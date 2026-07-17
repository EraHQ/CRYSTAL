"""Consolidation Service — idle phase memory consolidation.

Production service that extracts the consolidation logic from
scripts/idle_phase.py into an importable, multi-tenant async service.

Runs BETWEEN learning rounds (not during). Performs:
  1. Gather behavior crystals for a customer from the crystal bank
  2. Gather contradicting knowledge (F1 facts that conflict with rules)
  3. Consolidate behavior rules via LLM → mandatory rules with UNLESS clauses
  4. Meta-reflection on systemic failure patterns
  5. Write results to DB tables (mandatory_rules, meta_patterns)

The consolidation service does NOT modify crystals. It reads from
the crystal bank and writes to the learning state tables. The
composer reads those tables at injection time.

RELATIONSHIP TO EXISTING CODE:
  - scripts/idle_phase.py — the benchmark version (file-based state,
    hardcoded paths). Stays for benchmarking.
  - This service reads crystals from DB and writes to mandatory_rules
    and meta_patterns tables (migration 0018).

TRIGGER MECHANISMS:
  - Schedule: every N learning events per customer
  - Manual: admin API endpoint
  - After batch learning: automatically after learn_from_failures()

USAGE:
    from crystal_cache.maintenance.consolidation_service import (
        ConsolidationService,
    )

    service = ConsolidationService(store=store)
    result = await service.consolidate(
        customer_id="cus_xxx",
        crystal_type="general:python_general",
    )

PORT NOTE (v2 Wave 7E, 2026-05-20): the two methods
`_write_mandatory_rules` and `_run_meta_reflection` in v1 carried
inline SQLAlchemy queries against `MandatoryRuleRow` and
`MetaPatternRow` — R9 violation. Per Wave 7E AN-7 refactor, those
queries moved to `LearningExtensionsMixin.replace_mandatory_rules`
and `replace_meta_patterns` on the store. This file is no longer
verbatim with v1; the refactor is documented in
PROJECT_LEDGER.md (AN-7, P0.8, Wave 7E close-out).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

from ..cost.emit import record_model_call
from ..llm import get_llm_client

if TYPE_CHECKING:
    from ..infrastructure.metadata_store import MetadataStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM prompts (extracted from idle_phase.py)
# ---------------------------------------------------------------------------

CONSOLIDATE_SYSTEM = """You are a memory consolidation engine.

You will receive TWO inputs:
1. A list of behavior crystal rules generated across multiple learning rounds
2. A list of task-specific knowledge that CONTRADICTS some of those rules

Your job:
1. Group behavior rules that say the same thing
2. Merge each group into ONE clean, precise rule
3. CRITICAL: For each merged rule, check if any contradicting knowledge
   applies. If it does, add an UNLESS clause that specifies when the
   rule should NOT be followed.
4. Flag any rules that contradict each other

Return JSON (no markdown, no backticks):
{
  "merged_rules": [
    {
      "rule": "The consolidated rule text WITH exception clause if needed",
      "source_count": 5,
      "priority": "mandatory" or "advisory",
      "has_exceptions": true or false
    }
  ],
  "contradictions": [
    {
      "rule_a": "...",
      "rule_b": "...",
      "explanation": "..."
    }
  ],
  "skip_rules": [
    {
      "rule": "...",
      "reason": "Superseded by merged rule X"
    }
  ]
}

Rules appearing in 5+ originals should get priority "mandatory".
Rules appearing in 1-2 originals stay "advisory".
Rules appearing in 3-4 originals use your judgment."""

META_REFLECTION_SYSTEM = """You are analyzing ALL failures from a
coding benchmark round as a group. Not individual tasks — the WHOLE SESSION.

Look for SYSTEMIC patterns. What are the most common categories of
failure? What single rules would fix the MOST failures at once?

Return JSON (no markdown, no backticks):
{
  "patterns": [
    {
      "pattern": "Description of the systemic pattern",
      "affected_count": 17,
      "proposed_rule": "The mandatory behavior rule that would fix this",
      "priority": "mandatory"
    }
  ],
  "total_failures": 63,
  "coverage": "X of Y failures covered by proposed rules"
}

Focus on patterns that affect 5+ tasks. Ignore one-off issues."""

# Signals that indicate contradicting knowledge
_CONTRADICTION_SIGNALS = [
    r"required parameter",
    r"must accept",
    r"must have.*parameter",
    r"signature must",
    r"explicitly requires",
    r"raise.*error",
    r"raise.*exception",
    r"should raise",
    r"expects.*exception",
    r"positional argument",
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ConsolidationResult:
    """Result of running consolidation for one customer."""
    customer_id: str
    behavior_rules_found: int = 0
    contradictions_found: int = 0
    mandatory_rules_written: int = 0
    advisory_rules_written: int = 0
    meta_patterns_written: int = 0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Consolidation Service
# ---------------------------------------------------------------------------

class ConsolidationService:
    """Async service for idle-phase memory consolidation.

    Reads behavior crystals and knowledge from the crystal bank,
    consolidates via LLM, writes mandatory rules and meta patterns
    to the V2 DB tables.
    """

    def __init__(self, store: "MetadataStore"):
        self._store = store

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    async def consolidate(
        self,
        customer_id: str,
        *,
        crystal_type: str = "general:python_general",
        run_meta: bool = True,
    ) -> ConsolidationResult:
        """Run full consolidation for one customer.

        Steps:
          1. Gather behavior crystals from the bank
          2. Gather contradicting knowledge
          3. Consolidate via LLM → mandatory rules with UNLESS clauses
          4. Write mandatory rules to DB (replacing existing)
          5. Optionally run meta-reflection on failure patterns
          6. Write meta patterns to DB (replacing existing)

        Args:
            customer_id: Which customer to consolidate.
            crystal_type: Crystal type to query.
            run_meta: Whether to run meta-reflection (slower, more tokens).

        Returns:
            ConsolidationResult with counts of what was written.
        """
        result = ConsolidationResult(customer_id=customer_id)

        # Step 1: Gather behavior crystals
        behavior_rules = await self._gather_behavior_crystals(
            customer_id, crystal_type
        )
        result.behavior_rules_found = len(behavior_rules)
        logger.info(
            "consolidation.behavior_gathered",
            customer_id=customer_id,
            count=len(behavior_rules),
        )

        if not behavior_rules:
            logger.info(
                "consolidation.skipped_no_behavior",
                customer_id=customer_id,
            )
            return result

        # Step 2: Gather contradicting knowledge
        contradictions = await self._gather_contradicting_knowledge(
            customer_id, crystal_type
        )
        result.contradictions_found = len(contradictions)

        # Step 3: Consolidate via LLM
        consolidated, _usage = self._consolidate_llm(behavior_rules, contradictions)
        if _usage is not None:
            await record_model_call(
                customer_id=customer_id,
                origin="consolidation",
                model=_usage.model,
                input_tokens=_usage.input_tokens,
                output_tokens=_usage.output_tokens,
                cache_creation_tokens=_usage.cache_creation_tokens,
                cache_read_tokens=_usage.cache_read_tokens,
                store=self._store,
            )
        if consolidated is None:
            result.error = "LLM consolidation failed"
            return result

        # Step 4: Write mandatory rules to DB (replace existing)
        mandatory, advisory = await self._write_mandatory_rules(
            customer_id, consolidated
        )
        result.mandatory_rules_written = mandatory
        result.advisory_rules_written = advisory

        # Step 5: Meta-reflection
        if run_meta:
            failure_reflections = await self._gather_failure_reflections(
                customer_id, crystal_type
            )
            if failure_reflections:
                meta_count = await self._run_meta_reflection(
                    customer_id, failure_reflections
                )
                result.meta_patterns_written = meta_count

        logger.info(
            "consolidation.complete",
            customer_id=customer_id,
            mandatory=result.mandatory_rules_written,
            advisory=result.advisory_rules_written,
            meta=result.meta_patterns_written,
        )
        return result

    # -----------------------------------------------------------------
    # Private: Data gathering from crystal bank
    # -----------------------------------------------------------------

    async def _gather_behavior_crystals(
        self, customer_id: str, crystal_type: str
    ) -> list[str]:
        """Get all behavior crystal texts for a customer."""
        rules = []
        crystals = await self._store.list_crystals_for_customer(customer_id)
        for crystal in crystals:
            if crystal.crystal_type != crystal_type:
                continue
            facts = await self._store.list_facts_for_crystal(crystal.id)
            for fact in facts:
                if (
                    fact.pair_type
                    and fact.pair_type.startswith("behavior_crystal_")
                    and fact.claim_text
                ):
                    rules.append(fact.claim_text)
        return rules

    async def _gather_contradicting_knowledge(
        self, customer_id: str, crystal_type: str
    ) -> list[dict[str, str]]:
        """Find F1 knowledge that contradicts behavior rules."""
        contradictions = []
        crystals = await self._store.list_crystals_for_customer(customer_id)
        for crystal in crystals:
            if crystal.crystal_type != crystal_type:
                continue
            facts = await self._store.list_facts_for_crystal(crystal.id)
            for fact in facts:
                if (
                    not fact.pair_type
                    or not fact.pair_type.startswith("knowledge_crystal_")
                    or not fact.claim_text
                ):
                    continue
                k_lower = fact.claim_text.lower()
                for signal in _CONTRADICTION_SIGNALS:
                    if re.search(signal, k_lower):
                        contradictions.append({
                            "knowledge": fact.claim_text[:300],
                            "signal": signal,
                        })
                        break
        return contradictions

    async def _gather_failure_reflections(
        self, customer_id: str, crystal_type: str
    ) -> list[dict[str, str]]:
        """Get all failure reflections for meta-reflection analysis."""
        reflections = []
        crystals = await self._store.list_crystals_for_customer(customer_id)
        for crystal in crystals:
            if crystal.crystal_type != crystal_type:
                continue
            facts = await self._store.list_facts_for_crystal(crystal.id)
            for fact in facts:
                if (
                    fact.source_kind == "failed_reasoning"
                    and fact.claim_text
                ):
                    reflections.append({
                        "reflection": fact.claim_text,
                        "prompt": fact.prompt_text or "",
                    })
        return reflections

    # -----------------------------------------------------------------
    # Private: LLM consolidation
    # -----------------------------------------------------------------

    def _consolidate_llm(
        self,
        behavior_rules: list[str],
        contradictions: list[dict[str, str]],
    ) -> Optional[dict[str, Any]]:
        """Use LLM to consolidate behavior rules with UNLESS clauses."""

        rules_text = "\n".join(f"  - {rule}" for rule in behavior_rules)

        contra_text = ""
        if contradictions:
            contra_lines = [
                f"  - {ck['knowledge'][:200]}"
                for ck in contradictions[:20]
            ]
            contra_text = (
                f"\n\nCONTRADICTING KNOWLEDGE ({len(contradictions)} items, "
                f"showing {min(20, len(contradictions))}):\n"
                f"These task-specific findings CONTRADICT some of the "
                f"behavior rules above. Use them to add UNLESS clauses:\n\n"
                + "\n".join(contra_lines)
            )

        _usage = None
        try:
            _client = get_llm_client()
            _detailed = getattr(_client, "complete_detailed", None)
            _do = _detailed if _detailed is not None else _client.complete
            _out = _do(
                tier="small",
                temperature=0.0,
                max_tokens=4000,
                system=CONSOLIDATE_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Here are {len(behavior_rules)} behavior crystal "
                        f"rules:\n\n{rules_text}{contra_text}\n\n"
                        f"Consolidate them. Add UNLESS clauses where "
                        f"contradicting knowledge exists."
                    ),
                }],
            )
            if _detailed is not None:
                text, _usage = _out.text, _out
            else:
                text = _out
            return self._parse_json_response(text), _usage
        except Exception as e:
            logger.error("consolidation LLM call failed: %s", e)
            return None, _usage

    def _run_meta_reflection_llm(
        self, failure_reflections: list[dict[str, str]]
    ) -> Optional[dict[str, Any]]:
        """Use LLM to find systemic patterns across failures."""

        summaries = [
            f"  Reflection: {f['reflection'][:200]}"
            for f in failure_reflections
        ]
        failures_text = "\n\n".join(summaries)

        _usages: list = []
        for attempt, max_tok in enumerate([8000, 12000], 1):
            try:
                _client = get_llm_client()
                _detailed = getattr(_client, "complete_detailed", None)
                _do = _detailed if _detailed is not None else _client.complete
                _out = _do(
                    tier="small",
                    temperature=0.0,
                    max_tokens=max_tok,
                    system=META_REFLECTION_SYSTEM,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Here are ALL {len(failure_reflections)} "
                            f"failure reflections:\n\n{failures_text}\n\n"
                            f"Find systemic patterns."
                        ),
                    }],
                )
                if _detailed is not None:
                    text = _out.text
                    _usages.append(_out)
                else:
                    text = _out
                parsed = self._parse_json_response(text)
                if parsed is not None:
                    return parsed, _usages
                if attempt < 2:
                    logger.warning(
                        "meta-reflection JSON parse failed, retrying "
                        "with more tokens"
                    )
                    continue
                return None, _usages
            except Exception as e:
                logger.error(
                    "meta-reflection attempt %d failed: %s", attempt, e
                )
                if attempt < 2:
                    continue
                return None, _usages
        return None, _usages

    @staticmethod
    def _parse_json_response(text: str) -> Optional[dict[str, Any]]:
        """Parse JSON from LLM response with fallback strategies."""
        # Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Extract from markdown fences
        if "```" in text:
            inner = text.split("```")[1]
            if inner.startswith("json"):
                inner = inner[4:]
            inner = inner.strip()
            try:
                return json.loads(inner)
            except json.JSONDecodeError:
                pass

        # Truncate to last complete JSON object
        last_brace = text.rfind("}")
        if last_brace > 0:
            try:
                return json.loads(text[: last_brace + 1])
            except json.JSONDecodeError:
                pass

        return None

    # -----------------------------------------------------------------
    # Private: Write to DB
    # -----------------------------------------------------------------

    async def _write_mandatory_rules(
        self, customer_id: str, consolidated: dict[str, Any]
    ) -> tuple[int, int]:
        """Replace existing mandatory rules with new consolidated ones.

        Returns (mandatory_count, advisory_count).

        Per Wave 7E AN-7 refactor: the inline `delete + session.add`
        loop moved to `LearningExtensionsMixin.replace_mandatory_rules`
        on the store. The UNLESS-clause splitting + (mandatory,
        advisory) bookkeeping live in the mixin to keep semantic
        responsibility with the database boundary; this method just
        forwards.
        """
        return await self._store.replace_mandatory_rules(
            customer_id=customer_id,
            rules=consolidated.get("merged_rules", []),
        )

    async def _run_meta_reflection(
        self,
        customer_id: str,
        failure_reflections: list[dict[str, str]],
    ) -> int:
        """Run meta-reflection and write patterns to DB.

        Returns count of patterns written.

        Per Wave 7E AN-7 refactor: the inline `delete + session.add`
        loop moved to `LearningExtensionsMixin.replace_meta_patterns`
        on the store. The LLM call stays here; only the persistence
        moved.
        """
        meta_result, _usages = self._run_meta_reflection_llm(failure_reflections)
        for _usage in _usages:
            await record_model_call(
                customer_id=customer_id,
                origin="meta_reflection",
                model=_usage.model,
                input_tokens=_usage.input_tokens,
                output_tokens=_usage.output_tokens,
                cache_creation_tokens=_usage.cache_creation_tokens,
                cache_read_tokens=_usage.cache_read_tokens,
                store=self._store,
            )
        if meta_result is None:
            return 0

        return await self._store.replace_meta_patterns(
            customer_id=customer_id,
            patterns=meta_result.get("patterns", []),
        )
