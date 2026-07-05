"""Pluggable composer strategies for crystal injection.

The composer takes retrieved crystals, mandatory rules, meta patterns,
and other context, and assembles them into a structured text block
for injection into the upstream LLM call.

Two strategies are validated:
  - InstructionComposer: flat imperative rules ("follow these rules")
  - BayesianComposer: evidence chains ("you assumed X, evidence shows Y")

The strategy is selected per-customer via config. New strategies can
be added by subclassing ComposerStrategy.

USAGE:
    from crystal_cache.retrieval.composer import get_composer

    composer = get_composer("instruction")  # or "bayesian"
    injection = await composer.compose(
        crystals=crystals,
        facts=facts,
        mandatory_rules=mandatory_rules,
        meta_patterns=meta_patterns,
        customer_id=customer_id,
        query_text=query_text,
    )

MIGRATION FROM SCRIPTS:
    This module extracts the core composition logic from:
    - scripts/bcb_composer.py → InstructionComposer
    - scripts/bcb_composer_bayesian.py → BayesianComposer
    The script versions remain for benchmark use; this module is
    the production interface.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from crystal_cache.infrastructure.schema import (
        MandatoryRuleRow,
        MetaPatternRow,
    )
    from crystal_cache.models import Crystal, Fact


# ---------------------------------------------------------------------------
# Data classes for composer inputs
# ---------------------------------------------------------------------------

@dataclass
class ComposerContext:
    """Everything the composer needs to build an injection block.

    Assembled by the retrieval pipeline before calling the composer.
    Separates retrieval (finding the right crystals) from composition
    (formatting them for the model).
    """
    # Retrieved crystals and facts (from cosine/ANN channels)
    matched_crystals: list[Any] = field(default_factory=list)
    recalled_facts: list[Any] = field(default_factory=list)

    # Mandatory rules and meta patterns (from DB, Item 3)
    mandatory_rules: list[Any] = field(default_factory=list)
    meta_patterns: list[Any] = field(default_factory=list)

    # Non-cosine channel results (direct DB lookups)
    failure_rules: list[str] = field(default_factory=list)
    knowledge_facts: list[str] = field(default_factory=list)
    process_crystals: list[str] = field(default_factory=list)
    behavior_rules: list[str] = field(default_factory=list)
    promoted_rules: list[str] = field(default_factory=list)

    # Blacklisted reflections (for cross-validation)
    blacklisted_hashes: set[str] = field(default_factory=set)

    # Bayesian-specific (only used by BayesianComposer)
    baseline_priors: dict[str, str] = field(default_factory=dict)
    prior_assumptions: dict[str, str] = field(default_factory=dict)

    # Query context
    customer_id: str = ""
    query_text: str = ""
    query_shapes: list[str] = field(default_factory=list)

    # Reference material (from cosine retrieval)
    reference_text: Optional[str] = None
    reference_prompt: Optional[str] = None
    reference_match_type: Optional[str] = None


# ---------------------------------------------------------------------------
# Strategy base class
# ---------------------------------------------------------------------------

class ComposerStrategy(ABC):
    """Base class for injection composition strategies.

    Subclasses implement compose() which takes a ComposerContext
    and returns the formatted injection text.
    """

    @abstractmethod
    async def compose(self, ctx: ComposerContext) -> str:
        """Build the injection text block from the given context.

        Args:
            ctx: ComposerContext with all retrieved crystals, rules,
                 patterns, and query context.

        Returns:
            The full injection text block. Empty string if no injection.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging and config."""
        ...


# ---------------------------------------------------------------------------
# Instruction Composer (validated: V1 ceiling 0.581)
# ---------------------------------------------------------------------------

class InstructionComposer(ComposerStrategy):
    """Flat imperative rule injection.

    Assembles crystals as structured sections with imperative voice:
    - Mandatory rules (highest priority, always first)
    - Meta patterns (systemic failure patterns)
    - Failure constraints (Level B reflections)
    - Domain knowledge (F1 knowledge crystals)
    - Reference material (cosine-matched solutions)
    - Process crystals (procedural guidance)
    - Behavior rules (customer preferences)
    - Promoted insights (session → permanent)

    Caps:
    - 3 mandatory rules
    - 3 meta patterns (deduped against mandatory)
    - 3 failure rules (cross-validated against mandatory)
    - 3 knowledge items
    - 3 process crystals
    - 3 behavior rules
    - 3 promoted insights
    """

    @property
    def name(self) -> str:
        return "instruction"

    async def compose(self, ctx: ComposerContext) -> str:
        sections: list[str] = []

        # ── Mandatory rules (highest priority) ──
        if ctx.mandatory_rules:
            rules = ctx.mandatory_rules[:3]
            formatted = "\n".join(
                f"- {_format_mandatory_rule(r)}" for r in rules
            )
            sections.append(
                "### MANDATORY — User requirements (DO NOT IGNORE)\n"
                "These rules override your default behavior. "
                "Violating them is considered a failure.\n\n"
                + formatted
            )

        # ── Meta patterns ──
        if ctx.meta_patterns:
            # Dedup against mandatory rules
            mandatory_texts = {
                _rule_text(r).lower() for r in ctx.mandatory_rules[:3]
            }
            deduped = [
                p for p in ctx.meta_patterns
                if not _overlaps_mandatory(p, mandatory_texts)
            ]
            if deduped:
                formatted = "\n".join(
                    f"- {_format_meta_pattern(p)}" for p in deduped[:3]
                )
                sections.append(
                    "### Systemic rules (from failure analysis)\n"
                    "These patterns caused repeated failures. "
                    "Follow them strictly.\n\n" + formatted
                )

        # ── Failure rules (cross-validated) ──
        rules = _cross_validate_failure_rules(
            ctx.failure_rules, ctx.mandatory_rules, ctx.blacklisted_hashes
        )
        if rules:
            formatted = "\n".join(f"- {rule}" for rule in rules[:3])
            sections.append(
                "### Constraints from prior attempts\n"
                "Earlier attempts on related problems failed. "
                "Apply these rules:\n\n" + formatted
            )

        # ── Domain knowledge ──
        if ctx.knowledge_facts:
            formatted = "\n\n".join(ctx.knowledge_facts[:3])
            sections.append(
                "### Domain knowledge\n"
                "The following knowledge may help:\n\n" + formatted
            )

        # ── Reference material ──
        if ctx.reference_text:
            if ctx.reference_match_type == "PERFECT":
                ref = "### Closely matching reference from memory\n"
                if ctx.reference_prompt:
                    ref += f"**Matched on:**\n{ctx.reference_prompt[:1000]}\n\n"
                ref += f"**Content:**\n{ctx.reference_text}"
            else:
                ref = (
                    "### Reference material from memory\n"
                    "Use what's relevant; ignore what isn't.\n\n"
                )
                if ctx.reference_prompt:
                    ref += f"**Matched on:**\n{ctx.reference_prompt[:800]}\n\n"
                ref += f"**Content:**\n{ctx.reference_text}"
            sections.append(ref)

        # ── Process crystals ──
        if ctx.process_crystals:
            formatted = "\n\n".join(ctx.process_crystals[:3])
            sections.append("### Recommended approach\n" + formatted)

        # ── Behavior rules ──
        if ctx.behavior_rules:
            formatted = "\n".join(
                f"- {rule}" for rule in ctx.behavior_rules[:3]
            )
            sections.append(
                "### User preferences\n"
                "Based on prior interactions:\n\n" + formatted
            )

        # ── Promoted insights ──
        if ctx.promoted_rules:
            formatted = "\n".join(
                f"- {rule}" for rule in ctx.promoted_rules[:3]
            )
            sections.append("### Learned insights\n" + formatted)

        if not sections:
            return ""

        return "## Relevant context from memory\n\n" + "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Bayesian Composer (validated: quick test +1.4pp over instruction)
# ---------------------------------------------------------------------------

class BayesianComposer(ComposerStrategy):
    """Evidence chain framing for crystal injection.

    Instead of flat imperative rules, frames crystals as evidence
    that updates the model's prior beliefs:

        YOUR BASELINE: You rounded averages to 2dp
        EVIDENCE: Test expected full precision 172.3437
        MECHANISM: csv.writer preserves values as-is
        UPDATE: Preserve full precision
        EXCEPTION: Unless spec explicitly says to round

    This framing is more robust than instruction framing because
    it gives the model the REASONING behind the rule, not just
    the rule itself.
    """

    @property
    def name(self) -> str:
        return "bayesian"

    async def compose(self, ctx: ComposerContext) -> str:
        sections: list[str] = []

        # ── Mandatory rules (same as instruction — mandatory is mandatory) ──
        if ctx.mandatory_rules:
            rules = ctx.mandatory_rules[:3]
            formatted = "\n".join(
                f"- {_format_mandatory_rule(r)}" for r in rules
            )
            sections.append(
                "### MANDATORY — Non-negotiable requirements\n\n" + formatted
            )

        # ── Evidence chains from failure rules + knowledge ──
        evidence_chains = _build_evidence_chains(
            ctx.failure_rules,
            ctx.knowledge_facts,
            ctx.baseline_priors,
            ctx.prior_assumptions,
        )
        if evidence_chains:
            sections.append(
                "### Evidence from prior attempts\n"
                "Your default assumptions on similar tasks were tested. "
                "Update your approach based on this evidence:\n\n"
                + "\n\n".join(evidence_chains[:5])
            )

        # ── Reference material (same framing as instruction) ──
        if ctx.reference_text:
            ref = (
                "### Reference material from memory\n"
                "Relevant prior knowledge — adapt it to the current "
                "context rather than reproducing it verbatim.\n\n"
            )
            if ctx.reference_prompt:
                ref += f"**Matched on:**\n{ctx.reference_prompt[:800]}\n\n"
            ref += f"**Content:**\n{ctx.reference_text}"
            sections.append(ref)

        # ── Process crystals ──
        if ctx.process_crystals:
            formatted = "\n\n".join(ctx.process_crystals[:3])
            sections.append("### Recommended approach\n" + formatted)

        # ── Behavior + promoted (same as instruction) ──
        if ctx.behavior_rules:
            formatted = "\n".join(
                f"- {rule}" for rule in ctx.behavior_rules[:3]
            )
            sections.append("### User preferences\n" + formatted)

        if ctx.promoted_rules:
            formatted = "\n".join(
                f"- {rule}" for rule in ctx.promoted_rules[:3]
            )
            sections.append("### Learned insights\n" + formatted)

        if not sections:
            return ""

        return "## Relevant context from memory\n\n" + "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_mandatory_rule(rule) -> str:
    """Format a MandatoryRuleRow or dict into a rule string."""
    if hasattr(rule, "rule_text"):
        text = rule.rule_text
        unless = getattr(rule, "unless_clause", None)
    elif isinstance(rule, dict):
        text = rule.get("rule", rule.get("rule_text", ""))
        unless = rule.get("unless_clause") or rule.get("unless")
    else:
        return str(rule)

    if unless:
        return f"{text} UNLESS {unless}"
    return text


def _rule_text(rule) -> str:
    """Extract rule text from a MandatoryRuleRow or dict."""
    if hasattr(rule, "rule_text"):
        return rule.rule_text
    if isinstance(rule, dict):
        return rule.get("rule", rule.get("rule_text", ""))
    return str(rule)


def _format_meta_pattern(pattern) -> str:
    """Format a MetaPatternRow or dict."""
    if hasattr(pattern, "pattern_text"):
        count = getattr(pattern, "affected_count", 0)
        return f"{pattern.pattern_text} ({count} prior failures)"
    if isinstance(pattern, dict):
        text = pattern.get("proposed_rule", pattern.get("pattern_text", ""))
        count = pattern.get("affected_count", 0)
        return f"{text} ({count} prior failures)"
    return str(pattern)


def _overlaps_mandatory(pattern, mandatory_texts: set[str]) -> bool:
    """Check if a meta pattern overlaps with mandatory rules (dedup)."""
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'to', 'in', 'of', 'for', 'with',
        'must', 'not', 'be', 'is', 'that', 'all', 'this', 'do', 'should',
    }
    if hasattr(pattern, "pattern_text"):
        text = pattern.pattern_text
    elif isinstance(pattern, dict):
        text = pattern.get("proposed_rule", pattern.get("pattern_text", ""))
    else:
        text = str(pattern)

    p_words = set(text.lower().split()) - stopwords
    for m_text in mandatory_texts:
        m_words = set(m_text.split()) - stopwords
        if len(m_words & p_words) >= 5:
            return True
    return False


def _cross_validate_failure_rules(
    failure_rules: list[str],
    mandatory_rules: list,
    blacklisted_hashes: set[str],
) -> list[str]:
    """Filter failure rules: remove blacklisted and contradictory.

    Args:
        failure_rules: Raw failure reflection texts.
        mandatory_rules: MandatoryRuleRow or dict objects.
        blacklisted_hashes: Set of SHA256 hashes (16 hex chars)
            of known-bad reflections.

    Returns:
        Filtered list with blacklisted and contradictory rules removed.
    """
    mandatory_texts = [_rule_text(r).lower() for r in mandatory_rules[:3]]
    filtered = []

    for rule in failure_rules:
        # Skip blacklisted
        rule_hash = hashlib.sha256(rule.encode()).hexdigest()[:16]
        if rule_hash in blacklisted_hashes:
            continue

        # Skip contradictions with mandatory rules
        rule_lower = rule.lower()
        contradicts = False
        for m_lower in mandatory_texts:
            if (("round" in rule_lower and "precision" in m_lower) or
                ("validate" in rule_lower and "gracefully" in m_lower
                 and "exception" in rule_lower)):
                contradicts = True
                break
        if not contradicts:
            filtered.append(rule)

    return filtered


def _build_evidence_chains(
    failure_rules: list[str],
    knowledge_facts: list[str],
    baseline_priors: dict[str, str],
    prior_assumptions: dict[str, str],
) -> list[str]:
    """Build Bayesian evidence chains from failure rules + knowledge.

    Attempts to pair each failure rule with a prior assumption that
    it corrects. When a prior is found, the chain shows the model
    what it assumed vs what the evidence shows. When no prior is
    found, the rule is presented as standalone evidence.

    Args:
        failure_rules: Imperative rules from Level B reflections.
        knowledge_facts: Domain knowledge from F1 crystals.
        baseline_priors: Task-level priors from baseline analysis
            (keyed by task_id — not used yet, reserved for when
            the composer receives task_id in ComposerContext).
        prior_assumptions: Per-crystal prior assumptions extracted
            during F1 knowledge generation.

    Returns:
        List of formatted evidence chain strings.
    """
    chains = []

    for rule in failure_rules:
        # Try to find a matching prior assumption
        prior = None
        for key, assumption in prior_assumptions.items():
            if any(word in rule.lower() for word in key.lower().split()[:3]):
                prior = assumption
                break

        if prior:
            chain = (
                f"YOUR DEFAULT: {prior}\n"
                f"EVIDENCE: {rule}\n"
                f"UPDATE: Follow the evidence, not your default assumption."
            )
        else:
            chain = (
                f"EVIDENCE: {rule}\n"
                f"UPDATE: Apply this constraint to your solution."
            )
        chains.append(chain)

    # Add knowledge facts as supporting evidence
    for fact in knowledge_facts[:3]:
        chains.append(f"DOMAIN KNOWLEDGE: {fact}")

    return chains


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_STRATEGIES: dict[str, type[ComposerStrategy]] = {
    "instruction": InstructionComposer,
    "bayesian": BayesianComposer,
}


def get_composer(strategy: str = "instruction") -> ComposerStrategy:
    """Get a composer strategy by name.

    Args:
        strategy: "instruction" or "bayesian"

    Returns:
        An instance of the requested ComposerStrategy.

    Raises:
        ValueError: If the strategy name is not recognized.
    """
    cls = _STRATEGIES.get(strategy.lower())
    if cls is None:
        valid = ", ".join(repr(k) for k in _STRATEGIES)
        raise ValueError(
            f"Unknown composer strategy {strategy!r}. Valid: {valid}"
        )
    return cls()


def register_composer(name: str, cls: type[ComposerStrategy]) -> None:
    """Register a custom composer strategy.

    Use this to add domain-specific composers without modifying
    this module.
    """
    _STRATEGIES[name.lower()] = cls
