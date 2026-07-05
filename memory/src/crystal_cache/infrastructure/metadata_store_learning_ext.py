"""Learning-state-table extension methods — Phase 7 Wave 7E / AN-7
(extended Wave 7F for read-side methods).

The three "secondary reference tables" for the learning subsystem —
`blacklisted_reflections`, `mandatory_rules`, `meta_patterns` — were
outside the original nine-table audit scope (D11 / AN-7). Phase 5's
AuditTablesMixin didn't cover them. v1's `learning_service.py`,
`maintenance/consolidation_service.py`, and `endpoints/sdk.py`
(/v1/retrieve) accessed them via inline SQLAlchemy, violating the
"no SQL outside the store layer" rule (R9).

Wave 7E closed the write-side gap by adding the four write methods
here (is_reflection_blacklisted, add_blacklisted_reflection,
replace_mandatory_rules, replace_meta_patterns). **Wave 7F extends
this mixin with two read-side methods** —
`list_mandatory_rules_for_customer` and
`list_meta_patterns_for_customer` — to support /v1/retrieve's
loading of mandatory_rules + meta_patterns into the
`ComposerContext` for injection composition. Same mixin-via-setattr
pattern as the audit tables (D12), customer extensions
(Phase 6.5 P4.1), and cognition extensions (Wave C).

Why these two read methods belong in this file (not a new mixin):
they read from the SAME two tables (`mandatory_rules`,
`meta_patterns`) that `replace_mandatory_rules` and
`replace_meta_patterns` write to. Splitting read and write for the
same table across two files would obscure the contract.

Wave 7E inventory correction (2026-05-20): an earlier framing of
AN-10 claimed `BlacklistedReflectionRow` was undefined in v1's
schema.py. That was an artifact of `Filesystem:search_files` doing
filename-glob matching, not content matching — no file is *named*
after the class, but the class is defined within v1's schema.py
under "V2 Learning State Tables (migration 0018)" alongside
`MandatoryRuleRow` and `MetaPatternRow`. All three were ported to
v2 verbatim in Phase 2. **No schema additions were required for
Wave 7E.** See ledger AN-10 + AN-11 corrections.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

import structlog
from sqlalchemy import delete, select

from .schema import (
    BlacklistedReflectionRow,
    MandatoryRuleRow,
    MetaPatternRow,
)

logger = structlog.get_logger(__name__)


class LearningExtensionsMixin:
    """Learning-state-table methods bound onto MetadataStore.

    Bound at import time via setattr in `infrastructure/__init__.py`,
    same pattern as AuditTablesMixin (D12), CustomerExtensionsMixin
    (Phase 6.5 P4.1), and CognitionExtensionsMixin (Wave C).
    """

    # -----------------------------------------------------------------
    # Blacklisted reflections — used by LearningService
    # -----------------------------------------------------------------

    async def is_reflection_blacklisted(
        self,
        customer_id: str,
        reflection_hash: str,
    ) -> bool:
        """Check if a reflection hash is blacklisted for this customer.

        The (customer_id, reflection_hash) pair has a unique index
        per migration 0018, so a single row lookup is sufficient.

        Returns True if the row exists, False otherwise.

        Per Wave 7E / AN-7 refactor of `LearningService._is_blacklisted`.
        """
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = select(BlacklistedReflectionRow).where(
                BlacklistedReflectionRow.customer_id == customer_id,
                BlacklistedReflectionRow.reflection_hash == reflection_hash,
            )
            row = (await session.execute(stmt)).scalar_one_or_none()
            return row is not None

    async def add_blacklisted_reflection(
        self,
        customer_id: str,
        reflection_hash: str,
        reflection_text: str,
        reason: Optional[str] = None,
    ) -> None:
        """Add a reflection to the blacklist if not already present.

        Idempotent: the unique index on (customer_id, reflection_hash)
        prevents duplicates at the DB level, and the in-method check
        avoids the round-trip to discover the constraint.

        Per Wave 7E / AN-7 refactor of `LearningService._add_to_blacklist`.
        """
        async with self.session() as session:  # type: ignore[attr-defined]
            # Check existing first to avoid relying on the unique-index
            # error path. Matches v1's pre-refactor behavior verbatim.
            stmt = select(BlacklistedReflectionRow).where(
                BlacklistedReflectionRow.customer_id == customer_id,
                BlacklistedReflectionRow.reflection_hash == reflection_hash,
            )
            existing = (await session.execute(stmt)).scalar_one_or_none()
            if existing is not None:
                return

            row = BlacklistedReflectionRow(
                id=uuid.uuid4().hex[:16],
                customer_id=customer_id,
                reflection_hash=reflection_hash,
                reflection_text=reflection_text,
                reason=reason,
            )
            session.add(row)

    # -----------------------------------------------------------------
    # Mandatory rules — used by ConsolidationService (write) +
    #                   sdk.sdk_retrieve (read, Wave 7F)
    # -----------------------------------------------------------------

    async def replace_mandatory_rules(
        self,
        customer_id: str,
        rules: list[dict[str, Any]],
    ) -> tuple[int, int]:
        """Replace this customer's mandatory rules with the given set.

        Existing rules are deleted; the new ones are inserted in one
        session. The (mandatory, advisory) counts are returned so the
        caller can report them.

        Each rule dict has keys:
          rule: str (the rule text, possibly with embedded UNLESS)
          priority: 'mandatory' | 'advisory'

        UNLESS clauses are split out at the boundary: if the rule
        text contains "UNLESS", everything before becomes `rule_text`
        and everything after becomes `unless_clause`. Matches v1's
        consolidation_service._write_mandatory_rules verbatim.

        Per Wave 7E / AN-7 refactor of
        `ConsolidationService._write_mandatory_rules`.
        """
        mandatory_count = 0
        advisory_count = 0

        async with self.session() as session:  # type: ignore[attr-defined]
            # Delete existing rules for this customer
            await session.execute(
                delete(MandatoryRuleRow).where(
                    MandatoryRuleRow.customer_id == customer_id
                )
            )

            # Write new rules
            for rule in rules:
                rule_text = rule.get("rule", "")
                priority = rule.get("priority", "advisory")

                # Split UNLESS clause if embedded in rule text
                unless_clause: Optional[str] = None
                if "UNLESS" in rule_text:
                    parts = rule_text.split("UNLESS", 1)
                    rule_text = parts[0].strip()
                    unless_clause = parts[1].strip()

                row = MandatoryRuleRow(
                    id=uuid.uuid4().hex[:16],
                    customer_id=customer_id,
                    rule_text=rule_text,
                    is_mandatory=(priority == "mandatory"),
                    unless_clause=unless_clause,
                    source_round=None,  # filled by caller if known
                )
                session.add(row)

                if priority == "mandatory":
                    mandatory_count += 1
                else:
                    advisory_count += 1

        return mandatory_count, advisory_count

    async def list_mandatory_rules_for_customer(
        self,
        customer_id: str,
    ) -> list[MandatoryRuleRow]:
        """List all mandatory rules for a customer (read side, Wave 7F).

        Returns the raw `MandatoryRuleRow` instances; callers
        typically pass these directly into `ComposerContext.mandatory_rules`,
        which is typed `list[Any]` for this reason.

        Per Wave 7F / AN-7 refactor of `endpoints/sdk.py::sdk_retrieve`,
        which v1 implemented with inline SQLAlchemy. Read-side
        complement of `replace_mandatory_rules`.
        """
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = select(MandatoryRuleRow).where(
                MandatoryRuleRow.customer_id == customer_id
            )
            return list((await session.execute(stmt)).scalars().all())

    # -----------------------------------------------------------------
    # Meta patterns — used by ConsolidationService (write) +
    #                  sdk.sdk_retrieve (read, Wave 7F)
    # -----------------------------------------------------------------

    async def replace_meta_patterns(
        self,
        customer_id: str,
        patterns: list[dict[str, Any]],
    ) -> int:
        """Replace this customer's meta patterns with the given set.

        Existing patterns are deleted; the new ones are inserted in
        one session. The count of patterns written is returned.

        Each pattern dict has keys:
          proposed_rule: str (the pattern text)
          affected_count: int (how many failures the pattern covers)

        Patterns with no proposed_rule or affected_count < 5 are
        skipped. Matches v1's
        consolidation_service._run_meta_reflection write-side
        behavior verbatim.

        Per Wave 7E / AN-7 refactor of
        `ConsolidationService._run_meta_reflection`.
        """
        count = 0

        async with self.session() as session:  # type: ignore[attr-defined]
            # Delete existing patterns for this customer
            await session.execute(
                delete(MetaPatternRow).where(
                    MetaPatternRow.customer_id == customer_id
                )
            )

            for pattern in patterns:
                proposed = pattern.get("proposed_rule", "")
                affected = pattern.get("affected_count", 0)
                if not proposed or affected < 5:
                    continue

                row = MetaPatternRow(
                    id=uuid.uuid4().hex[:16],
                    customer_id=customer_id,
                    pattern_text=proposed,
                    affected_count=affected,
                )
                session.add(row)
                count += 1

        return count

    async def list_meta_patterns_for_customer(
        self,
        customer_id: str,
    ) -> list[MetaPatternRow]:
        """List all meta patterns for a customer (read side, Wave 7F).

        Returns the raw `MetaPatternRow` instances; callers
        typically pass these directly into `ComposerContext.meta_patterns`,
        which is typed `list[Any]` for this reason.

        Per Wave 7F / AN-7 refactor of `endpoints/sdk.py::sdk_retrieve`,
        which v1 implemented with inline SQLAlchemy. Read-side
        complement of `replace_meta_patterns`.
        """
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = select(MetaPatternRow).where(
                MetaPatternRow.customer_id == customer_id
            )
            return list((await session.execute(stmt)).scalars().all())
