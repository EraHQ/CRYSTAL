"""Cognition worker primitives — Phase 6 Wave C.

Phase 6 Wave C ports v1's `cognition/` package. The worker
`_worker_crystal_key_scan` in v1 had inline SQLAlchemy
(`sa_select(FactRow).join(CrystalRow,...)`) to perform a prefix
scan on facts' prompt_text within a tenant. v2's "no SQL outside
the store" rule (R9) requires this to land as a store method.

The method does not fit naturally into AuditTablesMixin (which is
scoped to the eight audit tables) and editing the Phase 3 verbatim
port of metadata_store.py is the explicit non-goal of the mixin
pattern (D12). So we follow the same shape Phase 6.5 P4.1
introduced for customer extensions: a separate file with its own
mixin, bound by `infrastructure/__init__.py` alongside the others.

If future cognition workers need additional store primitives
(rerank-by-recency, cross-crystal aggregate scans, etc.), they
belong here next to `list_facts_by_key_prefix`.
"""
from __future__ import annotations

from typing import Optional

import structlog
from sqlalchemy import and_, or_, select

from ..models import Fact
from .schema import CrystalRow, FactRow

logger = structlog.get_logger(__name__)


def _fact_from_row_minimal(row: FactRow) -> Fact:
    """Local converter — mirrors `metadata_store._fact_from_row`.

    Duplicated here rather than imported because the Phase 3 file's
    `_fact_from_row` is a module-private helper not exposed at the
    package boundary. The duplication is cheap (one function); the
    alternative is exposing internal converters or importing from
    a sibling private module, both of which add coupling.
    """
    return Fact(
        id=row.id,
        crystal_id=row.crystal_id,
        claim_text=row.claim_text,
        pair_type=row.pair_type,
        source_kind=row.source_kind,  # type: ignore[arg-type]
        answer_value=row.answer_value,
        prompt_text=row.prompt_text or "",
        vector=row.vector or [],
        source_doc_id=row.source_doc_id,
        extracted_by=row.extracted_by,
        verified_by=row.verified_by,
        grating_strength=row.grating_strength,
        hit_count=row.hit_count,
        last_hit_at=row.last_hit_at,
        created_at=row.created_at,
    )


class CognitionExtensionsMixin:
    """Cognition-worker store primitives, bound onto MetadataStore.

    Same binding pattern as AuditTablesMixin and
    CustomerExtensionsMixin: `infrastructure/__init__.py` iterates
    public methods at import time and `setattr`s them onto
    MetadataStore via `_bind_mixin_methods`. The mixin is NOT in
    MetadataStore's MRO — the binding is attribute-level.
    """

    async def list_facts_by_key_prefix(
        self,
        customer_id: str,
        *,
        key_prefix: str,
        subject_contains: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[Fact]:
        """Prefix scan on facts' `prompt_text` (sparse key) within a tenant.

        Returns all matching facts ordered by prompt_text ascending so
        cognition's `crystal_key_scan` worker can produce a stable,
        enumerable listing.

        Distinct from semantic vector search (FactVectorStore.search):
        this is a literal-string prefix scan on the indexed
        `prompt_text` column. Use it for enumeration questions
        ("how many scenes does the script have", "list every chapter")
        where vector top-k would return a truncated similarity-ranked
        sample instead of the full set.

        Args:
            customer_id: tenant scope. Joined through crystals.customer_id.
                Since the general-crystals merge (2026-06-12), scope is
                the tenant's crystals PLUS the general bank (customer_id
                NULL) for every type the tenant subscribes to — resolved
                here via get_customer_general_types so all key-scan
                consumers (agent key_scan tool, cognition
                crystal_key_scan) inherit general knowledge with zero
                call-site changes. Empty subscription = tenant-only,
                exactly the old behavior.
            key_prefix: literal prefix the fact's prompt_text must start
                with (e.g. `Script|Scene `). Caller is responsible for
                escaping any SQL LIKE metacharacters; current usage
                doesn't need it because v1's sparse keys are
                pipe-separated ASCII without `_` or `%`.
            subject_contains: optional secondary filter — prompt_text
                must additionally contain this substring. Used by
                cognition to narrow `Script|Scene |*` to a specific
                document subject.
            limit: optional row cap. None = no cap; callers that need
                cross-corpus enumeration pass None deliberately. The
                in-tenant set is bounded by document count so unbounded
                is generally fine at inspector scale.

        Returns:
            list of Fact in ascending prompt_text order.
        """
        subscribed = await self.get_customer_general_types(customer_id)  # type: ignore[attr-defined]
        scope = CrystalRow.customer_id == customer_id
        if subscribed:
            scope = or_(
                scope,
                and_(
                    CrystalRow.customer_id.is_(None),
                    CrystalRow.crystal_type.in_(subscribed),
                ),
            )
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(FactRow)
                .join(CrystalRow, FactRow.crystal_id == CrystalRow.id)
                .where(
                    and_(
                        scope,
                        FactRow.prompt_text.like(f"{key_prefix}%"),
                    )
                )
                .order_by(FactRow.prompt_text)
            )
            if subject_contains:
                stmt = stmt.where(FactRow.prompt_text.contains(subject_contains))
            if limit is not None:
                stmt = stmt.limit(limit)

            result = await session.execute(stmt)
            return [
                _fact_from_row_minimal(r)
                for r in result.scalars().all()
            ]

    async def list_recent_facts_for_customer(
        self,
        customer_id: str,
        *,
        limit: Optional[int] = None,
    ) -> list[Fact]:
        """A customer's OWN facts, newest first — candidate source for the
        contradiction scan (docs/NEVER_IDLE_CONVERGENCE.md).

        Deliberately distinct from `list_facts_by_key_prefix`:
          * OWN FACTS ONLY (D8). Joined through crystals.customer_id ==
            customer_id, with NO general-subscription merge. A customer's
            contradictions in v1 are within its own bank; general-vs-customer
            conflicts are a fast-follow. (list_facts_by_key_prefix folds in
            the general bank for subscribed types — wrong scope here.)
          * RECENCY ORDER (D2). Ordered by FactRow.created_at DESC so the
            generator's candidate cap keeps the freshest facts and the scan
            stays bounded; ties broken by id for determinism.
          * NO prefix filter. The generator buckets by crystal + sparse-key
            Subject in Python over the returned set; this method just hands
            it the (bounded) recent own-fact population.

        Args:
            customer_id: tenant scope (own crystals only).
            limit: optional cap on facts returned (the generator passes a
                cap so a huge bank doesn't enumerate unbounded). None = all.

        Returns:
            list of Fact, newest first.
        """
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(FactRow)
                .join(CrystalRow, FactRow.crystal_id == CrystalRow.id)
                .where(CrystalRow.customer_id == customer_id)
                .order_by(FactRow.created_at.desc(), FactRow.id.desc())
            )
            if limit is not None:
                stmt = stmt.limit(limit)
            result = await session.execute(stmt)
            return [
                _fact_from_row_minimal(r)
                for r in result.scalars().all()
            ]
