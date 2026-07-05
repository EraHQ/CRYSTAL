"""Knowledge-conflict store primitives — Never-Idle Convergence.

The CRUD behind the `knowledge_conflicts` table (the first-class peer of
`knowledge_gaps`). The contradiction-scan generator
(`scan/contradiction.py`) writes `open` rows here when its CONTRADICTS
discriminator fires; the admin Conflicts surface + the unified backlog
read these; a later curation gate transitions them to resolved/dismissed.

Same binding pattern as AuditTablesMixin / CognitionExtensionsMixin: this
mixin is NOT in MetadataStore's MRO — `infrastructure/__init__.py` iterates
its public methods at import time and `setattr`s each onto MetadataStore via
`_bind_mixin_methods`. `self.session()` inside a bound method resolves to
MetadataStore.session by normal attribute lookup on the bound callable.

SURFACING-ONLY (D5): the scan only ever calls `create_knowledge_conflict`.
The resolution mutators (`mark_knowledge_conflict_resolved` / `_dismissed`)
exist for the curation gate and the admin surface — never the scan.

IDEMPOTENT CREATE (D4): `create_knowledge_conflict` is a pre-checked insert.
If a row with this (customer_id, pair_key) already exists it returns that row
unchanged (no duplicate, no status reset) — so a re-scan over an unchanged
bank is a no-op even before the unique-index backstop. A fact whose claim
text CHANGED yields a different pair_key (the generator folds claim hashes
in), so it lands as a NEW conflict to evaluate. Atomic under SQLite's
single-writer SERIALIZABLE transaction (the claim_pending_documents_batch
precedent); the `ux_knowledge_conflicts_pair_key` unique index is the
backstop under Postgres concurrency.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import structlog
from sqlalchemy import func, or_, select

from ..models import KnowledgeConflict
from .schema import BlacklistedReflectionRow, FactRow, KnowledgeConflictRow

logger = structlog.get_logger(__name__)


class ConflictExtensionsMixin:
    """knowledge_conflicts CRUD, bound onto MetadataStore."""

    async def create_knowledge_conflict(
        self,
        customer_id: str,
        *,
        fact_a_id: str,
        fact_b_id: str,
        claim_a: str,
        claim_b: str,
        pair_key: str,
        crystal_a_id: Optional[str] = None,
        crystal_b_id: Optional[str] = None,
        subject: Optional[str] = None,
        provenance_a: Optional[str] = None,
        provenance_b: Optional[str] = None,
        detector: str = "contradiction_scan",
    ) -> KnowledgeConflict:
        """Insert one `open` conflict, idempotent on (customer_id, pair_key).

        If a row with this pair_key already exists for the customer, returns
        it unchanged — a re-scan never duplicates or reopens. Otherwise
        inserts a fresh `open` row and returns it.
        """
        import uuid

        async with self.session() as session:  # type: ignore[attr-defined]
            existing_stmt = (
                select(KnowledgeConflictRow)
                .where(KnowledgeConflictRow.customer_id == customer_id)
                .where(KnowledgeConflictRow.pair_key == pair_key)
                .limit(1)
            )
            existing = (
                await session.execute(existing_stmt)
            ).scalar_one_or_none()
            if existing is not None:
                return _knowledge_conflict_from_row(existing)

            conflict_id = f"kc_{uuid.uuid4().hex[:16]}"
            now = datetime.now(timezone.utc)
            row = KnowledgeConflictRow(
                id=conflict_id,
                customer_id=customer_id,
                fact_a_id=fact_a_id,
                fact_b_id=fact_b_id,
                crystal_a_id=crystal_a_id,
                crystal_b_id=crystal_b_id,
                subject=subject,
                claim_a=claim_a,
                claim_b=claim_b,
                provenance_a=provenance_a,
                provenance_b=provenance_b,
                detector=detector,
                status="open",
                resolution=None,
                pair_key=pair_key,
                created_at=now,
            )
            session.add(row)
            return _knowledge_conflict_from_row(row)

    async def knowledge_conflict_exists(
        self, customer_id: str, *, pair_key: str
    ) -> bool:
        """True if a row with this (customer_id, pair_key) exists in ANY
        status (open / resolved / dismissed).

        The generator's cheap pre-check (D4): a pair already recorded —
        including a terminal resolved/dismissed one — is skipped before the
        (expensive) discriminator call, so the loop quiesces and dismissed
        conflicts never re-surface.
        """
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(KnowledgeConflictRow.id)
                .where(KnowledgeConflictRow.customer_id == customer_id)
                .where(KnowledgeConflictRow.pair_key == pair_key)
                .limit(1)
            )
            return (await session.execute(stmt)).scalar_one_or_none() is not None

    async def list_knowledge_conflicts(
        self,
        customer_id: str,
        *,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[KnowledgeConflict]:
        """Paginated per-customer list, newest first, optional status
        filter. Backs the admin Conflicts surface and the backlog
        read-model."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(KnowledgeConflictRow)
                .where(KnowledgeConflictRow.customer_id == customer_id)
                .order_by(KnowledgeConflictRow.created_at.desc())
                .limit(limit)
            )
            if status is not None:
                stmt = stmt.where(KnowledgeConflictRow.status == status)
            result = await session.execute(stmt)
            return [
                _knowledge_conflict_from_row(r)
                for r in result.scalars().all()
            ]

    async def count_open_conflicts_for_crystal(
        self, customer_id: str, crystal_id: str,
    ) -> int:
        """Open conflicts touching one crystal on either side (the
        tier-promotion demotion signal)."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(func.count())
                .select_from(KnowledgeConflictRow)
                .where(
                    KnowledgeConflictRow.customer_id == customer_id,
                    KnowledgeConflictRow.status == "open",
                    or_(
                        KnowledgeConflictRow.crystal_a_id == crystal_id,
                        KnowledgeConflictRow.crystal_b_id == crystal_id,
                    ),
                )
            )
            return int((await session.execute(stmt)).scalar() or 0)

    async def count_knowledge_conflicts(
        self, customer_id: str, *, status: Optional[str] = "open"
    ) -> int:
        """Count conflicts for a customer (default: open only). Cheap
        signal for the backlog summary / worth gate without paging rows."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(func.count(KnowledgeConflictRow.id))
                .where(KnowledgeConflictRow.customer_id == customer_id)
            )
            if status is not None:
                stmt = stmt.where(KnowledgeConflictRow.status == status)
            return int((await session.execute(stmt)).scalar_one() or 0)

    async def mark_knowledge_conflict_resolved(
        self,
        conflict_id: str,
        *,
        resolution: str,
        resolved_at: datetime,
    ) -> None:
        """Curation gate / operator settled a conflict. `resolution` is one
        of qualified | superseded | blacklisted. NOT called by the scan
        (surfacing-only, D5)."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(KnowledgeConflictRow, conflict_id)
            if row is not None:
                row.status = "resolved"
                row.resolution = resolution
                row.resolved_at = resolved_at

    async def mark_knowledge_conflict_dismissed(
        self,
        conflict_id: str,
        *,
        resolved_at: datetime,
    ) -> None:
        """Operator judged this not a real conflict. Terminal — the pair_key
        stays recorded so the scan never re-surfaces it. NOT called by the
        scan (surfacing-only, D5)."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(KnowledgeConflictRow, conflict_id)
            if row is not None:
                row.status = "dismissed"
                row.resolution = "dismissed"
                row.resolved_at = resolved_at

    async def apply_conflict_resolution(
        self,
        conflict_id: str,
        *,
        resolution: str,
        resolved_at: datetime,
        loser: Optional[str] = None,
    ) -> Optional[KnowledgeConflict]:
        """Curation gate (B2): settle a conflict AND apply its effect to the
        bank, atomically. The operator's choice IS the gate — no second review
        queue. One session, so the fact effect + the status transition commit
        together (never a deactivated fact with a still-open conflict).

        resolution:
          dismissed   — not a real conflict. No fact effect; status→dismissed.
          qualified   — both true under different conditions. Keep BOTH facts
                        active; status→resolved. (The qualifier-attachment
                        mechanism — an UNLESS-clause conditioning the two
                        claims so they formally coexist — is a deferred
                        follow-up, B2.1; v1 just closes the conflict.)
          superseded  — `loser` ('a'|'b') is outdated. Deactivate it from
                        retrieval (grating_strength→0, reversible); the winner
                        is untouched. status→resolved.
          blacklisted — `loser` ('a'|'b') is wrong. Deactivate it AND record a
                        blacklisted_reflections row for its claim so it is not
                        re-learned or re-surfaced. status→resolved.

        NON-DESTRUCTIVE (fork 1): nothing is hard-deleted — a deactivated fact
        stays in the bank at grating 0 and can be restored. superseded /
        blacklisted require `loser`; qualified / dismissed ignore it.

        Returns the updated conflict, or None if no conflict matched (caller
        404s). Raises ValueError on an unknown resolution, or a missing loser
        where one is required (caller 400s). NOT called by the scan
        (surfacing-only, D5).
        """
        import hashlib
        import uuid

        valid = {"dismissed", "qualified", "superseded", "blacklisted"}
        if resolution not in valid:
            raise ValueError(
                f"resolution must be one of {sorted(valid)}; got {resolution!r}"
            )
        needs_loser = resolution in {"superseded", "blacklisted"}
        if needs_loser and loser not in {"a", "b"}:
            raise ValueError(
                f"resolution {resolution!r} requires loser='a' or 'b'"
            )

        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(KnowledgeConflictRow, conflict_id)
            if row is None:
                return None

            if needs_loser:
                loser_fact_id = row.fact_a_id if loser == "a" else row.fact_b_id
                loser_claim = row.claim_a if loser == "a" else row.claim_b

                # Deactivate the losing fact from retrieval (reversible). The
                # fact may already be gone under REPLACE — then this is a no-op
                # (effectively already deactivated); the claim snapshot on the
                # conflict row still drives the blacklist below.
                fact = await session.get(FactRow, loser_fact_id)
                if fact is not None:
                    fact.grating_strength = 0.0

                if resolution == "blacklisted":
                    # Record the wrong claim so learning/the scan won't bring
                    # it back. Keyed by a hash of the claim (the
                    # add_blacklisted_reflection convention); pre-checked so a
                    # re-click doesn't duplicate the row.
                    rhash = hashlib.sha256(
                        (loser_claim or "").encode("utf-8")
                    ).hexdigest()[:64]
                    already = (
                        await session.execute(
                            select(BlacklistedReflectionRow.id)
                            .where(
                                BlacklistedReflectionRow.customer_id
                                == row.customer_id
                            )
                            .where(
                                BlacklistedReflectionRow.reflection_hash == rhash
                            )
                            .limit(1)
                        )
                    ).scalar_one_or_none()
                    if already is None:
                        session.add(BlacklistedReflectionRow(
                            id=uuid.uuid4().hex[:16],
                            customer_id=row.customer_id,
                            reflection_hash=rhash,
                            reflection_text=(loser_claim or ""),
                            reason=f"conflict {conflict_id}: blacklisted losing claim",
                        ))

            row.status = "dismissed" if resolution == "dismissed" else "resolved"
            row.resolution = resolution
            row.resolved_at = resolved_at
            return _knowledge_conflict_from_row(row)


# ---------------------------------------------------------------------------
# Row → Pydantic converter (mirrors metadata_store_audit._knowledge_gap_from_row)
# ---------------------------------------------------------------------------

def _knowledge_conflict_from_row(
    row: KnowledgeConflictRow,
) -> KnowledgeConflict:
    return KnowledgeConflict(
        id=row.id,
        customer_id=row.customer_id,
        fact_a_id=row.fact_a_id,
        fact_b_id=row.fact_b_id,
        crystal_a_id=row.crystal_a_id,
        crystal_b_id=row.crystal_b_id,
        subject=row.subject,
        claim_a=row.claim_a,
        claim_b=row.claim_b,
        provenance_a=row.provenance_a,
        provenance_b=row.provenance_b,
        detector=row.detector,  # type: ignore[arg-type]
        status=row.status,  # type: ignore[arg-type]
        resolution=row.resolution,  # type: ignore[arg-type]
        pair_key=row.pair_key,
        created_at=row.created_at,
        resolved_at=row.resolved_at,
    )
