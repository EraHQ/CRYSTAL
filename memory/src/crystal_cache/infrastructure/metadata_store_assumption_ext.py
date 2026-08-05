"""Assumption store reads/writes — the Assumptions worker's substrate.

Slice 1 of the Assumptions build (design ratified 2026-07-20, refreshed
2026-07-31: RQ1=B own worker role, RQ2=C supersession deferred, RQ3=B
ledger origin from birth; Q1=B type registered create-if-missing with
scope='customer'; Q2=B crystal-row origin='assumptions').

An assumption is an ORDINARY CRYSTAL born epistemically humble:
`crystal_type="assumption"`, `quality_tier="quarantine"`,
`recall_gated=True`, `origin="assumptions"` — all existing columns,
zero migrations. Parentage is TWO CrystalChainRow edges
(assumption -> parentA, assumption -> parentB, direction
source_uses_target), which makes parent-death invalidation one join
and gives recall the parents' facts in the cleanup codebook when the
assumption is eventually promoted. `parent_crystal_id` carries the
PRIMARY parent for lineage display.

The direct-write shape mirrors `add_reflection_fact` (metadata_store.py):
an assumption must NEVER bond into an existing knowledge crystal — it is
its own reviewable unit — so this bypasses the bonder entirely and
constructs the rows itself. Vectors ride the serialized encoder lane
(encoding/executor), same as every other write path.

Same binding pattern as GapExtensionsMixin: this mixin is NOT in
MetadataStore's MRO — infrastructure/__init__.py setattrs its public
methods onto MetadataStore via _bind_mixin_methods. `self.session()`
and `self.add_chain(...)` inside a bound method resolve on the
MetadataStore instance by normal attribute lookup.

R9: ALL assumption SQL lives in this file. The scan (scan/assumptions.py)
and the future `assume` agent tool call these methods only.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog
from sqlalchemy import select
from sqlalchemy.orm import aliased

from ..models import CrystalChain, CrystalType
from .schema import CrystalChainRow, CrystalRow, FactRow

logger = structlog.get_logger(__name__)

# The one type id for assumption crystals. Bare (no `customer:` prefix)
# per the reflection-crystal precedent; registered in the crystal_types
# registry create-if-missing (Q1=B) so discovery surfaces that list
# types from the registry can see it — the 2026-06-12 invisible-banks
# lesson.
ASSUMPTION_CRYSTAL_TYPE = "assumption"


class AssumptionExtensionsMixin:
    """Assumption reads/writes, bound onto MetadataStore."""

    async def list_chained_crystal_pairs(
        self,
        customer_id: str,
        *,
        limit: int = 50,
        exclude_types: tuple[str, ...] = (ASSUMPTION_CRYSTAL_TYPE,),
    ) -> list[tuple[str, str]]:
        """Distinct canonicalized crystal-id pairs joined by a chain edge.

        The Q1=C pairing input: two crystals an author (ingestion,
        curation, or a prior worker) explicitly connected are the
        highest-signal candidates for a bridging assumption. Both
        endpoints must belong to `customer_id`; either endpoint whose
        crystal_type is in `exclude_types` drops the pair — by default
        that excludes assumptions themselves, so the worker never
        compounds speculation on top of quarantined speculation.

        Canonicalization: bidirectional chains are stored as TWO rows
        (audit fix #7), so (A,B) and (B,A) both exist for them; each
        pair is reduced to its sorted-id form and returned once.
        Ordering is newest-edge-first (recency bias, matching the
        pairwise scans). `limit` bounds the RETURNED pair count; the
        underlying fetch reads up to 2x limit rows to absorb the
        two-row bidirectional collapse.
        """
        src = aliased(CrystalRow)
        tgt = aliased(CrystalRow)
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(
                    CrystalChainRow.source_crystal_id,
                    CrystalChainRow.target_crystal_id,
                )
                .join(src, src.id == CrystalChainRow.source_crystal_id)
                .join(tgt, tgt.id == CrystalChainRow.target_crystal_id)
                .where(src.customer_id == customer_id)
                .where(tgt.customer_id == customer_id)
                .order_by(CrystalChainRow.created_at.desc())
                .limit(max(limit, 1) * 2)
            )
            if exclude_types:
                stmt = stmt.where(src.crystal_type.notin_(exclude_types))
                stmt = stmt.where(tgt.crystal_type.notin_(exclude_types))
            rows = (await session.execute(stmt)).all()

        pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for source_id, target_id in rows:
            canonical = (
                (source_id, target_id)
                if source_id <= target_id
                else (target_id, source_id)
            )
            if canonical in seen:
                continue
            seen.add(canonical)
            pairs.append(canonical)
            if len(pairs) >= limit:
                break
        return pairs

    async def find_assumption_for_parents(
        self,
        customer_id: str,
        parent_a_id: str,
        parent_b_id: str,
    ) -> Optional[str]:
        """The id of an existing assumption crystal whose chain edges hit
        BOTH parents, or None.

        The idempotency read: a re-scan over an unchanged bank must not
        pile duplicate assumptions onto the same parent pair. Order-
        insensitive — the two aliased joins cover (a,b) and (b,a) alike
        because each join only requires that SOME outgoing edge reaches
        the named parent.
        """
        edge_a = aliased(CrystalChainRow)
        edge_b = aliased(CrystalChainRow)
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(CrystalRow.id)
                .join(edge_a, edge_a.source_crystal_id == CrystalRow.id)
                .join(edge_b, edge_b.source_crystal_id == CrystalRow.id)
                .where(CrystalRow.customer_id == customer_id)
                .where(CrystalRow.crystal_type == ASSUMPTION_CRYSTAL_TYPE)
                .where(
                    edge_a.target_crystal_id.in_(
                        (parent_a_id, parent_b_id)
                    )
                )
                .where(
                    edge_b.target_crystal_id.in_(
                        (parent_a_id, parent_b_id)
                    )
                )
                .where(
                    edge_a.target_crystal_id != edge_b.target_crystal_id
                )
                .limit(1)
            )
            found = (await session.execute(stmt)).scalar_one_or_none()
            return found

    async def create_assumption_crystal(
        self,
        customer_id: str,
        *,
        statement: str,
        subject: str,
        parent_a_id: str,
        parent_b_id: str,
        confidence: float,
        encoder,
        gap_id: Optional[str] = None,
    ) -> dict[str, str]:
        """Write one assumption crystal: row + fact + two parent edges.

        Birth fields (ratified): crystal_type='assumption',
        quality_tier='quarantine', recall_gated=True,
        origin='assumptions' (Q2=B), parent_crystal_id=parent_a_id
        (the PRIMARY parent, for lineage display; the full parentage
        lives in the two chain edges).

        Zero-migration bookkeeping: `confidence` and the seeding gap id
        land in `diagnostic_tags` (the schema's open tag list) as
        'assumption_confidence:<x.xx>' and 'assumption_gap:<id>' — no
        new column, readable by the Inspector and the promotion
        machinery.

        The fact indexes the STATEMENT (encode_native) — retrieval by
        meaning — while `prompt_text` carries the 'Assumptions|<subject>'
        key so key_scan finds the namespace, mirroring the Reflections|
        precedent.

        Raises ValueError when either parent is missing, belongs to a
        different tenant, or the parents are the same crystal (the
        chain layer would reject the self-loop anyway; failing before
        any write keeps the operation atomic-in-effect). Callers with
        fail-safe semantics (the scan) catch and log.
        """
        from ..encoding.executor import encode_async, encode_native_async

        if parent_a_id == parent_b_id:
            raise ValueError(
                "assumption parents must be two distinct crystals; got "
                f"{parent_a_id!r} twice"
            )

        # Q1=B: register the type create-if-missing so registry-driven
        # surfaces see it. Create-only — never overwrite an operator-
        # customized display_name (general-bank import precedent).
        if await self.get_crystal_type(  # type: ignore[attr-defined]
            ASSUMPTION_CRYSTAL_TYPE
        ) is None:
            await self.upsert_crystal_type(  # type: ignore[attr-defined]
                CrystalType(
                    id=ASSUMPTION_CRYSTAL_TYPE,
                    display_name="Assumptions",
                    scope="customer",
                )
            )
            logger.info(
                "assumptions.type_registered",
                crystal_type=ASSUMPTION_CRYSTAL_TYPE,
            )

        now = datetime.now(timezone.utc)
        crystal_id = f"asm_{uuid.uuid4().hex[:16]}"
        tags = [f"assumption_confidence:{confidence:.2f}"]
        if gap_id:
            tags.append(f"assumption_gap:{gap_id}")

        summary_vec = await encode_async(encoder, statement)
        fact_vec = await encode_native_async(encoder, statement)

        async with self.session() as session:  # type: ignore[attr-defined]
            # Tenancy + existence check on both parents in one read.
            parent_rows = (await session.execute(
                select(CrystalRow.id, CrystalRow.customer_id).where(
                    CrystalRow.id.in_((parent_a_id, parent_b_id))
                )
            )).all()
            by_id = {r.id: r.customer_id for r in parent_rows}
            for pid in (parent_a_id, parent_b_id):
                if pid not in by_id:
                    raise ValueError(
                        f"assumption parent {pid!r} does not exist"
                    )
                if by_id[pid] != customer_id:
                    raise ValueError(
                        f"assumption parent {pid!r} belongs to a "
                        "different tenant"
                    )

            session.add(CrystalRow(
                id=crystal_id,
                customer_id=customer_id,
                summary_vector=[float(x) for x in summary_vec],
                summary_text=statement,
                crystal_type=ASSUMPTION_CRYSTAL_TYPE,
                quality_tier="quarantine",
                recall_gated=True,
                origin="assumptions",
                source_kind="agent_inferred",
                build_method="assumption",
                parent_crystal_id=parent_a_id,
                diagnostic_tags=tags,
                fact_count=1,
                created_at=now,
                last_activity=now,
            ))
            session.add(FactRow(
                id=f"asf_{uuid.uuid4().hex}",
                crystal_id=crystal_id,
                pair_type="question_answer",
                prompt_text=f"Assumptions|{subject}",
                claim_text=statement,
                source_kind="agent_inferred",
                vector=[float(x) for x in fact_vec],
            ))
            await session.commit()

        # Parentage edges via the existing chain primitive (self-loop
        # guard, idempotent upsert, direction normalization all reused).
        for parent_id in (parent_a_id, parent_b_id):
            await self.add_chain(CrystalChain(  # type: ignore[attr-defined]
                source_crystal_id=crystal_id,
                target_crystal_id=parent_id,
                direction="source_uses_target",
                created_at=now,
            ))

        logger.info(
            "assumptions.crystal_created",
            customer_id=customer_id,
            crystal_id=crystal_id,
            parent_a=parent_a_id,
            parent_b=parent_b_id,
            confidence=confidence,
            gap_id=gap_id,
        )
        return {"crystal_id": crystal_id, "parent_a": parent_a_id,
                "parent_b": parent_b_id}

    async def sweep_orphaned_assumptions(self, *, limit: int = 50) -> int:
        """Invalidate assumptions whose chain edges point at crystals
        that no longer exist — the out-of-band-death safety net
        (Q3=B, slice 4).

        The capture-at-delete inside delete_crystal handles every
        normal death; on Postgres the chain FKs make a dangling edge
        unreachable via SQL deletes (the D2 note), so this sweep
        covers dev/SQLite deletions and FK-disabled surgery. Same
        marking as capture: quality_tier='blacklist' + the
        'assumption_invalidated:parent:<id>' audit tag (deduped),
        NULL a dangling primary-parent FK, and DELETE the dangling
        edge rows — which is what makes the sweep idempotent (a swept
        edge is gone; a rerun finds nothing).

        Global by design (a system-integrity pass, the reclaim
        posture), bounded by `limit` edges per call. Returns the
        number of assumption crystals invalidated this pass.
        """
        tgt = aliased(CrystalRow)
        invalidated: set[str] = set()
        async with self.session() as session:  # type: ignore[attr-defined]
            pairs = (await session.execute(
                select(CrystalRow, CrystalChainRow)
                .join(
                    CrystalChainRow,
                    CrystalChainRow.source_crystal_id == CrystalRow.id,
                )
                .outerjoin(
                    tgt, tgt.id == CrystalChainRow.target_crystal_id
                )
                .where(CrystalRow.crystal_type == ASSUMPTION_CRYSTAL_TYPE)
                .where(tgt.id.is_(None))
                .limit(max(limit, 1))
            )).all()
            for row, edge in pairs:
                dead_id = edge.target_crystal_id
                row.quality_tier = "blacklist"
                tags = list(row.diagnostic_tags or [])
                tag = f"assumption_invalidated:parent:{dead_id}"
                if tag not in tags:
                    tags.append(tag)
                row.diagnostic_tags = tags
                if row.parent_crystal_id == dead_id:
                    row.parent_crystal_id = None
                invalidated.add(row.id)
                await session.delete(edge)
        if invalidated:
            logger.info(
                "assumptions.sweep_invalidated", count=len(invalidated)
            )
        return len(invalidated)

    async def list_assumption_crystals(
        self, customer_id: str, *, limit: int = 200,
    ) -> list[dict]:
        """All of a customer's assumption crystals, newest first — the
        Inspector review-surface read (slice 5). Includes BOTH pending
        (quarantine, recall-gated) and invalidated (blacklist) rows;
        the caller renders state from quality_tier/recall_gated and
        parses confidence/provenance from diagnostic_tags.

        Returns trimmed dicts (the headline_facts precedent) — the
        review list needs seven fields, not 10k-dim vectors.
        """
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(
                select(
                    CrystalRow.id,
                    CrystalRow.summary_text,
                    CrystalRow.quality_tier,
                    CrystalRow.recall_gated,
                    CrystalRow.diagnostic_tags,
                    CrystalRow.parent_crystal_id,
                    CrystalRow.created_at,
                )
                .where(CrystalRow.customer_id == customer_id)
                .where(CrystalRow.crystal_type == ASSUMPTION_CRYSTAL_TYPE)
                .order_by(CrystalRow.created_at.desc(), CrystalRow.id.desc())
                .limit(max(limit, 1))
            )).all()
        return [
            {
                "id": r.id,
                "statement": r.summary_text,
                "quality_tier": r.quality_tier,
                "recall_gated": bool(r.recall_gated),
                "diagnostic_tags": list(r.diagnostic_tags or []),
                "parent_crystal_id": r.parent_crystal_id,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]
