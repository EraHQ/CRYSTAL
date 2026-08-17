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
from .schema import (
    CitationRow, CrystalChainRow, CrystalEdgeRow, CrystalRow, FactRow,
    KnowledgeGapRow, QueryLogRow,
)

logger = structlog.get_logger(__name__)

# The one type id for assumption crystals. Bare (no `customer:` prefix)
# per the reflection-crystal precedent; registered in the crystal_types
# registry create-if-missing (Q1=B) so discovery surfaces that list
# types from the registry can see it — the 2026-06-12 invisible-banks
# lesson.
ASSUMPTION_CRYSTAL_TYPE = "assumption"


def parse_assumption_tags(tags: list[str]) -> dict:
    """Presentation split of the assumption diagnostic tags:
    'assumption_confidence:<x.xx>' -> float, 'assumption_gap:<id>' ->
    seeding provenance, 'assumption_invalidated:parent:<id>' -> the
    dead parents. Unknown tags pass through untouched in the raw list
    the caller already has.

    C1 (2026-08-07): promoted from endpoints/admin.py to THIS module —
    the ext owns the tag vocabulary (create_assumption_crystal writes
    it), and the retrieval annotation read below is a second consumer.
    Module-level on purpose: _bind_mixin_methods only binds the mixin
    CLASS's callables onto MetadataStore, so a free function stays a
    plain import surface.
    """
    confidence = None
    gap_id = None
    invalidated_parents: list[str] = []
    for tag in tags:
        if tag.startswith("assumption_confidence:"):
            try:
                confidence = float(tag.split(":", 1)[1])
            except ValueError:
                pass
        elif tag.startswith("assumption_gap:"):
            gap_id = tag.split(":", 1)[1]
        elif tag.startswith("assumption_invalidated:parent:"):
            invalidated_parents.append(tag.rsplit(":", 1)[1])
    return {
        "confidence": confidence,
        "gap_id": gap_id,
        "invalidated_parents": invalidated_parents,
    }


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

        # C2 Q3=A (2026-08-08): the birth witness — worker/tool writes
        # become visible in the curation activity feed. Best-effort.
        try:
            await self.record_curation_event(  # type: ignore[attr-defined]
                customer_id,
                event_type="assumption_written",
                subject_id=crystal_id,
                label=f"Assumption written - {subject}"[:256],
                payload={
                    "confidence": confidence,
                    "gap_id": gap_id,
                    "parent_a": parent_a_id,
                    "parent_b": parent_b_id,
                },
            )
        except Exception:  # noqa: BLE001 — witness never breaks the write
            logger.debug("curation_event.emit_failed", exc_info=True)
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

    async def tag_assumption_verification(
        self, customer_id: str, crystal_id: str, task_id: str,
    ) -> bool:
        """C4 (2026-08-11): stamp `verification_task:<task_id>` on an
        assumption crystal — the durable once-per-assumption spawn
        record the verification scan's idempotence reads. Tenant-
        guarded; returns False when the crystal isn't this tenant's
        assumption. Appends (never replaces): a manual respawn adds a
        second tag and the history stays legible."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = (await session.execute(
                select(CrystalRow)
                .where(CrystalRow.id == crystal_id)
                .where(CrystalRow.customer_id == customer_id)
                .where(CrystalRow.crystal_type == ASSUMPTION_CRYSTAL_TYPE)
            )).scalar_one_or_none()
            if row is None:
                return False
            tags = list(row.diagnostic_tags or [])
            tag = f"verification_task:{task_id}"
            if tag not in tags:
                tags.append(tag)
            row.diagnostic_tags = tags
            await session.commit()
            return True

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

    async def list_assumption_annotations(
        self, customer_id: str, crystal_ids: list[str],
    ) -> dict[str, dict]:
        """C1 (ratified 2026-08-07, Q1=C): {assumption_crystal_id:
        annotation} for whichever of the given crystals are assumption
        crystals — the retrieval-time framing read behind
        tier_signal.assumption_note. Empty dict when none are.

        TWO batched selects regardless of result-set size (the hot
        path already runs two fail-safe annotation reads — tiers and
        conflicts — this stays in that cost class, never the admin
        surface's per-crystal hydration): (1) id/tags/tier for the
        assumption-typed subset, tenant-guarded; (2) live parents via
        chains joined to the parent crystal for summary_text, the
        parent side tenant-guarded too (parents are the tenant's own
        bank by design — slice-3 hardening posture). Dead parents come
        from the audit tags; capture-at-delete removed their edges.

        Annotation shape per id: {"quality_tier", "confidence",
        "gap_id", "invalidated_parents": [crystal_id],
        "parents": [{"id", "summary_text"}]}.
        """
        ids = [c for c in (crystal_ids or []) if c]
        if not ids:
            return {}
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(
                select(
                    CrystalRow.id,
                    CrystalRow.quality_tier,
                    CrystalRow.diagnostic_tags,
                )
                .where(CrystalRow.id.in_(ids))
                .where(CrystalRow.customer_id == customer_id)
                .where(CrystalRow.crystal_type == ASSUMPTION_CRYSTAL_TYPE)
            )).all()
            if not rows:
                return {}
            annotations: dict[str, dict] = {}
            for r in rows:
                parsed = parse_assumption_tags(list(r.diagnostic_tags or []))
                annotations[r.id] = {
                    "quality_tier": r.quality_tier,
                    "confidence": parsed["confidence"],
                    "gap_id": parsed["gap_id"],
                    "invalidated_parents": parsed["invalidated_parents"],
                    "parents": [],
                }
            parent = aliased(CrystalRow)
            chain_rows = (await session.execute(
                select(
                    CrystalChainRow.source_crystal_id,
                    parent.id,
                    parent.summary_text,
                )
                .join(parent, parent.id == CrystalChainRow.target_crystal_id)
                .where(CrystalChainRow.source_crystal_id.in_(list(annotations)))
                .where(parent.customer_id == customer_id)
            )).all()
            for cr in chain_rows:
                annotations[cr.source_crystal_id]["parents"].append({
                    "id": cr.id,
                    "summary_text": cr.summary_text,
                })
        return annotations

    async def close_gap_for_approved_assumption(
        self, crystal_id: str, customer_id: str,
    ) -> Optional[str]:
        """C2 Q1=A (2026-08-08): approval closes the seeding gap.

        The moment an assumption becomes recallable (approve = the
        ratified promotion act), the question it answers stops being
        open — otherwise the fill sweep can spend research budget on a
        question the bank already answers. Guarded: the crystal must be
        this tenant's assumption carrying assumption_gap provenance,
        and the gap must be this tenant's and still 'open' (a gap
        filled by anything else is never overwritten). Returns the
        gap_id when it closed, else None. The caller (approve endpoint)
        emits the witness events.
        """
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(CrystalRow, crystal_id)
            if (
                row is None
                or row.customer_id != customer_id
                or row.crystal_type != ASSUMPTION_CRYSTAL_TYPE
            ):
                return None
            gap_id = parse_assumption_tags(
                list(row.diagnostic_tags or [])
            )["gap_id"]
            if not gap_id:
                return None
            gap = await session.get(KnowledgeGapRow, gap_id)
            if (
                gap is None
                or gap.customer_id != customer_id
                or gap.status != "open"
            ):
                return None
            gap.status = "filled"
            gap.filled_by_crystal_id = crystal_id
            gap.resolved_at = datetime.now(timezone.utc)
        return gap_id

    # ------------------------------------------------------------------
    # Pairing-funnel substrate (F1, Q6=A) — the crystal_edges writer's
    # reads + the batched upsert. All funnel SQL lives here (R9).
    # ------------------------------------------------------------------

    async def upsert_crystal_edges(
        self, edges: "list[tuple[str, str, str, float]]",
    ) -> int:
        """Batch-upsert (crystal_a_id, crystal_b_id, edge_type,
        weight_delta) tuples. Composite-PK accumulate: an existing edge
        gains weight and a fresh last_reinforced_at; a new one is
        inserted. Callers pass CANONICAL a<=b ordering — this method
        enforces it defensively so the (a,b,type) PK never splits one
        logical edge into two rows."""
        if not edges:
            return 0
        now = datetime.now(timezone.utc)
        written = 0
        async with self.session() as session:  # type: ignore[attr-defined]
            for a, b, edge_type, weight_delta in edges:
                if a == b:
                    continue
                if a > b:
                    a, b = b, a
                row = await session.get(
                    CrystalEdgeRow, (a, b, edge_type)
                )
                if row is None:
                    session.add(CrystalEdgeRow(
                        crystal_a_id=a,
                        crystal_b_id=b,
                        edge_type=edge_type,
                        weight=weight_delta,
                        last_reinforced_at=now,
                    ))
                else:
                    row.weight = (row.weight or 0.0) + weight_delta
                    row.last_reinforced_at = now
                written += 1
        return written

    async def list_grounded_citations_since(
        self,
        customer_id: str,
        *,
        since: "Optional[datetime]" = None,
        limit: int = 2000,
    ) -> list[dict]:
        """Grounded citations newer than the watermark, oldest first
        (so the caller's watermark advance is monotonic). Trimmed dicts:
        the funnel needs the turn pointer + crystal id, not spans."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(
                    CitationRow.query_log_id,
                    CitationRow.crystal_id,
                    CitationRow.created_at,
                )
                .where(CitationRow.customer_id == customer_id)
                .where(CitationRow.grounded.is_(True))
                .order_by(CitationRow.created_at.asc())
                .limit(max(limit, 1))
            )
            if since is not None:
                stmt = stmt.where(CitationRow.created_at > since)
            rows = (await session.execute(stmt)).all()
        return [
            {
                "query_log_id": r.query_log_id,
                "crystal_id": r.crystal_id,
                "created_at": r.created_at,
            }
            for r in rows
        ]

    async def list_query_routings_since(
        self,
        customer_id: str,
        *,
        since: "Optional[datetime]" = None,
        limit: int = 2000,
    ) -> list[dict]:
        """Per-turn routing observations newer than the watermark,
        oldest first: the conversation anchor (sequence_id), the routed
        crystal, and the retrieved fact ids (matched_facts) — the
        co-routing tier's raw material. JSON list unnesting happens in
        Python at the caller (dialect-safe; no JSON SQL)."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(
                    QueryLogRow.sequence_id,
                    QueryLogRow.routed_crystal_id,
                    QueryLogRow.matched_facts,
                    QueryLogRow.timestamp,
                )
                .where(QueryLogRow.customer_id == customer_id)
                .order_by(QueryLogRow.timestamp.asc())
                .limit(max(limit, 1))
            )
            if since is not None:
                stmt = stmt.where(QueryLogRow.timestamp > since)
            rows = (await session.execute(stmt)).all()
        return [
            {
                "sequence_id": r.sequence_id,
                "routed_crystal_id": r.routed_crystal_id,
                "matched_facts": list(r.matched_facts or []),
                "timestamp": r.timestamp,
            }
            for r in rows
        ]

    async def list_crystal_pairing_info(
        self, customer_id: str, *, limit: int = 300,
    ) -> list[dict]:
        """Every crystal's pairing-relevant fields: id, type (assumption
        exclusion happens at the funnel), and the stored routing_vector
        for the vector_similar tier (None for pre-6.3 crystals — the
        funnel skips them there). Bounded; id-ordered for the canonical
        structural enumeration."""
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(
                select(
                    CrystalRow.id,
                    CrystalRow.crystal_type,
                    CrystalRow.routing_vector,
                )
                .where(CrystalRow.customer_id == customer_id)
                .order_by(CrystalRow.id.asc())
                .limit(max(limit, 1))
            )).all()
        return [
            {
                "id": r.id,
                "crystal_type": r.crystal_type,
                "routing_vector": (
                    list(r.routing_vector) if r.routing_vector else None
                ),
            }
            for r in rows
        ]

    async def list_candidate_edges(
        self, customer_id: str, *, limit: int = 500,
    ) -> list[dict]:
        """The funnel graph for one customer — the F2 spend queue's
        input. Tenant-scoped via the crystal join on BOTH endpoints
        (funnel emit already guarantees same-tenant pairs; the join is
        defense in depth). Trimmed dicts; tier ordering happens at the
        caller against EDGE_TIER_ORDER (Python sort — no dialect CASE
        gymnastics)."""
        src = aliased(CrystalRow)
        tgt = aliased(CrystalRow)
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(
                select(
                    CrystalEdgeRow.crystal_a_id,
                    CrystalEdgeRow.crystal_b_id,
                    CrystalEdgeRow.edge_type,
                    CrystalEdgeRow.weight,
                    CrystalEdgeRow.last_reinforced_at,
                )
                .join(src, src.id == CrystalEdgeRow.crystal_a_id)
                .join(tgt, tgt.id == CrystalEdgeRow.crystal_b_id)
                .where(src.customer_id == customer_id)
                .where(tgt.customer_id == customer_id)
                .order_by(CrystalEdgeRow.weight.desc())
                .limit(max(limit, 1))
            )).all()
        return [
            {
                "crystal_a_id": r.crystal_a_id,
                "crystal_b_id": r.crystal_b_id,
                "edge_type": r.edge_type,
                "weight": float(r.weight or 0.0),
            }
            for r in rows
        ]
