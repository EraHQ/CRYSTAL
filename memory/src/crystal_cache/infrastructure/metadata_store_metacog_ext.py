"""Metacognitive artifact CRUD — Phase 10A + 10B (2026-05-27).

The SIXTH mixin on MetadataStore (per D12, AN-7 pattern). Lands the
CRUD surface for the metacognitive artifact tables introduced in
Phase 10A (`item_alignments`, `critique_syntheses`) and Phase 10B
(`critic_calibrations`), plus the trace-eligibility query methods
the Phase 10B scheduler worker needs.

This mixin is conceptually distinct from McrExtensionsMixin (Phase
8.5) per P0.70:
  - McrExtensionsMixin owns what CRITICS write (traces, critiques,
    action items).
  - MetacognitionExtensionsMixin owns what the METACOGNITIVE LAYER
    writes (alignments, syntheses, calibrations) AND the worker-
    eligibility scans.

The Phase 10 `metacognition/` package consumes McrExtensionsMixin's
reads (list_critiques_for_trace, list_action_items_for_critique,
update_action_item_status) and writes through THIS mixin's methods.

CU-18 note: this is the sixth mixin file, so CLAUDE.md R9's prose
"FIVE metadata_store files" is now stale; the rule itself is
count-agnostic.

Method surface:

  Item alignments (Phase 10A):
    create_item_alignment(...)               — write a new alignment
    list_alignments_for_trace(trace_id)      — primary read
    get_alignment_for_item(focus_item_id)    — per-item lookup

  Critique syntheses (Phase 10A):
    create_critique_synthesis(...)           — write a new synthesis
    list_syntheses_for_trace(trace_id)       — newest-first
    list_syntheses_for_customer(...)         — chronological scan

  Scheduler queries (Phase 10B — P0.77):
    list_traces_needing_shadow_review(...)   — agent_self-only traces
    list_traces_needing_synthesis(...)       — un-synthesized traces

  Critic calibrations (Phase 10B — P0.78):
    upsert_critic_calibration(...)           — select-then-update-or-insert
    get_critic_calibration(...)              — by (customer, role, model)
    list_calibrations_for_customer(...)      — all critics for a customer

  Substrate review (Phase 10.5 — P0.91):
    list_substrate_action_items(...)         — deferred substrate observations
                                              for human review (D-MCR-13 V1)

All methods route through `self.session()` per R9. None bypass the
store layer. Each method is bound to MetadataStore via
`_bind_mixin_methods` in `infrastructure/__init__.py`.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import structlog
from sqlalchemy import select

from ..models.critic_calibration import CriticCalibration
from ..models.critique_synthesis import CritiqueSynthesis
from ..models.item_alignment import ItemAlignment
from ..models.action_item import ActionItem
from .schema import (
    ActionItemRow,
    CriticCalibrationRow,
    CritiqueRow,
    CritiqueSynthesisRow,
    ItemAlignmentRow,
    ReasoningTraceRow,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Row → Pydantic converters
# ---------------------------------------------------------------------------

def _alignment_from_row(row: ItemAlignmentRow) -> ItemAlignment:
    return ItemAlignment(
        id=row.id,
        customer_id=row.customer_id,
        trace_id=row.trace_id,
        focus_item_id=row.focus_item_id,
        alignment_class=row.alignment_class,  # type: ignore[arg-type]
        paired_item_ids=list(row.paired_item_ids or []),
        confidence=row.confidence,
        computed_at=row.computed_at,
    )


def _synthesis_from_row(row: CritiqueSynthesisRow) -> CritiqueSynthesis:
    return CritiqueSynthesis(
        id=row.id,
        customer_id=row.customer_id,
        trace_id=row.trace_id,
        review_window_start=row.review_window_start,
        review_window_end=row.review_window_end,
        promoted_item_ids=list(row.promoted_item_ids or []),
        deferred_item_ids=list(row.deferred_item_ids or []),
        dropped_item_ids=list(row.dropped_item_ids or []),
        promotion_rationales=dict(row.promotion_rationales or {}),
        critic_calibration_updates=list(row.critic_calibration_updates or []),
        cross_trace_patterns=list(row.cross_trace_patterns or []),
        created_at=row.created_at,
    )


def _calibration_from_row(row: CriticCalibrationRow) -> CriticCalibration:
    return CriticCalibration(
        id=row.id,
        customer_id=row.customer_id,
        critic_role=row.critic_role,
        critic_model=row.critic_model,
        total_proposals=row.total_proposals or 0,
        promoted_count=row.promoted_count or 0,
        deferred_count=row.deferred_count or 0,
        dropped_count=row.dropped_count or 0,
        last_synthesis_at=row.last_synthesis_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class MetacognitionExtensionsMixin:
    """Metacognitive artifact CRUD methods bound onto MetadataStore.

    Bound at import time via setattr in `infrastructure/__init__.py`
    via the shared `_bind_mixin_methods` helper. Same pattern as
    `McrExtensionsMixin` and the four other Phase-1-thru-9 mixins.

    All methods are async and route through `self.session()` per R9.
    """

    # ====================================================================
    # Item alignments
    # ====================================================================

    async def create_item_alignment(
        self,
        customer_id: str,
        focus_item_id: str,
        alignment_class: str,
        *,
        trace_id: Optional[str] = None,
        paired_item_ids: Optional[list[str]] = None,
        confidence: Optional[float] = None,
    ) -> ItemAlignment:
        """Create an item-alignment record for one focus action item.

        Per P0.71: one row per (trace, focus_item). Callers compute
        the classification via `metacognition.alignment.classify_pair`
        and pass the result here for persistence.
        """
        alignment_id = uuid.uuid4().hex[:16]
        async with self.session() as session:  # type: ignore[attr-defined]
            row = ItemAlignmentRow(
                id=alignment_id,
                customer_id=customer_id,
                trace_id=trace_id,
                focus_item_id=focus_item_id,
                alignment_class=alignment_class,
                paired_item_ids=list(paired_item_ids or []),
                confidence=confidence,
            )
            session.add(row)
            await session.flush()
            return _alignment_from_row(row)

    async def list_alignments_for_trace(
        self,
        trace_id: str,
    ) -> list[ItemAlignment]:
        """List alignment rows for a trace, oldest-first.

        The primary read pattern: the metacognitive layer's synthesis
        step walks a trace's alignments in deterministic order to
        produce a synthesis row.
        """
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(ItemAlignmentRow)
                .where(ItemAlignmentRow.trace_id == trace_id)
                .order_by(ItemAlignmentRow.computed_at.asc())
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [_alignment_from_row(r) for r in rows]

    async def get_alignment_for_item(
        self,
        focus_item_id: str,
    ) -> Optional[ItemAlignment]:
        """Get the alignment record for a specific action item.

        Returns the most recent if multiple exist (which can happen
        if a trace is re-synthesized — Phase 10B's scheduler may
        re-compute alignments as critic calibrations shift).
        Returns None when no alignment has been computed yet.
        """
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(ItemAlignmentRow)
                .where(ItemAlignmentRow.focus_item_id == focus_item_id)
                .order_by(ItemAlignmentRow.computed_at.desc())
                .limit(1)
            )
            row = (await session.execute(stmt)).scalar_one_or_none()
            return _alignment_from_row(row) if row is not None else None

    # ====================================================================
    # Critique syntheses
    # ====================================================================

    async def create_critique_synthesis(
        self,
        customer_id: str,
        *,
        trace_id: Optional[str] = None,
        review_window_start: Optional[datetime] = None,
        review_window_end: Optional[datetime] = None,
        promoted_item_ids: Optional[list[str]] = None,
        deferred_item_ids: Optional[list[str]] = None,
        dropped_item_ids: Optional[list[str]] = None,
        promotion_rationales: Optional[dict[str, str]] = None,
        critic_calibration_updates: Optional[list[dict[str, Any]]] = None,
        cross_trace_patterns: Optional[list[dict[str, Any]]] = None,
    ) -> CritiqueSynthesis:
        """Create a critique-synthesis row recording metacognitive decisions.

        Per P0.72: one row per (trace, review_window). Re-syntheses
        APPEND new rows rather than updating existing ones, preserving
        the audit trail per D-MCR-7 (critics are fallible — past
        decisions stay legible).

        The three lists (promoted/deferred/dropped) must be mutually
        exclusive within a single synthesis row. The Pydantic model
        doesn't enforce this; the caller (Phase 10A's synthesis
        algorithm) is responsible for valid assignments.
        """
        synthesis_id = uuid.uuid4().hex[:16]
        async with self.session() as session:  # type: ignore[attr-defined]
            row = CritiqueSynthesisRow(
                id=synthesis_id,
                customer_id=customer_id,
                trace_id=trace_id,
                review_window_start=review_window_start,
                review_window_end=review_window_end,
                promoted_item_ids=list(promoted_item_ids or []),
                deferred_item_ids=list(deferred_item_ids or []),
                dropped_item_ids=list(dropped_item_ids or []),
                promotion_rationales=dict(promotion_rationales or {}),
                critic_calibration_updates=list(
                    critic_calibration_updates or []
                ),
                cross_trace_patterns=list(cross_trace_patterns or []),
            )
            session.add(row)
            await session.flush()
            return _synthesis_from_row(row)

    async def list_syntheses_for_trace(
        self,
        trace_id: str,
    ) -> list[CritiqueSynthesis]:
        """List synthesis rows for a trace, newest-first.

        Multiple syntheses can exist for one trace when Phase 10B's
        scheduler re-reviews. The newest is the current decision;
        older rows are the audit trail.
        """
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(CritiqueSynthesisRow)
                .where(CritiqueSynthesisRow.trace_id == trace_id)
                .order_by(CritiqueSynthesisRow.created_at.desc())
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [_synthesis_from_row(r) for r in rows]

    async def list_syntheses_for_customer(
        self,
        customer_id: str,
        *,
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> list[CritiqueSynthesis]:
        """List syntheses for a customer, newest-first.

        Feeds Phase 10.5's substrate review surface and the calibration
        scans Phase 10B will need. `since` filters created_at >= since;
        `limit` caps the result count.
        """
        async with self.session() as session:  # type: ignore[attr-defined]
            conditions = [CritiqueSynthesisRow.customer_id == customer_id]
            if since is not None:
                conditions.append(CritiqueSynthesisRow.created_at >= since)
            stmt = (
                select(CritiqueSynthesisRow)
                .where(*conditions)
                .order_by(CritiqueSynthesisRow.created_at.desc())
            )
            if limit is not None:
                stmt = stmt.limit(limit)
            rows = (await session.execute(stmt)).scalars().all()
            return [_synthesis_from_row(r) for r in rows]

    # ====================================================================
    # Scheduler eligibility queries (Phase 10B — P0.77)
    # ====================================================================

    async def list_traces_needing_shadow_review(
        self,
        *,
        customer_id: Optional[str] = None,
        limit: int = 10,
        max_age_hours: int = 24,
    ) -> list[Any]:
        """Traces with an agent_self critique but no shadow critique.

        Returns ReasoningTrace rows oldest-first within limit. The
        `max_age_hours` guard bounds the scan: traces older than this
        are considered stale and handled by manual backfill, not the
        scheduler. customer_id optional for cross-tenant scans.

        Phase 10B worker's shadow pass calls this each cycle.
        """
        agent_self_exists = (
            select(CritiqueRow.id)
            .where(
                CritiqueRow.trace_id == ReasoningTraceRow.id,
                CritiqueRow.critic_role == "agent_self",
            )
            .exists()
        )
        shadow_exists = (
            select(CritiqueRow.id)
            .where(
                CritiqueRow.trace_id == ReasoningTraceRow.id,
                CritiqueRow.critic_role == "shadow",
            )
            .exists()
        )
        cutoff = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)
        from datetime import datetime as _dt
        cutoff_dt = _dt.fromtimestamp(cutoff, tz=timezone.utc)

        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(ReasoningTraceRow)
                .where(agent_self_exists)
                .where(~shadow_exists)
                .where(ReasoningTraceRow.created_at >= cutoff_dt)
                .order_by(ReasoningTraceRow.created_at.asc())
                .limit(limit)
            )
            if customer_id is not None:
                stmt = stmt.where(ReasoningTraceRow.customer_id == customer_id)
            rows = (await session.execute(stmt)).scalars().all()

            # Convert to Pydantic via the existing converter pattern.
            # Imported lazily to avoid circular imports with mcr_ext.
            from .metadata_store_mcr_ext import _trace_from_row
            return [_trace_from_row(r) for r in rows]

    async def list_traces_needing_synthesis(
        self,
        *,
        customer_id: Optional[str] = None,
        limit: int = 20,
        settling_seconds: int = 60,
    ) -> list[Any]:
        """Traces with ≥1 critique, no synthesis, age > settling_seconds.

        The settling guard prevents Pass 2 from running before the
        shadow's LLM call completes. settling_seconds=0 disables
        the guard (used by tests). customer_id optional for cross-
        tenant scans.

        Returns ReasoningTrace rows oldest-first within limit. Phase
        10B worker's synthesis pass calls this each cycle.
        """
        any_critique_exists = (
            select(CritiqueRow.id)
            .where(CritiqueRow.trace_id == ReasoningTraceRow.id)
            .exists()
        )
        synthesis_exists = (
            select(CritiqueSynthesisRow.id)
            .where(CritiqueSynthesisRow.trace_id == ReasoningTraceRow.id)
            .exists()
        )
        settling_cutoff = datetime.now(timezone.utc).timestamp() - settling_seconds
        from datetime import datetime as _dt
        cutoff_dt = _dt.fromtimestamp(settling_cutoff, tz=timezone.utc)

        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(ReasoningTraceRow)
                .where(any_critique_exists)
                .where(~synthesis_exists)
                .where(ReasoningTraceRow.created_at <= cutoff_dt)
                .order_by(ReasoningTraceRow.created_at.asc())
                .limit(limit)
            )
            if customer_id is not None:
                stmt = stmt.where(ReasoningTraceRow.customer_id == customer_id)
            rows = (await session.execute(stmt)).scalars().all()

            from .metadata_store_mcr_ext import _trace_from_row
            return [_trace_from_row(r) for r in rows]

    # ====================================================================
    # Critic calibrations (Phase 10B — P0.78)
    # ====================================================================

    async def upsert_critic_calibration(
        self,
        customer_id: str,
        critic_role: str,
        critic_model: str,
        *,
        promoted_delta: int = 0,
        deferred_delta: int = 0,
        dropped_delta: int = 0,
    ) -> CriticCalibration:
        """Select-then-update-or-insert per P0.78 + P0.79.

        Used by `metacognition.calibration.update_calibrations_from_
        synthesis` after each synthesis row is persisted. Atomic-per-
        row within a single worker cycle (cycle is single-threaded so
        no race concerns); Phase 11+ Postgres deployment may swap for
        INSERT ... ON CONFLICT DO UPDATE.

        Cold-start handling (P0.81): if no row exists for the
        (customer_id, critic_role, critic_model) triple, INSERT a
        new one with counters starting at 0 plus the deltas.
        """
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = select(CriticCalibrationRow).where(
                CriticCalibrationRow.customer_id == customer_id,
                CriticCalibrationRow.critic_role == critic_role,
                CriticCalibrationRow.critic_model == critic_model,
            )
            row = (await session.execute(stmt)).scalar_one_or_none()
            now = datetime.now(timezone.utc)
            delta_total = promoted_delta + deferred_delta + dropped_delta

            if row is None:
                row = CriticCalibrationRow(
                    id=uuid.uuid4().hex[:16],
                    customer_id=customer_id,
                    critic_role=critic_role,
                    critic_model=critic_model,
                    total_proposals=delta_total,
                    promoted_count=promoted_delta,
                    deferred_count=deferred_delta,
                    dropped_count=dropped_delta,
                    last_synthesis_at=now,
                )
                session.add(row)
            else:
                row.total_proposals = (row.total_proposals or 0) + delta_total
                row.promoted_count = (row.promoted_count or 0) + promoted_delta
                row.deferred_count = (row.deferred_count or 0) + deferred_delta
                row.dropped_count = (row.dropped_count or 0) + dropped_delta
                row.last_synthesis_at = now

            await session.flush()
            return _calibration_from_row(row)

    async def get_critic_calibration(
        self,
        customer_id: str,
        critic_role: str,
        critic_model: str,
    ) -> Optional[CriticCalibration]:
        """Get the calibration row for one critic identity. None if absent."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = select(CriticCalibrationRow).where(
                CriticCalibrationRow.customer_id == customer_id,
                CriticCalibrationRow.critic_role == critic_role,
                CriticCalibrationRow.critic_model == critic_model,
            )
            row = (await session.execute(stmt)).scalar_one_or_none()
            return _calibration_from_row(row) if row is not None else None

    async def list_calibrations_for_customer(
        self,
        customer_id: str,
    ) -> list[CriticCalibration]:
        """List all critic calibration rows for a customer.

        Ordered by total_proposals descending so the most-active
        critics surface first. Used by the future calibration
        dashboard (Phase 11+).
        """
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(CriticCalibrationRow)
                .where(CriticCalibrationRow.customer_id == customer_id)
                .order_by(CriticCalibrationRow.total_proposals.desc())
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [_calibration_from_row(r) for r in rows]

    # ====================================================================
    # Substrate review (Phase 10.5 — P0.91)
    # ====================================================================

    async def list_substrate_action_items(
        self,
        *,
        customer_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 50,
    ) -> list[ActionItem]:
        """List deferred substrate_observation action items.

        The substrate review surface (D-MCR-13 V1, MCR §9) reads
        these for human review. Filters are hardcoded to the
        substrate-review use case:
          - status='deferred' (always — substrate items NEVER
            auto-promote per Phase 10A's synthesis Rule 1)
          - action_type='substrate_observation' (always)

        Optional filters:
          - customer_id: scope to one customer; None = cross-tenant
            (operator wants system-wide view)
          - since: only items with created_at >= since
          - limit: cap result count, default 50

        Ordered most-recent-first (deviates from the mixin's general
        oldest-first convention) because the substrate review
        surface is for human consumption — recent observations are
        the most interesting to operators reviewing what their
        agents have been complaining about.

        Cross-tenant scans (customer_id=None) follow the precedent
        of `list_open_knowledge_gaps_cross_tenant` (Phase 6.5).
        """
        from .schema import ActionItemRow as _AIR  # avoid stale alias risk

        conditions = [
            _AIR.action_type == "substrate_observation",
            _AIR.status == "deferred",
        ]
        if customer_id is not None:
            conditions.append(_AIR.customer_id == customer_id)
        if since is not None:
            conditions.append(_AIR.created_at >= since)

        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(_AIR)
                .where(*conditions)
                .order_by(_AIR.created_at.desc())
                .limit(limit)
            )
            rows = (await session.execute(stmt)).scalars().all()

            # Reuse the MCR mixin's converter (imported lazily to
            # avoid circular imports at module load).
            from .metadata_store_mcr_ext import _action_item_from_row
            return [_action_item_from_row(r) for r in rows]
