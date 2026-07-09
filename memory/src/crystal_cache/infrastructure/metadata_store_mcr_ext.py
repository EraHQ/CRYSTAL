"""MCR artifact CRUD — Phase 8.5 (2026-05-27).

The fifth mixin on MetadataStore (per D12, AN-7 pattern). Lands the
CRUD surface for the three MCR artifact tables introduced in
Phase 8.5: `reasoning_traces`, `critiques`, `action_items`.

Phase 8.5 scope (P0.34): schema + Pydantic models + this mixin
ONLY. No writers, no readers, no behavior change. Writers ship in
Phase 9 (agent self-trace emission) and Phase 9.5 (shadow critic);
the metacognitive layer that reads these and produces item
alignments + critique synthesis ships in Phase 10.

Method surface (CRUD per artifact):

  Reasoning traces:
    create_reasoning_trace(...)            — write a new trace
    get_reasoning_trace(trace_id)          — read by id
    list_traces_for_sequence(...)          — soft-join by sequence

  Critiques:
    create_critique(...)                   — write a new critique
    get_critique(critique_id)              — by id (added Phase 10.5)
    list_critiques_for_trace(trace_id)     — by hard pointer
    list_critiques_for_sequence(...)       — by soft-join key
    list_critiques_by_role(...)            — calibration scans

  Action items:
    create_action_item(...)                — write a new item
    list_action_items_for_critique(...)    — by FK
    list_action_items_by_status(...)       — metacognitive queue
    update_action_item_status(...)         — lifecycle transitions
    reconcile_total_action_items(...)      — CU-20 drift fixer
                                            (added Phase 11.5)

All methods route through `self.session()` per R9. None bypass the
store layer. Each method's bound to MetadataStore via the
`_bind_mixin_methods` shared helper in `infrastructure/__init__.py`.

R3 wire-format strings used here (P0.40):
  - Action item statuses: 'pending', 'promoted', 'deferred',
    'dropped', 'acted'.
  - Critic roles passed through unchanged: 'agent_self', 'shadow',
    'specialist'.
  - The 7 action types and 8 observation types are stored as-is in
    JSON columns / String columns; validation lives at the Pydantic
    layer.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

import structlog
from sqlalchemy import select

from ..models.action_item import ActionItem
from ..models.critique import Critique
from ..models.reasoning_trace import ReasoningTrace
from .schema import (
    ActionItemRow,
    CritiqueRow,
    ReasoningTraceRow,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Row → Pydantic converters (module-level helpers per D12 pattern)
# ---------------------------------------------------------------------------

def _trace_from_row(row: ReasoningTraceRow) -> ReasoningTrace:
    return ReasoningTrace(
        id=row.id,
        customer_id=row.customer_id,
        sequence_id=row.sequence_id,
        turn_index=row.turn_index,
        query_log_id=row.query_log_id,
        events=list(row.events or []),
        crystals_used=list(row.crystals_used or []),
        tool_calls=list(row.tool_calls or []),
        inferences=list(row.inferences or []),
        borders_crossed=list(row.borders_crossed or []),
        gaps_felt=list(row.gaps_felt or []),
        created_at=row.created_at,
    )


def _critique_from_row(row: CritiqueRow) -> Critique:
    return Critique(
        id=row.id,
        customer_id=row.customer_id,
        trace_id=row.trace_id,
        sequence_id=row.sequence_id,
        turn_index=row.turn_index,
        critic_role=row.critic_role,  # type: ignore[arg-type]
        critic_model=row.critic_model,
        observations=list(row.observations or []),
        summary_text=row.summary_text,
        total_action_items=row.total_action_items,
        created_at=row.created_at,
    )


def _action_item_from_row(row: ActionItemRow) -> ActionItem:
    return ActionItem(
        id=row.id,
        critique_id=row.critique_id,
        customer_id=row.customer_id,
        action_type=row.action_type,  # type: ignore[arg-type]
        content=dict(row.content or {}),
        critic_confidence=row.critic_confidence,
        status=row.status,  # type: ignore[arg-type]
        metacog_decision_at=row.metacog_decision_at,
        acted_artifact_id=row.acted_artifact_id,
        created_at=row.created_at,
    )


class McrExtensionsMixin:
    """MCR artifact CRUD methods bound onto MetadataStore.

    Bound at import time via setattr in `infrastructure/__init__.py`
    via the shared `_bind_mixin_methods` helper. Same pattern as
    `AuditTablesMixin`, `CustomerExtensionsMixin`,
    `CognitionExtensionsMixin`, `LearningExtensionsMixin`.

    All methods are async and route through `self.session()` per R9.
    """

    # ====================================================================
    # Reasoning traces
    # ====================================================================

    async def create_reasoning_trace(
        self,
        customer_id: str,
        events: Optional[list[dict[str, Any]]] = None,
        *,
        sequence_id: Optional[str] = None,
        turn_index: Optional[int] = None,
        query_log_id: Optional[str] = None,
        crystals_used: Optional[list[str]] = None,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        inferences: Optional[list[dict[str, Any]]] = None,
        borders_crossed: Optional[list[dict[str, Any]]] = None,
        gaps_felt: Optional[list[dict[str, Any]]] = None,
    ) -> ReasoningTrace:
        """Create a reasoning trace.

        All trace-content kwargs are optional with sensible empty
        defaults so the Phase 9 emitter can build the trace
        incrementally and call this once at the end.
        """
        trace_id = uuid.uuid4().hex[:16]
        async with self.session() as session:  # type: ignore[attr-defined]
            row = ReasoningTraceRow(
                id=trace_id,
                customer_id=customer_id,
                sequence_id=sequence_id,
                turn_index=turn_index,
                query_log_id=query_log_id,
                events=list(events or []),
                crystals_used=list(crystals_used or []),
                tool_calls=list(tool_calls or []),
                inferences=list(inferences or []),
                borders_crossed=list(borders_crossed or []),
                gaps_felt=list(gaps_felt or []),
            )
            session.add(row)
            await session.flush()
            return _trace_from_row(row)

    async def get_reasoning_trace(
        self,
        trace_id: str,
    ) -> Optional[ReasoningTrace]:
        """Get a reasoning trace by id. Returns None when not found."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = select(ReasoningTraceRow).where(
                ReasoningTraceRow.id == trace_id
            )
            row = (await session.execute(stmt)).scalar_one_or_none()
            return _trace_from_row(row) if row is not None else None

    async def list_traces_for_sequence(
        self,
        customer_id: str,
        sequence_id: str,
    ) -> list[ReasoningTrace]:
        """List all reasoning traces for a sequence, oldest-first.

        Returns ordered by turn_index ascending (NULLs last). The
        common consumer is the metacognitive layer reviewing a
        conversation's traces in order.
        """
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(ReasoningTraceRow)
                .where(
                    ReasoningTraceRow.customer_id == customer_id,
                    ReasoningTraceRow.sequence_id == sequence_id,
                )
                .order_by(
                    ReasoningTraceRow.turn_index.asc().nullslast(),
                    ReasoningTraceRow.created_at.asc(),
                )
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [_trace_from_row(r) for r in rows]

    # ====================================================================
    # Critiques
    # ====================================================================

    async def create_critique(
        self,
        customer_id: str,
        critic_role: str,
        critic_model: str,
        *,
        trace_id: Optional[str] = None,
        sequence_id: Optional[str] = None,
        turn_index: Optional[int] = None,
        observations: Optional[list[dict[str, Any]]] = None,
        summary_text: Optional[str] = None,
        total_action_items: int = 0,
    ) -> Critique:
        """Create a critique row.

        critic_role and critic_model are required positional kwargs —
        every critique must be attributable per D-MCR-6 (calibration
        by track record) and §7 (per-model calibration).

        trace_id is optional because the agent's self-critique may
        write before the trace finishes streaming (Phase 9 open Q2).
        At least one of (trace_id) or (sequence_id, turn_index)
        SHOULD be provided so the critique resolves back to a trace,
        but neither is required at the DB layer.
        """
        critique_id = uuid.uuid4().hex[:16]
        async with self.session() as session:  # type: ignore[attr-defined]
            row = CritiqueRow(
                id=critique_id,
                customer_id=customer_id,
                trace_id=trace_id,
                sequence_id=sequence_id,
                turn_index=turn_index,
                critic_role=critic_role,
                critic_model=critic_model,
                observations=list(observations or []),
                summary_text=summary_text,
                total_action_items=total_action_items,
            )
            session.add(row)
            await session.flush()
            return _critique_from_row(row)

    async def list_critiques_for_trace(
        self,
        trace_id: str,
    ) -> list[Critique]:
        """List critiques pointing at a specific trace via trace_id.

        Returns oldest-first. Does NOT include critiques that share
        the trace's (sequence_id, turn_index) but have a NULL
        trace_id — use `list_critiques_for_sequence` for that
        broader scan.
        """
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(CritiqueRow)
                .where(CritiqueRow.trace_id == trace_id)
                .order_by(CritiqueRow.created_at.asc())
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [_critique_from_row(r) for r in rows]

    async def get_critique(
        self,
        critique_id: str,
    ) -> Optional[Critique]:
        """Get a critique by ID. Returns None when not found.

        Added Phase 10.5 (P0.90) for the substrate review surface,
        which composes per-item views by following action_item→
        critique→trace pointers. Reusable beyond Phase 10.5; any
        caller needing a single critique by ID can use this.
        """
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = select(CritiqueRow).where(
                CritiqueRow.id == critique_id
            )
            row = (await session.execute(stmt)).scalar_one_or_none()
            return _critique_from_row(row) if row is not None else None

    async def list_critiques_for_sequence(
        self,
        customer_id: str,
        sequence_id: str,
        *,
        turn_index: Optional[int] = None,
    ) -> list[Critique]:
        """List critiques for a sequence via the soft-join key.

        When turn_index is provided, narrows to a single turn.
        When omitted, returns all critiques across the sequence's
        turns, oldest-first.

        Returns critiques whose `sequence_id` matches; this catches
        critiques written before their trace existed (trace_id is
        NULL) AND critiques with both fields populated.
        """
        async with self.session() as session:  # type: ignore[attr-defined]
            conditions = [
                CritiqueRow.customer_id == customer_id,
                CritiqueRow.sequence_id == sequence_id,
            ]
            if turn_index is not None:
                conditions.append(CritiqueRow.turn_index == turn_index)
            stmt = (
                select(CritiqueRow)
                .where(*conditions)
                .order_by(
                    CritiqueRow.turn_index.asc().nullslast(),
                    CritiqueRow.created_at.asc(),
                )
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [_critique_from_row(r) for r in rows]

    async def list_critiques_by_role(
        self,
        customer_id: str,
        critic_role: str,
        *,
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> list[Critique]:
        """List critiques for a customer filtered by critic_role.

        Supports the Phase 10 calibration scan ("how reliable is
        this critic over the last N days?") and the human-surfacing
        path ("show me every shadow critique this customer has
        received this week").

        `since` filters to critiques created_at >= since when set.
        `limit` caps the result count when set; ordered
        most-recent-first.
        """
        async with self.session() as session:  # type: ignore[attr-defined]
            conditions = [
                CritiqueRow.customer_id == customer_id,
                CritiqueRow.critic_role == critic_role,
            ]
            if since is not None:
                conditions.append(CritiqueRow.created_at >= since)
            stmt = (
                select(CritiqueRow)
                .where(*conditions)
                .order_by(CritiqueRow.created_at.desc())
            )
            if limit is not None:
                stmt = stmt.limit(limit)
            rows = (await session.execute(stmt)).scalars().all()
            return [_critique_from_row(r) for r in rows]

    async def list_recent_critiques(
        self,
        *,
        customer_id: Optional[str] = None,
        critic_role: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 200,
    ) -> list[Critique]:
        """List recent critiques across roles (S11, 2026-07-09).

        The quality-review surface needs every critic's observations
        (shadow + agent_self) in one bounded, most-recent-first read,
        optionally cross-tenant (customer_id=None → operator's
        system-wide view — same semantics as the substrate surface).
        """
        async with self.session() as session:  # type: ignore[attr-defined]
            conditions = []
            if customer_id is not None:
                conditions.append(CritiqueRow.customer_id == customer_id)
            if critic_role is not None:
                conditions.append(CritiqueRow.critic_role == critic_role)
            if since is not None:
                conditions.append(CritiqueRow.created_at >= since)
            stmt = select(CritiqueRow).order_by(
                CritiqueRow.created_at.desc()
            ).limit(limit)
            if conditions:
                stmt = (
                    select(CritiqueRow)
                    .where(*conditions)
                    .order_by(CritiqueRow.created_at.desc())
                    .limit(limit)
                )
            rows = (await session.execute(stmt)).scalars().all()
            return [_critique_from_row(r) for r in rows]

    # ====================================================================
    # Action items
    # ====================================================================

    async def create_action_item(
        self,
        critique_id: str,
        customer_id: str,
        action_type: str,
        *,
        content: Optional[dict[str, Any]] = None,
        critic_confidence: Optional[float] = None,
    ) -> ActionItem:
        """Create a new action item linked to a critique.

        New items start at status='pending'. Metacognitive transitions
        (promoted / deferred / dropped / acted) flow through
        `update_action_item_status`.
        """
        item_id = uuid.uuid4().hex[:16]
        async with self.session() as session:  # type: ignore[attr-defined]
            row = ActionItemRow(
                id=item_id,
                critique_id=critique_id,
                customer_id=customer_id,
                action_type=action_type,
                content=dict(content or {}),
                critic_confidence=critic_confidence,
                status="pending",
            )
            session.add(row)
            await session.flush()
            return _action_item_from_row(row)

    async def get_session_tool_calls(
        self, customer_id: str, sequence_id: str, *, limit: int = 200
    ) -> list[list[dict]]:
        """S8 (2026-07-08): per-turn tool_calls for one chat session,
        ordered by trace creation. The agent endpoint is stateless
        (turn_index=None by design), so alignment with the session's
        query_logs is POSITIONAL — both rows are written by the same
        finalize call in the same order. Customer-scoped."""
        stmt = (
            select(ReasoningTraceRow.tool_calls)
            .where(
                ReasoningTraceRow.customer_id == customer_id,
                ReasoningTraceRow.sequence_id == sequence_id,
            )
            .order_by(ReasoningTraceRow.created_at.asc())
            .limit(limit)
        )
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(stmt)).scalars().all()
            return [list(r or []) for r in rows]

    async def list_action_items_for_critique(
        self,
        critique_id: str,
    ) -> list[ActionItem]:
        """List action items linked to a specific critique, oldest-first."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(ActionItemRow)
                .where(ActionItemRow.critique_id == critique_id)
                .order_by(ActionItemRow.created_at.asc())
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [_action_item_from_row(r) for r in rows]

    async def list_action_items_by_status(
        self,
        customer_id: str,
        status: str,
        *,
        action_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[ActionItem]:
        """List action items for a customer filtered by status.

        The metacognitive layer's primary read query: "give me
        pending items to review." Optionally narrows by action_type
        for the scheduler's per-type queues (e.g. all pending
        research_tasks).

        Ordered oldest-first within status so the metacognitive
        layer processes items roughly FIFO; tweak with `limit` to
        bound the per-pass workload.
        """
        async with self.session() as session:  # type: ignore[attr-defined]
            conditions = [
                ActionItemRow.customer_id == customer_id,
                ActionItemRow.status == status,
            ]
            if action_type is not None:
                conditions.append(ActionItemRow.action_type == action_type)
            stmt = (
                select(ActionItemRow)
                .where(*conditions)
                .order_by(ActionItemRow.created_at.asc())
            )
            if limit is not None:
                stmt = stmt.limit(limit)
            rows = (await session.execute(stmt)).scalars().all()
            return [_action_item_from_row(r) for r in rows]

    async def update_action_item_status(
        self,
        action_item_id: str,
        status: str,
        *,
        metacog_decision_at: Optional[datetime] = None,
        acted_artifact_id: Optional[str] = None,
    ) -> Optional[ActionItem]:
        """Update an action item's lifecycle status.

        Transitions: pending → {promoted, deferred, dropped, acted}.
        Caller is responsible for valid transitions (e.g. don't move
        a 'dropped' item back to 'pending'); the DB doesn't enforce
        the state machine.

        `metacog_decision_at` defaults to now when transitioning out
        of 'pending'. `acted_artifact_id` is required only for the
        transition to 'acted'; callers ensure it points at a real
        artifact row (cognition_task, knowledge_gap, crystal_edit,
        etc.).

        Returns the updated ActionItem, or None if the row was not
        found.
        """
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = select(ActionItemRow).where(
                ActionItemRow.id == action_item_id
            )
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row is None:
                return None
            row.status = status
            if metacog_decision_at is not None:
                row.metacog_decision_at = metacog_decision_at
            elif status != "pending":
                # Default the timestamp when transitioning out of
                # pending — callers don't have to pass it explicitly.
                from datetime import timezone
                row.metacog_decision_at = datetime.now(timezone.utc)
            if acted_artifact_id is not None:
                row.acted_artifact_id = acted_artifact_id
            await session.flush()
            return _action_item_from_row(row)

    async def reconcile_total_action_items(
        self,
        critique_id: str,
    ) -> Optional[int]:
        """Reconcile a critique's `total_action_items` counter against
        the actual count of ActionItemRow rows pointing at it.

        Phase 11.5 / CU-20 (P0.101). The CritiqueRow has a denormalized
        `total_action_items` counter that the writer sets at
        `create_critique` time. Subsequent calls to `create_action_item`
        do NOT auto-update the counter — if a future writer adds
        action items to an existing critique after creation, the
        counter drifts below the actual count.

        This helper COUNTs the actual ActionItemRow rows for the
        critique, UPDATEs the CritiqueRow.total_action_items to match,
        and returns the corrected count. Idempotent: calling twice on
        a critique that's already in sync is a no-op (counter set to
        the same value it already had).

        Currently shipped code does not produce drift (Phase 9A/9B/9C/
        9.5 emitters set the count at create time; Phase 10A+
        transitions existing items rather than creating new ones).
        This helper is a tool for future writers + the Phase 11.5
        invariant test M1.

        Args:
            critique_id: the critique whose counter to reconcile.

        Returns:
            The actual count of ActionItemRow rows pointing at the
            critique. None when the critique doesn't exist.
        """
        from sqlalchemy import func

        async with self.session() as session:  # type: ignore[attr-defined]
            # Fetch the critique first — we need to confirm it exists
            # before COUNTing items + updating the counter.
            stmt = select(CritiqueRow).where(
                CritiqueRow.id == critique_id
            )
            critique_row = (await session.execute(stmt)).scalar_one_or_none()
            if critique_row is None:
                return None

            # COUNT ActionItemRow rows for this critique.
            count_stmt = (
                select(func.count(ActionItemRow.id))
                .where(ActionItemRow.critique_id == critique_id)
            )
            actual_count = (await session.execute(count_stmt)).scalar_one()

            # Update the counter to match the actual count. SQLAlchemy
            # detects the change on the in-session row.
            critique_row.total_action_items = int(actual_count)
            await session.flush()
            return int(actual_count)
