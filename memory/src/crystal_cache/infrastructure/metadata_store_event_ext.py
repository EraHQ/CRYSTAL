"""Agent event-stream primitives — the Unify-Agents store surface (2026-06-15).

The append-only record of everything CRYS does, bound onto MetadataStore as a
mixin (R9 puts the SQL here, same binding pattern as the other extensions —
see infrastructure/__init__.py). One session writes a stream of events
(`agent_events`); the "Agents" Inspector surface renders it as a per-session
timeline, the unified interaction log unions it with proxy query logs, and cost
rollups sum it. CRYS is the product — this is how its every turn, tool call,
delegated subagent, crystal, and gap becomes visible.

Methods return plain dicts (the session / agent-task precedent — operational
rows, not domain entities).

`seq` is per-session monotonic, assigned MAX+1 at write inside the same
transaction. Single-writer per session in practice (the REPL or the daemon owns
its own session id), so the read-then-insert race is the documented
single-writer assumption; append-only means a tie only affects ordering, broken
by created_at then id. record_event is best-effort by contract — its CRYS-side
caller (SessionHandle.record_event) swallows failures so observability can
never break a turn — but the store method itself raises normally so tests and
server callers see real errors.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import structlog
from sqlalchemy import func, select

from .schema import AgentEventRow

logger = structlog.get_logger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _event_to_dict(row: AgentEventRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "session_id": row.session_id,
        "team_id": row.team_id,
        "seq": row.seq,
        "turn_index": row.turn_index,
        "parent_session_id": row.parent_session_id,
        "event_type": row.event_type,
        "phase": row.phase,
        "label": row.label,
        "payload": row.payload,
        "status": row.status,
        "duration_ms": row.duration_ms,
        "tokens_input": row.tokens_input,
        "tokens_output": row.tokens_output,
        "cost_micro_usd": row.cost_micro_usd,
        "created_at": row.created_at,
    }


class EventLogMixin:
    """`agent_events` append + read, bound onto MetadataStore."""

    async def record_event(
        self,
        session_id: str,
        *,
        event_type: str,
        team_id: Optional[str] = None,
        label: str = "",
        phase: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
        status: Optional[str] = None,
        turn_index: Optional[int] = None,
        parent_session_id: Optional[str] = None,
        duration_ms: Optional[int] = None,
        tokens_input: Optional[int] = None,
        tokens_output: Optional[int] = None,
        cost_micro_usd: Optional[int] = None,
    ) -> dict[str, Any]:
        """Append one event to a session's stream; returns the row as a dict.

        `seq` is COALESCE(MAX(seq), -1) + 1 for the session (0-based,
        monotonic). All pointers are soft — the session row need not exist
        (events may predate its materialization). Caller-side recording is
        fail-safe (SessionHandle.record_event); this method raises normally.
        """
        async with self.session() as session:  # type: ignore[attr-defined]
            next_seq = (
                await session.execute(
                    select(func.coalesce(func.max(AgentEventRow.seq), -1) + 1)
                    .where(AgentEventRow.session_id == session_id)
                )
            ).scalar_one()
            row = AgentEventRow(
                id=f"aev_{uuid.uuid4().hex}",
                session_id=session_id,
                team_id=team_id,
                seq=int(next_seq),
                turn_index=turn_index,
                parent_session_id=parent_session_id,
                event_type=event_type,
                phase=phase,
                label=label,
                payload=payload,
                status=status,
                duration_ms=duration_ms,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                cost_micro_usd=cost_micro_usd,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return _event_to_dict(row)

    async def list_events_for_session(
        self,
        session_id: str,
        *,
        after_seq: Optional[int] = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """A session's events in stream order (seq asc). `after_seq` returns
        only events newer than a seq the caller already has — the incremental
        poll the live timeline uses to avoid re-fetching the whole stream."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = select(AgentEventRow).where(
                AgentEventRow.session_id == session_id
            )
            if after_seq is not None:
                stmt = stmt.where(AgentEventRow.seq > after_seq)
            stmt = stmt.order_by(AgentEventRow.seq.asc()).limit(limit)
            rows = (await session.execute(stmt)).scalars().all()
            return [_event_to_dict(r) for r in rows]

    async def list_events_for_team(
        self,
        team_id: str,
        *,
        event_types: Optional[list[str]] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """A team's events newest-first, optionally filtered to certain
        event_types. The unified interaction log uses this (e.g. filtered to
        turn_completed) to union agent turns with proxy query logs."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = select(AgentEventRow).where(AgentEventRow.team_id == team_id)
            if event_types:
                stmt = stmt.where(AgentEventRow.event_type.in_(event_types))
            stmt = stmt.order_by(AgentEventRow.created_at.desc()).limit(limit)
            rows = (await session.execute(stmt)).scalars().all()
            return [_event_to_dict(r) for r in rows]
