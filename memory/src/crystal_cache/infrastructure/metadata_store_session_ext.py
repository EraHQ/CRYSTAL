"""Session-registry primitives — the F4 surface-consolidation store surface.

Foundation F4 makes the unified-surfaces law real: a live session registry
every surface (the CRYS terminal now; the Inspector reads it) writes to. A
session heartbeats its status + current_action; **liveness is inferred from
staleness, never self-reported** — a crashed agent can't report its own
crash, so a row stale beyond the threshold is presumed crashed and its
dependencies orphaned. This is the daemon's stale-window logic
(coding-agent daemon.py `_heartbeat_loop` / `daemon_running` /
`fail_stale_running_tasks`) generalized to a DB row, because sessions surface
server-side across machines.

R9 puts the SQL here, bound onto MetadataStore via the same setattr pattern
as the other extension mixins (see infrastructure/__init__.py). Like the
agent-task mixin, methods return plain dicts — these are operational rows the
Inspector + control plane format and branch on, not domain entities.

Two facets of staleness:
  - READ-TIME derivation (`is_stale` / `effective_status` on every returned
    dict): a non-terminal session whose last heartbeat is older than the
    threshold reads as crashed, WITHOUT mutating the row. So the Inspector
    shows the truth even if the sweep hasn't run.
  - MATERIALIZING sweep (`mark_stale_sessions`): flips stale non-terminal
    sessions to status='crashed' and their active dependencies to
    'orphaned'. Mirrors `fail_stale_running_tasks`; the control plane / a
    worker calls it so pending requests can be voided and deps reclaimed.
Both use the same threshold, so the two facets never disagree.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import structlog
from sqlalchemy import select, update

from .schema import AgentSessionRow, SessionDependencyRow

logger = structlog.get_logger(__name__)

# Liveness threshold. A session whose last heartbeat is older than this is
# presumed crashed. Generous relative to the runtime's heartbeat cadence
# (transitions + a timer) so network latency on a remote session never
# false-positives. Module-level (not a class attr — the mixin binding copies
# public callables only) and overridable per call.
DEFAULT_SESSION_STALE_SECONDS = 90

# Terminal lifecycle states — the staleness derivation + sweep leave these
# alone (an exited/crashed session is already settled).
TERMINAL_SESSION_STATUSES = frozenset({"exited", "crashed"})


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt: Optional[datetime]) -> datetime:
    """Coerce a possibly-naive datetime to tz-aware UTC (SQLite round-trips
    can drop tzinfo). None defends as 'now'."""
    if dt is None:
        return _utcnow()
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _is_stale(
    last_heartbeat_at: Optional[datetime],
    status: str,
    stale_seconds: float,
    now: datetime,
) -> bool:
    """A non-terminal session is stale once its last heartbeat is older than
    stale_seconds. Terminal sessions are never 'stale' (they're settled)."""
    if status in TERMINAL_SESSION_STATUSES:
        return False
    return (now - _aware(last_heartbeat_at)).total_seconds() > stale_seconds


def _session_to_dict(
    row: AgentSessionRow, *, stale_seconds: float, now: datetime
) -> dict[str, Any]:
    stale = _is_stale(row.last_heartbeat_at, row.status, stale_seconds, now)
    return {
        "session_id": row.session_id,
        "team_id": row.team_id,
        "operator_id": row.operator_id,
        "host": row.host,
        "pid": row.pid,
        "project_dir": row.project_dir,
        "model": row.model,
        "status": row.status,
        # Derived liveness: stale non-terminal sessions read as crashed
        # without mutating the stored status.
        "effective_status": "crashed" if stale else row.status,
        "is_stale": stale,
        "current_action": row.current_action,
        "awaiting_payload": row.awaiting_payload,
        "parent_session_id": row.parent_session_id,
        "started_at": row.started_at,
        "last_heartbeat_at": row.last_heartbeat_at,
        "cost_usd_cumulative": row.cost_usd_cumulative,
    }


def _dependency_to_dict(row: SessionDependencyRow) -> dict[str, Any]:
    return {
        "dependency_id": row.dependency_id,
        "session_id": row.session_id,
        "kind": row.kind,
        "descriptor": row.descriptor,
        "pid": row.pid,
        "status": row.status,
        "spawned_at": row.spawned_at,
    }


class SessionRegistryMixin:
    """agent_sessions + session_dependencies CRUD bound onto MetadataStore."""

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    async def register_session(
        self,
        session_id: str,
        team_id: str,
        *,
        operator_id: Optional[str] = None,
        host: Optional[str] = None,
        pid: Optional[int] = None,
        project_dir: Optional[str] = None,
        model: Optional[str] = None,
        status: str = "running",
        current_action: Optional[str] = None,
        parent_session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Register a session (insert) or refresh it (update) — idempotent.

        First call stamps started_at; every call sets last_heartbeat_at=now
        and the provided fields. A re-register on reconnect updates the
        existing row rather than duplicating it.
        """
        now = _utcnow()
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(AgentSessionRow, session_id)
            if row is None:
                row = AgentSessionRow(
                    session_id=session_id,
                    team_id=team_id,
                    operator_id=operator_id,
                    host=host,
                    pid=pid,
                    project_dir=project_dir,
                    model=model,
                    status=status,
                    current_action=current_action,
                    parent_session_id=parent_session_id,
                    started_at=now,
                    last_heartbeat_at=now,
                )
                session.add(row)
            else:
                row.team_id = team_id
                if operator_id is not None:
                    row.operator_id = operator_id
                if host is not None:
                    row.host = host
                if pid is not None:
                    row.pid = pid
                if project_dir is not None:
                    row.project_dir = project_dir
                if model is not None:
                    row.model = model
                row.status = status
                row.current_action = current_action
                if parent_session_id is not None:
                    row.parent_session_id = parent_session_id
                row.last_heartbeat_at = now
            await session.commit()
            await session.refresh(row)
            result = _session_to_dict(
                row, stale_seconds=DEFAULT_SESSION_STALE_SECONDS, now=_utcnow()
            )
        logger.info(
            "session_registry.registered",
            session_id=session_id, team_id=team_id, status=status,
        )
        return result

    async def heartbeat_session(
        self,
        session_id: str,
        *,
        status: Optional[str] = None,
        current_action: Optional[str] = None,
        awaiting_payload: Optional[dict[str, Any]] = None,
        cost_usd_cumulative: Optional[int] = None,
    ) -> bool:
        """Refresh a session's last_heartbeat_at (+ optional fields). The
        per-transition / on-a-timer beat. Returns False for an unknown
        session id rather than raising — a heartbeat is best-effort."""
        now = _utcnow()
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(AgentSessionRow, session_id)
            if row is None:
                return False
            row.last_heartbeat_at = now
            if status is not None:
                row.status = status
            if current_action is not None:
                row.current_action = current_action
            if awaiting_payload is not None:
                row.awaiting_payload = awaiting_payload
            if cost_usd_cumulative is not None:
                row.cost_usd_cumulative = cost_usd_cumulative
            await session.commit()
        return True

    async def get_session(
        self,
        session_id: str,
        *,
        stale_seconds: float = DEFAULT_SESSION_STALE_SECONDS,
    ) -> Optional[dict[str, Any]]:
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(AgentSessionRow, session_id)
            if row is None:
                return None
            return _session_to_dict(
                row, stale_seconds=stale_seconds, now=_utcnow()
            )

    async def list_sessions_for_team(
        self,
        team_id: str,
        *,
        operator_id: Optional[str] = None,
        stale_seconds: float = DEFAULT_SESSION_STALE_SECONDS,
        include_terminal: bool = True,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Sessions for a team, newest-heartbeat first. Optionally filtered to
        one operator, and optionally excluding terminal (exited/crashed)
        sessions. Each dict carries the derived is_stale / effective_status.
        """
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = select(AgentSessionRow).where(
                AgentSessionRow.team_id == team_id
            )
            if operator_id is not None:
                stmt = stmt.where(AgentSessionRow.operator_id == operator_id)
            if not include_terminal:
                stmt = stmt.where(
                    AgentSessionRow.status.notin_(
                        list(TERMINAL_SESSION_STATUSES)
                    )
                )
            stmt = stmt.order_by(
                AgentSessionRow.last_heartbeat_at.desc()
            ).limit(limit)
            rows = (await session.execute(stmt)).scalars().all()
            now = _utcnow()
            return [
                _session_to_dict(r, stale_seconds=stale_seconds, now=now)
                for r in rows
            ]

    async def mark_stale_sessions(
        self,
        *,
        stale_seconds: float = DEFAULT_SESSION_STALE_SECONDS,
    ) -> int:
        """Materialize staleness: flip stale non-terminal sessions to
        'crashed' and their still-active dependencies to 'orphaned'.

        Fetch-then-filter in Python (robust tz handling across SQLite /
        Postgres), then update by id — mirroring reopen_stale_retrying_gaps.
        Returns the number of sessions newly marked crashed.
        """
        now = _utcnow()
        crashed = 0
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(
                select(AgentSessionRow).where(
                    AgentSessionRow.status.notin_(
                        list(TERMINAL_SESSION_STATUSES)
                    )
                )
            )).scalars().all()
            stale_ids: list[str] = []
            for row in rows:
                age = (now - _aware(row.last_heartbeat_at)).total_seconds()
                if age > stale_seconds:
                    row.status = "crashed"
                    stale_ids.append(row.session_id)
                    crashed += 1
            if stale_ids:
                await session.execute(
                    update(SessionDependencyRow)
                    .where(
                        SessionDependencyRow.session_id.in_(stale_ids),
                        SessionDependencyRow.status == "active",
                    )
                    .values(status="orphaned")
                )
                await session.commit()
        if crashed:
            logger.warning(
                "session_registry.marked_stale_crashed", count=crashed
            )
        return crashed

    # ------------------------------------------------------------------
    # Dependencies
    # ------------------------------------------------------------------

    async def register_dependency(
        self,
        session_id: str,
        *,
        kind: str,
        descriptor: str = "",
        pid: Optional[int] = None,
        status: str = "active",
    ) -> dict[str, Any]:
        """Record a resource a session spawned (kind ∈ mcp_server |
        subprocess | browser | queued_task | pip_env)."""
        dep_id = f"sdep_{uuid.uuid4().hex[:16]}"
        async with self.session() as session:  # type: ignore[attr-defined]
            row = SessionDependencyRow(
                dependency_id=dep_id,
                session_id=session_id,
                kind=kind,
                descriptor=descriptor,
                pid=pid,
                status=status,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return _dependency_to_dict(row)

    async def update_dependency_status(
        self, dependency_id: str, *, status: str
    ) -> bool:
        """Transition a dependency (active → exited | orphaned). Returns False
        if no such dependency."""
        async with self.session() as session:  # type: ignore[attr-defined]
            result = await session.execute(
                update(SessionDependencyRow)
                .where(SessionDependencyRow.dependency_id == dependency_id)
                .values(status=status)
            )
            await session.commit()
            return result.rowcount > 0

    async def list_dependencies_for_session(
        self, session_id: str
    ) -> list[dict[str, Any]]:
        """A session's dependencies, oldest-spawned first."""
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(
                select(SessionDependencyRow)
                .where(SessionDependencyRow.session_id == session_id)
                .order_by(SessionDependencyRow.spawned_at)
            )).scalars().all()
            return [_dependency_to_dict(r) for r in rows]
