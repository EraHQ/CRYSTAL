"""CRYS-side session registry — Foundation F4 (surface consolidation).

Registers the running CRYS session in the metadata store so it appears in
the Inspector (the unified-surfaces law made real: "see CRYS activity in
the Inspector"), heartbeats its status + current action, records the
dependencies it spawned, and marks it exited on a clean shutdown.

D4 — offline is complete, local stays local: this writes to whatever store
CRYS is already using — the local default SQLite when offline, the team's
shared DB when logged in. Offline sessions therefore land in a local store
the server can't read and are never synced. No special-casing: the store IS
the boundary.

EVERYTHING HERE IS FAIL-SAFE. Observability must never break the agent's
real work, so every store call is guarded — a registry hiccup is logged at
debug and swallowed, the same posture as the daemon's heartbeat
(`except OSError: pass`). The agent runs whether or not the registry is
reachable.

v1 scope, stated plainly: the REPL blocks on input() between turns, so there
is NO background heartbeat task (a thread would either fight the event loop
for the async store or need raw cross-thread SQL, violating R9). Liveness is
beaten at turn boundaries instead — 'idle' before each prompt, 'running'
before each agent turn. A REPL left idle past the stale window therefore
reads as crashed and self-corrects on the next turn (heartbeat_session
resurrects it). The fully-async daemon could carry a real heartbeat loop;
that, and richer dependency tracking (the browser, shell subprocesses), are
deferred follow-ups — the store + API already support them.
"""
from __future__ import annotations

import os
import socket
import uuid
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)

# current_action is free text; keep it bounded so a giant pasted prompt
# doesn't bloat the row.
_ACTION_MAX = 160


def new_session_id() -> str:
    """A stable id for this process's session."""
    return f"crys_{uuid.uuid4().hex[:16]}"


def _clip(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    flat = " ".join(text.split())
    return flat if len(flat) <= _ACTION_MAX else flat[: _ACTION_MAX - 1] + "…"


def _safe_hostname() -> Optional[str]:
    try:
        return socket.gethostname()
    except Exception:  # noqa: BLE001 — a hostname lookup must never matter
        return None


class SessionHandle:
    """Tracks one CRYS session against a metadata store. Constructed once at
    startup, bound to the store after the agent is built, beaten at turn
    boundaries, and closed on shutdown. Every store interaction is fail-safe;
    an unbound handle's methods are no-ops."""

    def __init__(
        self,
        *,
        project_dir: str,
        model: Optional[str],
        host: Optional[str] = None,
        pid: Optional[int] = None,
    ) -> None:
        self.session_id = new_session_id()
        self.project_dir = project_dir
        self.model = model
        self.host = host or _safe_hostname()
        self.pid = pid if pid is not None else os.getpid()
        self._store: Any = None  # the store the session lives in (set on bind)
        self._team_id: Optional[str] = None
        self._bound = False
        # Per-session turn counter (advanced at each turn boundary). Events
        # recorded during a turn default to it; -1 = no turn yet.
        self._turn: int = -1

    async def bind(self, store: Any, team_id: str, *, status: str = "running") -> None:
        """Register (or refresh) the session against `store`. Idempotent —
        safe to call again. Remembers the store so later beats land in the
        same place. CRYS authenticates with a team key, so no operator is
        attached (operator_id stays None — a team-root session)."""
        self._store = store
        self._team_id = team_id
        try:
            await store.register_session(
                self.session_id,
                team_id,
                host=self.host,
                pid=self.pid,
                project_dir=self.project_dir,
                model=self.model,
                status=status,
            )
            self._bound = True
        except Exception as e:  # noqa: BLE001 — observability is best-effort
            logger.debug("crys_session.bind_failed", error=str(e))

    async def beat(
        self,
        *,
        status: Optional[str] = None,
        current_action: Optional[str] = None,
    ) -> None:
        """Heartbeat the session (refresh last_heartbeat + optional fields).
        No-op if the session was never bound."""
        if not self._bound or self._store is None:
            return
        try:
            await self._store.heartbeat_session(
                self.session_id,
                status=status,
                current_action=_clip(current_action),
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("crys_session.beat_failed", error=str(e))

    def begin_turn(self) -> int:
        """Advance the per-session turn counter at a turn boundary (called
        alongside guard.begin_turn). Returns the new turn index; events
        recorded during the turn default to it. Pure/sync — no store call."""
        self._turn += 1
        return self._turn

    async def record_event(
        self,
        *,
        event_type: str,
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
    ) -> None:
        """Append one event to this session's stream — the record the Agents
        timeline + unified log read. No-op if unbound; fail-safe like beat()
        (observability must never break a turn). turn_index defaults to the
        current turn."""
        if not self._bound or self._store is None:
            return
        if turn_index is None and self._turn >= 0:
            turn_index = self._turn
        try:
            await self._store.record_event(
                self.session_id,
                event_type=event_type,
                team_id=self._team_id,
                label=label,
                phase=phase,
                payload=payload,
                status=status,
                turn_index=turn_index,
                parent_session_id=parent_session_id,
                duration_ms=duration_ms,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                cost_micro_usd=cost_micro_usd,
            )
        except Exception as e:  # noqa: BLE001 — observability is best-effort
            logger.debug("crys_session.record_event_failed", error=str(e))

    async def register_dependency(
        self, *, kind: str, descriptor: str = "", pid: Optional[int] = None
    ) -> None:
        """Record a resource the session spawned (e.g. an MCP server)."""
        if not self._bound or self._store is None:
            return
        try:
            await self._store.register_dependency(
                self.session_id, kind=kind, descriptor=descriptor, pid=pid
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("crys_session.dependency_failed", error=str(e))

    async def close(self, *, status: str = "exited") -> None:
        """Mark the session terminal on a clean shutdown, so the Inspector
        shows it as exited immediately rather than waiting out the stale
        window. (A kill or crash skips this and goes stale → crashed.)"""
        if not self._bound or self._store is None:
            return
        try:
            await self._store.heartbeat_session(self.session_id, status=status)
        except Exception as e:  # noqa: BLE001
            logger.debug("crys_session.close_failed", error=str(e))
