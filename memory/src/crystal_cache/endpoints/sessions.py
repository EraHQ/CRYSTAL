"""Session-registry endpoints — Foundation F4 (surface consolidation).

The control-plane + read API over the session registry
(metadata_store_session_ext.py). Three routes:

  POST /v1/sessions/heartbeat        — a session pushes its current state
       (register-or-refresh, idempotent). This is the remote-session
       outbound push (D4): local sessions sharing the server DB can write
       rows directly; remote sessions POST here. Authenticated by the
       operator credential (resolve_principal) — the principal attributes
       the session to its team (+ operator). Each beat also sweeps stale
       OTHER sessions (mark_stale_sessions) so the registry self-heals as
       agents check in.
  GET  /v1/sessions                  — the Inspector's Activity view: the
       team's sessions, newest-heartbeat first, optionally filtered to one
       operator. Read-only; liveness (is_stale / effective_status) is
       derived per row, so a crashed session reads correctly even between
       sweeps.
  GET  /v1/sessions/{id}/dependencies — a session's spawned dependencies
       (team-scoped: a session id from another team 404s rather than
       leaking).

This is F4's API; the React Activity view is a Growth surface. The request
model is endpoint-local; responses are the store's dicts (FastAPI's encoder
serializes the datetimes to ISO). `awaiting_payload` (the approval-gate
payload) is deliberately NOT wired through the heartbeat here — that's G2's
control-plane flow; the column exists as the forward-ref only.
"""
from __future__ import annotations

from typing import Annotated, Any, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..infrastructure import MetadataStore
from ..infrastructure.metadata_store import get_metadata_store
from ..ingress.auth import resolve_principal
from ..models import Customer, Operator

logger = structlog.get_logger(__name__)

router = APIRouter()


class HeartbeatRequest(BaseModel):
    """A session's self-reported state. The agent sends its complete current
    state each beat and the server upserts it (register_session). status is
    always set; the optional metadata fields are applied when present."""
    session_id: str
    status: str = "running"
    current_action: Optional[str] = None
    host: Optional[str] = None
    pid: Optional[int] = None
    project_dir: Optional[str] = None
    model: Optional[str] = None
    parent_session_id: Optional[str] = None


@router.post("/v1/sessions/heartbeat")
async def session_heartbeat(
    body: HeartbeatRequest,
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(resolve_principal)
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> dict[str, Any]:
    """Register-or-refresh the calling session, then sweep stale others.

    Authenticated by the operator credential: the principal's team owns the
    session, and the operator (when the key was an operator key) is recorded.
    Idempotent — the first beat creates the row, later beats update it.
    """
    customer, operator = principal
    session = await store.register_session(
        body.session_id,
        customer.id,
        operator_id=operator.id if operator is not None else None,
        host=body.host,
        pid=body.pid,
        project_dir=body.project_dir,
        model=body.model,
        status=body.status,
        current_action=body.current_action,
        parent_session_id=body.parent_session_id,
    )
    # Self-healing: each check-in materializes stale OTHER sessions
    # (idempotent; global staleness hygiene, no cross-team data exposure —
    # the sweep only flips dead rows to 'crashed').
    await store.mark_stale_sessions()
    return {"session": session}


@router.get("/v1/sessions")
async def list_sessions(
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(resolve_principal)
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    operator_id: Optional[str] = None,
    include_terminal: bool = True,
) -> dict[str, Any]:
    """The team's sessions (Activity view), newest-heartbeat first.

    Any team member may read (viewers audit); scoped to the principal's team.
    Optional ?operator_id= filters to one operator's sessions; set
    include_terminal=false to hide exited/crashed sessions. Read-only —
    liveness is derived per row.
    """
    customer, _actor = principal
    sessions = await store.list_sessions_for_team(
        customer.id,
        operator_id=operator_id,
        include_terminal=include_terminal,
    )
    return {"sessions": sessions}


@router.get("/v1/sessions/{session_id}/dependencies")
async def list_session_dependencies(
    session_id: str,
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(resolve_principal)
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> dict[str, Any]:
    """A session's spawned dependencies. Team-scoped: a session id belonging
    to another team (or unknown) returns 404 rather than leaking."""
    customer, _actor = principal
    session = await store.get_session(session_id)
    if session is None or session["team_id"] != customer.id:
        raise HTTPException(status_code=404, detail="session not found")
    deps = await store.list_dependencies_for_session(session_id)
    return {"dependencies": deps}
