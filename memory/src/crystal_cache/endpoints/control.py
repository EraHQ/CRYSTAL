"""Control-plane endpoints — Growth G2 (outbound-poll command channel).

The server-side half of the control plane. The agent always initiates:
it heartbeats (F4) and POLLS here for commands; nothing connects inbound
(NAT-safe). Four routes:

  POST /v1/control/decisions        — an operator answers an agent's approval
       gate (approve/deny). require_role("operator"): viewers can't authorize
       dangerous actions; the team root key is admitted. The decision carries
       the operator's SIGNATURE (signature/nonce/signed_timestamp) for the
       AGENT to verify against the pinned public key before acting — the
       server is a courier that cannot forge (control/signing.py). Writes a
       pending `approval_decision` command on the target session.
  POST /v1/control/terminate        — an operator terminates a session
       (graceful: the agent tears down its own dependency tree) or, with a
       dependency_id, just one dependency. Same role gate + signed-auth shape.
  POST /v1/sessions/{id}/commands/claim — the AGENT claims its next pending
       command (first-wins compare-and-set). Authenticated by the team key
       (resolve_principal); team-scoped (a session from another team 404s).
  GET  /v1/sessions/{id}/commands   — the Inspector's read of a session's
       command history. Team-scoped, read-only.

This is G2's API. The agent-side poll/verify wiring (coding-agent guard.py +
runtime teardown) is built separately and gated on CC_CONTROL_PLANE_ENABLED,
pending live-test — like the F4 runtime. The decision state machine
(first-wins, void-on-staleness) lives in metadata_store_control_ext.py.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..infrastructure import MetadataStore
from ..infrastructure.metadata_store import get_metadata_store
from ..ingress.auth import require_role, resolve_principal
from ..models import Customer, Operator

logger = structlog.get_logger(__name__)

router = APIRouter()


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    """Best-effort parse of a signed-timestamp string for the display column;
    the *signed* value of record is the exact string in the payload."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


async def _require_team_session(
    store: MetadataStore, session_id: str, team_id: str
) -> dict[str, Any]:
    """Resolve a session and assert it belongs to the team, else 404 (no
    cross-team leak)."""
    session = await store.get_session(session_id)
    if session is None or session["team_id"] != team_id:
        raise HTTPException(status_code=404, detail="session not found")
    return session


class DecisionRequest(BaseModel):
    """An operator's signed approval decision for a session's approval gate."""
    session_id: str
    request_id: str
    decision: str  # approve | deny
    # Signed-authorization envelope (the agent verifies). signed_timestamp is
    # the EXACT string the operator signed over (the canonical-payload
    # timestamp); the agent reconstructs the payload with this same string.
    signature: Optional[str] = None
    nonce: Optional[str] = None
    signed_timestamp: Optional[str] = None


class TerminateRequest(BaseModel):
    """An operator's signed terminate command. dependency_id present → kill
    just that dependency (e.g. a runaway Chromium); absent → graceful session
    teardown."""
    session_id: str
    dependency_id: Optional[str] = None
    signature: Optional[str] = None
    nonce: Optional[str] = None
    signed_timestamp: Optional[str] = None


@router.post("/v1/control/decisions")
async def submit_decision(
    body: DecisionRequest,
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(require_role("operator"))
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> dict[str, Any]:
    """Write a pending approval decision for the agent to claim + verify."""
    if body.decision not in ("approve", "deny"):
        raise HTTPException(
            status_code=400, detail="decision must be 'approve' or 'deny'"
        )
    customer, operator = principal
    await _require_team_session(store, body.session_id, customer.id)

    payload: dict[str, Any] = {}
    if body.signed_timestamp:
        # The exact string the signature was computed over — the agent uses
        # this (not the parsed datetime) to rebuild the canonical payload.
        payload["signed_timestamp"] = body.signed_timestamp

    command = await store.create_control_command(
        body.session_id,
        customer.id,
        body.request_id,
        "approval_decision",
        decision=body.decision,
        payload=payload or None,
        signature=body.signature,
        nonce=body.nonce,
        signed_at=_parse_iso(body.signed_timestamp),
        issued_by_operator_id=operator.id if operator is not None else None,
    )
    return {"command": command}


@router.post("/v1/control/terminate")
async def submit_terminate(
    body: TerminateRequest,
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(require_role("operator"))
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> dict[str, Any]:
    """Write a pending terminate command (session or single dependency)."""
    customer, operator = principal
    await _require_team_session(store, body.session_id, customer.id)

    command_type = (
        "terminate_dependency" if body.dependency_id else "terminate"
    )
    payload: dict[str, Any] = {}
    if body.signed_timestamp:
        payload["signed_timestamp"] = body.signed_timestamp

    command = await store.create_control_command(
        body.session_id,
        customer.id,
        # Terminate is operator-initiated (no prior approval request); mint a
        # fresh request_id so first-wins + the agent's verification still key
        # on a unique id.
        f"term_{uuid.uuid4().hex[:16]}",
        command_type,
        dependency_id=body.dependency_id,
        payload=payload or None,
        signature=body.signature,
        nonce=body.nonce,
        signed_at=_parse_iso(body.signed_timestamp),
        issued_by_operator_id=operator.id if operator is not None else None,
    )
    return {"command": command}


@router.post("/v1/sessions/{session_id}/commands/claim")
async def claim_command(
    session_id: str,
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(resolve_principal)
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> dict[str, Any]:
    """The agent's poll: claim the next pending command for this session.

    Authenticated by the team key (the agent's credential); team-scoped. The
    claim is first-wins (pending→consumed in one transaction). Returns
    {command: <command>|null}. The agent verifies the signature on an
    approval_decision before acting.
    """
    customer, _actor = principal
    await _require_team_session(store, session_id, customer.id)
    command = await store.claim_next_command_for_session(session_id)
    return {"command": command}


@router.get("/v1/sessions/{session_id}/commands")
async def list_commands(
    session_id: str,
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(resolve_principal)
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    status: Optional[str] = None,
) -> dict[str, Any]:
    """The session's command history (Inspector view). Team-scoped, read-only.
    Optional ?status= filters (pending|consumed|voided)."""
    customer, _actor = principal
    await _require_team_session(store, session_id, customer.id)
    commands = await store.list_commands_for_session(session_id, status=status)
    return {"commands": commands}
