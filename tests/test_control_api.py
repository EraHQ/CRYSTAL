"""Growth G2 — control-plane API (endpoints/control.py) tests.

Direct-call convention (principal injected). Role-gating is enforced by the
shared require_role dependency and covered at the auth layer; the command
state machine itself is covered in test_control_plane.py. These focus on
endpoint wiring: decision validation, team-scoping + the cross-team 404 guard,
and delegation to the store.
"""
from __future__ import annotations

import uuid

import pytest
from fastapi import HTTPException

from crystal_cache.endpoints.control import (
    DecisionRequest,
    TerminateRequest,
    claim_command,
    list_commands,
    submit_decision,
    submit_terminate,
)


async def _register_session(store, customer, status="awaiting_approval"):
    sid = f"sess_{uuid.uuid4().hex[:12]}"
    await store.register_session(sid, customer.id, status=status)
    return sid


async def test_submit_decision_creates_pending_command(store, customer):
    sid = await _register_session(store, customer)
    resp = await submit_decision(
        body=DecisionRequest(
            session_id=sid, request_id="req_1", decision="approve",
            signature="sig", nonce="n1",
            signed_timestamp="2026-06-15T09:00:00+00:00",
        ),
        principal=(customer, None),
        store=store,
    )
    cmd = resp["command"]
    assert cmd["status"] == "pending"
    assert cmd["command_type"] == "approval_decision"
    assert cmd["decision"] == "approve"
    assert cmd["signature"] == "sig"
    # The exact signed-timestamp string is preserved in the payload so the
    # agent can rebuild the canonical payload it verifies against.
    assert cmd["payload"]["signed_timestamp"] == "2026-06-15T09:00:00+00:00"


async def test_submit_decision_rejects_bad_decision(store, customer):
    sid = await _register_session(store, customer)
    with pytest.raises(HTTPException) as exc:
        await submit_decision(
            body=DecisionRequest(
                session_id=sid, request_id="r", decision="maybe",
            ),
            principal=(customer, None),
            store=store,
        )
    assert exc.value.status_code == 400


async def test_submit_decision_cross_team_404(store, customer):
    other = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="x",
    )
    foreign_sid = f"sess_{uuid.uuid4().hex[:12]}"
    await store.register_session(
        foreign_sid, other.id, status="awaiting_approval",
    )
    with pytest.raises(HTTPException) as exc:
        await submit_decision(
            body=DecisionRequest(
                session_id=foreign_sid, request_id="r", decision="approve",
            ),
            principal=(customer, None),
            store=store,
        )
    assert exc.value.status_code == 404


async def test_terminate_session_and_dependency(store, customer):
    sid = await _register_session(store, customer, status="running")
    resp = await submit_terminate(
        body=TerminateRequest(session_id=sid),
        principal=(customer, None),
        store=store,
    )
    assert resp["command"]["command_type"] == "terminate"

    resp2 = await submit_terminate(
        body=TerminateRequest(session_id=sid, dependency_id="sdep_x"),
        principal=(customer, None),
        store=store,
    )
    assert resp2["command"]["command_type"] == "terminate_dependency"
    assert resp2["command"]["dependency_id"] == "sdep_x"


async def test_claim_and_list_commands(store, customer):
    sid = await _register_session(store, customer)
    await submit_decision(
        body=DecisionRequest(session_id=sid, request_id="r1", decision="deny"),
        principal=(customer, None),
        store=store,
    )
    claimed = await claim_command(
        session_id=sid, principal=(customer, None), store=store,
    )
    assert claimed["command"] is not None
    assert claimed["command"]["decision"] == "deny"
    # First-wins: nothing left to claim.
    again = await claim_command(
        session_id=sid, principal=(customer, None), store=store,
    )
    assert again["command"] is None
    # The history shows the (now consumed) command.
    listed = await list_commands(
        session_id=sid, principal=(customer, None), store=store,
    )
    assert len(listed["commands"]) == 1


async def test_claim_cross_team_404(store, customer):
    other = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="x",
    )
    foreign_sid = f"sess_{uuid.uuid4().hex[:12]}"
    await store.register_session(foreign_sid, other.id, status="running")
    with pytest.raises(HTTPException) as exc:
        await claim_command(
            session_id=foreign_sid, principal=(customer, None), store=store,
        )
    assert exc.value.status_code == 404
