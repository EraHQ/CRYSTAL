"""Foundation F4 — session API (endpoints/sessions.py) tests.

The control-plane heartbeat + read endpoints over the session registry.
Direct-call convention (principal injected); the registry behavior itself is
covered in test_session_registry.py, so these focus on endpoint wiring —
principal → team/operator attribution, team scoping, the cross-team 404
guard, and the sweep-on-heartbeat.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException

from crystal_cache.endpoints.sessions import (
    HeartbeatRequest,
    list_session_dependencies,
    list_sessions,
    session_heartbeat,
)
from crystal_cache.infrastructure.schema import AgentSessionRow


async def _age_session_heartbeat(store, session_id, seconds):
    async with store.session() as session:
        row = await session.get(AgentSessionRow, session_id)
        row.last_heartbeat_at = datetime.now(timezone.utc) - timedelta(
            seconds=seconds
        )
        await session.commit()


async def test_heartbeat_registers_and_refreshes(store, customer):
    sid = f"sess_{uuid.uuid4().hex[:12]}"
    resp = await session_heartbeat(
        body=HeartbeatRequest(
            session_id=sid, status="running", host="laptop", pid=99,
            project_dir="/p", model="claude-opus-4-8",
            current_action="planning",
        ),
        principal=(customer, None),
        store=store,
    )
    assert resp["session"]["session_id"] == sid
    assert resp["session"]["team_id"] == customer.id
    assert resp["session"]["status"] == "running"
    assert resp["session"]["operator_id"] is None

    # A second beat updates the same row (idempotent upsert).
    resp2 = await session_heartbeat(
        body=HeartbeatRequest(session_id=sid, status="idle"),
        principal=(customer, None),
        store=store,
    )
    assert resp2["session"]["status"] == "idle"
    assert (await store.get_session(sid))["status"] == "idle"


async def test_heartbeat_attributes_operator(store, customer):
    op, _ = await store.create_operator(
        team_id=customer.id, display_name="Ada",
    )
    sid = f"sess_{uuid.uuid4().hex[:12]}"
    resp = await session_heartbeat(
        body=HeartbeatRequest(session_id=sid, status="running"),
        principal=(customer, op),
        store=store,
    )
    assert resp["session"]["operator_id"] == op.id


async def test_list_sessions_team_scoped_and_operator_filter(store, customer):
    op, _ = await store.create_operator(
        team_id=customer.id, display_name="Ada",
    )
    s_op = f"sess_{uuid.uuid4().hex[:12]}"
    s_team = f"sess_{uuid.uuid4().hex[:12]}"
    await store.register_session(
        s_op, customer.id, operator_id=op.id, status="running",
    )
    await store.register_session(s_team, customer.id, status="running")
    other = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="x",
    )
    await store.register_session(
        f"sess_{uuid.uuid4().hex[:12]}", other.id, status="running",
    )

    resp = await list_sessions(principal=(customer, None), store=store)
    assert {s["session_id"] for s in resp["sessions"]} == {s_op, s_team}

    resp_op = await list_sessions(
        principal=(customer, None), store=store, operator_id=op.id,
    )
    assert {s["session_id"] for s in resp_op["sessions"]} == {s_op}


async def test_dependencies_endpoint_team_scoped_404(store, customer):
    # A session under ANOTHER team must not be readable here.
    other = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="x",
    )
    foreign_sid = f"sess_{uuid.uuid4().hex[:12]}"
    await store.register_session(foreign_sid, other.id, status="running")
    await store.register_dependency(
        foreign_sid, kind="browser", descriptor="chromium",
    )

    with pytest.raises(HTTPException) as exc:
        await list_session_dependencies(
            session_id=foreign_sid, principal=(customer, None), store=store,
        )
    assert exc.value.status_code == 404

    # A session under the caller's own team returns its dependencies.
    own_sid = f"sess_{uuid.uuid4().hex[:12]}"
    await store.register_session(own_sid, customer.id, status="running")
    await store.register_dependency(
        own_sid, kind="mcp_server", descriptor="filesystem",
    )
    resp = await list_session_dependencies(
        session_id=own_sid, principal=(customer, None), store=store,
    )
    assert len(resp["dependencies"]) == 1
    assert resp["dependencies"][0]["kind"] == "mcp_server"


async def test_heartbeat_sweeps_stale_sessions(store, customer):
    stale_sid = f"sess_{uuid.uuid4().hex[:12]}"
    await store.register_session(stale_sid, customer.id, status="running")
    await _age_session_heartbeat(store, stale_sid, 600)

    # A heartbeat from ANY session triggers the stale sweep.
    live_sid = f"sess_{uuid.uuid4().hex[:12]}"
    await session_heartbeat(
        body=HeartbeatRequest(session_id=live_sid, status="running"),
        principal=(customer, None),
        store=store,
    )
    assert (await store.get_session(stale_sid))["status"] == "crashed"
