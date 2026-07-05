"""Foundation F4 — session registry (surface consolidation) store tests.

The substrate behind "see CRYS activity in the Inspector": register a
session, heartbeat it, scope listings by team + operator, and — the
load-bearing property — infer liveness from staleness (a session whose
heartbeat goes stale reads as crashed, and the sweep materializes that +
orphans its dependencies). Direct against the in-memory store fixture;
asyncio_mode=auto.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from crystal_cache.infrastructure.schema import AgentSessionRow


async def _age_session_heartbeat(store, session_id, seconds):
    """Push a session's last_heartbeat_at into the past (test-only — the
    store methods always stamp 'now', so staleness needs a manual nudge)."""
    async with store.session() as session:
        row = await session.get(AgentSessionRow, session_id)
        row.last_heartbeat_at = datetime.now(timezone.utc) - timedelta(
            seconds=seconds
        )
        await session.commit()


async def test_register_and_get_session(store, customer):
    sid = f"sess_{uuid.uuid4().hex[:12]}"
    await store.register_session(
        sid, customer.id,
        host="laptop", pid=4242, project_dir="/home/x/proj",
        model="claude-opus-4-8", status="running", current_action="planning",
    )
    s = await store.get_session(sid)
    assert s is not None
    assert s["session_id"] == sid
    assert s["team_id"] == customer.id
    assert s["host"] == "laptop"
    assert s["pid"] == 4242
    assert s["status"] == "running"
    assert s["is_stale"] is False
    assert s["effective_status"] == "running"


async def test_heartbeat_updates_status_and_action(store, customer):
    sid = f"sess_{uuid.uuid4().hex[:12]}"
    await store.register_session(
        sid, customer.id, status="running", current_action="planning",
    )
    ok = await store.heartbeat_session(
        sid, status="awaiting_approval", current_action="write_file guard",
    )
    assert ok is True
    s = await store.get_session(sid)
    assert s["status"] == "awaiting_approval"
    assert s["current_action"] == "write_file guard"

    # A heartbeat for an unknown session is a clean False, not an error.
    assert await store.heartbeat_session("sess_nope", status="running") is False


async def test_list_sessions_scoped_by_team_and_operator(store, customer):
    op, _ = await store.create_operator(
        team_id=customer.id, display_name="Ada",
    )
    s_op = f"sess_{uuid.uuid4().hex[:12]}"
    s_team = f"sess_{uuid.uuid4().hex[:12]}"
    await store.register_session(
        s_op, customer.id, operator_id=op.id, status="running",
    )
    await store.register_session(s_team, customer.id, status="running")

    # Another team's session must not leak into this team's listing.
    other = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="x",
    )
    await store.register_session(
        f"sess_{uuid.uuid4().hex[:12]}", other.id, status="running",
    )

    all_team = await store.list_sessions_for_team(customer.id)
    assert {s["session_id"] for s in all_team} == {s_op, s_team}

    op_only = await store.list_sessions_for_team(
        customer.id, operator_id=op.id,
    )
    assert {s["session_id"] for s in op_only} == {s_op}


async def test_stale_session_reads_crashed_and_sweep_orphans_deps(
    store, customer,
):
    sid = f"sess_{uuid.uuid4().hex[:12]}"
    await store.register_session(
        sid, customer.id, status="running", current_action="long task",
    )
    dep = await store.register_dependency(
        sid, kind="browser", descriptor="chromium pid 9", pid=9,
    )

    # Fresh → live.
    fresh = await store.get_session(sid, stale_seconds=90)
    assert fresh["is_stale"] is False

    # Age it past the window → derived liveness reports crashed with NO
    # mutation: the self-reported status is still 'running'.
    await _age_session_heartbeat(store, sid, 600)
    stale = await store.get_session(sid, stale_seconds=90)
    assert stale["is_stale"] is True
    assert stale["effective_status"] == "crashed"
    assert stale["status"] == "running"

    # Sweep materializes the crash and orphans the dependency.
    assert await store.mark_stale_sessions(stale_seconds=90) == 1
    after = await store.get_session(sid)
    assert after["status"] == "crashed"
    deps = await store.list_dependencies_for_session(sid)
    assert len(deps) == 1
    assert deps[0]["dependency_id"] == dep["dependency_id"]
    assert deps[0]["status"] == "orphaned"

    # Sweeping again is a no-op — 'crashed' is terminal.
    assert await store.mark_stale_sessions(stale_seconds=90) == 0


async def test_dependencies_lifecycle(store, customer):
    sid = f"sess_{uuid.uuid4().hex[:12]}"
    await store.register_session(sid, customer.id, status="running")
    d1 = await store.register_dependency(
        sid, kind="mcp_server", descriptor="filesystem",
    )
    d2 = await store.register_dependency(
        sid, kind="subprocess", descriptor="pytest", pid=123,
    )

    deps = await store.list_dependencies_for_session(sid)
    assert len(deps) == 2
    assert {d["kind"] for d in deps} == {"mcp_server", "subprocess"}

    assert await store.update_dependency_status(
        d1["dependency_id"], status="exited",
    ) is True
    by_id = {
        d["dependency_id"]: d
        for d in await store.list_dependencies_for_session(sid)
    }
    assert by_id[d1["dependency_id"]]["status"] == "exited"
    assert by_id[d2["dependency_id"]]["status"] == "active"
