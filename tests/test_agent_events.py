"""Tests for the Unify-Agents event stream (agent_events).

The append-only per-session activity record CRYS writes. Covers seq
monotonicity (per session, independent across sessions), incremental
after_seq reads (the live-timeline poll), the team-scoped filtered read
(the unified interaction log), and payload/cost round-tripping.

In-memory store per test (conftest `store` fixture creates all tables via
init(), so agent_events exists without the migration — the migration covers
the persistent Alembic DB). asyncio_mode=auto, so plain `async def` tests.
"""
from __future__ import annotations


async def test_record_and_list_single_event(store):
    ev = await store.record_event(
        "crys_sess_1",
        event_type="turn_started",
        team_id="cus_x",
        label="build a game",
        phase="turn",
        turn_index=0,
        payload={"prompt": "build a game"},
    )
    assert ev["seq"] == 0
    assert ev["event_type"] == "turn_started"
    assert ev["id"].startswith("aev_")

    events = await store.list_events_for_session("crys_sess_1")
    assert len(events) == 1
    assert events[0]["label"] == "build a game"
    assert events[0]["payload"] == {"prompt": "build a game"}
    assert events[0]["turn_index"] == 0
    assert events[0]["phase"] == "turn"


async def test_seq_is_monotonic_per_session(store):
    for i in range(5):
        await store.record_event("s1", event_type="tool_called", label=f"t{i}")
    events = await store.list_events_for_session("s1")
    assert [e["seq"] for e in events] == [0, 1, 2, 3, 4]
    # Stream order matches insertion order.
    assert [e["label"] for e in events] == [f"t{i}" for i in range(5)]


async def test_seq_independent_across_sessions(store):
    await store.record_event("a", event_type="turn_started")
    await store.record_event("b", event_type="turn_started")
    await store.record_event("a", event_type="tool_called")
    a = await store.list_events_for_session("a")
    b = await store.list_events_for_session("b")
    assert [e["seq"] for e in a] == [0, 1]
    assert [e["seq"] for e in b] == [0]


async def test_after_seq_returns_only_newer(store):
    for i in range(4):
        await store.record_event("s", event_type="tool_called", label=f"t{i}")
    newer = await store.list_events_for_session("s", after_seq=1)
    assert [e["seq"] for e in newer] == [2, 3]
    # Nothing newer than the last seq.
    assert await store.list_events_for_session("s", after_seq=3) == []


async def test_list_for_team_filters_by_type_and_scopes(store):
    await store.record_event("s", event_type="turn_started", team_id="T")
    await store.record_event("s", event_type="tool_called", team_id="T")
    await store.record_event(
        "s",
        event_type="turn_completed",
        team_id="T",
        tokens_input=100,
        tokens_output=50,
        cost_micro_usd=1234,
    )
    await store.record_event("s2", event_type="turn_started", team_id="OTHER")

    turns = await store.list_events_for_team(
        "T", event_types=["turn_started", "turn_completed"]
    )
    types = {e["event_type"] for e in turns}
    assert types == {"turn_started", "turn_completed"}  # tool_called filtered out
    assert all(e["team_id"] == "T" for e in turns)  # OTHER team excluded

    completed = next(e for e in turns if e["event_type"] == "turn_completed")
    assert completed["cost_micro_usd"] == 1234
    assert completed["tokens_input"] == 100
    assert completed["tokens_output"] == 50


async def test_list_for_team_unfiltered_returns_all_for_team(store):
    await store.record_event("s", event_type="turn_started", team_id="T")
    await store.record_event("s", event_type="tool_called", team_id="T")
    rows = await store.list_events_for_team("T")
    assert len(rows) == 2


async def test_unknown_session_lists_empty(store):
    assert await store.list_events_for_session("does_not_exist") == []


async def test_subagent_event_carries_parent(store):
    ev = await store.record_event(
        "worker_sess",
        event_type="subagent_started",
        team_id="T",
        parent_session_id="crys_sess_1",
        phase="subagent",
        label="delegating research: map module Y",
        payload={"task": "map module Y"},
    )
    assert ev["parent_session_id"] == "crys_sess_1"
    assert ev["phase"] == "subagent"
