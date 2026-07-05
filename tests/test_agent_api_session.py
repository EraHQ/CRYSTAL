"""D — agent-API session registration (HTTP surface in the Agents view).

The agent endpoint is stateless, so each request registers an ephemeral
session (register → turn_started → run → turn_completed → exited) directly via
the store's session methods — `SessionHandle` lives in the CRYS
package, which the library can't import. These lock the two helpers
`run_agent_messages` wires in: `_register_agent_api_session` creates a running
session + a turn_started event; `_complete_agent_api_session` appends
turn_completed (with tokens + cost) and marks the session exited (so it reads
as a finished turn, not a session that went stale → 'crashed'). Both are
best-effort — a registry failure must never break the API response.

The full run_agent_messages happy path needs a live upstream call + app.state
(see test_admin_chat's note), so it isn't exercised here; these cover the D
logic the endpoint dispatches to.

R14 note: verified by pytest; describes expected behavior, not yet run at
authoring time.
"""
from __future__ import annotations

from crystal_cache.endpoints.agent import (
    _complete_agent_api_session,
    _register_agent_api_session,
)


async def test_register_creates_running_session_and_turn_started(store, customer):
    await _register_agent_api_session(
        store, session_id="crysapi_1", team_id=customer.id,
        model="claude-sonnet-4-5-20250929", label="build me a thing",
    )

    s = await store.get_session("crysapi_1")
    assert s is not None
    assert s["status"] == "running"
    assert s["model"] == "claude-sonnet-4-5-20250929"
    assert s["team_id"] == customer.id
    assert s["project_dir"] is None  # HTTP turn has no project

    events = await store.list_events_for_session("crysapi_1")
    assert len(events) == 1
    assert events[0]["event_type"] == "turn_started"
    assert events[0]["label"] == "build me a thing"
    assert events[0]["payload"] == {"surface": "agent_api"}
    assert events[0]["turn_index"] == 0


async def test_complete_records_turn_completed_and_exits(store, customer):
    await _register_agent_api_session(
        store, session_id="crysapi_2", team_id=customer.id,
        model="m", label="q",
    )
    result = {
        "final_text": "Here is the answer.",
        "prompt_tokens": 800,
        "completion_tokens": 120,
        "iterations": 2,
        "stop_reason": "end_turn",
    }
    await _complete_agent_api_session(
        store, session_id="crysapi_2", team_id=customer.id,
        result=result, cost_micro_usd=4242, duration_ms=1500,
    )

    # Closed cleanly (exited) — not left to go stale → 'crashed'.
    s = await store.get_session("crysapi_2")
    assert s["status"] == "exited"

    events = await store.list_events_for_session("crysapi_2")
    # turn_started (from register) then turn_completed (from complete).
    assert [e["event_type"] for e in events] == [
        "turn_started", "turn_completed",
    ]
    done = events[1]
    assert done["status"] == "ok"
    assert done["tokens_input"] == 800
    assert done["tokens_output"] == 120
    assert done["cost_micro_usd"] == 4242
    assert done["duration_ms"] == 1500
    assert done["payload"]["stop_reason"] == "end_turn"


async def test_register_is_failsafe_on_store_error(store, customer, monkeypatch):
    # A registry hiccup must never raise into the API response.
    async def _boom(*a, **k):
        raise RuntimeError("registry down")
    monkeypatch.setattr(store, "register_session", _boom)

    # Must not raise.
    await _register_agent_api_session(
        store, session_id="crysapi_3", team_id=customer.id,
        model="m", label="q",
    )
    # Nothing was registered, but the failure was swallowed.
    assert await store.get_session("crysapi_3") is None


async def test_complete_is_failsafe_on_store_error(store, customer, monkeypatch):
    await _register_agent_api_session(
        store, session_id="crysapi_4", team_id=customer.id, model="m", label="q",
    )

    async def _boom(*a, **k):
        raise RuntimeError("events down")
    monkeypatch.setattr(store, "record_event", _boom)

    # Must not raise even though recording the completion event fails.
    await _complete_agent_api_session(
        store, session_id="crysapi_4", team_id=customer.id,
        result={"final_text": "x"}, cost_micro_usd=None, duration_ms=10,
    )
