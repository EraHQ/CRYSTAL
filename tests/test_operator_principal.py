"""Phase 1.4 Gate 1 — the operator principal at the doors (Q1=A, Q4=C).

Covers, in layer order:
  1. `agent/principal.py` contract: default None (the system lane),
     set/get/reset, and the detached-task snapshot semantics the agent
     door depends on (contextvars copy at task creation, so resetting in
     the handler cannot un-pin an already-created pipeline task).
  2. The agent door: viewers 403 outright; the acting operator is pinned
     for the duration of `run_agent_messages` and reset after.
  3. The MCP door: operator keys are first-class (tried before the team
     path), suspended operators 403, team keys act as the Default Admin
     (P1), unresolvable tokens 401 — and both contextvars are set for
     the inner app and reset after.
  4. Q4=C at tool grain: a viewer principal is refused by the mutating
     memory_* tools (structured error result — JSON-RPC has no per-tool
     HTTP status) while the read tools still dispatch.

Nothing here asserts retrieval filtering — Gate 1 is deliberately inert
below the doors; the routers learn the operator kwarg in Gate 2.
asyncio_mode=auto.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from crystal_cache.agent.principal import (
    get_current_operator,
    reset_current_operator,
    set_current_operator,
)


# ---------------------------------------------------------------------------
# 1. principal.py contract
# ---------------------------------------------------------------------------

async def test_operator_contextvar_default_set_reset():
    assert get_current_operator() is None  # system lane default
    op = SimpleNamespace(id="op_x", role="operator", team_id="cus_x")
    token = set_current_operator(op)
    try:
        assert get_current_operator() is op
    finally:
        reset_current_operator(token)
    assert get_current_operator() is None


async def test_operator_contextvar_detached_task_snapshot():
    """The door's ordering guarantee: a task created while the var is set
    keeps seeing it after the parent resets — the detached agent pipeline
    inherits the principal for its whole life."""
    op = SimpleNamespace(id="op_snap", role="operator", team_id="cus_x")

    async def reader():
        await asyncio.sleep(0)
        return get_current_operator()

    token = set_current_operator(op)
    task = asyncio.create_task(reader())
    reset_current_operator(token)

    assert get_current_operator() is None  # parent context restored
    assert (await task) is op  # the task's snapshot survived the reset


# ---------------------------------------------------------------------------
# 2. The agent door (endpoints/agent.py::agent_messages)
# ---------------------------------------------------------------------------

def _agent_body():
    from crystal_cache.endpoints.agent import AgentRequest
    return AgentRequest(messages=[{"role": "user", "content": "hi"}])


async def test_agent_door_rejects_viewer(store, customer):
    from crystal_cache.endpoints.agent import agent_messages

    viewer, _key = await store.create_operator(
        team_id=customer.id, display_name="V", role="viewer",
    )
    # The viewer gate fires before the pipeline touches the request, so a
    # bare stand-in suffices (the suite's injected-principal convention).
    with pytest.raises(HTTPException) as exc:
        await agent_messages(
            body=_agent_body(),
            request=SimpleNamespace(),
            principal=(customer, viewer),
            store=store,
        )
    assert exc.value.status_code == 403


async def test_agent_door_pins_operator_for_run_and_resets(
    store, customer, monkeypatch,
):
    import crystal_cache.endpoints.agent as agent_ep

    op, _key = await store.create_operator(
        team_id=customer.id, display_name="A", role="operator",
    )
    seen: dict = {}

    async def _fake_run(**kwargs):
        # What the pipeline (and, transitively, any detached task it
        # creates) observes as the acting operator.
        seen["operator"] = get_current_operator()
        return JSONResponse(content={"ok": True})

    monkeypatch.setattr(agent_ep, "run_agent_messages", _fake_run)

    resp = await agent_ep.agent_messages(
        body=_agent_body(),
        request=SimpleNamespace(),
        principal=(customer, op),
        store=store,
    )
    assert resp.status_code == 200
    assert seen["operator"] is op
    assert get_current_operator() is None  # reset after the handler


# ---------------------------------------------------------------------------
# 3. The MCP door (agent/mcp_server.py middleware)
# ---------------------------------------------------------------------------

def _http_scope(token: str | None) -> dict:
    headers = []
    if token is not None:
        headers.append((b"authorization", f"Bearer {token}".encode("latin-1")))
    return {"type": "http", "headers": headers}


class _SendRecorder:
    def __init__(self) -> None:
        self.frames: list[dict] = []

    async def __call__(self, frame: dict) -> None:
        self.frames.append(frame)

    @property
    def status(self):
        for f in self.frames:
            if f.get("type") == "http.response.start":
                return f.get("status")
        return None


async def _recv():  # pragma: no cover - middleware never reads the body here
    return {"type": "http.request", "body": b""}


def _middleware(inner, monkeypatch, store_obj):
    import crystal_cache.agent.mcp_server as mcp_srv
    monkeypatch.setattr(mcp_srv, "get_metadata_store", lambda: store_obj)
    return mcp_srv._CustomerKeyAuthMiddleware(inner), mcp_srv


async def test_mcp_door_operator_key_sets_both_contextvars(
    store, customer, monkeypatch,
):
    op, op_key = await store.create_operator(
        team_id=customer.id, display_name="A", role="operator",
    )
    seen: dict = {}

    async def inner(scope, receive, send):
        import crystal_cache.agent.mcp_server as mcp_srv
        seen["customer_id"] = mcp_srv._current_customer_id.get()
        seen["operator"] = get_current_operator()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    mw, _ = _middleware(inner, monkeypatch, store)
    send = _SendRecorder()
    await mw(_http_scope(op_key), _recv, send)

    assert send.status == 200
    assert seen["customer_id"] == customer.id
    assert seen["operator"].id == op.id
    # Reset after the request — the next context starts clean.
    assert get_current_operator() is None


async def test_mcp_door_team_key_acts_as_default_admin(
    store, customer, monkeypatch,
):
    seen: dict = {}

    async def inner(scope, receive, send):
        seen["operator"] = get_current_operator()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    mw, _ = _middleware(inner, monkeypatch, store)
    send = _SendRecorder()
    await mw(_http_scope(customer.api_key), _recv, send)

    assert send.status == 200
    assert seen["operator"] is not None  # P1: never an operator-less request
    assert seen["operator"].role == "admin"
    assert seen["operator"].team_id == customer.id


async def test_mcp_door_suspended_operator_403(monkeypatch):
    suspended = SimpleNamespace(status="suspended", team_id="cus_x")

    class _Store:
        async def get_operator_by_api_key(self, token):
            return suspended

        async def get_customer_by_id(self, team_id):  # pragma: no cover
            return SimpleNamespace(id=team_id)

    called = {"inner": False}

    async def inner(scope, receive, send):  # pragma: no cover - must not run
        called["inner"] = True

    mw, _ = _middleware(inner, monkeypatch, _Store())
    send = _SendRecorder()
    await mw(_http_scope("op_key_sus"), _recv, send)

    assert send.status == 403
    assert called["inner"] is False


async def test_mcp_door_invalid_key_401(store, monkeypatch):
    async def inner(scope, receive, send):  # pragma: no cover - must not run
        raise AssertionError("inner app reached without auth")

    mw, _ = _middleware(inner, monkeypatch, store)
    send = _SendRecorder()
    await mw(_http_scope("not-a-real-key"), _recv, send)
    assert send.status == 401


# ---------------------------------------------------------------------------
# 4. Q4=C at tool grain — viewers refused by the mutating memory_* tools
# ---------------------------------------------------------------------------

_VIEWER = SimpleNamespace(id="op_v", role="viewer", team_id="cus_x")


async def test_mcp_viewer_blocked_on_mutating_tools(monkeypatch):
    import crystal_cache.agent.mcp_server as mcp_srv

    dispatched: list[str] = []

    async def _fake_dispatch(name, **kwargs):
        dispatched.append(name)
        return {"ok": True}

    monkeypatch.setattr(mcp_srv, "_dispatch", _fake_dispatch)

    token = set_current_operator(_VIEWER)
    try:
        # Every mutating tool refuses without dispatching.
        assert (await mcp_srv.memory_store(key="k", value="v"))["code"] == "viewer_forbidden"
        assert (await mcp_srv.memory_learn(prompt="p", response="r"))["code"] == "viewer_forbidden"
        assert (await mcp_srv.memory_record_gap(
            question="q", disposition="researchable",
        ))["code"] == "viewer_forbidden"
        assert (await mcp_srv.memory_forget(crystal_id="c1"))["code"] == "viewer_forbidden"
        assert (await mcp_srv.memory_ingest(text="doc text"))["code"] == "viewer_forbidden"
        assert (await mcp_srv.memory_import(records=[{"key": "k", "value": "v"}]))[
            "code"
        ] == "viewer_forbidden"
        assert dispatched == []

        # Read tools still dispatch for a viewer (Q4=C: read-only MCP seat).
        out = await mcp_srv.memory_search(query="anything")
        assert out == {"ok": True}
        assert dispatched == ["knowledge_search"]
    finally:
        reset_current_operator(token)


async def test_mcp_non_viewer_writes_pass_the_block(monkeypatch):
    import crystal_cache.agent.mcp_server as mcp_srv

    async def _fake_dispatch(name, **kwargs):
        return {"dispatched": name}

    monkeypatch.setattr(mcp_srv, "_dispatch", _fake_dispatch)

    op = SimpleNamespace(id="op_w", role="operator", team_id="cus_x")
    token = set_current_operator(op)
    try:
        out = await mcp_srv.memory_store(key="k", value="v")
        assert out == {"dispatched": "crystal_write"}
    finally:
        reset_current_operator(token)
