"""Activity trace (2026-10-19). Anthony: "I need to be able to review
every command, skill, tool, mcp used. I need to know every little detail
about any run, chat, or background process."

Pins:
  - every MCP tool call lands in tool_invocations through _tool_boundary
    (the one place every tool passes): tool, client, shortened args,
    ok / refusal code, duration; a recall links to its query_logs row;
    a refusal is recorded AND still raised as the protocol error;
  - arguments are bounded at the door (a pasted document never lands
    whole; a key-shaped argument is hidden);
  - the trace route returns the turn's self-report (every agent tool
    call), its model calls and the connected-tool call; dollars reach a
    platform admin only;
  - the model-calls and tool-calls routes list newest first.

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from crystal_cache.infrastructure.metadata_store_tool_ext import shorten, shorten_args, summarize_result
from crystal_cache.models.query_log import QueryLog

_T0 = datetime(2026, 10, 19, 22, 0, 0, tzinfo=timezone.utc)


def _app(store, encoder, vector_store, fact_vector_store):
    try:
        from tests.test_endpoint_smoke import _build_app
    except ModuleNotFoundError:
        from test_endpoint_smoke import _build_app
    return _build_app(store, encoder, vector_store, fact_vector_store)


def test_arguments_are_bounded_and_keys_hidden():
    long = "x" * 1000
    out = shorten_args({"text": long, "api_key": "sk-123", "key": "Rate|Now", "k": 5, "none": None, "tags": list(range(9)),
                        "nested": {"a": {"b": 1}, "c": "d"}})
    assert out["text"].startswith("x" * 300) and "+700 chars" in out["text"]
    assert out["api_key"] == "(hidden)" and out["key"] == "Rate|Now" and out["k"] == 5 and "none" not in out
    assert out["tags"][:5] == [0, 1, 2, 3, 4] and out["tags"][5] == "… 4 more"
    assert out["nested"] == {"a": "{1 keys}", "c": "d"}
    assert shorten(3.5) == 3.5 and shorten(True) is True
    assert summarize_result({"crystals": [1, 2], "count": 2, "note": "n" * 100, "meta": {"a": 1}}) == \
        "crystals: 2; count: 2; note: 100 chars; meta: {1 keys}"
    assert summarize_result("plain") == "plain" and summarize_result([1, 2, 3]) == "3 items"


@pytest.mark.asyncio
async def test_every_mcp_tool_call_is_recorded_with_its_outcome(store, customer, monkeypatch):
    from mcp.server.fastmcp.exceptions import ToolError

    from crystal_cache.agent import mcp_server

    recall_out = {"crystals": [{"crystal_id": "cR", "facts": [{"fact_id": "f1", "claim_text": "rate is 95", "score": 0.91}]}], "count": 1}
    tools = {"crystal_recall": SimpleNamespace(impl=lambda customer_id, **kw: _ret(recall_out))}
    monkeypatch.setattr(mcp_server, "get_registry", lambda: SimpleNamespace(get=lambda n: tools.get(n)))
    monkeypatch.setattr(mcp_server, "get_metadata_store", lambda: store)

    async def remember(text: str, key: str = "") -> dict:
        return {"stored": True, "crystal_id": "c1"}

    async def recall(query: str, k: int = 5) -> dict:
        return await mcp_server._dispatch("crystal_recall", query=query, k=k)

    async def forget(crystal_id: str) -> dict:
        return {"error": "not yours", "code": "forbidden"}

    async def boom(x: int) -> dict:
        raise ValueError("bad input")

    w_remember = mcp_server._tool_boundary(remember, name="remember")
    w_recall = mcp_server._tool_boundary(recall, name="recall")
    w_forget = mcp_server._tool_boundary(forget, name="forget")
    w_boom = mcp_server._tool_boundary(boom, name="memory_boom")
    tok = mcp_server._current_customer_id.set(customer.id)
    tok_c = mcp_server._current_client.set("Claude-User")
    try:
        assert (await w_remember(text="the rate is 95", key="Rate|Now"))["stored"] is True
        assert (await w_recall(query="what is the rate?"))["count"] == 1
        with pytest.raises(ToolError) as e1:
            await w_forget(crystal_id="c9")
        assert "[forbidden]" in str(e1.value)
        with pytest.raises(ValueError):
            await w_boom(x=1)
    finally:
        mcp_server._current_client.reset(tok_c)
        mcp_server._current_customer_id.reset(tok)

    rows = await store.list_tool_invocations(customer.id, limit=10)
    assert [r["tool"] for r in rows] == ["memory_boom", "forget", "recall", "remember"]
    by = {r["tool"]: r for r in rows}
    assert by["remember"]["ok"] is True and by["remember"]["client"] == "Claude-User"
    assert by["remember"]["args"] == {"text": "the rate is 95", "key": "Rate|Now"}
    assert by["remember"]["result_summary"] == "stored: True; crystal_id: c1"
    assert isinstance(by["remember"]["duration_ms"], int) and by["remember"]["surface"] == "mcp"
    # The recall links to the query log row it wrote.
    _total, qrows = await store.list_query_logs_for_customer(customer.id, limit=5)
    assert len(qrows) == 1 and by["recall"]["query_log_id"] == qrows[0].id
    assert by["remember"]["query_log_id"] is None
    # A refusal is recorded with its code; an exception with its type.
    assert by["forget"]["ok"] is False and by["forget"]["code"] == "forbidden" and by["forget"]["result_summary"] == "not yours"
    assert by["memory_boom"]["ok"] is False and by["memory_boom"]["code"] == "ValueError"
    assert by["memory_boom"]["result_summary"].startswith("ValueError: bad input")


async def _ret(v):
    return v


@pytest.mark.asyncio
async def test_trace_and_list_routes(store, customer, semantic_encoder_stub, vector_store, fact_vector_store, monkeypatch):
    from httpx import ASGITransport, AsyncClient

    from crystal_cache.config import Settings
    from crystal_cache.ingress import auth as auth_mod

    # A chat turn: its query log row, its self-report, its model calls.
    await store.write_query_log(QueryLog(
        id="qlog_turn", customer_id=customer.id, query_text="how does a crystal bank work?", match_type="high",
        injection_method="agent_tools", matched_facts=["f1"], client="console", sequence_id="seq_1", turn_index=0,
        model="nvidia/nemotron-3-super-120b-a12b", latency_ms=14043, timestamp=_T0,
    ))
    calls = [{"iteration": 1, "tool_name": "knowledge_search", "tool_use_id": "tu_1", "input": {"query": "crystal bank"},
              "output": {"matched_fact_ids": ["f1"], "injection_text": "..."}, "is_error": False, "duration_ms": 812}]
    await store.create_reasoning_trace(customer.id, [{"kind": "x"}], sequence_id="seq_1", turn_index=0, query_log_id="qlog_turn",
                                       crystals_used=["f1"], tool_calls=calls, inferences=[{"claim": "a"}],
                                       borders_crossed=[], gaps_felt=[{"gap": "pricing"}])
    for i, (origin, when) in enumerate([("agent", _T0 - timedelta(seconds=3)), ("self_critique", _T0 + timedelta(seconds=20)),
                                        ("agent", _T0 - timedelta(hours=2))]):
        await store.record_llm_call(customer.id, model="nvidia/nemotron-3-super-120b-a12b", input_tokens=1000 + i,
                                    output_tokens=50, session_id="seq_1", origin=origin)
        async with store.session() as s:
            from sqlalchemy import update
            from crystal_cache.infrastructure.schema import LlmCallRow
            await s.execute(update(LlmCallRow).where(LlmCallRow.input_tokens == 1000 + i).values(created_at=when))
    await store.record_tool_invocation(customer.id, tool="recall", client="Cursor", args={"query": "rate"}, ok=True,
                                       result_summary="count: 1", duration_ms=40, query_log_id="qlog_turn")
    await store.record_tool_invocation(customer.id, tool="remember", client="Cursor", args={"text": "x"}, ok=True, duration_ms=12)

    app = _app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    base = f"/admin/api/customers/{customer.id}"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as cl:
        r = await cl.get(f"{base}/query_logs/qlog_turn/trace")
        assert r.status_code == 200, r.text[:300]
        body = r.json()
        assert body["model"] == "nvidia/nemotron-3-super-120b-a12b" and body["sequence_id"] == "seq_1"
        assert body["trace"]["tool_calls"] == calls and body["trace"]["gaps_felt"] == [{"gap": "pricing"}]
        # Only the turn's model calls (the one two hours earlier is another turn).
        assert sorted(c["origin"] for c in body["model_calls"]) == ["agent", "self_critique"]
        assert body["tool_calls"][0]["tool"] == "recall"
        assert (await cl.get(f"{base}/query_logs/qlog_missing/trace")).status_code == 404

        r = await cl.get(f"{base}/model-calls", params={"limit": 10})
        assert r.status_code == 200 and r.json()["count"] == 3
        assert [c["input_tokens"] for c in r.json()["calls"]] == [1001, 1000, 1002]  # newest first
        r = await cl.get(f"{base}/model-calls", params={"session_id": "seq_1", "limit": 2})
        assert r.json()["count"] == 2

        r = await cl.get(f"{base}/tool-calls")
        assert [c["tool"] for c in r.json()["calls"]] == ["remember", "recall"]

        # Dollars: the gate off in tests makes every caller the platform admin.
        assert all("cost_micro_usd" in c for c in (await cl.get(f"{base}/model-calls")).json()["calls"])

    # With the admin gate ON, a tenant's own key sees tokens, never dollars.
    settings = Settings(environment="development", admin_api_key="adm_test_key", api_key_pepper="")
    monkeypatch.setattr(auth_mod, "get_settings", lambda: settings)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as cl:
        r = await cl.get(f"{base}/model-calls", headers={"Authorization": f"Bearer {customer.api_key}"})
        assert r.status_code == 200, r.text[:300]
        assert all("cost_micro_usd" not in c for c in r.json()["calls"]) and r.json()["count"] == 3
        r = await cl.get(f"{base}/model-calls", headers={"Authorization": "Bearer adm_test_key"})
        assert r.status_code == 200 and all("cost_micro_usd" in c for c in r.json()["calls"])
