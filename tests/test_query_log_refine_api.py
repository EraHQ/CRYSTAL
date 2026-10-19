"""Console Refine PR 11 (docs/CONSOLE_UX_AUDIT.md 5.7, Q65=A): the Query
log shows what a connected TOOL asked.

- Every query log row carries the client: the User-Agent family for API
  and MCP callers, "console" for the console's own routes.
- Recall-class MCP tools (memory_recall and the search tools) write a
  query log row with the client, the matched facts and the match
  strength from the customer's own thresholds; writes do not.
- The list route hydrates matched ids (facts, or the proxy rows' crystal
  ids) into memory text, so no fact id reaches a tenant.
- first_tool_query_at ignores console rows: Getting Started's last row
  completes on the first recall from a tool.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from crystal_cache.infrastructure.schema import CrystalRow, FactRow
from crystal_cache.ingress.client import client_from_request, ua_family
from crystal_cache.models.query_log import QueryLog

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
_T1 = datetime(2026, 3, 1, tzinfo=timezone.utc)


def _app(store, encoder, vector_store, fact_vector_store):
    try:
        from tests.test_endpoint_smoke import _build_app
    except ModuleNotFoundError:
        from test_endpoint_smoke import _build_app
    return _build_app(store, encoder, vector_store, fact_vector_store)


def _client(app):
    from httpx import ASGITransport, AsyncClient

    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t")


def test_client_label_is_the_user_agent_family_or_the_console():
    assert ua_family("Claude-User/1.0 (+https://claude.ai)") == "Claude-User"
    assert ua_family("node") == "node"
    assert ua_family("  python-httpx/0.27.0 ") == "python-httpx"
    assert ua_family("") is None and ua_family(None) is None
    assert len(ua_family("x" * 500)) == 64
    req = SimpleNamespace(url=SimpleNamespace(path="/admin/api/customers/c/agent"), headers={"user-agent": "Mozilla/5.0"})
    assert client_from_request(req) == "console"
    req = SimpleNamespace(url=SimpleNamespace(path="/v1/agent/messages"), headers={"user-agent": "Cursor/0.42 Electron"})
    assert client_from_request(req) == "Cursor"
    assert client_from_request(SimpleNamespace(url=SimpleNamespace(path="/v1/x"), headers={})) is None


@pytest.mark.asyncio
async def test_mcp_recall_tools_write_a_query_log_row_with_the_client(store, customer, monkeypatch):
    from crystal_cache.agent import mcp_server
    from crystal_cache.retrieval.match_classifier import MatchClassifier

    calls: list[tuple[str, dict]] = []

    class _Tool:
        def __init__(self, name, out):
            self.name, self.out = name, out

        async def impl(self, customer_id, **kwargs):
            calls.append((self.name, {"customer_id": customer_id, **kwargs}))
            return self.out

    recall_out = {"crystals": [{"crystal_id": "cR", "facts": [
        {"fact_id": "f1", "claim_text": "rate is 95", "score": 0.91},
        {"fact_id": "f2", "claim_text": "rate was 120", "score": 0.42},
    ]}], "count": 1}
    tools = {
        "crystal_recall": _Tool("crystal_recall", recall_out),
        "knowledge_search": _Tool("knowledge_search", {"injection_text": None, "matched_fact_ids": [], "matched_crystal_ids": [], "top_score": 0.0}),
        "crystal_write": _Tool("crystal_write", {"stored": True}),
    }
    monkeypatch.setattr(mcp_server, "get_registry", lambda: SimpleNamespace(get=lambda n: tools.get(n)))
    monkeypatch.setattr(mcp_server, "get_metadata_store", lambda: store)
    tok = mcp_server._current_customer_id.set(customer.id)
    tok_c = mcp_server._current_client.set("Claude-User")
    try:
        out = await mcp_server._dispatch("crystal_recall", query="what is the rate?", k=5)
        assert out == recall_out
        await mcp_server._dispatch("knowledge_search", query="nothing here", k=5, hints=None)
        await mcp_server._dispatch("crystal_write", key="k", value="v")
    finally:
        mcp_server._current_client.reset(tok_c)
        mcp_server._current_customer_id.reset(tok)

    total, rows = await store.list_query_logs_for_customer(customer.id, limit=10)
    assert total == 2, [r.query_text for r in rows]
    by_text = {r.query_text: r for r in rows}
    hit = by_text["what is the rate?"]
    assert hit.client == "Claude-User" and hit.injection_method == "mcp_tool"
    assert hit.matched_facts == ["f1", "f2"] and hit.upstream_call_made is False
    assert hit.match_type == MatchClassifier().classify(0.91, customer.retrieval_thresholds)
    assert isinstance(hit.latency_ms, int)
    miss = by_text["nothing here"]
    assert miss.match_type == "none" and miss.matched_facts == []
    assert mcp_server.recall_summary("crystal_recall", recall_out) == (["f1", "f2"], 0.91)
    assert mcp_server.recall_summary("content_search", {"matched_fact_ids": ["a"], "top_score": "0.7"}) == (["a"], 0.7)
    # The first recall from a tool is this one; the console never counts.
    first = await store.first_tool_query_at(customer.id)
    assert first is not None and first.tzinfo is not None
    # SQLite hands row timestamps back naive; they are UTC by construction.
    earliest = min(r.timestamp for r in rows)
    if earliest.tzinfo is None:
        earliest = earliest.replace(tzinfo=timezone.utc)
    assert first == earliest


@pytest.mark.asyncio
async def test_list_route_reads_memories_as_text_and_carries_the_client(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    async with store.session() as s:
        s.add(CrystalRow(id="cQ", customer_id=customer.id, crystal_type="customer:legacy", summary_vector=[], source_path="docs/rates.md"))
        s.add(FactRow(id="fq1", crystal_id="cQ", pair_type="question_answer", prompt_text="Rate|Now", claim_text="the rate is 95",
                      source_kind="operator_stated", vector=[], created_at=_T0))
        s.add(FactRow(id="fq2", crystal_id="cQ", pair_type="question_answer", prompt_text="Rate|Then", claim_text="the rate was 120",
                      source_kind="document_extraction", vector=[], created_at=_T1))
    assert await store.first_tool_query_at(customer.id) is None
    await store.write_query_log(QueryLog(id="qlog_console", customer_id=customer.id, query_text="console asked", match_type="high",
                                         injection_method="agent_tools", matched_facts=["fq1"], client="console",
                                         model="nvidia/nemotron-3-super-120b-a12b", timestamp=_T0))
    assert await store.first_tool_query_at(customer.id) is None
    await store.write_query_log(QueryLog(id="qlog_cursor", customer_id=customer.id, query_text="what is the rate?", match_type="high",
                                         injection_method="mcp_tool", matched_facts=["fq1", "fq_gone"], client="Cursor", timestamp=_T1))
    await store.write_query_log(QueryLog(id="qlog_proxy", customer_id=customer.id, query_text="proxy asked", match_type="medium",
                                         injection_method="text", matched_facts=["cQ"], client=None, timestamp=_T1))
    assert (await store.first_tool_query_at(customer.id)) == _T1

    app = _app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    async with _client(app) as cl:
        r = await cl.get(f"/admin/api/customers/{customer.id}/query_logs", params={"limit": 10})
        assert r.status_code == 200, r.text[:300]
        rows = {x["id"]: x for x in r.json()["items"]}
        assert rows["qlog_cursor"]["client"] == "Cursor"
        assert rows["qlog_cursor"]["matched_memories"] == [{"id": "fq1", "text": "the rate is 95"}]
        assert rows["qlog_console"]["client"] == "console"
        # M3 follow-up: the model that answered rides along; recall-only
        # rows have none.
        assert rows["qlog_console"]["model"] == "nvidia/nemotron-3-super-120b-a12b"
        assert rows["qlog_cursor"]["model"] is None
        # A crystal id (proxy rows) reads as that crystal's first fact.
        assert rows["qlog_proxy"]["matched_memories"] == [{"id": "cQ", "text": "the rate is 95"}]
        assert rows["qlog_proxy"]["client"] is None
