"""Audit item (d) — /mcp metering + limits (Q1=B, Q2=A, Q3=A, Q4=A, Q5=A,
ratified 2026-08-25).

Pins, in layer order:
  1. Rate-limit classing: /mcp is its own class with its own knob; the
     three R11-found unlimited routes (/v1/agent/messages, /v1/export,
     /v1/import) are in the expensive class; distinct bearer tokens get
     distinct buckets; unclassified paths stay unlimited; and the
     pre-existing two-arg factory call shape leaves MCP unlimited
     (back-compat, byte-for-byte).
  2. memory_ingest refuses over-ceiling text BEFORE any store write or
     model call (proven with an exploding create_document_upload).
  3. memory_export pagination: stable disjoint pages, has_more/total
     honest, a full page-walk equals the whole bank, and the store
     method's ordering is deterministic.
  4. The idle-stamp predicate counts /v1/* and /mcp, nothing else.

asyncio_mode=auto.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from crystal_cache.ingress.rate_limit import build_rate_limit_middleware
from crystal_cache.workers.idle import is_substantive_path


# ---------------------------------------------------------------------------
# 1. Rate-limit classing
# ---------------------------------------------------------------------------

def _tiny_app(**limits) -> FastAPI:
    app = FastAPI()
    app.middleware("http")(build_rate_limit_middleware(**limits))

    @app.get("/mcp")
    @app.post("/mcp")
    @app.get("/v1/agent/messages")
    @app.get("/v1/export")
    @app.get("/v1/import")
    @app.get("/v1/retrieve")
    @app.get("/unclassified")
    async def _ok():
        return {"ok": True}

    return app


def _client(app) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t")


async def test_mcp_has_its_own_class_and_knob():
    app = _tiny_app(auth_per_minute=100, expensive_per_minute=100,
                    mcp_per_minute=1)
    async with _client(app) as client:
        h = {"Authorization": "Bearer tok_a"}
        assert (await client.get("/mcp", headers=h)).status_code == 200
        r = await client.get("/mcp", headers=h)
        assert r.status_code == 429
        assert r.headers.get("Retry-After") == "30"
        # The expensive class is untouched by MCP exhaustion — separate
        # limiters, separate buckets.
        assert (
            await client.get("/v1/retrieve", headers=h)
        ).status_code == 200


async def test_expensive_class_covers_the_three_rider_routes():
    app = _tiny_app(auth_per_minute=100, expensive_per_minute=1,
                    mcp_per_minute=100)
    async with _client(app) as client:
        for path in ("/v1/agent/messages", "/v1/export", "/v1/import"):
            h = {"Authorization": f"Bearer tok_{path}"}
            assert (await client.get(path, headers=h)).status_code == 200
            assert (await client.get(path, headers=h)).status_code == 429


async def test_distinct_tokens_get_distinct_buckets():
    app = _tiny_app(auth_per_minute=100, expensive_per_minute=100,
                    mcp_per_minute=1)
    async with _client(app) as client:
        assert (await client.get(
            "/mcp", headers={"Authorization": "Bearer tok_x"},
        )).status_code == 200
        assert (await client.get(
            "/mcp", headers={"Authorization": "Bearer tok_y"},
        )).status_code == 200  # different key, fresh bucket


async def test_unclassified_paths_stay_unlimited():
    app = _tiny_app(auth_per_minute=1, expensive_per_minute=1,
                    mcp_per_minute=1)
    async with _client(app) as client:
        h = {"Authorization": "Bearer tok_a"}
        for _ in range(5):
            assert (
                await client.get("/unclassified", headers=h)
            ).status_code == 200


async def test_two_arg_factory_call_keeps_mcp_unlimited():
    """Back-compat pin: the pre-existing call shape (no mcp_per_minute)
    must leave /mcp exactly as unlimited as it was before this slice."""
    app = _tiny_app(auth_per_minute=1, expensive_per_minute=1)
    async with _client(app) as client:
        h = {"Authorization": "Bearer tok_a"}
        for _ in range(5):
            assert (await client.get("/mcp", headers=h)).status_code == 200


# ---------------------------------------------------------------------------
# 2. memory_ingest ceiling (Q2=A)
# ---------------------------------------------------------------------------

async def test_ingest_refuses_over_ceiling_before_any_write(
    store, customer, semantic_encoder_stub, vector_store, monkeypatch,
):
    from crystal_cache.agent import mcp_server
    from crystal_cache.agent.tools.retrievers import set_tool_state
    from crystal_cache.config import settings

    monkeypatch.setattr(settings, "mcp_ingest_max_chars", 50)

    async def _explode(**kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("create_document_upload reached past the cap")

    monkeypatch.setattr(store, "create_document_upload", _explode)
    set_tool_state({
        "store": store,
        "encoder": semantic_encoder_stub,
        "vector_store": vector_store,
    })
    token = mcp_server._current_customer_id.set(customer.id)
    try:
        out = await mcp_server.memory_ingest(text="x" * 100, label="big")
    finally:
        mcp_server._current_customer_id.reset(token)

    assert out["code"] == "ingest_too_large"
    assert out["crystals_written"] == 0
    assert out["max_chars"] == 50
    assert out["received_chars"] == 100


async def test_ingest_cap_zero_disables(
    store, customer, semantic_encoder_stub, vector_store, monkeypatch,
):
    """0 = the codebase's 'no cap' idiom — the check must not fire."""
    from crystal_cache.agent import mcp_server
    from crystal_cache.agent.tools.retrievers import set_tool_state
    from crystal_cache.config import settings

    monkeypatch.setattr(settings, "mcp_ingest_max_chars", 0)

    sentinel = {}

    async def _capture(**kwargs):
        sentinel.update(kwargs)
        raise RuntimeError("stop here — cap check passed, that's the pin")

    monkeypatch.setattr(store, "create_document_upload", _capture)
    set_tool_state({
        "store": store,
        "encoder": semantic_encoder_stub,
        "vector_store": vector_store,
    })
    token = mcp_server._current_customer_id.set(customer.id)
    try:
        with pytest.raises(RuntimeError, match="stop here"):
            await mcp_server.memory_ingest(text="x" * 100, label="big")
    finally:
        mcp_server._current_customer_id.reset(token)
    assert sentinel["text"] == "x" * 100


# ---------------------------------------------------------------------------
# 3. memory_export pagination (Q3=A)
# ---------------------------------------------------------------------------

async def _seed_pairs(store, customer, encoder, vector_store, n) -> list[str]:
    keys = []
    for i in range(n):
        key = f"Export|Item{i:02d}"
        await store.add_pair_for_customer(
            customer_id=customer.id,
            prompt_text=key,
            answer_text=f"value {i}",
            pair_type="question_answer",
            encoder=encoder,
            vector_store=vector_store,
            vector_index=None,
            crystal_type="customer:legacy",
            source_kind="model_reasoning",
            answer_value=None,
        )
        keys.append(key)
    return keys


async def test_export_pages_are_disjoint_and_complete(
    store, customer, semantic_encoder_stub, vector_store,
):
    from crystal_cache.agent import mcp_server
    from crystal_cache.agent.tools.retrievers import set_tool_state

    keys = await _seed_pairs(
        store, customer, semantic_encoder_stub, vector_store, 5,
    )
    set_tool_state({"store": store})
    token = mcp_server._current_customer_id.set(customer.id)
    try:
        page1 = await mcp_server.memory_export(limit=2, offset=0)
        page2 = await mcp_server.memory_export(limit=2, offset=2)
        page3 = await mcp_server.memory_export(limit=2, offset=4)
    finally:
        mcp_server._current_customer_id.reset(token)

    assert page1["total_records"] == 5
    assert page1["record_count"] == 2 and page1["has_more"] is True
    assert page2["record_count"] == 2 and page2["has_more"] is True
    assert page3["record_count"] == 1 and page3["has_more"] is False

    walked = [r["key"] for p in (page1, page2, page3) for r in p["data"]]
    assert sorted(walked) == sorted(keys)  # complete, no dupes
    assert len(set(walked)) == 5  # disjoint pages
    # Records carry the crystal-joined fields.
    assert page1["data"][0]["crystal_type"] == "customer:legacy"
    assert page1["data"][0]["source_kind"] == "model_reasoning"
    assert page1["export_format"] == "jsonl"


async def test_export_limit_is_clamped(
    store, customer, semantic_encoder_stub, vector_store,
):
    from crystal_cache.agent import mcp_server
    from crystal_cache.agent.tools.retrievers import set_tool_state

    await _seed_pairs(store, customer, semantic_encoder_stub, vector_store, 2)
    set_tool_state({"store": store})
    token = mcp_server._current_customer_id.set(customer.id)
    try:
        out = await mcp_server.memory_export(limit=999999, offset=-5)
    finally:
        mcp_server._current_customer_id.reset(token)
    assert out["limit"] == 1000
    assert out["offset"] == 0
    assert out["record_count"] == 2


async def test_store_pagination_is_deterministic(
    store, customer, semantic_encoder_stub, vector_store,
):
    await _seed_pairs(store, customer, semantic_encoder_stub, vector_store, 4)
    total_a, page_a = await store.list_facts_for_customer_paginated(
        customer.id, limit=2, offset=0,
    )
    total_b, page_b = await store.list_facts_for_customer_paginated(
        customer.id, limit=2, offset=0,
    )
    assert total_a == total_b == 4
    assert [f.id for f in page_a] == [f.id for f in page_b]


# ---------------------------------------------------------------------------
# 4. Idle-stamp predicate (Q5=A)
# ---------------------------------------------------------------------------

def test_substantive_paths():
    assert is_substantive_path("/v1/retrieve")
    assert is_substantive_path("/v1/agent/messages")
    assert is_substantive_path("/mcp")
    assert is_substantive_path("/mcp/")
    assert not is_substantive_path("/admin/api/cognition/runs")
    assert not is_substantive_path("/health")
    assert not is_substantive_path("/")
