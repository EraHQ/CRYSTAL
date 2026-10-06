"""RC-07 (2026-10-05): the fact-level export round-trips.

Before: export paged at 1,000 while import refused over 500; records
carried the crystal's source_kind, not the fact's; origin, gate and
scope were dropped, so an import turned a pending assumption into an
approved fact and derived facts into billable ones. Now both exports
carry origin, recall_gated, scope and the fact's own source_kind, both
imports restore them, and the MCP page cap is the import cap.

The pin builds a bank with a direct fact, a background-derived fact and a
gated assumption, exports through the real HTTP route, imports into a
fresh tenant through the real HTTP route (key_is_path keeps keys
verbatim, so no model call), and compares gate, origin and billable
count.
"""
from types import SimpleNamespace

import pytest

from crystal_cache.endpoints import sdk as sdk_mod
from crystal_cache.ingress.schema import ImportRequest


async def _seed(store, customer, encoder, vector_store):
    await store.add_pair_for_customer(
        customer_id=customer.id, prompt_text="Billing|terms", answer_text="Net 30",
        encoder=encoder, vector_store=vector_store, origin="direct",
    )
    await store.add_pair_for_customer(
        customer_id=customer.id, prompt_text="Billing|derived", answer_text="Invoices monthly",
        encoder=encoder, vector_store=vector_store, origin="background_worker",
        source_kind="agent_inferred",
    )
    gated = await store.add_pair_for_customer(
        customer_id=customer.id, prompt_text="Assumptions|due", answer_text="Due on the 31st",
        encoder=encoder, vector_store=vector_store, origin="assumptions", recall_gated=True,
    )
    return gated


def _req():
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()), headers={})


@pytest.mark.asyncio
async def test_http_export_imports_into_a_fresh_tenant_as_it_was(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store
):
    await _seed(store, customer, semantic_encoder_stub, vector_store)
    billable_before = await store.count_billable_facts(customer.id)
    assert billable_before == 1  # only the direct fact counts (Q5=B)

    export = await sdk_mod.sdk_export(customer, store)
    records = export.data
    assert len(records) == 3
    by_key = {r["key"]: r for r in records}
    assert by_key["Assumptions|due"]["recall_gated"] is True
    assert by_key["Assumptions|due"]["origin"] == "assumptions"
    assert by_key["Billing|derived"]["origin"] == "background_worker"
    assert by_key["Billing|derived"]["source_kind"] == "agent_inferred"
    assert by_key["Billing|terms"]["origin"] == "direct"

    other = await store.create_customer(provider="anthropic", model_id="m", api_key_ref="")
    req = _req()
    req.app.state.prompt_encoder = semantic_encoder_stub
    req.app.state.vector_store = vector_store
    req.app.state.fact_vector_store = fact_vector_store
    req.app.state.vector_index = None
    out = await sdk_mod.sdk_import(ImportRequest(records=records), req, other, store)
    assert out.records_processed == 3 and out.errors == 0

    # Same shape on the other side: gate, origin, and the billable count.
    assert await store.count_billable_facts(other.id) == billable_before
    crystals = await store.list_crystals_for_customer(other.id)
    by_origin = {}
    for c in crystals:
        by_origin.setdefault(c.origin, []).append(c)
    assert len(by_origin.get("assumptions", [])) == 1
    assert by_origin["assumptions"][0].recall_gated is True
    assert len(by_origin.get("background_worker", [])) == 1
    assert len(by_origin.get("direct", [])) == 1


@pytest.mark.asyncio
async def test_mcp_export_page_cap_matches_the_import_cap_and_carries_the_fields(
    store, customer, semantic_encoder_stub, vector_store, monkeypatch
):
    from crystal_cache.agent import mcp_server

    await _seed(store, customer, semantic_encoder_stub, vector_store)
    monkeypatch.setattr(mcp_server, "_get_state", lambda: {"store": store})
    fn = getattr(mcp_server.memory_export, "fn", mcp_server.memory_export)
    token = mcp_server._current_customer_id.set(customer.id)
    try:
        out = await fn(limit=10_000)
    finally:
        mcp_server._current_customer_id.reset(token)
    assert out["limit"] == mcp_server.MEMORY_IMPORT_MAX_RECORDS
    rec = next(r for r in out["data"] if r["key"] == "Assumptions|due")
    assert rec["recall_gated"] is True and rec["origin"] == "assumptions"

    import inspect
    assert "default 500, the import cap" in inspect.getsource(mcp_server)
