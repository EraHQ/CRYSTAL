"""Console Refine PR 6 (docs/CONSOLE_UX_AUDIT.md 5.4 item 5): every gap
row has a verb the server honours.

- /answer writes the typed text as a fact under the gap's key, born
  operator_stated, and marks the gap filled by that crystal.
- /dismiss closes an open gap; /reopen is its Undo. Nothing is deleted.
- A foreign tenant sees a uniform 404 on all three.
"""
from __future__ import annotations

import pytest


def _app(store, encoder, vector_store, fact_vector_store):
    try:
        from tests.test_endpoint_smoke import _build_app
    except ModuleNotFoundError:
        from test_endpoint_smoke import _build_app
    return _build_app(store, encoder, vector_store, fact_vector_store)


def _client(app):
    from httpx import ASGITransport, AsyncClient

    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t")


@pytest.mark.asyncio
async def test_answer_fills_the_gap_with_an_operator_stated_fact(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    gap = await store.create_knowledge_gap(
        customer.id, domain="Ops", subject="Deploy window", missing="When do deploys run?",
        full_key="Ops|Deploy window|Schedule",
    )
    app = _app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    h = {"Authorization": f"Bearer {customer.api_key}"}
    async with _client(app) as c:
        r = await c.post(f"/v1/gaps/{gap.id}/answer", headers=h, json={"text": "   "})
        assert r.status_code in (400, 422)
        r = await c.post(f"/v1/gaps/{gap.id}/answer", headers=h, json={"text": "Deploys run Tuesdays at 10:00 UTC."})
        assert r.status_code == 200, r.text[:300]
        body = r.json()
        assert body["status"] == "filled" and body["gap_id"] == gap.id
        crystal = await store.get_crystal(body["crystal_id"])
        assert crystal is not None and crystal.customer_id == customer.id
        facts = await store.list_facts_for_crystal(crystal.id)
        assert [f.claim_text for f in facts] == ["Deploys run Tuesdays at 10:00 UTC."]
        assert facts[0].prompt_text == "Ops|Deploy window|Schedule"
        assert facts[0].source_kind == "operator_stated"
        fresh = await store.get_knowledge_gap(gap.id)
        assert fresh.status == "filled" and fresh.filled_by_crystal_id == crystal.id
        assert fresh.resolved_at is not None
        # A filled gap cannot be answered or dismissed again.
        r = await c.post(f"/v1/gaps/{gap.id}/answer", headers=h, json={"text": "again"})
        assert r.status_code == 409
        r = await c.post(f"/v1/gaps/{gap.id}/dismiss", headers=h)
        assert r.status_code == 409


@pytest.mark.asyncio
async def test_dismiss_and_reopen_and_tenant_isolation(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    gap = await store.create_knowledge_gap(
        customer.id, domain=None, subject="Pricing", missing="What is the enterprise price?",
    )
    other = await store.create_customer(
        provider="anthropic", model_id="claude-sonnet-4-5-20250929", api_key_ref="sk-test-other",
    )
    app = _app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    h = {"Authorization": f"Bearer {customer.api_key}"}
    foreign = {"Authorization": f"Bearer {other.api_key}"}
    async with _client(app) as c:
        for verb in ("answer", "dismiss", "reopen"):
            r = await c.post(f"/v1/gaps/{gap.id}/{verb}", headers=foreign, json={"text": "x"})
            assert r.status_code == 404, verb
        r = await c.post(f"/v1/gaps/{gap.id}/reopen", headers=h)
        assert r.status_code == 409  # only a dismissed gap reopens
        r = await c.post(f"/v1/gaps/{gap.id}/dismiss", headers=h)
        assert r.status_code == 200 and r.json()["status"] == "closed"
        fresh = await store.get_knowledge_gap(gap.id)
        assert fresh.status == "closed" and fresh.resolved_at is not None
        r = await c.post(f"/v1/gaps/{gap.id}/reopen", headers=h)
        assert r.status_code == 200 and r.json()["status"] == "open"
        fresh = await store.get_knowledge_gap(gap.id)
        assert fresh.status == "open" and fresh.resolved_at is None
        # The list the console reads still carries it as open.
        r = await c.get("/admin/api/knowledge-gaps", params={"customer_id": customer.id})
        assert r.status_code == 200
        assert [g["status"] for g in r.json()["gaps"] if g["id"] == gap.id] == ["open"]


@pytest.mark.asyncio
async def test_runs_carry_a_capacity_share_instead_of_dollars_for_tenants(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    """5.4 items 3/4: a run's cost as a percent of the tenant's daily
    allowance (free tier: $0.50/day, so $0.007 is 1.4%); None without a
    tier. The dollar figure stays in the payload for the admin view; the
    console hides it from tenants."""
    await store.upsert_cognition_run(
        "env_share", customer.id, status="complete", trigger_type="research",
        goal_title="Find the thing",
        summary={"id": "env_share", "status": "complete", "cost_usd": 0.007, "tokens_used": 1200},
        detail={"id": "env_share", "status": "complete", "cost_usd": 0.007, "steps": {}},
    )
    app = _app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    async with _client(app) as c:
        r = await c.get("/admin/api/cognition/environments", params={"customer_id": customer.id})
        assert r.status_code == 200, r.text[:300]
        row = next(x for x in r.json()["environments"] if x["id"] == "env_share")
        assert row["capacity_share_pct"] is None  # no tier yet
        await store.set_customer_subscription(customer.id, "free", None)
        r = await c.get("/admin/api/cognition/environments", params={"customer_id": customer.id})
        row = next(x for x in r.json()["environments"] if x["id"] == "env_share")
        assert row["capacity_share_pct"] == 1.4
        r = await c.get("/admin/api/cognition/environments/env_share")
        assert r.status_code == 200 and r.json()["capacity_share_pct"] == 1.4
