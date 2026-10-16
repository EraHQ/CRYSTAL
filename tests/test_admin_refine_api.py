"""Console Refine PR 13 (docs/CONSOLE_UX_AUDIT.md 5.10-5.11): what the
admin pages need from the server.

- POST /admin/api/metacognition/substrate-observations/{id}/restore is
  the Undo of dismiss: the observation renders on the review surface
  again; a missing id is a 404.
- POST /admin/api/agents/tasks/{id}/retry re-enqueues a FAILED daemon
  task as a fresh queued row with parent_task_id lineage; a queued task
  409s; a foreign customer's task reads as missing.
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
async def test_restore_is_the_undo_of_dismiss(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    critique = await store.create_critique(
        customer.id, critic_role="structural", critic_model="scan",
        observations=[], summary_text="s", total_action_items=1,
    )
    item = await store.create_action_item(
        critique.id, customer.id, "substrate_observation",
        content={"subsystem": "ingestion", "complaint": "blob", "severity": "medium"},
        critic_confidence=1.0,
    )
    await store.update_action_item_status(item.id, "deferred")
    app = _app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    base = "/admin/api/metacognition/substrate-observations"
    async with _client(app) as cl:
        r = await cl.get(base, params={"customer_id": customer.id})
        assert r.status_code == 200 and r.json()["total"] == 1
        r = await cl.post(f"{base}/{item.id}/dismiss", json={})
        assert r.status_code == 200 and r.json()["status"] == "dropped"
        r = await cl.get(base, params={"customer_id": customer.id})
        assert r.json()["total"] == 0
        r = await cl.post(f"{base}/{item.id}/restore", json={})
        assert r.status_code == 200 and r.json()["status"] == "deferred"
        r = await cl.get(base, params={"customer_id": customer.id})
        assert r.json()["total"] == 1
        r = await cl.post(f"{base}/aitem_missing/restore", json={})
        assert r.status_code == 404


@pytest.mark.asyncio
async def test_retry_requeues_only_a_failed_task(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    other = await store.create_customer(
        provider="anthropic", model_id="claude-sonnet-4-5-20250929", api_key_ref="sk-other",
    )
    failed = await store.create_agent_task(
        customer.id, project_dir="/repo", task="fix the build", branch="main",
    )
    claimed = await store.claim_next_agent_task()
    assert claimed is not None and claimed["id"] == failed["id"]
    await store.finish_agent_task(failed["id"], status="failed", error="boom")
    queued = await store.create_agent_task(customer.id, project_dir="/repo", task="later")
    app = _app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    async with _client(app) as cl:
        r = await cl.post(
            f"/admin/api/agents/tasks/{failed['id']}/retry", params={"customer_id": customer.id},
        )
        assert r.status_code == 200, r.text[:300]
        new = r.json()["task"]
        assert new["status"] == "queued" and new["task"] == "fix the build"
        assert new["branch"] == "main" and new["parent_task_id"] == failed["id"]
        assert new["source"] == "retry" and new["id"] != failed["id"]
        r = await cl.get("/admin/api/agents/tasks", params={"customer_id": customer.id})
        assert {t["id"] for t in r.json()["tasks"]} == {failed["id"], queued["id"], new["id"]}
        r = await cl.post(
            f"/admin/api/agents/tasks/{queued['id']}/retry", params={"customer_id": customer.id},
        )
        assert r.status_code == 409
        r = await cl.post(
            f"/admin/api/agents/tasks/{failed['id']}/retry", params={"customer_id": other.id},
        )
        assert r.status_code == 404
