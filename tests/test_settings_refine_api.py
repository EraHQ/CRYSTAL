"""Console Refine PR 9 (docs/CONSOLE_UX_AUDIT.md 5.8): what the Settings
Team tab needs from the server.

- PATCH /admin/api/customers/{id}/operators/{op} takes status
  active | suspended: Remove is a suspension (the row survives, the
  list shows it), Undo reactivates; anything else is a 400; a foreign
  operator id reads as missing.
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
async def test_operator_status_round_trips_through_the_admin_patch(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    op, _key = await store.create_operator(customer.id, "Dana", role="operator")
    other = await store.create_customer(provider="anthropic", model_id="claude-sonnet-4-5-20250929", api_key_ref="sk-other")
    app = _app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    base = f"/admin/api/customers/{customer.id}/operators"
    async with _client(app) as cl:
        r = await cl.patch(f"{base}/{op.id}", json={"status": "suspended"})
        assert r.status_code == 200, r.text[:300]
        assert r.json()["status"] == "suspended" and r.json()["display_name"] == "Dana"
        r = await cl.get(base)
        row = next(x for x in r.json()["operators"] if x["id"] == op.id)
        assert row["status"] == "suspended"
        # Undo.
        r = await cl.patch(f"{base}/{op.id}", json={"status": "active"})
        assert r.status_code == 200 and r.json()["status"] == "active"
        r = await cl.get(base)
        assert next(x for x in r.json()["operators"] if x["id"] == op.id)["status"] == "active"
        # Only the two states exist; a rename without status leaves it alone.
        r = await cl.patch(f"{base}/{op.id}", json={"status": "deleted"})
        assert r.status_code == 400
        r = await cl.patch(f"{base}/{op.id}", json={"display_name": "Dana R."})
        assert r.status_code == 200 and r.json()["status"] == "active" and r.json()["display_name"] == "Dana R."
        # A foreign tenant's path never finds it.
        r = await cl.patch(f"/admin/api/customers/{other.id}/operators/{op.id}", json={"status": "suspended"})
        assert r.status_code == 404
