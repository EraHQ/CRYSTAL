"""Console Refine PR 5 (docs/CONSOLE_UX_AUDIT.md 5.3): the two backend
seams the Knowledge page needs.

- 5.3 item 7: a watch whose GitHub token was revoked is repaired in
  place with PATCH {token}; the new credential resolves, the standing
  error clears and the clock is aged so the next poll retries now.
- 5.3 item 4: "View {n} crystals" on a document card lists exactly the
  crystals born from that document (crystals?document_id=).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from crystal_cache.infrastructure.schema import CrystalRow


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
async def test_watch_token_is_replaced_in_place(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    from crystal_cache.ingestion.source_handlers import encrypt_watch_token, resolve_watch_token

    enc = await encrypt_watch_token(store, customer.id, "ghp_old")
    w = await store.create_source_watch(
        customer.id, scheme="git", source_name="repo-a",
        config={"repo": "o/r", "branch": "main"}, encrypted_token=enc,
    )
    await store.update_source_watch_state(
        w.id, customer.id, last_error="The sync failed. (ref abc)",
        checked_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    app = _app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    async with _client(app) as c:
        r = await c.patch(f"/admin/api/watches/{w.id}", params={"customer_id": customer.id},
                          json={"token": "ghp_new"})
        assert r.status_code == 200, r.text[:300]
        body = r.json()
        assert body["has_token"] is True and "token" not in body and "encrypted_token" not in body
        assert body["last_error"] is None
        # A non-string token is refused; an empty one is not a replacement.
        r = await c.patch(f"/admin/api/watches/{w.id}", params={"customer_id": customer.id},
                          json={"token": 42})
        assert r.status_code in (400, 422)
        r = await c.patch(f"/admin/api/watches/{w.id}", params={"customer_id": customer.id},
                          json={"token": "", "status": "paused"})
        assert r.status_code == 200 and r.json()["status"] == "paused"
    fresh = await store.get_source_watch(w.id, customer.id)
    assert await resolve_watch_token(store, fresh) == "ghp_new"
    assert fresh.last_error is None
    assert fresh.last_checked_at.year == 1970  # due immediately


@pytest.mark.asyncio
async def test_crystals_filter_by_source_document(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    when = datetime(2026, 1, 1, tzinfo=timezone.utc)
    async with store.session() as s:
        for i in range(6):
            s.add(CrystalRow(
                id=f"cr_doc_{i}", customer_id=customer.id, summary_vector=[],
                summary_text=f"crystal {i}", quality_tier="neutral", crystal_type="customer:legacy",
                source_document_id=("doc_A" if i < 4 else "doc_B"),
                created_at=when, last_activity=when,
            ))
    app = _app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    base = f"/admin/api/customers/{customer.id}/crystals"
    async with _client(app) as c:
        r = await c.get(base, params={"document_id": "doc_A"})
        assert r.status_code == 200, r.text[:300]
        assert r.json()["total"] == 4
        assert {x["id"] for x in r.json()["crystals"]} == {f"cr_doc_{i}" for i in range(4)}
        # Composes with the search term; the overview still counts the whole bank.
        r = await c.get(base, params={"document_id": "doc_B", "q": "crystal 5"})
        assert [x["id"] for x in r.json()["crystals"]] == ["cr_doc_5"]
        assert r.json()["bank"]["crystals"] == 6
        r = await c.get(base, params={"document_id": "doc_none"})
        assert r.json()["total"] == 0
