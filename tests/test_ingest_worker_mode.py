"""L7a gate 5 (ratified 2026-08-29: Q1=A, Q2=A, Q3=A): ingest routed to
the worker under CC_INGEST_MODE=worker.

Before: POST /v1/documents/{id}/approve ran the whole write leg inside the
request on crystal-api, and POST /crystallize ran extraction there too;
the worker's poll loop only ever claimed 'pending' rows. Now: with
CC_INGEST_MODE=worker the approve handler saves the edits, marks the row
'approved' (new status, Literal addition) and returns 202; the worker
claims 'approved' rows exactly the way it claims 'pending' ones and runs
the write leg. /crystallize and /crystallize-all return 202 in worker
mode because their rows are already 'pending'. The write leg is ONE
workflow function, `write_approved_document`, that the inline handler and
the worker both call. GET /v1/documents/{id} is the poll target.
CC_INGEST_MODE=inline (the default) is byte-for-byte the old behaviour.

Pinned here:
  1. Store: mark_document_approved saves edits and sets 'approved';
     claim_approved_documents_batch claims only 'approved' rows and marks
     them 'crystallizing'; claim_pending_documents_batch ignores them.
  2. Workflow: write_approved_document writes the crystals, marks the row
     crystallized with the counts, and returns the endpoint's response
     dict; a pipeline failure marks the row 'error' and re-raises.
  3. Worker: one poll_once processes an 'approved' row (write leg) and
     claims a 'pending' row (extraction leg) in the same pass.
  4. Endpoints: worker mode -> approve returns 202/'approved' and writes
     nothing; crystallize returns 202 and leaves the row 'pending'.
     Inline mode -> approve returns 200/'crystallized' with the crystals
     written. GET /v1/documents/{id} returns the status envelope.
  5. Knob default is inline.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from crystal_cache.config import Settings
from crystal_cache.endpoints.documents import (
    sdk_approve_document,
    sdk_crystallize_document,
    sdk_get_document,
)
from crystal_cache.workers import crystallization as wk


ITEMS = [
    {"key": "batch call count", "value": "one per approve", "type": "fact",
     "sparse_key": "Docs|Ingest|batch"},
    {"key": "fallback", "value": "per-text encodes", "type": "definition",
     "sparse_key": "Docs|Ingest|fallback"},
]
CHUNKS = [
    {"index": 0, "label": "Section 0", "text": "Chunk zero body text.",
     "locator": "Section 0", "subject": "Ingest", "domain": "Docs", "doc_type": "general"},
]


async def _doc_in_review(store, customer):
    doc = await store.create_document_upload(customer.id, "notes.txt", "raw text")
    await store.mark_document_review_ready(
        doc.id, detected_type="general",
        content_chunks=[dict(c) for c in CHUNKS],
        extracted_items=[dict(i) for i in ITEMS],
        items_extracted_count=len(ITEMS),
    )
    return doc.id


def _fake_request(encoder, vector_store, fact_vector_store):
    class _Req:
        headers = {"content-type": "application/json"}
        app = SimpleNamespace(state=SimpleNamespace(
            prompt_encoder=encoder, vector_store=vector_store,
            fact_vector_store=fact_vector_store, vector_index=None,
        ))
        async def json(self):
            return {}
    return _Req()


def _body(resp) -> dict:
    return json.loads(resp.body)


# ---------------------------------------------------------------------------
# 1. Store primitives
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mark_approved_then_claim_approved_not_pending(store, customer):
    doc_id = await _doc_in_review(store, customer)
    pending = await store.create_document_upload(customer.id, "later.txt", "more raw text")

    edited = [dict(ITEMS[0], value="edited value")]
    await store.mark_document_approved(doc_id, items=edited, content_chunks=[dict(c) for c in CHUNKS])
    row = await store.get_document_upload(doc_id, customer.id)
    assert row.status == "approved"
    assert row.extracted_items[0]["value"] == "edited value"

    # The pending claim does not see it ...
    claimed_pending = await store.claim_pending_documents_batch(limit=10)
    assert [d.id for d in claimed_pending] == [pending.id]
    # ... the approved claim does, and marks it crystallizing.
    claimed = await store.claim_approved_documents_batch(limit=10)
    assert [d.id for d in claimed] == [doc_id]
    assert (await store.get_document_upload(doc_id, customer.id)).status == "crystallizing"
    assert await store.claim_approved_documents_batch(limit=10) == []      # claimed once


# ---------------------------------------------------------------------------
# 2. The write-leg workflow
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_approved_document_writes_and_marks(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    doc_id = await _doc_in_review(store, customer)
    await store.mark_document_approved(doc_id, items=[dict(i) for i in ITEMS],
                                       content_chunks=[dict(c) for c in CHUNKS])
    await store.claim_approved_documents_batch(limit=1)

    result = await wk.write_approved_document(
        store=store, encoder=semantic_encoder_stub, vector_store=vector_store,
        fact_vector_store=fact_vector_store, document_id=doc_id, customer_id=customer.id,
    )
    assert result["status"] == "crystallized"
    assert result["crystals_written"] == len(ITEMS) + 1          # items + the file crystal
    assert result["errors"] == 0
    row = await store.get_document_upload(doc_id, customer.id)
    assert row.status == "crystallized"
    assert row.crystals_written == len(ITEMS) + 1
    assert row.crystallized_at is not None
    assert len(await store.list_crystals_for_customer(customer.id)) >= 2


@pytest.mark.asyncio
async def test_write_approved_document_failure_marks_error(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store, monkeypatch,
):
    from crystal_cache.ingestion.document_pipeline import DocumentPipeline
    async def _boom(self, **kw):
        raise RuntimeError("pipeline boom")
    monkeypatch.setattr(DocumentPipeline, "approve_and_crystallize", _boom)

    doc_id = await _doc_in_review(store, customer)
    await store.mark_document_approved(doc_id, items=[dict(i) for i in ITEMS], content_chunks=[])
    await store.claim_approved_documents_batch(limit=1)
    with pytest.raises(RuntimeError, match="pipeline boom"):
        await wk.write_approved_document(
            store=store, encoder=semantic_encoder_stub, vector_store=vector_store,
            fact_vector_store=fact_vector_store, document_id=doc_id, customer_id=customer.id,
        )
    assert (await store.get_document_upload(doc_id, customer.id)).status == "error"


# ---------------------------------------------------------------------------
# 3. One worker poll handles both legs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_poll_once_writes_approved_and_claims_pending(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store, monkeypatch,
):
    import asyncio
    extracted: list[str] = []
    async def _fake_extract(*, store, encoder, vector_store, document_id, **kw):
        extracted.append(document_id)
    monkeypatch.setattr(wk, "crystallize_document", _fake_extract)

    approved_id = await _doc_in_review(store, customer)
    await store.mark_document_approved(approved_id, items=[dict(i) for i in ITEMS],
                                       content_chunks=[dict(c) for c in CHUNKS])
    pending = await store.create_document_upload(customer.id, "later.txt", "more raw text")

    n = await wk.poll_once(
        store=store, encoder=semantic_encoder_stub, vector_store=vector_store,
        fact_vector_store=fact_vector_store, vector_index=None,
        sem=asyncio.Semaphore(1), concurrency=1,
    )
    assert n == 2
    assert (await store.get_document_upload(approved_id, customer.id)).status == "crystallized"
    assert extracted == [pending.id]


# ---------------------------------------------------------------------------
# 4. Endpoint dispatch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_approve_in_worker_mode_queues_without_writing(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store, monkeypatch,
):
    monkeypatch.setattr("crystal_cache.config.get_settings", lambda: Settings(ingest_mode="worker"))
    doc_id = await _doc_in_review(store, customer)
    req = _fake_request(semantic_encoder_stub, vector_store, fact_vector_store)

    resp = await sdk_approve_document(doc_id, req, customer, store)
    assert resp.status_code == 202
    assert _body(resp)["status"] == "approved" and _body(resp)["queued"] is True
    assert (await store.get_document_upload(doc_id, customer.id)).status == "approved"
    assert await store.list_crystals_for_customer(customer.id) == []

    # /crystallize on a pending row: 202, row untouched
    pending = await store.create_document_upload(customer.id, "later.txt", "more raw text")
    resp = await sdk_crystallize_document(pending.id, req, customer, store)
    assert resp.status_code == 202
    assert _body(resp)["status"] == "pending" and _body(resp)["queued"] is True
    assert (await store.get_document_upload(pending.id, customer.id)).status == "pending"

    # the poll target
    resp = await sdk_get_document(doc_id, customer, store)
    assert resp.status_code == 200
    assert _body(resp)["id"] == doc_id and _body(resp)["status"] == "approved"


@pytest.mark.asyncio
async def test_approve_in_inline_mode_writes_in_the_request(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store, monkeypatch,
):
    monkeypatch.setattr("crystal_cache.config.get_settings", lambda: Settings(ingest_mode="inline"))
    doc_id = await _doc_in_review(store, customer)
    req = _fake_request(semantic_encoder_stub, vector_store, fact_vector_store)

    resp = await sdk_approve_document(doc_id, req, customer, store)
    assert resp.status_code == 200
    body = _body(resp)
    assert body["status"] == "crystallized" and body["crystals_written"] == len(ITEMS) + 1
    assert (await store.get_document_upload(doc_id, customer.id)).status == "crystallized"


def test_ingest_mode_default_is_inline():
    assert Settings.model_fields["ingest_mode"].default == "inline"
