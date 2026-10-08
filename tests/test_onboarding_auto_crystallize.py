"""RC-17 (2026-10-08): an upload with auto_crystallize becomes memory.

Onboarding's "About you" upload sent auto_crystallize=true; the route
accepted the flag and dropped it, so the document sat waiting for an
approval the wizard never mentioned. The flag is stored
(document_uploads.auto_approve, migration f6c8d0e2a4b6) and extraction
sends the row straight to 'approved', where the write-leg claim picks it
up. The pin drives the real JSON route and the real review-ready
transition, then confirms the row is claimable for the write leg.
"""
from types import SimpleNamespace

import pytest

from crystal_cache.endpoints import documents as docs_mod
from crystal_cache.infrastructure.schema import DocumentUploadRow
from crystal_cache.ingress.schema import DocumentUploadRequest


@pytest.mark.asyncio
async def test_auto_crystallize_upload_is_approved_after_extraction(store, customer):
    body = DocumentUploadRequest(text="I run a two-person consultancy in Raleigh.", label="About you",
                                 auto_crystallize=True)
    out = await docs_mod.sdk_upload_document(body, (customer, None), store)
    import json

    doc_id = json.loads(out.body)["id"]
    async with store.session() as s:
        row = await s.get(DocumentUploadRow, doc_id)
        assert row.status == "pending" and row.auto_approve is True

    # The worker's extraction step ends here; an auto-approve row goes to
    # 'approved' instead of 'review'.
    await store.mark_document_review_ready(
        doc_id, detected_type="customer:legacy", content_chunks=[{"index": 0, "text": "x", "label": "Chunk 0"}],
        extracted_items=[{"key": "Business|size", "value": "two-person consultancy", "type": "fact"}],
        items_extracted_count=1,
    )
    async with store.session() as s:
        assert (await s.get(DocumentUploadRow, doc_id)).status == "approved"

    # And the write leg can claim it without anyone pressing Approve.
    claimed = await store.claim_approved_documents_batch(limit=5)
    assert [d.id for d in claimed] == [doc_id]


@pytest.mark.asyncio
async def test_plain_upload_still_waits_for_review(store, customer):
    body = DocumentUploadRequest(text="Plain upload.", label="notes")
    out = await docs_mod.sdk_upload_document(body, (customer, None), store)
    import json

    doc_id = json.loads(out.body)["id"]
    await store.mark_document_review_ready(
        doc_id, detected_type="customer:legacy", content_chunks=[], extracted_items=[], items_extracted_count=0,
    )
    async with store.session() as s:
        row = await s.get(DocumentUploadRow, doc_id)
        assert row.status == "review" and not row.auto_approve
