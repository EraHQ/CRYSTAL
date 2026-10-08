"""RC-16 / B6 (2026-10-08): an explicit empty approve selection is refused.

The approve route read `body.get("items") or doc.extracted_items`, so a
reviewer who unticked everything and pressed approve crystallized every
item. An explicit empty selection is now 400; omitting the keys still
means "as extracted" for callers that never edited.
"""
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from crystal_cache.endpoints import documents as docs_mod
from crystal_cache.infrastructure.schema import DocumentUploadRow


class _Req:
    def __init__(self, body):
        self._body = body
        self.headers = {"content-type": "application/json"}
        self.app = SimpleNamespace(state=SimpleNamespace())

    async def json(self):
        return self._body


@pytest.mark.asyncio
async def test_explicit_empty_selection_is_400_and_changes_nothing(store, customer, monkeypatch):
    async with store.session() as s:
        s.add(DocumentUploadRow(id="doc_e", customer_id=customer.id, label="d", text="t",
                                status="review", extracted_items=[{"key": "k", "value": "v"}],
                                content_chunks=[]))
    from crystal_cache.config import settings
    monkeypatch.setattr(settings, "ingest_mode", "worker")
    with pytest.raises(HTTPException) as e:
        await docs_mod.sdk_approve_document("doc_e", _Req({"items": [], "content_chunks": []}), customer, store)
    assert e.value.status_code == 400
    assert "selected" in e.value.detail.lower()
    async with store.session() as s:
        assert (await s.get(DocumentUploadRow, "doc_e")).status == "review"


@pytest.mark.asyncio
async def test_omitted_keys_still_mean_as_extracted(store, customer, monkeypatch):
    async with store.session() as s:
        s.add(DocumentUploadRow(id="doc_o", customer_id=customer.id, label="d", text="t",
                                status="review", extracted_items=[{"key": "k", "value": "v"}],
                                content_chunks=[]))
    from crystal_cache.config import settings
    monkeypatch.setattr(settings, "ingest_mode", "worker")
    out = await docs_mod.sdk_approve_document("doc_o", _Req({}), customer, store)
    assert getattr(out, "status_code", 202) == 202
    async with store.session() as s:
        assert (await s.get(DocumentUploadRow, "doc_o")).status == "approved"
