"""RC-04 (2026-10-05): ingest identity and approve are idempotent.

Three defects, three pins:
- E-B3: unlabeled ingests shared the label "Untitled"; the pipeline keys
  source identity on the label, so the second replaced the first. An
  unlabeled ingest now gets a content-derived label.
- D9: re-ingest deleted the old crystal BEFORE the new one encoded, so a
  failed encode lost both. Deletion now follows a successful write.
- E-S9: approve was not compare-and-set; two approves encoded twice.
"""
import pytest

from crystal_cache.infrastructure.schema import CrystalRow, DocumentUploadRow, FactRow


# ----------------------------------------------------------------------------
# E-B3: two unlabeled ingests are two sources
# ----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unlabeled_ingests_get_distinct_content_labels(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store, monkeypatch
):
    from crystal_cache.agent import mcp_server

    seen: list[str] = []
    real_create = store.create_document_upload

    async def _spy(customer_id, label, text, **kw):
        seen.append(label)
        return await real_create(customer_id=customer_id, label=label, text=text, **kw)

    monkeypatch.setattr(store, "create_document_upload", _spy)
    # The same keys the real MCP state carries (app.py), from the real fixtures.
    monkeypatch.setattr(mcp_server, "_get_state", lambda: {
        "store": store, "encoder": semantic_encoder_stub, "vector_store": vector_store,
        "vector_index": None, "fact_vector_store": fact_vector_store,
    })

    async def _open(**kw):
        return None

    monkeypatch.setattr(mcp_server, "_write_admission_block", _open)

    class _Halt(Exception):
        pass

    async def _halt(*a, **k):
        raise _Halt()

    from crystal_cache.workers import crystallization
    monkeypatch.setattr(crystallization, "crystallize_document", _halt)

    fn = getattr(mcp_server.memory_ingest, "fn", mcp_server.memory_ingest)
    token = mcp_server._current_customer_id.set(customer.id)
    try:
        for text in ("first document", "second document", "first document"):
            with pytest.raises(_Halt):
                await fn(text=text)
    finally:
        mcp_server._current_customer_id.reset(token)
    assert len(seen) == 3
    assert seen[0] != seen[1]          # distinct texts, distinct sources
    assert seen[0] == seen[2]          # same text, same source (dedup as unchanged)
    assert all(s.startswith("Untitled ") and s != "Untitled" for s in seen)


# ----------------------------------------------------------------------------
# D9: a failed re-encode keeps the previous version
# ----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_failed_reencode_keeps_the_old_version(store, customer, semantic_encoder_stub,
                                                      vector_store, fact_vector_store, monkeypatch):
    from crystal_cache.ingestion.document_pipeline import DocumentPipeline

    uri = "upload://doc_v1"
    async with store.session() as s:
        s.add(CrystalRow(id="cr_old", customer_id=customer.id, summary_vector=[],
                         summary_text="old version", source_uri=uri, source_path="notes",
                         content_hash="hash_old"))
        await s.flush()
        s.add(FactRow(id="f_old", crystal_id="cr_old", claim_text="old", prompt_text="k",
                      pair_type="content_chunk", vector=[1.0, 0.0]))
        s.add(DocumentUploadRow(id="doc_v1", customer_id=customer.id, label="notes",
                                text="new version text", status="review", source_uri=uri,
                                extracted_items=[],
                                content_chunks=[{"index": 0, "text": "new version text",
                                                 "label": "Chunk 0"}]))

    pipeline = DocumentPipeline(store=store, encoder=semantic_encoder_stub,
                                vector_store=vector_store, fact_vector_store=fact_vector_store)

    async def _boom(*a, **k):
        raise RuntimeError("encode exploded")

    monkeypatch.setattr(store, "add_pair_to_crystal", _boom)
    try:
        await pipeline.approve_and_crystallize(customer.id, "doc_v1", items=[],
                                               content_chunks=[{"index": 0, "text": "new version text",
                                                                "label": "Chunk 0"}])
    except Exception:  # noqa: BLE001  (the pipeline may or may not swallow)
        pass
    # The old version is still there: nothing was written, so nothing was replaced.
    assert await store.get_crystal("cr_old") is not None
    facts = await store.list_facts_for_crystal("cr_old", include_deactivated=True)
    assert {f.id for f in facts} == {"f_old"}


# ----------------------------------------------------------------------------
# E-S9: approve is compare-and-set
# ----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_two_approves_transition_once(store, customer):
    async with store.session() as s:
        s.add(DocumentUploadRow(id="doc_cas", customer_id=customer.id, label="d",
                                text="t", status="review"))
    first = await store.mark_document_approved(document_id="doc_cas", items=[], content_chunks=[])
    second = await store.mark_document_approved(document_id="doc_cas", items=[], content_chunks=[])
    assert (first, second) == (True, False)

    async with store.session() as s:
        s.add(DocumentUploadRow(id="doc_cas2", customer_id=customer.id, label="d",
                                text="t", status="review"))
    first = await store.save_approval_edits_and_mark_crystallizing(document_id="doc_cas2", items=[], content_chunks=[])
    second = await store.save_approval_edits_and_mark_crystallizing(document_id="doc_cas2", items=[], content_chunks=[])
    assert (first, second) == (True, False)
