"""Document provenance (2026-09-26, the 243/262 defect).

LIVE-FOUND during the 262-crystal ingest test: every claim the
extraction pipeline distilled from operator-supplied documents was
stamped source_kind='model_reasoning' (the writer hard-coded the store
default), so 243 of 262 crystals claimed the model reasoned them out of
nothing. That corrupts the axis decision 29 trusts (operator-supplied
vs model-inferred). Extracted items now carry 'document_extraction';
verbatim chunks keep 'document_chunk'; 'model_reasoning' never appears
on document-born crystals.

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

from crystal_cache.ingestion.document_pipeline import DocumentPipeline


class _FakeVS:
    def invalidate(self, *a, **k):
        pass

    async def search(self, *a, **k):
        # add_pair's dedup pass; empty means every write is novel.
        return []

    async def note_pair_written(self, *a, **k):
        # post-write index bookkeeping; a no-op here.
        return None


async def test_document_born_crystals_carry_document_provenance(
    store, customer, semantic_encoder_stub
):
    pipe = DocumentPipeline(
        store=store, encoder=semantic_encoder_stub, vector_store=_FakeVS(),
        vector_index=None, fact_vector_store=_FakeVS(),
    )
    doc = await store.create_document_upload(
        customer_id=customer.id, label="prov", text="The sky is blue.",
        detected_type="general",
    )
    await pipe.approve_and_crystallize(
        customer_id=customer.id,
        document_id=doc.id,
        items=[{
            "key": "what color is the sky",
            "segments": ["Nature", "Sky", "Color"],
            "value": "The sky is blue.",
            "citation": "section 1",
        }],
        content_chunks=[{"index": 0, "text": "The sky is blue.",
                         "locator": "chunk 0"}],
    )
    crystals = await store.list_crystals_for_customer(customer.id)
    assert crystals, "expected crystals from one chunk + one item"

    # Provenance lives on the FACT (Phase 2, migration 0011: voicing and
    # the cache-hit path read the matched fact; the crystal-level column
    # is a legacy default every fresh spawn receives). Assert the lane
    # the system actually reads.
    fact_kinds: set[str] = set()
    for c in crystals:
        for f in await store.list_facts_for_crystal(c.id):
            fact_kinds.add(f.source_kind)

    # Both writer paths ran, each with its own honest provenance.
    assert "document_chunk" in fact_kinds
    assert "document_extraction" in fact_kinds
    # THE pin: nothing born from a document ever claims the model
    # reasoned it out of nothing.
    assert "model_reasoning" not in fact_kinds
