"""Background-worker write path (2026-07-03, step 5 part 1).

Autonomous-worker memory must be born recall_gated so it cannot be used
until reviewed. Cognition writes such output to document_uploads tagged
detected_type='inferred_knowledge'; when those documents are crystallized,
the resulting crystals get origin='background_worker', recall_gated=True.
User/agent uploads stay origin='direct', ungated — exactly as before.

This test proves:
  - recall_stamps() maps origin to the right add_pair kwargs (pure unit);
  - approve_and_crystallize(origin='background_worker') births gated
    crystals held out of recall; the default births ungated ones.

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

import pytest

from crystal_cache.ingestion.document_pipeline import (
    DocumentPipeline,
    recall_stamps,
)


# --- recall_stamps pure unit -----------------------------------------------

def test_recall_stamps_vocabulary():
    # direct => no extra kwargs => add_pair defaults (ungated, direct).
    assert recall_stamps("direct") == {}
    # any non-direct origin => born gated + origin-tagged.
    assert recall_stamps("background_worker") == {
        "origin": "background_worker", "recall_gated": True,
    }


# --- end-to-end: origin stamps the crystals --------------------------------

class _FakeVS:
    def invalidate(self, *a, **k):
        pass


async def _crystallize(store, semantic_encoder_stub, customer_id, *, origin):
    """Write one content chunk through approve_and_crystallize with the
    given origin; return the crystals born."""
    pipe = DocumentPipeline(
        store=store, encoder=semantic_encoder_stub, vector_store=_FakeVS(),
        vector_index=None, fact_vector_store=_FakeVS(),
    )
    doc = await store.create_document_upload(
        customer_id=customer_id, label="wk", text="some finding",
        detected_type=("inferred_knowledge" if origin == "background_worker"
                       else "general"),
    )
    await pipe.approve_and_crystallize(
        customer_id=customer_id,
        document_id=doc.id,
        items=[],
        content_chunks=[{"index": 0, "text": "some finding",
                         "locator": "chunk 0"}],
        origin=origin,
    )
    return await store.list_crystals_for_customer(customer_id)


async def test_background_worker_births_gated_crystals(store, customer, semantic_encoder_stub):
    crystals = await _crystallize(
        store, semantic_encoder_stub, customer.id, origin="background_worker",
    )
    assert crystals, "expected at least one crystal written"
    assert all(c.recall_gated is True for c in crystals)
    assert all(c.origin == "background_worker" for c in crystals)
    # Held out of the recall load.
    recall_view = await store.list_crystals_for_customer(
        customer.id, include_recall_gated=False,
    )
    assert recall_view == []


async def test_direct_upload_births_ungated_crystals(store, customer, semantic_encoder_stub):
    crystals = await _crystallize(
        store, semantic_encoder_stub, customer.id, origin="direct",
    )
    assert crystals, "expected at least one crystal written"
    assert all(c.recall_gated is False for c in crystals)
    assert all(c.origin == "direct" for c in crystals)
    # Present in the recall load.
    recall_view = await store.list_crystals_for_customer(
        customer.id, include_recall_gated=False,
    )
    assert len(recall_view) == len(crystals)


async def test_default_origin_is_direct(store, customer, semantic_encoder_stub):
    """Omitting origin entirely must behave exactly like 'direct'."""
    pipe = DocumentPipeline(
        store=store, encoder=semantic_encoder_stub, vector_store=_FakeVS(),
        vector_index=None, fact_vector_store=_FakeVS(),
    )
    doc = await store.create_document_upload(
        customer_id=customer.id, label="d", text="t", detected_type="general",
    )
    await pipe.approve_and_crystallize(
        customer_id=customer.id, document_id=doc.id, items=[],
        content_chunks=[{"index": 0, "text": "t", "locator": "chunk 0"}],
    )
    crystals = await store.list_crystals_for_customer(customer.id)
    assert crystals
    assert all(c.recall_gated is False and c.origin == "direct"
               for c in crystals)
