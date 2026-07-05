"""Versioned crystals — VS-D2/D3 replace semantics + delete_crystal (CU-9).

Covers the 2026-06-10 locked decisions
(docs/VERSIONED_CRYSTALS_AND_SOURCE_SYNC.md):

  - MetadataStore.delete_crystal: whole-crystal deletion with fact
    cascade, tenancy scoping, and vector-cache invalidation.
  - approve_and_crystallize REPLACE semantics: every content chunk is
    stamped with source_path / content_hash / source_modified_at; an
    unchanged source is skipped on re-ingest (dedup); a CHANGED source
    DELETES its prior crystals and writes a fresh set. No stale
    crystals are ever kept — there is no is_current flag.

Integration tests against the in-memory store + semantic encoder stub
(conftest), exercising the real write path end-to-end. The encoder
stub's pseudo-random vectors make distinct keys spawn distinct
crystals, so counts below are deterministic.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from crystal_cache.ingestion.document_pipeline import DocumentPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunks(f_body: str = "def f(): pass") -> list[dict]:
    """Two code symbols in one file + one script chunk (label-keyed)."""
    return [
        {"index": 0, "label": "a.py::f", "text": f_body,
         "locator": "a.py::f", "subject": None, "doc_type": "code"},
        {"index": 1, "label": "a.py::g", "text": "def g(): pass",
         "locator": "a.py::g", "subject": None, "doc_type": "code"},
        {"index": 2, "label": "Scene 1", "text": "INT. OFFICE — DAY",
         "locator": "Scene 1", "subject": "Corporate Mistletoe",
         "doc_type": "script"},
    ]


async def _make_doc(store, customer_id: str, label: str = "mydoc.txt"):
    return await store.create_document_upload(
        customer_id,
        label,
        "irrelevant raw text",
        source_modified_at=datetime(2026, 6, 10, tzinfo=timezone.utc),
    )


def _pipeline(store, semantic_encoder_stub, vector_store, fact_vector_store):
    return DocumentPipeline(
        store=store,
        encoder=semantic_encoder_stub,
        vector_store=vector_store,
        fact_vector_store=fact_vector_store,
    )


# ---------------------------------------------------------------------------
# delete_crystal (CU-9)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_crystal_cascades_facts(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store
):
    crystal, fact = await store.add_pair_for_customer(
        customer_id=customer.id,
        prompt_text="Code|a.py|f",
        answer_text="def f(): pass",
        pair_type="content_chunk",
        encoder=semantic_encoder_stub,
        vector_store=vector_store,
        crystal_type="customer:legacy",
        source_kind="document_chunk",
    )
    assert await store.get_crystal(crystal.id) is not None
    assert len(await store.list_facts_for_crystal(crystal.id)) == 1

    deleted = await store.delete_crystal(
        crystal.id,
        customer.id,
        vector_store=vector_store,
        fact_vector_store=fact_vector_store,
    )

    assert deleted is True
    assert await store.get_crystal(crystal.id) is None
    assert await store.list_facts_for_crystal(crystal.id) == []
    assert await store.list_all_facts_for_customer(customer.id) == []


@pytest.mark.asyncio
async def test_delete_crystal_is_tenancy_scoped(
    store, customer, semantic_encoder_stub, vector_store
):
    crystal, _ = await store.add_pair_for_customer(
        customer_id=customer.id,
        prompt_text="Code|a.py|f",
        answer_text="def f(): pass",
        pair_type="content_chunk",
        encoder=semantic_encoder_stub,
        vector_store=vector_store,
        crystal_type="customer:legacy",
        source_kind="document_chunk",
    )

    deleted = await store.delete_crystal(crystal.id, "cus_someone_else")

    assert deleted is False
    assert await store.get_crystal(crystal.id) is not None


@pytest.mark.asyncio
async def test_delete_crystal_missing_returns_false(store, customer):
    assert await store.delete_crystal("crys_nope", customer.id) is False


# ---------------------------------------------------------------------------
# approve_and_crystallize — stamp / skip / replace
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_first_ingest_stamps_all_content_chunks(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store
):
    doc = await _make_doc(store, customer.id)
    p = _pipeline(store, semantic_encoder_stub, vector_store, fact_vector_store)

    result = await p.approve_and_crystallize(
        customer_id=customer.id, document_id=doc.id,
        items=[], content_chunks=_chunks(),
    )

    assert result.crystals_written == 3
    crystals = await store.list_crystals_for_customer(customer.id)
    stamped = {c.source_path for c in crystals if c.source_path}
    # Code chunks keyed by file path; the script chunk by the doc label.
    assert stamped == {"a.py", "mydoc.txt"}
    for c in crystals:
        assert c.content_hash, f"crystal {c.id} missing content_hash"
        assert c.source_modified_at is not None


@pytest.mark.asyncio
async def test_unchanged_reingest_is_skipped(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store
):
    doc = await _make_doc(store, customer.id)
    p = _pipeline(store, semantic_encoder_stub, vector_store, fact_vector_store)

    await p.approve_and_crystallize(
        customer_id=customer.id, document_id=doc.id,
        items=[], content_chunks=_chunks(),
    )
    before = {c.id for c in await store.list_crystals_for_customer(customer.id)}

    # Same content again (e.g. user re-uploads the identical file).
    doc2 = await _make_doc(store, customer.id)
    result = await p.approve_and_crystallize(
        customer_id=customer.id, document_id=doc2.id,
        items=[], content_chunks=_chunks(),
    )

    after = {c.id for c in await store.list_crystals_for_customer(customer.id)}
    assert result.crystals_written == 0
    assert after == before  # no duplicates, no deletions


@pytest.mark.asyncio
async def test_changed_source_is_replaced_not_duplicated(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store
):
    doc = await _make_doc(store, customer.id)
    p = _pipeline(store, semantic_encoder_stub, vector_store, fact_vector_store)

    await p.approve_and_crystallize(
        customer_id=customer.id, document_id=doc.id,
        items=[], content_chunks=_chunks(),
    )
    crystals = await store.list_crystals_for_customer(customer.id)
    old_code_ids = {c.id for c in crystals if c.source_path == "a.py"}
    old_code_hash = next(c.content_hash for c in crystals if c.source_path == "a.py")
    script_ids = {c.id for c in crystals if c.source_path == "mydoc.txt"}

    # Re-ingest with ONE symbol's body changed -> the whole a.py path
    # is replaced; the script chunk (unchanged) is skipped.
    doc2 = await _make_doc(store, customer.id)
    result = await p.approve_and_crystallize(
        customer_id=customer.id, document_id=doc2.id,
        items=[], content_chunks=_chunks(f_body="def f(): return 42"),
    )

    crystals2 = await store.list_crystals_for_customer(customer.id)
    new_code = [c for c in crystals2 if c.source_path == "a.py"]
    new_code_ids = {c.id for c in new_code}

    # Both a.py symbols rewritten; old crystals GONE (replace, not version).
    assert result.crystals_written == 2
    assert new_code_ids.isdisjoint(old_code_ids)
    assert old_code_ids.isdisjoint({c.id for c in crystals2})
    # Fresh hash everywhere on the new set; the stale hash survives nowhere.
    assert all(c.content_hash != old_code_hash for c in new_code)
    # Script crystal untouched.
    assert {c.id for c in crystals2 if c.source_path == "mydoc.txt"} == script_ids
    # THE point: re-ingesting a changed file does not grow the bank.
    assert len(crystals2) == len(crystals)
    # And the old crystals' facts are gone with them.
    all_facts = await store.list_all_facts_for_customer(customer.id)
    assert all(f.crystal_id not in old_code_ids for f in all_facts)
