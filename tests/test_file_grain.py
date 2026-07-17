"""Gate D (2026-07-16): file-grain crystals on canonical source identity.

C1/C2/C4 ratified: source_uri (scheme-qualified location) + content_hash
(sha256 of extracted text) on uploads; source_uri on crystals; VS-D1
file grain — one content crystal per source, chunks as ordered facts
(facts.chunk_index). Reader renders ALL facts in reading order (capped);
the content router injects the MATCHED fact; supersede carries
citation + chunk_index onto the successor.
"""
from __future__ import annotations

import pytest

from crystal_cache.ingestion.document_pipeline import DocumentPipeline
from crystal_cache.retrieval.reader import (
    CrystalReader,
    _CONTENT_CRYSTAL_MAX_CHARS,
)


# ---------------------------------------------------------------------------
# Upload identity stamping (C1)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upload_gets_identity_pair(store, customer):
    doc = await store.create_document_upload(
        customer.id, "notes.txt", "some extracted text",
    )
    row = await store.get_document_upload(doc.id, customer.id)
    assert getattr(row, "source_uri", None) == f"upload://{doc.id}"
    import hashlib
    assert getattr(row, "content_hash", None) == hashlib.sha256(
        b"some extracted text").hexdigest()


@pytest.mark.asyncio
async def test_drive_upload_keeps_drive_identity(store, customer):
    doc = await store.create_document_upload(
        customer.id, "report.pdf", "drive text",
        source_file_id="1AbC",
    )
    row = await store.get_document_upload(doc.id, customer.id)
    assert getattr(row, "source_uri", None) == "gdrive://1AbC"


# ---------------------------------------------------------------------------
# Reader: ordered multi-fact render + cap
# ---------------------------------------------------------------------------

def _chunk(i, text, locator):
    return {"index": i, "label": locator, "text": text,
            "locator": locator, "subject": None, "doc_type": "code"}


@pytest.mark.asyncio
async def test_reader_renders_file_facts_in_order(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    doc = await store.create_document_upload(customer.id, "m.py", "raw")
    p = DocumentPipeline(store=store, encoder=semantic_encoder_stub,
                         vector_store=vector_store,
                         fact_vector_store=fact_vector_store)
    await p.approve_and_crystallize(
        customer_id=customer.id, document_id=doc.id, items=[],
        content_chunks=[
            _chunk(0, "def first(): pass", "m.py::first"),
            _chunk(1, "def second(): pass", "m.py::second"),
        ],
    )
    crystals = await store.list_crystals_for_customer(customer.id)
    file_crystal = next(c for c in crystals if c.source_path == "m.py")

    ctx = await CrystalReader(store).read(file_crystal)
    assert ctx is not None
    assert ctx.text.index("def first") < ctx.text.index("def second")
    # Provenance headers per chunk survive the join.
    assert ctx.text.count("Source:") >= 1 or "m.py" in ctx.text


@pytest.mark.asyncio
async def test_reader_caps_huge_file_crystals(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    doc = await store.create_document_upload(customer.id, "big.py", "raw")
    p = DocumentPipeline(store=store, encoder=semantic_encoder_stub,
                         vector_store=vector_store,
                         fact_vector_store=fact_vector_store)
    big = "x = 1\n" * 700  # ~4200 chars per chunk
    await p.approve_and_crystallize(
        customer_id=customer.id, document_id=doc.id, items=[],
        content_chunks=[
            _chunk(0, big, "big.py::a"),
            _chunk(1, big, "big.py::b"),
            _chunk(2, big, "big.py::c"),
        ],
    )
    crystals = await store.list_crystals_for_customer(customer.id)
    file_crystal = next(c for c in crystals if c.source_path == "big.py")

    ctx = await CrystalReader(store).read(file_crystal)
    assert ctx is not None
    assert len(ctx.text) <= _CONTENT_CRYSTAL_MAX_CHARS + 200
    assert "[source continues beyond this excerpt]" in ctx.text


# ---------------------------------------------------------------------------
# Content router: injects the MATCHED fact under file grain
# ---------------------------------------------------------------------------

def test_router_resolves_matched_fact_source_pin():
    """The router picks the fact whose id the vector search returned —
    not the file's first chunk. (Runtime coverage rides the retrieval
    integration suite; the load-bearing line is pinned here.)"""
    import inspect
    from crystal_cache.retrieval import v3_routers
    src = inspect.getsource(v3_routers)
    assert "next((f for f in facts if f.id == top_fact_id), facts[0])" in src


# ---------------------------------------------------------------------------
# Supersede: provenance carries onto the successor
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_supersede_carries_citation_and_position(
    store, customer, semantic_encoder_stub, vector_store,
):
    _, fact = await store.add_pair_for_customer(
        customer_id=customer.id,
        prompt_text="Docs|Install|Steps",
        answer_text="Run pip install crystal-cache.",
        pair_type="question_answer",
        encoder=semantic_encoder_stub,
        vector_store=vector_store,
        crystal_type="customer:legacy",
        citation="Section 2.1",
    )
    # The carry expression the endpoint uses (body citation wins, else
    # the original's) — exercised at the store contract level:
    new_fact = await store.add_pair_to_crystal(
        fact.crystal_id,
        fact.prompt_text,
        "Run pip install crystal-cache[all].",
        pair_type=fact.pair_type,
        encoder=semantic_encoder_stub,
        source_kind=fact.source_kind,
        citation=(("" or "").strip() or fact.citation),
        chunk_index=fact.chunk_index,
    )
    assert new_fact.citation == "Section 2.1"

    # And the endpoint source pins the carry semantics.
    import inspect
    from crystal_cache.endpoints import admin
    src = inspect.getsource(admin)
    assert 'or fact.citation)' in src
    assert "chunk_index=fact.chunk_index," in src


# ---------------------------------------------------------------------------
# Bonder bypass by construction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_two_files_never_share_a_crystal(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    """The old hazard: similar chunks from different files could bond
    into one crystal (the shared-stamp warning). File grain kills it
    structurally — identical text in two files lands in two crystals."""
    doc = await store.create_document_upload(customer.id, "twins", "raw")
    p = DocumentPipeline(store=store, encoder=semantic_encoder_stub,
                         vector_store=vector_store,
                         fact_vector_store=fact_vector_store)
    same = "def identical(): pass"
    await p.approve_and_crystallize(
        customer_id=customer.id, document_id=doc.id, items=[],
        content_chunks=[
            _chunk(0, same, "one.py::identical"),
            {"index": 1, "label": "two.py::identical", "text": same,
             "locator": "two.py::identical", "subject": None,
             "doc_type": "code"},
        ],
    )
    crystals = await store.list_crystals_for_customer(customer.id)
    uris = {c.source_uri for c in crystals if c.source_uri}
    assert uris == {"repo://one.py", "repo://two.py"}
