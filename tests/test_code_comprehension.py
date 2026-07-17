"""Gate D2 (2026-07-17): code comprehension at ingest.

Amends Gate A's code-extraction exclusion (ratified Q1=B, Q2=C):
mechanical import facts + resolved import CHAINS between file
crystals, and the describer's judgment promoted to queryable purpose
facts — all living on the file crystal so supersede retires them with
the version they describe. Plus the load-bearing find: delete_crystal
now cascades chains (no FK CASCADE in schema — Postgres would reject
the replace path otherwise).
"""
from __future__ import annotations

import pytest

from crystal_cache.ingestion.code_structure import (
    extract_imports,
    resolve_import_target,
)
from crystal_cache.ingestion.document_pipeline import DocumentPipeline
from crystal_cache.models.crystal_type import CrystalChain


def _chunk(i, text, locator, description=None):
    d = {"index": i, "label": locator, "text": text,
         "locator": locator, "subject": None, "doc_type": "code"}
    if description:
        d["description"] = description
    return d


def _pipeline(store, enc, vs, fvs):
    return DocumentPipeline(store=store, encoder=enc, vector_store=vs,
                            fact_vector_store=fvs)


# ---------------------------------------------------------------------------
# Mechanical extraction unit tests
# ---------------------------------------------------------------------------

def test_extract_imports_shapes():
    text = (
        "import json\n"
        "from ..cost.emit import record_model_call\n"
        "from crystal_cache.llm import get_llm_client\n"
        "import numpy as np\n"
    )
    assert extract_imports(text) == [
        "json", "..cost.emit", "crystal_cache.llm", "numpy",
    ]


class _C:
    def __init__(self, cid, path, uri):
        self.id = cid
        self.source_path = path
        self.source_uri = uri


def test_resolver_requires_unique_two_segment_match():
    cands = [
        _C("c1", "crystal_cache/cost/emit.py", "repo://crystal_cache/cost/emit.py"),
        _C("c2", "other/cost/emit.py", "repo://other/cost/emit.py"),
    ]
    # Fully-qualified: unique via the longest suffix.
    assert resolve_import_target(
        "crystal_cache.cost.emit", "x.py", cands).id == "c1"
    # Relative form matches BOTH at 'cost/emit.py' -> ambiguous -> None.
    assert resolve_import_target("..cost.emit", "x.py", cands) is None
    # External package: nothing matches.
    assert resolve_import_target("numpy", "x.py", cands) is None


# ---------------------------------------------------------------------------
# Ingest integration: facts + chains
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_import_facts_and_intra_upload_chain(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    doc = await store.create_document_upload(customer.id, "pkg", "raw")
    p = _pipeline(store, semantic_encoder_stub, vector_store, fact_vector_store)
    await p.approve_and_crystallize(
        customer_id=customer.id, document_id=doc.id, items=[],
        content_chunks=[
            _chunk(0, "def helper(): pass", "pkg/util.py::helper"),
            _chunk(1, "from pkg.util import helper\nimport json",
                   "pkg/main.py::<module>"),
        ],
    )
    crystals = await store.list_crystals_for_customer(customer.id)
    by_uri = {c.source_uri: c for c in crystals if c.source_uri}
    main = by_uri["repo://pkg/main.py"]
    util = by_uri["repo://pkg/util.py"]

    facts = await store.list_facts_for_crystal(main.id)
    rels = [f for f in facts if f.pair_type == "entity_relationship"]
    answers = {f.claim_text for f in rels}
    # Resolved import names its in-bank target; external stays plain.
    assert any(
        "imports pkg.util" in a and "in this bank: pkg/util.py" in a
        for a in answers
    )
    assert any(a.endswith("imports json") for a in answers)

    # The resolved import produced a directed chain main -> util.
    chains = await store.list_chains_from_source(main.id)
    assert any(ch.target_crystal_id == util.id for ch in chains)
    # External import produced NO chain (facts-only).
    assert all(ch.target_crystal_id == util.id for ch in chains)


@pytest.mark.asyncio
async def test_describer_judgment_promoted_to_facts(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    doc = await store.create_document_upload(customer.id, "m.py", "raw")
    p = _pipeline(store, semantic_encoder_stub, vector_store, fact_vector_store)
    await p.approve_and_crystallize(
        customer_id=customer.id, document_id=doc.id, items=[],
        content_chunks=[
            _chunk(0, '"""doc"""\nimport json', "m.py::<module>",
                   description="m.py: derives retrieval keys from text"),
            _chunk(1, "def gen(): pass", "m.py::gen",
                   description="Builds the wide-to-specific key path"),
        ],
    )
    crystals = await store.list_crystals_for_customer(customer.id)
    m = next(c for c in crystals if c.source_uri == "repo://m.py")
    facts = await store.list_facts_for_crystal(m.id)
    qa = {f.claim_text for f in facts if f.pair_type == "question_answer"}
    assert "m.py: derives retrieval keys from text" in qa
    assert "Builds the wide-to-specific key path" in qa
    # Purpose facts carry their locator as citation.
    gen_fact = next(
        f for f in facts
        if f.claim_text == "Builds the wide-to-specific key path"
    )
    assert gen_fact.citation == "m.py::gen"


@pytest.mark.asyncio
async def test_prose_documents_get_no_comprehension_pass(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    doc = await store.create_document_upload(customer.id, "notes.txt", "raw")
    p = _pipeline(store, semantic_encoder_stub, vector_store, fact_vector_store)
    await p.approve_and_crystallize(
        customer_id=customer.id, document_id=doc.id, items=[],
        content_chunks=[{
            "index": 0, "label": "Section 1",
            "text": "We should import better habits from the platform team.",
            "locator": "Section 1", "subject": None, "doc_type": "general",
        }],
    )
    crystals = await store.list_crystals_for_customer(customer.id)
    c = next(c for c in crystals if c.source_path == "notes.txt")
    facts = await store.list_facts_for_crystal(c.id)
    assert all(f.pair_type == "content_chunk" for f in facts)


# ---------------------------------------------------------------------------
# The load-bearing cascade: chains die with their crystals
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_crystal_cascades_chains(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    doc = await store.create_document_upload(customer.id, "pkg2", "raw")
    p = _pipeline(store, semantic_encoder_stub, vector_store, fact_vector_store)
    await p.approve_and_crystallize(
        customer_id=customer.id, document_id=doc.id, items=[],
        content_chunks=[
            _chunk(0, "def h(): pass", "pkg2/base.py::h"),
            _chunk(1, "from pkg2.base import h", "pkg2/app.py::<module>"),
        ],
    )
    crystals = await store.list_crystals_for_customer(customer.id)
    by_uri = {c.source_uri: c for c in crystals if c.source_uri}
    app = by_uri["repo://pkg2/app.py"]
    base = by_uri["repo://pkg2/base.py"]
    assert await store.list_chains_from_source(app.id)

    # Deleting the TARGET removes the edge pointing at it (Postgres FK
    # would otherwise reject this delete outright).
    assert await store.delete_crystal(
        base.id, customer.id,
        vector_store=vector_store, fact_vector_store=fact_vector_store,
    )
    assert not await store.list_chains_from_source(app.id)

    # And re-ingesting the changed importer re-derives cleanly.
    assert await store.delete_crystal(
        app.id, customer.id,
        vector_store=vector_store, fact_vector_store=fact_vector_store,
    )
