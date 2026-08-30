"""L7a gate 2 (ratified 2026-08-28: Q1=B, Q2=A, Q3=B, Q4=A): batch/dedupe
encoding on the ingest write path.

The approve phase used to run three or four encoder forward passes per
pair — key for routing, key again inside add_pair_to_crystal, answer to
HDC, answer again to native — one text at a time through the single
encoder lane. encode() is encode_native() + a numpy projection, so one
native pass yields both vectors; and sentence-transformers encodes a
list far faster than n single calls. Now approve_and_crystallize
pre-encodes every distinct text of the document in ONE batched native
call and hands the vectors into add_pair_for_customer /
add_pair_to_crystal, which then make zero encoder calls.

Pinned here:
  1. Encoder math on the REAL semantic.py methods (model-free
     SemanticTextEncoder with a fake sentence-transformer): batch row i
     == per-text encode_native; project(row) == encode; empty text is a
     zero row on both; project() refuses a batch.
  1b. Gate 2b (Q1=A): the batch is split into power-of-two token-length
     buckets, one model call each, so a short text is never padded up
     to a 512-token chunk (the padded first batch measured 36-49 s per
     session, flat in the text count). No call mixes a chunk-length
     text with a short one; rows still equal the per-text encode.
  2. Store, vectors supplied: zero model calls; Fact.vector is the
     supplied native; the crystal's summary_vector matches a text-path
     write of the same pair.
  3. Store, no vectors, project-capable encoder: TWO model calls per
     pair (key once — the routing vector is handed down — answer once,
     HDC by projection); embed_text set takes the encode(answer) branch.
  4. Store, an encoder with no batch surface (the conftest stub shape):
     supports_batch_encode is False; three model calls per pair (key
     once, answer twice) — the pre-gate-2 four minus the key re-encode.
  5. Pipeline: exactly one batch call (all these texts fall in one
     length bucket) whose input is the distinct texts in first-seen
     order (a sparse key shared by two items appears once), zero
     per-text calls, fact vectors == the batched natives, and the same
     crystal/fact counts as the no-batch path.
  6. Pipeline fail-safe: the batch call raising still writes every pair
     via per-text encodes and logs pre_encode_failed.
  7. Executor: encode_native_batch_async is one job on the cc-encoder
     lane.
"""
from __future__ import annotations

import hashlib
import logging
import threading

import numpy as np
import pytest

from crystal_cache.encoding.executor import (
    encode_native_batch_async,
    supports_batch_encode,
)
from crystal_cache.encoding.semantic import SemanticTextEncoder
from crystal_cache.ingestion.document_pipeline import DocumentPipeline


# ---------------------------------------------------------------------------
# A SemanticTextEncoder without gtr-t5-base: real methods, fake model
# ---------------------------------------------------------------------------

class _FakeSentenceTransformer:
    """Deterministic stand-in for the sentence-transformers model:
    unit-norm vectors seeded from the text, str or list input, and a
    record of every call so the pins can count forward passes."""

    def __init__(self, dim: int):
        self.dim = dim
        self.calls: list[object] = []
        self.threads: list[str] = []

    def _one(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        v = np.random.default_rng(seed).standard_normal(self.dim, dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-12)

    def encode(self, x, **_kw):
        self.calls.append(x)
        self.threads.append(threading.current_thread().name)
        if isinstance(x, str):
            return self._one(x)
        return np.stack([self._one(t) for t in x])


def _model_free_encoder(native_dim: int, d_hdc: int) -> SemanticTextEncoder:
    """Build the real class without loading a model: same P construction
    as __init__, same seed, so every method under test is the shipped
    one."""
    enc = SemanticTextEncoder.__new__(SemanticTextEncoder)
    enc.model_name = "fake/for-tests"
    enc.d_hdc = d_hdc
    enc.native_dim = native_dim
    enc._seed = 42
    enc.P = np.random.RandomState(42).choice(
        [-1.0, 1.0], size=(native_dim, d_hdc)
    ).astype(np.float32)
    enc._model = _FakeSentenceTransformer(native_dim)
    return enc


class _NoBatchEncoder:
    """The conftest semantic stub's SHAPE — encode / encode_native /
    fingerprint only, no project, no batch — over the same fake model so
    the pins can count its forward passes too."""

    def __init__(self, inner: SemanticTextEncoder):
        self._inner = inner

    def encode(self, text):
        return self._inner.encode(text)

    def encode_native(self, text):
        return self._inner.encode_native(text)

    def fingerprint(self):
        return self._inner.fingerprint()


@pytest.fixture(scope="module")
def _shared_P():
    # 768 x 10_000 to match the production geometry the store expects;
    # built once per module (the choice() call is the slow part).
    return np.random.RandomState(42).choice(
        [-1.0, 1.0], size=(768, 10_000)
    ).astype(np.float32)


@pytest.fixture
def batch_encoder(_shared_P) -> SemanticTextEncoder:
    enc = SemanticTextEncoder.__new__(SemanticTextEncoder)
    enc.model_name = "fake/for-tests"
    enc.d_hdc = 10_000
    enc.native_dim = 768
    enc._seed = 42
    enc.P = _shared_P
    enc._model = _FakeSentenceTransformer(768)
    return enc


def _cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0.0 or nb == 0.0 else float(a @ b / (na * nb))


# ---------------------------------------------------------------------------
# 1. Encoder math
# ---------------------------------------------------------------------------

def test_batch_rows_equal_per_text_encodes_and_project_equals_encode():
    enc = _model_free_encoder(native_dim=32, d_hdc=256)
    texts = ["alpha beta", "", "gamma", "alpha beta"]

    nb = enc.encode_native_batch(texts)
    assert nb.shape == (4, 32) and nb.dtype == np.float32
    assert not nb[1].any()                                   # empty -> zero row
    for i in (0, 2, 3):
        assert np.allclose(nb[i], enc.encode_native(texts[i]), atol=1e-6)
        assert np.allclose(enc.project(nb[i]), enc.encode(texts[i]), atol=1e-6)
        assert abs(np.linalg.norm(enc.project(nb[i])) - 1.0) < 1e-5
    assert np.array_equal(nb[0], nb[3])                      # order kept, dup identical
    # The empty sentinel survives projection: encode("") == project(zero row).
    assert not enc.encode("").any()
    assert not enc.project(nb[1]).any()
    # The batch method hands the model every LIVE text (dedupe is the
    # pipeline's job, filtering empties is the encoder's).
    assert enc._model.calls[0] == ["alpha beta", "gamma", "alpha beta"]


def test_project_refuses_a_batch():
    enc = _model_free_encoder(native_dim=32, d_hdc=256)
    with pytest.raises(ValueError):
        enc.project(np.zeros((3, 32), dtype=np.float32))


def test_length_buckets_are_power_of_two_bands_floored_and_capped():
    enc = _model_free_encoder(native_dim=32, d_hdc=256)
    enc._model.max_seq_length = 512
    texts = [
        "x" * 10,      # ~2 tokens  -> floor band 32
        "x" * 200,     # ~50 tokens -> band 64
        "x" * 1000,    # ~250       -> band 256
        "x" * 3000,    # ~750       -> capped to 512
        "x" * 12000,   # ~3000      -> capped to 512 (same bucket)
        "x" * 40,      # ~10        -> floor band 32 (joins the first)
    ]
    buckets = enc.length_buckets(texts)
    assert buckets == [(32, [0, 5]), (64, [1]), (256, [2]), (512, [3, 4])]   # ascending bands, input order inside


def test_batch_never_pads_short_texts_up_to_a_chunk():
    """Gate 2b: chunk-length and item-length texts go to the model in
    separate calls, and the reassembled rows are still the per-text
    encodes in input order."""
    enc = _model_free_encoder(native_dim=32, d_hdc=256)
    enc._model.max_seq_length = 512
    chunk_a, chunk_b = "chunk a " * 400, "chunk b " * 400        # ~3200 chars each
    texts = ["short one", chunk_a, "", "short two", chunk_b, "short three"]

    nb = enc.encode_native_batch(texts)

    calls = [c for c in enc._model.calls if isinstance(c, list)]
    assert len(calls) == 2                                     # one per band that is populated
    for call in calls:
        lengths = [len(t) for t in call]
        assert max(lengths) < 200 or min(lengths) > 2000     # no long/short mixing
    assert sorted(t for c in calls for t in c) == sorted(t for t in texts if t)
    assert not nb[2].any()                                     # empty sentinel kept
    for i, t in enumerate(texts):
        if t:
            assert np.allclose(nb[i], enc.encode_native(t), atol=1e-6)


def test_supports_batch_encode_is_the_duck_check(batch_encoder):
    assert supports_batch_encode(batch_encoder)
    assert not supports_batch_encode(_NoBatchEncoder(batch_encoder))


# ---------------------------------------------------------------------------
# 2-4. Store
# ---------------------------------------------------------------------------

KEY = "Topic|Ingest|batch encoding"
ANSWER = "One native pass per distinct text; HDC vectors by projection."


@pytest.mark.asyncio
async def test_store_makes_zero_model_calls_when_vectors_supplied(
    store, vector_store, batch_encoder,
):
    # Text-path reference write first (a different customer).
    _, ref_fact = await store.add_pair_for_customer(
        customer_id="g2-text", prompt_text=KEY, answer_text=ANSWER,
        encoder=batch_encoder, vector_store=vector_store,
        crystal_type="customer:legacy",
    )
    ref_crystal = await store.get_crystal(ref_fact.crystal_id)

    # Vectors produced the way the pipeline produces them.
    native = batch_encoder.encode_native_batch([KEY, ANSWER])
    batch_encoder._model.calls.clear()
    crystal, fact = await store.add_pair_for_customer(
        customer_id="g2-vec", prompt_text=KEY, answer_text=ANSWER,
        encoder=batch_encoder, vector_store=vector_store,
        crystal_type="customer:legacy",
        prompt_hdc=batch_encoder.project(native[0]),
        answer_hdc=batch_encoder.project(native[1]),
        answer_native=native[1],
    )
    assert batch_encoder._model.calls == []                  # zero forward passes
    assert _cos(fact.vector, native[1]) > 0.9999             # the supplied native
    assert fact.claim_text == ANSWER
    assert _cos(crystal.summary_vector, ref_crystal.summary_vector) > 0.9999


@pytest.mark.asyncio
async def test_store_no_vectors_project_capable_is_two_passes(
    store, vector_store, batch_encoder,
):
    _, fact = await store.add_pair_for_customer(
        customer_id="g2-two", prompt_text=KEY, answer_text=ANSWER,
        encoder=batch_encoder, vector_store=vector_store,
        crystal_type="customer:legacy",
    )
    calls = batch_encoder._model.calls
    assert calls == [KEY, ANSWER]                             # key once, answer once
    assert _cos(fact.vector, batch_encoder.encode_native(ANSWER)) > 0.9999


@pytest.mark.asyncio
async def test_store_embed_text_takes_the_encode_answer_branch(
    store, vector_store, batch_encoder,
):
    desc = "a plain-language description that indexes the pair"
    _, fact = await store.add_pair_for_customer(
        customer_id="g2-embed", prompt_text=KEY, answer_text=ANSWER,
        encoder=batch_encoder, vector_store=vector_store,
        crystal_type="customer:legacy", embed_text=desc,
    )
    # key (routing), description (native), answer (HDC) — the answer's
    # HDC cannot come from the description's native vector.
    assert batch_encoder._model.calls == [KEY, desc, ANSWER]
    assert _cos(fact.vector, batch_encoder.encode_native(desc)) > 0.9999


@pytest.mark.asyncio
async def test_store_no_batch_surface_is_three_passes(
    store, vector_store, batch_encoder,
):
    stub = _NoBatchEncoder(batch_encoder)
    _, fact = await store.add_pair_for_customer(
        customer_id="g2-stub", prompt_text=KEY, answer_text=ANSWER,
        encoder=stub, vector_store=vector_store,
        crystal_type="customer:legacy",
    )
    # key once (routing vector handed down), answer to native, answer to
    # HDC — no project(), so the third pass stays.
    assert batch_encoder._model.calls == [KEY, ANSWER, ANSWER]
    assert _cos(fact.vector, batch_encoder.encode_native(ANSWER)) > 0.9999


# ---------------------------------------------------------------------------
# 5-6. Pipeline
# ---------------------------------------------------------------------------

def _chunk(i: int, text: str, description: str | None = None) -> dict:
    d = {"index": i, "label": f"Section {i}", "text": text,
         "locator": f"Section {i}", "subject": "Ingest", "domain": "Docs",
         "doc_type": "general"}
    if description is not None:
        d["description"] = description
    return d


ITEMS = [
    {"key": "batch call count", "value": "one per approve", "type": "fact",
     "sparse_key": "Docs|Ingest|batch"},
    {"key": "projection cost", "value": "about one millisecond", "type": "fact",
     "sparse_key": "Docs|Ingest|batch"},          # shared key -> encoded once
    {"key": "fallback", "value": "per-text encodes", "type": "definition",
     "sparse_key": "Docs|Ingest|fallback"},
]
CHUNKS = [
    _chunk(0, "Chunk zero body text."),
    _chunk(1, "Chunk one body text.", description="What chunk one is about."),
]


async def _approve(store, customer, encoder, vector_store, fact_vector_store):
    doc = await store.create_document_upload(customer.id, "notes.txt", "raw")
    p = DocumentPipeline(store=store, encoder=encoder, vector_store=vector_store,
                         fact_vector_store=fact_vector_store)
    result = await p.approve_and_crystallize(
        customer_id=customer.id, document_id=doc.id,
        items=[dict(it) for it in ITEMS], content_chunks=[dict(c) for c in CHUNKS],
    )
    crystals = await store.list_crystals_for_customer(customer.id)
    return result, crystals


@pytest.mark.asyncio
async def test_pipeline_one_batch_call_over_distinct_texts(
    store, customer, vector_store, fact_vector_store, batch_encoder,
):
    result, crystals = await _approve(
        store, customer, batch_encoder, vector_store, fact_vector_store,
    )
    assert result.errors == 0
    calls = batch_encoder._model.calls
    assert len(calls) == 1 and isinstance(calls[0], list)    # ONE batch, no per-text
    sk0 = DocumentPipeline._chunk_sparse_key(CHUNKS[0])
    sk1 = DocumentPipeline._chunk_sparse_key(CHUNKS[1])
    assert calls[0] == [
        sk0, "Chunk zero body text.",                        # desc-or-text == text: once
        sk1, "Chunk one body text.", "What chunk one is about.",
        "Docs|Ingest|batch", "one per approve",              # shared key: once
        "about one millisecond",
        "Docs|Ingest|fallback", "per-text encodes",
    ]
    # The batched natives are what landed on the facts.
    file_crystal = next(c for c in crystals if c.build_method == "content_chunk")
    facts = await store.list_facts_for_crystal(file_crystal.id)
    by_idx = {f.chunk_index: f for f in facts}
    assert _cos(by_idx[0].vector, batch_encoder.encode_native("Chunk zero body text.")) > 0.9999
    assert _cos(by_idx[1].vector, batch_encoder.encode_native("What chunk one is about.")) > 0.9999
    assert result.crystals_written == len(ITEMS) + 1          # 3 items + 1 file crystal
    assert result.items_extracted == len(ITEMS)


@pytest.mark.asyncio
async def test_pipeline_no_batch_surface_writes_the_same_bank(
    store, customer, vector_store, fact_vector_store, batch_encoder,
):
    stub = _NoBatchEncoder(batch_encoder)
    result, crystals = await _approve(
        store, customer, stub, vector_store, fact_vector_store,
    )
    assert result.errors == 0
    assert all(isinstance(c, str) for c in batch_encoder._model.calls)   # per-text only
    assert result.crystals_written == len(ITEMS) + 1          # same bank shape as batched
    assert result.items_extracted == len(ITEMS)
    file_crystal = next(c for c in crystals if c.build_method == "content_chunk")
    assert len(await store.list_facts_for_crystal(file_crystal.id)) == len(CHUNKS)


@pytest.mark.asyncio
async def test_pipeline_batch_failure_falls_back_to_per_text(
    store, customer, vector_store, fact_vector_store, batch_encoder, caplog,
):
    def _boom(_texts):
        raise RuntimeError("batch boom")
    batch_encoder.encode_native_batch = _boom              # instance override
    assert supports_batch_encode(batch_encoder)

    with caplog.at_level(logging.ERROR, logger="crystal_cache.ingestion.document_pipeline"):
        result, crystals = await _approve(
            store, customer, batch_encoder, vector_store, fact_vector_store,
        )
    assert result.errors == 0
    assert result.crystals_written == len(ITEMS) + 1
    assert result.items_extracted == len(ITEMS)
    assert any(r.getMessage() == "document_pipeline.pre_encode_failed" for r in caplog.records)
    assert all(isinstance(c, str) for c in batch_encoder._model.calls)   # fell back
    file_crystal = next(c for c in crystals if c.build_method == "content_chunk")
    assert len(await store.list_facts_for_crystal(file_crystal.id)) == len(CHUNKS)


# ---------------------------------------------------------------------------
# 7. Executor lane
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_batch_async_is_one_job_on_the_encoder_lane(batch_encoder):
    out = await encode_native_batch_async(batch_encoder, ["a", "b", ""])
    assert out.shape == (3, 768)
    assert len(batch_encoder._model.calls) == 1
    assert batch_encoder._model.threads[0].startswith("cc-encoder")
