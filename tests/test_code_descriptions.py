"""Code descriptions (Checkpoint 2, CRYS June 2026).

Two layers:

  * describe_code_file (unit) — maps each symbol's index to a functional NL
    description + a file_summary. Small files are described whole in one call;
    large files use sequential code-budget batches (each carrying prior
    summary + descriptions forward) plus an end-of-file synopsis pass. Lenient
    JSON parsing; best-effort fallback to empty on any failure.
  * the ingest wiring (integration) — with CC_ENABLE_CODE_DESCRIPTIONS on,
    crystallize_document attaches descriptions to the content-chunk dicts, and
    approve_and_crystallize indexes the resulting fact by the description
    (embed_text) while claim_text still returns the verbatim code.

Uses the in-memory store + deterministic semantic encoder stub + hand-rolled
Anthropic fake from conftest. The stub's encode_native is a pure function of
the text, so "indexed by the description, not the code" is checkable by cosine.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from crystal_cache import config
from crystal_cache.ingestion.code_describer import _parse_json_object, describe_code_file
from crystal_cache.ingestion.document_pipeline import DocumentPipeline
from crystal_cache.workers.crystallization import crystallize_document


def _cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(a @ b / (na * nb))


_CHUNKS = [
    {"index": 0, "label": "alpha()", "text": "def alpha():\n    return 1"},
    {"index": 1, "label": "beta()", "text": "def beta():\n    return 2"},
]


# ---------------------------------------------------------------------------
# describe_code_file — unit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_describe_maps_indices_and_summary(fake_anthropic):
    fake_anthropic.script_text(json.dumps({
        "file_summary": "Two tiny functions.",
        "symbols": {"0": "alpha returns the first value.", "1": "beta returns the second."},
    }))
    out = await describe_code_file(
        chunks=_CHUNKS, client=fake_anthropic, file_label="m.py",
    )
    assert out["file_summary"] == "Two tiny functions."
    assert out["by_index"] == {
        0: "alpha returns the first value.",
        1: "beta returns the second.",
    }
    fake_anthropic.assert_call_count(1)


@pytest.mark.asyncio
async def test_describe_handles_fenced_json(fake_anthropic):
    fake_anthropic.script_text(
        '```json\n{"file_summary": "f", "symbols": {"0": "does a thing"}}\n```'
    )
    out = await describe_code_file(
        chunks=_CHUNKS[:1], client=fake_anthropic, file_label="m.py",
    )
    assert out["by_index"] == {0: "does a thing"}
    assert out["file_summary"] == "f"


@pytest.mark.asyncio
async def test_describe_bad_json_returns_empty(fake_anthropic):
    fake_anthropic.script_text("sorry, I couldn't do that")
    out = await describe_code_file(
        chunks=_CHUNKS[:1], client=fake_anthropic, file_label="m.py",
    )
    assert out == {"file_summary": "", "by_index": {}}


@pytest.mark.asyncio
async def test_describe_no_client_returns_empty_without_calling():
    out = await describe_code_file(
        chunks=_CHUNKS[:1], client=None, file_label="m.py",
    )
    assert out == {"file_summary": "", "by_index": {}}


def test_parse_json_object_variants():
    assert _parse_json_object('{"a": 1}') == {"a": 1}
    assert _parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert _parse_json_object('here you go: {"a": 1} done') == {"a": 1}
    assert _parse_json_object("not json at all") is None
    assert _parse_json_object('[1, 2, 3]') is None  # array is not an object


class _RecordingClient:
    """Seam-shaped client that returns queued responses in order and
    records each call, so a test can assert what carry-forward context each
    batch was given."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0) if self._responses else "{}"


@pytest.mark.asyncio
async def test_describe_batched_with_synopsis_and_chaining():
    """A file too large for one call uses sequential batches (symbols keyed by
    GLOBAL index, each batch carrying prior summary + descriptions forward),
    then a final synopsis call over ALL descriptions sets file_summary."""
    # 5 symbols of ~4500-char bodies, capped to 4000, force two batches at the
    # 18000 budget: batch 1 = globals 0..3 (16000), batch 2 = global 4.
    chunks = [
        {"index": i, "label": f"sym_{i}()", "text": f"# symbol {i}\n" + "y" * 4500}
        for i in range(5)
    ]
    big_file_text = "y" * 20000  # > WHOLE_FILE_BUDGET -> batched mode
    resp_b1 = json.dumps({
        "file_summary": "part one",
        "symbols": {"0": "d0", "1": "d1", "2": "d2", "3": "d3"},
    })
    resp_b2 = json.dumps({"file_summary": "part two", "symbols": {"0": "d4"}})
    synopsis = "This file orchestrates the widget lifecycle."
    client = _RecordingClient([resp_b1, resp_b2, synopsis])

    out = await describe_code_file(
        file_text=big_file_text, chunks=chunks, client=client, file_label="big.py",
    )

    # 2 batches + 1 synopsis call
    assert len(client.calls) == 3
    # all 5 symbols described, keyed by GLOBAL index
    assert set(out["by_index"]) == set(range(5))
    assert out["by_index"][4] == "d4"          # batch 2 local 0 -> global 4
    # file_summary comes from the SYNOPSIS pass, not a batch's running summary
    assert out["file_summary"] == synopsis

    b2_prompt = client.calls[1]["messages"][0]["content"]
    syn_prompt = client.calls[2]["messages"][0]["content"]
    # chaining: batch 2 carried batch 1's summary + a prior description
    assert "part 2 of 2" in b2_prompt
    assert "part one" in b2_prompt
    assert "d3" in b2_prompt
    # synopsis saw the complete description set
    assert all(f"d{i}" in syn_prompt for i in range(5))
    # determinism: every call pinned temperature=0
    assert all(c.get("temperature") == 0.0 for c in client.calls)


@pytest.mark.asyncio
async def test_describe_batch_failure_is_isolated():
    """One batch returning junk doesn't sink the others; the synopsis still runs
    over the surviving descriptions."""
    chunks = [
        {"index": i, "label": f"sym_{i}()", "text": f"# symbol {i}\n" + "y" * 4500}
        for i in range(5)
    ]
    big_file_text = "y" * 20000
    client = _RecordingClient([
        "not json",                                                  # batch 1 -> skipped
        json.dumps({"file_summary": "p2", "symbols": {"0": "d4"}}),   # batch 2 ok
        "Synopsis from the one surviving symbol.",                   # synopsis
    ])
    out = await describe_code_file(
        file_text=big_file_text, chunks=chunks, client=client, file_label="big.py",
    )
    assert len(client.calls) == 3
    assert set(out["by_index"]) == {4}            # only batch 2's symbol survived
    assert out["by_index"][4] == "d4"
    assert out["file_summary"] == "Synopsis from the one surviving symbol."


# ---------------------------------------------------------------------------
# Ingest wiring — integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_crystallize_attaches_descriptions_when_flag_on(
    store, customer, semantic_encoder_stub, vector_store, fake_anthropic, monkeypatch
):
    """Flag on + code doc + a client → content-chunk dicts gain a description."""
    monkeypatch.setattr(config.settings, "enable_code_descriptions", True)
    code = "def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n"
    doc = await store.create_document_upload(
        customer_id=customer.id, label="m.py", text=code,
    )
    fake_anthropic.script_text(json.dumps({
        "file_summary": "Tiny module.",
        "symbols": {"0": "alpha returns one.", "1": "beta returns two."},
    }))

    await crystallize_document(
        store=store, encoder=semantic_encoder_stub, vector_store=vector_store,
        document_id=doc.id, client=fake_anthropic,
    )

    row = await store.get_document_upload(doc.id, customer.id)
    descs = [c.get("description") for c in (row.content_chunks or [])]
    assert any(d for d in descs), descs
    # exactly one model call (the description pass); code skips extraction
    fake_anthropic.assert_call_count(1)


@pytest.mark.asyncio
async def test_crystallize_skips_descriptions_when_flag_off(
    store, customer, semantic_encoder_stub, vector_store, fake_anthropic
):
    """Flag off (default) → no description, no model call (today's behavior)."""
    code = "def alpha():\n    return 1\n"
    doc = await store.create_document_upload(
        customer_id=customer.id, label="m.py", text=code,
    )
    await crystallize_document(
        store=store, encoder=semantic_encoder_stub, vector_store=vector_store,
        document_id=doc.id, client=fake_anthropic,
    )
    row = await store.get_document_upload(doc.id, customer.id)
    assert all(not c.get("description") for c in (row.content_chunks or []))
    fake_anthropic.assert_call_count(0)


@pytest.mark.asyncio
async def test_approve_indexes_content_chunk_by_description(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store
):
    """A content chunk carrying a description is indexed by it (embed_text);
    the returned body stays the verbatim code."""
    pipeline = DocumentPipeline(
        store=store, encoder=semantic_encoder_stub,
        vector_store=vector_store, fact_vector_store=fact_vector_store,
    )
    doc = await store.create_document_upload(
        customer_id=customer.id, label="z.py", text="def z(): pass",
    )
    CODE = "def z():\n    pass"
    DESC = "z is a placeholder function that intentionally does nothing."
    chunk = {
        "index": 0, "label": "z()", "text": CODE, "locator": "z.py::z",
        "subject": "z", "doc_type": "code", "description": DESC,
    }

    result = await pipeline.approve_and_crystallize(
        customer_id=customer.id, document_id=doc.id, items=[],
        content_chunks=[chunk], crystal_type="customer:legacy",
    )
    assert result.crystals_written == 1

    crystals = await store.list_crystals_for_customer(customer.id)
    cc = [c for c in crystals if c.build_method == "content_chunk"]
    assert len(cc) == 1
    facts = await store.list_facts_for_crystal(cc[0].id)
    fact = facts[0]

    # indexed by the description, NOT the code; body is still the code
    assert _cos(fact.vector, semantic_encoder_stub.encode_native(DESC)) > 0.999
    assert _cos(fact.vector, semantic_encoder_stub.encode_native(CODE)) < 0.2
    assert fact.claim_text == CODE
