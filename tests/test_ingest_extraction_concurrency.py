"""L7a gate 1 (ratified 2026-08-28): concurrent chunk extraction.

extract_items used to await one chunk's extraction call at a time —
7-13 independent small-tier calls per session, back to back. On the
LongMemEval bench that was 60-100s of a 130s session; for a customer it
is the same wait on every upload. Chunk extractions now run under a
semaphore sized by CC_INGEST_EXTRACTION_CONCURRENCY.

Pinned here:
  1. Calls overlap up to the bound and no further (max in-flight ==
     the knob), and wall-clock shrinks accordingly.
  2. Order is preserved: items come back in window order with the
     right chunk_index, regardless of which call finished first.
  3. Concurrency 1 is the old serial behaviour.
  4. One window's failure yields nothing for that window and does not
     disturb its neighbours.
  5. A ledger (record_model_call) failure logs and KEEPS the items —
     the old loop's try/except dropped the window on that path.
"""
from __future__ import annotations

import asyncio
import json
import re
import threading
import time

import pytest

from crystal_cache.config import get_settings
from crystal_cache.ingestion.document_pipeline import DocumentPipeline


class _SlowFakeExtractClient:
    """Each call sleeps `delay` in its thread and reports the section it
    was asked about, so the test can check ordering and overlap."""

    def __init__(self, delay: float = 0.15, fail_sections: set[int] = frozenset()):
        self.delay = delay
        self.fail_sections = set(fail_sections)
        self._lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls = 0

    def complete(self, *, system, messages, max_tokens, temperature=0.0, tier="small"):
        prompt = messages[-1]["content"]
        section = int(re.search(r"Section (\d+):", prompt).group(1))
        with self._lock:
            self.calls += 1
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            time.sleep(self.delay)
            if section in self.fail_sections:
                raise RuntimeError(f"section {section} boom")
            return json.dumps([{
                "key": f"fact from section {section}",
                "segments": ["S", str(section)],
                "value": f"value {section}",
                "type": "fact",
            }])
        finally:
            with self._lock:
                self.in_flight -= 1


def _six_paragraph_text() -> str:
    # chunk_size=50 below -> every paragraph is its own window (6 windows).
    return "\n\n".join(f"paragraph number {i} with enough text to fill" for i in range(6))


def _pipeline(fake) -> DocumentPipeline:
    return DocumentPipeline(store=None, encoder=None, vector_store=None, client=fake)


@pytest.mark.asyncio
async def test_extraction_runs_bounded_concurrent_and_ordered(monkeypatch):
    monkeypatch.setattr(get_settings(), "ingest_extraction_concurrency", 3)
    fake = _SlowFakeExtractClient(delay=0.15)
    t0 = time.monotonic()
    items = await _pipeline(fake).extract_items(
        text=_six_paragraph_text(), label="L", chunk_size=50,
    )
    elapsed = time.monotonic() - t0

    assert fake.calls == 6
    assert fake.max_in_flight == 3                 # bounded by the knob
    assert elapsed < 6 * 0.15 * 0.8                # clearly not serial (0.9s)
    # Order preserved: section numbers 1..6 in window order, chunk_index 0..5
    assert [it.key for it in items] == [f"fact from section {i}" for i in range(1, 7)]
    assert [it.chunk_index for it in items] == list(range(6))


@pytest.mark.asyncio
async def test_concurrency_one_is_serial(monkeypatch):
    monkeypatch.setattr(get_settings(), "ingest_extraction_concurrency", 1)
    fake = _SlowFakeExtractClient(delay=0.02)
    items = await _pipeline(fake).extract_items(
        text=_six_paragraph_text(), label="L", chunk_size=50,
    )
    assert fake.max_in_flight == 1
    assert len(items) == 6


@pytest.mark.asyncio
async def test_failed_window_does_not_disturb_neighbours(monkeypatch):
    monkeypatch.setattr(get_settings(), "ingest_extraction_concurrency", 4)
    fake = _SlowFakeExtractClient(delay=0.02, fail_sections={3})
    items = await _pipeline(fake).extract_items(
        text=_six_paragraph_text(), label="L", chunk_size=50,
    )
    keys = [it.key for it in items]
    assert "fact from section 3" not in keys
    assert keys == [f"fact from section {i}" for i in (1, 2, 4, 5, 6)]
    assert [it.chunk_index for it in items] == [0, 1, 3, 4, 5]


@pytest.mark.asyncio
async def test_ledger_failure_keeps_items(monkeypatch):
    """record_model_call raising must not cost the extracted knowledge."""
    from crystal_cache.ingestion import document_pipeline as dp

    class _Usage:
        model = "m"; input_tokens = 1; output_tokens = 1
        cache_creation_tokens = 0; cache_read_tokens = 0
        text = json.dumps([{"key": "k", "segments": ["K"], "value": "v", "type": "fact"}])

    class _DetailedFake:
        def complete_detailed(self, **kwargs):
            return _Usage()

    async def _boom(**kwargs):
        raise RuntimeError("ledger down")

    monkeypatch.setattr(dp, "record_model_call", _boom)
    monkeypatch.setattr(get_settings(), "ingest_extraction_concurrency", 2)
    items = await _pipeline(_DetailedFake()).extract_items(
        text=_six_paragraph_text(), label="L", chunk_size=50,
        customer_id="cus_x", store=object(),
    )
    assert len(items) == 6  # kept, not dropped


def test_knob_default_and_floor():
    s = get_settings()
    assert s.ingest_extraction_concurrency >= 1
    assert s.model_fields["ingest_extraction_concurrency"].default == 6
