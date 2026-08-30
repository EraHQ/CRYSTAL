"""L7a gate 4 (ratified 2026-08-29: Q1=A, Q2=A, Q3=A): the encoder lane is
a priority queue, and the ingest batch is windowed BULK work.

Before: `encoding/executor.py` was ONE `ThreadPoolExecutor(max_workers=1)`
with a FIFO queue, so a chat turn's preflight encode submitted while an
ingest batch ran waited behind the whole batch (45 s on CPU before gate 2b)
in any process that serves chat AND ingests. Now: one daemon thread
draining a priority queue — INTERACTIVE (the default for every wrapper,
no call site changed) runs ahead of BULK (the ingest pre-encode), FIFO
within a class, still one encode at a time. The batch is cut into windows
of ~CC_INGEST_ENCODE_WINDOW_TOKENS tokens inside each 2b length band, each
window its own BULK job, so an INTERACTIVE encode waits at most one window.

Pinned here:
  1. An INTERACTIVE job submitted while BULK jobs are queued runs before
     the queued BULK jobs (and after the one already executing).
  2. FIFO within a priority class.
  3. Windowing: budget // band texts per job (4 chunks or 64 items at
     2048), one model call per window, rows identical to the unwindowed
     batch and in input order; no window mixes bands.
  4. A job that raises delivers the exception to its awaiter and the lane
     keeps serving.
  5. An awaiter cancelled mid-job does not break the lane or raise
     InvalidStateError when the job later completes.
  6. The lane thread is named cc-encoder; wrappers default to INTERACTIVE.
  7. The knob defaults to 2048; the pipeline floors it at 512.
"""
from __future__ import annotations

import asyncio
import hashlib
import threading

import numpy as np
import pytest

from crystal_cache.config import Settings
from crystal_cache.encoding.executor import (
    Priority,
    encode_native_batch_async,
    run_encoder_bound,
)
from crystal_cache.encoding.semantic import SemanticTextEncoder


# ---------------------------------------------------------------------------
# A SemanticTextEncoder without gtr-t5-base (same shape as the gate-2 pins)
# ---------------------------------------------------------------------------

class _FakeSentenceTransformer:
    def __init__(self, dim: int):
        self.dim = dim
        self.calls: list[object] = []
        self.max_seq_length = 512

    def _one(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        v = np.random.default_rng(seed).standard_normal(self.dim, dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-12)

    def encode(self, x, **_kw):
        self.calls.append(x)
        if isinstance(x, str):
            return self._one(x)
        return np.stack([self._one(t) for t in x])


def _model_free_encoder(native_dim: int = 16, d_hdc: int = 64) -> SemanticTextEncoder:
    enc = SemanticTextEncoder.__new__(SemanticTextEncoder)
    enc.model_name = "fake/for-tests"
    enc.d_hdc = d_hdc
    enc.native_dim = native_dim
    enc._seed = 42
    enc.P = np.random.RandomState(42).choice([-1.0, 1.0], size=(native_dim, d_hdc)).astype(np.float32)
    enc._model = _FakeSentenceTransformer(native_dim)
    return enc


async def _wait_for(event: threading.Event) -> None:
    """Block the test coroutine (not the loop) until a lane job signals."""
    await asyncio.get_running_loop().run_in_executor(None, event.wait)


# ---------------------------------------------------------------------------
# 1-2. Priority and order
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_interactive_jumps_queued_bulk_jobs():
    order: list[str] = []
    started, release = threading.Event(), threading.Event()

    def bulk_1():
        started.set()
        release.wait()                      # hold the lane
        order.append("bulk_1")

    def make(name):
        def fn():
            order.append(name)
        return fn

    t1 = asyncio.ensure_future(run_encoder_bound(bulk_1, priority=Priority.BULK))
    await _wait_for(started)                # bulk_1 is executing, lane busy
    t2 = asyncio.ensure_future(run_encoder_bound(make("bulk_2"), priority=Priority.BULK))
    t3 = asyncio.ensure_future(run_encoder_bound(make("bulk_3"), priority=Priority.BULK))
    await asyncio.sleep(0)                  # let both reach the queue
    t4 = asyncio.ensure_future(run_encoder_bound(make("interactive")))   # default priority
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(t1, t2, t3, t4)

    assert order == ["bulk_1", "interactive", "bulk_2", "bulk_3"]


@pytest.mark.asyncio
async def test_fifo_within_a_class():
    order: list[int] = []
    started, release = threading.Event(), threading.Event()

    def head():
        started.set()
        release.wait()

    def make(i):
        def fn():
            order.append(i)
        return fn

    t0 = asyncio.ensure_future(run_encoder_bound(head, priority=Priority.BULK))
    await _wait_for(started)
    tasks = [asyncio.ensure_future(run_encoder_bound(make(i), priority=Priority.BULK)) for i in range(5)]
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(t0, *tasks)
    assert order == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# 3. Windowing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_windowed_batch_cuts_jobs_by_token_budget_inside_each_band():
    enc = _model_free_encoder()
    chunks = [f"chunk {i} " * 400 for i in range(6)]          # ~3200 chars -> band 512
    items = [f"item {i}" for i in range(100)]                  # band 32
    texts = items[:50] + chunks + items[50:]                   # interleaved on purpose

    reference = enc.encode_native_batch(texts)                 # unwindowed, in-process
    enc._model.calls.clear()

    out = await encode_native_batch_async(enc, texts, priority=Priority.BULK, window_tokens=2048)

    windows = [c for c in enc._model.calls if isinstance(c, list)]
    sizes = sorted(len(w) for w in windows)
    assert sizes == [2, 4, 36, 64]                             # 6 chunks @4/job, 100 items @64/job
    for w in windows:                                          # no window mixes bands
        assert all(len(t) > 2000 for t in w) or all(len(t) < 200 for t in w)
    assert np.array_equal(out, reference)                      # same rows, same order


@pytest.mark.asyncio
async def test_no_budget_means_one_job():
    enc = _model_free_encoder()
    texts = ["a", "b " * 2000, "c"]
    await encode_native_batch_async(enc, texts)                # window_tokens=None
    # one lane job; the encoder's own bucketing still makes one model call per band
    calls = [c for c in enc._model.calls if isinstance(c, list)]
    assert sorted(len(c) for c in calls) == [1, 2]


# ---------------------------------------------------------------------------
# 4-5. Failure and cancellation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_job_exception_reaches_awaiter_and_lane_survives():
    def boom():
        raise RuntimeError("lane boom")
    with pytest.raises(RuntimeError, match="lane boom"):
        await run_encoder_bound(boom)
    assert await run_encoder_bound(lambda: 42) == 42


@pytest.mark.asyncio
async def test_cancelled_awaiter_does_not_break_the_lane():
    started, release = threading.Event(), threading.Event()

    def slow():
        started.set()
        release.wait()
        return "late"

    task = asyncio.ensure_future(run_encoder_bound(slow, priority=Priority.BULK))
    await _wait_for(started)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()                                              # job finishes after the awaiter left
    assert await run_encoder_bound(lambda: "next") == "next"   # no InvalidStateError, lane serving


# ---------------------------------------------------------------------------
# 6-7. Identity and knob
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lane_thread_name_and_default_priority():
    assert await run_encoder_bound(lambda: threading.current_thread().name) == "cc-encoder"
    order: list[str] = []
    started, release = threading.Event(), threading.Event()

    def head():
        started.set()
        release.wait()

    t0 = asyncio.ensure_future(run_encoder_bound(head, priority=Priority.BULK))
    await _wait_for(started)
    tb = asyncio.ensure_future(run_encoder_bound(lambda: order.append("bulk"), priority=Priority.BULK))
    await asyncio.sleep(0)
    td = asyncio.ensure_future(run_encoder_bound(lambda: order.append("default")))
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(t0, tb, td)
    assert order == ["default", "bulk"]                         # default == INTERACTIVE


def test_window_knob_default():
    assert Settings.model_fields["ingest_encode_window_tokens"].default == 2048
