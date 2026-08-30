"""Async encoder execution — the run_in_executor fix (2026-06-11), now a
priority lane (L7a gate 4, 2026-08-29).

THE PROBLEM (documented in the v1 context doc as "ISSUE: encoder
blocks event loop"): SemanticTextEncoder.encode() is CPU-bound
(sentence-transformer forward pass, ~10-50ms on CPU). Every async path
that called it directly — retrieval per request, crystallization per
chunk, learning per fact — froze the event loop for that long, which
violates the first core principle: no component starves another. One
document being crystallized could stutter every concurrent API request.

THE FIX: a dedicated single-thread lane for all encoder work, plus
awaitable wrappers. Why ONE thread, deliberately:

  * One encode at a time keeps "the encoder is a shared resource"
    TRUE in the scheduler instead of aspirational in a comment —
    concurrent callers queue here, visibly, instead of contending for
    torch threads invisibly.
  * sentence-transformers is thread-safe for inference, but parallel
    forward passes on CPU fight over the same cores and ALL get slower.
    Serialized, each encode finishes at full speed.
  * The event loop never blocks either way — that's the point.

THE SECOND PROBLEM (L7a gate 4): one thread with a FIFO queue treats a
chat turn's preflight encode and an ingest batch identically. After
gate 2 the ingest pre-encode became ONE job of a few hundred texts
(45 s on CPU before 2b's buckets), so a chat encode submitted mid-batch
waited behind all of it — in any process that serves chat AND ingests
(self-host single-process; the api's inline crystallize). THE FIX: the
lane is a priority queue. INTERACTIVE (the default for every wrapper,
so no caller changed) runs ahead of BULK (opt-in: the ingest batch,
which also splits itself into windowed jobs so a BULK job is short).
FIFO within a class. Still one thread, still one encode at a time.

Callers in async code use these wrappers. Sync code (CLI startup,
scripts, tests) keeps calling encoder.encode() directly — there is no
loop to starve there.

The wrappers take the encoder as an argument (rather than living on a
base class) so both encoder families — SemanticTextEncoder and the
legacy HashTextEncoder — get the same treatment with zero changes to
either, and so the lane stays one process-wide singleton no matter
how many encoder instances exist.
"""
from __future__ import annotations

import asyncio
import enum
import functools
import itertools
import queue
import threading
from typing import Any, Callable, Optional, Sequence

import numpy as np


class Priority(enum.IntEnum):
    """Lane priority. Lower runs first. INTERACTIVE is the default for
    every wrapper; BULK is opted into by callers that encode on nobody's
    behalf in particular (the ingest batch)."""
    INTERACTIVE = 0
    BULK = 1


class _EncoderLane:
    """One daemon thread draining a priority queue of (priority, seq, fn,
    loop, future). Results and exceptions are delivered back onto the
    submitting loop with call_soon_threadsafe; a future whose awaiter was
    cancelled is left alone (its result is simply dropped). The thread is
    named cc-encoder so it is identifiable in a thread dump."""

    def __init__(self) -> None:
        self._q: "queue.PriorityQueue[tuple[int, int, Callable[[], Any], asyncio.AbstractEventLoop, asyncio.Future]]" = queue.PriorityQueue()
        self._seq = itertools.count()          # FIFO within a priority class
        self._thread = threading.Thread(target=self._run, name="cc-encoder", daemon=True)
        self._thread.start()

    def submit(self, fn: Callable[[], Any], priority: Priority) -> "asyncio.Future[Any]":
        loop = asyncio.get_running_loop()
        fut: "asyncio.Future[Any]" = loop.create_future()
        self._q.put((int(priority), next(self._seq), fn, loop, fut))
        return fut

    def _run(self) -> None:
        while True:
            _, _, fn, loop, fut = self._q.get()
            try:
                result = fn()
            except BaseException as exc:      # same contract as ThreadPoolExecutor
                self._deliver(loop, fut, exc=exc)
            else:
                self._deliver(loop, fut, result=result)

    @staticmethod
    def _deliver(
        loop: asyncio.AbstractEventLoop, fut: "asyncio.Future[Any]",
        result: Any = None, exc: Optional[BaseException] = None,
    ) -> None:
        def _set() -> None:
            if fut.cancelled():
                return
            if exc is not None:
                fut.set_exception(exc)
            else:
                fut.set_result(result)
        try:
            loop.call_soon_threadsafe(_set)
        except RuntimeError:
            # The submitting loop is closed (shutdown, test teardown):
            # nobody can be waiting on this future any more.
            pass


_LANE = _EncoderLane()


async def encode_async(
    encoder: Any, text: str, *, priority: Priority = Priority.INTERACTIVE,
) -> np.ndarray:
    """`encoder.encode(text)` off the event loop, serialized on the lane."""
    return await _LANE.submit(functools.partial(encoder.encode, text), priority)


async def encode_native_async(
    encoder: Any, text: str, *, priority: Priority = Priority.INTERACTIVE,
) -> np.ndarray:
    """`encoder.encode_native(text)` off the event loop, serialized on the lane."""
    return await _LANE.submit(functools.partial(encoder.encode_native, text), priority)


async def encode_messages_async(
    encoder: Any,
    messages: Sequence[dict],
    *,
    priority: Priority = Priority.INTERACTIVE,
    **kwargs: Any,
) -> np.ndarray:
    """`encoder.encode_messages(messages, **kwargs)` off the event loop."""
    fn = functools.partial(encoder.encode_messages, messages, **kwargs)
    return await _LANE.submit(fn, priority)


def supports_batch_encode(encoder: Any) -> bool:
    """True when the encoder can produce native vectors for many texts in
    one model call, derive HDC vectors from them by projection, and report
    its length buckets — the three things the ingest pre-encode path
    needs. The semantic encoder does; the legacy hash encoder and most
    test doubles don't, and they take the per-text path unchanged."""
    return (
        callable(getattr(encoder, "encode_native_batch", None))
        and callable(getattr(encoder, "project", None))
        and callable(getattr(encoder, "length_buckets", None))
    )


async def encode_native_batch_async(
    encoder: Any,
    texts: Sequence[str],
    *,
    priority: Priority = Priority.INTERACTIVE,
    window_tokens: Optional[int] = None,
) -> np.ndarray:
    """`encoder.encode_native_batch(texts)` off the event loop (L7a gate 2).

    `window_tokens=None`: ONE lane job for the whole list. With a budget
    (L7a gate 4): the list is cut into windows — inside each 2b length
    band, budget // band texts per window (at least one) — and every
    window is its own lane job at `priority`, so a job never holds the
    lane for more than roughly the budget's worth of encoder compute and
    an INTERACTIVE encode waits at most one window. Windows are submitted
    together (an INTERACTIVE job still jumps them all), awaited together,
    and reassembled so row i is texts[i]. A window of one band is one
    model call inside the encoder (its own bucketing is a no-op on it).
    """
    texts = list(texts)
    if window_tokens is None:
        return await _LANE.submit(
            functools.partial(encoder.encode_native_batch, texts), priority
        )
    budget = max(1, int(window_tokens))
    windows: list[tuple[list[int], "asyncio.Future[Any]"]] = []
    for band, idx in encoder.length_buckets(texts):
        per_job = max(1, budget // band)
        for start in range(0, len(idx), per_job):
            window = idx[start:start + per_job]
            fn = functools.partial(encoder.encode_native_batch, [texts[i] for i in window])
            windows.append((window, _LANE.submit(fn, priority)))
    out = np.zeros((len(texts), encoder.native_dim), dtype=np.float32)
    results = await asyncio.gather(*(fut for _, fut in windows))
    for (window, _), vecs in zip(windows, results):
        out[window] = vecs
    return out


async def run_encoder_bound(
    fn: Any, *args: Any, priority: Priority = Priority.INTERACTIVE, **kwargs: Any,
) -> Any:
    """Run any encoder-bound SYNC callable on the encoder lane.

    For sync helpers that encode internally (e.g. learning's
    `payload_agreement`, crystallizer's `_build_crystal_row`) and are
    called from async code: wrapping the WHOLE helper here moves its
    encodes (and the small math around them) off the event loop in one
    hop, without making the helper itself async — it stays directly
    callable from sync code and tests.
    """
    return await _LANE.submit(functools.partial(fn, *args, **kwargs), priority)
