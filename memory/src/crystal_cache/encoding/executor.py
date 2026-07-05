"""Async encoder execution — the run_in_executor fix (2026-06-11).

THE PROBLEM (documented in the v1 context doc as "ISSUE: encoder
blocks event loop"): SemanticTextEncoder.encode() is CPU-bound
(sentence-transformer forward pass, ~10-50ms on CPU). Every async path
that called it directly — retrieval per request, crystallization per
chunk, learning per fact — froze the event loop for that long, which
violates the first core principle: no component starves another. One
document being crystallized could stutter every concurrent API request.

THE FIX: a dedicated single-thread executor for all encoder work, plus
three awaitable wrappers. Why max_workers=1, deliberately:

  * One encode at a time keeps "the encoder is a shared resource"
    TRUE in the scheduler instead of aspirational in a comment —
    concurrent callers queue here, visibly, instead of contending for
    torch threads invisibly.
  * sentence-transformers is thread-safe for inference, but parallel
    forward passes on CPU fight over the same cores and ALL get slower.
    Serialized, each encode finishes at full speed.
  * The event loop never blocks either way — that's the point.

Callers in async code use these wrappers. Sync code (CLI startup,
scripts, tests) keeps calling encoder.encode() directly — there is no
loop to starve there.

The wrappers take the encoder as an argument (rather than living on a
base class) so both encoder families — SemanticTextEncoder and the
legacy HashTextEncoder — get the same treatment with zero changes to
either, and so the executor stays one process-wide singleton no matter
how many encoder instances exist.
"""
from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import numpy as np

# One process-wide encoder lane. Named so it's identifiable in thread
# dumps when someone asks "what is this thread doing".
_ENCODER_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cc-encoder")


async def encode_async(encoder: Any, text: str) -> np.ndarray:
    """`encoder.encode(text)` off the event loop, serialized."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_ENCODER_EXECUTOR, encoder.encode, text)


async def encode_native_async(encoder: Any, text: str) -> np.ndarray:
    """`encoder.encode_native(text)` off the event loop, serialized."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_ENCODER_EXECUTOR, encoder.encode_native, text)


async def encode_messages_async(
    encoder: Any,
    messages: Sequence[dict],
    **kwargs: Any,
) -> np.ndarray:
    """`encoder.encode_messages(messages, **kwargs)` off the event loop."""
    loop = asyncio.get_running_loop()
    fn = functools.partial(encoder.encode_messages, messages, **kwargs)
    return await loop.run_in_executor(_ENCODER_EXECUTOR, fn)


async def run_encoder_bound(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Run any encoder-bound SYNC callable on the encoder lane.

    For sync helpers that encode internally (e.g. learning's
    `payload_agreement`, crystallizer's `_build_crystal_row`) and are
    called from async code: wrapping the WHOLE helper here moves its
    encodes (and the small math around them) off the event loop in one
    hop, without making the helper itself async — it stays directly
    callable from sync code and tests.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _ENCODER_EXECUTOR, functools.partial(fn, *args, **kwargs)
    )
