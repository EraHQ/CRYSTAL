"""TextEncoder — the text-space encoder interface.

Every text-space encoder implements this Protocol. The retrieval
pipeline (PromptEncoder, VectorStore, CrystalRouter) depends on this
interface, not on any particular implementation.

IMPLEMENTATIONS
---------------
- `HashTextEncoder` (encoding/prompt_encoder.py): deterministic
  hash-based encoding. Zero deps. Fast. Token-overlap matching.
  The v0 default — stable vocabulary matching is all the GSM8K research
  use case needs.

- `SemanticTextEncoder` (encoding/semantic.py): sentence-transformers
  backend. Optional dep. Produces semantic near-neighbor matches
  ("car" ≈ "automobile"). Required for arbitrary-domain customers.

CHOOSING AN ENCODER
-------------------
The process-wide default is constructed via `build_text_encoder()`
from `CC_TEXT_ENCODER` env var. Customers can't pick their own encoder
yet — a customer bank encoded with one must be queried with the same
one or scores collapse. Per-customer encoding is future work once we
have the operational story for bank re-encoding.

WHY THE SAME DIMENSIONALITY
---------------------------
All implementations produce vectors of the same size (settings.d_hdc,
default 10_000). This is because:
  - `Crystal.summary_vector` is a fixed-schema JSON list of floats.
    Mixing dims breaks VectorStore's stacking.
  - Cosine similarity is invariant to trailing zeros, so a 384-dim
    semantic embedding can be zero-padded to 10k without losing signal.
  - Switching encoders without a schema migration is one of the
    design goals — we don't want to ship 0002_summary_text every time
    someone plugs in a different model.

The d_hdc=10_000 size is wasteful for semantic embeddings (most slots
are zero). Once the encoder story stabilizes we can re-consider the
schema and store native dims per-encoder. That's not on the critical
path today.
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class TextEncoder(Protocol):
    """Encodes text into a fixed-dim numpy vector.

    Implementations must:
      - Produce vectors of shape (d_hdc,) dtype float32.
      - Be deterministic: same input → same output, across processes.
      - Be safe to call from multiple asyncio tasks concurrently
        (no shared mutable state OR properly-guarded state).
      - Handle empty input by returning a zero vector.
    """

    def encode(self, text: str) -> np.ndarray:
        """Encode full text to a unit-norm d_hdc-dim vector."""
        ...

    def encode_messages(
        self,
        messages: Sequence[dict],
        *,
        include_roles: Sequence[str] = ("user",),
    ) -> np.ndarray:
        """Encode an OpenAI-shaped message list.

        By default only `user` messages contribute. `system` is usually
        boilerplate added by the caller, not the actual query. `assistant`
        is prior model output that would bias retrieval.
        """
        ...


@runtime_checkable
class BindCapableEncoder(Protocol):
    """A TextEncoder that ALSO supports bind-storage writes.

    Bind-storage (Phase 1.1+) requires three things beyond the basic
    TextEncoder contract:

    1. `encode_native(text)` returns the pre-projection embedding
       (e.g. raw 768-dim gtr-t5-base output). This is what gets stored
       on `Fact.vector` for codebook cleanup at recall time, and it is
       what the bind-v1 decoder was trained to consume.

    2. `P` exposes the projection matrix (shape `(native_dim, d_hdc)`).
       The recall path's reverse-projection step needs `recovered_hdc @ P.T`
       to land back in native space where the codebook lives. Storage
       paths don't strictly need P, but exposing it consistently keeps
       the encoder self-describing.

    3. `fingerprint()` returns a short stable string that identifies
       this encoder's full storage geometry: model identity, native
       dim, d_hdc, projection seed, and any other parameter that would
       change the recovered-vector distribution at recall time. The
       fingerprint gets stamped onto a Crystal on its first bind-storage
       write; subsequent writes verify it matches; recall paths check it
       before invoking the decoder. Mismatched fingerprints between
       writer and reader produce out-of-distribution recovered vectors
       that bind-v1 silently misdecodes — the fingerprint catches that.

    The hash encoder satisfies the basic TextEncoder Protocol but does
    NOT implement BindCapableEncoder. Per CLAUDE.md Hard Rule 15, hash
    encoder is back-compat only and must not participate in bind-storage.
    The runtime_checkable Protocol gives a type-level signal; the
    runtime check in `add_pair_to_crystal` gives a friendly error message
    at the call site.
    """

    def encode(self, text: str) -> np.ndarray:
        ...

    def encode_native(self, text: str) -> np.ndarray:
        """Encode text to a unit-norm vector in NATIVE dim (pre-projection).

        For SemanticTextEncoder this is the raw gtr-t5-base 768-dim output.
        Used as the codebook entry on Fact.vector and as the decoder input
        at recall time.
        """
        ...

    def encode_messages(
        self,
        messages: Sequence[dict],
        *,
        include_roles: Sequence[str] = ("user",),
    ) -> np.ndarray:
        ...

    @property
    def P(self) -> np.ndarray:
        """Projection matrix of shape (native_dim, d_hdc)."""
        ...

    def fingerprint(self) -> str:
        """Stable string identifying this encoder's storage geometry.

        Two encoders with the same fingerprint produce vectors in the
        same recovered-vector distribution at recall time, so the same
        decoder works for both. Two encoders with different fingerprints
        produce out-of-distribution outputs and must not be mixed in
        one bank.

        Stable across processes (must be), versioned by every parameter
        that affects recovered geometry (model name, native_dim, d_hdc,
        seed, scale, projection convention).
        """
        ...
