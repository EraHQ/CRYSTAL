"""Prompt encoder — LEGACY hash-based encoder. NOT the default after April 2026.

This is a deterministic bag-of-token-hashes encoder. Used as the v0
default before the spike work in April 2026 validated semantic encoding.
Kept for:
  - Tests that pinned encoded vectors (changing the encoder would
    invalidate them and the tests run faster than re-encoding everything).
  - Banks imported before the semantic switch — those vectors live in
    hash-encoder space and must be queried with this encoder.
  - Environments that explicitly choose CC_TEXT_ENCODER=hash for
    deterministic, zero-dep operation (e.g. air-gapped CI).

For new banks, use SemanticTextEncoder (the new default). It produces
actual semantic near-neighbor matches and lives in the same geometry
as the April 2026 decoder fine-tunes (text-v1, bind-v1) that depend on
gtr-t5-base embeddings.

WHY THIS WAS WRONG AS A DEFAULT
-------------------------------
Hash encoder cosine similarity correlates with token overlap, not
semantic similarity. "car" and "automobile" hash to unrelated vectors,
so paraphrase queries miss every time. CLAUDE.md Finding 9 documents
why this matters: the spike work needed real embeddings to even be
worth running, and the rest of the architecture depends on that.

HOW IT WORKS (kept here for the back-compat case)
-------------------------------------------------
  1. Tokenize input text (whitespace + lowercase).
  2. For each token, derive a deterministic 10k-dim vector via
     SHA-256 hashing (token → indices, signs).
  3. Sum all token vectors. Normalize.
  4. Return.

The ENCODER_SEED constant pins the hash so different processes/
machines produce identical vectors. Don't change it without
invalidating every existing hash-bank.
"""
from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np


# Default dimension matches the Lili research bank and the Crystal.summary_vector
# schema comment ("10k-dim JSON; pgvector.Vector(10000) in production").
DEFAULT_D_HDC = 10_000

# How many dimensions each token activates. Trade-off:
#  - Too few (1): vectors are sparse, cosine similarity is brittle.
#  - Too many (thousands): every token lights up the whole vector; similarity
#    collapses toward 1.0 for all pairs.
# 64 is the research-proven sweet spot for 10k-dim HDC with ~50-token queries.
DEFAULT_COMPONENTS_PER_TOKEN = 64

# Fixed seed for the hash. Change only if you want to invalidate every bank
# in existence — encoded vectors will become incompatible.
ENCODER_SEED = b"crystal_cache.v1"


# Whitespace + strip ASCII punctuation. Cheap and deterministic.
# We deliberately don't split on apostrophes (keeps "don't" as one token).
_TOKEN_RE = re.compile(r"[a-z0-9']+")


class HashTextEncoder:
    """Hash-based HDC text encoder. Thread-safe and stateless after construction.

    Implements the TextEncoder Protocol in `encoding.base`. The old
    name `PromptEncoder` is kept as an alias at the bottom of this
    module for back-compat.
    """

    def __init__(
        self,
        d_hdc: int = DEFAULT_D_HDC,
        components_per_token: int = DEFAULT_COMPONENTS_PER_TOKEN,
    ) -> None:
        if d_hdc <= 0:
            raise ValueError(f"d_hdc must be positive, got {d_hdc}")
        if components_per_token <= 0 or components_per_token > d_hdc:
            raise ValueError(
                f"components_per_token must be in (0, d_hdc]; "
                f"got {components_per_token} with d_hdc={d_hdc}"
            )
        self.d_hdc = d_hdc
        self.components_per_token = components_per_token

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def encode(self, text: str) -> np.ndarray:
        """Encode full text to a unit-norm 10k-dim vector."""
        tokens = self._tokenize(text)
        if not tokens:
            # Empty input → zero vector. Cosine similarity vs anything is 0.
            return np.zeros(self.d_hdc, dtype=np.float32)
        return self._encode_tokens(tokens)

    def encode_messages(
        self,
        messages: Sequence[dict],
        *,
        include_roles: Sequence[str] = ("user",),
        window: int | None = None,
    ) -> np.ndarray:
        """Encode an OpenAI-shaped message list.

        By default only `user` messages contribute. `system` and `assistant`
        are filtered out because:
          - system is usually a boilerplate prompt the caller added, not the
            actual question we want to route on
          - assistant is prior model output; using it for retrieval biases
            toward whatever the model already said

        Pass `include_roles=("user", "system")` if you want system included.

        Phase 1.5.3: `window` limits to the last N messages (after role
        filtering). None = all messages (legacy behavior). When set, only
        the last `window` matching messages contribute to the encoding.
        """
        allowed = set(include_roles)
        chunks: list[str] = []
        for m in messages:
            if m.get("role") not in allowed:
                continue
            content = m.get("content", "")
            if isinstance(content, str):
                chunks.append(content)
            elif isinstance(content, list):
                # OpenAI structured content: flatten text parts
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        chunks.append(part.get("text", ""))
        # Phase 1.5.3: windowed context — take only the last N chunks.
        if window is not None and window > 0:
            chunks = chunks[-window:]
        return self.encode("\n".join(chunks))

    def encode_sentences(self, text: str) -> list[np.ndarray]:
        """Return one unit-norm vector per sentence. Cheap sentence split
        on punctuation; good enough for the "sentence-level vectors" the
        research path wanted."""
        # Split on sentence-ending punctuation followed by whitespace or end-of-string.
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [self.encode(s) for s in sentences if s]

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    def _encode_tokens(self, tokens: Sequence[str]) -> np.ndarray:
        vec = np.zeros(self.d_hdc, dtype=np.float32)
        for tok in tokens:
            indices, signs = self._token_to_components(tok)
            # `np.add.at` handles duplicate indices correctly (rare but possible
            # on dimension collisions across components_per_token slots).
            np.add.at(vec, indices, signs)

        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def _token_to_components(self, token: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (indices, signs) for a single token.

        Uses SHA-256 as a deterministic PRNG. We draw enough bytes to cover
        both indices (4 bytes each, mod d_hdc) and signs (1 bit each).
        """
        # 4 bytes per index + 1 bit per sign (packed as 1 byte for simplicity)
        bytes_needed = self.components_per_token * 5

        buf = bytearray()
        counter = 0
        while len(buf) < bytes_needed:
            h = hashlib.sha256()
            h.update(ENCODER_SEED)
            h.update(token.encode("utf-8"))
            h.update(counter.to_bytes(4, "little"))
            buf.extend(h.digest())
            counter += 1

        raw = bytes(buf[:bytes_needed])

        # Split raw: first 4*k bytes for indices, next k bytes for signs
        k = self.components_per_token
        idx_raw = np.frombuffer(raw[: 4 * k], dtype=np.uint32)
        sign_raw = np.frombuffer(raw[4 * k : 5 * k], dtype=np.uint8)

        indices = (idx_raw % self.d_hdc).astype(np.int64)
        signs = np.where(sign_raw & 1, 1.0, -1.0).astype(np.float32)

        return indices, signs


# Back-compat alias. Existing code (tests, scripts, pipeline) imports
# `PromptEncoder` from this module or from `crystal_cache.encoding`.
# We keep the name pointing at HashTextEncoder so nothing breaks.
PromptEncoder = HashTextEncoder
