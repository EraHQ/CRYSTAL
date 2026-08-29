"""SemanticTextEncoder — sentence-transformers backed encoder, projects to HDC space.

Wraps a pretrained sentence-transformer (gtr-t5-base by default) and lifts
its output into the d_hdc-dim space VectorStore expects, using the same
random-bipolar projection P that KnowledgeCrystal uses for its HDC math.

THE GEOMETRY THIS USES (validated April 2026 spikes)
-----------------------------------------------------
  Text  --(gtr-t5-base)-->  768-dim native embedding
        --(/ ||v||)-->      unit-norm 768
        --(@ P)-->          d_hdc-dim HDC vector
        --(/ ||v||)-->      unit-norm d_hdc

  P is a fixed (768, d_hdc) bipolar ±1 matrix derived from a seed.
  Two processes with the same seed produce the same P; query and bank
  vectors land in the same space; cosine similarity is meaningful.

This is the SAME math as `KnowledgeCrystal.project()` from the research
module. We don't import KnowledgeCrystal directly because that's a
research dependency — instead we re-derive P from the same seed and
keep this encoder self-contained.

CRITICAL: zero-padding is NOT used. Earlier versions of this file padded
the 768-dim native vector with zeros up to d_hdc. That preserves cosine
similarity between two vectors encoded the same way, but it does NOT lift
into HDC space — the bipolar randomness needed for HDC's near-orthogonality
guarantees lives in P, not in the zero pads. Routing scores correlated with
token overlap rather than semantic similarity. Fixed by switching to
P-projection.

WHY gtr-t5-base
---------------
The April 2026 fine-tune work used `sentence-transformers/gtr-t5-base`
(768 native dim) as the encoder for both text-v1 and bind-v1 decoders.
The decoders were trained against gtr-t5-base embeddings. If the encoder
that produces vectors at query time is anything else, the decoder
geometry no longer matches and decoding produces nonsense.

We default to gtr-t5-base because the rest of the pipeline depends on
it. Customers can override with `CC_SEMANTIC_MODEL=...` but should
understand they will need their own decoder fine-tunes if they do.

OPTIONAL DEPENDENCY
-------------------
Requires `sentence-transformers`. Install via `pip install sentence-transformers`
or `pip install 'crystal-cache[embeddings]'`. If missing, the import error
fires at construction time with a clear install instruction.

THREAD / ASYNC SAFETY
---------------------
sentence-transformers models are thread-safe for inference. P is a numpy
array mutated only at construction. encode() is safe to call from any
worker.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..config import settings


# Default to gtr-t5-base — what the April 2026 decoders were trained on.
# Override only if you also retrain the decoders for the new encoder.
DEFAULT_MODEL_NAME = "sentence-transformers/gtr-t5-base"

# Fixed seed for the random bipolar projection P. MUST match the seed used
# by KnowledgeCrystal in the research module (currently 42) so that vectors
# encoded here are interpretable by the same HDC math the research code
# uses. Changing this seed invalidates every existing bank.
PROJECTION_SEED = 42


class SemanticTextEncoder:
    """Sentence-transformers-backed encoder with HDC-space projection.

    Constructor loads the model (~440MB for gtr-t5-base) and builds the
    fixed projection matrix P. Cold start is dominated by the model load
    (~3-5 seconds for gtr-t5-base on CPU, faster on GPU). After load,
    encode() is fast.

    Attributes:
        model_name: HuggingFace id of the sentence-transformer.
        d_hdc: Target dimensionality (matches settings.d_hdc, default 10000).
        native_dim: Native dim of the underlying model (768 for gtr-t5-base).
        P: Bipolar projection matrix of shape (native_dim, d_hdc).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        d_hdc: Optional[int] = None,
        seed: int = PROJECTION_SEED,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "SemanticTextEncoder requires 'sentence-transformers'. "
                "Install with: pip install sentence-transformers"
            ) from e

        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.d_hdc = d_hdc or settings.d_hdc
        self._model = SentenceTransformer(self.model_name)
        # Newer sentence-transformers (>=3.x) renamed this to
        # get_embedding_dimension(). Try the new name first, fall back
        # to the old one for back-compat with pinned older versions.
        if hasattr(self._model, "get_embedding_dimension"):
            self.native_dim = int(self._model.get_embedding_dimension())
        else:
            self.native_dim = int(self._model.get_sentence_embedding_dimension())

        if self.native_dim > self.d_hdc:
            raise ValueError(
                f"model {self.model_name!r} native dim {self.native_dim} "
                f"exceeds d_hdc {self.d_hdc}; pick a smaller model or "
                f"raise d_hdc"
            )

        # Build the bipolar projection matrix. Same construction the
        # research KnowledgeCrystal uses, same seed, same shape — vectors
        # we produce here are interoperable with the research module's
        # bind/unbind operations.
        rng = np.random.RandomState(seed)
        self.P = rng.choice([-1.0, 1.0], size=(self.native_dim, self.d_hdc)).astype(np.float32)
        self._seed = seed

    # -----------------------------------------------------------------
    # Public API — same shape as HashTextEncoder
    # -----------------------------------------------------------------

    def encode(self, text: str) -> np.ndarray:
        """Encode text → unit-norm d_hdc-dim vector via P-projection.

        Steps:
          1. sentence-transformer produces unit-norm native_dim embedding
          2. Project to d_hdc via @ P
          3. Re-normalize (P is bipolar ±1, post-projection norm depends
             on input direction; we want unit-norm output for cosine math)

        Empty input -> zero vector. Cosine vs anything is 0, classifier
        downgrades to 'low' / 'no match'.

        L7a gate 2 (2026-08-28): steps 2-3 live in `project()` so a
        caller holding a native vector (from encode_native or the batch
        path) can derive the HDC vector without a second forward pass.
        encode() == project(encode_native()) by construction.
        """
        if not text or not text.strip():
            return np.zeros(self.d_hdc, dtype=np.float32)
        return self.project(self.encode_native(text))

    def project(self, native: np.ndarray) -> np.ndarray:
        """P-projection + re-normalise: ONE native (unit-norm) vector of
        shape (native_dim,) → HDC (unit-norm) of shape (d_hdc,).

        An all-zero input (the empty-text sentinel) stays all-zero,
        matching encode()'s empty-input contract. Pure numpy — no model
        call — which is why the store and the ingest pre-encode table
        call it directly on the event loop (L7a gate 2, Q2=A / Q3=B):
        ~1 ms at 768x10000, an order of magnitude under the forward
        pass the encoder lane keeps off the loop.
        """
        native = np.asarray(native, dtype=np.float32)
        if native.ndim != 1:
            raise ValueError(
                f"project() takes one vector of shape ({self.native_dim},); "
                f"got shape {native.shape}"
            )
        # Project into d_hdc HDC space. This is the HDC "lift" step —
        # each output dim is a sum of ± input components, distributing
        # the signal across the high-dim space.
        projected = native @ self.P
        # Re-normalize. Without this, dot products between two encoded
        # vectors are not bounded to [-1, 1] and downstream cosine
        # thresholds miscalibrate.
        norm = float(np.linalg.norm(projected))
        if norm > 0.0:
            projected = projected / norm
        return projected.astype(np.float32)

    # Length bucketing (L7a gate 2b, Q1=A, 2026-08-28). sentence-
    # transformers pads every batch to its longest member. A document's
    # 7-13 chunks are all ~512 tokens and never fill a batch of 32, so
    # the first batch used to be those chunks plus 19-25 item texts ALL
    # padded to 512 tokens: 32x512 tokens of encoder compute regardless
    # of how many real chunks there were (measured: 36-49 s per session,
    # flat in the text count). Texts are grouped into power-of-two token
    # bands first and each band is its own model call, so a short text is
    # never padded past ~2x its own length. Tokens are estimated as
    # chars/4 (gtr-t5-base's sentencepiece on English); the estimate
    # only chooses the band, never truncation, so being off is harmless.
    _BUCKET_FLOOR_TOKENS = 32

    def _length_buckets(self, texts: Sequence[str]) -> list[list[int]]:
        """Indices of `texts` grouped by estimated token length into
        power-of-two bands, floored at 32 and capped at the model's
        max_seq_length (everything at/over the cap is truncated to the
        same length anyway). Bands ascend; input order is kept inside a
        band."""
        max_len = int(getattr(self._model, "max_seq_length", None) or 512)
        cap = 1 << (max_len - 1).bit_length()
        bands: dict[int, list[int]] = {}
        for i, t in enumerate(texts):
            est = max(1, len(t) // 4)
            band = 1 << (est - 1).bit_length()
            band = max(self._BUCKET_FLOOR_TOKENS, min(band, cap))
            bands.setdefault(band, []).append(i)
        return [bands[b] for b in sorted(bands)]

    def encode_native_batch(self, texts: Sequence[str]) -> np.ndarray:
        """encode_native for many texts in one model call PER LENGTH
        BUCKET → (n, native_dim).

        L7a gate 2: sentence-transformers batches a list far more
        efficiently than n single calls (length-sorted padding, one
        forward pass per batch_size) — provided the batches are not
        padded up to an outlier; see `_length_buckets`. Empty texts get
        a zero row, same sentinel as encode_native(). Row i corresponds
        to texts[i].
        """
        out = np.zeros((len(texts), self.native_dim), dtype=np.float32)
        live = [i for i, t in enumerate(texts) if t and t.strip()]
        if not live:
            return out
        for bucket in self._length_buckets([texts[i] for i in live]):
            idx = [live[j] for j in bucket]
            vecs = self._model.encode(
                [texts[i] for i in idx],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            out[idx] = np.asarray(vecs, dtype=np.float32).reshape(len(idx), self.native_dim)
        return out

    def encode_messages(
        self,
        messages: Sequence[dict],
        *,
        include_roles: Sequence[str] = ("user",),
        window: int | None = None,
    ) -> np.ndarray:
        """Encode an OpenAI-shaped message list.

        Joins the relevant turns into one text and encodes that.
        Default include_roles=("user",) to avoid biasing retrieval on
        prior assistant output or system boilerplate.

        Phase 1.5.3: `window` limits to the last N messages (after role
        filtering). None = all messages (legacy behavior). When set:
          - Take only the last `window` matching messages.
          - Recency weighting: the most recent turn is duplicated at the
            end of the concatenated text (separated by ``\\n---\\n``). This
            biases the sentence-transformer's attention toward recent
            tokens — cheap and effective without requiring a new encoder
            or a separate weighted-average embedding path.
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
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        chunks.append(part.get("text", ""))
        # Phase 1.5.3: windowed context — take only the last N chunks.
        if window is not None and window > 0:
            chunks = chunks[-window:]
        # Phase 1.5.3: recency weighting. When windowed and we have
        # more than one chunk, duplicate the most recent turn at the
        # end. The sentence-transformer's attention will naturally
        # weight the duplicated tokens higher. Separator distinguishes
        # the context window from the recency emphasis.
        if window is not None and len(chunks) > 1:
            text = "\n---\n".join(chunks) + "\n---\n" + chunks[-1]
        else:
            text = "\n".join(chunks)
        return self.encode(text)

    def encode_native(self, text: str) -> np.ndarray:
        """Encode → native_dim (pre-projection) unit-norm vector.

        Diagnostic helper. The native vector is what the inverter
        decoders (text-v1, bind-v1) expect as input — they were trained
        on raw gtr-t5-base 768-dim embeddings, not P-projected ones.

        For routing / VectorStore we use encode() (projected). For
        decoding via the inverter we use encode_native(). Both call
        the same underlying sentence-transformer; only the post-processing
        differs.
        """
        if not text or not text.strip():
            return np.zeros(self.native_dim, dtype=np.float32)
        native = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        return native

    # -----------------------------------------------------------------
    # Bind-storage geometry self-describe
    # -----------------------------------------------------------------

    def fingerprint(self) -> str:
        """Stable identifier for this encoder's storage geometry.

        Format: ``semantic:<model>/native=<n>/hdc=<n>/seed=<n>``

        Stamped onto Crystal.encoder_fingerprint on first bind-storage
        write. Re-checked on every subsequent write into the same
        crystal and again at recall time before the decoder is
        invoked. A mismatch means the recovered-vector distribution
        will not match what bind-v1 was trained on — the fingerprint
        is the only thing standing between that mismatch and silent
        decoder garbage.

        What the fingerprint covers:
          - Model identity (different sentence-transformer → different
            native embedding distribution).
          - Native dim (d_input in the research module's vocabulary;
            should be 768 for gtr-t5-base).
          - d_hdc (10000 in production; affects projection geometry).
          - Projection seed (42 in production; same seed → same P matrix).

        What it does NOT cover (intentionally):
          - Library versions of sentence-transformers / numpy. Those
            don't change recovered-vector geometry; the math is the
            same. If a library upgrade ever changes outputs, the
            fingerprint should grow a version field then.
          - Decoder identity. Decoders are downstream of storage; a
            fingerprint match guarantees the storage geometry is
            consistent, not that any particular decoder was trained
            against it.
        """
        return (
            f"semantic:{self.model_name}"
            f"/native={self.native_dim}"
            f"/hdc={self.d_hdc}"
            f"/seed={self._seed}"
        )
