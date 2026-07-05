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
        """
        if not text or not text.strip():
            return np.zeros(self.d_hdc, dtype=np.float32)

        # normalize_embeddings=True returns L2-normalized native vectors.
        native = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        # native shape: (native_dim,), unit-norm.

        # Project into d_hdc HDC space. This is the HDC "lift" step —
        # each output dim is a sum of ± input components, distributing
        # the signal across the high-dim space.
        projected = native @ self.P  # (d_hdc,)

        # Re-normalize. Without this, dot products between two encoded
        # vectors are not bounded to [-1, 1] and downstream cosine
        # thresholds miscalibrate.
        norm = float(np.linalg.norm(projected))
        if norm > 0.0:
            projected = projected / norm
        return projected.astype(np.float32)

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
