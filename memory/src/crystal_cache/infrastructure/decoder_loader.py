"""Decoder loader — vec2text inverter checkpoints loaded once per process.

Loads the April 2026 fine-tuned inverters (text-v1, bind-v1) from
their checkpoint directories under experiments/finetune/. Provides a
single shared instance per process so the FastAPI worker doesn't
reload ~3GB of weights on every request.

WHEN THIS LOADS
---------------
Gated on the CC_ENABLE_DECODER environment variable. When unset (or
not "true"/"1"), `get_decoder_loader()` returns None and synthesis
is skipped — the pipeline degrades to "spread match falls back to
top-1 injection."

This gate exists because:
  - Loading both decoders costs ~3GB GPU memory (or ~4GB CPU)
  - Tests don't need them — synthesis tests skip when unset
  - Dev environments often don't have a GPU
  - The decoder is only useful on SPREAD-decision queries (~5–10%
    of typical traffic), so paying the load cost on a low-traffic
    dev box is wasteful

In production, set CC_ENABLE_DECODER=true at startup. The loader
warms up on the first call (cold start ~5–10 seconds for both
checkpoints on GPU; longer on CPU). After that, decode() is
~50–100ms per call.

OPTIONAL DEPENDENCIES
---------------------
Requires `vec2text` and `torch`. Neither is in the base install:
  pip install vec2text torch

If either is missing and someone tries to construct a DecoderLoader,
we raise ImportError with install instructions. Same pattern as
SemanticTextEncoder.

WHY NOT CHECK BOTH MODELS LAZILY
--------------------------------
We could load text-v1 and bind-v1 on first use rather than at
construction. Decided against: the production answer is always
"both load at startup so the first SPREAD query doesn't pay a
multi-second penalty." Lazy-load risks the first user query
hitting a 10-second cold start. Eat the cost on lifespan, not on
the request hot path.

THREAD / ASYNC SAFETY
---------------------
PyTorch inference is thread-safe (no autograd state mutated). The
vec2text generate() call is reentrant. We don't add locking; if
two requests hit synthesis concurrently, both proceed in parallel
on the same model.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import structlog


logger = structlog.get_logger(__name__)


# Default checkpoint locations. The eval_inverter spike used these
# exact paths. Production would point at a hash-pinned blob store
# (S3, GCS) but for now we read from the local checkpoint dir.
DEFAULT_TEXT_V1_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "experiments" / "finetune" / "checkpoints-text-v1" / "final"
)
DEFAULT_BIND_V1_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "experiments" / "finetune" / "checkpoints-bind-v1" / "final"
)

# Generation hyperparams. Match what eval_inverter.py used so synthesis
# outputs match what we measured during the bind-v1 eval.
DEFAULT_MAX_LENGTH = 96
DEFAULT_NUM_BEAMS = 4


def is_decoder_enabled() -> bool:
    """Read the env flag once. Returns True only for explicit opt-in."""
    val = (os.environ.get("CC_ENABLE_DECODER") or "").strip().lower()
    return val in ("1", "true", "yes", "on")


class DecoderLoader:
    """Loads + holds text-v1 and bind-v1 inverters for the lifetime of the
    process. Construct once; call decode_text() and decode_bind() per request.

    Attributes:
        device: "cuda" or "cpu", picked at construction time.
        text_v1: the text-vector inverter (or None if checkpoint missing).
        bind_v1: the bind-vector inverter (or None if checkpoint missing).
    """

    def __init__(
        self,
        text_v1_dir: Optional[Path] = None,
        bind_v1_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> None:
        # Fail fast on missing deps. Better than blowing up halfway
        # through model load with a confusing trace.
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "DecoderLoader requires 'torch'. "
                "Install with: pip install torch"
            ) from e

        # vec2text 0.0.13 imports `resource` at module load on some
        # submodules. Stub it on Windows before importing vec2text or
        # the import itself dies. Same trick eval_inverter.py uses.
        self._stub_resource_on_windows()

        try:
            import vec2text  # noqa: F401  (triggers model registration)
            from vec2text.models import InversionModel
        except ImportError as e:
            raise ImportError(
                "DecoderLoader requires 'vec2text'. "
                "Install with: pip install vec2text"
            ) from e

        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        text_dir = Path(text_v1_dir) if text_v1_dir else DEFAULT_TEXT_V1_DIR
        bind_dir = Path(bind_v1_dir) if bind_v1_dir else DEFAULT_BIND_V1_DIR

        # Each model is optional. Synthesis only needs bind-v1 today;
        # text-v1 is reserved for future direct-decode-from-recovered-vector
        # use cases. If either is missing we log loudly and continue.
        self.text_v1 = self._maybe_load(InversionModel, text_dir, "text-v1")
        self.bind_v1 = self._maybe_load(InversionModel, bind_dir, "bind-v1")

        if self.text_v1 is None and self.bind_v1 is None:
            logger.warning(
                "decoder_loader.no_checkpoints_found",
                text_v1_dir=str(text_dir),
                bind_v1_dir=str(bind_dir),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decode_text(self, embedding: np.ndarray) -> Optional[str]:
        """Decode a single 768-dim native embedding via text-v1.

        Returns None if text-v1 isn't loaded. Caller should treat None
        as "synthesis unavailable" and fall back to non-decoder paths.
        """
        if self.text_v1 is None:
            return None
        return self._decode(self.text_v1, embedding)

    def decode_bind(self, embedding: np.ndarray) -> Optional[str]:
        """Decode a single 768-dim native embedding via bind-v1.

        Bind-v1 is the right model for synthesis vectors (output of
        bind(proj_A, proj_B) reverse-projected). text-v1 stalls in
        loops on those inputs — see Finding 12.
        """
        if self.bind_v1 is None:
            return None
        return self._decode(self.bind_v1, embedding)

    @property
    def available(self) -> bool:
        """True if at least one decoder is loaded. Used by callers
        that want to short-circuit when decoders aren't around."""
        return self.text_v1 is not None or self.bind_v1 is not None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _maybe_load(self, InversionModel, ckpt_dir: Path, tag: str):
        """Try to load a checkpoint. Return None and log if missing.

        We don't raise on missing checkpoints because partial setups
        should still boot — operators may have only one model trained
        in a dev environment, and the synthesis path should degrade
        gracefully rather than crashing the whole FastAPI worker.

        VERSION NOTE: vec2text 0.0.13 is compatible with transformers
        4.x but NOT 5.x. transformers 5.x's `from_pretrained` introduced
        a meta-device check that rejects vec2text's nested loader. The
        venv pins transformers 4.55.4 (see requirements / dist-info).
        If `python -c "import transformers; print(transformers.__file__)"`
        ever points OUTSIDE the venv (e.g. user-site), the load will
        fail with a misleading "meta device context manager" error —
        the actual fix is to ensure the venv's transformers shadows
        any user-site install. See CLAUDE.md Rule 13 for the cautionary
        tale.
        """
        if not ckpt_dir.is_dir():
            logger.info(
                "decoder_loader.checkpoint_absent",
                tag=tag, path=str(ckpt_dir),
            )
            return None
        try:
            logger.info(
                "decoder_loader.loading",
                tag=tag,
                path=str(ckpt_dir),
            )
            model = InversionModel.from_pretrained(str(ckpt_dir))
            model = model.to(self.device)
            model.eval()
            logger.info(
                "decoder_loader.loaded", tag=tag, device=self.device,
            )
            return model
        except Exception as e:
            import traceback
            logger.error(
                "decoder_loader.load_failed",
                tag=tag,
                path=str(ckpt_dir),
                error_type=type(e).__name__,
                error_repr=repr(e),
                traceback=traceback.format_exc(),
            )
            return None

    def _decode(self, model, embedding: np.ndarray) -> str:
        """Run a single-pass decode with beam=4. Same generation kwargs
        as eval_inverter.py — outputs match what we measured against
        the validation set."""
        import torch

        v = np.asarray(embedding, dtype=np.float32)
        # The decoder was trained on unit-norm gtr-t5-base outputs.
        # Renormalize defensively in case the caller handed us a
        # post-bind-and-reverse-project vector that drifted from unit norm.
        norm = float(np.linalg.norm(v))
        if norm > 0.0:
            v = v / norm

        emb = torch.from_numpy(v).unsqueeze(0).to(self.device)
        # vec2text's generate() requires these placeholder inputs even
        # when we hand it frozen_embeddings. The values don't matter;
        # the model only uses frozen_embeddings.
        dummy_ids = torch.zeros((1, 1), dtype=torch.long).to(self.device)
        dummy_mask = torch.zeros((1, 1), dtype=torch.long).to(self.device)

        with torch.no_grad():
            output_ids = model.generate(
                inputs={
                    "embedder_input_ids": dummy_ids,
                    "embedder_attention_mask": dummy_mask,
                    "frozen_embeddings": emb,
                },
                generation_kwargs={
                    "max_length": DEFAULT_MAX_LENGTH,
                    "num_beams": DEFAULT_NUM_BEAMS,
                    "do_sample": False,
                    "early_stopping": True,
                },
            )

        if output_ids.dim() == 2:
            output_ids = output_ids[0]
        return model.tokenizer.decode(
            output_ids.tolist(), skip_special_tokens=True
        )

    @staticmethod
    def _stub_resource_on_windows() -> None:
        """vec2text 0.0.13 imports `resource` (POSIX-only) on import.
        Stub it before importing vec2text on Windows."""
        if sys.platform.startswith("win") and "resource" not in sys.modules:
            import types
            stub = types.ModuleType("resource")
            stub.RLIMIT_AS = 0  # type: ignore[attr-defined]
            stub.RLIMIT_DATA = 0  # type: ignore[attr-defined]
            stub.RLIM_INFINITY = -1  # type: ignore[attr-defined]
            stub.getrlimit = lambda which: (-1, -1)  # type: ignore[attr-defined]
            stub.setrlimit = lambda which, limits: None  # type: ignore[attr-defined]
            sys.modules["resource"] = stub


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_loader: Optional[DecoderLoader] = None


def set_decoder_loader(loader: Optional[DecoderLoader]) -> None:
    """Install (or clear) the process-wide DecoderLoader. Called from
    the FastAPI lifespan."""
    global _loader
    _loader = loader


def get_decoder_loader() -> Optional[DecoderLoader]:
    """Return the active loader, or None if decoders are disabled.

    Synthesis-using code should treat None as "skip synthesis, fall
    through to top-1 only" rather than raising. The pipeline always
    has a non-decoder path that works.
    """
    return _loader
