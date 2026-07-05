"""Encoding layer — §7 of BUILD_PROPOSAL.md.

Text → 10k-dim vectors. Two encoder families share this module:

  - TextEncoder family (encoder used by the retrieval hot path):
      SemanticTextEncoder     — sentence-transformers backend (DEFAULT after April 2026)
      HashTextEncoder         — LEGACY hash-based encoder, kept for back-compat

For back-compat, `PromptEncoder` remains as an alias for
`HashTextEncoder`. Existing tests that construct `PromptEncoder()`
continue to work unchanged.

CHOICE AT STARTUP
-----------------
`build_text_encoder()` constructs the process-wide encoder from the
`CC_TEXT_ENCODER` setting:
    semantic  → SemanticTextEncoder (DEFAULT, requires sentence-transformers)
    hash      → HashTextEncoder (legacy, zero-deps)

One encoder per process. Different customers CAN'T currently use
different encoders — a bank encoded with one must be queried with the
same one. Per-customer encoding is future work once we have bank
re-encoding tooling.

NOTE: settings is imported LAZILY inside build_text_encoder() rather
than at module scope. Module-level `from ..config import settings`
binds a local reference that doesn't update when tests reassign
`config_module.settings = get_settings()`. By resolving the import
per-call we always read the current value, which is what tests and
the lifespan both expect.
"""
from __future__ import annotations

from .base import TextEncoder
from .executor import encode_async, encode_messages_async, encode_native_async
from .prompt_encoder import HashTextEncoder, PromptEncoder
from .sparse_keys import generate_sparse_key


def build_text_encoder() -> TextEncoder:
    """Construct the process-wide text encoder from settings.

    Returns a TextEncoder. Default is SemanticTextEncoder (gtr-t5-base).
    If CC_TEXT_ENCODER=hash, returns the legacy HashTextEncoder —
    use only for back-compat with banks built before April 2026.

    Raises ImportError if the semantic encoder is requested without
    sentence-transformers installed. That's a startup configuration
    error, not a runtime error — the process should refuse to boot
    rather than silently fall back.
    """
    # Lazy import — see module docstring.
    from ..config import settings

    # Default to semantic if unset (mirrors the config default). We
    # don't fall through to hash silently — that would be a confusing
    # downgrade if the env var is mistyped.
    choice = (settings.text_encoder or "semantic").lower()
    if choice == "semantic":
        from .semantic import SemanticTextEncoder
        return SemanticTextEncoder(
            model_name=settings.semantic_model or None,
        )
    if choice == "hash":
        return HashTextEncoder()
    raise ValueError(
        f"Unknown CC_TEXT_ENCODER={choice!r}. Valid: 'semantic', 'hash'."
    )


__all__ = [
    "TextEncoder",
    "HashTextEncoder",
    "PromptEncoder",  # alias of HashTextEncoder for back-compat
    "build_text_encoder",
    "generate_sparse_key",
    # Async wrappers — the run_in_executor fix; async paths use these
    # instead of calling encoder.encode()/encode_native() directly.
    "encode_async",
    "encode_native_async",
    "encode_messages_async",
]
