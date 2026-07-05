"""Unified sparse key generation — derive a wide->specific PATH from free text.

See docs/UNIFIED_SPARSE_KEY.md. A sparse key is an ordered path of segments
running WIDE -> SPECIFIC (broad category first, exact subject last),
'|'-delimited, of variable length. This module derives such a path from a
piece of free text (an SDK /v1/store fact, an imported record, a cached
solution, a failure reflection) with a capped LLM call that returns an
ordered segment list; retrieval.sparse_key.format_key owns the join and the
sanitation (it drops the '|' delimiter, collapses whitespace, caps lengths,
and drops empties), so freeform pipe junk can never leak into a key.

HISTORY: this module previously emitted a single 3-8 word semantic phrase
(a depth-1 key). The document pipeline already builds multi-segment paths
from document structure; this brings the same path shape to the SDK
store / learn / import write paths, which until now collapsed to one
segment. Depth-1 keys remain valid (the unified-key spec tolerates them),
so existing keys are unaffected.

CALL STYLE: a plain chat completion (via the provider-neutral crystal_cache.llm
seam) that asks for a JSON array of segments, parsed tolerantly — the same
style document_pipeline uses for its proven extraction path (not
structured-output `output_config`).

KEYLESS / ON FAILURE: degrades to a depth-1 key (the first 8 words as one
sanitized segment) so storage and meaning-based retrieval still work with no
LLM. Multi-segment paths require an LLM provider key (whichever provider
crystal_cache.llm is configured for); without one you get the depth-1 fallback.

Cost: ~one small-tier model call per distinct text (LRU-cached by text hash).
"""
from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# System prompt: produce an ordered, general->specific segment path for ONE
# piece of knowledge. Mirrors the per-item "segments" instruction the
# document-extraction prompt uses, narrowed to a single fact and a bare
# JSON-array reply.
SPARSE_KEY_SEGMENTS_SYSTEM = (
    "You build a retrieval KEY for a single piece of knowledge.\n\n"
    "Output an ordered list of segments naming WHERE this knowledge sits in a "
    "knowledge hierarchy, from GENERAL (first) to SPECIFIC (last). Use 2-5 "
    "segments. Each segment is 1-4 plain words: a broad category first, the "
    "exact subject last. No '|' character, no punctuation inside a segment.\n\n"
    "Examples:\n"
    "  Text: We use PostgreSQL 16 for all production services.\n"
    '  ["Infrastructure", "Database", "Production", "PostgreSQL 16"]\n'
    "  Text: PTO accrues at 1.5 days per month after the first year.\n"
    '  ["Policy", "Employee Handbook", "PTO", "Accrual Rate"]\n\n'
    "Return ONLY a JSON array of strings, no markdown, no explanation."
)

# Token cap. A JSON array of up to five short segments fits comfortably;
# headroom avoids truncating the array (which would fail the parse and drop
# to the depth-1 fallback).
SPARSE_KEY_MAX_TOKENS = 128

# Input truncation. The path captures where the knowledge sits, not every
# detail; 500 chars is enough for the model to place it.
MAX_INPUT_CHARS = 500

def _text_hash(text: str) -> str:
    """SHA256 hash of text for cache keying."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _parse_segment_array(text: str) -> list[str]:
    """Tolerantly parse a JSON array of segment strings from the model reply.

    Accepts a bare array, a ```json fenced array, or an array embedded in
    surrounding prose. Coerces each element to a stripped string and drops
    empties. Returns [] if nothing parseable is found (caller falls back).
    """
    def _coerce(parsed) -> list[str]:
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
        return []

    try:
        out = _coerce(json.loads(text))
        if out:
            return out
    except json.JSONDecodeError:
        pass

    if "```" in text:
        inner = text.split("```")[1]
        if inner.startswith("json"):
            inner = inner[4:]
        try:
            out = _coerce(json.loads(inner.strip()))
            if out:
                return out
        except json.JSONDecodeError:
            pass

    start, end = text.find("["), text.rfind("]")
    if start >= 0 and end > start:
        try:
            out = _coerce(json.loads(text[start:end + 1]))
            if out:
                return out
        except json.JSONDecodeError:
            pass

    return []


# LRU cache: same text -> same segment list. Avoids redundant API calls when
# the same text recurs (e.g. batch import). Cache size 4096 keeps memory
# bounded. A tuple is returned so the cached value is immutable.
@lru_cache(maxsize=4096)
def _cached_generate(text_hash: str, truncated_text: str) -> tuple[str, ...]:
    """Generate the ordered segment list for `truncated_text` (cached by hash).

    Routes through the provider-neutral LLM seam (crystal_cache.llm): the
    configured provider's small-tier model returns the JSON segment array.
    """
    from ..llm import get_llm_client

    text = get_llm_client().complete(
        system=SPARSE_KEY_SEGMENTS_SYSTEM,
        messages=[{"role": "user", "content": truncated_text}],
        max_tokens=SPARSE_KEY_MAX_TOKENS,
        temperature=0.0,
        tier="small",
    )
    return tuple(_parse_segment_array(text))


def generate_sparse_key(
    text: str,
    *,
    fallback: bool = True,
) -> str:
    """Derive a unified sparse key — a wide->specific PATH — from free text.

    Asks the model for an ordered general->specific segment list and joins it
    through retrieval.sparse_key.format_key (sanitize + drop '|' + cap +
    drop empties). The result is a multi-segment path like
    'Infrastructure|Database|Production|PostgreSQL 16'.

    Pass the richest text available — for a stored fact, the key AND the value
    together produce a better path than the key alone.

    Args:
        text: the knowledge to key.
        fallback: if True, on model failure return a depth-1 key (the first 8
                  words as one sanitized segment). If False, re-raise.

    Returns:
        A '|'-delimited sparse key. Multi-segment when the model succeeds; a
        single segment (depth-1) on the keyless/failure fallback or when the
        model returns nothing usable.

    Raises:
        Exception: if fallback=False and the model call fails.

    Examples:
        >>> generate_sparse_key("We use PostgreSQL 16 for all production services.")
        'Infrastructure|Database|Production|PostgreSQL 16'
    """
    truncated = text[:MAX_INPUT_CHARS] if len(text) > MAX_INPUT_CHARS else text
    text_h = _text_hash(truncated)

    # Lazy import — a module-level import would cycle through
    # retrieval/__init__ -> pipeline -> encoding/__init__ -> this module.
    from ..retrieval.sparse_key import format_key

    try:
        segments = _cached_generate(text_h, truncated)
    except Exception as e:
        if not fallback:
            raise
        # Fallback: first 8 words of the original text as one clean segment.
        # Better than nothing — the embedding still captures SOME topic, and
        # the key stays a valid depth-1 unified key.
        logger.warning("Sparse key generation failed, using fallback: %s", e)
        return format_key(" ".join(text.split()[:8]))

    key = format_key(list(segments))
    if not key:
        # Model returned nothing usable — degrade to a depth-1 key.
        key = format_key(" ".join(text.split()[:8]))
    return key


def clear_cache() -> None:
    """Clear the sparse key LRU cache.

    Call after changing the model or during test teardown.
    """
    _cached_generate.cache_clear()
