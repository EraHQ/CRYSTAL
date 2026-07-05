"""Identity-query routing — memory blend Inc 4 (D-MB6).

"Where is X / what file defines X / which module has X" is a navigation
question, not a resemblance one. Vector recall answers it only indirectly
— it finds the chunk that *mentions* X and leans on a provenance header —
whereas a sparse-key scan answers it directly and precisely.

This module detects identity/location queries, extracts the symbol, and
scans the sparse-key registry (facts' prompt_text) for a structured key
that names that symbol — ideally at its SPECIFIC (right) end, which is
exactly what an identity query is: entering the unified wide->specific
key from the specific end. On a confident hit it builds a breadcrumb
("A > B > C" + claim body) injection and returns a RetrievalOutcome so the
chat proxy can skip recall. On ANY miss, ambiguity, or error it returns
None and the caller falls through to the existing recall path (which
already surfaces a provenance header) — so this is purely additive and
cannot regress identity answers that already work.

Built on the precise `list_facts_by_key_prefix` store primitive (the one
the cognition worker's key-scan uses), NOT the verbose NavigationRouter
overview: an identity answer wants one crisp location, not a knowledge
inventory.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

import structlog

from .pipeline import RetrievalOutcome
from .reader import _provenance_header
from .sparse_key import parse_key
from ..execution.text_injection import inject_text_context

if TYPE_CHECKING:
    from ..infrastructure.metadata_store import MetadataStore

logger = structlog.get_logger(__name__)


# How many candidate facts to pull before precise Python-side filtering.
# The scan is already subject_contains-narrowed; this just bounds work
# when a symbol substring happens to be common.
_SCAN_LIMIT = 50

# Max chars of claim body injected alongside the location header. The
# value of an identity answer is the location; the body is supporting
# context, so keep it bounded.
_MAX_BODY_CHARS = 800


# Identity/location query patterns, each capturing the symbol in group 1,
# most-specific first. Deliberately narrow: these must fire ONLY on real
# "where does X live" phrasing, never on resemblance questions ("how does
# X work", "what is X"), so a miss is the common case and recall stays the
# default.
_IDENTITY_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(
        r"^\s*where\s+is\s+(.+?)\s+(?:defined|declared|implemented|located|set)\b",
        re.I,
    ),
    re.compile(
        r"^\s*where\s+(?:is|are|can\s+i\s+find|do\s+i\s+find)\s+(.+?)\s*\??$",
        re.I,
    ),
    re.compile(
        r"^\s*what\s+file\s+(?:is|has|contains|defines)\s+(.+?)\s*(?:\s+in)?\s*\??$",
        re.I,
    ),
    re.compile(
        r"^\s*which\s+(?:file|module|class|function)\s+(?:is|has|contains|defines)\s+(.+?)\s*(?:\s+in)?\s*\??$",
        re.I,
    ),
    re.compile(r"^\s*(?:what|which)\s+(?:file|module)\s+defines\s+(.+?)\s*\??$", re.I),
    re.compile(r"^\s*location\s+of\s+(.+?)\s*\??$", re.I),
)

# Noise to strip off an extracted symbol (locational verbs / articles that
# slipped past the capture group).
_TRAILING_NOISE = re.compile(
    r"\s*\b(?:defined|declared|implemented|located|set|in|the|a|an)\b\s*$", re.I
)
_LEADING_NOISE = re.compile(r"^\s*(?:the|a|an)\s+", re.I)


def is_identity_query(query_text: str) -> Optional[str]:
    """Return the symbol an identity/location query asks about, else None.

    Matches (symbol in parens):
        "where is generate_sparse_key defined"  -> (generate_sparse_key)
        "what file is CrystalReader in"         -> (CrystalReader)
        "location of the drive watcher"         -> (drive watcher)
    Does NOT match (returns None — recall handles these):
        "how does crystallization work"
        "what is a sparse key"
    """
    if not query_text:
        return None
    for pat in _IDENTITY_PATTERNS:
        m = pat.search(query_text)
        if not m:
            continue
        symbol = m.group(1).strip()
        # Trim leading/trailing noise a few times (handles stacked words).
        for _ in range(3):
            stripped = _TRAILING_NOISE.sub("", symbol).strip()
            stripped = _LEADING_NOISE.sub("", stripped).strip()
            if stripped == symbol:
                break
            symbol = stripped
        symbol = symbol.strip(" \t?.\"'`")
        if symbol and len(symbol) >= 2:
            return symbol
    return None


def _literal_match(symbol: str, field_value: str) -> bool:
    """Case-insensitive LITERAL substring test (no LIKE wildcards)."""
    return symbol.lower() in (field_value or "").lower()


async def try_identity_injection(
    *,
    query_text: str,
    customer_id: str,
    store: "MetadataStore",
    messages: list[dict],
) -> Optional[RetrievalOutcome]:
    """Answer an identity/location query from a sparse-key scan, or give up.

    Returns a RetrievalOutcome (recall skipped) on a confident hit, or None
    to fall through to recall. Returns None for routine misses rather than
    raising; the caller also wraps this in try/except as a backstop.
    """
    symbol = is_identity_query(query_text)
    if not symbol:
        return None

    # Precise scan: narrow by substring on the key, then verify literally
    # in Python (the store's LIKE treats '_' as a wildcard, and code
    # symbols are full of underscores).
    facts = await store.list_facts_by_key_prefix(
        customer_id,
        key_prefix="",
        subject_contains=symbol,
        limit=_SCAN_LIMIT,
    )
    if not facts:
        return None

    # Keep only facts whose STRUCTURED key (a path of >= 2 segments)
    # actually names the symbol in some segment; anything else is a
    # coincidental substring.
    candidates: list[tuple] = []
    for f in facts:
        sk = parse_key(getattr(f, "prompt_text", "") or "")
        if sk.depth < 2:
            continue  # not a structured key — can't form a location answer
        if any(_literal_match(symbol, seg) for seg in sk.segments):
            candidates.append((f, sk))

    if not candidates:
        return None

    # Rank: a key whose SPECIFIC (right) end names the symbol — the
    # identity entry point — beats one that only names it mid-path;
    # among specific-end matches, prefer the deeper (more-qualified)
    # path. Stable within the scan's ascending prompt_text order.
    candidates.sort(
        key=lambda fs: (
            0 if _literal_match(symbol, fs[1].specific) else 1,
            -fs[1].depth,
        )
    )
    fact, sk = candidates[0]

    header = _provenance_header(getattr(fact, "prompt_text", "") or "")
    if not header:
        return None  # shouldn't happen for a structured key, but be safe
    body = (getattr(fact, "claim_text", None) or getattr(fact, "answer_value", None) or "").strip()
    if len(body) > _MAX_BODY_CHARS:
        body = body[: _MAX_BODY_CHARS - 1].rstrip() + "\u2026"
    injection = f"{header}\n{body}" if body else header

    new_messages = inject_text_context(
        messages, context_text=injection, voicing="informational"
    )

    crystal_id = getattr(fact, "crystal_id", None)
    logger.info(
        "navigation_dispatch.identity_hit",
        customer_id=customer_id,
        symbol=symbol,
        key=str(sk),
        specific=sk.specific,
        candidates=len(candidates),
        crystal_id=crystal_id,
    )

    return RetrievalOutcome(
        messages=new_messages,
        match_type="high",
        injection_method="text",
        matched_crystal_ids=[crystal_id] if crystal_id else [],
        top_score=1.0,
        injected_text=injection,
    )
