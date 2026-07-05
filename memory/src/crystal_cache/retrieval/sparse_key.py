"""Unified sparse key — bidirectional, variable-length.

See docs/UNIFIED_SPARSE_KEY.md. A sparse key is an ordered path of
segments running WIDE -> SPECIFIC, left to right, of unbounded length:

    Film | Corporate Mistletoe | Script | Scene 5 | Props | Mistletoe
    └ wide (broadest)                                       specific ┘

The STRUCTURE is fixed (general->specific ordering, delimited by '|',
matchable at any segment). The LENGTH is not — a fact is keyed as deep
as its knowledge warrants. Segments are plain values; generality is
encoded by position (leftmost = widest), not by labeled role slots.

This supersedes the two prior key systems (the 3-8 word semantic key and
the fixed Source|Locator|Subject|Domain key). There are no role-named
accessors and no fixed-arity builders: generation calls format_key with
a context-appropriate, ordered segment list.

Pure module — stdlib only, no SQL, no async, no DB.

The two query shapes enter from opposite ends of the same key:
  * identity   -> match the SPECIFIC (right) end, traverse left.
  * resemblance-> match the WIDE (left) end, fan right.
  * anywhere   -> match any shared segment, traverse either way.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional, Union

DELIMITER = "|"

# Guards (not semantics). A key must have >= 1 segment.
MAX_SEGMENTS = 12
MAX_SEGMENT_CHARS = 64
MAX_KEY_CHARS = 256

_WS_RE = re.compile(r"\s+")
_VAGUE_TERMS = ("stuff", "things", "misc", "info", "various", "general", "other")

SegmentsInput = Union[str, Iterable[str]]


def _sanitize_segment(seg: str) -> str:
    """One segment: drop the delimiter, collapse whitespace, strip, cap length."""
    s = str(seg).replace(DELIMITER, " ")
    s = _WS_RE.sub(" ", s).strip()
    if len(s) > MAX_SEGMENT_CHARS:
        s = s[:MAX_SEGMENT_CHARS].strip()
    return s


def _coerce_segments(segments) -> list[str]:
    """Accept format_key('a','b') OR format_key(['a','b']) OR format_key('a|b')."""
    if len(segments) == 1 and isinstance(segments[0], (list, tuple)):
        raw = list(segments[0])
    elif len(segments) == 1 and isinstance(segments[0], str) and DELIMITER in segments[0]:
        raw = segments[0].split(DELIMITER)
    else:
        raw = list(segments)
    out = [_sanitize_segment(s) for s in raw]
    return [s for s in out if s]  # drop empties


@dataclass(frozen=True)
class SparseKey:
    """An ordered path of segments, wide (first) -> specific (last)."""
    segments: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Allow construction from any iterable; freeze to a tuple.
        object.__setattr__(self, "segments", tuple(self.segments))

    @property
    def wide(self) -> str:
        """The broadest segment (or '' for an empty key)."""
        return self.segments[0] if self.segments else ""

    @property
    def specific(self) -> str:
        """The most specific segment (or '' for an empty key)."""
        return self.segments[-1] if self.segments else ""

    @property
    def depth(self) -> int:
        return len(self.segments)

    @property
    def is_empty(self) -> bool:
        return not self.segments

    def lowered(self) -> tuple[str, ...]:
        return tuple(s.lower() for s in self.segments)

    def __str__(self) -> str:
        return DELIMITER.join(self.segments)


def format_key(*segments: SegmentsInput) -> str:
    """Build a sparse key string from an ordered, wide->specific segment list.

    Sanitizes each segment (drops '|', collapses whitespace, caps length),
    drops empties, and joins with the delimiter. Accepts varargs, a single
    list/tuple, or a pre-joined 'a|b|c' string.
    """
    return DELIMITER.join(_coerce_segments(segments))


def parse_key(key: str) -> SparseKey:
    """Parse a key string into a SparseKey (split, sanitize, drop empties)."""
    if not key:
        return SparseKey(())
    parts = [_sanitize_segment(p) for p in str(key).split(DELIMITER)]
    return SparseKey(tuple(p for p in parts if p))


def validate_key(key: str) -> tuple[bool, list[str]]:
    """Validate a key. Returns (is_valid, issues).

    is_valid is False only for structural problems (no segments, over the
    guards). Vague-term hits are quality warnings that don't fail validity.
    """
    issues: list[str] = []
    sk = parse_key(key)

    if sk.depth == 0:
        issues.append("Empty key (no segments)")
    if sk.depth > MAX_SEGMENTS:
        issues.append(f"Too many segments ({sk.depth}, max {MAX_SEGMENTS})")
    if len(str(sk)) > MAX_KEY_CHARS:
        issues.append(f"Key too long ({len(str(sk))} chars, max {MAX_KEY_CHARS})")
    for seg in sk.segments:
        if len(seg) > MAX_SEGMENT_CHARS:
            issues.append(f"Segment '{seg[:20]}...' over {MAX_SEGMENT_CHARS} chars")

    low = str(sk).lower()
    for term in _VAGUE_TERMS:
        if re.search(rf"\b{term}\b", low):
            issues.append(f"Key contains vague term '{term}'")

    structural = [i for i in issues if "Empty" in i or "Too many" in i or "too long" in i]
    return (len(structural) == 0 and sk.depth >= 1), issues


def is_structured_key(key: str) -> bool:
    """A key is structured if it is a path of >= 2 segments (wide + specific)."""
    return parse_key(key).depth >= 2


# ---------------------------------------------------------------------------
# Traversal / Index-of-Indexes (case-insensitive matching)
# ---------------------------------------------------------------------------

def _as_key(k: Union[str, SparseKey]) -> SparseKey:
    return k if isinstance(k, SparseKey) else parse_key(k)


def _seglist(x: SegmentsInput) -> list[str]:
    if isinstance(x, str):
        return _coerce_segments((x,))
    return _coerce_segments((list(x),))


def common_prefix(a: Union[str, SparseKey], b: Union[str, SparseKey]) -> int:
    """Number of shared WIDE segments (how much of the general path matches)."""
    la, lb = _as_key(a).lowered(), _as_key(b).lowered()
    n = 0
    for x, y in zip(la, lb):
        if x != y:
            break
        n += 1
    return n


def common_suffix(a: Union[str, SparseKey], b: Union[str, SparseKey]) -> int:
    """Number of shared SPECIFIC segments (how much of the precise tail matches)."""
    la, lb = _as_key(a).lowered(), _as_key(b).lowered()
    n = 0
    for x, y in zip(reversed(la), reversed(lb)):
        if x != y:
            break
        n += 1
    return n


def contains_segment(key: Union[str, SparseKey], segment: str) -> bool:
    """True if `segment` appears anywhere in the key (enter-anywhere)."""
    return _sanitize_segment(segment).lower() in _as_key(key).lowered()


def index_of(key: Union[str, SparseKey], segment: str) -> Optional[int]:
    low = _as_key(key).lowered()
    target = _sanitize_segment(segment).lower()
    return low.index(target) if target in low else None


def matches(query: Union[str, SparseKey], key: Union[str, SparseKey], *, mode: str = "anywhere") -> bool:
    """Does `key` match `query` under the given entry mode?

    identity    -> shares the specific (right) end: common_suffix >= 1.
    resemblance -> shares the wide (left) end: common_prefix >= 1.
    anywhere    -> shares at least one segment at any position.
    """
    q, k = _as_key(query), _as_key(key)
    if q.is_empty or k.is_empty:
        return False
    if mode == "identity":
        return common_suffix(q, k) >= 1
    if mode == "resemblance":
        return common_prefix(q, k) >= 1
    if mode == "anywhere":
        return bool(set(q.lowered()) & set(k.lowered()))
    raise ValueError(f"unknown match mode {mode!r} (identity|resemblance|anywhere)")


def _starts_with(sk: SparseKey, prefix: list[str]) -> bool:
    if len(prefix) > sk.depth:
        return False
    low = sk.lowered()
    return all(low[i] == prefix[i].lower() for i in range(len(prefix)))


def _ends_with(sk: SparseKey, suffix: list[str]) -> bool:
    if len(suffix) > sk.depth:
        return False
    low = sk.lowered()
    off = sk.depth - len(suffix)
    return all(low[off + i] == suffix[i].lower() for i in range(len(suffix)))


def scan_keys(
    keys: Iterable[str],
    *,
    contains: Optional[str] = None,
    prefix: Optional[SegmentsInput] = None,
    suffix: Optional[SegmentsInput] = None,
    at_depth: Optional[int] = None,
) -> list[SparseKey]:
    """Index-of-Indexes scan. Filter the registry by any combination of:

      contains  - a segment appears anywhere (enter-anywhere)
      prefix    - the key starts with these wide segments (resemblance)
      suffix    - the key ends with these specific segments (identity)
      at_depth  - exact segment count
    """
    pre = _seglist(prefix) if prefix is not None else None
    suf = _seglist(suffix) if suffix is not None else None
    seg = _sanitize_segment(contains).lower() if contains else None

    results: list[SparseKey] = []
    for key_str in keys:
        sk = parse_key(key_str)
        if sk.is_empty:
            continue
        if seg is not None and seg not in sk.lowered():
            continue
        if pre is not None and not _starts_with(sk, pre):
            continue
        if suf is not None and not _ends_with(sk, suf):
            continue
        if at_depth is not None and sk.depth != at_depth:
            continue
        results.append(sk)
    return results


def detect_gaps(
    keys: Iterable[str],
    *,
    prefix: SegmentsInput = (),
    pattern: str = r"(\d+)",
) -> list[str]:
    """Detect gaps in a sequential segment within a shared wide prefix.

    Among keys that start with `prefix`, find segments matching `pattern`
    (one numeric group), collect the numbers, and return the missing ones
    reconstructed from the matched segment's template. E.g. Scene 1-4 and
    6-68 with prefix=[Film, Corporate Mistletoe, Script] -> ["Scene 5"].
    """
    pre = _seglist(prefix)
    numbers: set[int] = set()
    template: Optional[str] = None
    rx = re.compile(pattern)

    for key_str in keys:
        sk = parse_key(key_str)
        if not _starts_with(sk, pre):
            continue
        for seg in sk.segments:
            m = rx.search(seg)
            if m:
                numbers.add(int(m.group(1)))
                if template is None:
                    # Reconstruct the template by replacing only the numeric
                    # group (not the whole match) so the segment's literal
                    # text survives: 'Scene 5' -> 'Scene {}'.
                    s, e = m.span(1)
                    template = seg[:s] + "{}" + seg[e:]

    if not numbers or template is None:
        return []
    full = set(range(min(numbers), max(numbers) + 1))
    return [template.format(n) for n in sorted(full - numbers)]
