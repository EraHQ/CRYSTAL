"""Segment layer — the ONE place scans read sparse keys (C3 Q1=A,
ratified 2026-08-11).

A sparse key is an ordered path of segments, wide -> specific, of
unbounded length, position-unlabeled (docs/UNIFIED_SPARSE_KEY.md;
primitives in retrieval/sparse_key.py). Cognition functions match on
ANY shared segment at ANY position — never on a numbered slot. This
module replaced four hand-rolled positional parsers that implemented
the RETIRED fixed `Source|Locator|Subject|Domain` contract
(contradiction._subject_of, gap_discovery._domain_of,
assumptions._subject_of_key, pairing_funnel._source_of_key) — parsers
that were blind to short namespace keys (`Assumptions|<subject>`) and
misread every wide-leftmost key written since the June unification.

Grouping discipline: every shared segment is a meeting point, but
common segments make huge noisy groups ("Film" shared by a whole
bank is namespace, not signal). Groups larger than
max(_MIN_EXCLUDE_FLOOR, ceil(fraction * population)) are excluded —
rarity counted from the scanned population itself, no stoplist to
maintain — and surviving groups return RAREST-FIRST so bounded
budgets spend on the strongest meeting points. The fraction knob is
settings.scan_segment_max_group_fraction (default 0.25); the floor
keeps small banks functional (a 3-fact group in an 8-fact bank is
signal, not noise).

Pure module: retrieval.sparse_key + stdlib. Facts are duck-typed
(.prompt_text / .claim_text / .crystal_id) so the funnel's Fact rows
and test fakes both pass through.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Optional

from ..retrieval.sparse_key import parse_key

# Below this population-fraction threshold never drops below this many
# facts — tiny banks would otherwise exclude their own best groups.
_MIN_EXCLUDE_FLOOR = 4


def segments_of(prompt_text: Optional[str]) -> tuple[str, ...]:
    """A key's segments in original casing; () for keyless/blank facts."""
    key = (prompt_text or "").strip()
    if not key:
        return ()
    return tuple(parse_key(key).segments)


def widest_segment_of(prompt_text: Optional[str]) -> Optional[str]:
    """The LEFTMOST (widest) segment — the broadest judgment the
    write-time model made (C3 Q3=A: read it back, don't re-ask)."""
    segs = segments_of(prompt_text)
    return segs[0] if segs else None


def has_segment(prompt_text: Optional[str], segment: str) -> bool:
    """Case-insensitive: does this key carry `segment` at ANY position?"""
    if not segment:
        return False
    want = segment.strip().lower()
    return any(s.lower() == want for s in segments_of(prompt_text))


def most_specific_shared_segment(
    key_a: Optional[str], key_b: Optional[str]
) -> Optional[str]:
    """The shared segment sitting furthest toward the SPECIFIC (right)
    end of both keys — the most precise common ground two facts have.
    Used as the human-facing subject label on conflict rows. None when
    the keys share nothing."""
    a_segs = segments_of(key_a)
    b_segs = segments_of(key_b)
    if not a_segs or not b_segs:
        return None
    b_index = {s.lower(): i for i, s in enumerate(b_segs)}
    best: Optional[tuple[int, str]] = None
    for i, seg in enumerate(a_segs):
        j = b_index.get(seg.lower())
        if j is None:
            continue
        rank = i + j
        # >= : on a rank tie, the LATER position in key A wins — ties
        # resolve toward the more specific end of the caller's key.
        if best is None or rank >= best[0]:
            best = (rank, seg)
    return best[1] if best else None


def exclusion_threshold(population: int, fraction: float) -> int:
    """Members-per-segment above which a segment is namespace-scale
    noise rather than a meeting point."""
    return max(_MIN_EXCLUDE_FLOOR, math.ceil(max(0.0, fraction) * population))


def group_by_shared_segment(
    facts: list[Any],
    *,
    max_group_fraction: float,
    min_group: int = 2,
) -> "OrderedDict[str, list[Any]]":
    """Bucket facts by EVERY segment they carry (case-insensitive
    matching; first-seen casing as the display key), keep groups with
    >= min_group members and <= the exclusion threshold, and return
    them RAREST-FIRST (ties keep insertion order — facts arrive
    newest-first, preserving the sibling scans' recency bias).

    A fact appears in one group per distinct segment it carries — a
    deep key meets the bank at every level of its path, which is the
    point of the unified model."""
    usable = [f for f in facts if (getattr(f, "claim_text", "") or "").strip()]
    grouped: "OrderedDict[str, list[Any]]" = OrderedDict()
    display: dict[str, str] = {}
    for f in usable:
        seen_here: set[str] = set()
        for seg in segments_of(getattr(f, "prompt_text", None)):
            low = seg.lower()
            if low in seen_here:
                continue  # a segment repeated within one key counts once
            seen_here.add(low)
            display.setdefault(low, seg)
            grouped.setdefault(low, []).append(f)

    cap = exclusion_threshold(len(usable), max_group_fraction)
    kept = [
        (low, fs) for low, fs in grouped.items()
        if min_group <= len(fs) <= cap
    ]
    kept.sort(key=lambda item: len(item[1]))  # rarest first; stable ties
    return OrderedDict((display[low], fs) for low, fs in kept)
