"""Tier as an epistemic signal to the model (RATIFIED 2026-07-02).

Ratified design (superseding the ranking-weight framing): quality tiers do
NOT change retrieval scores anywhere. They are a SIGN to the LLM about how
vetted a piece of knowledge is — "maybe search for updated information
and/or ask the user" — surfaced as data alongside results, never as a
weight.

Semantics (the one legend both prompts and notes use):
  whitelist   — evidence-backed: earned grounded citations, survived
                conflict scans, still fresh (decay window).
  neutral     — ordinary standing: not yet strongly vetted either way.
  quarantine  — unvetted origin: treat with care.
  blacklist   — operator-flagged: do not rely on it.

This module is the ONE place that renders the signal so every surface
says the same thing.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..infrastructure import MetadataStore

TIER_SEMANTICS = (
    "Knowledge quality tiers: whitelist = evidence-backed (cited, "
    "conflict-free, fresh); neutral = not yet strongly vetted; "
    "quarantine = unvetted, treat with care; blacklist = operator-flagged, "
    "do not rely on it. Tiers never change ranking - they are a signal: "
    "for neutral/quarantine knowledge that is load-bearing to your answer, "
    "consider verifying via web_search or asking the user; state the "
    "uncertainty rather than presenting unvetted knowledge as settled."
)

CONFLICT_SEMANTICS = (
    "Contested knowledge: when a retrieval result carries a conflict_note, "
    "one or more retrieved facts are party to an OPEN knowledge conflict - "
    "the bank itself has flagged a disagreement it has not resolved. Never "
    "present a contested fact as settled. Surface BOTH sides to the user, "
    "reason about which is likely current (recency, provenance, "
    "specificity), state your lean, and ASK the user to confirm before "
    "relying on either. When the user confirms, update memory accordingly."
)


async def tier_map(
    store: "MetadataStore",
    customer_id: str,
    crystal_ids: list[str],
) -> dict[str, str]:
    """{crystal_id: quality_tier} for the given crystals (one read)."""
    if not crystal_ids:
        return {}
    return await store.get_quality_tiers(crystal_ids, customer_id=customer_id)


def conflict_note(
    contested: dict[str, list[dict[str, str]]],
) -> Optional[str]:
    """CONF-R (2026-07-23): the contested-knowledge line for a result
    set, or None when nothing retrieved is under an open conflict.

    Same philosophy as tier_note: a SIGN the model reasons about, never
    a filter — the contested fact still arrives, accompanied by the
    other side's claim so the model can reason about the disagreement
    in the moment instead of answering on half of it."""
    if not contested:
        return None
    n = len(contested)
    plural = "facts are" if n > 1 else "fact is"
    lines = [
        f"CONTESTED: {n} retrieved {plural} party to an open knowledge "
        "conflict. Surface both sides, reason about which is current, "
        "state your lean, and ask the user to confirm before relying on "
        "either. The opposing claims:"
    ]
    shown = 0
    for fact_id, entries in contested.items():
        for entry in entries:
            if shown >= 3:
                break
            claim = (entry.get("counterpart_claim") or "").strip()
            if len(claim) > 240:
                claim = claim[:240].rstrip() + "\u2026"
            lines.append(f"- vs {fact_id}: {claim}")
            shown += 1
        if shown >= 3:
            break
    remaining = sum(len(v) for v in contested.values()) - shown
    if remaining > 0:
        lines.append(f"(+{remaining} more open conflict(s) on this result set)")
    return "\n".join(lines)


def tier_note(tiers: dict[str, str]) -> Optional[str]:
    """The epistemic note for a result set, or None when nothing needs one.

    None when every contributing crystal is whitelist (or the set is
    empty) — no noise when the knowledge is fully vetted. Otherwise a
    compact count line plus the action guidance.
    """
    if not tiers:
        return None
    counts: dict[str, int] = {}
    for tier in tiers.values():
        counts[tier] = counts.get(tier, 0) + 1
    non_whitelist = {t: n for t, n in counts.items() if t != "whitelist"}
    if not non_whitelist:
        return None
    parts = ", ".join(f"{n} {t}" for t, n in sorted(non_whitelist.items()))
    note = (
        f"Quality: {parts}"
        + (f", {counts['whitelist']} whitelist" if counts.get("whitelist") else "")
        + ". Non-whitelist knowledge is not fully vetted - if it is "
        "load-bearing, consider verifying (web_search) or asking the user"
    )
    if counts.get("blacklist"):
        note += "; blacklist items are operator-flagged - do not rely on them"
    return note + "."
