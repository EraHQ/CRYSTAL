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
    "relying on either. When the user confirms which side is right, settle "
    "it with the resolve_conflict tool, quoting their own words - that "
    "retires the outdated fact and closes the conflict together. NEVER "
    "settle a conflict by writing a new, better-matching fact: the outdated "
    "fact stays live and the conflict stays open, so retrieval merely "
    "prefers the newer claim while the memory goes on disagreeing with "
    "itself out of sight."
)


ASSUMPTION_SEMANTICS = (
    "Assumption knowledge: retrieval results may include ASSUMPTION "
    "crystals - bridging inferences this system generated from two "
    "pieces of existing knowledge, NOT facts anyone stated. Each is "
    "named in the assumption_note with its confidence and the parent "
    "knowledge it was inferred from. Treat an assumption as a "
    "hypothesis: weigh it below stated knowledge, attribute it as an "
    "inference ('the bank infers...', never 'the bank says...'), and "
    "when it is load-bearing to your answer, verify it (web_search) or "
    "ask the user before relying on it. An INVALIDATED assumption lost "
    "a parent it was inferred from - do not rely on it at all."
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


def _parent_phrase(parents: list[dict], dead_count: int) -> str:
    """The 'inferred from ...' clause for one assumption line."""
    quoted: list[str] = []
    for p in parents[:2]:
        summary = (p.get("summary_text") or "").strip() or p.get("id", "?")
        if len(summary) > 80:
            summary = summary[:80].rstrip() + "\u2026"
        quoted.append(f'"{summary}"')
    if quoted and dead_count:
        quoted.append("a since-deleted parent")
    if not quoted:
        return "knowledge since deleted"
    return " + ".join(quoted)


def assumption_note(annotations: dict[str, dict]) -> Optional[str]:
    """C1 (ratified 2026-08-07, Q1=C): the assumption-framing note for a
    result set, or None when no retrieved crystal is an assumption.

    Same philosophy as tier_note / conflict_note: a SIGN the model
    reasons about, never a filter — the assumption's content still
    arrives; this names it as an INFERENCE with its confidence and
    parents so the model can attribute it honestly instead of
    presenting a system-generated hypothesis as a stated fact.
    conflict_note's cap discipline: 3 detailed lines + a remainder.
    """
    if not annotations:
        return None
    n = len(annotations)
    noun = "crystals are" if n > 1 else "crystal is"
    lines = [
        f"ASSUMPTIONS: {n} retrieved {noun} system-generated "
        "inference(s) bridging existing knowledge - hypotheses, NOT "
        "stated facts. Weigh them below stated knowledge; verify "
        "(web_search) or ask the user before relying on one that is "
        "load-bearing. The inferences:"
    ]
    shown = 0
    for crystal_id, info in annotations.items():
        if shown >= 3:
            break
        dead = list(info.get("invalidated_parents") or [])
        invalidated = bool(dead) or info.get("quality_tier") == "blacklist"
        if invalidated:
            lines.append(
                f"- {crystal_id}: INVALIDATED - a parent it was inferred "
                "from was deleted; do not rely on it"
            )
        else:
            conf = info.get("confidence")
            conf_part = (
                f" (confidence {conf:.2f})" if isinstance(conf, (int, float))
                else ""
            )
            phrase = _parent_phrase(list(info.get("parents") or []), len(dead))
            lines.append(f"- {crystal_id}{conf_part}: inferred from {phrase}")
        shown += 1
    if n > shown:
        lines.append(f"(+{n - shown} more assumption(s) in this result set)")
    return "\n".join(lines)
