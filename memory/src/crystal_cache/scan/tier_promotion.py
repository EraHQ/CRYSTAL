"""Tier promotion — quality tiers that MOVE (launch-prep sweep, 2026-07-02).

The convergence family's cheapest member: no model calls, pure store
signals. One pass walks a customer's crystals and moves quality_tier one
rung at a time based on evidence the bank already produces:

  PROMOTE  quarantine → neutral    when the crystal has at least one
           grounded citation and no open conflicts.
  PROMOTE  neutral → whitelist     when grounded citations ≥ the knob
           (default 3), age ≥ the knob (default 7 days), and no open
           conflicts touch it.
  DEMOTE   whitelist → neutral     the moment an open conflict touches it
           (the contradiction/dedup scans produce the signal).
  DECAY    whitelist → neutral     when the newest grounded citation is
           older than the decay window (RATIFIED 2026-07-02: 30 days) —
           trust must stay earned. Staleness alone NEVER demotes below
           neutral: age is not evidence of wrongness; only conflicts
           touch anything harder, and nothing is ever deleted here.
  NEVER    blacklist               human-set; the promoter never reads or
           writes it.

One rung per pass is deliberate: a quarantined crystal earns neutral this
cycle and whitelist no earlier than a later cycle, so trust accrues across
real usage rather than jumping on one good day. Surfacing tier movement to
retrieval/ranking is tracked separately (BACKLOG §13 — the badge moves
first; consuming it is the follow-on).

Vocabulary note: quality_tier is four-valued per the research §3 merge —
whitelist / neutral / quarantine / blacklist (models/crystal.py). There is
no "trusted" tier; whitelist is the top rung.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

import structlog

from ..config import settings

if TYPE_CHECKING:
    from ..infrastructure import MetadataStore

logger = structlog.get_logger(__name__)

_PROMOTABLE = ("quarantine", "neutral")


@dataclass
class TierPromotionResult:
    """Outcome of one tier-promotion pass for one customer."""

    customer_id: str
    crystals_scanned: int
    promoted: int
    demoted: int
    skipped: int


def _age_days(created_at: Optional[datetime]) -> float:
    """Crystal age in days, tolerating naive datetimes from the DB round-trip."""
    if created_at is None:
        return 0.0
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - created_at).total_seconds() / 86400.0


async def run_tier_promotion_scan(
    *,
    store: "MetadataStore",
    customer_id: str,
    min_citations: Optional[int] = None,
    min_age_days: Optional[int] = None,
    decay_days: Optional[int] = None,
    max_crystals: int = 200,
    log: Any = None,
) -> TierPromotionResult:
    """One promotion/demotion pass over a customer's crystals.

    Knobs default from settings (tier_promotion_min_citations /
    tier_promotion_min_age_days). ``max_crystals`` caps the walk so an idle
    cycle stays cheap on large banks; the pass is stateless, so uncovered
    crystals are simply picked up on later cycles as the list is re-walked.
    No model calls anywhere — the gate this replaces was a v1 stub; the
    signals (grounded citations, open conflicts) are produced by the
    citation pipeline and the convergence scans respectively.
    """
    log = log or logger
    if min_citations is None:
        min_citations = settings.tier_promotion_min_citations
    if min_age_days is None:
        min_age_days = settings.tier_promotion_min_age_days
    if decay_days is None:
        decay_days = settings.tier_promotion_decay_days

    crystals = await store.list_crystals_for_customer(customer_id)
    scanned = 0
    promoted = 0
    demoted = 0
    skipped = 0

    for crystal in crystals[:max_crystals]:
        scanned += 1
        tier = crystal.quality_tier

        if tier == "blacklist":
            skipped += 1
            continue

        open_conflicts = await store.count_open_conflicts_for_crystal(
            customer_id, crystal.id,
        )
        if open_conflicts > 0:
            if tier == "whitelist":
                await store.set_crystal_quality_tier(
                    crystal.id, customer_id, "neutral",
                )
                demoted += 1
                log.info(
                    "tier_promotion.demoted",
                    customer_id=customer_id,
                    crystal_id=crystal.id,
                    open_conflicts=open_conflicts,
                )
            else:
                skipped += 1
            continue

        if tier not in _PROMOTABLE:
            if tier == "whitelist":
                # DECAY (ratified 2026-07-02, 30 days): trust must stay
                # earned — no grounded citation inside the window drifts
                # the crystal back to neutral. Never below neutral.
                latest = await store.latest_grounded_citation_at(
                    customer_id, crystal.id,
                )
                if latest is None or _age_days(latest) >= decay_days:
                    await store.set_crystal_quality_tier(
                        crystal.id, customer_id, "neutral",
                    )
                    demoted += 1
                    log.info(
                        "tier_promotion.decayed",
                        customer_id=customer_id,
                        crystal_id=crystal.id,
                        latest_grounded_citation=str(latest),
                    )
                    continue
            skipped += 1
            continue

        cites = await store.count_grounded_citations_for_crystal(
            customer_id, crystal.id,
        )
        if tier == "quarantine" and cites >= 1:
            await store.set_crystal_quality_tier(crystal.id, customer_id, "neutral")
            promoted += 1
        elif (
            tier == "neutral"
            and cites >= min_citations
            and _age_days(crystal.created_at) >= min_age_days
        ):
            await store.set_crystal_quality_tier(crystal.id, customer_id, "whitelist")
            promoted += 1
        else:
            skipped += 1

    result = TierPromotionResult(
        customer_id=customer_id,
        crystals_scanned=scanned,
        promoted=promoted,
        demoted=demoted,
        skipped=skipped,
    )
    if promoted or demoted:
        log.info(
            "tier_promotion.pass_complete",
            customer_id=customer_id,
            crystals_scanned=scanned,
            promoted=promoted,
            demoted=demoted,
        )
    return result
