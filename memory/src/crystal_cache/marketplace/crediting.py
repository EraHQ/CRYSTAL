"""Shard crediting policy (Growth G4). PURE — no DB, no I/O.

Metering is CITATIONS (the G1 rail): a *cited* crystal accrues a shard, not a
merely-injected one. This module holds the anti-gaming + weighting decisions
so the ledger mixin stays pure-SQL:

  - is_marketplace_crystal — only general-scoped (marketplace) crystals earn;
    private/team citations never do.
  - is_self_traffic — a team citing its own crystal earns nothing (the
    seeder-decoupling instinct; the key anti-gaming rule alongside grounding).
  - split_weight — when several crystals are co-cited for one claim, the
    usefulness weight is split among them (credit-split).
  - shards_from_weight — maps a usefulness weight to an INTEGER shard count.

**D7 (the bounded reward pool) is deferred.** Until it lands, every eligible
grounded citation is worth a fixed integer shard, and the *fractional* weight
is preserved in the ledger's raw_weight column so the future pool can
apportion proportionally without losing information. Convertibility (shards
offsetting subscription) stays OFF at launch regardless.
"""
from __future__ import annotations

from typing import Optional


def is_marketplace_crystal(
    crystal_type: Optional[str],
    crystal_customer_id: Optional[str],
) -> bool:
    """True iff the crystal is a general/marketplace crystal — the only tier
    that earns shards.

    Two equivalent signals of the general tier (either suffices): a
    `general:*` crystal type, or a NULL customer_id (the general-bank
    convention — general crystals are not owned by any one team).
    """
    if crystal_type is not None and crystal_type.startswith("general:"):
        return True
    if crystal_customer_id is None:
        return True
    return False


def is_self_traffic(
    crystal_group_team_id: Optional[str],
    consuming_team_id: Optional[str],
) -> bool:
    """True iff the citing team is (an owner of) the cited crystal — excluded
    from earning.

    v1 rule: self-traffic when the crystal's owning team equals the consuming
    team. A general crystal with no owning team (group_team_id is None) is
    never self-traffic. Deeper contributor-team exclusion (a promoted
    crystal's original team) is future work; the contributor provenance F3
    captured at merge is where that will read from.
    """
    if crystal_group_team_id is None or consuming_team_id is None:
        return False
    return crystal_group_team_id == consuming_team_id


def split_weight(total_weight: float, num_crystals: int) -> float:
    """Split a claim's usefulness weight equally among co-cited crystals.

    One claim citing N crystals splits the credit N ways (credit-split). N <= 1
    returns the whole weight. The result is the per-crystal raw_weight recorded
    in the ledger; integer shards are derived separately (shards_from_weight),
    and the fraction is preserved for D7's pool apportionment.
    """
    if num_crystals <= 1:
        return total_weight
    return total_weight / float(num_crystals)


def shards_from_weight(weight: float) -> int:
    """Map a usefulness weight to an INTEGER shard count (placeholder, D7).

    Until the bounded reward pool lands, any positive weight earns exactly one
    shard — the unit "one cited crystal = one shard claim." The fractional
    `weight` is preserved in the ledger (raw_weight) so the future pool can
    apportion proportionally. Non-positive weight earns nothing.
    """
    return 1 if weight > 0 else 0
