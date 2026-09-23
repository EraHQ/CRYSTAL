"""T1a pins (pricing pass ratified 2026-09-23, alignment=A 2026-09-24).

What must not regress: the canonical tier names resolve (every
starter_29 tenant silently fell to default before this), the ratified
cap numbers, the Q5=B grace boundaries, and the two invariants that
protect everyone: tier-None never caps, and reads never touch any of
this by construction (no read path calls these gates).
"""
import pytest
from fastapi import HTTPException

from crystal_cache.control.admission import (
    GRACE_FACTOR,
    TIER_TABLE,
    TierLimits,
    crystal_admission,
    resolve_tier,
)
from crystal_cache.ingress.auth import require_write_capacity


def test_canonical_stamped_names_resolve_directly():
    assert resolve_tier("free").crystal_cap == 500
    assert resolve_tier("free").daily_managed_budget_micro_usd == 500_000
    assert resolve_tier("starter_29").crystal_cap == 25_000
    assert resolve_tier("starter_29").daily_managed_budget_micro_usd == 5_000_000
    assert resolve_tier("scale_49_seat").crystal_cap is None


def test_legacy_aliases_resolve_to_modern_rows():
    assert resolve_tier("trial_29") is TIER_TABLE["starter_29"]
    assert resolve_tier("pro") is TIER_TABLE["starter_29"]
    assert resolve_tier("scale") is TIER_TABLE["scale_49_seat"]


def test_unknown_and_none_fall_to_deployment_default():
    assert resolve_tier(None) is TIER_TABLE["free"]
    assert resolve_tier("no_such_tier") is TIER_TABLE["free"]


def test_grace_boundaries_free_tier():
    free = TIER_TABLE["free"]
    assert crystal_admission(0, free) == "ok"
    assert crystal_admission(449, free) == "ok"
    assert crystal_admission(450, free) == "warning"   # 90% of 500
    assert crystal_admission(500, free) == "warning"   # at cap: grace
    assert crystal_admission(549, free) == "warning"
    assert crystal_admission(550, free) == "blocked"   # cap × 1.1
    assert GRACE_FACTOR == 1.1


def test_uncapped_tier_never_blocks():
    scale = TIER_TABLE["scale_49_seat"]
    assert crystal_admission(10_000_000, scale) == "ok"
    assert crystal_admission(1, TierLimits(1, 1, 1, 1, False)) == "ok"


@pytest.mark.asyncio
async def test_capacity_wall_tier_none_never_caps(store, customer, monkeypatch):
    # Tier None (self-host / legacy): the gate returns before ever
    # counting — a raising counter proves the early exit.
    async def _boom(cid):
        raise AssertionError("count must not be called for tier None")

    monkeypatch.setattr(store, "count_crystals_for_customer", _boom)
    await require_write_capacity(customer, store)  # fixture tier is None


@pytest.mark.asyncio
async def test_capacity_wall_grace_and_block(store, customer, monkeypatch):
    await store.set_customer_subscription(customer.id, "free", None)
    c = await store.get_customer_by_id(customer.id)

    counts = {"n": 500}

    async def _count(cid):
        return counts["n"]

    monkeypatch.setattr(store, "count_crystals_for_customer", _count)
    # At cap: grace zone, writes pass.
    await require_write_capacity(c, store)
    # Past cap × 1.1: the wall, with the humane message.
    counts["n"] = 550
    with pytest.raises(HTTPException) as e:
        await require_write_capacity(c, store)
    assert e.value.status_code == 402
    assert "recallable and exportable" in e.value.detail
