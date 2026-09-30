"""T1a pins (pricing pass ratified 2026-09-23, alignment=A 2026-09-24;
crystal-facts unit switch + Q3=A caps + Q5=B billable origin, ratified
2026-09-30).

What must not regress: the canonical tier names resolve (every
starter_29 tenant silently fell to default before this), the ratified
cap numbers, the Q5=B grace boundaries at EVERY capped tier, the
fact-counted walls on both write surfaces (HTTP and MCP), the billable
count (direct-origin facts only), and the two invariants that protect
everyone: tier-None never caps, and reads never touch any of this by
construction (no read path calls these gates).
"""
import uuid

import pytest
from fastapi import HTTPException

from crystal_cache.control.admission import (
    GRACE_FACTOR,
    TIER_TABLE,
    TierLimits,
    fact_admission,
    resolve_tier,
)
from crystal_cache.ingress.auth import require_write_capacity


def test_canonical_stamped_names_resolve_directly():
    assert resolve_tier("free").fact_cap == 2_500
    assert resolve_tier("free").daily_managed_budget_micro_usd == 500_000
    assert resolve_tier("starter_29").fact_cap == 50_000
    assert resolve_tier("starter_29").daily_managed_budget_micro_usd == 5_000_000
    assert resolve_tier("scale_49_seat").fact_cap is None


def test_legacy_aliases_resolve_to_modern_rows():
    assert resolve_tier("trial_29") is TIER_TABLE["starter_29"]
    assert resolve_tier("pro") is TIER_TABLE["starter_29"]
    assert resolve_tier("scale") is TIER_TABLE["scale_49_seat"]


def test_unknown_and_none_fall_to_deployment_default():
    assert resolve_tier(None) is TIER_TABLE["free"]
    assert resolve_tier("no_such_tier") is TIER_TABLE["free"]


@pytest.mark.parametrize("tier_name,cap", [("free", 2_500), ("starter_29", 50_000)])
def test_grace_boundaries_every_capped_tier(tier_name, cap):
    t = TIER_TABLE[tier_name]
    warn = int(cap * 0.9)
    wall = int(cap * GRACE_FACTOR)
    assert fact_admission(0, t) == "ok"
    assert fact_admission(warn - 1, t) == "ok"
    assert fact_admission(warn, t) == "warning"
    assert fact_admission(cap, t) == "warning"      # at cap: grace
    assert fact_admission(wall - 1, t) == "warning"
    assert fact_admission(wall, t) == "blocked"     # cap x 1.1
    assert GRACE_FACTOR == 1.1


def test_free_tier_exact_boundaries():
    free = TIER_TABLE["free"]
    assert fact_admission(2_249, free) == "ok"
    assert fact_admission(2_250, free) == "warning"
    assert fact_admission(2_749, free) == "warning"
    assert fact_admission(2_750, free) == "blocked"


def test_uncapped_tier_never_blocks():
    scale = TIER_TABLE["scale_49_seat"]
    assert fact_admission(10_000_000, scale) == "ok"
    assert fact_admission(1, TierLimits(1, 1, 1, 1, False)) == "ok"


@pytest.mark.asyncio
async def test_capacity_wall_tier_none_never_caps(store, customer, monkeypatch):
    # Tier None (self-host / legacy): the gate returns before ever
    # counting; a raising counter proves the early exit.
    async def _boom(cid):
        raise AssertionError("count must not be called for tier None")

    monkeypatch.setattr(store, "count_billable_facts", _boom)
    await require_write_capacity(customer, store)  # fixture tier is None


@pytest.mark.asyncio
async def test_capacity_wall_uncapped_scale_never_counts(store, customer, monkeypatch):
    async def _boom(cid):
        raise AssertionError("uncapped tiers must not count")

    await store.set_customer_subscription(customer.id, "scale_49_seat", None)
    c = await store.get_customer_by_id(customer.id)
    monkeypatch.setattr(store, "count_billable_facts", _boom)
    await require_write_capacity(c, store)


@pytest.mark.asyncio
@pytest.mark.parametrize("tier_name,cap", [("free", 2_500), ("starter_29", 50_000)])
async def test_capacity_wall_counts_facts_every_tier(
    store, customer, monkeypatch, tier_name, cap
):
    await store.set_customer_subscription(customer.id, tier_name, None)
    c = await store.get_customer_by_id(customer.id)
    counts = {"n": cap}

    async def _facts(cid):
        return counts["n"]

    async def _crystals(cid):
        raise AssertionError("the wall must count facts, not crystals")

    monkeypatch.setattr(store, "count_billable_facts", _facts)
    monkeypatch.setattr(store, "count_crystals_for_customer", _crystals)
    # At cap: grace zone, writes pass.
    await require_write_capacity(c, store)
    # Past cap x 1.1: the wall, with the humane message in the new unit.
    counts["n"] = int(cap * GRACE_FACTOR)
    with pytest.raises(HTTPException) as e:
        await require_write_capacity(c, store)
    assert e.value.status_code == 402
    assert "crystal facts" in e.value.detail
    assert "recallable and exportable" in e.value.detail


@pytest.mark.asyncio
async def test_mcp_write_gate_counts_facts(store, customer, monkeypatch):
    from crystal_cache.agent import mcp_server

    await store.set_customer_subscription(customer.id, "free", None)
    monkeypatch.setattr(mcp_server, "_get_state", lambda: {"store": store})

    async def _facts(cid):
        return 2_750

    monkeypatch.setattr(store, "count_billable_facts", _facts)
    token = mcp_server._current_customer_id.set(customer.id)
    try:
        denied = await mcp_server._write_admission_block()
    finally:
        mcp_server._current_customer_id.reset(token)
    assert denied is not None
    assert denied["code"] == "memory_full"
    assert "crystal facts" in denied["error"]


async def _seed(store, customer_id, origin, n_facts):
    from crystal_cache.infrastructure.schema import CrystalRow, FactRow

    crystal_id = f"cr_{uuid.uuid4().hex[:12]}"
    async with store.session() as s:
        s.add(CrystalRow(
            id=crystal_id, customer_id=customer_id,
            summary_vector=[], origin=origin,
        ))
        await s.flush()
        for i in range(n_facts):
            s.add(FactRow(
                id=f"f_{uuid.uuid4().hex[:12]}", crystal_id=crystal_id,
                claim_text=f"claim {i}",
            ))


@pytest.mark.asyncio
async def test_billable_count_is_direct_origin_facts_only(store, customer):
    # Q5=B: customer-originated facts count; system-derived ride free.
    await _seed(store, customer.id, "direct", 3)
    await _seed(store, customer.id, "direct", 2)
    await _seed(store, customer.id, "background_worker", 4)
    await _seed(store, customer.id, "assumptions", 5)
    assert await store.count_billable_facts(customer.id) == 5
    # 4 crystals and 14 facts in the bank; the meter reads 5.
    assert await store.count_crystals_for_customer(customer.id) == 4


@pytest.mark.asyncio
async def test_me_reports_fact_meter_and_ai_capacity(store, customer, monkeypatch):
    """T1c: today's ledger spend vs the tier daily allowance, as a
    percent (Q4=A, never dollars), plus the crystal-facts meter.
    Free allowance $0.50/day; $0.25 spent -> 50%."""
    from crystal_cache.config import Settings
    from crystal_cache.endpoints import me as me_mod
    from crystal_cache.ingress import auth as auth_mod

    await store.set_customer_subscription(customer.id, "free", None)
    await store.create_user("uid_cap_1", "cap1@test.dev", customer.id, "owner")
    monkeypatch.setattr(
        auth_mod, "get_settings",
        lambda: Settings(firebase_project_id="test-proj"),
    )
    monkeypatch.setattr(
        auth_mod, "_verify_firebase_jwt",
        lambda tok, proj: {"sub": "uid_cap_1", "email": "cap1@test.dev"},
    )

    async def _spend(cid, *, since=None):
        assert since is not None  # midnight bound must be passed
        return [{"origin": "interactive", "cost_micro_usd": 200_000},
                {"origin": "cognition", "cost_micro_usd": 50_000}]

    async def _facts(cid):
        return 2_300

    monkeypatch.setattr(store, "cost_by_origin", _spend)
    monkeypatch.setattr(store, "count_billable_facts", _facts)

    class _Req:
        headers = {"authorization": "Bearer eyJx.eyJy.sig"}

    out = await me_mod.get_me(_Req(), store)
    assert out["usage"]["ai_capacity_pct"] == 50
    assert out["usage"]["facts_used"] == 2_300
    assert out["usage"]["fact_cap"] == 2_500
    assert out["usage"]["fact_state"] == "warning"
    assert "crystals_used" not in out["usage"]


def test_reads_never_call_the_capacity_gates():
    """Q3=A 'no read limits': the recall tools never reach admission.
    Pinned structurally: the read tool bodies do not call the write gate."""
    import inspect

    from crystal_cache.agent import mcp_server

    for name in (
        "memory_recall", "memory_search", "memory_search_documents",
        "memory_outline", "memory_keys", "memory_synthesize",
        "memory_stats", "memory_list", "memory_export",
    ):
        fn = getattr(mcp_server, name)  # a rename must break this pin loudly
        src = inspect.getsource(getattr(fn, "fn", fn))
        assert "_write_admission_block" not in src, name
        assert "count_billable_facts" not in src, name
