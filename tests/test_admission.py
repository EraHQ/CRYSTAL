"""Phase 3 slice 3: hosted-plane admission control (G6, ratified).

Enqueue refuses early — queue depth, tier ceilings, GPU gating — with
below-ceiling requests passing through unchanged; dispatch caps per-
tenant concurrency under a global cap. Tested against the REAL
agent_tasks queue (the store's count is the admission read).

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

import asyncio

import pytest

from crystal_cache.control.admission import (
    TIER_TABLE,
    DispatchGate,
    admit_task,
    resolve_tier,
)


# --- tier resolution ------------------------------------------------------------

def test_null_and_unknown_tiers_fall_back_to_default():
    assert resolve_tier(None) == TIER_TABLE["free"]
    assert resolve_tier("no-such-tier") == TIER_TABLE["free"]
    assert resolve_tier("scale") == TIER_TABLE["scale"]


# --- enqueue gate ----------------------------------------------------------------

async def test_defaults_to_tier_ceilings_when_nothing_requested(store, customer):
    d = await admit_task(
        store, customer_id=customer.id, subscription_tier="pro",
    )
    assert d.allowed
    assert d.deadline_seconds == TIER_TABLE["pro"].max_deadline_seconds
    assert d.budget_micro_usd == TIER_TABLE["pro"].max_budget_micro_usd


async def test_tighter_requests_pass_through_unchanged(store, customer):
    d = await admit_task(
        store, customer_id=customer.id, subscription_tier="pro",
        requested_deadline_seconds=60, requested_budget_micro_usd=1000,
    )
    assert d.allowed
    assert d.deadline_seconds == 60
    assert d.budget_micro_usd == 1000


async def test_ceiling_violations_are_named(store, customer):
    over_time = await admit_task(
        store, customer_id=customer.id, subscription_tier="free",
        requested_deadline_seconds=999_999,
    )
    assert not over_time.allowed and over_time.reason == "deadline_exceeds_tier"

    over_money = await admit_task(
        store, customer_id=customer.id, subscription_tier="free",
        requested_budget_micro_usd=10**9,
    )
    assert not over_money.allowed and over_money.reason == "budget_exceeds_tier"


async def test_gpu_gated_by_tier(store, customer):
    no = await admit_task(
        store, customer_id=customer.id, subscription_tier="free", gpu=True,
    )
    assert not no.allowed and no.reason == "gpu_not_in_tier"

    yes = await admit_task(
        store, customer_id=customer.id, subscription_tier="scale", gpu=True,
    )
    assert yes.allowed


async def test_queue_full_counts_real_agent_tasks(store, customer):
    """Depth is read from the REAL queue: fill queued+concurrent for the
    free tier (3 + 1 = 4 active) and the fifth enqueue is refused."""
    cap = (TIER_TABLE["free"].max_queued_tasks
           + TIER_TABLE["free"].max_concurrent_tasks)
    for i in range(cap):
        await store.create_agent_task(
            customer.id, project_dir="/p", task=f"t{i}",
        )
    d = await admit_task(
        store, customer_id=customer.id, subscription_tier="free",
    )
    assert not d.allowed and d.reason == "queue_full"


async def test_terminal_tasks_free_the_queue(store, customer):
    cap = (TIER_TABLE["free"].max_queued_tasks
           + TIER_TABLE["free"].max_concurrent_tasks)
    rows = []
    for i in range(cap):
        rows.append(await store.create_agent_task(
            customer.id, project_dir="/p", task=f"t{i}",
        ))
    # One finishes — its slot opens.
    await store.finish_agent_task(rows[0]["id"], status="done")
    d = await admit_task(
        store, customer_id=customer.id, subscription_tier="free",
    )
    assert d.allowed


# --- dispatch gate -----------------------------------------------------------------

async def test_per_tenant_semaphore_caps_concurrency():
    gate = DispatchGate(global_max=10)
    tier = TIER_TABLE["free"]           # max_concurrent_tasks = 1
    running = {"n": 0, "peak": 0}

    async def one_task():
        async with gate.acquire("cust-a", tier):
            running["n"] += 1
            running["peak"] = max(running["peak"], running["n"])
            await asyncio.sleep(0.02)
            running["n"] -= 1

    await asyncio.gather(*(one_task() for _ in range(4)))
    assert running["peak"] == 1          # never two at once for this tenant


async def test_global_cap_binds_across_tenants():
    gate = DispatchGate(global_max=2)
    tier = TIER_TABLE["scale"]          # per-tenant cap is high (10)
    running = {"n": 0, "peak": 0}

    async def one_task(cust):
        async with gate.acquire(cust, tier):
            running["n"] += 1
            running["peak"] = max(running["peak"], running["n"])
            await asyncio.sleep(0.02)
            running["n"] -= 1

    await asyncio.gather(*(one_task(f"c{i}") for i in range(6)))
    assert running["peak"] == 2          # the plane-wide ceiling held
