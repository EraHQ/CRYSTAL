"""Phase 3 slice 3: hosted-plane admission control (G6, ratified).

Dispatch caps per-tenant concurrency under a global cap; tier names
resolve with aliases and a safe default.

2026-10-04 (RC-10, AUDIT_FINAL S8): the enqueue gate `admit_task` had no
caller in the running system, so its six pins here guarded dead code
and were removed with it. The tenant door is require_write_capacity /
_write_admission_block and the cognition worker's per-task check,
pinned in tests/test_http_write_wall.py and tests/test_spend_gates.py.

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

import asyncio


from crystal_cache.control.admission import (
    TIER_TABLE,
    DispatchGate,
    resolve_tier,
)


# --- tier resolution ------------------------------------------------------------

def test_null_and_unknown_tiers_fall_back_to_default():
    assert resolve_tier(None) == TIER_TABLE["free"]
    assert resolve_tier("no-such-tier") == TIER_TABLE["free"]
    # T1a alignment=A (2026-09-24): legacy "scale" resolves via alias to
    # the canonical stamped name.
    assert resolve_tier("scale") == TIER_TABLE["scale_49_seat"]


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
    tier = TIER_TABLE["scale_49_seat"]  # per-tenant cap is high (10)
    running = {"n": 0, "peak": 0}

    async def one_task(cust):
        async with gate.acquire(cust, tier):
            running["n"] += 1
            running["peak"] = max(running["peak"], running["n"])
            await asyncio.sleep(0.02)
            running["n"] -= 1

    await asyncio.gather(*(one_task(f"c{i}") for i in range(6)))
    assert running["peak"] == 2          # the plane-wide ceiling held
