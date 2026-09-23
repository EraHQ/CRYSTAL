"""Hosted-plane admission control (Phase 3 G6, 2026-07-03, ratified).

The tier table maps a tenant's subscription_tier to hard ceilings on
disposable-task deadline, budget, queue depth, concurrency, and GPU
access. Two enforcement points, refusing EARLY rather than killing late:

  ENQUEUE (admit_task): reject when the tenant's queue is at depth, when
  requested limits exceed the tier ceiling, or when GPU is requested on
  a tier without it. Requests BELOW the ceiling pass through unchanged —
  a tenant may want a tighter budget than their tier allows — and absent
  requests default to the ceiling.

  DISPATCH (DispatchGate): a per-tenant semaphore sized from the tier
  caps concurrent running tasks, under a global plane-wide cap.

Self-host is untouched by all of this: operators set DisposableLimits
directly and never construct these objects. The tier values here are
launch defaults, deliberately conservative; pricing iteration changes
numbers, not shapes.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from ..config import settings

# coding-agent is not importable from src (separate tree); the limits
# shape is duplicated as a structural twin on purpose — admission is
# plane-side and must not import the agent package. The runner's
# DisposableLimits is constructed FROM an AdmissionDecision at the
# enqueue endpoint (slice 5 wiring).

__all__ = [
    "AdmissionDecision",
    "DispatchGate",
    "TierLimits",
    "TIER_TABLE",
    "GRACE_FACTOR",
    "admit_task",
    "crystal_admission",
    "resolve_tier",
]


@dataclass(frozen=True)
class TierLimits:
    """One tier's ceilings (ratified G6 shape; E4 monthly cap added
    Accounts Phase B, 2026-07-06; T1 pricing pass 2026-09-23 — the
    "pending pricing pass" placeholders below are now the RATIFIED launch
    values, and two customer-facing capacity dimensions join the task
    ceilings)."""
    max_deadline_seconds: float
    max_budget_micro_usd: int
    max_concurrent_tasks: int
    max_queued_tasks: int
    gpu_allowed: bool
    # E4: month-to-date ceiling on MANAGED-inference proxy spend. Enforced
    # at the proxy door. T1: this is the internal per-account LOSS
    # BACKSTOP — never customer-facing (Q4=A: customers see capacity, not
    # dollars).
    monthly_managed_budget_micro_usd: int = 0
    # T1 (ratified 2026-09-23): bank-size cap — the need-based upgrade
    # wall. None = unlimited. Grace: warnings from 90%, writes keep
    # working to cap × GRACE_FACTOR, hard wall there (Q5=B). Reads NEVER
    # degrade at any boundary.
    crystal_cap: Optional[int] = None
    # T1: daily managed-AI allowance — surfaced to customers ONLY as a
    # capacity percent (Q4=A), soft by construction (resets at midnight).
    # 0 = no tier default (explicit spend-budget rows and the global
    # setting still apply).
    daily_managed_budget_micro_usd: int = 0


# Q5=B (2026-09-23): the overage grace — soft warnings from 90% of a
# cap, hard wall at cap × this factor.
GRACE_FACTOR: float = 1.1


def crystal_admission(count: int, tier: TierLimits) -> str:
    """Capacity state for a bank of `count` crystals under `tier`:
    'ok' | 'warning' (>= 90% of cap) | 'blocked' (>= cap × GRACE_FACTOR).
    Uncapped tiers (crystal_cap None — Scale, and tier-None legacy /
    self-host via resolve_tier's caller checks) are always 'ok'."""
    cap = tier.crystal_cap
    if not cap:
        return "ok"
    if count >= int(cap * GRACE_FACTOR):
        return "blocked"
    if count >= int(cap * 0.9):
        return "warning"
    return "ok"


# Launch values — RATIFIED 2026-09-23 (Q1=B ladder, Q2/Q3=A caps).
# Conservative on purpose — raising a ceiling is a painless change;
# lowering one on live tenants is not. Keys are the CANONICAL STAMPED
# STRINGS (what signup and the billing webhook actually write — the old
# free/pro/scale names silently fell through resolve_tier to the default
# for every starter_29 tenant; alignment=A 2026-09-24).
TIER_TABLE: dict[str, TierLimits] = {
    "free": TierLimits(
        max_deadline_seconds=1800,          # 30 min
        max_budget_micro_usd=500_000,       # $0.50
        max_concurrent_tasks=1,
        max_queued_tasks=3,
        gpu_allowed=False,
        monthly_managed_budget_micro_usd=10_000_000,    # $10/mo backstop
        crystal_cap=500,
        daily_managed_budget_micro_usd=500_000,         # $0.50/day
    ),
    "starter_29": TierLimits(
        max_deadline_seconds=7200,          # 2 h
        max_budget_micro_usd=5_000_000,     # $5
        max_concurrent_tasks=3,
        max_queued_tasks=10,
        gpu_allowed=False,
        monthly_managed_budget_micro_usd=100_000_000,   # $100/mo backstop
        crystal_cap=25_000,
        daily_managed_budget_micro_usd=5_000_000,       # $5/day
    ),
    "scale_49_seat": TierLimits(
        max_deadline_seconds=21_600,        # 6 h
        max_budget_micro_usd=25_000_000,    # $25
        max_concurrent_tasks=10,
        max_queued_tasks=50,
        gpu_allowed=True,
        monthly_managed_budget_micro_usd=250_000_000,   # $250/mo backstop
        crystal_cap=None,                                # unlimited
        daily_managed_budget_micro_usd=25_000_000,       # $25/day backstop
    ),
}

# Legacy and historical tier strings resolve to their modern rows —
# trial_29 accounts predate the free-tier pivot and were Starter with a
# clock; "pro"/"scale" never shipped to a customer but appear in old
# fixtures and docs.
TIER_ALIASES: dict[str, str] = {
    "trial_29": "starter_29",
    "pro": "starter_29",
    "scale": "scale_49_seat",
}

# Statuses that count against the queue-depth ceiling: everything the
# tenant has in flight that has not reached a terminal state.
ACTIVE_STATUSES: tuple[str, ...] = ("queued", "running")


async def enforce_managed_budget(store, customer) -> None:
    """The E4 monthly spend door (2026-07-06) — ONE implementation, called
    by EVERY per-tenant inference surface (chat proxy AND agent; ratified:
    the agent has everything the proxy has, in the same commit). A managed
    tenant at or over its tier's month-to-date cap gets 429 before any
    upstream work; byok tenants never touch the read.
    """
    from fastapi import HTTPException

    if getattr(customer, "inference_mode", "byok") != "managed":
        return
    cap = resolve_tier(
        getattr(customer, "subscription_tier", None)
    ).monthly_managed_budget_micro_usd
    if cap <= 0:
        return
    spent = await store.managed_spend_micro_usd_this_month(customer.id)
    if spent >= cap:
        raise HTTPException(
            status_code=429,
            detail=(
                "Monthly managed-inference budget reached for this "
                "plan. It resets on the 1st (UTC). Upgrade your "
                "plan or switch to your own API key in Settings "
                "to continue immediately."
            ),
        )


async def function_budget_allows(
    store,
    customer,
    function: str,
    *,
    origin: str,
    operator_id: Optional[str] = None,
    default_cap_micro_usd: int = 0,
) -> bool:
    """The spend-budget door for AUTONOMOUS paths (S4, 2026-07-08 —
    docs/GAP_ENGINE_AND_LEARN_REDESIGN.md). Resolution: operator row →
    tenant row → default_cap_micro_usd (0 = the function is OFF: the
    manual-by-default posture ratified in B-1). When a cap applies, the
    meter is the llm_calls ledger filtered by `origin` for the budget's
    period — the ledger IS the meter, no second counter to drift.

    Returns bool (never raises): auto paths SKIP quietly when
    disallowed; interactive paths have their own doors (E4)."""
    budget = await store.get_spend_budget(
        customer.id, function=function, operator_id=operator_id
    )
    if budget is None:
        cap = int(default_cap_micro_usd)
        period = "monthly"
    else:
        cap = int(budget.cap_micro_usd)
        period = budget.period
    if cap <= 0:
        return False
    spent = await store.origin_spend_micro_usd_this_period(
        customer.id, origin=origin, period=period
    )
    return spent < cap


def enforce_managed_model(customer, model_id) -> None:
    """E4 model policy (2026-07-06): a MANAGED tenant's calls run on the
    platform's key, so the effective model must be one the platform
    serves. byok tenants are unrestricted — their key, their model.
    Applied wherever a model is chosen per-request (proxy + agent) and on
    the Settings PATCH.
    """
    from fastapi import HTTPException

    from ..endpoints.me import MANAGED_ALLOWED_MODELS

    if getattr(customer, "inference_mode", "byok") != "managed":
        return
    if not model_id or model_id in MANAGED_ALLOWED_MODELS:
        return
    raise HTTPException(
        status_code=400,
        detail=(
            "Managed inference supports: "
            + ", ".join(sorted(MANAGED_ALLOWED_MODELS))
            + ". Switch to your own key for other models."
        ),
    )


def resolve_tier(subscription_tier: Optional[str]) -> TierLimits:
    """Tier string → limits. Canonical stamped strings hit the table
    directly; legacy strings resolve through TIER_ALIASES; anything
    unknown (and None — though callers gate tier-None BEFORE resolving,
    since self-host/legacy never caps) falls to the deployment default."""
    name = (subscription_tier or settings.default_subscription_tier).strip()
    name = TIER_ALIASES.get(name, name)
    return TIER_TABLE.get(name) or TIER_TABLE[settings.default_subscription_tier]


@dataclass(frozen=True)
class AdmissionDecision:
    allowed: bool
    reason: Optional[str] = None
    # The limits the task will actually run under when allowed: the
    # request where given (already validated <= ceiling), else the
    # ceiling itself.
    deadline_seconds: Optional[float] = None
    budget_micro_usd: Optional[int] = None


async def admit_task(
    store,
    *,
    customer_id: str,
    subscription_tier: Optional[str],
    requested_deadline_seconds: Optional[float] = None,
    requested_budget_micro_usd: Optional[int] = None,
    gpu: bool = False,
) -> AdmissionDecision:
    """The enqueue gate. Reject early with a named reason; otherwise
    return the effective limits the runner must enforce."""
    tier = resolve_tier(subscription_tier)

    if gpu and not tier.gpu_allowed:
        return AdmissionDecision(
            allowed=False, reason="gpu_not_in_tier",
        )
    if (requested_deadline_seconds is not None
            and requested_deadline_seconds > tier.max_deadline_seconds):
        return AdmissionDecision(
            allowed=False, reason="deadline_exceeds_tier",
        )
    if (requested_budget_micro_usd is not None
            and requested_budget_micro_usd > tier.max_budget_micro_usd):
        return AdmissionDecision(
            allowed=False, reason="budget_exceeds_tier",
        )

    active = await store.count_agent_tasks_by_status(
        customer_id, ACTIVE_STATUSES,
    )
    if active >= tier.max_queued_tasks + tier.max_concurrent_tasks:
        return AdmissionDecision(allowed=False, reason="queue_full")

    return AdmissionDecision(
        allowed=True,
        deadline_seconds=(
            requested_deadline_seconds
            if requested_deadline_seconds is not None
            else tier.max_deadline_seconds
        ),
        budget_micro_usd=(
            requested_budget_micro_usd
            if requested_budget_micro_usd is not None
            else tier.max_budget_micro_usd
        ),
    )


class DispatchGate:
    """Per-tenant concurrency semaphores under a global cap (the dispatch
    half of G6). acquire() is an async context manager: holding it means
    one running slot for that tenant AND one of the plane's global slots.

    Semaphores are per-process — correct for the single dispatcher the
    plane runs (the worker service owns dispatch); a multi-dispatcher
    future moves this into the claim query, not into Redis.
    """

    def __init__(self, global_max: int = 50):
        self._global = asyncio.Semaphore(global_max)
        self._tenants: dict[str, asyncio.Semaphore] = {}

    def _tenant_sem(self, customer_id: str, tier: TierLimits) -> asyncio.Semaphore:
        sem = self._tenants.get(customer_id)
        if sem is None:
            sem = asyncio.Semaphore(tier.max_concurrent_tasks)
            self._tenants[customer_id] = sem
        return sem

    def acquire(self, customer_id: str, tier: TierLimits):
        gate = self

        class _Slot:
            async def __aenter__(self):
                await gate._global.acquire()
                self._sem = gate._tenant_sem(customer_id, tier)
                await self._sem.acquire()
                return self

            async def __aexit__(self, *exc):
                self._sem.release()
                gate._global.release()
                return False

        return _Slot()
