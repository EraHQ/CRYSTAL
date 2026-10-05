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
from datetime import datetime, timedelta, timezone
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
    "fact_admission",
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
    # T1 (ratified 2026-09-23): bank-size cap, the need-based upgrade
    # wall. None = unlimited. Grace: warnings from 90%, writes keep
    # working to cap × GRACE_FACTOR, hard wall there (Q5=B). Reads NEVER
    # degrade at any boundary.
    # Unit switch (ratified 2026-09-30): the cap counts CRYSTAL FACTS
    # (bound pairs), customer-facing name "crystal facts", not crystals.
    # Crystal-capping mis-incentivized binding; facts are the honest
    # unit. Billable = facts on origin='direct' crystals only (Q5=B:
    # system-derived facts ride free). See count_billable_facts.
    fact_cap: Optional[int] = None
    # T1: daily managed-AI allowance — surfaced to customers ONLY as a
    # capacity percent (Q4=A), soft by construction (resets at midnight).
    # 0 = no tier default (explicit spend-budget rows and the global
    # setting still apply).
    # v108 (Q11=A, 2026-10-01): ENFORCED at the door of every LLM-spending
    # path except read-only recall/search; see enforce_managed_budget.
    daily_managed_budget_micro_usd: int = 0
    # v108 (Q13=A): the managed models this tier may run. Opus starts at
    # Starter. byok tenants are unrestricted (their key, their model).
    allowed_models: tuple[str, ...] = ()
    # v108 (Q15=A): the largest max_tokens a single managed turn may ask
    # for. Unclamped, one Opus turn at 64k output was ~$45.
    max_output_tokens: int = 4096


# Model ids the platform serves (the full managed set). Tier rows pick
# from this; enforce_managed_model refuses anything outside the tier's
# pick. The free tier does NOT get Opus (Q13=A).
MANAGED_MODELS_ALL: tuple[str, ...] = (
    "claude-haiku-4-5", "claude-sonnet-5", "claude-opus-4-8",
)
MANAGED_MODELS_FREE: tuple[str, ...] = ("claude-haiku-4-5", "claude-sonnet-5")


# Q5=B (2026-09-23): the overage grace — soft warnings from 90% of a
# cap, hard wall at cap × this factor.
GRACE_FACTOR: float = 1.1


def fact_admission(count: int, tier: TierLimits) -> str:
    """Capacity state for a bank holding `count` billable crystal facts
    under `tier`: 'ok' | 'warning' (>= 90% of cap) | 'blocked'
    (>= cap × GRACE_FACTOR). Uncapped tiers (fact_cap None: Scale, and
    tier-None legacy / self-host via the callers' early exits) are
    always 'ok'."""
    cap = tier.fact_cap
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
        # Q14=A (2026-10-01): monthly = 30 x daily, counts ALL origins.
        # A guard against gate bugs, never the first wall a customer hits.
        monthly_managed_budget_micro_usd=15_000_000,    # $15/mo backstop
        fact_cap=2_500,                     # Q3=A 2026-09-30
        daily_managed_budget_micro_usd=500_000,         # $0.50/day
        allowed_models=MANAGED_MODELS_FREE,
        max_output_tokens=4_096,
    ),
    "starter_29": TierLimits(
        max_deadline_seconds=7200,          # 2 h
        max_budget_micro_usd=5_000_000,     # $5
        max_concurrent_tasks=3,
        max_queued_tasks=10,
        gpu_allowed=False,
        monthly_managed_budget_micro_usd=150_000_000,   # $150/mo backstop
        fact_cap=50_000,                    # Q3=A 2026-09-30
        daily_managed_budget_micro_usd=5_000_000,       # $5/day
        allowed_models=MANAGED_MODELS_ALL,
        max_output_tokens=16_384,
    ),
    # Scale-solo (ratified 2026-09-30): $49/mo, uncapped, NO seats at
    # launch (seats return later as an additive). The stamped key keeps
    # its historical name on purpose: renaming a stamped tier string is
    # the exact bug alignment=A fixed.
    "scale_49_seat": TierLimits(
        max_deadline_seconds=21_600,        # 6 h
        max_budget_micro_usd=25_000_000,    # $25
        max_concurrent_tasks=10,
        max_queued_tasks=50,
        gpu_allowed=True,
        monthly_managed_budget_micro_usd=750_000_000,   # $750/mo backstop
        fact_cap=None,                                   # unlimited
        daily_managed_budget_micro_usd=25_000_000,       # $25/day backstop
        allowed_models=MANAGED_MODELS_ALL,
        max_output_tokens=32_768,
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


def _utc_midnight(now: Optional[datetime] = None) -> datetime:
    now = now or datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


async def daily_capacity(store, customer) -> dict:
    """v108 (Q11=A, Q17=A, Q22=A): today's managed-AI capacity for one
    tenant. Counts every ledger origin that cost the platform money
    (billing != 'byok') since UTC midnight, plus yesterday's overage
    carried forward (Q17: an admitted ingest always finishes; what it
    overran comes off today's allowance; Q22=A: computed from the ledger,
    one day only, no column). Returns micro-USD figures and the state
    the meters show: ok | warning (>= 90%) | blocked (>= 100%).
    """
    tier = resolve_tier(getattr(customer, "subscription_tier", None))
    allowance = tier.daily_managed_budget_micro_usd
    if allowance <= 0:
        return {"allowance": 0, "spent": 0, "carried": 0, "state": "ok", "pct": None}
    midnight = _utc_midnight()
    yesterday = midnight - timedelta(days=1)
    spent_today = await store.platform_spend_micro_usd(customer.id, since=midnight)
    spent_yesterday = await store.platform_spend_micro_usd(
        customer.id, since=yesterday, until=midnight,
    )
    carried = max(0, spent_yesterday - allowance)
    # RC-10 (2026-10-04): the carry is capped at ONE day's allowance. An
    # ingest that overran by three days' worth would otherwise lock the
    # account for three days while the meter said "resets at midnight".
    # With the cap, tomorrow always starts clean, so resets_at is
    # exactly the next UTC midnight.
    carried = min(carried, allowance)
    used = spent_today + carried
    pct = min(100, round(used * 100 / allowance))
    state = "blocked" if used >= allowance else "warning" if pct >= 90 else "ok"
    return {
        "allowance": allowance, "spent": spent_today, "carried": carried,
        "state": state, "pct": pct,
    }


DAILY_CAPACITY_MESSAGE = (
    "Daily AI capacity is used up for this plan. It powers ingesting "
    "documents, curation, gap filling and agent runs, and resets at "
    "midnight UTC. Remembering and recall keep working. Upgrade your "
    "plan in the console for more capacity."
)


async def enforce_managed_budget(store, customer) -> None:
    """The ONE spend door, called at the top of EVERY path that can
    spend the platform's LLM money (chat proxy, agent, MCP write tools,
    ingest, cognition, consolidate, feedback, import). Reads never call
    it. byok tenants never touch it: their key, their money.

    E4 (2026-07-06): month-to-date backstop, 429.
    v108 (Q11=A): the daily allowance is enforced here too, checked ONCE
    at the door so an admitted job always finishes (Q17=A).
    """
    from fastapi import HTTPException

    if getattr(customer, "inference_mode", "byok") != "managed":
        return
    # RC-10 (2026-10-04): one monthly backstop shared with the MCP door.
    monthly = await monthly_backstop_block(store, customer)
    if monthly:
        raise PlanWallError("monthly_budget", monthly)
    cap_state = await daily_capacity(store, customer)
    if cap_state["state"] == "blocked":
        raise PlanWallError("daily_capacity", DAILY_CAPACITY_MESSAGE)


async def monthly_backstop_block(store, customer) -> Optional[str]:
    """RC-10 (2026-10-04): the month-to-date backstop, as one function
    BOTH doors call, so the MCP door and the HTTP door cannot disagree.
    Returns the refusal message, or None when under the cap."""
    tier = resolve_tier(getattr(customer, "subscription_tier", None))
    cap = tier.monthly_managed_budget_micro_usd
    if cap <= 0:
        return None
    spent = await store.platform_spend_micro_usd(customer.id, since=_utc_month_start())
    if spent >= cap:
        return (
            "Monthly managed-inference budget reached for this plan. It "
            "resets on the 1st (UTC). Upgrade your plan or switch to your "
            "own API key in Settings to continue immediately."
        )
    return None


async def daily_capacity_block(store, customer) -> Optional[dict]:
    """The MCP shape of the same door (Q12=A): a structured error the
    customer's AI client can show, in the same shape as memory_full.
    None when the tenant may proceed. RC-10: checks the monthly backstop
    first, exactly as enforce_managed_budget does."""
    if getattr(customer, "inference_mode", "byok") != "managed":
        return None
    monthly = await monthly_backstop_block(store, customer)
    if monthly:
        return {
            "error": monthly,
            "code": "monthly_budget",
            "resets_at": _next_month_start().isoformat(),
        }
    cap_state = await daily_capacity(store, customer)
    if cap_state["state"] != "blocked":
        return None
    return {
        "error": DAILY_CAPACITY_MESSAGE,
        "code": "daily_capacity",
        "resets_at": (_utc_midnight() + timedelta(days=1)).isoformat(),
    }


def _next_month_start() -> datetime:
    start = _utc_month_start()
    return (start.replace(day=28) + timedelta(days=4)).replace(day=1)


def _utc_month_start() -> datetime:
    return datetime.now(timezone.utc).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )


class PlanWallError(Exception):
    """Placeholder replaced below; kept so the name exists for type hints."""


def _make_plan_wall():
    from fastapi import HTTPException

    class _PlanWallError(HTTPException):
        """v109 (Q28, 2026-10-01): a plan wall is an HTTPException whose
        detail stays the human message (every existing reader keeps
        working) and whose X-Plan-Wall header carries a machine-readable
        code so the console can open the right upgrade modal: memory_full,
        daily_capacity, monthly_budget, model_not_in_plan, trial_expired."""

        def __init__(self, code: str, message: str, status_code: int = 429):
            self.code = code
            self.message = message
            super().__init__(
                status_code=status_code,
                detail=message,
                headers={"X-Plan-Wall": code},
            )

    return _PlanWallError


PlanWallError = _make_plan_wall()


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
    the Settings PATCH. v108 (Q13=A): the allow-list is per TIER; free
    gets Haiku and Sonnet, Opus from Starter.
    """
    from fastapi import HTTPException

    if getattr(customer, "inference_mode", "byok") != "managed":
        return
    allowed = resolve_tier(getattr(customer, "subscription_tier", None)).allowed_models
    if not model_id or model_id in allowed:
        return
    raise PlanWallError(
        "model_not_in_plan",
        "This plan's managed inference supports: "
        + ", ".join(allowed)
        + ". Upgrade for more models, or switch to your own key.",
        status_code=400,
    )


def clamp_max_tokens(customer, requested: Optional[int]) -> Optional[int]:
    """v108 (Q15=A): a managed turn never asks the model for more output
    than its tier allows. byok tenants keep what they asked for (their
    key). None stays None (the caller's own default applies)."""
    if not requested or requested <= 0:
        return requested
    if getattr(customer, "inference_mode", "byok") != "managed":
        return requested
    ceiling = resolve_tier(getattr(customer, "subscription_tier", None)).max_output_tokens
    return min(int(requested), ceiling)


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


# RC-10 (2026-10-04): `admit_task`, the original per-task enqueue gate,
# had no caller anywhere in the tree (AUDIT_FINAL S8). The tenant door is
# require_write_capacity (HTTP) / _write_admission_block (MCP) and the
# cognition worker's per-task check; a second, unused gate with its own
# rules is exactly how doors come to disagree. Removed.


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
