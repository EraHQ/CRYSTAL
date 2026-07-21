"""The daily background-spend gate (cost slice 1c, 2026-07-21).

Mechanism in code: the llm_calls ledger is the ground truth; each
worker cycle asks "is today's total under CC_DAILY_LLM_BUDGET_USD?"
before doing model-calling work. Cached for 60s so the check itself
costs one cheap SUM a minute across all loops. The agent's interactive
lane never consults this — a drained budget stops the bank's
background thinking, never a user's question.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from ..infrastructure.metadata_store import MetadataStore

logger = structlog.get_logger(__name__)

_CACHE_TTL_SECONDS = 60.0
_state = {"checked_at": 0.0, "exhausted": False}


def _budget_micro() -> "int | None":
    from ..config import get_settings
    budget = get_settings().daily_llm_budget_usd
    if budget is None or budget <= 0:
        return None
    return int(budget * 1_000_000)


async def llm_budget_exhausted(store: "MetadataStore") -> bool:
    """True when today's ledger spend has crossed the daily budget.
    False when no budget is configured. 60s-cached."""
    budget = _budget_micro()
    if budget is None:
        return False
    now = time.monotonic()
    if (now - _state["checked_at"]) < _CACHE_TTL_SECONDS:
        return _state["exhausted"]
    day_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    spent = await store.sum_llm_cost_since_micro(day_start)
    exhausted = spent >= budget
    if exhausted and not _state["exhausted"]:
        logger.warning(
            "worker.budget_exhausted",
            spent_micro_usd=spent, budget_micro_usd=budget,
        )
    _state["checked_at"] = now
    _state["exhausted"] = exhausted
    return exhausted


_customer_state: dict = {}


async def budget_for_customer(customer_id: str, store) -> "int | None":
    """THE DYNAMIC SWITCH POINT (rails laid 2026-07-21): today this
    returns the static CC_DAILY_LLM_BUDGET_PER_CUSTOMER_USD for every
    customer; when pricing plans exist, THIS function reads the
    customer's plan and returns its limit — no caller changes. Returns
    micro-USD, or None for no per-customer gate."""
    from ..config import get_settings
    budget = get_settings().daily_llm_budget_per_customer_usd
    if budget is None or budget <= 0:
        return None
    return int(budget * 1_000_000)


async def customer_llm_budget_exhausted(
    store, customer_id: str,
) -> bool:
    """Per-customer daily gate (60s-cached per customer). The company
    stop-loss (llm_budget_exhausted) and this gate compose: work
    proceeds only when BOTH say yes."""
    budget = await budget_for_customer(customer_id, store)
    if budget is None:
        return False
    now = time.monotonic()
    entry = _customer_state.get(customer_id)
    if entry and (now - entry["checked_at"]) < _CACHE_TTL_SECONDS:
        return entry["exhausted"]
    day_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    spent = await store.sum_llm_cost_since_micro(
        day_start, customer_id=customer_id,
    )
    exhausted = spent >= budget
    if exhausted and not (entry or {}).get("exhausted"):
        logger.warning(
            "worker.customer_budget_exhausted",
            customer_id=customer_id,
            spent_micro_usd=spent, budget_micro_usd=budget,
        )
    _customer_state[customer_id] = {
        "checked_at": now, "exhausted": exhausted,
    }
    return exhausted


def reset_budget_cache() -> None:
    """Test seam."""
    _state["checked_at"] = 0.0
    _state["exhausted"] = False
    _customer_state.clear()
