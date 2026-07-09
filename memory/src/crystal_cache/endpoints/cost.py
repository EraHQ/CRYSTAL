"""Cost-accounting endpoints — Growth G3 (cost + budgets).

The Inspector's read API over the cost ledger (llm_calls via
metadata_store_cost_ext.py). Cost visibility is **financial** — role-gated to
operator+ (viewers, the audit role, are denied; the team root key is always
admitted). Money is returned as INTEGER micro-USD; the client formats it.

Four read routes, all team-scoped to the principal's team:

  GET /v1/cost/summary       — all-time totals + average spend per agent (D6:
       the session is the averaging unit).
  GET /v1/cost/sessions      — per-session cost rollup, costliest first (the
       sortable all-time-per-agent view).
  GET /v1/cost/operators     — per-operator cost rollup, costliest first.
  GET /v1/cost/timeseries    — cost bucketed daily or weekly (the cost-over-
       time chart), ?bucket=day|week & ?days=N.

This is G3's read API. The single write choke point is record_llm_call()
(metadata_store_cost_ext.py); the chat-proxy emitter is wired behind
CC_ENABLE_COST_ACCOUNTING (other call sites — cognition / agent loop / depth /
metacognition / inline research — are wired as they're unified through the
adapter, deferred). The budget-breach → G2 auto-pause tie-back is deferred
(needs the agent-side control channel).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException

from ..infrastructure import MetadataStore
from ..infrastructure.metadata_store import get_metadata_store
from ..ingress.auth import require_role
from ..models import Customer, Operator

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.get("/v1/cost/summary")
async def cost_summary(
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(require_role("operator"))
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> dict[str, Any]:
    """All-time team totals + average spend per agent (micro-USD)."""
    customer, _actor = principal
    totals = await store.cost_totals_for_team(customer.id)
    avg_per_agent = await store.average_cost_per_agent(customer.id)
    return {
        "summary": {
            **totals,
            "average_cost_micro_usd_per_agent": avg_per_agent,
        }
    }


@router.get("/v1/cost/origins")
async def cost_origins(
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(require_role("operator"))
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    days: Optional[int] = None,
) -> dict[str, Any]:
    """Spend grouped by ledger origin (S12) — where the money goes.

    ?days=N bounds the window (default: all time). Highest spend first.
    """
    customer, _actor = principal
    since = None
    if days is not None:
        if days < 1:
            raise HTTPException(
                status_code=400, detail="days must be >= 1"
            )
        since = datetime.now(timezone.utc) - timedelta(days=days)
    origins = await store.cost_by_origin(customer.id, since=since)
    return {"origins": origins}


@router.get("/v1/cost/sessions")
async def cost_by_session(
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(require_role("operator"))
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    limit: int = 100,
) -> dict[str, Any]:
    """Per-session cost rollup for the team, costliest first."""
    customer, _actor = principal
    rows = await store.cost_by_session(customer.id, limit=limit)
    return {"sessions": rows}


@router.get("/v1/cost/operators")
async def cost_by_operator(
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(require_role("operator"))
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    limit: int = 100,
) -> dict[str, Any]:
    """Per-operator cost rollup for the team, costliest first."""
    customer, _actor = principal
    rows = await store.cost_by_operator(customer.id, limit=limit)
    return {"operators": rows}


@router.get("/v1/cost/timeseries")
async def cost_timeseries(
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(require_role("operator"))
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    bucket: str = "day",
    days: int = 30,
) -> dict[str, Any]:
    """Cost over time, bucketed daily or weekly, oldest bucket first."""
    if bucket not in ("day", "week"):
        raise HTTPException(
            status_code=400, detail="bucket must be 'day' or 'week'"
        )
    customer, _actor = principal
    series = await store.cost_timeseries(customer.id, bucket=bucket, days=days)
    return {"bucket": bucket, "days": days, "series": series}
