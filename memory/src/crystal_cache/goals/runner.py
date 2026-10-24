"""The goals pass: a minute loop beside the app pull, in the same worker
(role app_sync; workers/app_sync.run_app_sync_worker starts it).

For each due goal (oldest first, at most goal_sessions_per_tick):
  1. skip a tenant being deleted, or without the goals feature (beta list)
  2. the tenant's door (require_write_capacity, spend=True): at the wall
     the goal waits an hour and says why, once
  3. claim it (compare-and-set lease) so two worker instances never run
     the same check-in
  4. unplanned -> planner.plan_goal (frontier); planned -> session.run_goal_session (large)
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

WALL_WAIT = timedelta(hours=1)


async def tick(store: Any, *, encoder: Any = None, fact_store: Any = None, llm: Any = None) -> dict[str, int]:
    from .. import beta
    from ..config import get_settings
    from ..ingress.auth import require_write_capacity
    from ..llm import get_llm_client
    from .planner import plan_goal
    from .session import run_goal_session

    settings = get_settings()
    out = {"planned": 0, "ran": 0, "walled": 0}
    if not settings.enable_goals:
        return out
    client = llm or get_llm_client()
    if hasattr(client, "is_ready") and not client.is_ready():
        return out
    limit = max(1, int(settings.goal_sessions_per_tick or 4))
    lease = timedelta(seconds=float(settings.goal_wall_seconds or 240) * 2 + 120)
    for goal in await store.list_due_goals(limit=limit * 3):
        if out["planned"] + out["ran"] >= limit:
            break
        customer = await store.get_customer_by_id(goal["customer_id"])
        if customer is None or getattr(customer, "deletion_scheduled_at", None) is not None:
            continue
        if not await beta.has_feature(store, customer.id, "goals"):
            continue
        now = datetime.now(timezone.utc)
        try:
            await require_write_capacity(customer, store, spend=True)
        except Exception as wall:  # noqa: BLE001 — HTTPException 402/429: the plan's door is shut
            msg = "Paused: today's capacity is used up. The goal picks up again when it refills."
            await store.update_goal(customer.id, goal["id"], next_run_at=now + WALL_WAIT,
                                    **({} if goal.get("last_summary") == msg else {"last_summary": msg}))
            out["walled"] += 1
            logger.info("goals.walled", goal_id=goal["id"], error_type=type(wall).__name__)
            continue
        if not await store.claim_goal(goal["id"], now=now, lease_until=now + lease):
            continue
        if goal.get("planned_at") is None:
            res = await plan_goal(store, customer, goal, llm=llm)
            out["planned"] += 1 if res.get("ok") else 0
            continue
        await run_goal_session(store=store, customer=customer, goal=goal, encoder=encoder, fact_store=fact_store,
                               llm=llm)
        out["ran"] += 1
    return out


async def run_goals_loop(*, store: Any, encoder: Any = None, fact_store: Any = None,
                         shutdown_event: asyncio.Event, interval: float = 60.0) -> None:
    logger.info("goals_loop.started", interval_seconds=interval)
    while not shutdown_event.is_set():
        try:
            out = await tick(store, encoder=encoder, fact_store=fact_store)
            if any(out.values()):
                logger.info("goals_loop.pass", **out)
        except Exception:  # noqa: BLE001
            logger.exception("goals_loop.pass_failed")
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            continue
    logger.info("goals_loop.stopped")
