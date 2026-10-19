"""Scheduler worker (launch feature, 2026-10-18; Q78-Q81).

Every `interval_seconds`:

Phase 1, settle: every schedule whose last run is still 'queued' has its
cognition task read. Completed -> the streak resets, the result summary
and crystal id land on the row, and when the schedule has a webhook the
result is delivered (Q90=A; at-most-once, outcome recorded). Failed or
cancelled -> the streak grows; the third consecutive failure PARKS the
schedule (the daemon's series cap, mirrored). Still running -> left
alone. This phase runs for paused and parked rows too, so a run that
was in flight when the person paused still settles and delivers.

Phase 2, enqueue: for each due active schedule,
1. Claims the occurrence by advancing next_run_at with a compare-and-set
   (two scheduler processes can tick the same row; one wins).
2. Q80=A: a managed tenant with no daily AI capacity has that occurrence
   SKIPPED and recorded ("skipped_capacity"); the next one runs normally.
3. Enqueues ONE cognition task (task_type agent_research, priority
   background) carrying the prompt and a run context (what "my memory"
   and "since the last run" mean); the cognition worker does the run
   unchanged, including its own tenant door as a second guard, and
   derived results stay behind human review.

Creating, pausing or deleting a schedule is never capacity-gated; only a
run spends capacity, and it pays the same door as any cognition run.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from ..scheduling import build_event, deliver, next_occurrence, run_context, spec_from_row

logger = structlog.get_logger(__name__)

SCHEDULER_INTERVAL_SECONDS = 60
SERIES_FAILURE_CAP = 3
COGNITION_MAX_ATTEMPTS = 3
SUMMARY_CHARS = 2000


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _summary_of(task) -> str:
    result = getattr(task, "result", None) or {}
    text = result.get("findings") or result.get("reason") or result.get("recommendation") or ""
    return str(text).strip()[:SUMMARY_CHARS]


async def _settle(store, sched, *, now: datetime, deliver_fn=deliver, transport=None) -> tuple[int, bool, bool]:
    """Read the last run's cognition task and settle the row. Returns
    (series_failures, parked, settled): the updated streak, whether this
    call parked the row, and whether the run reached a terminal state."""
    failures = int(sched.series_failures or 0)
    if not sched.last_task_id or sched.last_outcome != "queued":
        return failures, False, False
    try:
        task = await store.get_cognition_task(sched.last_task_id)
    except Exception:  # noqa: BLE001 — an unreadable task is not a failure of the schedule
        return failures, False, False
    if task is None:
        return failures, False, False
    status = getattr(task, "status", None)
    ran_at = sched.last_run_at or now
    if status == "complete":
        summary = _summary_of(task)
        crystal_id = getattr(task, "result_crystal_id", None) or (getattr(task, "result", None) or {}).get("crystal_id")
        await store.record_scheduled_run(sched.id, outcome="completed", task_id=sched.last_task_id,
                                         ran_at=ran_at, series_failures=0)
        await store.record_scheduled_result(sched.id, summary=summary, crystal_id=crystal_id)
        if sched.deliver_url:
            from ..config import get_settings
            event = build_event(sched, summary=summary, crystal_id=crystal_id, ran_at=ran_at.isoformat(),
                                console_url=get_settings().console_url)
            try:
                ok, outcome = await deliver_fn(sched.deliver_url, sched.deliver_secret, event, transport=transport)
            except Exception as e:  # noqa: BLE001 — delivery never breaks settlement
                ok, outcome = False, f"failed: {type(e).__name__}"
            await store.record_scheduled_delivery(sched.id, outcome=outcome, at=now)
            (logger.info if ok else logger.warning)(
                "scheduler.delivered" if ok else "scheduler.delivery_failed",
                schedule_id=sched.id, customer_id=sched.customer_id, outcome=outcome,
            )
        return 0, False, True
    if status in ("failed", "cancelled"):
        failures += 1
        parked = failures >= SERIES_FAILURE_CAP and sched.status == "active"
        await store.record_scheduled_run(
            sched.id, outcome="failed", task_id=sched.last_task_id,
            ran_at=ran_at, series_failures=failures,
            status="parked" if parked else None,
        )
        if parked:
            logger.warning("scheduler.parked", schedule_id=sched.id, customer_id=sched.customer_id,
                           series_failures=failures)
        return failures, parked, True
    # Still pending or running: leave it; the next occurrence may still enqueue.
    return failures, False, False


async def settle_in_flight(
    store, *, now: Optional[datetime] = None, limit: int = 200, deliver_fn=deliver, transport=None,
) -> dict[str, int]:
    """Phase 1: settle every queued run whose task has finished."""
    now = now or _utcnow()
    out = {"in_flight": 0, "completed": 0, "failed": 0, "parked": 0}
    rows = await store.list_scheduled_tasks_in_flight(limit=limit)
    out["in_flight"] = len(rows)
    for sched in rows:
        try:
            failures, parked, settled = await _settle(store, sched, now=now, deliver_fn=deliver_fn, transport=transport)
            if not settled:
                continue
            if parked:
                out["parked"] += 1
            if failures == 0:
                out["completed"] += 1
            else:
                out["failed"] += 1
        except Exception as e:  # noqa: BLE001 — one bad row never stops the pass
            logger.error("scheduler.settle_failed", schedule_id=sched.id, error=str(e)[:300],
                         error_type=type(e).__name__)
    return out


async def tick(
    store, *, now: Optional[datetime] = None, limit: int = 50, deliver_fn=deliver, transport=None,
) -> dict[str, int]:
    """One scheduler pass (both phases). Separate from the loop so a test
    drives it."""
    from ..control.admission import daily_capacity_block

    now = now or _utcnow()
    out = {"due": 0, "queued": 0, "skipped_capacity": 0, "parked": 0, "lost_claim": 0, "done": 0}
    settled = await settle_in_flight(store, now=now, deliver_fn=deliver_fn, transport=transport)
    out["settled"] = settled["completed"] + settled["failed"]
    out["parked"] += settled["parked"]
    due = await store.list_due_scheduled_tasks(now, limit=limit)
    out["due"] = len(due)
    for sched in due:
        try:
            # A run still in flight at the next occurrence (hourly cadence,
            # slow run) was not settled above; read it once more here so a
            # third failure parks before anything new is enqueued.
            failures, parked, _ = await _settle(store, sched, now=now, deliver_fn=deliver_fn, transport=transport)
            if parked:
                out["parked"] += 1
                continue
            following = next_occurrence(spec_from_row(sched), now)
            claimed = await store.advance_scheduled_task(
                sched.id, expected_next_run_at=sched.next_run_at, next_run_at=following,
            )
            if not claimed:
                out["lost_claim"] += 1
                continue
            if following is None:
                out["done"] += 1
            customer = await store.get_customer_by_id(sched.customer_id)
            if customer is None:
                await store.record_scheduled_run(sched.id, outcome="failed", task_id=None, ran_at=now,
                                                 series_failures=failures + 1)
                continue
            block = await daily_capacity_block(store, customer)
            if block:
                await store.record_scheduled_run(sched.id, outcome="skipped_capacity", task_id=None, ran_at=now)
                out["skipped_capacity"] += 1
                logger.info("scheduler.skipped_capacity", schedule_id=sched.id,
                            customer_id=sched.customer_id, code=block.get("code"))
                continue
            payload: dict[str, Any] = {
                "topic": sched.prompt,
                "conversation_context": run_context(sched, now=now),
                "source_crystal_id": "",
                "output_type": "crystal",
                "max_attempts": COGNITION_MAX_ATTEMPTS,
                "schedule_id": sched.id,
                "schedule_kind": sched.kind,
                "schedule_title": sched.title,
            }
            task = await store.create_cognition_task(
                sched.customer_id, task_type="agent_research", payload=payload, priority="background",
            )
            await store.record_scheduled_run(sched.id, outcome="queued", task_id=task.id, ran_at=now,
                                             series_failures=failures)
            out["queued"] += 1
            logger.info("scheduler.queued", schedule_id=sched.id, customer_id=sched.customer_id,
                        task_id=task.id, kind=sched.kind, next_run_at=following.isoformat() if following else None)
        except Exception as e:  # noqa: BLE001 — one bad row never stops the pass
            logger.error("scheduler.row_failed", schedule_id=sched.id, error=str(e)[:300],
                         error_type=type(e).__name__)
    return out


async def run_scheduler_worker(
    *, store, shutdown_event: asyncio.Event, interval_seconds: int = SCHEDULER_INTERVAL_SECONDS,
) -> None:
    logger.info("scheduler_worker.started", interval_seconds=interval_seconds)
    while not shutdown_event.is_set():
        try:
            result = await tick(store)
            if result["due"] or result.get("settled"):
                logger.info("scheduler_worker.tick", **result)
        except Exception as e:  # noqa: BLE001
            logger.error("scheduler_worker.tick_failed", error=str(e)[:300], error_type=type(e).__name__)
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            pass
    logger.info("scheduler_worker.stopped")
