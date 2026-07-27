"""Cognition API endpoints for the admin UI.

Provides real-time visibility into active and completed cognition
environments. Verbatim port from v1 — the router prefix matches v1
(`/admin/api/cognition`) so the inspector's existing fetches work
without modification (per R3, wire-format strings are public
contracts).

Mounted in `app.py` via `app.include_router(cognition.api.router)`.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import structlog
from fastapi import APIRouter, Request
from starlette.responses import JSONResponse

from ..infrastructure.metadata_store import get_metadata_store

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/admin/api/cognition", tags=["cognition"])


@router.get("/environments")
async def list_environments(request: Request, customer_id: str = ""):
    """List active cognition environments.

    Tenant principals reach this route pinned (Accounts Phase A): the
    guard middleware stashes request.state.tenant_pin, which OVERRIDES any
    caller-supplied customer_id — a tenant sees exactly its own
    environments, never more, regardless of the query string. Platform
    admins arrive unpinned and keep the cross-tenant view.
    """
    pin = getattr(request.state, "tenant_pin", None)
    if pin:
        customer_id = pin
    # S9 (2026-07-08): read cognition_runs — the in-memory registry is
    # process-local (runs live in the worker; this API is a different
    # process) and completed runs deserve a surface. The stored rows
    # carry the exact summary wire shape this endpoint always served.
    store = get_metadata_store()
    runs = await store.list_cognition_runs(customer_id)
    # Q2B (2026-07-15): open-critique badges on the run list.
    counts = await store.count_open_critiques_by_run(
        [r.get("id") for r in runs if r.get("id")]
    )
    for r in runs:
        r["open_critiques"] = counts.get(r.get("id"), 0)
    return JSONResponse(content={
        "total": len(runs),
        "environments": runs,
    })


@router.get("/environments/{env_id}")
async def get_environment_detail(request: Request, env_id: str):
    """Get full detail for a specific cognition environment.

    Pinned tenants may only see their own environments: a foreign env id
    returns the same 404 as a nonexistent one (never an existence oracle
    — same posture as the B1 customer routes).
    """
    store = get_metadata_store()
    run = await store.get_cognition_run(env_id)
    pin = getattr(request.state, "tenant_pin", None)
    if not run or (pin and run.get("customer_id") != pin):
        return JSONResponse(
            status_code=404,
            content={"error": f"Environment {env_id} not found"},
        )
    return JSONResponse(content=run)


# ---------------------------------------------------------------------------
# Run critiques (Q2B, ratified 2026-07-15)
# ---------------------------------------------------------------------------
# Operator critiques pinned to parts of a run's anatomy. These are the
# ONE console write tenants may make (see ingress.auth._tenant_writable);
# ownership is enforced here with the same 404-not-an-oracle posture as
# the detail route. Open critiques feed the orchestrator on retries and
# on future runs of the same trigger — operator judgment enters the
# ratchet instead of sitting as a sticky note.


async def _owned_run(request: Request, env_id: str):
    """Load the run and enforce tenant ownership. None => respond 404."""
    store = get_metadata_store()
    run = await store.get_cognition_run(env_id)
    pin = getattr(request.state, "tenant_pin", None)
    if not run or (pin and run.get("customer_id") != pin):
        return None
    return run


@router.get("/environments/{env_id}/critiques")
async def list_critiques(request: Request, env_id: str):
    run = await _owned_run(request, env_id)
    if run is None:
        return JSONResponse(status_code=404,
                            content={"error": f"Environment {env_id} not found"})
    store = get_metadata_store()
    critiques = await store.list_run_critiques(env_id)
    return JSONResponse(content={"total": len(critiques),
                                 "critiques": critiques})


@router.post("/environments/{env_id}/critiques")
async def create_critique(request: Request, env_id: str):
    run = await _owned_run(request, env_id)
    if run is None:
        return JSONResponse(status_code=404,
                            content={"error": f"Environment {env_id} not found"})
    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        return JSONResponse(status_code=422,
                            content={"error": "text is required"})
    pin = getattr(request.state, "tenant_pin", None)
    store = get_metadata_store()
    critique = await store.create_run_critique(
        env_id,
        run.get("customer_id") or "",
        target_path=(body.get("target_path") or "run"),
        text=text[:4000],
        author="tenant" if pin else "platform_admin",
        trigger_id=run.get("trigger_id") or None,
    )
    return JSONResponse(status_code=201, content=critique)


# Stale-run reclaim (2026-07-26). A `running` task whose newest run's
# heartbeat is older than this is treated as abandoned and may be
# requeued. Chosen to sit well above the longest legitimate single step
# and far below the 40-minute hang that motivated it. The heartbeat's
# resolution is per-lifecycle-transition today, so this is deliberately
# generous rather than tight.
_STALE_RUN_MINUTES = 10


def _as_utc(value: datetime) -> datetime:
    """SQLite hands back NAIVE datetimes even for DateTime(timezone=True),
    while Postgres returns aware ones. Subtracting a naive from an aware
    raises TypeError, which would turn a reclaim into a 500 on the
    self-host shape only. Treat naive as UTC, which is what every writer
    in this codebase stores."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


@router.post("/tasks/{task_id}/requeue")
async def requeue_task(request: Request, task_id: str):
    """Manual Re-run (cognition cycles, 2026-07-16): the operator half
    of the requeue mechanism the worker uses automatically. Same task
    row → same trigger → the fresh run's orchestrator sees the prior
    verdicts and any open critiques. Ownership = 404-not-an-oracle.

    Stale-run reclaim (2026-07-26): `running` is no longer a blanket
    409. An api+worker deploy replaces the executor mid-run, leaving
    the row 'running' with no process behind it — and since
    claim_pending_cognition_task only takes 'pending', the one state
    that needed reclaiming was the one state nothing could reclaim.
    A running task whose newest run has not heartbeat in
    _STALE_RUN_MINUTES is presumed abandoned and may be requeued; a
    task still showing signs of life still 409s, because requeueing a
    LIVE run would have two executors on one task_id.
    """
    store = get_metadata_store()
    task = await store.get_cognition_task(task_id)
    pin = getattr(request.state, "tenant_pin", None)
    if task is None or (
        pin is not None and task.customer_id != pin
    ):
        return JSONResponse(status_code=404,
                            content={"error": f"Task {task_id} not found"})
    if task.status == "pending":
        return JSONResponse(
            status_code=409,
            content={"error": f"Task {task_id} is already pending"},
        )
    if task.status == "running":
        stale_after = timedelta(minutes=_STALE_RUN_MINUTES)
        beat = await store.latest_run_heartbeat_for_trigger(
            task_id, customer_id=task.customer_id,
        )
        # No run row at all: the task was claimed but the engine never
        # wrote a snapshot. Fall back to the claim time, which
        # claim_pending_cognition_task stamps on started_at.
        last_sign_of_life = beat or task.started_at
        if last_sign_of_life is None:
            return JSONResponse(
                status_code=409,
                content={
                    "error": (
                        f"Task {task_id} is running and has no heartbeat or "
                        "claim time to judge staleness by"
                    ),
                },
            )
        age = datetime.now(timezone.utc) - _as_utc(last_sign_of_life)
        if age < stale_after:
            return JSONResponse(
                status_code=409,
                content={
                    "error": (
                        f"Task {task_id} is running and still alive "
                        f"(last progress {int(age.total_seconds())}s ago; "
                        f"stale after {_STALE_RUN_MINUTES}m)"
                    ),
                },
            )
        logger.warning(
            "cognition.stale_run_reclaimed",
            task_id=task_id,
            customer_id=task.customer_id,
            stale_seconds=int(age.total_seconds()),
        )
        # Gravestone cleanup (2026-07-27): the abandoned run's snapshot
        # row would otherwise sit at 'working' forever — only its dead
        # executor could finalize it — polluting the active list beside
        # the fresh run the requeue spawns. 'failed' not 'cancelled':
        # nobody stopped it; its executor died under a deploy.
        finalized = await store.finalize_stale_runs_for_trigger(
            task_id, customer_id=task.customer_id, status="failed",
        )
        if finalized:
            logger.info("cognition.stale_runs_finalized",
                        task_id=task_id, count=finalized)
    ok = await store.requeue_cognition_task(task_id)
    if not ok:
        return JSONResponse(status_code=409,
                            content={"error": "requeue failed"})
    return JSONResponse(status_code=200, content={
        "task_id": task_id, "status": "pending", "requeued": True,
    })


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(request: Request, task_id: str):
    """Cooperative cancellation (2026-07-27). Ownership =
    404-not-an-oracle, agent-queue outcome vocabulary.

    Three live shapes, one mechanism each:
      pending — never claimed, nothing to cooperate with: finalized
        directly to 'cancelled'.
      running + fresh heartbeat — a live executor: set
        cancel_requested and let the engine stop at its next step/
        attempt boundary (never mid-LLM-call). The task goes terminal
        when the engine exits; this endpoint reports 'requested'.
      running + stale heartbeat — an orphan (executor replaced by a
        deploy; nothing alive to cooperate): finalized directly, and
        its frozen run rows are finalized too so the gravestone leaves
        the active list.
    Terminal tasks no-op with their current status, matching
    cancel_agent_task's no-op-on-terminal shape.
    """
    store = get_metadata_store()
    task = await store.get_cognition_task(task_id)
    pin = getattr(request.state, "tenant_pin", None)
    if task is None or (
        pin is not None and task.customer_id != pin
    ):
        return JSONResponse(status_code=404,
                            content={"error": f"Task {task_id} not found"})

    now = datetime.now(timezone.utc)

    if task.status in ("complete", "failed", "cancelled"):
        return JSONResponse(status_code=200, content={
            "task_id": task_id, "status": task.status,
            "cancelled": False,
            "note": f"already terminal ({task.status}) — no-op",
        })

    if task.status == "pending":
        await store.mark_cognition_task_cancelled(
            task_id, completed_at=now,
            reason="cancelled by operator before start",
        )
        logger.info("cognition.task_cancelled_pending", task_id=task_id)
        return JSONResponse(status_code=200, content={
            "task_id": task_id, "status": "cancelled", "cancelled": True,
        })

    # running — liveness decides the MECHANISM. Finalizing a LIVE run
    # directly would race its executor: a later mark_complete would
    # overwrite 'cancelled' with 'complete'. So a live run gets the
    # flag; only a dead one is finalized from here.
    beat = await store.latest_run_heartbeat_for_trigger(
        task_id, customer_id=task.customer_id,
    )
    last_sign_of_life = beat or task.started_at
    age_seconds = (
        int((now - _as_utc(last_sign_of_life)).total_seconds())
        if last_sign_of_life is not None else None
    )
    is_stale = (
        age_seconds is None
        or age_seconds >= _STALE_RUN_MINUTES * 60
    )

    if not is_stale:
        await store.request_cognition_cancel(task_id)
        logger.info("cognition.cancel_requested",
                    task_id=task_id, last_progress_seconds=age_seconds)
        return JSONResponse(status_code=200, content={
            "task_id": task_id, "status": "running",
            "cancelled": "requested",
            "note": (
                "live run — will stop at its next step boundary "
                f"(last progress {age_seconds}s ago)"
            ),
        })

    # Orphan: belt-and-braces — set the flag FIRST so a
    # pathologically-silent-but-alive executor still stops at its next
    # boundary instead of overwriting the terminal status later.
    await store.request_cognition_cancel(task_id)
    reason = (
        "orphaned run cancelled by operator "
        + (f"(no heartbeat for {age_seconds}s)" if age_seconds is not None
           else "(no heartbeat or claim time recorded)")
    )
    await store.mark_cognition_task_cancelled(
        task_id, completed_at=now, reason=reason,
    )
    finalized = await store.finalize_stale_runs_for_trigger(
        task_id, customer_id=task.customer_id, status="cancelled",
    )
    logger.info("cognition.task_cancelled_orphan",
                task_id=task_id, runs_finalized=finalized,
                stale_seconds=age_seconds)
    return JSONResponse(status_code=200, content={
        "task_id": task_id, "status": "cancelled", "cancelled": True,
        "runs_finalized": finalized,
    })


@router.patch("/critiques/{critique_id}")
async def update_critique(request: Request, critique_id: str):
    """Flip open|resolved. Tenants may only touch critiques on their
    own runs (404 on foreign, never an existence oracle)."""
    store = get_metadata_store()
    critique = await store.get_run_critique(critique_id)
    pin = getattr(request.state, "tenant_pin", None)
    if not critique or (pin and critique.get("customer_id") != pin):
        return JSONResponse(status_code=404,
                            content={"error": "Critique not found"})
    body = await request.json()
    status = (body.get("status") or "").strip()
    if status not in ("open", "resolved"):
        return JSONResponse(status_code=422,
                            content={"error": "status must be open|resolved"})
    await store.set_run_critique_status(critique_id, status)
    critique = await store.get_run_critique(critique_id)
    return JSONResponse(content=critique)
