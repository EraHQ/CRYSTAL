"""Cognition API endpoints for the admin UI.

Provides real-time visibility into active and completed cognition
environments. Verbatim port from v1 — the router prefix matches v1
(`/admin/api/cognition`) so the inspector's existing fetches work
without modification (per R3, wire-format strings are public
contracts).

Mounted in `app.py` via `app.include_router(cognition.api.router)`.
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from starlette.responses import JSONResponse

from ..infrastructure.metadata_store import get_metadata_store

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


@router.post("/tasks/{task_id}/requeue")
async def requeue_task(request: Request, task_id: str):
    """Manual Re-run (cognition cycles, 2026-07-16): the operator half
    of the requeue mechanism the worker uses automatically. Same task
    row → same trigger → the fresh run's orchestrator sees the prior
    verdicts and any open critiques. Ownership = 404-not-an-oracle."""
    store = get_metadata_store()
    task = await store.get_cognition_task(task_id)
    pin = getattr(request.state, "tenant_pin", None)
    if task is None or (
        pin is not None and task.customer_id != pin
    ):
        return JSONResponse(status_code=404,
                            content={"error": f"Task {task_id} not found"})
    if task.status in ("pending", "running"):
        return JSONResponse(
            status_code=409,
            content={"error": f"Task {task_id} is already {task.status}"},
        )
    ok = await store.requeue_cognition_task(task_id)
    if not ok:
        return JSONResponse(status_code=409,
                            content={"error": "requeue failed"})
    return JSONResponse(status_code=200, content={
        "task_id": task_id, "status": "pending", "requeued": True,
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
