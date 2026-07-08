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
