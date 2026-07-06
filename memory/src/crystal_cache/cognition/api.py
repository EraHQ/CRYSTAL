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

from .engine import get_active_environments, get_environment

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
    envs = get_active_environments(customer_id)
    return JSONResponse(content={
        "total": len(envs),
        "environments": [_env_summary(e) for e in envs],
    })


@router.get("/environments/{env_id}")
async def get_environment_detail(request: Request, env_id: str):
    """Get full detail for a specific cognition environment.

    Pinned tenants may only see their own environments: a foreign env id
    returns the same 404 as a nonexistent one (never an existence oracle
    — same posture as the B1 customer routes).
    """
    env = get_environment(env_id)
    pin = getattr(request.state, "tenant_pin", None)
    if not env or (pin and getattr(env, "customer_id", None) != pin):
        return JSONResponse(
            status_code=404,
            content={"error": f"Environment {env_id} not found"},
        )
    return JSONResponse(content=env.to_dict())


def _env_summary(env) -> dict:
    """Compact summary for the list view."""
    step_statuses = {}
    for sid, step in env.step_outputs.items():
        step_statuses[str(sid)] = {
            "action": step.action,
            "status": step.status.value,
            "duration_ms": step.duration_ms,
        }

    return {
        "id": env.id,
        "customer_id": env.customer_id,
        "status": env.status.value,
        "trigger_type": env.trigger_type,
        "goal_title": env.goal.title if env.goal else "",
        "output_type": env.output_type.value,
        "attempts": env.attempts,
        "max_attempts": env.max_attempts,
        "step_count": len(env.plan.steps) if env.plan else 0,
        "steps_complete": sum(
            1 for s in env.step_outputs.values()
            if s.status.value == "complete"
        ),
        "steps": step_statuses,
        "validation": {
            "approved": env.validation.approved,
            "score": env.validation.score,
        } if env.validation else None,
        "tokens_used": env.tokens_used,
        "cost_usd": round(env.total_cost_usd, 6),
        "created_at": env.created_at.isoformat(),
    }
