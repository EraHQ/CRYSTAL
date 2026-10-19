"""Scheduled runs (launch feature, 2026-10-18): the console's routes.

Tenant-pathed under /admin/api/customers/{customer_id}/schedules so the
guard admits the owning tenant and platform admins alike (ingress/auth.py
_TENANT_PATH_RE). Creating, pausing, resuming and deleting are plain
writes: never capacity-gated, never counted against the fact cap (the
run itself pays the daily door when the cognition worker executes it).
Q81=B caps active + paused schedules per tier.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Annotated, Any, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from ..control.admission import resolve_tier
from ..infrastructure import MetadataStore
from ..infrastructure.metadata_store import get_metadata_store
from ..ingress.limits import ID64, Bound, require_bounds
from ..scheduling import build_schedule, check_deliver_url, describe, new_secret, next_occurrence, spec_from_row

logger = structlog.get_logger(__name__)

router = APIRouter()


def _view(s) -> dict[str, Any]:
    d = s.model_dump()
    d.pop("deliver_secret", None)  # shown once at creation, never again
    for k in ("run_once_at", "next_run_at", "last_run_at", "last_delivery_at", "created_at", "updated_at"):
        v = d.get(k)
        if isinstance(v, datetime):
            d[k] = (v if v.tzinfo else v.replace(tzinfo=timezone.utc)).isoformat()
    d["when"] = describe(spec_from_row(s))
    d["delivers"] = bool(s.deliver_url)
    return d


async def _checked_deliver_url(raw: Any) -> Optional[str]:
    """'' / None -> None (no delivery). Otherwise the URL, or a 400 with
    the delivery module's sentence (https only, public host)."""
    url = (str(raw).strip() if raw is not None else "")
    if not url:
        return None
    reason = await asyncio.to_thread(check_deliver_url, url)
    if reason:
        raise HTTPException(status_code=400, detail=reason)
    return url


async def _customer_or_404(store: MetadataStore, customer_id: str):
    customer = await store.get_customer_by_id(customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer


@router.get("/admin/api/customers/{customer_id}/schedules")
async def list_schedules(
    customer_id: ID64,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    customer = await _customer_or_404(store, customer_id)
    rows = await store.list_scheduled_tasks(customer_id)
    cap = resolve_tier(getattr(customer, "subscription_tier", None)).max_active_schedules
    active = sum(1 for r in rows if r.status in ("active", "paused"))
    return JSONResponse(content={
        "schedules": [_view(r) for r in rows], "count": len(rows),
        "active": active, "max_active": cap,
    })


@router.post("/admin/api/customers/{customer_id}/schedules", status_code=201)
async def create_schedule(
    customer_id: ID64,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    customer = await _customer_or_404(store, customer_id)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="A JSON object is required.")
    require_bounds(body, {
        "kind": Bound(max_len=16), "cadence": Bound(max_len=16), "tz": Bound(max_len=64),
        "time": Bound(max_len=8), "weekday": Bound(max_len=16), "run_at": Bound(max_len=40),
        "question": Bound(max_len=4000), "title": Bound(max_len=200), "deliver_url": Bound(max_len=2048),
    })
    # Shape first, cap second: a malformed request is told what is wrong
    # with it before it is told the plan is full.
    kwargs, reason = build_schedule(
        kind=str(body.get("kind") or ""), cadence=str(body.get("cadence") or ""),
        tz=str(body.get("tz") or "UTC"), time=body.get("time"), weekday=body.get("weekday"),
        every_hours=body.get("every_hours"), run_at=body.get("run_at"),
        question=body.get("question"), title=body.get("title"),
    )
    if reason:
        raise HTTPException(status_code=400, detail=reason)
    deliver_url = await _checked_deliver_url(body.get("deliver_url"))
    cap = resolve_tier(getattr(customer, "subscription_tier", None)).max_active_schedules
    held = await store.count_active_scheduled_tasks(customer_id)
    if held >= cap:
        raise HTTPException(
            status_code=409,
            detail=f"This plan allows {cap} schedule{'s' if cap != 1 else ''}. Pause or delete one, or upgrade.",
        )
    secret = new_secret() if deliver_url else None
    created = await store.create_scheduled_task(
        customer_id, deliver_url=deliver_url, deliver_secret=secret, **kwargs,
    )
    logger.info("schedule.created", customer_id=customer_id, schedule_id=created.id,
                kind=created.kind, cadence=created.cadence, delivers=bool(deliver_url))
    out: dict[str, Any] = {"schedule": _view(created)}
    if secret:
        out["deliver_secret"] = secret  # the one time it is ever returned
    return JSONResponse(status_code=201, content=out)


@router.patch("/admin/api/customers/{customer_id}/schedules/{schedule_id}")
async def update_schedule(
    customer_id: ID64,
    schedule_id: ID64,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """status: paused | active (resume recomputes the next occurrence and
    clears the failure streak, which is also how a parked schedule comes
    back). deliver_url: set a webhook ('' clears it); a new secret is
    minted when a URL is set on a schedule that has none, and returned
    once."""
    await _customer_or_404(store, customer_id)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="A JSON object is required.")
    require_bounds(body, {"status": Bound(max_len=16), "deliver_url": Bound(max_len=2048)})
    current = await store.get_scheduled_task(schedule_id, customer_id)
    if current is None:
        raise HTTPException(status_code=404, detail="Schedule not found")
    out: dict[str, Any] = {}
    if "deliver_url" in body:
        deliver_url = await _checked_deliver_url(body.get("deliver_url"))
        secret = new_secret() if (deliver_url and not current.deliver_secret) else None
        current = await store.set_scheduled_task_delivery(
            schedule_id, customer_id, deliver_url=deliver_url, deliver_secret=secret,
        )
        if current is None:
            raise HTTPException(status_code=404, detail="Schedule not found")
        if secret:
            out["deliver_secret"] = secret
        if "status" not in body:
            out["schedule"] = _view(current)
            return JSONResponse(content=out)
    status = (body.get("status") or "").strip().lower()
    if status not in ("paused", "active"):
        raise HTTPException(status_code=400, detail="status must be 'paused' or 'active'.")
    next_run: Optional[datetime] = None
    if status == "active":
        next_run = next_occurrence(spec_from_row(current), datetime.now(timezone.utc))
        if next_run is None:
            raise HTTPException(status_code=409, detail="That one-time run is in the past; create a new one.")
    updated = await store.set_scheduled_task_status(schedule_id, customer_id, status, next_run_at=next_run)
    if updated is None:
        raise HTTPException(status_code=404, detail="Schedule not found")
    out["schedule"] = _view(updated)
    return JSONResponse(content=out)


@router.delete("/admin/api/customers/{customer_id}/schedules/{schedule_id}")
async def delete_schedule(
    customer_id: ID64,
    schedule_id: ID64,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    await _customer_or_404(store, customer_id)
    ok = await store.delete_scheduled_task(schedule_id, customer_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return JSONResponse(content={"deleted": True, "schedule_id": schedule_id})
