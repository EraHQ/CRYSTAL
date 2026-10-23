"""The Steward's console routes (Q100 A, 2026-10-23). Tenant-pathed under
/admin/api/customers/{customer_id}/... so the tenant itself and a platform
admin reach them (ingress/auth _TENANT_PATH_RE), nobody else.

GET   .../steward                      the switch, today's state, the
                                       latest brief, the action ledger
PATCH .../steward {enabled}            "Let Crystal run itself"
POST  .../steward/actions/{id}/undo    the 24-hour undo
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from ..infrastructure import MetadataStore
from ..infrastructure.metadata_store import get_metadata_store
from ..ingress.errors import build_error_envelope
from ..ingress.limits import ID64

router = APIRouter()


def _utc_midnight() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


async def steward_status(store: MetadataStore, customer_id: str) -> dict[str, Any]:
    """The one read behind the console view and the status tool."""
    from ..config import get_settings

    customer = await store.get_customer_by_id(customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    midnight = _utc_midnight()
    events = await store.list_curation_events(customer_id, limit=50)
    paused = None
    latest_session = None
    for e in events:
        at = e.get("created_at")
        if at is not None and at.tzinfo is None:
            at = at.replace(tzinfo=timezone.utc)
        if e["event_type"] == "steward_paused" and paused is None and at is not None and at >= midnight:
            payload = e.get("payload") or {}
            paused = {"code": payload.get("code"), "message": payload.get("message"), "at": at.isoformat(),
                      "resets_at": (midnight + timedelta(days=1)).isoformat()}
        if e["event_type"] == "steward_session" and latest_session is None:
            latest_session = e
    sessions_today = sum(
        1 for e in events
        if e["event_type"] == "steward_session" and e.get("created_at") is not None
        and (e["created_at"].replace(tzinfo=timezone.utc) if e["created_at"].tzinfo is None else e["created_at"]) >= midnight
    )
    actions = await store.list_steward_actions(customer_id, limit=100)
    actions_today = await store.count_steward_actions_since(customer_id, midnight)
    latest: dict[str, Any] | None = None
    if latest_session is not None:
        run_id = (latest_session.get("payload") or {}).get("run_id") or latest_session.get("subject_id")
        task = await store.get_cognition_task(run_id) if run_id else None
        result = (getattr(task, "result", None) or {}) if task is not None else {}
        at = latest_session.get("created_at")
        latest = {
            "run_id": run_id,
            "at": at.isoformat() if at is not None else None,
            "brief": (result.get("brief") or ""),
            "actions": int((latest_session.get("payload") or {}).get("actions") or 0),
            "ok": bool((latest_session.get("payload") or {}).get("ok", True)),
            "status": getattr(task, "status", None),
        }
    settings = get_settings()
    return {
        "customer_id": customer_id,
        "enabled": bool(getattr(customer, "steward_enabled", False)),
        "platform_enabled": bool(getattr(settings, "enable_steward", True)),
        "today": {"sessions": sessions_today, "actions": actions_today,
                  "sessions_per_day": int(getattr(settings, "steward_sessions_per_day", 12))},
        "paused": paused,
        "latest": latest,
        "actions": actions,
    }


@router.get("/admin/api/customers/{customer_id}/steward")
async def get_steward(
    customer_id: ID64,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    return JSONResponse(content=await steward_status(store, customer_id))


@router.patch("/admin/api/customers/{customer_id}/steward")
async def set_steward(
    customer_id: ID64,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """The switch. Body: {"enabled": true | false}."""
    body = await request.json()
    enabled = body.get("enabled") if isinstance(body, dict) else None
    if not isinstance(enabled, bool):
        raise HTTPException(status_code=400, detail="enabled must be true or false")
    updated = await store.set_customer_steward_enabled(customer_id, enabled)
    if updated is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    try:
        await store.record_curation_event(
            customer_id, event_type="steward_enabled" if enabled else "steward_disabled", subject_id=None,
            label="Crystal now runs itself" if enabled else "Crystal no longer runs itself",
        )
    except Exception:  # noqa: BLE001 — witness only
        pass
    return JSONResponse(content={"customer_id": customer_id, "enabled": enabled})


@router.post("/admin/api/customers/{customer_id}/steward/actions/{action_id}/undo")
async def undo_steward(
    customer_id: ID64,
    action_id: ID64,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    from ..cognition.steward import undo_steward_action

    out = await undo_steward_action(store, customer_id, action_id)
    if not out.get("undone"):
        code = out.get("code") or "undo_failed"
        status = 404 if code == "not_found" else 409
        raise HTTPException(status_code=status, detail=build_error_envelope(
            out.get("error") or _UNDO_MESSAGES.get(code, "Could not undo that."),
            error_type="invalid_request_error", param="action_id", code=code,
        ))
    return JSONResponse(content=out)


_UNDO_MESSAGES = {
    "already_undone": "That was already undone.",
    "undo_expired": "The 24-hour undo window has passed.",
    "not_undoable": "That action has nothing to undo.",
    "already_open": "That conflict is already open.",
}
