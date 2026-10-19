"""Scheduled runs as agent tools (launch feature, 2026-10-18; Q78-Q81).

Three registry tools the MCP wrappers (memory_schedule / memory_schedules
/ memory_unschedule) and the chat agent share: schedule_run creates a
Brief or a Research schedule, list_schedules shows them with the last
outcome, unschedule_run deletes one. Creating is a plain write (never
capacity-gated); the run pays the daily door when the cognition worker
executes it. The same builder as the console routes, so both surfaces
accept the same shapes.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from ..tool_registry import register_tool
from .retrievers import _get_state
from ...control.admission import resolve_tier
from ...scheduling import build_schedule, check_deliver_url, describe, new_secret, spec_from_row

logger = structlog.get_logger(__name__)


def _iso(v: Optional[datetime]) -> Optional[str]:
    if v is None:
        return None
    return (v if v.tzinfo else v.replace(tzinfo=timezone.utc)).isoformat()


def _public(s) -> dict[str, Any]:
    return {
        "schedule_id": s.id, "kind": s.kind, "title": s.title, "when": describe(spec_from_row(s)),
        "question": s.prompt if s.kind == "research" else None, "status": s.status,
        "next_run_at": _iso(s.next_run_at), "last_run_at": _iso(s.last_run_at),
        "last_outcome": s.last_outcome, "tz": s.tz,
        "last_result_summary": s.last_result_summary, "last_crystal_id": s.last_crystal_id,
        "delivers": bool(s.deliver_url), "deliver_url": s.deliver_url,
        "last_delivery": s.last_delivery, "last_delivery_at": _iso(s.last_delivery_at),
    }


@register_tool(
    name="schedule_run",
    description=(
        "Schedule a run that happens on its own. question: what the run "
        "should do, in the person's words (it can read the web and this "
        "knowledge bank; it does not act on other services yet). kind is "
        "optional: 'research' (default, open-ended) or 'brief' (the fixed "
        "wording: what changed in memory, open gaps, what waits for review; "
        "no question needed). cadence: once | daily | weekdays | weekly | "
        "hourly. Times are HH:MM in tz (an IANA zone, default UTC). "
        "deliver_url (optional, https, public host): a Slack or Discord "
        "webhook or any URL (Zapier, Make, n8n) the result is posted to when "
        "a run completes; the signing secret is returned once. A run spends "
        "the tenant's daily AI capacity; when there is none that occurrence "
        "is skipped. Write-side: agent-only."
    ),
    contexts={"agent"},
    parameters_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "kind": {"type": "string", "enum": ["brief", "research"], "description": "Optional; research (default) or brief."},
            "cadence": {"type": "string", "enum": ["once", "daily", "weekdays", "weekly", "hourly"]},
            "time": {"type": "string", "maxLength": 8, "description": "HH:MM in tz, for daily / weekdays / weekly."},
            "weekday": {"type": "string", "maxLength": 16, "description": "monday .. sunday, for weekly."},
            "every_hours": {"type": "integer", "minimum": 1, "maximum": 168, "description": "For hourly."},
            "run_at": {"type": "string", "maxLength": 40, "description": "ISO date-time with zone, for once."},
            "tz": {"type": "string", "maxLength": 64, "description": "IANA time zone, e.g. America/New_York.", "default": "UTC"},
            "question": {"type": "string", "maxLength": 4000, "description": "What the run should do, in the person's words (not needed for brief)."},
            "title": {"type": "string", "maxLength": 200, "description": "Optional name shown in the console."},
            "deliver_url": {"type": "string", "maxLength": 2048, "description": "Optional https webhook the result is posted to."},
        },
        "required": ["cadence"],
    },
    returns_description="{'scheduled': bool, 'schedule'?: {...}, 'deliver_secret'?: str, 'error'?: str, 'code'?: str}",
)
async def schedule_run(
    customer_id: str,
    cadence: str,
    kind: Optional[str] = None,
    time: Optional[str] = None,
    weekday: Optional[str] = None,
    every_hours: Optional[int] = None,
    run_at: Optional[str] = None,
    tz: str = "UTC",
    question: Optional[str] = None,
    title: Optional[str] = None,
    deliver_url: Optional[str] = None,
) -> dict[str, Any]:
    store = _get_state()["store"]
    customer = await store.get_customer_by_id(customer_id)
    if customer is None:
        return {"scheduled": False, "error": "Customer not found.", "code": "not_found"}
    # Shape first, cap second (same order as the console route).
    kwargs, reason = build_schedule(
        kind=kind, cadence=cadence, tz=tz, time=time, weekday=weekday,
        every_hours=every_hours, run_at=run_at, question=question, title=title,
    )
    if reason:
        return {"scheduled": False, "error": reason, "code": "bad_arguments"}
    deliver_url = (deliver_url or "").strip() or None
    if deliver_url:
        bad = await asyncio.to_thread(check_deliver_url, deliver_url)
        if bad:
            return {"scheduled": False, "error": bad, "code": "bad_deliver_url"}
    cap = resolve_tier(getattr(customer, "subscription_tier", None)).max_active_schedules
    held = await store.count_active_scheduled_tasks(customer_id)
    if held >= cap:
        return {
            "scheduled": False,
            "error": f"This plan allows {cap} schedule{'s' if cap != 1 else ''}. Pause or delete one, or upgrade.",
            "code": "schedule_cap",
        }
    secret = new_secret() if deliver_url else None
    created = await store.create_scheduled_task(
        customer_id, deliver_url=deliver_url, deliver_secret=secret, **kwargs,
    )
    logger.info("schedule.created_by_tool", customer_id=customer_id, schedule_id=created.id,
                kind=created.kind, cadence=created.cadence, delivers=bool(deliver_url))
    out: dict[str, Any] = {"scheduled": True, "schedule": _public(created)}
    if secret:
        out["deliver_secret"] = secret
        out["note"] = "Keep the secret: it signs every delivery (X-Crystal-Signature) and is not shown again."
    return out


@register_tool(
    name="list_schedules",
    description="List this memory's scheduled runs with their next run and last outcome.",
    contexts={"agent"},
    parameters_schema={"type": "object", "additionalProperties": False, "properties": {}},
    returns_description="{'schedules': [...], 'count': int}",
)
async def list_schedules(customer_id: str) -> dict[str, Any]:
    store = _get_state()["store"]
    rows = await store.list_scheduled_tasks(customer_id)
    return {"schedules": [_public(r) for r in rows], "count": len(rows)}


@register_tool(
    name="unschedule_run",
    description="Delete a scheduled run by its schedule_id (from list_schedules). Write-side: agent-only.",
    contexts={"agent"},
    parameters_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {"schedule_id": {"type": "string", "maxLength": 64}},
        "required": ["schedule_id"],
    },
    returns_description="{'deleted': bool, 'error'?: str, 'code'?: str}",
)
async def unschedule_run(customer_id: str, schedule_id: str) -> dict[str, Any]:
    store = _get_state()["store"]
    ok = await store.delete_scheduled_task((schedule_id or "").strip(), customer_id)
    if not ok:
        return {"deleted": False, "error": "No schedule with that id.", "code": "not_found"}
    return {"deleted": True, "schedule_id": schedule_id}
