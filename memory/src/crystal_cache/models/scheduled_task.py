"""Scheduled run (launch feature, 2026-10-18): the Pydantic view of a
scheduled_tasks row the routes and MCP tools return."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel

ScheduleStatus = Literal["active", "paused", "parked", "done"]


class ScheduledTask(BaseModel):
    id: str
    customer_id: str
    kind: str
    title: str
    prompt: str
    cadence: str
    at_minute: Optional[int] = None
    weekday: Optional[int] = None
    every_hours: Optional[int] = None
    run_once_at: Optional[datetime] = None
    tz: str = "UTC"
    next_run_at: Optional[datetime] = None
    status: str = "active"
    series_failures: int = 0
    last_task_id: Optional[str] = None
    last_run_at: Optional[datetime] = None
    last_outcome: Optional[str] = None
    last_result_summary: Optional[str] = None
    last_crystal_id: Optional[str] = None
    deliver_url: Optional[str] = None
    # Never serialized to a client after creation; the routes and tools
    # return it ONCE from the create call and strip it everywhere else.
    deliver_secret: Optional[str] = None
    last_delivery: Optional[str] = None
    last_delivery_at: Optional[datetime] = None
    created_by_operator_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
