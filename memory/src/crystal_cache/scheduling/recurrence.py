"""Recurrence presets (Q79=A) in the tenant's time zone.

Presets: once (a UTC instant), daily / weekdays / weekly at a local
wall-clock minute, every N hours. Local wall-clock is resolved through
zoneinfo so a daily 8:00 AM America/New_York stays 8:00 AM across the
DST change (2026-11-01). Missed occurrences are never backfilled: the
next occurrence is always computed from "now", not from the one that
was missed (Q79 note).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

KINDS: tuple[str, ...] = ("brief", "research")
CADENCES: tuple[str, ...] = ("once", "daily", "weekdays", "weekly", "hourly")

# Q78=A: the Brief is a fixed prompt.
BRIEF_PROMPT = (
    "Summarize what changed in my memory since the last brief, the open "
    "gaps, and what is waiting for my review."
)


@dataclass(frozen=True)
class ScheduleSpec:
    cadence: str
    tz: str = "UTC"
    at_minute: Optional[int] = None      # minutes after local midnight
    weekday: Optional[int] = None        # 0 = Monday .. 6 = Sunday
    every_hours: Optional[int] = None
    run_once_at: Optional[datetime] = None  # UTC


def validate_spec(spec: ScheduleSpec) -> Optional[str]:
    """None when the spec is complete and sane, else the reason (a
    user-facing sentence; callers return it with code bad_arguments)."""
    if spec.cadence not in CADENCES:
        return f"cadence must be one of {', '.join(CADENCES)}."
    try:
        ZoneInfo(spec.tz)
    except (ZoneInfoNotFoundError, ValueError, TypeError):
        return f"Unknown time zone {spec.tz!r}; use an IANA name such as America/New_York."
    if spec.cadence == "once":
        if spec.run_once_at is None or spec.run_once_at.tzinfo is None:
            return "A one-time run needs a run_at date and time with a time zone."
        return None
    if spec.cadence == "hourly":
        if not spec.every_hours or not (1 <= spec.every_hours <= 168):
            return "every_hours must be between 1 and 168."
        return None
    if spec.at_minute is None or not (0 <= spec.at_minute < 24 * 60):
        return "A time of day is needed (0 to 1439 minutes after midnight)."
    if spec.cadence == "weekly" and (spec.weekday is None or not (0 <= spec.weekday <= 6)):
        return "A weekly run needs a weekday (0 = Monday .. 6 = Sunday)."
    return None


def _local_at(day: datetime, at_minute: int, zone: ZoneInfo) -> datetime:
    """The wall-clock instant at `at_minute` on `day` (a local date),
    resolved through the zone so DST transitions keep the wall clock."""
    local = datetime(day.year, day.month, day.day, at_minute // 60, at_minute % 60, tzinfo=zone)
    return local.astimezone(timezone.utc)


def next_occurrence(spec: ScheduleSpec, after: datetime) -> Optional[datetime]:
    """The first occurrence strictly after `after` (UTC-aware), or None
    when the schedule has nothing left to run (a one-shot in the past).
    Pure: the caller stores the result as next_run_at."""
    if after.tzinfo is None:
        after = after.replace(tzinfo=timezone.utc)
    after = after.astimezone(timezone.utc)
    if spec.cadence == "once":
        if spec.run_once_at is None:
            return None
        at = spec.run_once_at.astimezone(timezone.utc)
        return at if at > after else None
    if spec.cadence == "hourly":
        hours = max(1, int(spec.every_hours or 1))
        base = after.replace(minute=0, second=0, microsecond=0)
        if hours < 24 and 24 % hours == 0:
            # Align to multiples of the interval from midnight UTC so the
            # instants are stable across ticks (every 6 hours = 00, 06, 12, 18).
            candidate = base + timedelta(hours=1)
            while candidate.hour % hours != 0:
                candidate += timedelta(hours=1)
            return candidate
        # Intervals that do not divide a day run N hours after this tick.
        return base + timedelta(hours=hours)
    zone = ZoneInfo(spec.tz)
    at_minute = int(spec.at_minute or 0)
    local_after = after.astimezone(zone)
    day = local_after.replace(hour=0, minute=0, second=0, microsecond=0)
    for _ in range(0, 15):
        candidate = _local_at(day, at_minute, zone)
        ok_day = (
            spec.cadence == "daily"
            or (spec.cadence == "weekdays" and day.weekday() < 5)
            or (spec.cadence == "weekly" and day.weekday() == int(spec.weekday or 0))
        )
        if ok_day and candidate > after:
            return candidate
        day = (day + timedelta(days=1, hours=12)).replace(hour=0, minute=0, second=0, microsecond=0)
    return None


def spec_from_row(row) -> ScheduleSpec:
    return ScheduleSpec(
        cadence=row.cadence,
        tz=row.tz or "UTC",
        at_minute=row.at_minute,
        weekday=row.weekday,
        every_hours=row.every_hours,
        run_once_at=row.run_once_at,
    )
