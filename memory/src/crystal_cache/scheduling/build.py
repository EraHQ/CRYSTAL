"""Turn a user's request (console form or MCP call) into a stored
schedule: validation, the Brief prompt, the first occurrence. Shared by
endpoints/schedules.py and the schedule_run agent tool so both surfaces
accept exactly the same shapes and refuse with the same sentences."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from .recurrence import BRIEF_PROMPT, CADENCES, KINDS, ScheduleSpec, next_occurrence, spec_from_row, validate_spec

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
MAX_PROMPT = 4000
MAX_TITLE = 200


def parse_time(text: Optional[str]) -> Optional[int]:
    """'08:00' / '8:00' / '17:30' -> minutes after midnight; None when
    absent or malformed."""
    if not text:
        return None
    s = str(text).strip().lower()
    try:
        hh, mm = s.split(":")
        h, m = int(hh), int(mm)
    except (ValueError, AttributeError):
        return None
    if not (0 <= h < 24 and 0 <= m < 60):
        return None
    return h * 60 + m


def parse_weekday(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value if 0 <= value <= 6 else None
    s = str(value).strip().lower()
    if s.isdigit():
        n = int(s)
        return n if 0 <= n <= 6 else None
    for i, name in enumerate(WEEKDAYS):
        if name.startswith(s[:3]):
            return i
    return None


def parse_run_at(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        return None
    return dt.astimezone(timezone.utc)


def build_schedule(
    *,
    cadence: str,
    kind: Optional[str] = None,
    tz: str = "UTC",
    time: Optional[str] = None,
    weekday: Any = None,
    every_hours: Any = None,
    run_at: Any = None,
    question: Optional[str] = None,
    title: Optional[str] = None,
    now: Optional[datetime] = None,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """(create kwargs, None) or (None, reason)."""
    # Anthony, 2026-10-18 23:32: "the what should be open ended". The
    # task is the person's own words; kind is optional and defaults to
    # research (the open-ended run: web + knowledge bank). "brief" stays
    # as the fixed-wording preset for callers that ask for it.
    kind = (kind or "research").strip().lower()
    cadence = (cadence or "").strip().lower()
    if kind not in KINDS:
        return None, f"kind must be one of {', '.join(KINDS)}."
    if cadence not in CADENCES:
        return None, f"cadence must be one of {', '.join(CADENCES)}."
    if kind == "research":
        prompt = (question or "").strip()
        if not prompt:
            return None, "Say what the run should do."
        if len(prompt) > MAX_PROMPT:
            return None, f"The task is too long (over {MAX_PROMPT} characters)."
    else:
        prompt = BRIEF_PROMPT
    try:
        hours = int(every_hours) if every_hours not in (None, "") else None
    except (TypeError, ValueError):
        return None, "every_hours must be a whole number of hours."
    spec = ScheduleSpec(
        cadence=cadence, tz=(tz or "UTC").strip() or "UTC",
        at_minute=parse_time(time) if cadence in ("daily", "weekdays", "weekly") else None,
        weekday=parse_weekday(weekday) if cadence == "weekly" else None,
        every_hours=hours if cadence == "hourly" else None,
        run_once_at=parse_run_at(run_at) if cadence == "once" else None,
    )
    if cadence in ("daily", "weekdays", "weekly") and spec.at_minute is None:
        return None, "A time of day is needed, as HH:MM in the schedule's time zone (for example 08:00)."
    if cadence == "weekly" and spec.weekday is None:
        return None, "A weekly run needs a weekday (monday .. sunday)."
    if cadence == "once" and spec.run_once_at is None:
        return None, "A one-time run needs run_at as an ISO date-time with a time zone (for example 2026-10-20T09:00:00-04:00)."
    reason = validate_spec(spec)
    if reason:
        return None, reason
    now = now or datetime.now(timezone.utc)
    first = next_occurrence(spec, now)
    if first is None:
        return None, "That time is already in the past."
    clean_title = (title or "").strip()[:MAX_TITLE] or default_title(kind, spec, prompt)
    return {
        "kind": kind, "title": clean_title, "prompt": prompt, "cadence": cadence, "tz": spec.tz,
        "at_minute": spec.at_minute, "weekday": spec.weekday, "every_hours": spec.every_hours,
        "run_once_at": spec.run_once_at, "next_run_at": first,
    }, None


def default_title(kind: str, spec: ScheduleSpec, prompt: str) -> str:
    what = "Brief" if kind == "brief" else prompt.strip()[:60]
    return f"{what} ({describe(spec)})"


def run_context(sched, *, now: Optional[datetime] = None) -> str:
    """The conversation_context a scheduled run hands the cognition
    engine. Without it "news relevant to my knowledge base" is a
    question about nothing: the engine must learn that "my memory" is
    this tenant's bank (survey it first), and that "new" means since the
    previous run."""
    when = describe(spec_from_row(sched))
    last = getattr(sched, "last_run_at", None)
    since = last.isoformat() if isinstance(last, datetime) else "never (this is the first run)"
    now_s = (now or datetime.now(timezone.utc)).isoformat()
    return (
        f"This is a scheduled run named '{sched.title}' ({sched.kind}; {when}), started {now_s} "
        f"with nobody in the conversation. 'My memory', 'my knowledge base' and 'my notes' mean "
        f"this tenant's Crystal bank: survey it first (key scan, then search the subjects you find) "
        f"and judge relevance against those subjects. 'News', 'new' and 'what changed' mean since "
        f"the previous run of this schedule: {since}. Prefer sources dated after that. Write the "
        f"result as a dated note that stands on its own, cite every source, and say plainly when "
        f"nothing relevant changed."
    )


def describe(spec: ScheduleSpec) -> str:
    """'Weekdays at 08:00 America/New_York', 'Every 6 hours', ..."""
    def hhmm(m: Optional[int]) -> str:
        m = int(m or 0)
        return f"{m // 60:02d}:{m % 60:02d}"
    if spec.cadence == "once":
        return f"Once at {spec.run_once_at.isoformat() if spec.run_once_at else '?'}"
    if spec.cadence == "hourly":
        return f"Every {spec.every_hours} hour{'s' if (spec.every_hours or 1) != 1 else ''}"
    when = f"at {hhmm(spec.at_minute)} {spec.tz}"
    if spec.cadence == "daily":
        return f"Daily {when}"
    if spec.cadence == "weekdays":
        return f"Weekdays {when}"
    day = WEEKDAYS[int(spec.weekday or 0)].capitalize()
    return f"Every {day} {when}"
