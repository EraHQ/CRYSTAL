"""Schedule parsing and recurrence math for CRYS background tasks.

Pure stdlib, no imports from the rest of the package — the daemon and
CLI call these; harnesses can exercise every edge without a database.

Design (locked 2026-06-12):
- Recurrence is FIXED-RATE against the wall clock, anchored at run_at:
  occurrences fire at anchor + k*interval. Never finish-time + interval
  — "daily at 09:00" must mean 09:00, not 09:00 plus yesterday's
  runtime creep.
- Missed occurrences SKIP: if the daemon was down three days, the task
  runs at the next future slot once — not three times in a burst.
- Times are local machine time, like the queue itself (project_dir is
  machine-local). Stated in docs, not apologized for.
- "AT 8" is implemented precisely (claimable the moment the clock hits
  8). "BY 8" is deadline logic — it needs duration prediction the
  system doesn't have yet; run-history timestamps are already being
  collected for a future auto-lead feature.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

# Boundary rule (the store stamps rows in UTC; users think in local
# time): PARSE local, STORE UTC, DISPLAY local. These two helpers are
# the only crossing points — everything between them stays UTC-aware.


def local_to_utc(local_naive: datetime) -> datetime:
    """A naive local datetime (what parse_at returns) -> aware UTC."""
    local_tz = datetime.now().astimezone().tzinfo
    return local_naive.replace(tzinfo=local_tz).astimezone(timezone.utc)


def utc_to_local(dt: datetime) -> datetime:
    """An aware-or-naive UTC datetime (store rows) -> aware local."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone()

# Floor for recurrence intervals. Each occurrence is a full agent run
# (LLM calls, encoder warm-up); anything tighter than a minute is a
# budget hammer nobody means.
MIN_RECUR_SECONDS = 60

_EVERY_SUGAR = {
    "hourly": 3600,
    "daily": 86400,
    "weekly": 604800,
}
_EVERY_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
_EVERY_RE = re.compile(r"^(\d+)\s*([smhdw])$")


def parse_every(text: str) -> int:
    """'30m' / '4h' / '1d' / '2w' / 'daily' / 'hourly' / 'weekly' -> seconds.

    Raises ValueError with a human-readable message on anything else —
    the message is shown to the user verbatim, so it carries examples.
    """
    s = text.strip().lower()
    if s in _EVERY_SUGAR:
        return _EVERY_SUGAR[s]
    m = _EVERY_RE.match(s)
    if not m:
        raise ValueError(
            f"couldn't read the interval {text!r} — use forms like "
            "'30m', '4h', '1d', '2w', or 'hourly'/'daily'/'weekly'."
        )
    seconds = int(m.group(1)) * _EVERY_UNITS[m.group(2)]
    if seconds < MIN_RECUR_SECONDS:
        raise ValueError(
            f"interval {text!r} is under the {MIN_RECUR_SECONDS}s minimum — "
            "each occurrence is a full agent run; tighter schedules only "
            "burn budget."
        )
    return seconds


def parse_at(text: str, now: datetime | None = None) -> datetime:
    """'09:00' / '2026-06-13 09:00' -> a future datetime (local time).

    A bare time that's already passed today means tomorrow — saying
    'at 09:00' at 10am is a request for the next 09:00 that exists.
    An explicit full datetime in the past is an error, not a guess.
    """
    now = now or datetime.now()
    s = text.strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
        try:
            dt = datetime.strptime(s, fmt)
        except ValueError:
            continue
        if dt <= now:
            raise ValueError(
                f"{text!r} is in the past — for a recurring series that "
                "should have started already, give the next future "
                "occurrence as the anchor."
            )
        return dt
    try:
        t = datetime.strptime(s, "%H:%M").time()
    except ValueError:
        raise ValueError(
            f"couldn't read the time {text!r} — use 'HH:MM' (next "
            "occurrence of that local time) or 'YYYY-MM-DD HH:MM'."
        ) from None
    candidate = datetime.combine(now.date(), t)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


def next_occurrence(anchor: datetime, interval_seconds: int, now: datetime) -> datetime:
    """The smallest anchor + k*interval that is strictly in the future.

    Called after a run completes to schedule its successor. The math
    does the skip-missed policy for free: however long the daemon was
    down or the run overran, k lands on the next FUTURE slot, exactly
    on the wall-clock grid the anchor defined.
    """
    if interval_seconds <= 0:
        raise ValueError("interval must be positive")
    if now < anchor:
        return anchor
    elapsed = (now - anchor).total_seconds()
    k = int(elapsed // interval_seconds) + 1
    return anchor + timedelta(seconds=k * interval_seconds)


def describe_schedule(run_at: datetime | None, recur_seconds: int | None) -> str:
    """One human line for prompts and /tasks: '', 'at 06-13 09:00',
    'every 1d from 06-13 09:00'. run_at is store-format (UTC);
    display is local."""
    local = utc_to_local(run_at) if run_at else None
    if recur_seconds:
        base = f"every {_human_interval(recur_seconds)}"
        if local:
            base += f" from {local.strftime('%m-%d %H:%M')}"
        return base
    if local:
        return f"at {local.strftime('%m-%d %H:%M')}"
    return ""


def _human_interval(seconds: int) -> str:
    for unit, size in (("w", 604800), ("d", 86400), ("h", 3600), ("m", 60)):
        if seconds % size == 0:
            return f"{seconds // size}{unit}"
    return f"{seconds}s"
