"""Load-aware idle detection (BACKLOG §3 remainder, 2026-07-02).

Core Principle #1: no component starves another — API routes have highest
priority. The cognition worker's opportunistic idle work (gap fill, the
convergence scans, tier promotion) should run when the deployment is QUIET,
not merely when the cognition queue is empty. This module is the quiet
signal: the API stamps request activity, the worker asks how long ago the
last substantive request was.

Scope (v1, deliberate): the stamp is IN-PROCESS (a monotonic timestamp).
That is exactly right for the single-process shape (uvicorn app with
CC_RUN_WORKERS=true — dev and the single-container self-host). In the
split-process compose shape (api + worker services) the worker process
never sees the api's stamps, so seconds_since_last_request() stays inf and
the gate is INERT — identical behavior to before this module existed, no
regression. The cross-process signal (a MAX(created_at) probe over recent
activity tables) is the noted follow-up on the same backlog item.

Only substantive traffic should count: the API stamps /v1/* paths only,
so an open admin dashboard's polling never starves the idle work.
"""
from __future__ import annotations

import time
from typing import Optional

_last_request_monotonic: Optional[float] = None


def note_request() -> None:
    """Stamp request activity (called by the app middleware for /v1/*)."""
    global _last_request_monotonic
    _last_request_monotonic = time.monotonic()


def seconds_since_last_request() -> float:
    """Seconds since the last stamped request; +inf when never stamped."""
    if _last_request_monotonic is None:
        return float("inf")
    return time.monotonic() - _last_request_monotonic


def is_quiet(quiet_seconds: int) -> bool:
    """True when the deployment has been quiet long enough for idle work.

    quiet_seconds <= 0 disables the gate (always quiet — the pre-gate
    behavior).
    """
    if quiet_seconds <= 0:
        return True
    return seconds_since_last_request() >= float(quiet_seconds)


def reset_for_tests() -> None:
    global _last_request_monotonic
    _last_request_monotonic = None
