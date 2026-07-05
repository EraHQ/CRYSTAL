"""In-process rate limiting (C3, 2026-07-03).

A sliding-window limiter guarding the auth-adjacent and expensive routes
before the public endpoint goes live. Deliberately dependency-free and
in-process: correct for self-host (one API container) and for the initial
single-instance Cloud Run deployment. When the hosted plane scales past
one API instance, the limiter's per-instance windows still bound abuse
(each instance enforces the limit independently) but the global rate is
limits × instances — the documented follow-on is a Redis-backed window
(DEPLOYMENT_GUIDE.md, scaling section).

Keying: the caller's Bearer token (hashed) when present — a per-customer
limit that survives NAT and proxies — else the client address, so
unauthenticated probes of the auth endpoints are bounded per source.

Route classes (config-driven, generous defaults so normal use never
trips):
  auth       — customer creation, Drive OAuth: tight (default 20/min)
  expensive  — chat completions, document upload, retrieval: moderate
               (default 120/min)
  everything else — unlimited (read paths are cheap and already authed)
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections import deque
from typing import Callable, Optional

from starlette.requests import Request
from starlette.responses import JSONResponse

_AUTH_PREFIXES = (
    "/v1/customers",
    "/v1/connectors/gdrive/auth-url",
    "/v1/connectors/gdrive/callback",
)
_EXPENSIVE_PREFIXES = (
    "/v1/chat/completions",
    "/v1/documents",
    "/v1/retrieve",
    "/v1/learn",
    "/v1/store",
)


class SlidingWindowLimiter:
    """Thread-safe sliding-window counter: allow() is O(evictions)."""

    def __init__(self, limit_per_minute: int, *, window_seconds: float = 60.0):
        self.limit = int(limit_per_minute)
        self.window = float(window_seconds)
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, *, now: Optional[float] = None) -> bool:
        if self.limit <= 0:
            return True  # 0/negative = this class is unlimited
        t = time.monotonic() if now is None else now
        cutoff = t - self.window
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and q[0] <= cutoff:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(t)
            return True


def _client_key(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer ") and len(auth) > 7:
        return "tok:" + hashlib.sha256(auth[7:].encode()).hexdigest()[:32]
    client = request.client.host if request.client else "unknown"
    return "ip:" + client


def build_rate_limit_middleware(
    *,
    auth_per_minute: int,
    expensive_per_minute: int,
) -> Callable:
    """A pure-ASGI-style HTTP middleware function for app.middleware("http").

    Built as a factory so limits come from settings at app assembly and
    tests can build a tiny app with tiny limits.
    """
    auth_limiter = SlidingWindowLimiter(auth_per_minute)
    expensive_limiter = SlidingWindowLimiter(expensive_per_minute)

    async def _middleware(request: Request, call_next):
        path = request.url.path
        limiter = None
        if any(path.startswith(p) for p in _AUTH_PREFIXES):
            limiter = auth_limiter
        elif any(path.startswith(p) for p in _EXPENSIVE_PREFIXES):
            limiter = expensive_limiter
        if limiter is not None and not limiter.allow(_client_key(request)):
            return JSONResponse(
                status_code=429,
                content={"error": {
                    "message": "rate limit exceeded — retry shortly",
                    "type": "rate_limit_exceeded",
                }},
                headers={"Retry-After": "30"},
            )
        return await call_next(request)

    return _middleware
