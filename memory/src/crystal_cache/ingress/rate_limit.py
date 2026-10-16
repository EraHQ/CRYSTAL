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
B5-5 (Lockdown PR-7, 2026-10-13): behind a proxy the client address is
read from X-Forwarded-For, `CC_RATE_LIMIT_TRUSTED_PROXY_HOPS` entries
from the RIGHT (the hop the last trusted proxy appended; Cloud Run's
front end appends the real client after anything the client sent, so
the hosted plane sets 1). 0 (the self-host default) ignores the header:
with no proxy a client could forge it. uvicorn's own
--forwarded-allow-ips="*" was NOT used: with every proxy trusted it
takes the LEFTMOST entry (uvicorn/middleware/proxy_headers.py,
`always_trust`), which is the one a client can forge. Buckets are
evicted when empty and swept when the table grows large, so random
bearers cannot grow memory without bound.

Route classes (config-driven, generous defaults so normal use never
trips):
  auth       — customer creation, Drive OAuth: tight (default 20/min)
  expensive  — chat completions, document upload, retrieval, agent
               turns, bulk export/import: moderate (default 120/min)
  mcp        — the whole /mcp mount (audit item (d), Q1=B 2026-08-25):
               its own class + knob (default 240/min) because MCP
               traffic is agent-shaped — bursty, chatty reads — and all
               tools POST the same URL, so the mount is one bucket by
               construction; per-tool bounds live in the tools
               themselves (ingest char cap, export pagination)
  everything else — unlimited (read paths are cheap and already authed)
"""
from __future__ import annotations

import hashlib
import re
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
    # B5-6 / B4-9 (Lockdown PR-7): identity minting and the OAuth
    # authorization server (dynamic registration, token exchange, the
    # consent bridge) and billing.
    "/v1/auth/",
    "/v1/oauth/",
    "/register",
    "/token",
    "/authorize",
    "/v1/billing/",
)
_EXPENSIVE_PREFIXES = (
    "/v1/chat/completions",
    "/v1/documents",
    "/v1/retrieve",
    "/v1/learn",
    "/v1/store",
    # Audit item (d) riders (Q4=A, ratified 2026-08-25): three routes the
    # R11 pass found outside every class — the agent door spends model
    # budget per turn (the stage-1.1 port item, landed here), and
    # export/import are whole-bank bulk ops.
    "/v1/agent/messages",
    "/v1/export",
    "/v1/import",
    # v108 (AUDIT_LLM_SPEND G6, Q15=A, 2026-10-01): consolidate runs a
    # paid call synchronously on the event loop (~$0.14/call) and
    # feedback thumbs-down triggers a paid learning call; both were in
    # no class at all.
    "/v1/consolidate",
    "/v1/feedback",
)
_MCP_PREFIXES = ("/mcp",)
# B5-6 (Lockdown PR-7): routes that spend model money or enqueue paid
# work from the console, matched by regex (they carry an id in the path).
_EXPENSIVE_PATTERNS = (
    re.compile(r"^/admin/api/customers/[^/]+/(?:chat|agent|learn)(?:/|$)"),
    re.compile(r"^/admin/api/assumptions/[^/]+/(?:respond|verify)(?:/|$)"),
    re.compile(r"^/admin/api/conflicts/scan(?:/|$)"),
    # Console Refine PR 7b: the on-demand explanation is one paid call.
    re.compile(r"^/admin/api/conflicts/[^/]+/explain(?:/|$)"),
    re.compile(r"^/admin/api/cognition/tasks/[^/]+/requeue(?:/|$)"),
    re.compile(r"^/v1/gaps/"),
)

# B5-5: the bucket table is swept (empty and expired deques dropped)
# whenever it grows past this many keys.
SWEEP_THRESHOLD = 100_000


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
            if len(self._hits) > SWEEP_THRESHOLD:
                self._sweep(cutoff)
            q = self._hits.setdefault(key, deque())
            while q and q[0] <= cutoff:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(t)
            return True

    def _sweep(self, cutoff: float) -> None:
        """B5-5: drop every bucket whose hits are all outside the window.
        Called under the lock."""
        dead = [k for k, q in self._hits.items() if not q or q[-1] <= cutoff]
        for k in dead:
            del self._hits[k]

    def bucket_count(self) -> int:
        with self._lock:
            return len(self._hits)


def _trusted_proxy_hops() -> int:
    from ..config import get_settings

    try:
        return max(0, int(getattr(get_settings(), "rate_limit_trusted_proxy_hops", 0) or 0))
    except (TypeError, ValueError):
        return 0


def client_address(request: Request, *, hops: Optional[int] = None) -> str:
    """The caller's address for an anonymous bucket (B5-5). With `hops`
    trusted proxies in front, the X-Forwarded-For entry `hops` from the
    right is the one the last trusted proxy appended (the real client
    behind Cloud Run with hops=1). With no trusted proxy (0) the header
    is ignored, because a direct client could forge it."""
    hops = _trusted_proxy_hops() if hops is None else hops
    if hops > 0:
        raw = request.headers.get("x-forwarded-for", "")
        entries = [e.strip() for e in raw.split(",") if e.strip()]
        if len(entries) >= hops:
            return entries[-hops]
    return request.client.host if request.client else "unknown"


def _client_key(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer ") and len(auth) > 7:
        return "tok:" + hashlib.sha256(auth[7:].encode()).hexdigest()[:32]
    return "ip:" + client_address(request)


def build_rate_limit_middleware(
    *,
    auth_per_minute: int,
    expensive_per_minute: int,
    mcp_per_minute: int = 0,
) -> Callable:
    """A pure-ASGI-style HTTP middleware function for app.middleware("http").

    Built as a factory so limits come from settings at app assembly and
    tests can build a tiny app with tiny limits. mcp_per_minute defaults
    to 0 (= unlimited) so pre-existing two-arg callers keep today's
    behavior byte-for-byte; app.py passes the real knob.
    """
    auth_limiter = SlidingWindowLimiter(auth_per_minute)
    expensive_limiter = SlidingWindowLimiter(expensive_per_minute)
    mcp_limiter = SlidingWindowLimiter(mcp_per_minute)

    async def _middleware(request: Request, call_next):
        path = request.url.path
        limiter = None
        if any(path.startswith(p) for p in _AUTH_PREFIXES):
            limiter = auth_limiter
        elif any(path.startswith(p) for p in _MCP_PREFIXES):
            limiter = mcp_limiter
        elif any(path.startswith(p) for p in _EXPENSIVE_PREFIXES) or any(
            p.search(path) for p in _EXPENSIVE_PATTERNS
        ):
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
