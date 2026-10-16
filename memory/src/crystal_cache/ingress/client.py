"""Which tool made a call — Console Refine PR 11 (Q65=A, 2026-10-16).

The Query log shows a Tool column and Getting Started completes its
last row on "the first recall from a connected tool", so every query
log row carries the client that made it. The MCP server runs
stateless HTTP, where the handshake's clientInfo is not available per
tool call, so the signal is the HTTP User-Agent's product family (the
first token, up to the first slash or space). The console's own calls
(under /admin/api/) are stamped "console" so they never count as a
tool. Nothing here is security-relevant: the value is a label.
"""
from __future__ import annotations

from typing import Any, Optional

CONSOLE_CLIENT = "console"
_MAX = 64


def ua_family(user_agent: Optional[str]) -> Optional[str]:
    """'Claude-User/1.0 (+https://…)' -> 'Claude-User'; '' / None -> None."""
    if not user_agent:
        return None
    token = user_agent.strip().split(" ", 1)[0].split("/", 1)[0].strip()
    return token[:_MAX] or None


def client_from_request(request: Any) -> Optional[str]:
    """The console for /admin/api/* callers, else the User-Agent family."""
    try:
        path = str(getattr(getattr(request, "url", None), "path", "") or "")
        if path.startswith("/admin/api/"):
            return CONSOLE_CLIENT
        headers = getattr(request, "headers", None) or {}
        return ua_family(headers.get("user-agent") or headers.get("User-Agent"))
    except Exception:  # noqa: BLE001 — a label must never break a request
        return None


def client_from_scope(scope: dict) -> Optional[str]:
    """ASGI variant for the MCP middleware."""
    for raw_name, raw_value in scope.get("headers", []) or []:
        if raw_name.lower() == b"user-agent":
            try:
                return ua_family(raw_value.decode("latin-1"))
            except Exception:  # noqa: BLE001
                return None
    return None
