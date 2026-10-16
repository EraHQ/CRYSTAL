"""OAuth discovery follows the host the client used (PR 12b, 2026-10-16).

The authorization server advertises ONE issuer (CC_OAUTH_ISSUER_URL;
the raw Cloud Run host until a product domain is mapped). When the
issuer moves to a product domain, clients will still reach the server
through the old host: the Anthropic connector directory submission
(2026-09-28, in review), the official MCP registry entry and every
snippet already pasted carry it. A client that connects through host X
and is told its resource is host Y may refuse (RFC 9728: the
protected-resource `resource` must match the server URL the client
used). So for every host in the allow-list (the issuer's own host plus
CC_OAUTH_ALT_ISSUER_HOSTS) the discovery documents and the 401
challenge are rewritten to that host. Tokens are host-agnostic (the
door checks the token, never the host), so a connection made through
either host keeps working through the other. Written 2026-10-16 after
a same-day mistake: the issuer was pointed at a domain the project does
not own for about two hours; this module is what makes the real move
safe when it happens.

Only allow-listed hosts are honoured: an arbitrary Host header never
steers discovery anywhere else.
"""
from __future__ import annotations

from typing import Any, Optional
from urllib.parse import urlsplit


def _host_only(host: str) -> str:
    return host.strip().lower().split(":", 1)[0]


def allowed_issuer_hosts(settings: Any) -> set[str]:
    out: set[str] = set()
    issuer_host = urlsplit(str(getattr(settings, "oauth_issuer_url", "") or "")).hostname
    if issuer_host:
        out.add(issuer_host.lower())
    for h in str(getattr(settings, "oauth_alt_issuer_hosts", "") or "").split(","):
        h = _host_only(h)
        if h:
            out.add(h)
    return out


def issuer_for(request_host: Optional[str], settings: Any) -> str:
    """The issuer base URL to advertise to a client on `request_host`:
    https://<host> when the host is allow-listed, else the configured
    issuer. Always https: the OAuth AS only exists on the hosted edge."""
    configured = str(getattr(settings, "oauth_issuer_url", "") or "").rstrip("/")
    if not request_host:
        return configured
    host = _host_only(request_host)
    if host in allowed_issuer_hosts(settings):
        # Keep a non-default port the client used (local dev), never a
        # scheme other than https.
        port = request_host.strip().split(":", 1)[1] if ":" in request_host.strip() else ""
        return f"https://{host}" + (f":{port}" if port and port != "443" else "")
    return configured


def rewrite_discovery(body: str, request_host: Optional[str], settings: Any) -> str:
    """Swap the configured issuer base for the request's issuer inside a
    discovery document (JSON text). A no-op when the host is not
    allow-listed or already matches."""
    configured = str(getattr(settings, "oauth_issuer_url", "") or "").rstrip("/")
    target = issuer_for(request_host, settings)
    if not configured or target == configured:
        return body
    return body.replace(configured, target)


DISCOVERY_PATHS = (
    "/.well-known/oauth-authorization-server",
    "/.well-known/oauth-protected-resource",
    "/.well-known/openid-configuration",
)


def is_discovery_path(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") for p in DISCOVERY_PATHS)


def host_from_scope(scope: dict) -> Optional[str]:
    for raw_name, raw_value in scope.get("headers", []) or []:
        if raw_name.lower() == b"host":
            try:
                return raw_value.decode("latin-1")
            except Exception:  # noqa: BLE001
                return None
    return None
