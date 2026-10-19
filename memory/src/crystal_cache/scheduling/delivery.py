"""Webhook delivery for scheduled runs (Q90=A, 2026-10-18).

One optional https URL per schedule. When a run completes the scheduler
POSTs the result there, shaped for the host:

  hooks.slack.com                     -> {"text": ...}      (Slack incoming webhook)
  discord.com / discordapp.com hooks  -> {"content": ...}   (Discord webhook)
  anything else                       -> the full JSON event, signed:
      X-Crystal-Signature: sha256=<HMAC-SHA256(deliver_secret, body)>
      X-Crystal-Event: scheduled_run.completed
      X-Crystal-Schedule: <schedule_id>

Zapier, Make and n8n catch hooks take the JSON shape, so one field covers
every destination without a connector of ours.

The URL is checked at creation AND at send time (the host is resolved
again) so a tenant cannot point the worker at the metadata service, a
private address, loopback, or a link-local address: https only, no
credentials in the URL, every resolved address public. No redirects are
followed. Delivery is at-most-once: the attempt and its result are
recorded on the row ("sent" / "failed: 404" / "failed: blocked address");
nothing retries, because a duplicate post to a chat channel is worse
than a visible failure the person can see in the console.
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import secrets
import socket
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

import structlog

logger = structlog.get_logger(__name__)

MAX_URL = 2048
TIMEOUT_SECONDS = 10.0
SLACK_HOSTS = frozenset({"hooks.slack.com"})
DISCORD_HOSTS = frozenset({"discord.com", "discordapp.com", "canary.discord.com", "ptb.discord.com"})
EVENT = "scheduled_run.completed"

Resolver = Callable[[str, int], list[str]]


def _resolve(host: str, port: int) -> list[str]:
    infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    return [i[4][0] for i in infos]


def _is_public(addr: ipaddress._BaseAddress) -> bool:
    if addr.version == 6 and getattr(addr, "ipv4_mapped", None) is not None:
        addr = addr.ipv4_mapped
    return bool(addr.is_global) and not addr.is_multicast


def check_deliver_url(url: Optional[str], *, resolve: Resolver = _resolve) -> Optional[str]:
    """None when the URL is acceptable, else the sentence to show."""
    if not url or not str(url).strip():
        return "A delivery URL is required."
    url = str(url).strip()
    if len(url) > MAX_URL:
        return "The delivery URL is too long."
    try:
        parts = urlsplit(url)
    except ValueError:
        return "That is not a valid URL."
    if parts.scheme != "https":
        return "Delivery URLs must start with https://."
    host = parts.hostname
    if not host:
        return "That URL has no host."
    if parts.username or parts.password:
        return "Delivery URLs cannot carry a username or password."
    try:
        port = parts.port or 443
    except ValueError:
        return "That URL has an invalid port."
    addrs: list[ipaddress._BaseAddress] = []
    try:
        addrs = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            raw = resolve(host, port)
        except (socket.gaierror, OSError, ValueError):
            return "That host could not be resolved."
        for a in raw:
            try:
                addrs.append(ipaddress.ip_address(a))
            except ValueError:
                continue
    if not addrs:
        return "That host could not be resolved."
    if any(not _is_public(a) for a in addrs):
        return "Delivery URLs must point at a public host."
    return None


def new_secret() -> str:
    return "whsec_" + secrets.token_urlsafe(24)


def sign(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def plain_text(event: dict[str, Any]) -> str:
    lines = [f"{event.get('title') or 'Scheduled run'} ({event.get('kind') or 'run'}) finished"]
    summary = (event.get("summary") or "").strip()
    if summary:
        lines.append(summary[:1500])
    link = event.get("console_url")
    if link:
        lines.append(str(link))
    return "\n".join(lines)


def shape(url: str, event: dict[str, Any]) -> dict[str, Any]:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if host in SLACK_HOSTS:
        return {"text": plain_text(event)}
    if host in DISCORD_HOSTS and "/api/webhooks/" in parts.path:
        return {"content": plain_text(event)[:2000]}
    return event


def build_event(sched, *, summary: Optional[str], crystal_id: Optional[str], ran_at: str, console_url: str) -> dict[str, Any]:
    base = console_url.rstrip("/")
    return {
        "event": EVENT,
        "schedule_id": sched.id,
        "title": sched.title,
        "kind": sched.kind,
        "question": sched.prompt if sched.kind == "research" else None,
        "ran_at": ran_at,
        "crystal_id": crystal_id,
        "summary": (summary or "")[:2000],
        "console_url": f"{base}/bank/{crystal_id}" if crystal_id else f"{base}/cognition",
    }


async def deliver(
    url: str, secret: Optional[str], event: dict[str, Any], *, transport: Any = None, resolve: Resolver = _resolve,
) -> tuple[bool, str]:
    """POST the event. Returns (ok, outcome) where outcome is what the
    row records: "sent", "failed: 404", "failed: blocked address",
    "failed: timeout", "failed: ConnectError"."""
    import httpx

    reason = check_deliver_url(url, resolve=resolve)
    if reason:
        return False, "failed: blocked address"
    payload = shape(url, event)
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "CrystalCache-Scheduler/1",
        "X-Crystal-Event": EVENT,
        "X-Crystal-Schedule": str(event.get("schedule_id") or ""),
    }
    if secret and payload is event:
        headers["X-Crystal-Signature"] = sign(secret, body)
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, follow_redirects=False, transport=transport) as client:
            resp = await client.post(url, content=body, headers=headers)
    except httpx.TimeoutException:
        return False, "failed: timeout"
    except httpx.HTTPError as e:
        return False, f"failed: {type(e).__name__}"[:120]
    if 200 <= resp.status_code < 300:
        return True, "sent"
    return False, f"failed: {resp.status_code}"
