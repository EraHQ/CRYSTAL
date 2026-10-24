"""Nango, the one door to a person's apps (2026-10-24).

Nango holds every user's OAuth tokens and runs the provider syncs; Crystal
never sees a provider token. This module is the only place that talks to
Nango's HTTP API:

  create_connect_session  POST /connect/sessions        (the console opens Connect UI with the token)
  reconnect_session       POST /connect/sessions/reconnect
  scripts_config          GET  /scripts/config           (which syncs return which models)
  list_records            GET  /records                  (Connection-Id, Provider-Config-Key, model, cursor)
  trigger_action          POST /action/trigger           (a pre-built action, e.g. Notion page as markdown)
  proxy                   ANY  /proxy/{path}             (any provider call: send mail, post a message)
  delete_connection       DELETE /connection/{id}?provider_config_key=
  verify_webhook          X-Nango-Hmac-Sha256 = HMAC-SHA256(raw body, webhook signing key)

Every call is bounded (timeout) and raises NangoError carrying the HTTP
status, never the provider's body, so a caller can log and refuse with a
code. Tests replace the transport (set_transport) instead of patching.
"""
from __future__ import annotations

import hashlib
import hmac
from typing import Any, Optional

import httpx
import structlog

logger = structlog.get_logger(__name__)

_TIMEOUT = httpx.Timeout(20.0, connect=5.0)
_transport: Optional[httpx.AsyncBaseTransport] = None


class NangoError(RuntimeError):
    def __init__(self, message: str, *, status: Optional[int] = None, code: str = "nango_error"):
        super().__init__(message)
        self.status = status
        self.code = code


def set_transport(transport: Optional[httpx.AsyncBaseTransport]) -> None:
    """Tests inject an httpx.MockTransport here."""
    global _transport
    _transport = transport


def configured() -> bool:
    from ..config import get_settings
    return bool((get_settings().nango_secret_key or "").strip())


def _base() -> str:
    from ..config import get_settings
    return (get_settings().nango_base_url or "https://api.nango.dev").rstrip("/")


def _headers(connection_id: Optional[str] = None, integration: Optional[str] = None) -> dict[str, str]:
    from ..config import get_settings
    key = (get_settings().nango_secret_key or "").strip()
    if not key:
        raise NangoError("Nango is not set up on this deployment", code="connections_unavailable")
    h = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    if connection_id:
        h["Connection-Id"] = connection_id
    if integration:
        h["Provider-Config-Key"] = integration
    return h


async def _request(method: str, path: str, *, headers: dict[str, str], params: Optional[dict] = None,
                   json: Any = None, extra_headers: Optional[dict[str, str]] = None) -> httpx.Response:
    kwargs: dict[str, Any] = {"timeout": _TIMEOUT}
    if _transport is not None:
        kwargs["transport"] = _transport
    async with httpx.AsyncClient(**kwargs) as client:
        try:
            resp = await client.request(method, f"{_base()}{path}", headers={**headers, **(extra_headers or {})},
                                        params=params, json=json)
        except httpx.HTTPError as e:
            raise NangoError(f"Nango unreachable ({type(e).__name__})", code="nango_unreachable") from e
    return resp


def _check(resp: httpx.Response, what: str) -> Any:
    if resp.status_code >= 400:
        code = "nango_auth" if resp.status_code in (401, 403) else (
            "nango_not_found" if resp.status_code == 404 else "nango_error")
        raise NangoError(f"Nango {what} failed (HTTP {resp.status_code})", status=resp.status_code, code=code)
    if not resp.content:
        return None
    try:
        return resp.json()
    except ValueError:
        return resp.text


async def create_connect_session(*, tags: dict[str, str], allowed_integrations: list[str]) -> dict[str, Any]:
    """{token, expires_at, connect_link?} for the console's Connect UI."""
    resp = await _request("POST", "/connect/sessions", headers=_headers(),
                          json={"tags": tags, "allowed_integrations": allowed_integrations})
    data = _check(resp, "connect session") or {}
    data = data.get("data", data) if isinstance(data, dict) else {}
    if not data.get("token"):
        raise NangoError("Nango returned no session token", code="nango_error")
    return {"token": data["token"], "expires_at": data.get("expires_at"), "connect_link": data.get("connect_link")}


async def reconnect_session(*, connection_id: str, integration: str) -> dict[str, Any]:
    resp = await _request("POST", "/connect/sessions/reconnect", headers=_headers(),
                          json={"connection_id": connection_id, "integration_id": integration})
    data = _check(resp, "reconnect session") or {}
    data = data.get("data", data) if isinstance(data, dict) else {}
    if not data.get("token"):
        raise NangoError("Nango returned no session token", code="nango_error")
    return {"token": data["token"], "expires_at": data.get("expires_at")}


async def list_integrations() -> list[dict[str, Any]]:
    """The integrations configured in this Nango environment:
    [{unique_key, provider, display_name, ...}] (GET /integrations,
    verified against @nangohq/node 0.71.13 listIntegrations)."""
    resp = await _request("GET", "/integrations", headers=_headers())
    data = _check(resp, "integrations")
    if isinstance(data, dict):
        data = data.get("data", [])
    return [d for d in (data or []) if isinstance(d, dict)]


async def scripts_config() -> list[dict[str, Any]]:
    """The environment's integrations and their enabled syncs and models."""
    resp = await _request("GET", "/scripts/config", headers=_headers())
    data = _check(resp, "scripts config")
    if isinstance(data, dict):
        data = data.get("data", [])
    return list(data or [])


async def list_records(*, connection_id: str, integration: str, model: str, cursor: Optional[str] = None,
                       limit: int = 100) -> dict[str, Any]:
    """One page: {records, next_cursor}."""
    params: dict[str, Any] = {"model": model, "limit": limit}
    if cursor:
        params["cursor"] = cursor
    resp = await _request("GET", "/records", headers=_headers(connection_id, integration), params=params)
    data = _check(resp, "records") or {}
    return {"records": list(data.get("records") or []), "next_cursor": data.get("next_cursor")}


async def trigger_action(*, connection_id: str, integration: str, action_name: str, input: Any) -> Any:
    resp = await _request("POST", "/action/trigger", headers=_headers(connection_id, integration),
                          json={"action_name": action_name, "input": input})
    return _check(resp, f"action {action_name}")


async def proxy(method: str, *, connection_id: str, integration: str, path: str, params: Optional[dict] = None,
                json: Any = None, raw: bool = False) -> Any:
    """A provider API call made with the person's token, which Nango adds.
    raw=True returns the bytes (a file download)."""
    path = path if path.startswith("/") else "/" + path
    resp = await _request(method, f"/proxy{path}", headers=_headers(connection_id, integration), params=params,
                          json=json, extra_headers={"Retries": "2"})
    if raw:
        if resp.status_code >= 400:
            _check(resp, f"proxy {method}")
        return resp.content
    return _check(resp, f"proxy {method}")


async def delete_connection(*, connection_id: str, integration: str) -> bool:
    resp = await _request("DELETE", f"/connection/{connection_id}", headers=_headers(),
                          params={"provider_config_key": integration})
    if resp.status_code == 404:
        return False
    _check(resp, "delete connection")
    return True


def verify_webhook(raw_body: bytes, signature: Optional[str]) -> bool:
    """X-Nango-Hmac-Sha256: hex HMAC-SHA256 of the raw body with the
    webhook signing key (not the secret key). Constant-time compare."""
    from ..config import get_settings
    key = (get_settings().nango_webhook_signing_key or "").strip()
    if not key or not signature:
        return False
    expected = hmac.new(key.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature.strip().lower())
