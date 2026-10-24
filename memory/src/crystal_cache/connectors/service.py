"""Connections: what the console, the webhook and MCP share (2026-10-24).

A person picks an app from the catalog (gated by the beta list), signs in
through Nango's Connect UI with a session token minted here, and Nango
tells us about the new connection by webhook. From then on the connection
row says what Crystal may do with the app:

  watch     read it on a schedule; goals see what it shows (kept 7 days)
  remember  turn items into memory, but only those the rules allow
  act       let goals send, post or create through it

Remember rules are an ordered allow/deny list, the same idea as
Automation (Q101): the first rule that matches an item decides, and the
default is "do not remember". Anthony: "Only make crystals for x, y, or z."
"""
from __future__ import annotations

from typing import Any, Optional

import structlog

from .. import beta
from . import catalog, nango

logger = structlog.get_logger(__name__)

USES = ("watch", "remember", "act")
RULE_FIELDS = ("from", "subject", "text", "folder", "label", "channel")
RULE_MATCHES = ("contains", "starts_with", "equals")
MAX_RULES = 30


class ConnectionRefused(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


# ------------------------------------------------------------------ rules

def validate_rules(rules: Any) -> list[dict[str, str]]:
    """Strict: a list of {decision, field, match, value}; unknown keys or
    values refuse with code rules_invalid."""
    if not isinstance(rules, list):
        raise ConnectionRefused("rules_invalid", "rules must be a list")
    if len(rules) > MAX_RULES:
        raise ConnectionRefused("rules_invalid", f"at most {MAX_RULES} rules per app")
    out = []
    for i, r in enumerate(rules, start=1):
        if not isinstance(r, dict):
            raise ConnectionRefused("rules_invalid", f"rule {i} must be an object")
        unknown = set(r) - {"decision", "field", "match", "value"}
        if unknown:
            raise ConnectionRefused("rules_invalid", f"rule {i}: unknown key(s) {sorted(unknown)}")
        decision, field, match = r.get("decision"), r.get("field"), r.get("match", "contains")
        value = str(r.get("value") or "").strip()
        if decision not in ("allow", "deny"):
            raise ConnectionRefused("rules_invalid", f"rule {i}: decision must be allow or deny")
        if field not in RULE_FIELDS:
            raise ConnectionRefused("rules_invalid", f"rule {i}: field must be one of {list(RULE_FIELDS)}")
        if match not in RULE_MATCHES:
            raise ConnectionRefused("rules_invalid", f"rule {i}: match must be one of {list(RULE_MATCHES)}")
        if not value or len(value) > 200:
            raise ConnectionRefused("rules_invalid", f"rule {i}: value must be 1 to 200 characters")
        out.append({"decision": decision, "field": field, "match": match, "value": value})
    return out


def _field_text(item: dict[str, Any], field: str) -> str:
    meta = item.get("meta") or {}
    if field == "from":
        return item.get("author") or ""
    if field == "subject":
        return item.get("title") or ""
    if field == "text":
        return " ".join([item.get("title") or "", item.get("snippet") or "", item.get("body") or ""])
    if field == "folder":
        return str(meta.get("path") or meta.get("parent_id") or "")
    if field == "label":
        labels = meta.get("labelIds") or []
        return " ".join(labels) if isinstance(labels, list) else str(labels)
    if field == "channel":
        return str(meta.get("channel") or meta.get("channel_id") or "")
    return ""


def rule_matches(item: dict[str, Any], rule: dict[str, str]) -> bool:
    hay = _field_text(item, rule["field"]).lower()
    needle = rule["value"].lower()
    if rule["match"] == "equals":
        return hay.strip() == needle
    if rule["match"] == "starts_with":
        return hay.strip().startswith(needle)
    return needle in hay


def should_remember(item: dict[str, Any], rules: list[dict[str, str]]) -> Optional[dict[str, str]]:
    """The first rule that matches decides; no match means no. Returns the
    allowing rule, or None."""
    for r in rules or []:
        if rule_matches(item, r):
            return r if r["decision"] == "allow" else None
    return None


def rule_text(rule: dict[str, str]) -> str:
    verb = "Remember" if rule["decision"] == "allow" else "Never remember"
    what = {"from": "items from", "subject": "items whose subject", "text": "items whose text",
            "folder": "files in folder", "label": "mail labelled", "channel": "messages in channel"}[rule["field"]]
    how = {"contains": "contains", "starts_with": "starts with", "equals": "is"}[rule["match"]]
    if rule["field"] in ("from", "folder", "label", "channel"):
        return f'{verb} {what} "{rule["value"]}"' + ("" if rule["match"] == "equals" else f" ({how})") + "."
    return f'{verb} {what} {how} "{rule["value"]}".'


# ---------------------------------------------------------------- catalog

async def catalog_view(store: Any, customer_id: str) -> dict[str, Any]:
    """The Apps card list: each app, whether this tenant may use it, and
    its connections."""
    emails = await beta.tenant_emails(store, customer_id)
    features = await beta.features_for(store, customer_id, emails)
    conns = await store.list_app_connections(customer_id)
    apps = []
    for app in catalog.APPS:
        allowed = features.get(catalog.app_feature(app), False)
        apps.append({
            "integration": app.integration, "name": app.name, "blurb": app.blurb, "kind": app.kind,
            "acts": [{"key": a, "label": catalog.ACT_LABEL.get(a, a)} for a in app.acts],
            "beta": app.beta, "available": allowed, "note": app.note, "remember_hint": app.remember_hint,
            "connections": [_public(c) for c in conns if c["integration"] == app.integration],
        })
    return {"configured": nango.configured(), "apps": apps, "features": features}


def _public(c: dict[str, Any]) -> dict[str, Any]:
    return {k: c[k] for k in ("id", "integration", "account_label", "status", "uses", "remember_rules",
                              "watch_cadence_minutes", "last_synced_at", "last_error", "created_at")} | {
        "rules_text": [rule_text(r) for r in c.get("remember_rules") or []]}


async def start_session(store: Any, customer_id: str, integration: str, *, operator_id: Optional[str] = None
                        ) -> dict[str, Any]:
    app = catalog.get(integration)
    if app is None:
        raise ConnectionRefused("unknown_app", "No such app.", 404)
    if not nango.configured():
        raise ConnectionRefused("connections_unavailable", "Connected apps are not set up on this deployment yet.", 503)
    emails = await beta.tenant_emails(store, customer_id)
    if not await beta.has_feature(store, customer_id, catalog.app_feature(app), emails):
        raise ConnectionRefused("beta_required", f"{app.name} is in private beta. Ask for an invite.", 403)
    tags = {"end_user_id": customer_id, "organization_id": customer_id, "integration": integration}
    if operator_id:
        tags["operator_id"] = operator_id
    session = await nango.create_connect_session(tags=tags, allowed_integrations=[integration])
    return {"token": session["token"], "expires_at": session.get("expires_at"), "integration": integration}


async def reconnect(store: Any, customer_id: str, connection_id: str) -> dict[str, Any]:
    c = await store.get_app_connection(customer_id, connection_id)
    if c is None:
        raise ConnectionRefused("not_found", "No such connection.", 404)
    session = await nango.reconnect_session(connection_id=c["nango_connection_id"], integration=c["integration"])
    return {"token": session["token"], "expires_at": session.get("expires_at"), "integration": c["integration"]}


async def set_uses(store: Any, customer_id: str, connection_id: str, *, uses: Optional[dict] = None,
                   rules: Optional[list] = None, cadence_minutes: Optional[int] = None,
                   paused: Optional[bool] = None) -> dict[str, Any]:
    c = await store.get_app_connection(customer_id, connection_id)
    if c is None:
        raise ConnectionRefused("not_found", "No such connection.", 404)
    fields: dict[str, Any] = {}
    if uses is not None:
        if not isinstance(uses, dict) or set(uses) - set(USES):
            raise ConnectionRefused("uses_invalid", f"uses takes {list(USES)}")
        merged = {**c["uses"], **{k: bool(v) for k, v in uses.items()}}
        app = catalog.get(c["integration"])
        if merged.get("act") and app is not None and not app.acts:
            raise ConnectionRefused("uses_invalid", f"{app.name} has nothing Crystal can do in it.")
        if merged.get("remember") and not merged.get("watch"):
            raise ConnectionRefused("uses_invalid", "Remember needs Watch on: Crystal remembers what it watches.")
        fields["uses"] = merged
    if rules is not None:
        fields["remember_rules"] = validate_rules(rules)
    if cadence_minutes is not None:
        if isinstance(cadence_minutes, bool) or not isinstance(cadence_minutes, int) or not 5 <= cadence_minutes <= 1440:
            raise ConnectionRefused("cadence_invalid", "check every 5 to 1440 minutes")
        fields["watch_cadence_minutes"] = cadence_minutes
    if paused is not None:
        fields["status"] = "paused" if paused else "active"
        if not paused:
            fields["sync_due"] = True
    updated = await store.update_app_connection(customer_id, connection_id, **fields) if fields else c
    return _public(updated)


async def remove(store: Any, customer_id: str, connection_id: str) -> bool:
    c = await store.get_app_connection(customer_id, connection_id)
    if c is None:
        raise ConnectionRefused("not_found", "No such connection.", 404)
    if nango.configured():
        try:
            await nango.delete_connection(connection_id=c["nango_connection_id"], integration=c["integration"])
        except nango.NangoError as e:
            logger.warning("connections.nango_delete_failed", connection_id=connection_id, code=e.code)
    return await store.delete_app_connection(customer_id, connection_id)


async def revoke_all_connections(store: Any, customer_id: str) -> int:
    """Account purge: every token at Nango goes before our rows do."""
    n = 0
    for c in await store.list_app_connections(customer_id):
        if nango.configured():
            try:
                await nango.delete_connection(connection_id=c["nango_connection_id"], integration=c["integration"])
                n += 1
            except nango.NangoError as e:
                logger.warning("connections.revoke_failed", connection_id=c["id"], code=e.code)
    return n


# ---------------------------------------------------------------- webhook

async def handle_webhook(store: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """A VERIFIED Nango webhook. auth: bind or refresh the connection from
    the tags we set when minting the session (the tenant id came from us,
    signed by Nango). sync: mark the connection due. Unknown types are
    acknowledged and ignored."""
    kind = payload.get("type")
    nango_id = str(payload.get("connectionId") or "")
    if kind == "auth":
        tags = payload.get("tags") or payload.get("endUser", {}).get("tags") or {}
        customer_id = str(tags.get("end_user_id") or tags.get("organization_id") or "")
        integration = str(payload.get("providerConfigKey") or tags.get("integration") or "")
        if not payload.get("success", True):
            if nango_id:
                existing = await store.find_app_connection_by_nango(nango_id)
                if existing is not None:
                    await store.update_app_connection(existing["customer_id"], existing["id"], status="needs_reauth",
                                                      last_error="The app's sign-in failed or expired; reconnect it.")
            return {"ok": True, "action": "auth_failed"}
        if not (nango_id and customer_id and catalog.get(integration)):
            logger.warning("connections.webhook_unbound", has_conn=bool(nango_id), has_tenant=bool(customer_id),
                           integration=integration)
            return {"ok": True, "action": "ignored"}
        if await store.get_customer_by_id(customer_id) is None:
            return {"ok": True, "action": "ignored"}
        label = str((payload.get("endUser") or {}).get("email") or tags.get("end_user_email") or "")
        conn = await store.upsert_app_connection(customer_id, integration=integration, nango_connection_id=nango_id,
                                                 account_label=label, created_by_operator_id=tags.get("operator_id"))
        try:
            await store.record_curation_event(customer_id, event_type="app_connected", subject_id=conn["id"],
                                              label=f"Connected {catalog.get(integration).name}"[:256],
                                              payload={"integration": integration, "operation": payload.get("operation")})
        except Exception:  # noqa: BLE001 — the feed is a witness
            pass
        return {"ok": True, "action": "bound", "connection_id": conn["id"]}
    if kind == "sync":
        existing = await store.find_app_connection_by_nango(nango_id) if nango_id else None
        if existing is None:
            return {"ok": True, "action": "ignored"}
        if payload.get("success") is False:
            await store.update_app_connection(existing["customer_id"], existing["id"],
                                              last_error="The app's last sync failed at Nango; it will retry.")
            return {"ok": True, "action": "sync_failed"}
        await store.update_app_connection(existing["customer_id"], existing["id"], sync_due=True)
        return {"ok": True, "action": "due", "connection_id": existing["id"]}
    return {"ok": True, "action": "ignored"}
