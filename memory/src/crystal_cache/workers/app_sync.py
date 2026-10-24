"""Watching and remembering a person's apps (2026-10-24).

Every `app_sync_interval_seconds` (60): the connections Nango flagged as
changed (sync webhook) or whose own cadence has passed are pulled, one
model at a time from the stored cursor, through Nango's records API. Each
record becomes a watched_items row (kept 7 days, Q108 A); a DELETED record
marks the item deleted and retires any memory made from it. When the
connection has Remember on and its rules allow the item, the item's text
goes through the same ingestion door a watched git file takes
(source_sync._ingest_envelope: size caps, extraction, chunking,
auto-approval born quarantine), under the source_uri
`nango://{integration}/{connection id}/{external id}` so a later delete
finds it. Once an hour, expired items are deleted. The goals loop
(goals/runner.py) runs beside this one in the same worker, so a long
goal session never holds up the pull.

Walls: a tenant over its fact cap or out of daily capacity still WATCHES
(no model spend: records are just read) but remembers nothing until the
wall lifts. A connection whose sign-in Nango rejects goes to
needs_reauth and stops being pulled; the console offers Reconnect.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Optional

import structlog

from ..connectors import catalog, nango
from ..connectors.service import should_remember

logger = structlog.get_logger(__name__)

MAX_PAGES_PER_MODEL = 5
MAX_REMEMBER_PER_CYCLE = 25
_CONFIG_TTL = 600.0
_config_cache: dict[str, Any] = {"at": 0.0, "models": {}}
_last_purge = {"at": 0.0}


def source_uri(integration: str, connection_id: str, external_id: str) -> str:
    return f"nango://{integration}/{connection_id}/{external_id}"


def reset_for_tests() -> None:
    _config_cache["at"] = 0.0
    _config_cache["models"] = {}
    _last_purge["at"] = 0.0


async def models_for(integration: str) -> list[str]:
    """The models this environment's enabled syncs return for an app,
    minus the syncs the catalog skips. Cached ten minutes."""
    now = time.monotonic()
    if now - _config_cache["at"] > _CONFIG_TTL or integration not in _config_cache["models"]:
        found: dict[str, list[str]] = {}
        for entry in await nango.scripts_config():
            key = str(entry.get("providerConfigKey") or entry.get("unique_key") or entry.get("integration") or "")
            app = catalog.get(key)
            skip = set(app.skip_syncs) if app else set()
            models: list[str] = found.setdefault(key, [])
            for sync in entry.get("syncs") or []:
                if not isinstance(sync, dict) or sync.get("enabled") is False or sync.get("name") in skip:
                    continue
                for m in sync.get("returns") or sync.get("models") or []:
                    name = m.get("name") if isinstance(m, dict) else str(m)
                    if name and name not in models:
                        models.append(name)
        _config_cache["models"] = found
        _config_cache["at"] = now
    return list(_config_cache["models"].get(integration, []))


async def _item_text(conn: dict[str, Any], app: catalog.App, item: dict[str, Any]) -> tuple[bytes, str, str]:
    """(bytes, label, mime) for ingestion. Drive files are downloaded
    (Google files exported as text); Notion pages come as markdown; every
    other app's record already carries its text."""
    title = item.get("title") or item.get("external_id") or "item"
    if app.content == "drive":
        mime = str((item.get("meta") or {}).get("mimeType") or "")
        fid = item["external_id"]
        if mime.startswith("application/vnd.google-apps."):
            export = "text/csv" if mime.endswith("spreadsheet") else "text/plain"
            data = await nango.proxy("GET", connection_id=conn["nango_connection_id"], integration=app.integration,
                                     path=f"/drive/v3/files/{fid}/export", params={"mimeType": export}, raw=True)
            return data, f"{app.name}/{title}.txt", "text/plain"
        data = await nango.proxy("GET", connection_id=conn["nango_connection_id"], integration=app.integration,
                                 path=f"/drive/v3/files/{fid}", params={"alt": "media"}, raw=True)
        return data, f"{app.name}/{title}", mime or "application/octet-stream"
    if app.content == "notion":
        out = await nango.trigger_action(connection_id=conn["nango_connection_id"], integration=app.integration,
                                         action_name="get-page-as-markdown", input={"pageId": item["external_id"]})
        md = ""
        if isinstance(out, dict):
            md = str(out.get("markdown") or out.get("content") or out.get("data") or "")
        elif isinstance(out, str):
            md = out
        text = f"# {title}\n\n{md or item.get('body') or item.get('snippet') or ''}"
        return text.encode("utf-8"), f"{app.name}/{title}.md", "text/markdown"
    head = [f"{app.name} {app.kind}: {title}"]
    if item.get("author"):
        head.append(f"From: {item['author']}")
    if item.get("occurred_at"):
        head.append(f"When: {item['occurred_at']}")
    text = "\n".join(head) + "\n\n" + (item.get("body") or item.get("snippet") or "")
    return text.encode("utf-8"), f"{app.name}/{title}.txt", "text/plain"


async def _retire(store, conn: dict[str, Any], external_id: str, *, vector_store, fact_vector_store) -> int:
    uri = source_uri(conn["integration"], conn["id"], external_id)
    n = 0
    for c in await store.list_crystals_for_customer(conn["customer_id"]):
        su = getattr(c, "source_uri", None) or ""
        if su == uri or su.startswith(uri + "#"):
            if await store.delete_crystal(c.id, conn["customer_id"], vector_store=vector_store,
                                          fact_vector_store=fact_vector_store):
                n += 1
    return n


async def _may_remember(store, customer_id: str) -> bool:
    """The tenant's write door, without raising (a walled tenant still
    watches)."""
    from ..ingress.auth import require_write_capacity
    customer = await store.get_customer_by_id(customer_id)
    if customer is None:
        return False
    try:
        await require_write_capacity(customer, store, spend=True)
    except Exception:  # noqa: BLE001 — HTTPException 402/429: the wall is up
        return False
    return True


async def sync_connection(store, conn: dict[str, Any], *, encoder=None, vector_store=None, fact_vector_store=None,
                          llm_client=None, ingest=None) -> dict[str, Any]:
    """One connection, one cycle. Returns counts (tested)."""
    from ..config import get_settings
    from ..ingestion.source_handlers import SourceEnvelope
    from .source_sync import _ingest_envelope

    app = catalog.get(conn["integration"])
    cid = conn["customer_id"]
    if app is None:
        await store.update_app_connection(cid, conn["id"], sync_due=False, last_error="This app is no longer offered.")
        return {"error": "unknown_app"}
    ttl = int(get_settings().watched_item_ttl_days or 7)
    ingest = ingest or _ingest_envelope
    counts = {"seen": 0, "deleted": 0, "remembered": 0, "retired": 0, "failed": 0}
    cursors = dict(conn.get("cursors") or {})
    remember_on = bool(conn["uses"].get("remember")) and bool(conn.get("remember_rules"))
    may_remember: Optional[bool] = None
    try:
        for model in await models_for(app.integration):
            cursor = cursors.get(model)
            for _page in range(MAX_PAGES_PER_MODEL):
                page = await nango.list_records(connection_id=conn["nango_connection_id"], integration=app.integration,
                                                model=model, cursor=cursor)
                for rec in page["records"]:
                    norm = catalog.normalize(app, rec)
                    meta_cursor = (rec.get("_nango_metadata") or {}).get("cursor")
                    if meta_cursor:
                        cursor = meta_cursor
                    if not norm["external_id"]:
                        continue
                    if norm["action"] == "DELETED":
                        await store.mark_watched_item_deleted(conn["id"], app.kind, norm["external_id"])
                        counts["deleted"] += 1
                        counts["retired"] += await _retire(store, conn, norm["external_id"], vector_store=vector_store,
                                                           fact_vector_store=fact_vector_store)
                        continue
                    item = await store.upsert_watched_item(
                        cid, conn["id"], kind=app.kind, external_id=norm["external_id"], title=norm["title"],
                        author=norm["author"], occurred_at=norm["occurred_at"], url=norm["url"],
                        snippet=norm["snippet"], body=norm["body"], meta=norm["meta"], ttl_days=ttl)
                    counts["seen"] += 1
                    if not remember_on or counts["remembered"] >= MAX_REMEMBER_PER_CYCLE:
                        continue
                    if item.get("remembered_document_id") and norm["action"] != "UPDATED":
                        continue
                    if should_remember(item, conn["remember_rules"]) is None:
                        continue
                    if may_remember is None:
                        may_remember = await _may_remember(store, cid)
                    if not may_remember:
                        continue
                    try:
                        data, label, mime = await _item_text(conn, app, item)
                        env = SourceEnvelope(payload_bytes=data, mime_type=mime,
                                             source_uri=source_uri(app.integration, conn["id"], norm["external_id"]),
                                             label=label, source_modified_at=norm["occurred_at"], connection_id=None)
                        holder = SimpleNamespace(customer_id=cid, review_mode="auto", id=conn["id"])
                        await ingest(store, encoder, vector_store, fact_vector_store, llm_client, holder, env)
                        await store.set_watched_item_remembered(item["id"], env.source_uri)
                        counts["remembered"] += 1
                    except Exception as e:  # noqa: BLE001 — one item never ends the cycle
                        counts["failed"] += 1
                        logger.warning("app_sync.remember_failed", connection_id=conn["id"], error_type=type(e).__name__)
                if not page.get("next_cursor") or not page["records"]:
                    break
            if cursor:
                cursors[model] = cursor
    except nango.NangoError as e:
        if e.code == "nango_auth" or e.status in (401, 403):
            await store.update_app_connection(cid, conn["id"], status="needs_reauth", sync_due=False, cursors=cursors,
                                              last_error="The app's sign-in expired; reconnect it.")
            return {**counts, "error": "needs_reauth"}
        await store.update_app_connection(cid, conn["id"], cursors=cursors,
                                          last_error="Couldn't reach the app just now; retrying.")
        return {**counts, "error": e.code}
    await store.update_app_connection(cid, conn["id"], cursors=cursors, sync_due=False, last_error=None,
                                      last_synced_at=datetime.now(timezone.utc))
    if counts["seen"] or counts["deleted"]:
        logger.info("app_sync.synced", connection_id=conn["id"], integration=app.integration, **counts)
    return counts


async def tick(store, *, encoder=None, vector_store=None, fact_vector_store=None, llm_client=None) -> dict[str, int]:
    """One pass: hourly expiry, then every due connection."""
    out = {"connections": 0, "purged": 0}
    now = time.monotonic()
    if now - _last_purge["at"] >= 3600:
        out["purged"] = await store.purge_expired_watched_items()
        _last_purge["at"] = now
    if nango.configured():
        out["connections"] = await _sync_due(store, encoder=encoder, vector_store=vector_store,
                                             fact_vector_store=fact_vector_store, llm_client=llm_client)
    return out


async def _sync_due(store, *, encoder, vector_store, fact_vector_store, llm_client) -> int:
    from ..llm.lanes import tenant_background
    n = 0
    for conn in await store.list_due_app_connections(limit=50):
        customer = await store.get_customer_by_id(conn["customer_id"])
        if customer is None:
            continue
        try:
            async with tenant_background(customer, store):
                await sync_connection(store, conn, encoder=encoder, vector_store=vector_store,
                                      fact_vector_store=fact_vector_store, llm_client=llm_client)
            n += 1
        except Exception:  # noqa: BLE001 — one tenant never blocks the rest
            logger.warning("app_sync.connection_failed", connection_id=conn["id"], exc_info=True)
    return n


async def run_app_sync_worker(*, store, encoder=None, vector_store=None, fact_vector_store=None,
                              shutdown_event: asyncio.Event) -> None:
    from ..config import get_settings
    interval = max(15, int(get_settings().app_sync_interval_seconds or 60))
    logger.info("app_sync_worker.started", interval_seconds=interval)
    from ..goals.runner import run_goals_loop
    goals_task = asyncio.create_task(run_goals_loop(store=store, encoder=encoder, fact_store=fact_vector_store,
                                                    vector_store=vector_store, shutdown_event=shutdown_event))
    while not shutdown_event.is_set():
        try:
            await tick(store, encoder=encoder, vector_store=vector_store, fact_vector_store=fact_vector_store)
        except Exception:  # noqa: BLE001
            logger.exception("app_sync_worker.pass_failed")
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            continue
    goals_task.cancel()
    try:
        await goals_task
    except (asyncio.CancelledError, Exception):  # noqa: BLE001
        pass
    logger.info("app_sync_worker.stopped")
