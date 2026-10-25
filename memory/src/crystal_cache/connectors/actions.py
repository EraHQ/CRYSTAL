"""What a goal (or the person) may DO through a connected app (Q107).

Anthony: "We can enable send and post too." A draft is a goal_items row
of kind "draft" whose target names the app, the connection and the act;
execute() carries it out through Nango's proxy with the person's own
token. Paths are relative to each provider's proxy base URL in Nango's
providers.yaml (verified 2026-10-24): gmail.googleapis.com,
www.googleapis.com (Calendar), slack.com/api, api.notion.com (Nango adds
the Notion-Version header), sheets.googleapis.com (Google Sheets: create a
spreadsheet; append rows, the body being a JSON array of rows).

Gates, every one checked here and nowhere else:
  1. the app offers the act (catalog.App.acts)
  2. the connection exists, is active, and has Act on (uses.act)
  3. a goal acting on its own needs the goal's permission: may_send for
     email, may_post for everything else. A person approving a draft in
     the console or through MCP IS the permission (actor != goal:...).
Every send lands in the ledger (controls.record, undoable=False, the
exact text in the label's payload) and the item keeps the text sent.
Refusals carry a code.
"""
from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Any, Optional

import structlog

from . import catalog, nango

logger = structlog.get_logger(__name__)

SEND_ACTS = ("send_email",)
MAX_BODY = 20_000


def refusal(code: str, error: str) -> dict[str, Any]:
    return {"ok": False, "code": code, "error": error}


def permission_for(act: str) -> str:
    return "may_send" if act in SEND_ACTS else "may_post"


def _iso_or_none(v: Any) -> Optional[str]:
    if not v:
        return None
    try:
        dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def validate_target(target: dict[str, Any], body: str) -> Optional[dict[str, Any]]:
    """The fields each act needs, checked before anything is stored as
    sendable. Returns a refusal or None."""
    act = target.get("action")
    if not body.strip() and act != "create_spreadsheet":
        return refusal("empty_body", "There is nothing to send.")
    if len(body) > MAX_BODY:
        return refusal("body_too_long", f"Keep it under {MAX_BODY} characters.")
    if act == "send_email":
        to = str(target.get("to") or "")
        if "@" not in to or len(to) > 320:
            return refusal("recipient_required", "An email needs a recipient address (to).")
        if not str(target.get("subject") or "").strip() and not target.get("thread_id"):
            return refusal("subject_required", "A new email needs a subject.")
    elif act == "post_message":
        if not str(target.get("channel") or "").strip():
            return refusal("channel_required", "A Slack post needs a channel id.")
    elif act == "create_event":
        if not _iso_or_none(target.get("start")):
            return refusal("start_required", "An event needs a start time (ISO 8601).")
        if target.get("end") and not _iso_or_none(target.get("end")):
            return refusal("bad_end", "The end time must be ISO 8601.")
    elif act == "append_page":
        if not str(target.get("page_id") or "").strip():
            return refusal("page_required", "Adding to a page needs the page id.")
    elif act == "append_rows":
        if not str(target.get("spreadsheet_id") or "").strip():
            return refusal("spreadsheet_required", "Adding rows needs the spreadsheet id (from create_spreadsheet or its URL).")
        if parse_rows(body) is None:
            return refusal("rows_invalid", "The body must be a JSON array of rows, each an array of cells (at most 500 rows).")
    elif act == "create_spreadsheet":
        pass  # the title is the draft's title; the body lists tab names, one per line
    else:
        return refusal("unknown_action", f"Unknown act {act!r}.")
    return None


def parse_rows(body: str) -> Optional[list[list[Any]]]:
    import json as _json
    try:
        rows = _json.loads(body)
    except ValueError:
        return None
    if not isinstance(rows, list) or not rows or len(rows) > 500:
        return None
    if not all(isinstance(r, list) and len(r) <= 100 and all(isinstance(c, (str, int, float, bool)) or c is None for c in r)
               for r in rows):
        return None
    return rows


def _email_raw(target: dict[str, Any], body: str) -> str:
    msg = EmailMessage()
    msg["To"] = str(target["to"])
    if target.get("cc"):
        msg["Cc"] = str(target["cc"])
    subject = str(target.get("subject") or "").strip()
    if target.get("thread_id") and subject and not subject.lower().startswith("re:"):
        subject = "Re: " + subject
    msg["Subject"] = subject or "Re:"
    if target.get("in_reply_to"):
        msg["In-Reply-To"] = str(target["in_reply_to"])
        msg["References"] = str(target["in_reply_to"])
    msg.set_content(body)
    return base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii").rstrip("=")


def _notion_blocks(body: str) -> list[dict[str, Any]]:
    blocks = []
    for para in [p for p in body.split("\n\n") if p.strip()][:90]:
        text = para.strip()[:1900]
        blocks.append({"object": "block", "type": "paragraph",
                       "paragraph": {"rich_text": [{"type": "text", "text": {"content": text}}]}})
    return blocks


async def _call(act: str, conn: dict[str, Any], target: dict[str, Any], title: str, body: str) -> dict[str, Any]:
    """The provider request. Returns {ref} (what the provider called it)."""
    cid, integ = conn["nango_connection_id"], conn["integration"]
    if act == "send_email":
        payload: dict[str, Any] = {"raw": _email_raw(target, body)}
        if target.get("thread_id"):
            payload["threadId"] = str(target["thread_id"])
        out = await nango.proxy("POST", connection_id=cid, integration=integ,
                                path="/gmail/v1/users/me/messages/send", json=payload)
        return {"ref": (out or {}).get("id") if isinstance(out, dict) else None}
    if act == "post_message":
        payload = {"channel": str(target["channel"]), "text": body}
        if target.get("thread_ts"):
            payload["thread_ts"] = str(target["thread_ts"])
        out = await nango.proxy("POST", connection_id=cid, integration=integ, path="/chat.postMessage", json=payload)
        if isinstance(out, dict) and out.get("ok") is False:  # Slack answers 200 with ok:false
            raise nango.NangoError(f"Slack refused the post: {out.get('error')}", status=400, code="provider_refused")
        return {"ref": (out or {}).get("ts") if isinstance(out, dict) else None}
    if act == "create_event":
        start = _iso_or_none(target.get("start"))
        end = _iso_or_none(target.get("end")) or (
            datetime.fromisoformat(start) + timedelta(minutes=int(target.get("minutes") or 30))).isoformat()
        payload = {"summary": title or "Event", "description": body, "start": {"dateTime": start},
                   "end": {"dateTime": end}}
        out = await nango.proxy("POST", connection_id=cid, integration=integ,
                                path="/calendar/v3/calendars/primary/events", json=payload)
        return {"ref": (out or {}).get("id") if isinstance(out, dict) else None}
    if act == "create_spreadsheet":
        tabs = [t.strip()[:100] for t in body.splitlines() if t.strip()][:20] or ["Sheet1"]
        payload = {"properties": {"title": title or "Crystal"},
                   "sheets": [{"properties": {"title": t}} for t in tabs]}
        out = await nango.proxy("POST", connection_id=cid, integration=integ, path="/v4/spreadsheets", json=payload)
        return {"ref": (out or {}).get("spreadsheetId") if isinstance(out, dict) else None}
    if act == "append_rows":
        sheet = str(target.get("sheet") or "Sheet1").replace("'", "")
        out = await nango.proxy("POST", connection_id=cid, integration=integ,
                                path=f"/v4/spreadsheets/{target['spreadsheet_id']}/values/'{sheet}'!A1:append",
                                params={"valueInputOption": "USER_ENTERED", "insertDataOption": "INSERT_ROWS"},
                                json={"values": parse_rows(body)})
        return {"ref": ((out or {}).get("updates") or {}).get("updatedRange") if isinstance(out, dict) else None}
    if act == "append_page":
        out = await nango.proxy("PATCH", connection_id=cid, integration=integ,
                                path=f"/v1/blocks/{target['page_id']}/children", json={"children": _notion_blocks(body)})
        return {"ref": str(target["page_id"])}
    raise ValueError(act)


async def execute(store: Any, customer_id: str, item: dict[str, Any], *, actor: str, reason: str,
                  goal: Optional[dict[str, Any]] = None, run_id: Optional[str] = None) -> dict[str, Any]:
    """Carry out one draft. `actor` is "goal:<id>" when a goal acts on its
    own, else the person ("console" / "chat")."""
    from ..cognition import controls

    if item.get("kind") != "draft":
        return refusal("not_a_draft", "Only a draft can be sent.")
    if item.get("status") != "open":
        return refusal("already_handled", f"This draft is already {item.get('status')}.")
    target = dict(item.get("target") or {})
    act = str(target.get("action") or "")
    app = catalog.get(str(target.get("app") or ""))
    if app is None or act not in app.acts:
        return refusal("act_not_offered", "That app can't do that.")
    conn = await store.get_app_connection(customer_id, str(target.get("connection_id") or ""))
    if conn is None or conn["integration"] != app.integration:
        return refusal("not_connected", f"{app.name} isn't connected.")
    if conn["status"] != "active":
        return refusal("connection_" + str(conn["status"]), f"{app.name} needs attention first ({conn['status']}).")
    if not conn["uses"].get("act"):
        return refusal("act_off", f"Act is off for {app.name}; turn it on under Settings > Connections.")
    if actor.startswith("goal:"):
        g = goal or {}
        perm = permission_for(act)
        if not g.get(perm):
            return refusal("goal_not_permitted",
                           f"This goal may not {'send email' if perm == 'may_send' else 'post'} on its own; the draft waits for you.")
    if (bad := validate_target(target, item.get("body") or "")):
        return bad
    if not nango.configured():
        return refusal("connections_unavailable", "Connections aren't set up on this server.")
    try:
        out = await _call(act, conn, target, item.get("title") or "", item.get("body") or "")
    except nango.NangoError as e:
        if e.code == "nango_auth" or e.status in (401, 403):
            await store.update_app_connection(customer_id, conn["id"], status="needs_reauth",
                                              last_error="The app's sign-in expired; reconnect it.")
        await store.update_goal_item(customer_id, item["id"], status="failed")
        logger.warning("connectors.action_failed", act=act, integration=app.integration, code=e.code)
        return refusal("send_failed", f"{app.name} didn't accept it ({e.code}).")
    sent = await store.update_goal_item(customer_id, item["id"], status="sent",
                                        target={**target, "ref": out.get("ref"),
                                                "sent_at": datetime.now(timezone.utc).isoformat(), "sent_by": actor})
    where = target.get("to") or target.get("channel") or target.get("page_id") or target.get("start") or ""
    label = f"{_who(actor, goal)} {catalog.ACT_LABEL.get(act, act).lower()}: {(item.get('title') or '')[:80]} ({where})"
    row = await controls.record(store, customer_id, actor=actor, run_id=run_id, action=act, subject_kind="goal_item",
                                subject_id=item["id"], label=label[:256], reason=(reason or "Approved draft")[:1000],
                                inverse=None, undoable=False)
    return {"ok": True, "item": sent, "action_id": row.get("id"), "ref": out.get("ref")}


def _who(actor: str, goal: Optional[dict[str, Any]]) -> str:
    if actor.startswith("goal:"):
        return f"Goal \"{(goal or {}).get('title', '')[:60]}\""
    return "You"
