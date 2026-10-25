"""The apps a person can connect (2026-10-24, Q106).

Anthony: "Everything but keep Gmail private." Each app is one Nango
integration backed by Crystal's own provider app (our name on the consent
screen, our scopes, tokens portable off Nango). `beta` names the beta
feature that gates the app; None means "connections" (open).

Q118 A (2026-10-24): an app is offered only once its integration is
configured in the Nango dashboard (GET /integrations, matched on the
integration's unique key, which must stay Nango's default: the provider
id). Configuring one there makes it appear here with no deploy. The
apps past the first five are read and remember only; an act is added
per app once it is built and tested.

Which syncs feed an app is read from Nango at runtime (scripts_config),
because Nango names the models; `skip_syncs` lists the pre-built syncs
that carry nothing to watch (folders, permissions, users).

normalize() turns any provider record into the watched_items shape
without trusting a fixed schema: it reads the common field names the
providers use and falls back to the record's id.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass(frozen=True)
class App:
    integration: str
    name: str
    blurb: str
    kind: str                       # what its items are: file | event | email | message | page
    acts: tuple[str, ...]           # what a goal may do through it
    beta: Optional[str] = None      # the beta feature gating it (None: "connections")
    skip_syncs: tuple[str, ...] = ()
    content: str = "record"         # where an item's text comes from: record | drive | notion
    note: str = ""                  # shown under the app card
    remember_hint: str = ""         # what "Remember" means for this app
    group: str = "Other"            # how the Add connection list groups it


APPS: tuple[App, ...] = (
    App("google-drive", "Google Drive", "Files you pick are watched; their contents can become memory.",
        kind="file", acts=(), skip_syncs=("folders", "permissions", "shared-drives"), content="drive",
        note="You choose the files and folders in Google's picker; Crystal sees nothing else.",
        remember_hint="Turn file contents into memory.", group="Files"),
    App("google-calendar", "Google Calendar", "Your events, for goals that plan around your time.",
        kind="event", acts=("create_event",), remember_hint="Keep event notes as memory.", group="Email and calendar"),
    App("notion", "Notion", "Pages you share with Crystal are watched; their contents can become memory.",
        kind="page", acts=("append_page",), skip_syncs=("users", "data-source-templates", "data-sources"),
        content="notion", remember_hint="Turn page contents into memory.", group="Docs and knowledge"),
    App("slack", "Slack", "Messages in the channels you choose, for goals and action items.",
        kind="message", acts=("post_message",), remember_hint="Keep messages that pass your rules as memory.",
        group="Chat"),
    App("google-mail", "Gmail", "New mail, checked against your goals; replies drafted or sent.",
        kind="email", acts=("send_email",), beta="gmail",
        note="Private beta. Until Google's review of mail access is done, Google asks you to sign in again every 7 days.",
        remember_hint="Keep mail that passes your rules as memory.", group="Email and calendar"),
    # 2026-10-24 (Anthony: "I should at least get an option for Google
    # Sheets"): the one app past the first five that writes, because a goal
    # that builds a catalog needs somewhere to put rows.
    App("google-sheet", "Google Sheets", "Sheets you share; goals can create a spreadsheet and add rows to it.",
        kind="record", acts=("create_spreadsheet", "append_rows"), group="Docs and knowledge",
        remember_hint="Keep rows that pass your rules as memory."),
) + tuple(
    # Q118 A: read and remember only. (integration id from Nango's providers.yaml, name, kind, group, blurb)
    App(i, n, b, kind=k, acts=(), group=g, remember_hint="Keep items that pass your rules as memory.")
    for i, n, k, g, b in (
        ("outlook", "Outlook", "email", "Email and calendar", "Mail and calendar from Microsoft 365, checked against your goals."),
        ("calendly", "Calendly", "event", "Email and calendar", "Meetings people book with you."),
        ("zoom", "Zoom", "meeting", "Email and calendar", "Your meetings and their details."),
        ("microsoft-teams", "Microsoft Teams", "message", "Chat", "Messages in the teams and chats you choose."),
        ("discord", "Discord", "message", "Chat", "Messages in the servers you choose."),
        ("one-drive", "OneDrive for Business", "file", "Files", "Work files in OneDrive."),
        ("one-drive-personal", "OneDrive", "file", "Files", "Your personal OneDrive files."),
        ("sharepoint-online", "SharePoint", "file", "Files", "Sites and documents in SharePoint."),
        ("dropbox", "Dropbox", "file", "Files", "Files in your Dropbox."),
        ("box", "Box", "file", "Files", "Files in your Box account."),
        ("confluence", "Confluence", "page", "Docs and knowledge", "Pages in the spaces you share."),
        ("airtable", "Airtable", "record", "Docs and knowledge", "Records in the bases you share."),
        ("github", "GitHub", "issue", "Engineering", "Issues and pull requests in your repositories."),
        ("gitlab", "GitLab", "issue", "Engineering", "Issues and merge requests in your projects."),
        ("linear", "Linear", "issue", "Engineering", "Issues and projects in Linear."),
        ("jira", "Jira", "issue", "Engineering", "Issues in your Jira projects."),
        ("asana", "Asana", "task", "Tasks and projects", "Tasks and projects in Asana."),
        ("trello", "Trello", "task", "Tasks and projects", "Cards on your Trello boards."),
        ("clickup", "ClickUp", "task", "Tasks and projects", "Tasks in your ClickUp workspace."),
        ("monday", "monday.com", "task", "Tasks and projects", "Items on your monday.com boards."),
        ("todoist", "Todoist", "task", "Tasks and projects", "Your Todoist tasks."),
        ("hubspot", "HubSpot", "record", "Sales and support", "Contacts, companies and deals in HubSpot."),
        ("salesforce", "Salesforce", "record", "Sales and support", "Accounts, contacts and opportunities in Salesforce."),
        ("zendesk", "Zendesk", "ticket", "Sales and support", "Support tickets in Zendesk."),
        ("intercom", "Intercom", "ticket", "Sales and support", "Conversations in Intercom."),
    )
)
BY_INTEGRATION: dict[str, App] = {a.integration: a for a in APPS}
ACT_LABEL = {"send_email": "Send email", "post_message": "Post a message", "create_event": "Create events",
             "append_page": "Add to pages", "create_spreadsheet": "Create spreadsheets", "append_rows": "Add rows"}


def get(integration: str) -> Optional[App]:
    return BY_INTEGRATION.get(integration)


def app_feature(app: App) -> str:
    return app.beta or "connections"


# ------------------------------------------------------------- normalizer

_TITLE = ("subject", "title", "name", "summary", "filename")
_AUTHOR = ("from", "sender", "author", "user_name", "user", "owner", "created_by", "organizer", "creator")
_WHEN = ("date", "receivedAt", "received_at", "sent_at", "start", "createdTime", "created_time", "created_at",
         "created", "modifiedTime", "last_edited_time", "updated_at", "updated", "ts")
_URL = ("webViewLink", "url", "htmlLink", "permalink", "web_url", "link")
_BODY = ("body", "text", "description", "content", "body_text", "markdown", "plain")
_SNIPPET = ("snippet", "preview", "excerpt")


def _first(rec: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        v = rec.get(k)
        if v not in (None, "", [], {}):
            return v
    return None


def _as_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        for k in ("email", "displayName", "display_name", "name", "real_name", "id", "dateTime", "date", "plain", "text"):
            if v.get(k):
                return str(v[k])
        return ""
    if isinstance(v, (list, tuple)):
        return ", ".join(t for t in (_as_text(x) for x in v) if t)
    return str(v)


def _as_time(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, dict):
        v = v.get("dateTime") or v.get("date") or v.get("value")
        if v is None:
            return None
    if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace(".", "", 1).isdigit()):
        try:
            n = float(v)
            if n > 1e12:  # milliseconds
                n /= 1000.0
            return datetime.fromtimestamp(n, tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            return None
    if isinstance(v, str):
        s = v.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            try:
                from email.utils import parsedate_to_datetime
                dt = parsedate_to_datetime(v)
            except (TypeError, ValueError):
                return None
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return None


def normalize(app: App, record: dict[str, Any]) -> dict[str, Any]:
    """{external_id, title, author, occurred_at, url, snippet, body, meta, action}"""
    meta_in = dict(record.get("_nango_metadata") or {})
    rec = {k: v for k, v in record.items() if k != "_nango_metadata"}
    body = _as_text(_first(rec, _BODY))
    snippet = _as_text(_first(rec, _SNIPPET)) or body[:300]
    keep = {}
    for k in ("threadId", "thread_id", "channel", "channel_id", "mimeType", "mime_type", "labelIds", "to", "cc",
              "messageId", "message_id", "parent_id", "location", "end", "attendees", "path"):
        if k in rec and rec[k] not in (None, "", [], {}):
            keep[k] = rec[k]
    return {
        "external_id": str(rec.get("id") or rec.get("messageId") or rec.get("ts") or ""),
        "title": _as_text(_first(rec, _TITLE))[:512],
        "author": _as_text(_first(rec, _AUTHOR))[:320],
        "occurred_at": _as_time(_first(rec, _WHEN)),
        "url": _as_text(_first(rec, _URL)) or None,
        "snippet": snippet[:2000],
        "body": body,
        "meta": keep,
        "action": str(meta_in.get("last_action") or "ADDED").upper(),
    }
