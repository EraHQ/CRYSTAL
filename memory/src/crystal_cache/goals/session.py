"""One goal check-in, run by the large tier (Q109 A).

The Steward's shape (cognition/steward.py): the agent loop with a FRESH
registry holding exactly this goal's tools, a tool-call budget, an action
cap and a wall clock, on the tenant's background lane, metered under
origin "goal". Nothing it does is silent: items it files show on the
Goals page, sends land in the ledger with the exact text, and its final
message becomes the goal's last_summary.

Tools:
  goal_plan        the goal, its plan and step states, permissions, apps
  watched_items    what the connected apps showed (since the last check-in
                   by default), with what this goal already filed
  read_item        one watched item's full text
  crystal_search / crystal_key_scan   the person's memory bank
  web_search / web_fetch   the web (Q113 A, 2026-10-24)
  save_fact        one short fact into memory, through the fact-cap door,
                   on the ledger and undoable for 24 hours (Q113 A)
  lattices / lattice_rows / lattice_stale   Crystal storage's tables
                   (2026-10-25): what exists, rows by filter, live cells
                   due for a refresh
  lattice_create / lattice_write   make a table, write up to 500 rows
                   per call by key (snapshot cells never overwritten;
                   ledgered and undoable for 24 hours)
  goal_items       this goal's open items and drafts
  add_action_item  a to-do for the person (one per watched item)
  write_draft      a reply, post, event or page addition, waiting for the
                   person unless the goal may send / post on its own
  send_draft       send or post a draft (connectors/actions.execute gates)
  close_item       mark an item done or dismissed
  mark_step        a plan step done (or not) with a note
  report_stuck     what the goal needs from the person
What it cannot do: change the plan, the permissions or the connections,
edit or delete memory, call APIs other than the connected apps.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Optional

import structlog

from ..config import get_settings
from ..connectors import actions as connector_actions
from ..connectors import catalog
from ..cost.emit import record_model_call

logger = structlog.get_logger(__name__)

GOAL_ORIGIN = "goal"
_GRACE_SECONDS = 90.0
_MODEL_CEILING = 150.0
_TOOL_WALL = 60.0
_BODY_CAP = 6000


def _head(s: Optional[str], n: int = 240) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _refusal(code: str, error: str) -> dict[str, Any]:
    return {"ok": False, "code": code, "error": error}


def goal_charter(goal: dict[str, Any], budget_calls: int, budget_actions: int) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    plan = goal.get("plan") or {}
    rules = plan.get("standing_instructions") or "(none)"
    return f"""You work one standing goal for one person (Crystal Cache). NOW IS {now} UTC. Nobody is watching this check-in; everything you file shows on their Goals page, and anything sent is recorded with its exact text.

The goal: {goal['title']}
In their words: {_head(goal['directive'], 1500)}
Success measure: {plan.get('success_measure') or '(none)'}
Standing instructions: {rules}

How to work:
1. Call goal_plan first, then goal_items (never file what is already filed; never redo a dismissed item).
2. Work the first step that is not done, and any repeating "each check-in" step. Look at watched_items for what arrived since the last check-in; read_item when the snippet is not enough. Search the memory bank (crystal_search) for context about people, projects and preferences before drafting.
3. When the goal needs facts from the world, web_search then web_fetch the best source; do not save what you only saw in a snippet. save_fact keeps what is worth remembering: one atomic fact per save, with its date and source in the value ("as of 2026-10-24, per nba.com: ..."). Never save what memory already says (crystal_search first).
3a. Pages that are mostly tables: web_tables shows each table's headers and first rows; lattice_import then moves a whole table into a lattice on the server (map its columns once; up to 2,000 rows in one action). That is how rows should arrive whenever the source has a table. Copy rows by hand with lattice_write only when there is no table to import.
3b. Rows go in lattices, never one fact each. Use the plan's lattices (goal_plan lists them; lattices shows every one). Read pages that list MANY records at once (a day's scoreboard, a season schedule, a standings table, a league's stat leaders) before pages about one record, and write everything a page gave you in ONE lattice_write (up to 500 rows per call, one action), with that page as source. A check-in should move hundreds of rows, not one. lattice_aggregate shows what is already filled (rows per date, per team) so you fetch only what is missing. Fill a lattice's key and list-level columns first; detail pages (one game's box score) come in later passes. Snapshot cells are fixed once filled (a final score); live cells take the new value (a season average). Each check-in, lattice_stale lists the live cells due for a refresh: recompute those first. Make a new lattice (lattice_create) only when no lattice fits.
4. File what the person must do as add_action_item (one per thing, with a due time when the goal has a clock). Write replies, posts, events and page additions with write_draft.
5. send_draft only when write_draft said this goal may send it on its own, and only when the draft follows the standing instructions exactly. When unsure, leave the draft for the person.
6. mark_step when a step is truly done. report_stuck when the goal needs something only the person can give (an app connected, Act turned on, a permission, an answer).
7. Budget: {budget_calls} tool calls and at most {budget_actions} items, drafts, saves or sends this check-in. Stop early when nothing more is useful.

End with the check-in's summary in plain prose, three short lines, no markdown headings:
Did: what you filed, drafted or sent.
Waiting: what needs the person.
Next: what the next check-in should do."""


class GoalSession:
    def __init__(self, *, store: Any, customer: Any, goal: dict[str, Any], run_id: str, encoder: Any,
                 fact_store: Any, max_actions: int, since: datetime, vector_store: Any = None) -> None:
        self.store = store
        self.customer = customer
        self.customer_id = customer.id
        self.goal = goal
        self.run_id = run_id
        self.encoder = encoder
        self.fact_store = fact_store
        self.vector_store = vector_store
        self.max_actions = max_actions
        self.since = since
        self.actions: list[dict[str, Any]] = []
        self.stuck: list[str] = []

    @property
    def actor(self) -> str:
        return f"goal:{self.goal['id']}"

    def cap(self) -> Optional[dict[str, Any]]:
        if len(self.actions) >= self.max_actions:
            return _refusal("session_action_cap",
                            f"This check-in already took {self.max_actions} actions; write the summary and stop.")
        return None


def build_goal_registry(session: GoalSession):
    from ..agent.tool_registry import Tool, ToolRegistry
    from ..cognition.agentic import _WORKER_TOOLS
    from ..cognition.retrieval_adapter import dispatch_cognition_retrieval

    store, goal = session.store, session.goal
    registry = ToolRegistry()

    def add(name: str, description: str, schema: dict, impl, returns: str = "") -> None:
        registry.register(Tool(name=name, description=description, contexts=frozenset({"agent"}),
                               parameters_schema=schema, impl=impl, returns_description=returns))

    def obj(props: dict, required: tuple[str, ...] = ()) -> dict:
        return {"type": "object", "properties": props, "required": list(required), "additionalProperties": False}

    async def _connections() -> list[dict[str, Any]]:
        return await store.list_app_connections(session.customer_id)

    # ------------------------------------------------------------- reads
    async def goal_plan(customer_id: str) -> dict:
        conns = await _connections()
        plan = goal.get("plan") or {}
        apps = []
        for c in conns:
            app = catalog.get(c["integration"])
            apps.append({"app": c["integration"], "name": app.name if app else c["integration"],
                         "connection_id": c["id"], "status": c["status"], "watch": c["uses"].get("watch"),
                         "act": c["uses"].get("act"), "acts": list(app.acts) if app else []})
        missing = [a for a in (plan.get("needs") or {}).get("apps", []) if a not in {c["integration"] for c in conns}]
        return {"title": goal["title"], "summary": plan.get("summary"), "success_measure": plan.get("success_measure"),
                "steps": plan.get("steps", []), "standing_instructions": plan.get("standing_instructions"),
                "may_send_email": bool(goal.get("may_send")), "may_post": bool(goal.get("may_post")),
                "connected_apps": apps, "missing_apps": missing,
                "missing": plan.get("missing", []),
                "lattices": [{"name": x.get("name"), "lattice_id": x.get("lattice_id")} for x in plan.get("lattices") or []],
                "last_check_in": goal.get("last_run_at"), "last_summary": _head(goal.get("last_summary"), 800)}
    add("goal_plan", "The goal, its plan with each step's state, the standing instructions, what it may do on its own, and the connected apps (connection ids, whether Act is on).",
        obj({}), goal_plan)

    async def watched_items(customer_id: str, app: Optional[str] = None, since_hours: Optional[float] = None,
                            query: Optional[str] = None, limit: int = 20) -> dict:
        conns = {c["id"]: c for c in await _connections()}
        conn_id = None
        if app:
            match = [c for c in conns.values() if c["integration"] == app]
            if not match:
                return _refusal("not_connected", f"{app} isn't connected.")
            conn_id = match[0]["id"]
        since = session.since if since_hours is None else datetime.now(timezone.utc) - timedelta(hours=float(since_hours))
        items = await store.list_watched_items(customer_id, connection_id=conn_id, since=since, query=query,
                                               limit=max(1, min(int(limit or 20), 50)))
        filed = await store.goal_items_for_watched(customer_id, goal["id"], [i["id"] for i in items])
        return {"since": since.isoformat(), "items": [
            {"item_id": i["id"], "app": (conns.get(i["connection_id"]) or {}).get("integration"), "kind": i["kind"],
             "title": _head(i["title"], 200), "from": _head(i["author"], 120), "when": i["occurred_at"],
             "snippet": _head(i["snippet"], 400), "already_filed": i["id"] in filed,
             "thread": (i.get("meta") or {}).get("threadId") or (i.get("meta") or {}).get("thread_id")}
            for i in items]}
    add("watched_items", "What the connected apps showed: emails, messages, events, files, pages. Defaults to what arrived since the last check-in; since_hours looks further back (up to 168). already_filed marks items this goal already handled.",
        obj({"app": {"type": "string", "enum": [a.integration for a in catalog.APPS]},
             "since_hours": {"type": "number", "minimum": 0, "maximum": 168},
             "query": {"type": "string", "maxLength": 200}, "limit": {"type": "integer", "minimum": 1, "maximum": 50}}),
        watched_items)

    async def read_item(customer_id: str, item_id: str) -> dict:
        it = await store.get_watched_item(customer_id, item_id)
        if it is None:
            return _refusal("not_found", "No such item (items are kept 7 days).")
        meta = it.get("meta") or {}
        return {"item_id": it["id"], "kind": it["kind"], "title": it["title"], "from": it["author"],
                "when": it["occurred_at"], "url": it["url"], "body": (it["body"] or it["snippet"])[:_BODY_CAP],
                "meta": {k: meta[k] for k in ("threadId", "thread_id", "messageId", "message_id", "channel", "channel_id",
                                              "to", "cc", "location", "end", "attendees") if k in meta}}
    add("read_item", "One watched item's full text and reply details (thread, message id, channel).",
        obj({"item_id": {"type": "string", "maxLength": 64}}, ("item_id",)), read_item)

    for spec in _WORKER_TOOLS:
        if spec["name"] not in ("crystal_search", "crystal_key_scan"):
            continue

        def _make(action_value: str):
            async def _impl(customer_id: str, **kwargs: Any) -> dict:
                return await dispatch_cognition_retrieval(
                    action_value=action_value, step_input=kwargs, customer_id=customer_id,
                    store=store, fact_store=session.fact_store, encoder=session.encoder)
            return _impl
        add(spec["name"], spec["description"], spec["schema"], _make(spec["action"]))

    from ..agent.tool_registry import get_registry, import_all_tools
    web = get_registry()
    if web.get("web_search") is None:  # a process that never loaded the agent's tools
        import_all_tools()
        web = get_registry()
    for name in ("web_search", "web_fetch", "web_tables"):
        spec = web.get(name)
        if spec is not None:
            add(name, spec.description, spec.parameters_schema, spec.impl, spec.returns_description or "")

    async def goal_items(customer_id: str, status: Optional[str] = "open", limit: int = 30) -> dict:
        rows = await store.list_goal_items(customer_id, goal["id"], status=status or None,
                                           limit=max(1, min(int(limit or 30), 100)))
        return {"items": [{"item_id": r["id"], "kind": r["kind"], "status": r["status"], "title": _head(r["title"], 200),
                           "body": _head(r["body"], 300), "watched_item_id": r["watched_item_id"], "due_at": r["due_at"],
                           "act": (r.get("target") or {}).get("action"), "ref": (r.get("target") or {}).get("ref"),
                           "created_at": r["created_at"]} for r in rows]}
    add("goal_items", "This goal's items: action items, drafts awaiting the person, sent messages, notes. status open (default), done, dismissed, sent, failed, or empty for all.",
        obj({"status": {"type": "string", "enum": ["", "open", "done", "dismissed", "sent", "failed"]},
             "limit": {"type": "integer", "minimum": 1, "maximum": 100}}), goal_items)

    # ------------------------------------------------------------ writes
    async def _not_filed(customer_id: str, watched_item_id: Optional[str]) -> Optional[dict]:
        if not watched_item_id:
            return None
        if await store.get_watched_item(customer_id, watched_item_id) is None:
            return _refusal("not_found", "No such watched item.")
        if await store.goal_items_for_watched(customer_id, goal["id"], [watched_item_id]):
            return _refusal("already_filed", "This goal already filed something for that item.")
        return None

    async def add_action_item(customer_id: str, title: str, detail: str = "", watched_item_id: Optional[str] = None,
                              due_in_minutes: Optional[int] = None) -> dict:
        if (cap := session.cap()):
            return cap
        if (bad := await _not_filed(customer_id, watched_item_id)):
            return bad
        due = datetime.now(timezone.utc) + timedelta(minutes=int(due_in_minutes)) if due_in_minutes else None
        row = await store.add_goal_item(customer_id, goal["id"], kind="action", title=title.strip()[:512],
                                        body=detail.strip(), watched_item_id=watched_item_id, due_at=due)
        session.actions.append({"id": row["id"], "action": "add_action_item", "title": row["title"]})
        return {"ok": True, "item_id": row["id"]}
    add("add_action_item", "File a to-do for the person (one per thing). Link the watched item it came from; give due_in_minutes when the goal has a clock.",
        obj({"title": {"type": "string", "maxLength": 300}, "detail": {"type": "string", "maxLength": 2000},
             "watched_item_id": {"type": "string", "maxLength": 64},
             "due_in_minutes": {"type": "integer", "minimum": 1, "maximum": 43200}}, ("title",)), add_action_item)

    async def write_draft(customer_id: str, app: str, title: str, body: str, watched_item_id: Optional[str] = None,
                          to: Optional[str] = None, subject: Optional[str] = None, thread_id: Optional[str] = None,
                          in_reply_to: Optional[str] = None, channel: Optional[str] = None,
                          thread_ts: Optional[str] = None, start: Optional[str] = None, end: Optional[str] = None,
                          page_id: Optional[str] = None, act: Optional[str] = None,
                          spreadsheet_id: Optional[str] = None, sheet: Optional[str] = None) -> dict:
        if (cap := session.cap()):
            return cap
        spec = catalog.get(app)
        if spec is None or not spec.acts:
            return _refusal("act_not_offered", f"Crystal can't act through {app}.")
        conns = [c for c in await _connections() if c["integration"] == app]
        if not conns:
            return _refusal("not_connected", f"{spec.name} isn't connected; report_stuck to ask for it.")
        if (bad := await _not_filed(customer_id, watched_item_id)):
            return bad
        if act is not None and act not in spec.acts:
            return _refusal("act_not_offered", f"{spec.name} can: {', '.join(spec.acts)}.")
        act = act or spec.acts[0]
        target = {"app": app, "connection_id": conns[0]["id"], "action": act}
        for k, v in (("to", to), ("subject", subject), ("thread_id", thread_id), ("in_reply_to", in_reply_to),
                     ("channel", channel), ("thread_ts", thread_ts), ("start", start), ("end", end), ("page_id", page_id),
                     ("spreadsheet_id", spreadsheet_id), ("sheet", sheet)):
            if v:
                target[k] = str(v)[:500]
        if (bad := connector_actions.validate_target(target, body)):
            return bad
        row = await store.add_goal_item(customer_id, goal["id"], kind="draft", title=title.strip()[:512],
                                        body=body.strip(), watched_item_id=watched_item_id, target=target)
        session.actions.append({"id": row["id"], "action": "write_draft", "title": row["title"], "act": act})
        may = bool(goal.get(connector_actions.permission_for(act))) and bool(conns[0]["uses"].get("act")) \
            and conns[0]["status"] == "active"
        return {"ok": True, "item_id": row["id"], "act": act, "may_send_on_its_own": may,
                "note": "send_draft when it follows the standing instructions." if may
                else "This draft waits for the person to approve it."}
    add("write_draft", "Write a reply, post, event or page addition through a connected app. google-mail: to, subject (Re: for replies), thread_id and in_reply_to from read_item. slack: channel (id), thread_ts. google-calendar: start, end (ISO 8601), title is the event name. notion: page_id. google-sheet: act create_spreadsheet (title is the spreadsheet name, body lists tab names one per line; its id comes back as the sent item's ref) or act append_rows (spreadsheet_id, sheet = tab name, body = a JSON array of rows, each an array of cells). The body is the exact text.",
        obj({"app": {"type": "string", "enum": [a.integration for a in catalog.APPS if a.acts]},
             "title": {"type": "string", "maxLength": 300}, "body": {"type": "string", "maxLength": 20000},
             "watched_item_id": {"type": "string", "maxLength": 64}, "to": {"type": "string", "maxLength": 320},
             "subject": {"type": "string", "maxLength": 300}, "thread_id": {"type": "string", "maxLength": 200},
             "in_reply_to": {"type": "string", "maxLength": 500}, "channel": {"type": "string", "maxLength": 100},
             "thread_ts": {"type": "string", "maxLength": 64}, "start": {"type": "string", "maxLength": 64},
             "end": {"type": "string", "maxLength": 64}, "page_id": {"type": "string", "maxLength": 100},
             "act": {"type": "string", "enum": sorted({x for a in catalog.APPS for x in a.acts})},
             "spreadsheet_id": {"type": "string", "maxLength": 200}, "sheet": {"type": "string", "maxLength": 100}},
            ("app", "title", "body")), write_draft)

    async def save_fact(customer_id: str, key: str, value: str, reason: str) -> dict:
        from ..cognition import controls
        from ..ingress.auth import require_write_capacity
        if (cap := session.cap()):
            return cap
        if (bad := controls.reason_check(reason)):
            return bad
        key, value = key.strip(), value.strip()
        if not key or not value:
            return _refusal("empty_fact", "A fact needs a key and a value.")
        if len(value) > 800:
            return _refusal("content_too_long", "One atomic fact, under 800 characters.")
        try:
            await require_write_capacity(session.customer, store, spend=False)
        except Exception:  # noqa: BLE001 — HTTPException 402: the fact cap is reached
            return _refusal("memory_full", "Memory is full on this plan; report_stuck so the person can make room.")
        if session.encoder is None or session.vector_store is None:
            return _refusal("memory_unavailable", "Memory writes aren't available to this worker.")
        crystal, fact = await store.add_pair_for_customer(
            customer_id=customer_id, prompt_text=key, answer_text=value, pair_type="question_answer",
            encoder=session.encoder, vector_store=session.vector_store, crystal_type="customer:legacy",
            source_kind="model_reasoning")
        row = await controls.record(store, customer_id, actor=session.actor, run_id=session.run_id, action="save_fact",
                                    subject_kind="fact", subject_id=fact.id,
                                    label=f"Goal \"{goal['title'][:60]}\" saved a fact - {_head(value, 120)}",
                                    reason=reason, inverse={"forget_fact": fact.id, "crystal_id": crystal.id},
                                    undoable=True)
        session.actions.append({"id": row["id"], "action": "save_fact", "fact_id": fact.id})
        return {"ok": True, "fact_id": fact.id, "crystal_id": crystal.id}
    add("save_fact", "Save one short, atomic fact to the person's memory (key: what it answers, value: the fact with its date and source). Checked against the memory cap; recorded and undoable for 24 hours. Search memory first; never save a duplicate.",
        obj({"key": {"type": "string", "maxLength": 300}, "value": {"type": "string", "maxLength": 800},
             "reason": {"type": "string", "maxLength": 500}}, ("key", "value", "reason")), save_fact)

    # ----------------------------------------------------------- lattices
    from ..lattices import service as lat

    def _lat_err(e: "lat.LatticeRefused") -> dict:
        out = _refusal(e.code, e.message)
        if e.detail:
            out["detail"] = e.detail
        return out

    async def lattices(customer_id: str) -> dict:
        return {"lattices": await lat.list_view(store, customer_id)}
    add("lattices", "Every lattice (Crystal storage's tables): name, columns (type, snapshot or live, entity), key columns, row count.",
        obj({}), lattices)

    async def lattice_rows(customer_id: str, lattice: str, filters: Optional[list] = None, search: Optional[str] = None,
                           sort: Optional[str] = None, descending: bool = False, limit: int = 30, offset: int = 0) -> dict:
        try:
            out = await lat.query(store, customer_id, lattice, filters=filters, search=search, sort=sort,
                                  descending=descending, limit=min(int(limit or 30), 50), offset=offset)
        except lat.LatticeRefused as e:
            return _lat_err(e)
        out["lattice"] = {"name": out["lattice"]["name"], "rows": out["lattice"]["rows"]}
        return out
    add("lattice_rows", "Read rows of a lattice (name or id). filters: [{column, op: eq | contains | gte | lte, value}]; search: text in any text column; sort a column (descending for highest first); limit up to 50, offset to page.",
        obj({"lattice": {"type": "string", "maxLength": 120},
             "filters": {"type": "array", "maxItems": 10, "items": {"type": "object", "properties": {
                 "column": {"type": "string"}, "op": {"type": "string", "enum": list(lat.FILTER_OPS)},
                 "value": {"type": ["string", "number", "boolean"]}}, "required": ["column", "value"]}},
             "search": {"type": "string", "maxLength": 200}, "sort": {"type": "string", "maxLength": 40},
             "descending": {"type": "boolean"}, "limit": {"type": "integer", "minimum": 1, "maximum": 50},
             "offset": {"type": "integer", "minimum": 0}}, ("lattice",)), lattice_rows)

    async def lattice_aggregate(customer_id: str, lattice: str, group_by: Optional[list] = None,
                                metrics: Optional[list] = None, filters: Optional[list] = None,
                                sort: Optional[str] = None, limit: int = 30) -> dict:
        try:
            return await lat.aggregate(store, customer_id, lattice, group_by=group_by, metrics=metrics,
                                       filters=filters, sort=sort, limit=min(int(limit or 30), 50))
        except lat.LatticeRefused as e:
            return _lat_err(e)
    add("lattice_aggregate", "Count, sum, average, min or max of a lattice's number columns, grouped by up to three columns: how many games each team has, which dates are already filled, averages. Use it to see what is missing before writing.",
        obj({"lattice": {"type": "string", "maxLength": 120},
             "group_by": {"type": "array", "maxItems": 3, "items": {"type": "string", "maxLength": 40}},
             "metrics": {"type": "array", "maxItems": 8, "items": {"type": "object", "properties": {
                 "fn": {"type": "string", "enum": list(lat.AGG_FNS)}, "column": {"type": "string", "maxLength": 40}},
                 "required": ["fn"]}},
             "filters": {"type": "array", "maxItems": 10, "items": {"type": "object", "properties": {
                 "column": {"type": "string"}, "op": {"type": "string", "enum": list(lat.FILTER_OPS)},
                 "value": {"type": ["string", "number", "boolean"]}}, "required": ["column", "value"]}},
             "sort": {"type": "string", "maxLength": 48}, "limit": {"type": "integer", "minimum": 1, "maximum": 50}},
            ("lattice",)), lattice_aggregate)

    async def lattice_import(customer_id: str, lattice: str, url: str, table: Any, columns: dict, reason: str,
                             constants: Optional[dict] = None, offset: int = 0) -> dict:
        from ..cognition import controls
        if (cap := session.cap()):
            return cap
        if (bad := controls.reason_check(reason)):
            return bad
        try:
            out = await lat.import_table(store, customer_id, lattice, url=url, table=table, columns=columns,
                                         constants=constants, offset=offset, actor=session.actor,
                                         customer=session.customer, reason=reason, run_id=session.run_id)
        except lat.LatticeRefused as e:
            return _lat_err(e)
        session.actions.append({"id": (out["action_ids"] or [None])[0], "action": "lattice_import", "lattice": out["lattice"],
                                "inserted": out["inserted"], "updated": out["updated"]})
        return out
    add("lattice_import", "Move a table from a web page into a lattice on the server, up to 2,000 rows per call (one action): look at the page with web_tables first, then map lattice columns to the table's headers. columns: {lattice column: header, or {from: header, link: true for the cell's link, pattern: regex whose first group is kept}} (a record's id often sits in a link). constants: values for every row (a season, a league). Dates in common formats are converted. next_offset continues a longer table.",
        obj({"lattice": {"type": "string", "maxLength": 120}, "url": {"type": "string", "maxLength": 2048},
             "table": {"type": "string", "maxLength": 120},
             "columns": {"type": "object", "maxProperties": 40}, "constants": {"type": "object", "maxProperties": 20},
             "offset": {"type": "integer", "minimum": 0, "maximum": 5000},
             "reason": {"type": "string", "maxLength": 500}},
            ("lattice", "url", "table", "columns", "reason")), lattice_import)

    async def lattice_stale(customer_id: str, lattice: str, limit: int = 50) -> dict:
        try:
            return await lat.stale(store, customer_id, lattice, limit=limit)
        except lat.LatticeRefused as e:
            return _lat_err(e)
    add("lattice_stale", "The live cells of a lattice that are past their refresh window, oldest first: what to recompute this check-in.",
        obj({"lattice": {"type": "string", "maxLength": 120}, "limit": {"type": "integer", "minimum": 1, "maximum": 200}},
            ("lattice",)), lattice_stale)

    async def lattice_create(customer_id: str, name: str, description: str, columns: list, key_columns: list) -> dict:
        if (cap := session.cap()):
            return cap
        try:
            made = await lat.create(store, customer_id, name=name, columns=columns, key_columns=key_columns,
                                    description=description, goal_id=goal["id"], actor=session.actor)
        except lat.LatticeRefused as e:
            return _lat_err(e)
        session.actions.append({"id": made["id"], "action": "lattice_create", "name": made["name"]})
        return {"ok": True, "lattice": lat.describe(made)}
    add("lattice_create", "Make a lattice when none fits. columns: [{key (lowercase_with_underscores), label, type: text | number | date (YYYY-MM-DD) | bool | url, kind: snapshot (true once) | live (changes), entity (true for columns to browse by, text only), refresh_minutes (live columns: how fresh it must stay)}]; key_columns: the snapshot columns that identify a row.",
        obj({"name": {"type": "string", "maxLength": 120}, "description": {"type": "string", "maxLength": 500},
             "columns": {"type": "array", "minItems": 1, "maxItems": lat.MAX_COLUMNS, "items": {"type": "object", "properties": {
                 "key": {"type": "string"}, "label": {"type": "string"},
                 "type": {"type": "string", "enum": list(lat.COLUMN_TYPES)},
                 "kind": {"type": "string", "enum": list(lat.CELL_KINDS)}, "entity": {"type": "boolean"},
                 "refresh_minutes": {"type": "integer", "minimum": 15, "maximum": 43200}}, "required": ["key"]}},
             "key_columns": {"type": "array", "minItems": 1, "maxItems": 5, "items": {"type": "string"}}},
            ("name", "description", "columns", "key_columns")), lattice_create)

    async def lattice_write(customer_id: str, lattice: str, rows: list, source: str, reason: str) -> dict:
        from ..cognition import controls
        if (cap := session.cap()):
            return cap
        if (bad := controls.reason_check(reason)):
            return bad
        try:
            out = await lat.write_rows(store, customer_id, lattice, rows, source=source, actor=session.actor,
                                       customer=session.customer)
        except lat.LatticeRefused as e:
            return _lat_err(e)
        action_id = await lat.ledger(store, customer_id, out, actor=session.actor, reason=reason, run_id=session.run_id)
        if action_id:
            session.actions.append({"id": action_id, "action": "lattice_write", "lattice": out["lattice"],
                                    "inserted": out["inserted"], "updated": out["updated"]})
        view = lat.public(out)
        view.pop("row_ids", None)
        return view
    add("lattice_write", "Add or update rows of a lattice by its key columns, up to 500 per call (one action). Empty cells fill; live cells take the new value; snapshot cells already filled are kept and listed under locked. source: the page the values came from. Rows count against the plan's lattice row cap.",
        obj({"lattice": {"type": "string", "maxLength": 120},
             "rows": {"type": "array", "minItems": 1, "maxItems": lat.MAX_BATCH, "items": {"type": "object"}},
             "source": {"type": "string", "maxLength": 300}, "reason": {"type": "string", "maxLength": 500}},
            ("lattice", "rows", "source", "reason")), lattice_write)

    async def send_draft(customer_id: str, item_id: str, reason: str) -> dict:
        if (cap := session.cap()):
            return cap
        item = await store.get_goal_item(customer_id, item_id)
        if item is None or item["goal_id"] != goal["id"]:
            return _refusal("not_found", "No such draft in this goal.")
        out = await connector_actions.execute(store, customer_id, item, actor=session.actor, reason=reason,
                                              goal=goal, run_id=session.run_id)
        if out.get("ok"):
            session.actions.append({"id": out.get("action_id"), "action": "send_draft", "item_id": item_id})
        return {k: v for k, v in out.items() if k != "item"}
    add("send_draft", "Send or post one of this goal's drafts. Refused unless the goal may do it on its own and the app's Act is on. Needs a reason a person will read.",
        obj({"item_id": {"type": "string", "maxLength": 64}, "reason": {"type": "string", "maxLength": 1000}},
            ("item_id", "reason")), send_draft)

    async def close_item(customer_id: str, item_id: str, status: str, reason: str = "") -> dict:
        item = await store.get_goal_item(customer_id, item_id)
        if item is None or item["goal_id"] != goal["id"]:
            return _refusal("not_found", "No such item in this goal.")
        if item["status"] != "open":
            return _refusal("already_handled", f"Already {item['status']}.")
        await store.update_goal_item(customer_id, item_id, status=status)
        return {"ok": True}
    add("close_item", "Mark one of this goal's open items done (it happened: the person replied, the event passed) or dismissed (no longer needed).",
        obj({"item_id": {"type": "string", "maxLength": 64}, "status": {"type": "string", "enum": ["done", "dismissed"]},
             "reason": {"type": "string", "maxLength": 500}}, ("item_id", "status")), close_item)

    async def mark_step(customer_id: str, n: int, done: bool, note: str = "") -> dict:
        plan = dict(goal.get("plan") or {})
        steps = [dict(s) for s in plan.get("steps", [])]
        hit = next((s for s in steps if int(s.get("n", 0)) == int(n)), None)
        if hit is None:
            return _refusal("no_such_step", f"The plan has steps 1 to {len(steps)}.")
        hit["done"] = bool(done)
        hit["note"] = note.strip()[:500]
        plan["steps"] = steps
        await store.update_goal(customer_id, goal["id"], plan=plan)
        goal["plan"] = plan
        return {"ok": True, "step": hit}
    add("mark_step", "Mark a plan step done (or not done again) with a short note. Repeating steps stay not done.",
        obj({"n": {"type": "integer", "minimum": 1, "maximum": 50}, "done": {"type": "boolean"},
             "note": {"type": "string", "maxLength": 500}}, ("n", "done")), mark_step)

    async def report_stuck(customer_id: str, need: str) -> dict:
        need = need.strip()[:500]
        if not need:
            return _refusal("need_required", "Say what the goal needs.")
        open_notes = await store.list_goal_items(customer_id, goal["id"], status="open", kind="note", limit=50)
        if any(n["title"].strip().lower() == need.lower() for n in open_notes):
            return {"ok": True, "already_reported": True}
        row = await store.add_goal_item(customer_id, goal["id"], kind="note", title=need, body="")
        session.stuck.append(need)
        return {"ok": True, "item_id": row["id"]}
    add("report_stuck", "Tell the person what this goal needs from them (connect an app, turn Act on, a permission, an answer). Shown on the Goals page; not repeated while it is open.",
        obj({"need": {"type": "string", "maxLength": 500}}, ("need",)), report_stuck)

    return registry


async def run_goal_session(*, store: Any, customer: Any, goal: dict[str, Any], encoder: Any = None,
                           fact_store: Any = None, vector_store: Any = None, llm: Any = None) -> dict[str, Any]:
    """One check-in. Never raises for the worker; schedules the next."""
    from ..agent.agent import Agent
    from ..llm import get_llm_client
    from ..llm.lanes import tenant_background

    settings = get_settings()
    budget_calls = int(settings.goal_max_tool_calls or 16)
    budget_actions = int(settings.goal_max_actions_per_session or 8)
    wall = float(settings.goal_wall_seconds or 240.0)
    max_tokens = int(settings.goal_max_tokens or 3000)
    now = datetime.now(timezone.utc)
    last = goal.get("last_run_at")
    since = datetime.fromisoformat(last) if last else now - timedelta(hours=24)

    run_id = await store.open_goal_run(customer.id, payload={
        "topic": f"Goal: {goal['title'][:80]}", "goal_id": goal["id"], "phase": "run",
        "budget": {"tool_calls": budget_calls, "actions": budget_actions}})
    session = GoalSession(store=store, customer=customer, goal=goal, run_id=run_id, encoder=encoder,
                          fact_store=fact_store, max_actions=budget_actions, since=since, vector_store=vector_store)
    registry = build_goal_registry(session)
    usage_sink: dict[str, Any] = {}
    trajectory: list[dict[str, Any]] = []
    model = None
    summary = ""
    ok = False
    stop_reason = None
    calls = 0
    try:
        async with tenant_background(customer, store):
            client = llm or get_llm_client()
            try:
                model = client._resolve_model("large", None) if hasattr(client, "_resolve_model") else None
            except Exception:  # noqa: BLE001
                model = None
            agent = Agent(customer=SimpleNamespace(id=customer.id), llm=client,
                          tool_state={"store": store, "fact_vector_store": fact_store, "encoder": encoder,
                                      "vector_store": vector_store},
                          model=model, max_tokens=max_tokens, max_iterations=budget_calls, registry=registry)
            run = await asyncio.wait_for(
                agent.run(messages=[{"role": "user", "content": (
                              f"Check in on this goal. Last check-in: {last or 'never'}. "
                              f"Budget: {budget_calls} tool calls, at most {budget_actions} actions. "
                              "Start with goal_plan and goal_items.")}],
                          system=goal_charter(goal, budget_calls, budget_actions), deadline_seconds=wall,
                          usage_sink=usage_sink, trajectory_sink=trajectory, default_tool_timeout=_TOOL_WALL,
                          model_timeout=_MODEL_CEILING),
                timeout=wall + _GRACE_SECONDS)
        summary = (run.get("final_text") or "").strip()
        calls = len(run.get("tool_calls") or [])
        stop_reason = run.get("stop_reason")
        await store.mark_cognition_task_complete(run_id, result={
            "output_type": "report", "deliverable": summary, "brief": summary, "actions": list(session.actions),
            "tool_calls": [{"tool": t.get("tool_name"), "input": t.get("input")} for t in (run.get("tool_calls") or [])][:40],
            "calls": calls, "stop_reason": stop_reason, "model": run.get("model") or model, "goal_id": goal["id"]},
            completed_at=datetime.now(timezone.utc))
        ok = True
    except asyncio.TimeoutError:
        stop_reason = "belt"
        await store.mark_cognition_task_failed(run_id, error_message=f"Goal session exceeded {wall + _GRACE_SECONDS:.0f}s",
                                               completed_at=datetime.now(timezone.utc))
    except Exception as e:  # noqa: BLE001 — one goal's failure never ends the pass
        stop_reason = "error"
        logger.warning("goals.session_failed", goal_id=goal["id"], error_type=type(e).__name__, error=str(e)[:300])
        try:
            await store.mark_cognition_task_failed(run_id, error_message=f"{type(e).__name__}: {str(e)[:300]}",
                                                   completed_at=datetime.now(timezone.utc))
        except Exception:  # noqa: BLE001
            pass

    pt, ct = int(usage_sink.get("prompt_tokens") or 0), int(usage_sink.get("completion_tokens") or 0)
    if pt or ct:
        try:
            await record_model_call(customer_id=customer.id, model=usage_sink.get("model") or model or "unknown",
                                    input_tokens=pt, output_tokens=ct,
                                    cache_read_tokens=usage_sink.get("cache_read_tokens"),
                                    cache_creation_tokens=usage_sink.get("cache_creation_tokens"),
                                    origin=GOAL_ORIGIN, session_id=run_id, store=store)
        except Exception as e:  # noqa: BLE001
            logger.warning("goals.metering_failed", run_id=run_id, error=str(e)[:200])

    from .planner import next_check_minutes
    every = next_check_minutes(goal.get("plan"), int(settings.goal_min_check_minutes or 15))
    finished = datetime.now(timezone.utc)
    await store.update_goal(customer.id, goal["id"], last_run_at=finished, last_task_id=run_id,
                            next_run_at=finished + timedelta(minutes=every if ok else max(every, 60)),
                            last_summary=(summary or ("The check-in didn't finish; trying again later." if not ok else ""))[:4000])
    n = len(session.actions)
    try:
        await store.record_curation_event(customer.id, event_type="goal_session", subject_id=goal["id"],
                                          label=(f"Goal check-in: {goal['title'][:80]} - {n} action{'s' if n != 1 else ''}"
                                                 if ok else f"Goal check-in failed: {goal['title'][:80]}"),
                                          payload={"goal_id": goal["id"], "run_id": run_id, "actions": n, "ok": ok,
                                                   "stuck": session.stuck[:5], "calls": calls})
    except Exception:  # noqa: BLE001
        logger.debug("goals.event_failed", exc_info=True)
    logger.info("goals.session_completed", goal_id=goal["id"], run_id=run_id, ok=ok, actions=n, calls=calls,
                stop_reason=stop_reason, prompt_tokens=pt, completion_tokens=ct)
    return {"ok": ok, "run_id": run_id, "actions": n, "summary": summary, "calls": calls, "stop_reason": stop_reason}
