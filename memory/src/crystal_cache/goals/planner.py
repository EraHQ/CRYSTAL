"""A goal's plan, written by the frontier tier (Q109 A).

One structured call: the person's words, the apps they connected (and
what each may do), and the goal's permissions go in; a plan comes out:

  summary            one sentence: what this goal is for
  success_measure    how anyone can tell it is working
  check_every_minutes how often a session should check in (floored at
                     goal_min_check_minutes; capped at a week)
  steps              ordered; each {title, detail, done}
  needs              {apps: [integration...], acts: [act...]} the plan
                     relies on; missing ones are reported, never assumed
  missing            Q112 A (2026-10-24): what the goal needs that this
                     account doesn't have yet (an app, a data source, a
                     tool, a permission, an answer), each with why, a next
                     step and whether it blocks. A needed app that isn't
                     connected is added here by the server, not left to the
                     model. Any blocking item holds the goal in
                     "needs_setup": no check-ins (no spend) until the person
                     fixes it or chooses Run anyway.
  decisions          (2026-10-24) planning is two steps when the goal
                     turns on choices only the person can make (which app,
                     where the data lives, scope). Anthony: "I don't think
                     the plan should plan for Notion specifically if I never
                     connected it. It should instead give decisions/options
                     as a step 2 to planning." The first call returns the
                     decisions (2-4 options, one recommended) and no steps;
                     the goal waits in "deciding"; the person's answers go
                     back to the planner, which then writes the plan on
                     them. Answers are final: the planner never re-asks.

normalize_plan() never trusts the model's shape: unknown apps and acts
are dropped, counts and lengths are capped, the cadence is clamped.
"""
from __future__ import annotations

import re

import json
from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from ..connectors import catalog

logger = structlog.get_logger(__name__)

PLAN_ORIGIN = "goal_plan"
PLAN_TIER = "frontier"
MAX_STEPS = 12
MAX_CHECK_MINUTES = 7 * 24 * 60
MAX_MISSING = 8
MISSING_KINDS = ("app", "data_source", "tool", "permission", "info")
MAX_DECISIONS = 4

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "success_measure": {"type": "string"},
        "check_every_minutes": {"type": "integer"},
        "steps": {"type": "array", "items": {
            "type": "object",
            "properties": {"title": {"type": "string"}, "detail": {"type": "string"}},
            "required": ["title", "detail"],
            "additionalProperties": False,
        }},
        "needs_apps": {"type": "array", "items": {"type": "string"}},
        "needs_acts": {"type": "array", "items": {"type": "string"}},
        "standing_instructions": {"type": "string"},
        "missing": {"type": "array", "items": {
            "type": "object",
            "properties": {"kind": {"type": "string", "enum": list(MISSING_KINDS)}, "need": {"type": "string"},
                           "why": {"type": "string"}, "suggestion": {"type": "string"}, "blocking": {"type": "boolean"}},
            "required": ["kind", "need", "why", "suggestion", "blocking"],
            "additionalProperties": False,
        }},
        "decisions": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "question": {"type": "string"}, "why": {"type": "string"},
                "options": {"type": "array", "items": {
                    "type": "object", "properties": {"label": {"type": "string"}, "detail": {"type": "string"}},
                    "required": ["label", "detail"], "additionalProperties": False}},
                "recommended": {"type": "integer"},
            },
            "required": ["question", "why", "options", "recommended"],
            "additionalProperties": False,
        }},
    },
    "required": ["summary", "success_measure", "check_every_minutes", "steps", "needs_apps", "needs_acts",
                 "standing_instructions", "missing", "decisions"],
    "additionalProperties": False,
}


APP_STATUS = {"connected": "connected", "ready": "ready to connect",
              "setup": "not set up on this server yet (choosing it means setup first)"}


def app_statuses(connections: list[dict[str, Any]], offered_keys: Optional[set[str]] = None) -> dict[str, str]:
    """integration -> connected | ready | setup, for every catalog app."""
    live = {c["integration"] for c in connections if c["status"] == "active"}
    keys = offered_keys if offered_keys is not None else {a.integration for a in catalog.APPS}
    return {a.integration: "connected" if a.integration in live else "ready" if a.integration in keys else "setup"
            for a in catalog.APPS}


def planner_system(min_minutes: int, statuses: Optional[dict[str, str]] = None) -> str:
    """statuses: every catalog app's status for this person (connected,
    ready to connect, not set up yet). The planner may offer any of them
    as an option in a decision, but builds steps only on connected ones."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    statuses = statuses or {a.integration: "ready" for a in catalog.APPS}
    apps = "\n".join(f"- {a.integration}: {a.name} ({a.kind}s; acts: {', '.join(a.acts) or 'none'}; "
                     f"{APP_STATUS[statuses.get(a.integration, 'setup')]})" for a in catalog.APPS)
    return f"""You plan a standing goal for one person. TODAY IS {today} (UTC). A smaller model will run this plan in short sessions, again and again, every check_every_minutes. Each session can: read what the person's connected apps showed in the last 7 days (emails, messages, events, files, pages), search the web and read web pages, search the person's memory bank, save short facts to it (one fact per save, at most a few per session), file action items for the person, write drafts (email replies, Slack posts, calendar events, additions to Notion pages), send or post a draft only when the goal is permitted to, mark plan steps done, and report when it is stuck.

What a session CANNOT do: call any API or data feed other than the connected apps below, run code, download files in bulk, or store tables or spreadsheets (memory holds short facts, not rows of data). Plans that need those must say so in missing.

Never build a plan around an app the person has not connected, or around a place for the data they did not choose. When the plan turns on a choice only the person can make (which app or connector to use, where results should live, how far back or how wide the scope goes, whether the goal may act on its own), FIRST return those choices as decisions and leave steps empty: each decision is one plain question, why it matters, 2 to 4 options (label plus one line of what it means), and the index of the option you recommend. Crystal's own memory is always one option for where results live. Ask only what changes the plan; at most {MAX_DECISIONS}. When the person has already answered decisions (given below), those answers are final: never ask them again, and write the full plan on them with decisions empty.

Every app Crystal supports, with its status for this person. Steps may only use apps marked connected. Any app may appear as an option in a decision, labelled with its status. Acts are the only writes possible through an app (for example Google Sheets: create_spreadsheet, append_rows; Notion: append_page, which adds to an existing page and cannot create databases):
{apps}

Write the plan so that smaller model can follow it without guessing:
- summary: one sentence, what this goal is for.
- success_measure: how anyone can tell it is working, concrete and checkable (a number, a deadline, a state).
- check_every_minutes: how often a session should check in. Match the goal's clock: "reply within 3 hours" needs about 30 to 60; a weekly review needs 10080. Never below {min_minutes}.
- steps: the ordered work, 2 to {MAX_STEPS} steps. Ongoing goals end with a repeating step (for example "Each check-in: ..."). Each step's detail says what to look at and what to produce.
- needs_apps / needs_acts: only the integration ids and acts above that the plan truly relies on.
- standing_instructions: the rules every session follows (what counts as needing a reply, tone of drafts, what never to do). Short and specific.
- A choice the person must make is a decision, never a missing item. missing is for what the person must DO or PROVIDE (connect something, hand over a key), not for what they must choose.
- missing: everything the goal needs that this account does not have right now, one item each: kind app (an app above that is not connected, or connected with Act off when the plan must send or post), data_source (an API, feed or dataset none of the apps provide), tool (an ability sessions lack, such as structured tables), permission (the goal must be allowed to send or post on its own), info (something only the person can tell you). need says what, why says which step needs it, suggestion is the person's next step. blocking is true only when no step of the plan can make real progress without it; when web search can carry the goal for now, even imperfectly, it is not blocking, and the step says so. An empty list when nothing is missing.
Do not invent facts about the person. If the goal would need an app that is not connected and the person has not chosen it in a decision, do not plan around it: ask which app to use (that one, its alternatives, or Crystal memory) as a decision."""


def answered_decisions(plan: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    return [d for d in (plan or {}).get("decisions", []) if d.get("answer")]


def planner_user(goal: dict[str, Any], connections: list[dict[str, Any]]) -> str:
    conns = [f"- {c['integration']}: watch {'on' if c['uses'].get('watch') else 'off'}, "
             f"act {'on' if c['uses'].get('act') else 'off'}, status {c['status']}" for c in connections]
    previous = ""
    p = goal.get("plan") or {}
    done = [s["title"] for s in p.get("steps", []) if s.get("done")]
    if done:
        # Only finished work anchors a revision; an untouched old plan is not
        # shown at all, so a replan starts from the person's words and answers.
        previous = (f"\n\nThe current plan (revise it; keep what is done):\n{json.dumps(p.get('steps', []))[:3000]}"
                    f"\nDone so far: {', '.join(done)}"
                    f"\nLast check-in: {(goal.get('last_summary') or 'none')[:1200]}")
    answered = answered_decisions(goal.get("plan"))
    if answered:
        previous = "\n\nThe person's decisions (final; plan on them):\n" + "\n".join(
            f"- {d['question']} -> {d['answer']}" for d in answered) + previous
    return (f"Goal title: {goal['title']}\n\nIn the person's words:\n{goal['directive'][:4000]}\n\n"
            f"Connected apps:\n{chr(10).join(conns) or '- none yet'}\n\n"
            f"Permissions: may send email on its own: {'yes' if goal.get('may_send') else 'no (drafts wait for the person)'}; "
            f"may post/create on its own: {'yes' if goal.get('may_post') else 'no (drafts wait for the person)'}."
            f"{previous}")


def normalize_missing(raw_items: Any, needs_apps: list[str], connections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The model's list, cleaned, plus what the server knows for certain:
    a needed app that isn't connected (or needs signing in again) blocks."""
    out: list[dict[str, Any]] = []
    seen_apps: set[str] = set()
    live = {c["integration"]: c for c in connections}
    for it in raw_items or []:
        if not isinstance(it, dict):
            continue
        need = str(it.get("need") or "").strip()[:200]
        if not need:
            continue
        kind = str(it.get("kind") or "info")
        kind = kind if kind in MISSING_KINDS else "info"
        item = {"kind": kind, "need": need, "why": str(it.get("why") or "").strip()[:400],
                "suggestion": str(it.get("suggestion") or "").strip()[:400], "blocking": bool(it.get("blocking"))}
        if kind == "app":
            app = next((a for a in catalog.APPS if a.integration in need.lower() or a.name.lower() in need.lower()), None)
            if app is not None:
                if app.integration in seen_apps:
                    continue
                c = live.get(app.integration)
                if c is not None and c["status"] == "active":
                    continue  # connected after all; the model was wrong
                item["integration"] = app.integration
                seen_apps.add(app.integration)
        out.append(item)
    for integ in needs_apps:
        c = live.get(integ)
        if integ in seen_apps or (c is not None and c["status"] == "active"):
            continue
        app = catalog.get(integ)
        name = app.name if app else integ
        out.append({"kind": "app", "integration": integ, "blocking": True,
                    "need": f"Connect {name}" if c is None else f"Sign in to {name} again",
                    "why": "The plan's steps read or act through it.",
                    "suggestion": "Add it from the Command center (Add connection)." if c is None
                    else "Settings > Connections > Sign in again."})
        seen_apps.add(integ)
    return out[:MAX_MISSING]


def normalize_decisions(raw_items: Any, previous: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    """The answered decisions carry over (final); new ones are cleaned and
    capped. A new decision that repeats an answered question is dropped."""
    kept = answered_decisions(previous)
    asked = {d["question"].strip().lower() for d in kept}
    out = list(kept)
    for it in raw_items or []:
        if len(out) >= MAX_DECISIONS + len(kept) or not isinstance(it, dict):
            continue
        q = str(it.get("question") or "").strip()[:300]
        if not q or q.lower() in asked:
            continue
        opts = [{"label": str(o.get("label") or "").strip()[:120], "detail": str(o.get("detail") or "").strip()[:300]}
                for o in (it.get("options") or []) if isinstance(o, dict) and str(o.get("label") or "").strip()][:4]
        if len(opts) < 2:
            continue
        try:
            rec = int(it.get("recommended") or 0)
        except (TypeError, ValueError):
            rec = 0
        out.append({"question": q, "why": str(it.get("why") or "").strip()[:400], "options": opts,
                    "recommended": rec if 0 <= rec < len(opts) else 0, "answer": None})
        asked.add(q.lower())
    return out


STORAGE_GROUPS = ("Docs and knowledge", "Files")
_STATUS_RANK = {"connected": 0, "ready": 1, "setup": 2}


def _option(app: catalog.App, status: str) -> dict[str, str]:
    can = (", can write: " + ", ".join(catalog.ACT_LABEL.get(a, a).lower() for a in app.acts)) if app.acts else ", read only"
    return {"label": app.name, "detail": f"{APP_STATUS[status].split(' (')[0].capitalize()}{can}"}


def app_decision(app: catalog.App, statuses: dict[str, str]) -> dict[str, Any]:
    """The question the server asks when a plan leans on an app the person
    hasn't connected or chosen: that app, up to two alternatives from the
    same group (writable and connected first), and Crystal memory where
    results could live there instead."""
    alts = sorted((a for a in catalog.APPS if a.group == app.group and a.integration != app.integration),
                  key=lambda a: (not a.acts, _STATUS_RANK[statuses.get(a.integration, "setup")]))[:2]
    options = []
    if app.group in STORAGE_GROUPS:
        options.append({"label": "Crystal memory", "detail": "No app needed; results are kept as facts in your bank"})
    options += [_option(a, statuses.get(a.integration, "setup")) for a in [app, *alts]]
    rec = next((i for i, a in enumerate([app, *alts]) if statuses.get(a.integration) == "connected"), None)
    recommended = (rec + (1 if app.group in STORAGE_GROUPS else 0)) if rec is not None else 0
    return {"question": f"This goal would use {app.name}, which isn't connected. What should it use?",
            "why": "The steps depend on it; Crystal won't plan around an app you haven't chosen.",
            "options": options[:4], "recommended": recommended, "answer": None, "asked_by": "crystal"}


def normalize_plan(raw: dict[str, Any], *, min_minutes: int, previous: Optional[dict[str, Any]] = None,
                   connections: Optional[list[dict[str, Any]]] = None,
                   statuses: Optional[dict[str, str]] = None, directive: str = "") -> dict[str, Any]:
    acts_known = {a for app in catalog.APPS for a in app.acts}
    try:
        every = int(raw.get("check_every_minutes") or 60)
    except (TypeError, ValueError):
        every = 60
    every = max(min_minutes, min(every, MAX_CHECK_MINUTES))
    was_done = {str(s.get("title", "")).strip().lower() for s in (previous or {}).get("steps", []) if s.get("done")}
    steps = []
    for s in (raw.get("steps") or [])[:MAX_STEPS]:
        if not isinstance(s, dict):
            continue
        title = str(s.get("title") or "").strip()[:200]
        if not title:
            continue
        steps.append({"n": len(steps) + 1, "title": title, "detail": str(s.get("detail") or "").strip()[:1200],
                      "done": title.lower() in was_done, "note": ""})
    decisions = normalize_decisions(raw.get("decisions"), previous)
    deciding = any(not d.get("answer") for d in decisions)
    if not steps and not deciding:
        raise ValueError("the plan has no steps")
    apps = [a for a in dict.fromkeys(str(x) for x in raw.get("needs_apps") or []) if catalog.get(a)]
    # The server's guarantee (2026-10-24, Anthony: "Instead of defaulting to
    # Notion I should at least get an option for Google Sheets"): a plan
    # that leans on an app the person hasn't connected, and hasn't chosen in
    # an answered decision, becomes a question instead of a plan.
    statuses = statuses or app_statuses(connections or [])
    model_apps = [m.get("integration") for m in normalize_missing(raw.get("missing"), [], connections or [])
                  if m.get("integration")]
    # Named in the person's own words counts as chosen ("post to #team in Slack").
    answered_text = (directive + " " + " ".join(str(d.get("answer") or "") for d in decisions)).lower()
    open_text = " ".join(d["question"] + " " + " ".join(o["label"] for o in d["options"])
                         for d in decisions if not d.get("answer")).lower()
    answered_q = {d["question"] for d in decisions if d.get("answer")}
    for integ in dict.fromkeys([*apps, *model_apps]):
        app = catalog.get(integ)
        if app is None or statuses.get(integ) == "connected" or app.name.lower() in answered_text:
            continue
        if app_decision(app, statuses)["question"] in answered_q:
            # Asked and answered with something else: the person said no to
            # this app. Never ask twice; drop it from what the plan needs.
            apps = [a for a in apps if a != integ]
            continue
        if app.name.lower() in open_text or sum(1 for d in decisions if not d.get("answer")) >= MAX_DECISIONS:
            continue
        decisions.append(app_decision(app, statuses))
        open_text += " " + app.name.lower()
    # The model's own app questions get the writable alternatives it left
    # out (it asked "Notion or Crystal memory?" and never mentioned Google
    # Sheets): every open question that offers an app gains, up to four
    # options, the same-group apps that can write, connected first.
    for d in decisions:
        if d.get("answer") or d.get("asked_by") == "crystal":
            continue
        labels = " ".join(o["label"] for o in d["options"]).lower()
        named = [a for a in catalog.APPS if a.name.lower() in labels]
        for app in named:
            for alt in sorted((a for a in catalog.APPS if a.group == app.group and a.acts and a.name.lower() not in labels),
                              key=lambda a: _STATUS_RANK[statuses.get(a.integration, "setup")]):
                if len(d["options"]) >= 4:
                    break
                d["options"].append(_option(alt, statuses.get(alt.integration, "setup")))
                labels += " " + alt.name.lower()
        # The console marks the recommendation itself; and an app that isn't
        # connected is never recommended over one that is or over memory.
        for o in d["options"]:
            o["label"] = re.sub(r"\s*\(recommended\)\s*$", "", o["label"], flags=re.I)

        def _cost(o: dict) -> int:
            hit = [x for x in catalog.APPS if x.name.lower() in o["label"].lower()]
            return _STATUS_RANK[statuses.get(hit[0].integration, "setup")] if hit else 0
        rec = d.get("recommended") or 0
        if 0 <= rec < len(d["options"]) and _cost(d["options"][rec]) > 0:
            d["recommended"] = min(range(len(d["options"])), key=lambda i: (_cost(d["options"][i]), i))
    if any(not d.get("answer") for d in decisions) and not deciding:
        deciding = True
    if deciding:
        steps = []
    # While the person is still choosing, nothing is "missing" yet: the app
    # they pick decides what to connect. Only the model's own notes stay.
    missing = normalize_missing(raw.get("missing"), [] if deciding else apps, connections or [])
    if deciding:
        missing = [{**m, "blocking": False} for m in missing if m["kind"] != "app"]
    return {
        "summary": str(raw.get("summary") or "").strip()[:400],
        "success_measure": str(raw.get("success_measure") or "").strip()[:400],
        "check_every_minutes": every,
        "steps": steps,
        "needs": {
            "apps": apps,
            "acts": [a for a in dict.fromkeys(str(x) for x in raw.get("needs_acts") or []) if a in acts_known],
        },
        "standing_instructions": str(raw.get("standing_instructions") or "").strip()[:2000],
        "missing": missing,
        "blocked": any(m["blocking"] for m in missing),
        "decisions": decisions,
        "deciding": deciding,
        # Run anyway survives a replan: the person already chose.
        "run_anyway": bool((previous or {}).get("run_anyway")),
    }


async def plan_goal(store: Any, customer: Any, goal: dict[str, Any], *, llm: Any = None) -> dict[str, Any]:
    """Write (or revise) one goal's plan. Opens a goal run so the spend
    shows in Activity; never raises for the worker."""
    from ..config import get_settings
    from ..llm import get_llm_client
    from ..llm.lanes import tenant_background
    from ..scan._seam import metered_call

    settings = get_settings()
    min_minutes = int(settings.goal_min_check_minutes or 15)
    run_id = await store.open_goal_run(customer.id, payload={
        "topic": f"Goal plan: {goal['title'][:80]}", "goal_id": goal["id"], "phase": "plan"})
    now = datetime.now(timezone.utc)
    try:
        connections = await store.list_app_connections(customer.id)
        from ..connectors.service import available_apps
        offered = await available_apps(store, customer.id)
        statuses = app_statuses(connections, {a.integration for a in offered})
        async with tenant_background(customer, store):
            client = llm or get_llm_client()
            raw = await metered_call(client, customer_id=customer.id, origin=PLAN_ORIGIN,
                                     system=planner_system(min_minutes, statuses), user=planner_user(goal, connections),
                                     max_tokens=int(settings.goal_plan_max_tokens or 4000), tier=PLAN_TIER,
                                     store=store, json_schema=PLAN_SCHEMA)
            try:
                model = client._resolve_model(PLAN_TIER, None) if hasattr(client, "_resolve_model") else None
            except Exception:  # noqa: BLE001
                model = None
        plan = normalize_plan(json.loads((raw or "").strip()), min_minutes=min_minutes, previous=goal.get("plan"),
                              connections=connections, statuses=statuses, directive=goal.get("directive") or "")
    except Exception as e:  # noqa: BLE001 — a failed plan leaves the goal as it was and retries in an hour
        logger.warning("goals.plan_failed", goal_id=goal["id"], error_type=type(e).__name__, error=str(e)[:200])
        await store.mark_cognition_task_failed(run_id, error_message=f"{type(e).__name__}: {str(e)[:300]}",
                                               completed_at=datetime.now(timezone.utc))
        from datetime import timedelta
        await store.update_goal(customer.id, goal["id"], next_run_at=now + timedelta(hours=1),
                                last_summary="Couldn't write a plan just now; trying again within the hour.")
        return {"ok": False, "code": "plan_failed", "run_id": run_id}
    deciding = plan["deciding"]
    hold = deciding or (plan["blocked"] and not plan["run_anyway"])
    status = "deciding" if deciding else "needs_setup" if hold else None
    await store.update_goal(customer.id, goal["id"], plan=plan, plan_model=model, planned_at=now,
                            next_run_at=None if hold else now, **({"status": status} if status else {}))
    await store.mark_cognition_task_complete(run_id, result={
        "output_type": "report", "deliverable": plan["summary"], "plan": plan}, completed_at=datetime.now(timezone.utc))
    try:
        await store.record_curation_event(customer.id, event_type=("goal_deciding" if deciding else "goal_needs_setup" if hold
                                                                   else "goal_planned"),
                                          subject_id=goal["id"],
                                          label=(f"Goal needs your answers: {goal['title'][:120]}" if deciding
                                                 else f"Goal needs setup: {goal['title'][:120]}" if hold
                                                 else f"Goal planned: {goal['title'][:120]}"),
                                          payload={"goal_id": goal["id"], "run_id": run_id, "steps": len(plan["steps"]),
                                                   "check_every_minutes": plan["check_every_minutes"]})
    except Exception:  # noqa: BLE001
        logger.debug("goals.event_failed", exc_info=True)
    logger.info("goals.planned", goal_id=goal["id"], steps=len(plan["steps"]), every=plan["check_every_minutes"],
                missing=len(plan["missing"]), held=hold)
    return {"ok": True, "plan": plan, "run_id": run_id, "held": hold}
