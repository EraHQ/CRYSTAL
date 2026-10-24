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


def planner_system(min_minutes: int, offered: Optional[list] = None) -> str:
    """offered: the apps this person can connect right now (Q118 A); the
    planner never proposes one that isn't offered."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    apps = "\n".join(f"- {a.integration}: {a.name} ({a.kind}s; acts: {', '.join(a.acts) or 'none'})"
                     for a in (catalog.APPS if offered is None else offered)) or "- none yet"
    return f"""You plan a standing goal for one person. TODAY IS {today} (UTC). A smaller model will run this plan in short sessions, again and again, every check_every_minutes. Each session can: read what the person's connected apps showed in the last 7 days (emails, messages, events, files, pages), search the web and read web pages, search the person's memory bank, save short facts to it (one fact per save, at most a few per session), file action items for the person, write drafts (email replies, Slack posts, calendar events, additions to Notion pages), send or post a draft only when the goal is permitted to, mark plan steps done, and report when it is stuck.

What a session CANNOT do: call any API or data feed other than the connected apps below, run code, download files in bulk, or store tables or spreadsheets (memory holds short facts, not rows of data). Plans that need those must say so in missing.

Never build a plan around an app the person has not connected, or around a place for the data they did not choose. When the plan turns on a choice only the person can make (which app or connector to use, where results should live, how far back or how wide the scope goes, whether the goal may act on its own), FIRST return those choices as decisions and leave steps empty: each decision is one plain question, why it matters, 2 to 4 options (label plus one line of what it means), and the index of the option you recommend. Crystal's own memory is always one option for where results live. Ask only what changes the plan; at most {MAX_DECISIONS}. When the person has already answered decisions (given below), those answers are final: never ask them again, and write the full plan on them with decisions empty.

Apps this person can connect (integration ids; connected or not, see below). Never propose any other app:
{apps}

Write the plan so that smaller model can follow it without guessing:
- summary: one sentence, what this goal is for.
- success_measure: how anyone can tell it is working, concrete and checkable (a number, a deadline, a state).
- check_every_minutes: how often a session should check in. Match the goal's clock: "reply within 3 hours" needs about 30 to 60; a weekly review needs 10080. Never below {min_minutes}.
- steps: the ordered work, 2 to {MAX_STEPS} steps. Ongoing goals end with a repeating step (for example "Each check-in: ..."). Each step's detail says what to look at and what to produce.
- needs_apps / needs_acts: only the integration ids and acts above that the plan truly relies on.
- standing_instructions: the rules every session follows (what counts as needing a reply, tone of drafts, what never to do). Short and specific.
- missing: everything the goal needs that this account does not have right now, one item each: kind app (an app above that is not connected, or connected with Act off when the plan must send or post), data_source (an API, feed or dataset none of the apps provide), tool (an ability sessions lack, such as structured tables), permission (the goal must be allowed to send or post on its own), info (something only the person can tell you). need says what, why says which step needs it, suggestion is the person's next step. blocking is true only when no step of the plan can make real progress without it; when web search can carry the goal for now, even imperfectly, it is not blocking, and the step says so. An empty list when nothing is missing.
Do not invent facts about the person. If the goal needs an app they have not connected, plan for it anyway and list it in needs_apps and missing."""


def answered_decisions(plan: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    return [d for d in (plan or {}).get("decisions", []) if d.get("answer")]


def planner_user(goal: dict[str, Any], connections: list[dict[str, Any]]) -> str:
    conns = [f"- {c['integration']}: watch {'on' if c['uses'].get('watch') else 'off'}, "
             f"act {'on' if c['uses'].get('act') else 'off'}, status {c['status']}" for c in connections]
    previous = ""
    if goal.get("plan"):
        p = goal["plan"]
        done = [s["title"] for s in p.get("steps", []) if s.get("done")]
        previous = (f"\n\nThe current plan (revise it; keep what is done):\n{json.dumps(p.get('steps', []))[:3000]}"
                    f"\nDone so far: {', '.join(done) or 'nothing'}"
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


def normalize_plan(raw: dict[str, Any], *, min_minutes: int, previous: Optional[dict[str, Any]] = None,
                   connections: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
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
        async with tenant_background(customer, store):
            client = llm or get_llm_client()
            raw = await metered_call(client, customer_id=customer.id, origin=PLAN_ORIGIN,
                                     system=planner_system(min_minutes, offered), user=planner_user(goal, connections),
                                     max_tokens=int(settings.goal_plan_max_tokens or 4000), tier=PLAN_TIER,
                                     store=store, json_schema=PLAN_SCHEMA)
            try:
                model = client._resolve_model(PLAN_TIER, None) if hasattr(client, "_resolve_model") else None
            except Exception:  # noqa: BLE001
                model = None
        plan = normalize_plan(json.loads((raw or "").strip()), min_minutes=min_minutes, previous=goal.get("plan"),
                              connections=connections)
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
