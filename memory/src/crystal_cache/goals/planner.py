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
    },
    "required": ["summary", "success_measure", "check_every_minutes", "steps", "needs_apps", "needs_acts",
                 "standing_instructions"],
    "additionalProperties": False,
}


def planner_system(min_minutes: int) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    apps = "\n".join(f"- {a.integration}: {a.name} ({a.kind}s; acts: {', '.join(a.acts) or 'none'})"
                     for a in catalog.APPS)
    return f"""You plan a standing goal for one person. TODAY IS {today} (UTC). A smaller model will run this plan in short sessions, again and again, every check_every_minutes. Each session can: read what the person's connected apps showed in the last 7 days (emails, messages, events, files, pages), search the person's memory bank, file action items for the person, write drafts (email replies, Slack posts, calendar events, additions to Notion pages), send or post a draft only when the goal is permitted to, mark plan steps done, and report when it is stuck.

Apps that exist (integration ids):
{apps}

Write the plan so that smaller model can follow it without guessing:
- summary: one sentence, what this goal is for.
- success_measure: how anyone can tell it is working, concrete and checkable (a number, a deadline, a state).
- check_every_minutes: how often a session should check in. Match the goal's clock: "reply within 3 hours" needs about 30 to 60; a weekly review needs 10080. Never below {min_minutes}.
- steps: the ordered work, 2 to {MAX_STEPS} steps. Ongoing goals end with a repeating step (for example "Each check-in: ..."). Each step's detail says what to look at and what to produce.
- needs_apps / needs_acts: only the integration ids and acts above that the plan truly relies on.
- standing_instructions: the rules every session follows (what counts as needing a reply, tone of drafts, what never to do). Short and specific.
Do not invent facts about the person. If the goal needs an app they have not connected, plan for it anyway and list it in needs_apps; the session reports it."""


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
    return (f"Goal title: {goal['title']}\n\nIn the person's words:\n{goal['directive'][:4000]}\n\n"
            f"Connected apps:\n{chr(10).join(conns) or '- none yet'}\n\n"
            f"Permissions: may send email on its own: {'yes' if goal.get('may_send') else 'no (drafts wait for the person)'}; "
            f"may post/create on its own: {'yes' if goal.get('may_post') else 'no (drafts wait for the person)'}."
            f"{previous}")


def normalize_plan(raw: dict[str, Any], *, min_minutes: int, previous: Optional[dict[str, Any]] = None) -> dict[str, Any]:
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
    if not steps:
        raise ValueError("the plan has no steps")
    return {
        "summary": str(raw.get("summary") or "").strip()[:400],
        "success_measure": str(raw.get("success_measure") or "").strip()[:400],
        "check_every_minutes": every,
        "steps": steps,
        "needs": {
            "apps": [a for a in dict.fromkeys(str(x) for x in raw.get("needs_apps") or []) if catalog.get(a)],
            "acts": [a for a in dict.fromkeys(str(x) for x in raw.get("needs_acts") or []) if a in acts_known],
        },
        "standing_instructions": str(raw.get("standing_instructions") or "").strip()[:2000],
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
        async with tenant_background(customer, store):
            client = llm or get_llm_client()
            raw = await metered_call(client, customer_id=customer.id, origin=PLAN_ORIGIN,
                                     system=planner_system(min_minutes), user=planner_user(goal, connections),
                                     max_tokens=int(settings.goal_plan_max_tokens or 4000), tier=PLAN_TIER,
                                     store=store, json_schema=PLAN_SCHEMA)
            try:
                model = client._resolve_model(PLAN_TIER, None) if hasattr(client, "_resolve_model") else None
            except Exception:  # noqa: BLE001
                model = None
        plan = normalize_plan(json.loads((raw or "").strip()), min_minutes=min_minutes, previous=goal.get("plan"))
    except Exception as e:  # noqa: BLE001 — a failed plan leaves the goal as it was and retries in an hour
        logger.warning("goals.plan_failed", goal_id=goal["id"], error_type=type(e).__name__, error=str(e)[:200])
        await store.mark_cognition_task_failed(run_id, error_message=f"{type(e).__name__}: {str(e)[:300]}",
                                               completed_at=datetime.now(timezone.utc))
        from datetime import timedelta
        await store.update_goal(customer.id, goal["id"], next_run_at=now + timedelta(hours=1),
                                last_summary="Couldn't write a plan just now; trying again within the hour.")
        return {"ok": False, "code": "plan_failed", "run_id": run_id}
    await store.update_goal(customer.id, goal["id"], plan=plan, plan_model=model, planned_at=now, next_run_at=now)
    await store.mark_cognition_task_complete(run_id, result={
        "output_type": "report", "deliverable": plan["summary"], "plan": plan}, completed_at=datetime.now(timezone.utc))
    try:
        await store.record_curation_event(customer.id, event_type="goal_planned", subject_id=goal["id"],
                                          label=f"Goal planned: {goal['title'][:120]}",
                                          payload={"goal_id": goal["id"], "run_id": run_id, "steps": len(plan["steps"]),
                                                   "check_every_minutes": plan["check_every_minutes"]})
    except Exception:  # noqa: BLE001
        logger.debug("goals.event_failed", exc_info=True)
    logger.info("goals.planned", goal_id=goal["id"], steps=len(plan["steps"]), every=plan["check_every_minutes"])
    return {"ok": True, "plan": plan, "run_id": run_id}
