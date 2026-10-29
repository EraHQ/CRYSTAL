"""Goals, as the console and MCP change them. One implementation, two
doors (endpoints/goals.py, agent/mcp_server.py). Every refusal carries a
code.

  create        a new goal goes straight to the planner (active, unplanned)
  update        title / directive / status / permissions; a new directive
                means a new plan
  replan        ask the frontier tier for a fresh plan (keeps what is done)
  run_now       check in on the next pass
  run_anyway    a goal held for setup (Q112 A) runs with what it has
  decide        answer the planner's questions (step 2 of planning); the
                planner then writes the plan on the answers
  item_action   on one item: send (a draft, optionally edited first),
                done, dismiss, reopen
  comment       what the person says after a check-in; the next check-in
                reads it first (2026-10-27)
  set_criteria  the person edits a step's completion criteria (Q136 A)
  review_step   the person approves or sends back a step that waits on
                them (a "person" criterion)
  rollback      back to any step (Q137 A; goals/rollback.py)
  runs_view     each planner and check-in run, counted
"""
from __future__ import annotations

from typing import Any, Optional

from .. import beta
from ..config import get_settings
from ..connectors import actions as connector_actions

STATUS_CHANGES = ("active", "paused", "done", "archived")
ITEM_ACTIONS = ("send", "done", "dismiss", "reopen", "answer")


class GoalRefused(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def _text(v: Any, field: str, lo: int, hi: int) -> str:
    s = str(v or "").strip()
    if len(s) < lo:
        raise GoalRefused(f"{field}_required", f"The {field} needs at least {lo} characters.")
    if len(s) > hi:
        raise GoalRefused(f"{field}_too_long", f"Keep the {field} under {hi} characters.")
    return s


def _flag(v: Any, field: str) -> bool:
    if not isinstance(v, bool):
        raise GoalRefused("invalid_request", f"{field} must be true or false")
    return v


async def _goal(store: Any, customer_id: str, goal_id: str) -> dict[str, Any]:
    g = await store.get_goal(customer_id, goal_id)
    if g is None:
        raise GoalRefused("not_found", "No such goal.", 404)
    return g


async def _feature(store: Any, customer_id: str) -> None:
    if not await beta.has_feature(store, customer_id, "goals", await beta.tenant_emails(store, customer_id)):
        raise GoalRefused("beta_required", "Goals are in private beta; ask for an invite.", 403)


async def create(store: Any, customer_id: str, *, title: Any, directive: Any, may_send: Any = False,
                 may_post: Any = False, operator_id: Optional[str] = None) -> dict[str, Any]:
    await _feature(store, customer_id)
    title_s = _text(title, "title", 2, 200)
    directive_s = _text(directive, "directive", 10, 4000)
    cap = int(get_settings().goals_max_per_tenant or 10)
    if await store.count_goals(customer_id) >= cap:
        raise GoalRefused("goal_limit", f"You can keep {cap} goals; archive one first.", 409)
    g = await store.create_goal(customer_id, title=title_s, directive=directive_s,
                                may_send=_flag(may_send, "may_send"), may_post=_flag(may_post, "may_post"),
                                created_by_operator_id=operator_id)
    return await store.update_goal(customer_id, g["id"], status="active")


async def update(store: Any, customer_id: str, goal_id: str, *, title: Any = None, directive: Any = None,
                 status: Any = None, may_send: Any = None, may_post: Any = None) -> dict[str, Any]:
    g = await _goal(store, customer_id, goal_id)
    fields: dict[str, Any] = {}
    if title is not None:
        fields["title"] = _text(title, "title", 2, 200)
    if directive is not None:
        d = _text(directive, "directive", 10, 4000)
        if d != g["directive"]:
            fields.update(directive=d, planned_at=None, next_run_at=None)
            if g["status"] in ("needs_setup", "deciding"):
                fields["status"] = "active"  # a new directive gets a fresh plan and a fresh check
                # ... and fresh questions: the old answers were about the old words.
                fields["plan"] = {**(g.get("plan") or {}), "decisions": []}
    if may_send is not None:
        fields["may_send"] = _flag(may_send, "may_send")
    if may_post is not None:
        fields["may_post"] = _flag(may_post, "may_post")
    if status is not None:
        if status not in STATUS_CHANGES:
            raise GoalRefused("invalid_status", f"status must be one of {', '.join(STATUS_CHANGES)}")
        if status == "active" and g["status"] != "active":
            await _feature(store, customer_id)
            fields["next_run_at"] = None
            if g["status"] == "needs_setup":
                # Q112 A: resuming a held goal is the person's "Run anyway".
                fields["plan"] = {**(g.get("plan") or {}), "run_anyway": True}
        fields["status"] = status
    if not fields:
        return g
    return await store.update_goal(customer_id, goal_id, **fields)


async def replan(store: Any, customer_id: str, goal_id: str) -> dict[str, Any]:
    g = await _goal(store, customer_id, goal_id)
    if g["status"] in ("done", "archived"):
        raise GoalRefused("goal_closed", "Reopen the goal first.", 409)
    await _feature(store, customer_id)
    return await store.update_goal(customer_id, goal_id, status="active", planned_at=None, next_run_at=None)


async def decide(store: Any, customer_id: str, goal_id: str, answers: Any) -> dict[str, Any]:
    """answers: [{index, choice?: option index, other?: free text}]. Every
    open decision needs an answer; then the goal goes back to the planner."""
    g = await _goal(store, customer_id, goal_id)
    if g["status"] != "deciding":
        raise GoalRefused("not_deciding", "This goal isn't waiting on answers.", 409)
    if not isinstance(answers, list):
        raise GoalRefused("invalid_request", "answers must be a list")
    plan = dict(g.get("plan") or {})
    decisions = [dict(d) for d in plan.get("decisions", [])]
    for a in answers:
        if not isinstance(a, dict) or not isinstance(a.get("index"), int) or not 0 <= a["index"] < len(decisions):
            raise GoalRefused("bad_answer", "Each answer names a question by its index.")
        d = decisions[a["index"]]
        other = str(a.get("other") or "").strip()
        choice = a.get("choice")
        if other:
            d["answer"] = other[:400]
        elif isinstance(choice, int) and not isinstance(choice, bool) and 0 <= choice < len(d["options"]):
            o = d["options"][choice]
            d["answer"] = f"{o['label']} ({o['detail']})" if o.get("detail") else o["label"]
        else:
            raise GoalRefused("bad_answer", f"Pick an option or write your own for: {d['question']}")
    if any(not d.get("answer") for d in decisions):
        raise GoalRefused("answers_missing", "Answer every question; the plan is written on all of them.")
    plan["decisions"] = decisions
    plan["deciding"] = False
    return await store.update_goal(customer_id, goal_id, plan=plan, status="active", planned_at=None, next_run_at=None)


async def run_anyway(store: Any, customer_id: str, goal_id: str) -> dict[str, Any]:
    g = await _goal(store, customer_id, goal_id)
    if g["status"] != "needs_setup":
        raise GoalRefused("not_held", "This goal isn't waiting on setup.", 409)
    return await update(store, customer_id, goal_id, status="active")


async def run_now(store: Any, customer_id: str, goal_id: str) -> dict[str, Any]:
    g = await _goal(store, customer_id, goal_id)
    if g["status"] != "active":
        raise GoalRefused("goal_not_active", "Only an active goal checks in.", 409)
    return await store.update_goal(customer_id, goal_id, next_run_at=None)


async def item_action(store: Any, customer_id: str, item_id: str, action: str, *, actor: str,
                      title: Optional[str] = None, body: Optional[str] = None) -> dict[str, Any]:
    if action not in ITEM_ACTIONS:
        raise GoalRefused("invalid_action", f"action must be one of {', '.join(ITEM_ACTIONS)}")
    item = await store.get_goal_item(customer_id, item_id)
    if item is None:
        raise GoalRefused("not_found", "No such item.", 404)
    if action == "reopen":
        if item["status"] not in ("done", "dismissed"):
            raise GoalRefused("not_closed", "Only a done or dismissed item reopens.", 409)
        return {"ok": True, "item": await store.update_goal_item(customer_id, item_id, status="open")}
    if item["status"] != "open":
        raise GoalRefused("already_handled", f"This item is already {item['status']}.", 409)
    if action in ("done", "dismiss"):
        status = "done" if action == "done" else "dismissed"
        out = await store.update_goal_item(customer_id, item_id, status=status)
        if item["kind"] == "note":
            await _resume_if_answered(store, customer_id, item["goal_id"])
        return {"ok": True, "item": out}
    if action == "answer":
        # Series 385 (2026-10-28): "Even if it asked for approval there is no
        # way I could approve." The answer to what the goal needs becomes a
        # note every check-in reads until it is handled.
        text = _text(body, "answer", 2, MAX_COMMENT)
        asked = ((item.get("target") or {}).get("problem") or {}).get("question") or item["title"]
        note = await store.add_goal_comment(customer_id, item["goal_id"], body=f"Answer to \"{asked[:300]}\": {text}",
                                            item_id=item_id)
        done = await store.update_goal_item(customer_id, item_id, status="done")
        await _resume_if_answered(store, customer_id, item["goal_id"])
        return {"ok": True, "item": done, "note": note}
    # send: the person's approval is the permission; the app's Act switch still applies
    if item["kind"] != "draft":
        raise GoalRefused("not_a_draft", "Only a draft can be sent.", 409)
    edits: dict[str, Any] = {}
    if title is not None:
        edits["title"] = _text(title, "title", 1, 512)
    if body is not None:
        edits["body"] = _text(body, "body", 1, connector_actions.MAX_BODY)
    if edits:
        item = await store.update_goal_item(customer_id, item_id, **edits)
    goal = await store.get_goal(customer_id, item["goal_id"])
    out = await connector_actions.execute(store, customer_id, item, actor=actor, reason="Approved by you", goal=goal)
    if not out.get("ok"):
        raise GoalRefused(out["code"], out["error"], 409 if out["code"] != "send_failed" else 502)
    return out


async def _resume_if_answered(store: Any, customer_id: str, goal_id: str) -> None:
    """Series 386: a goal waiting on the person checks in again as soon as
    every open question is answered or handled."""
    from . import problems as pb
    g = await store.get_goal(customer_id, goal_id)
    if g is None or not pb.waiting(g):
        return
    if not await pb.open_problems(store, customer_id, goal_id):
        await store.update_goal(customer_id, goal_id, plan={**(g.get("plan") or {}), "waiting_on_you": False},
                                next_run_at=None)


# ------------------------------------------------------------- 2026-10-27

MAX_COMMENT = 4000


async def comment(store: Any, customer_id: str, goal_id: str, body: Any, *, always: Any = False) -> dict[str, Any]:
    g = await _goal(store, customer_id, goal_id)
    text = _text(body, "comment", 2, MAX_COMMENT)
    return await store.add_goal_comment(customer_id, g["id"], body=text, after_run_id=g.get("last_task_id"),
                                        always=_flag(always, "always") if always is not None else False)


async def set_note(store: Any, customer_id: str, goal_id: str, note_id: str, *, status: Any = None,
                   always: Any = None) -> dict[str, Any]:
    """The person closes, reopens or dismisses a note, or makes it a
    standing one (Series 385)."""
    g = await _goal(store, customer_id, goal_id)
    note = await store.get_goal_comment(customer_id, note_id)
    if note is None or note["goal_id"] != g["id"]:
        raise GoalRefused("not_found", "No such note.", 404)
    fields: dict[str, Any] = {}
    if status is not None:
        if status not in ("open", "done", "dismissed"):
            raise GoalRefused("invalid_request", "status is open, done or dismissed")
        fields["status"] = status
    if always is not None:
        fields["always"] = _flag(always, "always")
    if not fields:
        raise GoalRefused("invalid_request", "Give status or always.")
    return await store.update_goal_comment(customer_id, note_id, **fields)


async def set_criteria(store: Any, customer_id: str, goal_id: str, n: Any, criteria: Any) -> dict[str, Any]:
    from .criteria import normalize_criteria
    g = await _goal(store, customer_id, goal_id)
    plan = dict(g.get("plan") or {})
    steps = [dict(s) for s in plan.get("steps") or []]
    hit = next((s for s in steps if isinstance(n, int) and int(s.get("n", 0)) == n), None)
    if hit is None:
        raise GoalRefused("no_such_step", f"The plan has steps 1 to {len(steps)}.", 404)
    if hit.get("repeats"):
        raise GoalRefused("step_repeats", "A repeating step has no completion criteria.", 409)
    if not isinstance(criteria, list):
        raise GoalRefused("invalid_request", "criteria must be a list")
    clean, dropped = normalize_criteria(criteria, plan.get("lattices") or [])
    if dropped or len(clean) != len(criteria):
        raise GoalRefused("bad_criteria", "Some criteria can't be checked: " + "; ".join(dropped or ["unknown kind"]))
    hit["criteria"] = clean
    hit.pop("last_check", None)
    if hit.get("state") == "rejected":
        hit.pop("state", None)
    plan["steps"] = steps
    return await store.update_goal(customer_id, goal_id, plan=plan)


async def review_step(store: Any, customer_id: str, goal_id: str, n: Any, *, approve: Any, note: Any = None) -> dict[str, Any]:
    from datetime import datetime, timezone

    from .session import snapshot
    g = await _goal(store, customer_id, goal_id)
    plan = dict(g.get("plan") or {})
    steps = [dict(s) for s in plan.get("steps") or []]
    hit = next((s for s in steps if isinstance(n, int) and int(s.get("n", 0)) == n), None)
    if hit is None:
        raise GoalRefused("no_such_step", f"The plan has steps 1 to {len(steps)}.", 404)
    if hit.get("state") != "awaiting_review":
        raise GoalRefused("not_awaiting_review", "This step isn't waiting for you.", 409)
    ok = _flag(approve, "approve")
    text = str(note or "").strip()[:MAX_COMMENT]
    if ok:
        hit.update(done=True, state="accepted", accepted_at=datetime.now(timezone.utc).isoformat())
    else:
        if len(text) < 2:
            raise GoalRefused("note_required", "Say what the step still needs; the next check-in reads it.")
        hit["state"] = "rejected"
    plan["steps"] = steps
    g = await store.update_goal(customer_id, goal_id, plan=plan)
    if ok:
        await snapshot(store, customer_id, g, step_n=int(n))
    if text:
        await store.add_goal_comment(customer_id, goal_id, after_run_id=g.get("last_task_id"),
                                     body=f"Step {n} {'approved' if ok else 'sent back'}: {text}")
    return g


async def rollback(store: Any, customer_id: str, goal_id: str, to_step: Any, *, reason: Any = None,
                   state: Optional[dict] = None, actor: str = "console", preview: bool = False) -> dict[str, Any]:
    from .rollback import rollback as do_rollback
    g = await _goal(store, customer_id, goal_id)
    if g["status"] in ("done", "archived"):
        raise GoalRefused("goal_closed", "Reopen the goal first.", 409)
    why = str(reason or "").strip()[:500]
    out = await do_rollback(store, customer_id, g, to_step, reason=why, state=state, actor=actor, preview=preview)
    if why and not preview:
        # Series 382: what it should do differently is said to the next
        # check-in, which reads comments first.
        out["comment"] = await store.add_goal_comment(customer_id, g["id"], body=f"Went back to step {to_step}: {why}"[:MAX_COMMENT],
                                                      after_run_id=g.get("last_task_id"))
    return out


def _seconds(a: Optional[str], b: Optional[str]) -> Optional[float]:
    from datetime import datetime
    if not a or not b:
        return None
    try:
        return round((datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds(), 1)
    except ValueError:
        return None


async def runs_view(store: Any, customer_id: str, goal_id: str, *, limit: int = 30) -> list[dict[str, Any]]:
    """Each run of a goal, newest first: what it did, in counts."""
    g = await _goal(store, customer_id, goal_id)
    comments = {c["id"]: c for c in await store.list_goal_comments(customer_id, g["id"])}
    out = []
    for r in await store.list_goal_runs(customer_id, g["id"], limit=limit):
        res, pay = r.get("result") or {}, r.get("payload") or {}
        tools: dict[str, int] = {}
        for t in res.get("tool_calls") or []:
            tools[t.get("tool") or "?"] = tools.get(t.get("tool") or "?", 0) + 1
        kinds: dict[str, int] = {}
        for a in res.get("actions") or []:
            kinds[a.get("action") or "?"] = kinds.get(a.get("action") or "?", 0) + 1
        view = {"run_id": r["id"], "phase": pay.get("phase"), "status": r["status"], "started_at": r["started_at"],
                "completed_at": r["completed_at"], "seconds": _seconds(r["started_at"], r["completed_at"]),
                "error": r.get("error"), "summary": res.get("deliverable") or "", "model": res.get("model"),
                "stop_reason": res.get("stop_reason")}
        if pay.get("phase") == "plan":
            plan = res.get("plan") or {}
            view.update(steps=len(plan.get("steps") or []), lattices=len(plan.get("lattices") or []),
                        survey=len(((plan.get("survey") or {}).get("findings")) or []))
        else:
            view.update(calls=res.get("calls", 0), tools=tools, actions=len(res.get("actions") or []), action_kinds=kinds,
                        rows=res.get("rows") or {"inserted": 0, "updated": 0}, submissions=res.get("submissions") or [],
                        budget=pay.get("budget"),
                        comments=[{"id": cid, "body": (comments.get(cid) or {}).get("body", "")[:300]}
                                  for cid in pay.get("comments") or []])
        out.append(view)
    return out
