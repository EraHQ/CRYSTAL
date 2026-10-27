"""Rollback to any step (Q137 A, 2026-10-27).

Anthony: "a rollback to any step should be possible". Rolling back to
step n returns the goal to where it stood when step n-1 was accepted (to
when the plan was written, for step 1): the plan of that snapshot comes
back with steps n onward open again, and every act the goal recorded
after it is undone, newest first, at any age (a goal's undo records are
kept for its whole life). Drafts and to-dos filed after it are
dismissed. What was sent cannot be unsent: those are listed, never
hidden.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)

UNDONE_BY_ROLLBACK = ("lattice_write", "save_fact")


def _when(iso: str) -> datetime:
    dt = datetime.fromisoformat(iso)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def base_snapshot(snapshots: list[dict[str, Any]], to_step: int) -> Optional[dict[str, Any]]:
    """The snapshot step to_step starts from: the latest plan written (step
    0), or step to_step-1's acceptance under that plan."""
    plans = [s for s in snapshots if s["step_n"] == 0]
    if not plans:
        return None
    plan = plans[-1]
    if to_step <= 1:
        return plan
    after = [s for s in snapshots if s["step_n"] == to_step - 1 and _when(s["created_at"]) >= _when(plan["created_at"])]
    return after[-1] if after else None


def reopened(plan: dict[str, Any], to_step: int) -> dict[str, Any]:
    plan = copy.deepcopy(plan)
    for s in plan.get("steps") or []:
        if int(s.get("n", 0)) >= to_step:
            for k in ("accepted_at", "last_check", "state", "attempts"):
                s.pop(k, None)
            s["done"] = False
            s["note"] = ""
    return plan


async def rollback(store: Any, customer_id: str, goal: dict[str, Any], to_step: int, *, reason: str,
                   state: Optional[dict] = None, actor: str = "console") -> dict[str, Any]:
    from ..cognition.controls import undo_action
    from ..lattices import service as lat
    from .service import GoalRefused

    steps = (goal.get("plan") or {}).get("steps") or []
    if not isinstance(to_step, int) or isinstance(to_step, bool) or not 1 <= to_step <= len(steps):
        raise GoalRefused("no_such_step", f"Roll back to a step from 1 to {len(steps)}.")
    base = base_snapshot(await store.list_goal_snapshots(customer_id, goal["id"]), to_step)
    if base is None:
        raise GoalRefused("no_snapshot", (f"Step {to_step - 1} was never accepted under this plan, so there is no point "
                                          f"to return to. Roll back to an earlier step." if to_step > 1 else
                                          "This plan was written before rollback existed; replan the goal instead."), 409)
    cut = _when(base["created_at"])
    undone, not_undone = 0, []
    for a in await store.list_steward_actions_by_actor(customer_id, f"goal:{goal['id']}", since=cut):
        if a["undone_at"]:
            continue
        if a["action"] in UNDONE_BY_ROLLBACK and a.get("has_inverse"):
            out = await undo_action(store, customer_id, a["id"], state=state, actor=actor, any_age=True)
            if out.get("undone"):
                undone += 1
                continue
            not_undone.append({"label": a["label"], "why": out.get("code") or "undo_failed"})
        else:
            not_undone.append({"label": a["label"], "why": "sent" if a["subject_kind"] == "goal_item" else "not_undoable"})
    dismissed, sent = 0, []
    for it in await store.goal_items_created_after(customer_id, goal["id"], cut):
        if it["status"] == "open":
            await store.update_goal_item(customer_id, it["id"], status="dismissed")
            dismissed += 1
        elif it["kind"] == "sent" or it["status"] == "sent":
            sent.append(it["title"])
    keep = {str(x.get("name") or "").lower() for x in (base["plan"].get("lattices") or [])}
    removed = []
    for x in await store.list_lattices(customer_id):
        made = x.get("created_at")
        if x.get("goal_id") != goal["id"] or not made or _when(made) <= cut or x["name"].lower() in keep:
            continue
        if await store.count_lattice_rows(customer_id, x["id"]) == 0:
            await lat.delete(store, customer_id, x["id"])
            removed.append(x["name"])
    plan = reopened(base["plan"], to_step)
    await store.delete_goal_snapshots_after(customer_id, goal["id"], cut)
    fields: dict[str, Any] = {"plan": plan}
    if goal["status"] == "active":
        fields["next_run_at"] = None
    updated = await store.update_goal(customer_id, goal["id"], **fields)
    label = f"Goal rolled back to step {to_step}: {goal['title'][:80]}"
    try:
        await store.record_curation_event(customer_id, event_type="goal_rolled_back", subject_id=goal["id"], label=label[:256],
                                          payload={"goal_id": goal["id"], "to_step": to_step, "undone": undone,
                                                   "not_undone": len(not_undone), "dismissed": dismissed, "sent": len(sent),
                                                   "reason": reason[:500], "by": actor})
    except Exception:  # noqa: BLE001
        logger.debug("goals.event_failed", exc_info=True)
    logger.info("goals.rolled_back", goal_id=goal["id"], to_step=to_step, undone=undone, not_undone=len(not_undone))
    return {"ok": True, "goal": updated, "to_step": to_step, "returned_to": base["created_at"], "undone": undone,
            "not_undone": not_undone[:50], "dismissed_items": dismissed, "sent_items": sent[:50],
            "lattices_removed": removed}
