"""Goals (2026-10-24, Q109 A): the SQL for goals and goal_items.

Anthony: "'goals' its just a long standing directive that the model
addresses periodically. And consistently works at the next step of each
goal's plans." A goal row holds the person's words and the plan a
frontier model made of them; goal_items holds what the goal's sessions
produced (action items, drafts awaiting a person, sent messages with
their exact text, notes). Same binding pattern as the other mixins.
R9: these tables' SQL lives here and nowhere else.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import delete, func, or_, select, update

from .schema import CognitionTaskRow, GoalCommentRow, GoalItemRow, GoalRow, GoalSnapshotRow

GOAL_STATUSES = ("draft", "active", "deciding", "needs_setup", "paused", "done", "archived")
ITEM_KINDS = ("action", "draft", "sent", "note")
ITEM_STATUSES = ("open", "done", "dismissed", "sent", "failed")
_GOAL_FIELDS = {
    "title", "directive", "status", "plan", "plan_model", "planned_at", "next_run_at", "last_run_at",
    "last_task_id", "last_summary", "may_send", "may_post",
}
_ITEM_FIELDS = {"title", "body", "target", "status", "due_at"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def goal_dict(r: GoalRow) -> dict[str, Any]:
    return {
        "id": r.id, "customer_id": r.customer_id, "title": r.title, "directive": r.directive,
        "status": r.status, "plan": dict(r.plan) if r.plan else None, "plan_model": r.plan_model,
        "planned_at": _iso(r.planned_at), "next_run_at": _iso(r.next_run_at), "last_run_at": _iso(r.last_run_at),
        "last_task_id": r.last_task_id, "last_summary": r.last_summary or "",
        "may_send": bool(r.may_send), "may_post": bool(r.may_post),
        "created_at": _iso(r.created_at), "updated_at": _iso(r.updated_at),
    }


def goal_comment_dict(r: GoalCommentRow) -> dict[str, Any]:
    return {"id": r.id, "goal_id": r.goal_id, "body": r.body, "after_run_id": r.after_run_id,
            "seen_by_run_id": r.seen_by_run_id, "seen_at": _iso(r.seen_at), "created_at": _iso(r.created_at)}


def goal_snapshot_dict(r: GoalSnapshotRow) -> dict[str, Any]:
    return {"id": r.id, "goal_id": r.goal_id, "step_n": int(r.step_n or 0), "plan": dict(r.plan or {}),
            "run_id": r.run_id, "rows": dict(r.rows or {}), "created_at": _iso(r.created_at)}


def _aware(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def goal_run_dict(r: CognitionTaskRow) -> dict[str, Any]:
    return {"id": r.id, "status": r.status, "payload": dict(r.payload or {}), "result": dict(r.result or {}),
            "error": r.error_message, "created_at": _iso(r.created_at), "started_at": _iso(r.started_at),
            "completed_at": _iso(r.completed_at)}


def goal_item_dict(r: GoalItemRow) -> dict[str, Any]:
    return {
        "id": r.id, "goal_id": r.goal_id, "customer_id": r.customer_id, "kind": r.kind, "title": r.title or "",
        "body": r.body or "", "watched_item_id": r.watched_item_id, "target": dict(r.target or {}),
        "status": r.status, "due_at": _iso(r.due_at),
        "created_at": _iso(r.created_at), "updated_at": _iso(r.updated_at),
    }


class GoalsExtensionsMixin:
    """goals / goal_items, bound onto MetadataStore."""

    # ---------------------------------------------------------------- goals

    async def create_goal(self, customer_id: str, *, title: str, directive: str, may_send: bool = False,
                          may_post: bool = False, created_by_operator_id: Optional[str] = None) -> dict[str, Any]:
        now = _now()
        row = GoalRow(id=f"goal_{uuid.uuid4().hex[:16]}", customer_id=customer_id, title=title[:200],
                      directive=directive, status="draft", plan=None, last_summary="",
                      may_send=bool(may_send), may_post=bool(may_post),
                      created_by_operator_id=created_by_operator_id, created_at=now, updated_at=now)
        async with self.session() as session:  # type: ignore[attr-defined]
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return goal_dict(row)

    async def list_goals(self, customer_id: str, *, include_archived: bool = False) -> list[dict[str, Any]]:
        q = select(GoalRow).where(GoalRow.customer_id == customer_id)
        if not include_archived:
            q = q.where(GoalRow.status != "archived")
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(q.order_by(GoalRow.created_at.asc()))).scalars().all()
            return [goal_dict(r) for r in rows]

    async def count_goals(self, customer_id: str) -> int:
        async with self.session() as session:  # type: ignore[attr-defined]
            n = await session.scalar(select(func.count()).select_from(GoalRow).where(
                GoalRow.customer_id == customer_id, GoalRow.status != "archived"))
            return int(n or 0)

    async def get_goal(self, customer_id: str, goal_id: str) -> Optional[dict[str, Any]]:
        async with self.session() as session:  # type: ignore[attr-defined]
            r = await session.get(GoalRow, goal_id)
            if r is None or r.customer_id != customer_id:
                return None
            return goal_dict(r)

    async def update_goal(self, customer_id: str, goal_id: str, **fields: Any) -> Optional[dict[str, Any]]:
        bad = set(fields) - _GOAL_FIELDS
        if bad:
            raise ValueError(f"update_goal: unknown field(s) {sorted(bad)}")
        async with self.session() as session:  # type: ignore[attr-defined]
            r = await session.get(GoalRow, goal_id)
            if r is None or r.customer_id != customer_id:
                return None
            for k, v in fields.items():
                setattr(r, k, v)
            r.updated_at = _now()
            await session.commit()
            await session.refresh(r)
            return goal_dict(r)

    async def delete_goal(self, customer_id: str, goal_id: str) -> bool:
        async with self.session() as session:  # type: ignore[attr-defined]
            r = await session.get(GoalRow, goal_id)
            if r is None or r.customer_id != customer_id:
                return False
            await session.execute(delete(GoalItemRow).where(GoalItemRow.goal_id == goal_id))
            await session.execute(delete(GoalCommentRow).where(GoalCommentRow.goal_id == goal_id))
            await session.execute(delete(GoalSnapshotRow).where(GoalSnapshotRow.goal_id == goal_id))
            await session.delete(r)
            await session.commit()
            return True

    async def list_due_goals(self, *, now: Optional[datetime] = None, limit: int = 20) -> list[dict[str, Any]]:
        """Active goals whose next run has come (NULL: never run or just
        (re)planned), oldest first. Unplanned goals (planned_at NULL) are
        the planner's queue; the runner tells them apart."""
        now = now or _now()
        q = (select(GoalRow).where(GoalRow.status == "active")
             .where(or_(GoalRow.next_run_at.is_(None), GoalRow.next_run_at <= now))
             .order_by(GoalRow.next_run_at.asc().nullsfirst()).limit(limit))
        async with self.session() as session:  # type: ignore[attr-defined]
            return [goal_dict(r) for r in (await session.execute(q)).scalars().all()]

    async def list_goals_by_status(self, status: str, *, limit: int = 100) -> list[dict[str, Any]]:
        """Every tenant's goals in one status (the runner re-checks held goals)."""
        q = select(GoalRow).where(GoalRow.status == status).order_by(GoalRow.updated_at.asc()).limit(limit)
        async with self.session() as session:  # type: ignore[attr-defined]
            return [goal_dict(r) for r in (await session.execute(q)).scalars().all()]

    async def claim_goal(self, goal_id: str, *, now: Optional[datetime] = None, lease_until: datetime) -> bool:
        """Compare-and-set: push a due goal's next_run_at to the lease so a
        second worker instance never runs the same check-in."""
        now = now or _now()
        stmt = (update(GoalRow).where(GoalRow.id == goal_id, GoalRow.status == "active")
                .where(or_(GoalRow.next_run_at.is_(None), GoalRow.next_run_at <= now))
                .values(next_run_at=lease_until))
        async with self.session() as session:  # type: ignore[attr-defined]
            res = await session.execute(stmt)
            await session.commit()
            return bool(res.rowcount)

    async def open_goal_run(self, customer_id: str, *, payload: dict[str, Any]) -> str:
        """A cognition task row of type "goal", opened RUNNING (the task lane
        never claims it), closed by the planner or session with
        mark_cognition_task_complete / _failed so it shows in Activity."""
        task_id = f"cog_{uuid.uuid4().hex[:16]}"
        now = _now()
        async with self.session() as session:  # type: ignore[attr-defined]
            session.add(CognitionTaskRow(id=task_id, customer_id=customer_id, task_type="goal", payload=payload,
                                         priority="background", status="running", created_at=now, started_at=now))
            await session.commit()
        return task_id

    # ---------------------------------------------------------------- items

    async def add_goal_item(self, customer_id: str, goal_id: str, *, kind: str, title: str = "", body: str = "",
                            watched_item_id: Optional[str] = None, target: Optional[dict[str, Any]] = None,
                            status: str = "open", due_at: Optional[datetime] = None) -> dict[str, Any]:
        if kind not in ITEM_KINDS:
            raise ValueError(f"add_goal_item: kind must be one of {ITEM_KINDS}")
        if status not in ITEM_STATUSES:
            raise ValueError(f"add_goal_item: status must be one of {ITEM_STATUSES}")
        now = _now()
        row = GoalItemRow(id=f"gi_{uuid.uuid4().hex[:16]}", goal_id=goal_id, customer_id=customer_id, kind=kind,
                          title=title[:512], body=body, watched_item_id=watched_item_id, target=dict(target or {}),
                          status=status, due_at=due_at, created_at=now, updated_at=now)
        async with self.session() as session:  # type: ignore[attr-defined]
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return goal_item_dict(row)

    async def get_goal_item(self, customer_id: str, item_id: str) -> Optional[dict[str, Any]]:
        async with self.session() as session:  # type: ignore[attr-defined]
            r = await session.get(GoalItemRow, item_id)
            if r is None or r.customer_id != customer_id:
                return None
            return goal_item_dict(r)

    async def update_goal_item(self, customer_id: str, item_id: str, **fields: Any) -> Optional[dict[str, Any]]:
        bad = set(fields) - _ITEM_FIELDS
        if bad:
            raise ValueError(f"update_goal_item: unknown field(s) {sorted(bad)}")
        if "status" in fields and fields["status"] not in ITEM_STATUSES:
            raise ValueError(f"update_goal_item: status must be one of {ITEM_STATUSES}")
        async with self.session() as session:  # type: ignore[attr-defined]
            r = await session.get(GoalItemRow, item_id)
            if r is None or r.customer_id != customer_id:
                return None
            for k, v in fields.items():
                setattr(r, k, v)
            r.updated_at = _now()
            await session.commit()
            await session.refresh(r)
            return goal_item_dict(r)

    async def list_goal_items(self, customer_id: str, goal_id: Optional[str] = None, *,
                              status: Optional[str] = None, kind: Optional[str] = None,
                              limit: int = 100) -> list[dict[str, Any]]:
        q = select(GoalItemRow).where(GoalItemRow.customer_id == customer_id)
        if goal_id:
            q = q.where(GoalItemRow.goal_id == goal_id)
        if status:
            q = q.where(GoalItemRow.status == status)
        if kind:
            q = q.where(GoalItemRow.kind == kind)
        q = q.order_by(GoalItemRow.created_at.desc()).limit(max(1, min(int(limit), 500)))
        async with self.session() as session:  # type: ignore[attr-defined]
            return [goal_item_dict(r) for r in (await session.execute(q)).scalars().all()]

    async def goal_items_for_watched(self, customer_id: str, goal_id: str, watched_item_ids: list[str]) -> set[str]:
        """Which of these watched items this goal already made an item for
        (a session never files the same email twice)."""
        if not watched_item_ids:
            return set()
        q = select(GoalItemRow.watched_item_id).where(
            GoalItemRow.customer_id == customer_id, GoalItemRow.goal_id == goal_id,
            GoalItemRow.watched_item_id.in_(watched_item_ids))
        async with self.session() as session:  # type: ignore[attr-defined]
            return {x for x in (await session.execute(q)).scalars().all() if x}

    async def goal_items_created_after(self, customer_id: str, goal_id: str, after: datetime) -> list[dict[str, Any]]:
        q = (select(GoalItemRow).where(GoalItemRow.customer_id == customer_id, GoalItemRow.goal_id == goal_id,
                                       GoalItemRow.created_at > after).order_by(GoalItemRow.created_at.desc()))
        async with self.session() as session:  # type: ignore[attr-defined]
            return [goal_item_dict(r) for r in (await session.execute(q)).scalars().all()]

    # ------------------------------------------------------------- comments

    async def add_goal_comment(self, customer_id: str, goal_id: str, *, body: str,
                               after_run_id: Optional[str] = None) -> dict[str, Any]:
        row = GoalCommentRow(id=f"gc_{uuid.uuid4().hex[:16]}", goal_id=goal_id, customer_id=customer_id,
                             body=body, after_run_id=after_run_id, created_at=_now())
        async with self.session() as session:  # type: ignore[attr-defined]
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return goal_comment_dict(row)

    async def list_goal_comments(self, customer_id: str, goal_id: str, *, unseen: bool = False,
                                 limit: int = 200) -> list[dict[str, Any]]:
        """Oldest first: a check-in reads them in the order they were said."""
        q = select(GoalCommentRow).where(GoalCommentRow.customer_id == customer_id, GoalCommentRow.goal_id == goal_id)
        if unseen:
            q = q.where(GoalCommentRow.seen_by_run_id.is_(None))
        q = q.order_by(GoalCommentRow.created_at.asc()).limit(max(1, min(int(limit), 500)))
        async with self.session() as session:  # type: ignore[attr-defined]
            return [goal_comment_dict(r) for r in (await session.execute(q)).scalars().all()]

    async def mark_goal_comments_seen(self, customer_id: str, comment_ids: list[str], run_id: str) -> int:
        if not comment_ids:
            return 0
        stmt = (update(GoalCommentRow).where(GoalCommentRow.customer_id == customer_id, GoalCommentRow.id.in_(comment_ids),
                                             GoalCommentRow.seen_by_run_id.is_(None))
                .values(seen_by_run_id=run_id, seen_at=_now()))
        async with self.session() as session:  # type: ignore[attr-defined]
            res = await session.execute(stmt)
            await session.commit()
            return int(res.rowcount or 0)

    # ------------------------------------------------------------ snapshots

    async def add_goal_snapshot(self, customer_id: str, goal_id: str, *, step_n: int, plan: dict[str, Any],
                                run_id: Optional[str] = None, rows: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        row = GoalSnapshotRow(id=f"gs_{uuid.uuid4().hex[:16]}", goal_id=goal_id, customer_id=customer_id,
                              step_n=int(step_n), plan=dict(plan or {}), run_id=run_id, rows=dict(rows or {}),
                              created_at=_now())
        async with self.session() as session:  # type: ignore[attr-defined]
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return goal_snapshot_dict(row)

    async def list_goal_snapshots(self, customer_id: str, goal_id: str) -> list[dict[str, Any]]:
        q = (select(GoalSnapshotRow).where(GoalSnapshotRow.customer_id == customer_id, GoalSnapshotRow.goal_id == goal_id)
             .order_by(GoalSnapshotRow.created_at.asc(), GoalSnapshotRow.id.asc()))
        async with self.session() as session:  # type: ignore[attr-defined]
            return [goal_snapshot_dict(r) for r in (await session.execute(q)).scalars().all()]

    async def delete_goal_snapshots_after(self, customer_id: str, goal_id: str, after: datetime) -> int:
        stmt = delete(GoalSnapshotRow).where(GoalSnapshotRow.customer_id == customer_id, GoalSnapshotRow.goal_id == goal_id,
                                             GoalSnapshotRow.created_at > after)
        async with self.session() as session:  # type: ignore[attr-defined]
            res = await session.execute(stmt)
            await session.commit()
            return int(res.rowcount or 0)

    # ----------------------------------------------------------------- runs

    async def list_goal_runs(self, customer_id: str, goal_id: str, *, limit: int = 50,
                             phase: Optional[str] = None) -> list[dict[str, Any]]:
        """This goal's planner and check-in runs (cognition tasks of type
        goal), newest first."""
        q = (select(CognitionTaskRow).where(CognitionTaskRow.customer_id == customer_id,
                                            CognitionTaskRow.task_type == "goal",
                                            CognitionTaskRow.payload["goal_id"].as_string() == goal_id))
        if phase:
            q = q.where(CognitionTaskRow.payload["phase"].as_string() == phase)
        q = q.order_by(CognitionTaskRow.created_at.desc(), CognitionTaskRow.id.desc()).limit(max(1, min(int(limit), 200)))
        async with self.session() as session:  # type: ignore[attr-defined]
            return [goal_run_dict(r) for r in (await session.execute(q)).scalars().all()]
