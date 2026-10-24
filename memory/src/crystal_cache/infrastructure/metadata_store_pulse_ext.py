"""The Command center's pulse (2026-10-24, Q110 B): the SQL behind the
system-wide stats. Read-only, one tenant, UTC days. Every series is a
GROUP BY on an indexed timestamp (func.date works on Postgres and
SQLite alike); nothing here writes. R9: the SQL lives here.

  daily_fact_counts      facts written per day (all origins)
  fact_total             facts in the bank now
  daily_recall_counts    questions per day, and how many found memory
                         (query_logs.match_type high | low; none = not found)
  daily_autonomous_acts  ledger rows by the Steward, Crystal's scans or a
                         goal, per day and action; plus rule approvals
                         (memory_allowed events)
  facts_by_tier          facts per crystal quality tier
  goal_items_since       goal items touched since a time (status, kind)
  reply_pairs            (email arrived, drafted, sent) for goal items made
                         from a watched email
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from sqlalchemy import func, or_, select

from .schema import (
    CrystalRow, CurationEventRow, FactRow, GoalItemRow, QueryLogRow, StewardActionRow, WatchedItemRow,
)

AUTONOMOUS = ("steward", "system")


def _day(v: Any) -> str:
    if isinstance(v, (datetime, date)):
        return v.isoformat()[:10]
    return str(v)[:10]


class PulseExtensionsMixin:
    async def daily_fact_counts(self, customer_id: str, since: datetime) -> dict[str, int]:
        d = func.date(FactRow.created_at)
        q = (select(d, func.count(FactRow.id)).join(CrystalRow, FactRow.crystal_id == CrystalRow.id)
             .where(CrystalRow.customer_id == customer_id, FactRow.created_at >= since).group_by(d))
        async with self.session() as session:  # type: ignore[attr-defined]
            return {_day(k): int(n) for k, n in (await session.execute(q)).all()}

    async def fact_total(self, customer_id: str) -> int:
        q = (select(func.count(FactRow.id)).join(CrystalRow, FactRow.crystal_id == CrystalRow.id)
             .where(CrystalRow.customer_id == customer_id))
        async with self.session() as session:  # type: ignore[attr-defined]
            return int((await session.execute(q)).scalar_one())

    async def daily_recall_counts(self, customer_id: str, since: datetime) -> dict[str, dict[str, int]]:
        d = func.date(QueryLogRow.timestamp)
        q = (select(d, QueryLogRow.match_type, func.count(QueryLogRow.id))
             .where(QueryLogRow.customer_id == customer_id, QueryLogRow.timestamp >= since)
             .group_by(d, QueryLogRow.match_type))
        out: dict[str, dict[str, int]] = {}
        async with self.session() as session:  # type: ignore[attr-defined]
            for k, mt, n in (await session.execute(q)).all():
                row = out.setdefault(_day(k), {"asked": 0, "found": 0})
                row["asked"] += int(n)
                if mt in ("high", "low"):
                    row["found"] += int(n)
        return out

    async def daily_autonomous_acts(self, customer_id: str, since: datetime) -> list[tuple[str, str, int]]:
        """(day, action, count) for acts nobody clicked: the Steward, a
        scan (actor system), a goal (actor goal:<id>), and rule approvals."""
        d = func.date(StewardActionRow.created_at)
        q = (select(d, StewardActionRow.action, func.count(StewardActionRow.id))
             .where(StewardActionRow.customer_id == customer_id, StewardActionRow.created_at >= since)
             .where(or_(StewardActionRow.actor.in_(AUTONOMOUS), StewardActionRow.actor.like("goal:%")))
             .group_by(d, StewardActionRow.action))
        e = func.date(CurationEventRow.created_at)
        q2 = (select(e, func.count(CurationEventRow.id))
              .where(CurationEventRow.customer_id == customer_id, CurationEventRow.created_at >= since,
                     CurationEventRow.event_type == "memory_allowed").group_by(e))
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = [(_day(k), str(a), int(n)) for k, a, n in (await session.execute(q)).all()]
            rows += [(_day(k), "rule_approval", int(n)) for k, n in (await session.execute(q2)).all()]
        return rows

    async def facts_by_tier(self, customer_id: str) -> dict[str, int]:
        q = (select(CrystalRow.quality_tier, func.count(FactRow.id)).join(CrystalRow, FactRow.crystal_id == CrystalRow.id)
             .where(CrystalRow.customer_id == customer_id).group_by(CrystalRow.quality_tier))
        async with self.session() as session:  # type: ignore[attr-defined]
            return {str(t or "neutral"): int(n) for t, n in (await session.execute(q)).all()}

    async def goal_items_since(self, customer_id: str, since: datetime) -> list[dict[str, Any]]:
        q = (select(GoalItemRow.kind, GoalItemRow.status, func.count(GoalItemRow.id))
             .where(GoalItemRow.customer_id == customer_id, GoalItemRow.updated_at >= since)
             .group_by(GoalItemRow.kind, GoalItemRow.status))
        async with self.session() as session:  # type: ignore[attr-defined]
            return [{"kind": k, "status": s, "count": int(n)} for k, s, n in (await session.execute(q)).all()]

    async def reply_pairs(self, customer_id: str, since: datetime) -> list[dict[str, Optional[datetime]]]:
        """Every goal draft made from a watched email since `since`: when the
        email arrived, when the draft was written, when it was sent."""
        q = (select(WatchedItemRow.occurred_at, WatchedItemRow.created_at, GoalItemRow.created_at,
                    GoalItemRow.updated_at, GoalItemRow.status)
             .join(WatchedItemRow, GoalItemRow.watched_item_id == WatchedItemRow.id)
             .where(GoalItemRow.customer_id == customer_id, GoalItemRow.kind == "draft",
                    WatchedItemRow.kind == "email", GoalItemRow.created_at >= since))
        out = []
        async with self.session() as session:  # type: ignore[attr-defined]
            for occurred, seen, drafted, updated, status in (await session.execute(q)).all():
                out.append({"arrived": occurred or seen, "drafted": drafted,
                            "sent": updated if status == "sent" else None})
        return out
