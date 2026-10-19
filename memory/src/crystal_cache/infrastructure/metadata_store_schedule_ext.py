"""scheduled_tasks CRUD (launch feature, 2026-10-18). Bound onto
MetadataStore like every other extension; all SQL lives here (R9).

The claim is a compare-and-set on next_run_at: the scheduler computes
the following occurrence and advances the row only if next_run_at is
still the value it read, so two scheduler processes can tick the same
row and exactly one enqueues the run.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import func, select, update

from ..models.scheduled_task import ScheduledTask
from .schema import ScheduledTaskRow


def _to_model(row: ScheduledTaskRow) -> ScheduledTask:
    return ScheduledTask(
        id=row.id, customer_id=row.customer_id, kind=row.kind, title=row.title,
        prompt=row.prompt, cadence=row.cadence, at_minute=row.at_minute,
        weekday=row.weekday, every_hours=row.every_hours, run_once_at=row.run_once_at,
        tz=row.tz or "UTC", next_run_at=row.next_run_at, status=row.status,
        series_failures=row.series_failures or 0, last_task_id=row.last_task_id,
        last_run_at=row.last_run_at, last_outcome=row.last_outcome,
        last_result_summary=row.last_result_summary, last_crystal_id=row.last_crystal_id,
        deliver_url=row.deliver_url, deliver_secret=row.deliver_secret,
        last_delivery=row.last_delivery, last_delivery_at=row.last_delivery_at,
        created_by_operator_id=row.created_by_operator_id,
        created_at=row.created_at, updated_at=row.updated_at,
    )


def _utc(dt: Optional[datetime]) -> Optional[datetime]:
    """SQLite hands naive datetimes back; the scheduler compares in UTC."""
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


class ScheduleExtensionsMixin:
    """scheduled_tasks CRUD, bound onto MetadataStore."""

    async def create_scheduled_task(
        self,
        customer_id: str,
        *,
        kind: str,
        title: str,
        prompt: str,
        cadence: str,
        tz: str,
        next_run_at: Optional[datetime],
        at_minute: Optional[int] = None,
        weekday: Optional[int] = None,
        every_hours: Optional[int] = None,
        run_once_at: Optional[datetime] = None,
        created_by_operator_id: Optional[str] = None,
        deliver_url: Optional[str] = None,
        deliver_secret: Optional[str] = None,
    ) -> ScheduledTask:
        row = ScheduledTaskRow(
            id=f"sched_{uuid.uuid4().hex[:16]}",
            customer_id=customer_id, kind=kind, title=title[:200], prompt=prompt,
            cadence=cadence, at_minute=at_minute, weekday=weekday, every_hours=every_hours,
            run_once_at=run_once_at, tz=tz, next_run_at=next_run_at, status="active",
            series_failures=0, created_by_operator_id=created_by_operator_id,
            deliver_url=deliver_url or None, deliver_secret=deliver_secret or None,
        )
        async with self.session() as session:  # type: ignore[attr-defined]
            session.add(row)
            await session.flush()
            return _to_model(row)

    async def get_scheduled_task(self, schedule_id: str, customer_id: str) -> Optional[ScheduledTask]:
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(ScheduledTaskRow, schedule_id)
            if row is None or row.customer_id != customer_id:
                return None
            return _to_model(row)

    async def list_scheduled_tasks(self, customer_id: str, *, limit: int = 100) -> list[ScheduledTask]:
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(
                select(ScheduledTaskRow)
                .where(ScheduledTaskRow.customer_id == customer_id)
                .order_by(ScheduledTaskRow.created_at.desc())
                .limit(limit)
            )).scalars().all()
            return [_to_model(r) for r in rows]

    async def count_active_scheduled_tasks(self, customer_id: str) -> int:
        async with self.session() as session:  # type: ignore[attr-defined]
            n = (await session.execute(
                select(func.count(ScheduledTaskRow.id))
                .where(ScheduledTaskRow.customer_id == customer_id)
                .where(ScheduledTaskRow.status.in_(("active", "paused")))
            )).scalar_one()
            return int(n or 0)

    async def set_scheduled_task_status(
        self, schedule_id: str, customer_id: str, status: str, *, next_run_at: Optional[datetime] = None,
    ) -> Optional[ScheduledTask]:
        """Pause / resume (resume passes the recomputed next_run_at and
        clears the failure streak) / park / done."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(ScheduledTaskRow, schedule_id)
            if row is None or row.customer_id != customer_id:
                return None
            row.status = status
            if status == "active":
                row.series_failures = 0
                if next_run_at is not None:
                    row.next_run_at = next_run_at
            return _to_model(row)

    async def set_scheduled_task_delivery(
        self, schedule_id: str, customer_id: str, *, deliver_url: Optional[str], deliver_secret: Optional[str],
    ) -> Optional[ScheduledTask]:
        """Set or clear the webhook. Clearing drops the secret and the
        last delivery record; setting keeps an existing secret unless a
        new one is passed."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(ScheduledTaskRow, schedule_id)
            if row is None or row.customer_id != customer_id:
                return None
            row.deliver_url = deliver_url or None
            if not deliver_url:
                row.deliver_secret = None
                row.last_delivery = None
                row.last_delivery_at = None
            elif deliver_secret:
                row.deliver_secret = deliver_secret
            return _to_model(row)

    async def list_scheduled_tasks_in_flight(self, *, limit: int = 200) -> list[ScheduledTask]:
        """Rows whose last run is still 'queued': the scheduler settles
        these every tick (completed -> result + delivery; failed ->
        streak), whatever the schedule's own status is now."""
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(
                select(ScheduledTaskRow)
                .where(ScheduledTaskRow.last_outcome == "queued")
                .where(ScheduledTaskRow.last_task_id.is_not(None))
                .order_by(ScheduledTaskRow.last_run_at.asc())
                .limit(limit)
            )).scalars().all()
            out = []
            for r in rows:
                m = _to_model(r)
                m.last_run_at = _utc(m.last_run_at)
                out.append(m)
            return out

    async def record_scheduled_result(
        self, schedule_id: str, *, summary: Optional[str], crystal_id: Optional[str],
    ) -> None:
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(ScheduledTaskRow, schedule_id)
            if row is None:
                return
            row.last_result_summary = (summary or "")[:2000] or None
            row.last_crystal_id = crystal_id or None

    async def record_scheduled_delivery(self, schedule_id: str, *, outcome: str, at: datetime) -> None:
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(ScheduledTaskRow, schedule_id)
            if row is None:
                return
            row.last_delivery = (outcome or "")[:120]
            row.last_delivery_at = at

    async def list_recent_scheduled_results(
        self, customer_id: str, *, since: datetime, limit: int = 10,
    ) -> list[ScheduledTask]:
        """Schedules with a result since `since`, newest first (MCP status)."""
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(
                select(ScheduledTaskRow)
                .where(ScheduledTaskRow.customer_id == customer_id)
                .where(ScheduledTaskRow.last_result_summary.is_not(None))
                .where(ScheduledTaskRow.last_run_at.is_not(None))
                .where(ScheduledTaskRow.last_run_at >= since)
                .order_by(ScheduledTaskRow.last_run_at.desc())
                .limit(limit)
            )).scalars().all()
            out = []
            for r in rows:
                m = _to_model(r)
                m.last_run_at = _utc(m.last_run_at)
                out.append(m)
            return out

    async def delete_scheduled_task(self, schedule_id: str, customer_id: str) -> bool:
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(ScheduledTaskRow, schedule_id)
            if row is None or row.customer_id != customer_id:
                return False
            await session.delete(row)
            return True

    async def list_due_scheduled_tasks(self, now: datetime, *, limit: int = 50) -> list[ScheduledTask]:
        """Active rows whose next_run_at has passed, oldest first."""
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(
                select(ScheduledTaskRow)
                .where(ScheduledTaskRow.status == "active")
                .where(ScheduledTaskRow.next_run_at.is_not(None))
                .where(ScheduledTaskRow.next_run_at <= now)
                .order_by(ScheduledTaskRow.next_run_at.asc())
                .limit(limit)
            )).scalars().all()
            out = []
            for r in rows:
                m = _to_model(r)
                m.next_run_at = _utc(m.next_run_at)
                m.last_run_at = _utc(m.last_run_at)
                out.append(m)
            return out

    async def advance_scheduled_task(
        self, schedule_id: str, *, expected_next_run_at: datetime, next_run_at: Optional[datetime],
    ) -> bool:
        """Compare-and-set claim: advance next_run_at (None = a one-shot
        that becomes done) only if it still holds the value this tick
        read. True = this caller owns the occurrence."""
        values: dict[str, Any] = {"next_run_at": next_run_at}
        if next_run_at is None:
            values["status"] = "done"
        async with self.session() as session:  # type: ignore[attr-defined]
            res = await session.execute(
                update(ScheduledTaskRow)
                .where(ScheduledTaskRow.id == schedule_id)
                .where(ScheduledTaskRow.status == "active")
                .where(ScheduledTaskRow.next_run_at == expected_next_run_at)
                .values(**values)
            )
            return (res.rowcount or 0) == 1

    async def record_scheduled_run(
        self, schedule_id: str, *, outcome: str, task_id: Optional[str], ran_at: datetime,
        series_failures: Optional[int] = None, status: Optional[str] = None,
    ) -> None:
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(ScheduledTaskRow, schedule_id)
            if row is None:
                return
            row.last_outcome = outcome
            row.last_task_id = task_id if task_id is not None else row.last_task_id
            row.last_run_at = ran_at
            if series_failures is not None:
                row.series_failures = series_failures
            if status is not None:
                row.status = status
