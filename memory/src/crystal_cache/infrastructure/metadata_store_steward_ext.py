"""The Steward's ledger (Q100 A, 2026-10-23).

Anthony: "We need to let a model have the hands. Basically an agent that
can fully control a crystal account... the agent keeps the mind going
with a simple task of keep improving my crystal bank until my cost runs
out. Then ask for a refill."

Every action the Steward takes lands here with the inverse that undoes
it, so a person can reverse any of it for 24 hours from the Activity
page. Same binding pattern as the other mixins (infrastructure/__init__
setattrs the public methods onto MetadataStore). R9: the steward_actions
SQL lives here and nowhere else.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import structlog
from sqlalchemy import func, select

from .schema import CognitionTaskRow, CrystalRow, StewardActionRow

logger = structlog.get_logger(__name__)

# How long an action can be undone (the plan: "every action on the
# Activity page with a 24-hour undo").
UNDO_WINDOW = timedelta(hours=24)


def _action_dict(r: StewardActionRow) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    created = r.created_at
    if created is not None and created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    expires = (created + UNDO_WINDOW) if created else None
    return {
        "id": r.id,
        "customer_id": r.customer_id,
        "run_id": r.run_id,
        "actor": getattr(r, "actor", None) or "steward",
        "action": r.action,
        "subject_kind": r.subject_kind,
        "subject_id": r.subject_id,
        "label": r.label or "",
        "reason": r.reason or "",
        "inverse": dict(r.inverse or {}),
        "undoable": bool(r.undoable) and r.undone_at is None and expires is not None and now < expires,
        "undone_at": r.undone_at.isoformat() if r.undone_at else None,
        "undo_until": expires.isoformat() if expires else None,
        "created_at": created.isoformat() if created else None,
    }


class StewardExtensionsMixin:
    """steward_actions reads/writes, bound onto MetadataStore."""

    async def open_steward_run(self, customer_id: str, *, payload: dict[str, Any]) -> str:
        """A cognition task row of type "steward", opened RUNNING (never
        pending: the task lane must not claim it as a research goal).
        The session closes it with mark_cognition_task_complete / _failed
        so it shows in Activity like every other run."""
        task_id = f"cog_{uuid.uuid4().hex[:16]}"
        now = datetime.now(timezone.utc)
        async with self.session() as session:  # type: ignore[attr-defined]
            session.add(CognitionTaskRow(
                id=task_id,
                customer_id=customer_id,
                task_type="steward",
                payload=payload,
                priority="background",
                status="running",
                created_at=now,
                started_at=now,
            ))
            await session.commit()
        return task_id

    async def record_steward_action(
        self,
        customer_id: str,
        *,
        run_id: Optional[str],
        action: str,
        subject_kind: str,
        subject_id: str,
        label: str,
        reason: str,
        inverse: Optional[dict[str, Any]],
        undoable: bool = True,
        actor: str = "steward",
    ) -> dict[str, Any]:
        row = StewardActionRow(
            id=f"stw_{uuid.uuid4().hex[:16]}",
            customer_id=customer_id,
            run_id=run_id,
            actor=(actor or "steward")[:64],
            action=action,
            subject_kind=subject_kind,
            subject_id=subject_id,
            label=(label or "")[:256],
            reason=(reason or "")[:4000],
            inverse=inverse or {},
            undoable=undoable,
            created_at=datetime.now(timezone.utc),
        )
        async with self.session() as session:  # type: ignore[attr-defined]
            session.add(row)
            await session.commit()
        return _action_dict(row)

    async def list_steward_actions(
        self, customer_id: str, *, limit: int = 100, since: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(StewardActionRow)
                .where(StewardActionRow.customer_id == customer_id)
                .order_by(StewardActionRow.created_at.desc(), StewardActionRow.id.desc())
                .limit(max(1, limit))
            )
            if since is not None:
                stmt = stmt.where(StewardActionRow.created_at >= since)
            rows = (await session.execute(stmt)).scalars().all()
        return [_action_dict(r) for r in rows]

    async def get_steward_action(self, action_id: str, customer_id: str) -> Optional[dict[str, Any]]:
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(StewardActionRow, action_id)
            if row is None or row.customer_id != customer_id:
                return None
            return _action_dict(row)

    async def mark_steward_action_undone(self, action_id: str, customer_id: str) -> bool:
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(StewardActionRow, action_id)
            if row is None or row.customer_id != customer_id or row.undone_at is not None:
                return False
            row.undone_at = datetime.now(timezone.utc)
            await session.commit()
            return True

    async def count_steward_actions_since(self, customer_id: str, since: datetime) -> int:
        async with self.session() as session:  # type: ignore[attr-defined]
            n = (await session.execute(
                select(func.count())
                .select_from(StewardActionRow)
                .where(StewardActionRow.customer_id == customer_id)
                .where(StewardActionRow.created_at >= since)
            )).scalar_one()
        return int(n or 0)

    async def strip_crystal_diagnostic_tags(
        self, crystal_id: str, customer_id: Optional[str], prefixes: tuple[str, ...],
    ) -> bool:
        """Remove every diagnostic tag starting with one of the prefixes
        (the inverse of append_crystal_diagnostic_tags; the Steward's
        undo of an approval drops assumption_approved_by:*)."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(CrystalRow, crystal_id)
            if row is None:
                return False
            if customer_id is not None and row.customer_id != customer_id:
                return False
            current = list(row.diagnostic_tags or [])
            kept = [t for t in current if not any(t.startswith(p) for p in prefixes)]
            if kept != current:
                row.diagnostic_tags = kept
            await session.commit()
            return True
