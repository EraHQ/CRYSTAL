"""Curation activity feed — C2 Q3=A (2026-08-08).

The watch-drawer pattern (`metadata_store_schema_ext.record_watch_event`)
generalized to self-curation: an append-only witness stream of every
state transition the system performs on its own knowledge — assumptions
written / approved / invalidated / deleted, gaps filled / reopened — so
none of it is silent. Vocabulary is string-backed; the ONE home for
emitters is the layer that performs the transition (the assumption ext's
write primitive, delete_crystal's capture path, the approve endpoint).

`record_curation_event` raises normally (tests and server callers see
real errors); every production call site wraps it best-effort, because
observability must never break the operation it witnesses — the same
contract as `agent_events.record_event`.

Bound onto MetadataStore via infrastructure/__init__._bind_mixin_methods
(R9 puts the SQL here; the binding pattern matches every other ext).

This table is also the feed substrate for the planned activity-drawer
UI (the unprompted-chat panel): event_type + label + payload is exactly
what its narrator consumes.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import structlog
from sqlalchemy import select

from .schema import CurationEventRow

logger = structlog.get_logger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class CurationEventsMixin:
    """curation_events CRUD, bound onto MetadataStore."""

    async def record_curation_event(
        self,
        customer_id: str,
        *,
        event_type: str,
        label: str = "",
        payload: Optional[dict[str, Any]] = None,
        subject_id: Optional[str] = None,
    ) -> None:
        """Append to the self-curation witness feed. subject_id is a
        soft pointer (crystal or gap id) — never an FK; events outlive
        their subjects."""
        row = CurationEventRow(
            id=f"cue_{uuid.uuid4().hex[:16]}",
            customer_id=customer_id,
            subject_id=subject_id,
            event_type=event_type,
            label=label,
            payload=payload,
            created_at=_utcnow(),
        )
        async with self.session() as session:  # type: ignore[attr-defined]
            session.add(row)
            await session.commit()

    async def list_curation_events(
        self, customer_id: str, *, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Newest-first feed for the Activity surface."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(CurationEventRow)
                .where(CurationEventRow.customer_id == customer_id)
                .order_by(CurationEventRow.created_at.desc())
                .limit(limit)
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [
                {
                    "id": r.id,
                    "subject_id": r.subject_id,
                    "event_type": r.event_type,
                    "label": r.label,
                    "payload": r.payload,
                    "created_at": r.created_at,
                }
                for r in rows
            ]
