"""Source-schema registry + watch-event feed — Gate G slice 1.

The C5 machinery (Q9-A, ratified 2026-07-16): one inference call per
NEW JSON shape produces a mapping spec; the spec lives here; every
future record of that shape applies it mechanically with zero LLM.
"One human judgment per shape of data, ever."

G-Q2=A: the `status` column on source_schemas IS the review queue —
proposed rows are the Inspector's review surface; approval flips
status and releases every parked document of that shape in one update
(G-Q3=A parking via document_uploads.source_schema_hash +
status='awaiting_schema').

Also home to the source_watch_events feed (G-Q4=A): the watcher's
durable activity stream, upgrading the derived drawer feed with
events the derivation structurally cannot show (retires, cycles).

Binding: `_bind_mixin_methods(MetadataStore, SourceSchemaExtensionsMixin)`
in `infrastructure/__init__.py`.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select, update

from ..models.source_schema import SourceSchema
from .schema import DocumentUploadRow, SourceSchemaRow, SourceWatchEventRow

# G-Q3=A: the parking status. Documents whose JSON shape has no
# approved mapping wait here; the crystallization worker never claims
# them. Approval releases to 'pending'; rejection parks terminally as
# 'schema_rejected' with the reason visible in the pipeline UI.
STATUS_AWAITING_SCHEMA = "awaiting_schema"
STATUS_SCHEMA_REJECTED = "schema_rejected"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _schema_from_row(row: SourceSchemaRow) -> SourceSchema:
    return SourceSchema(
        id=row.id,
        customer_id=row.customer_id,
        schema_hash=row.schema_hash,
        mapping=dict(row.mapping or {}),
        status=row.status,
        sample=list(row.sample or []),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class SourceSchemaExtensionsMixin:
    """`source_schemas` registry + `source_watch_events` feed."""

    # ------------------------------------------------------------------
    # Schema registry
    # ------------------------------------------------------------------

    async def get_source_schema(
        self, customer_id: str, schema_hash: str
    ) -> Optional[SourceSchema]:
        async with self.session() as session:
            stmt = select(SourceSchemaRow).where(
                SourceSchemaRow.customer_id == customer_id,
                SourceSchemaRow.schema_hash == schema_hash,
            )
            row = (await session.execute(stmt)).scalars().first()
            return _schema_from_row(row) if row else None

    async def get_source_schema_by_id(
        self, schema_id: str, customer_id: str
    ) -> Optional[SourceSchema]:
        """Tenancy-checked fetch for the admin surface (G3): a schema id
        from another tenant is a 404, same posture as get_source_watch."""
        async with self.session() as session:
            row = await session.get(SourceSchemaRow, schema_id)
            if row is None or row.customer_id != customer_id:
                return None
            return _schema_from_row(row)

    async def parked_counts_by_schema(
        self, customer_id: str
    ) -> dict[str, int]:
        """How many documents wait on each shape (G3 card badge +
        approve button label): {schema_hash: awaiting count}."""
        from sqlalchemy import func

        async with self.session() as session:
            rows = (await session.execute(
                select(
                    DocumentUploadRow.source_schema_hash,
                    func.count(DocumentUploadRow.id),
                )
                .where(
                    DocumentUploadRow.customer_id == customer_id,
                    DocumentUploadRow.status == STATUS_AWAITING_SCHEMA,
                    DocumentUploadRow.source_schema_hash.is_not(None),
                )
                .group_by(DocumentUploadRow.source_schema_hash)
            )).all()
            return {h: n for h, n in rows}

    async def label_for_schema_hash(
        self, customer_id: str, schema_hash: str
    ) -> Optional[str]:
        """A human handle for a shape: the newest upload carrying its
        hash (parked or released — released docs keep the column)."""
        async with self.session() as session:
            row = (await session.execute(
                select(DocumentUploadRow.label)
                .where(
                    DocumentUploadRow.customer_id == customer_id,
                    DocumentUploadRow.source_schema_hash == schema_hash,
                )
                .order_by(DocumentUploadRow.created_at.desc())
                .limit(1)
            )).scalars().first()
            return row

    async def create_source_schema_proposal(
        self,
        *,
        customer_id: str,
        schema_hash: str,
        mapping: dict[str, Any],
        sample: list[Any],
    ) -> SourceSchema:
        """First-contact registration (idempotent): if the shape is
        already known — any status — return the existing row rather
        than re-proposing. One judgment per shape, ever, includes not
        asking twice."""
        existing = await self.get_source_schema(customer_id, schema_hash)
        if existing is not None:
            return existing
        now = _utcnow()
        row = SourceSchemaRow(
            id=f"schema_{uuid.uuid4().hex[:16]}",
            customer_id=customer_id,
            schema_hash=schema_hash,
            mapping=mapping,
            status="proposed",
            sample=sample,
            created_at=now,
            updated_at=now,
        )
        async with self.session() as session:
            session.add(row)
            await session.commit()
        return _schema_from_row(row)

    async def list_source_schemas(
        self, customer_id: str, *, status: Optional[str] = None
    ) -> list[SourceSchema]:
        async with self.session() as session:
            stmt = select(SourceSchemaRow).where(
                SourceSchemaRow.customer_id == customer_id
            )
            if status:
                stmt = stmt.where(SourceSchemaRow.status == status)
            stmt = stmt.order_by(SourceSchemaRow.created_at)
            rows = (await session.execute(stmt)).scalars().all()
            return [_schema_from_row(r) for r in rows]

    async def approve_source_schema(self, schema_id: str) -> int:
        """Approve the mapping and release every parked document of
        this shape back to 'pending' in one update. Returns the count
        released — the number the approval card showed as waiting."""
        async with self.session() as session:
            row = await session.get(SourceSchemaRow, schema_id)
            if row is None:
                raise ValueError(f"source schema {schema_id!r} not found")
            row.status = "approved"
            row.updated_at = _utcnow()
            released = await session.execute(
                update(DocumentUploadRow)
                .where(
                    DocumentUploadRow.customer_id == row.customer_id,
                    DocumentUploadRow.source_schema_hash == row.schema_hash,
                    DocumentUploadRow.status == STATUS_AWAITING_SCHEMA,
                )
                .values(status="pending")
            )
            await session.commit()
            return released.rowcount or 0

    async def reject_source_schema(self, schema_id: str) -> int:
        """Reject the shape; parked documents park terminally with a
        status the pipeline UI can explain. Re-arrivals of the same
        shape hit the existing rejected row and park directly — no
        re-proposal (idempotent first-contact)."""
        async with self.session() as session:
            row = await session.get(SourceSchemaRow, schema_id)
            if row is None:
                raise ValueError(f"source schema {schema_id!r} not found")
            row.status = "rejected"
            row.updated_at = _utcnow()
            parked = await session.execute(
                update(DocumentUploadRow)
                .where(
                    DocumentUploadRow.customer_id == row.customer_id,
                    DocumentUploadRow.source_schema_hash == row.schema_hash,
                    DocumentUploadRow.status == STATUS_AWAITING_SCHEMA,
                )
                .values(status=STATUS_SCHEMA_REJECTED)
            )
            await session.commit()
            return parked.rowcount or 0

    async def update_source_schema_mapping(
        self, schema_id: str, mapping: dict[str, Any]
    ) -> None:
        """Edit-forward (C5): the new mapping applies to future
        arrivals; staleness/re-ingest semantics ride G2 with the
        pipeline. No status change — an approved mapping stays
        approved through edits."""
        async with self.session() as session:
            row = await session.get(SourceSchemaRow, schema_id)
            if row is None:
                raise ValueError(f"source schema {schema_id!r} not found")
            row.mapping = mapping
            row.updated_at = _utcnow()
            await session.commit()

    async def park_document_for_schema(
        self, document_id: str, schema_hash: str, *, terminal: bool = False,
    ) -> None:
        """G-Q3=A: park an upload against the shape it waits on.
        terminal=True parks as schema_rejected (arrivals AFTER the
        shape's rejection — no waiting, the verdict already exists)."""
        async with self.session() as session:
            await session.execute(
                update(DocumentUploadRow)
                .where(DocumentUploadRow.id == document_id)
                .values(
                    status=(
                        STATUS_SCHEMA_REJECTED if terminal
                        else STATUS_AWAITING_SCHEMA
                    ),
                    source_schema_hash=schema_hash,
                )
            )
            await session.commit()

    # ------------------------------------------------------------------
    # Watch events (G-Q4=A)
    # ------------------------------------------------------------------

    async def record_watch_event(
        self,
        watch_id: str,
        *,
        customer_id: str,
        event_type: str,
        label: str = "",
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        """Append to the watcher's durable activity feed. Vocabulary is
        string-backed and lives with the sync worker (one home)."""
        row = SourceWatchEventRow(
            id=f"swe_{uuid.uuid4().hex[:16]}",
            customer_id=customer_id,
            watch_id=watch_id,
            event_type=event_type,
            label=label,
            payload=payload,
            created_at=_utcnow(),
        )
        async with self.session() as session:
            session.add(row)
            await session.commit()

    async def list_watch_events(
        self, watch_id: str, *, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Newest-first feed for the Activity drawer."""
        async with self.session() as session:
            stmt = (
                select(SourceWatchEventRow)
                .where(SourceWatchEventRow.watch_id == watch_id)
                .order_by(SourceWatchEventRow.created_at.desc())
                .limit(limit)
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [
                {
                    "id": r.id,
                    "watch_id": r.watch_id,
                    "event_type": r.event_type,
                    "label": r.label,
                    "payload": r.payload,
                    "created_at": r.created_at,
                }
                for r in rows
            ]
