"""Entity registry CRUD — bound onto MetadataStore (mixin pattern D12).

Entities layer (design gate 2026-07-22, SESSION_HANDOFF 0c): the
registry that makes people and orgs DETERMINISTICALLY resolvable to
their dedicated crystals. Resolution is registry name/alias lookup —
word-boundary, case-insensitive, NEVER vector similarity — because
referencing the wrong person is a category error, not a ranking miss.
Everything else about an entity's crystal is ordinary machinery.

Binding: `_bind_mixin_methods(MetadataStore, EntityExtensionsMixin)` in
`infrastructure/__init__.py`, alongside the other ext mixins.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select

from ..models.crystal import Crystal
from ..models.entity import Entity
from .schema import EntityRow


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _entity_from_row(row: EntityRow) -> Entity:
    return Entity(
        id=row.id,
        customer_id=row.customer_id,
        kind=row.kind,
        display_name=row.display_name,
        aliases=list(row.aliases or []),
        crystal_id=row.crystal_id,
        operator_id=row.operator_id,
        created_at=row.created_at,
    )


class EntityExtensionsMixin:
    """`entities` registry CRUD + deterministic resolution."""

    async def create_entity(
        self,
        *,
        customer_id: str,
        display_name: str,
        kind: str = "person",
        aliases: Optional[list[str]] = None,
        operator_id: Optional[str] = None,
    ) -> Entity:
        """Insert an entity row. Crystal creation is separate (lazy) —
        see ensure_entity_crystal — so read paths stay side-effect free.
        """
        row = EntityRow(
            id=f"ent_{uuid.uuid4().hex[:16]}",
            customer_id=customer_id,
            kind=kind,
            display_name=display_name,
            aliases=list(aliases or []),
            crystal_id=None,
            operator_id=operator_id,
            created_at=_utcnow(),
        )
        async with self.session() as session:
            session.add(row)
            await session.commit()
        return _entity_from_row(row)

    async def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        async with self.session() as session:
            row = await session.get(EntityRow, entity_id)
            return _entity_from_row(row) if row else None

    async def get_entity_for_operator(
        self, operator_id: str
    ) -> Optional[Entity]:
        """The entity row whose operator_id links F1's operator — the
        "who am I talking to" lookup. At most one by construction
        (ensure_entity_for_operator is the only creator of linked rows).
        """
        async with self.session() as session:
            stmt = select(EntityRow).where(
                EntityRow.operator_id == operator_id
            )
            row = (await session.execute(stmt)).scalars().first()
            return _entity_from_row(row) if row else None

    async def list_entities_for_customer(
        self, customer_id: str
    ) -> list[Entity]:
        async with self.session() as session:
            stmt = (
                select(EntityRow)
                .where(EntityRow.customer_id == customer_id)
                .order_by(EntityRow.created_at)
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [_entity_from_row(r) for r in rows]

    async def resolve_entities_in_text(
        self, customer_id: str, text: str, *, limit: int = 3
    ) -> list[Entity]:
        """Deterministic mention detection: which of this customer's
        entities appear in `text` by display_name or alias?

        Word-boundary, case-insensitive, EXACT names only — no fuzzy
        matching by design (the gate's scope guard; fuzzy merging lives
        on the idle-scan board). First-mention order, capped at `limit`.
        """
        entities = await self.list_entities_for_customer(customer_id)
        hits: list[tuple[int, Entity]] = []
        for entity in entities:
            names = [entity.display_name] + list(entity.aliases)
            best: Optional[int] = None
            for name in names:
                if not name:
                    continue
                m = re.search(
                    r"\b" + re.escape(name) + r"\b", text, re.IGNORECASE
                )
                if m and (best is None or m.start() < best):
                    best = m.start()
            if best is not None:
                hits.append((best, entity))
        hits.sort(key=lambda pair: pair[0])
        return [entity for _, entity in hits[:limit]]

    async def set_entity_crystal(
        self, entity_id: str, crystal_id: str
    ) -> None:
        async with self.session() as session:
            row = await session.get(EntityRow, entity_id)
            if row is None:
                raise ValueError(f"entity {entity_id!r} not found")
            row.crystal_id = crystal_id
            await session.commit()

    async def ensure_entity_crystal(self, entity_id: str) -> str:
        """The entity's dedicated crystal id, creating the crystal on
        first need (lazy by design — Q5A).

        The crystal is ordinary in every way except identity: canonical
        source_uri = entity://{entity_id} (Gate D's source-identity
        machinery), summary_text = the display name, and — per D4-A's
        born-quarantine-unless-vouched posture — born NEUTRAL, because
        an entity crystal is created deliberately by the operator's own
        hand or the agent's provenance-gated write tool, which is a
        vouching event. Facts inside carry their own provenance split
        (Q4): operator-stated vs agent-inferred.
        """
        entity = await self.get_entity_by_id(entity_id)
        if entity is None:
            raise ValueError(f"entity {entity_id!r} not found")
        if entity.crystal_id:
            return entity.crystal_id
        crystal_id = f"cry_{uuid.uuid4().hex[:16]}"
        crystal = Crystal(
            id=crystal_id,
            customer_id=entity.customer_id,
            summary_vector=[],
            summary_text=f"{entity.display_name} ({entity.kind})",
            build_method="content_chunk",
            source_kind="model_reasoning",
            quality_tier="neutral",
            source_uri=f"entity://{entity.id}",
            owner_operator_id=entity.operator_id,
        )
        await self.upsert_crystal(crystal)
        await self.set_entity_crystal(entity.id, crystal_id)
        return crystal_id

    async def ensure_entity_for_operator(
        self,
        operator_id: str,
        *,
        customer_id: str,
        display_name: str,
        aliases: Optional[list[str]] = None,
    ) -> Entity:
        """Get-or-create the operator's own entity row (idempotent).

        The seeding path for the "who am I talking to" case: called by
        demo/ops seeding and (slice B) by entity_memory_write when the
        subject is the operator. Does NOT create the crystal — that
        stays lazy via ensure_entity_crystal.
        """
        existing = await self.get_entity_for_operator(operator_id)
        if existing is not None:
            return existing
        return await self.create_entity(
            customer_id=customer_id,
            display_name=display_name,
            kind="person",
            aliases=aliases,
            operator_id=operator_id,
        )
