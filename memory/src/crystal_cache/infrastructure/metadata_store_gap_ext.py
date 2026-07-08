"""Knowledge-gap store reads — Never-Idle Convergence (peer of conflicts).

The READ half for `knowledge_gaps`, mirroring metadata_store_conflict_ext.py —
KnowledgeGap is the first-class peer of KnowledgeConflict (models/knowledge_gap.py).
Gap rows are WRITTEN by several producers (the LLM crystal_pull miss path, the
navigation-router miss, the idle gap-discovery scan); this mixin adds the
dedicated per-customer READ that the `memory_gaps` MCP tool and the
`knowledge_gaps` agent tool consume.

The unified backlog (metadata_store_backlog_ext.list_backlog) also reads gaps,
but normalizes them to {kind,id,subject,...} and DROPS the `missing` text — so
this reader returns the full KnowledgeGap, letting callers see WHAT is missing
(the point of a gaps surface), not just that something is.

Same binding pattern as ConflictExtensionsMixin: this mixin is NOT in
MetadataStore's MRO — infrastructure/__init__.py iterates its public methods at
import time and setattrs each onto MetadataStore via _bind_mixin_methods.
`self.session()` inside a bound method resolves to MetadataStore.session by
normal attribute lookup on the bound callable.

Read-only: no gap-writing or gap-resolution mutators live here. The producers
own writes; a later curation gate owns fill/close transitions.
"""
from __future__ import annotations

from typing import Optional

import structlog
from sqlalchemy import func, select

from ..models import KnowledgeGap
from .schema import KnowledgeGapRow

logger = structlog.get_logger(__name__)


def _knowledge_gap_from_row(row: "KnowledgeGapRow") -> KnowledgeGap:
    """Map a KnowledgeGapRow to the KnowledgeGap domain model.

    Explicit field mapping (mirrors _knowledge_conflict_from_row) rather than
    pydantic from_attributes, so a schema column rename surfaces here loudly
    instead of silently. The three least-load-bearing columns (source,
    filled_by_crystal_id, resolved_at) are read defensively so an older row
    shape can't break a read.
    """
    return KnowledgeGap(
        id=row.id,
        customer_id=row.customer_id,
        domain=row.domain,
        subject=row.subject,
        missing=row.missing,
        full_key=getattr(row, "full_key", None),
        triggering_query=getattr(row, "triggering_query", None),
        priority=row.priority,
        status=row.status,
        source=getattr(row, "source", "llm_observation"),
        filled_by_crystal_id=getattr(row, "filled_by_crystal_id", None),
        created_at=row.created_at,
        resolved_at=getattr(row, "resolved_at", None),
    )


class GapExtensionsMixin:
    """knowledge_gaps reads, bound onto MetadataStore."""

    async def list_knowledge_gaps(
        self,
        customer_id: str,
        *,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[KnowledgeGap]:
        """Paginated per-customer gap list, newest first, optional status
        filter ('open' / 'filled' / 'closed'). Mirrors
        list_knowledge_conflicts."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(KnowledgeGapRow)
                .where(KnowledgeGapRow.customer_id == customer_id)
                .order_by(KnowledgeGapRow.created_at.desc())
                .limit(limit)
            )
            if status is not None:
                stmt = stmt.where(KnowledgeGapRow.status == status)
            result = await session.execute(stmt)
            return [_knowledge_gap_from_row(r) for r in result.scalars().all()]

    async def count_knowledge_gaps(
        self, customer_id: str, *, status: Optional[str] = "open"
    ) -> int:
        """Count gaps for a customer (default: open only). Cheap signal for a
        summary / worth gate without paging rows."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(func.count(KnowledgeGapRow.id))
                .where(KnowledgeGapRow.customer_id == customer_id)
            )
            if status is not None:
                stmt = stmt.where(KnowledgeGapRow.status == status)
            return int((await session.execute(stmt)).scalar_one() or 0)
