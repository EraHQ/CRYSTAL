"""Citation-record primitives — the Growth G1 store surface.

G1 makes the model's answers attributable: it cites its sources (a source is
an injected crystal), and a *cited* crystal — not merely an injected one — is
what proves load-bearing. This mixin persists the per-claim citation record
the proxy's post-response step produces (parse [[cc:N]] → ground → record),
and reads it back for the Inspector and for G4's metering rail.

R9 puts the SQL here, bound onto MetadataStore via the same setattr pattern
as the other extension mixins (see infrastructure/__init__.py). Methods deal
in plain dicts — these are operational rows the proxy/Inspector/G4 format and
branch on, not domain entities (matching the session + agent-task mixins).

The table carries NO uniqueness: a turn may cite one crystal in several
claims, and G4's shard ledger is what dedupes on (interaction, crystal) when
it mints credit. `grounded` gates that credit — only grounded=True citations
are load-bearing; cited-but-ungrounded spans are spurious, kept for telemetry.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

import structlog
from sqlalchemy import func, select

from .schema import CitationRow

logger = structlog.get_logger(__name__)


def _citation_to_dict(row: CitationRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "customer_id": row.customer_id,
        "query_log_id": row.query_log_id,
        "crystal_id": row.crystal_id,
        "crystal_version": row.crystal_version,
        "handle": row.handle,
        "claim_span": row.claim_span,
        "grounding_score": row.grounding_score,
        "grounded": row.grounded,
        "created_at": row.created_at,
    }


class CitationExtensionsMixin:
    """citations CRUD bound onto MetadataStore (Growth G1)."""

    async def record_citations(
        self,
        customer_id: str,
        *,
        query_log_id: Optional[str],
        citations: list[dict[str, Any]],
    ) -> list[str]:
        """Persist parsed + grounding-checked citations for one response turn.

        Each citation dict carries: crystal_id (required), and optionally
        version, handle, claim_span, grounding_score, grounded. Plain inserts
        (one row per cited claim) — G4's shard ledger dedupes on
        (interaction, crystal) when it mints credit, so this raw record allows
        repeats. Empty list is a no-op. Returns the created row ids.
        """
        if not citations:
            return []
        created: list[str] = []
        async with self.session() as session:  # type: ignore[attr-defined]
            for c in citations:
                row_id = f"cite_{uuid.uuid4().hex[:16]}"
                row = CitationRow(
                    id=row_id,
                    customer_id=customer_id,
                    query_log_id=query_log_id,
                    crystal_id=(c.get("crystal_id") or ""),
                    crystal_version=c.get("version"),
                    handle=str(c.get("handle") or ""),
                    claim_span=(c.get("claim_span") or "")[:2000],
                    grounding_score=c.get("grounding_score"),
                    grounded=bool(c.get("grounded", False)),
                )
                session.add(row)
                created.append(row_id)
            await session.commit()
        logger.info(
            "citations.recorded",
            customer_id=customer_id,
            query_log_id=query_log_id,
            count=len(created),
            grounded=sum(1 for c in citations if c.get("grounded")),
        )
        return created

    async def list_citations_for_query(
        self, query_log_id: str
    ) -> list[dict[str, Any]]:
        """All citations recorded for one response turn, oldest first."""
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(
                select(CitationRow)
                .where(CitationRow.query_log_id == query_log_id)
                .order_by(CitationRow.created_at)
            )).scalars().all()
            return [_citation_to_dict(r) for r in rows]

    async def latest_grounded_citation_at(
        self, customer_id: str, crystal_id: str,
    ):
        """Timestamp of the crystal's newest grounded citation, or None.

        The tier-DECAY signal (2026-07-02, ratified 30 days): a whitelist
        crystal whose newest grounded citation is older than the decay
        window drifts back to neutral — trust must stay earned."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(func.max(CitationRow.created_at))
                .where(
                    CitationRow.customer_id == customer_id,
                    CitationRow.crystal_id == crystal_id,
                    CitationRow.grounded.is_(True),
                )
            )
            return (await session.execute(stmt)).scalar()

    async def count_grounded_citations_for_crystal(
        self, customer_id: str, crystal_id: str,
    ) -> int:
        """Grounded-citation count for one crystal (the tier-promotion
        signal — only grounded citations accrue trust, mirroring G4's
        only-grounded-mints-credit rule)."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(func.count())
                .select_from(CitationRow)
                .where(
                    CitationRow.customer_id == customer_id,
                    CitationRow.crystal_id == crystal_id,
                    CitationRow.grounded.is_(True),
                )
            )
            return int((await session.execute(stmt)).scalar() or 0)

    async def list_citations_for_crystal(
        self,
        customer_id: str,
        crystal_id: str,
        *,
        grounded_only: bool = True,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Citations of one crystal for a customer, newest first. Defaults to
        grounded-only — the G4-relevant set (only grounded citations accrue
        credit). Pass grounded_only=False for the full telemetry view."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = select(CitationRow).where(
                CitationRow.customer_id == customer_id,
                CitationRow.crystal_id == crystal_id,
            )
            if grounded_only:
                stmt = stmt.where(CitationRow.grounded.is_(True))
            stmt = stmt.order_by(CitationRow.created_at.desc()).limit(limit)
            rows = (await session.execute(stmt)).scalars().all()
            return [_citation_to_dict(r) for r in rows]
