"""Promotion provenance extension methods — Foundation F3.

The F3 promotion engine (maintenance/promotion_service.py) merges
operator-private crystals up to the team tier, leaving one survivor and
superseding the rest. At merge it records, durably, WHO contributed each
source crystal and the credit share reserved for them — the forward-
reference to G4's shard ledger (capture-at-merge is cheap; reconstruct-later
is impossible).

Those reads/writes touch the `crystal_contributions` table, which neither
the core store nor the existing mixins cover. Per R9 (SQL only in
metadata_store* files) the SQL lives here, bound onto MetadataStore via the
same setattr pattern as the other extension mixins (see
infrastructure/__init__.py).

The engine otherwise uses only existing store primitives (get_crystal,
upsert_crystal, delete_crystal, list_crystals_for_customer) for the
crystal-level merge, so the only NEW persistence F3 needs is this provenance
pair — hence the mixin is deliberately just two methods.
"""
from __future__ import annotations

import uuid
from typing import Any

import structlog
from sqlalchemy import select

from .schema import CrystalContributionRow

logger = structlog.get_logger(__name__)


class PromotionExtensionsMixin:
    """crystal_contributions CRUD bound onto MetadataStore (Foundation F3)."""

    async def record_promotion_contributions(
        self,
        merged_crystal_id: str,
        contributions: list[dict[str, Any]],
    ) -> int:
        """Persist contributor-provenance + reserved-share rows for a merge.

        Each contribution dict carries:
          source_crystal_id: str            — the contributing original's id
          contributor_operator_id: str|None — the operator who owned it
          share_basis_points: int           — reserved credit share (1/10000)

        One row per contribution (one per source crystal, including the
        survivor's own). Append-only: a merge writes its provenance once;
        there is no update path. Returns the number of rows written.
        """
        written = 0
        async with self.session() as session:  # type: ignore[attr-defined]
            for c in contributions:
                session.add(CrystalContributionRow(
                    id=f"ccontrib_{uuid.uuid4().hex[:16]}",
                    merged_crystal_id=merged_crystal_id,
                    contributor_operator_id=c.get("contributor_operator_id"),
                    source_crystal_id=c["source_crystal_id"],
                    share_basis_points=int(c.get("share_basis_points", 0)),
                ))
                written += 1
        logger.info(
            "promotion.contributions_recorded",
            merged_crystal_id=merged_crystal_id,
            contributors=written,
        )
        return written

    async def list_promotion_contributions(
        self,
        merged_crystal_id: str,
    ) -> list[dict[str, Any]]:
        """Return the provenance rows for a merged crystal as dicts.

        Ordered by share descending then source id, so the largest
        contributor reads first. Dicts (not ORM rows) keep the store's
        "return entities, not rows" contract; the shape mirrors the
        record_promotion_contributions input plus id + created_at.
        """
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(CrystalContributionRow)
                .where(
                    CrystalContributionRow.merged_crystal_id
                    == merged_crystal_id
                )
                .order_by(
                    CrystalContributionRow.share_basis_points.desc(),
                    CrystalContributionRow.source_crystal_id,
                )
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [
                {
                    "id": r.id,
                    "merged_crystal_id": r.merged_crystal_id,
                    "contributor_operator_id": r.contributor_operator_id,
                    "source_crystal_id": r.source_crystal_id,
                    "share_basis_points": r.share_basis_points,
                    "created_at": r.created_at,
                }
                for r in rows
            ]
