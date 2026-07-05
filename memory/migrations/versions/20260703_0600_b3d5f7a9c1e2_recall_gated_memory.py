"""recall_gated + origin on crystals (recall-gated memory, option b).

Revision ID: b3d5f7a9c1e2
Revises: a1c3e5b7d9f2
Create Date: 2026-07-03

Recall-gated memory (2026-07-03): two columns on crystals.

  recall_gated (bool, default false) — the "can this crystal be USED at
    all" bit, orthogonal to quality_tier (which stays a pure epistemic
    signal, never a recall gate). Default false = unchanged behavior for
    every existing row. True = held out of the recall candidate set until
    approved (human or a system_rules promotion rule).

  origin (text, default 'direct') — WHAT created the crystal, distinct from
    source_kind (KIND of evidence). 'direct' = foreground/user ingest (the
    default); 'background_worker' = autonomous task output (born gated).

Both are additive with safe defaults, so the upgrade is a pure add-column
with no data backfill: existing crystals load as ungated, direct-origin —
identical to pre-migration behavior.
"""
from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "b3d5f7a9c1e2"
down_revision: Union[str, None] = "a1c3e5b7d9f2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "crystals",
        sa.Column(
            "recall_gated", sa.Boolean(), nullable=False,
            server_default=sa.text("0"),
        ),
    )
    op.add_column(
        "crystals",
        sa.Column(
            "origin", sa.String(length=32), nullable=False,
            server_default="direct",
        ),
    )
    op.create_index(
        "ix_crystals_recall_gated", "crystals", ["recall_gated"],
    )
    op.create_index(
        "ix_crystals_origin", "crystals", ["origin"],
    )


def downgrade() -> None:
    op.drop_index("ix_crystals_origin", table_name="crystals")
    op.drop_index("ix_crystals_recall_gated", table_name="crystals")
    op.drop_column("crystals", "origin")
    op.drop_column("crystals", "recall_gated")
