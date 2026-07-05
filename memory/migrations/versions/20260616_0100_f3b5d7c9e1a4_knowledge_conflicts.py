"""knowledge conflicts (never-idle convergence)

Revision ID: f3b5d7c9e1a4
Revises: e1a3c5b7d2f4
Create Date: 2026-06-16 01:00:00.000000

Never-Idle Convergence (docs/NEVER_IDLE_CONVERGENCE.md) — the first-class peer
of knowledge_gaps. Creates one table:

  knowledge_conflicts — two stored facts that contradict each other, surfaced
                        by the contradiction-scan generator when its
                        CONTRADICTS discriminator fires over subject-adjacent
                        facts. A gap = "we lack knowledge about X"; a conflict
                        = "we hold two facts about X that can't both be true."
                        Mirrors knowledge_gaps' lifecycle; the deltas are
                        intrinsic to a conflict being about a PAIR (two fact
                        ids + crystal ids as SOFT pointers / no FK — REPLACE
                        deletes facts/crystals; two claim snapshots so the row
                        reads without joins and survives the facts changing;
                        two provenance strings; a pair_key idempotence hash).
                        status/resolution are String (not enum) so a new
                        lifecycle/verb lands without a migration.

IDEMPOTENCE (D4): unique on (customer_id, pair_key) — a re-scan can't write a
duplicate (the shard_events ux_*_idempotent precedent). pair_key folds in a
hash of both claim texts, so a changed claim yields a new pair_key and is
re-evaluated; terminal rows keep their pair_key and are never re-surfaced.

CREATE TABLE (not ALTER) so the SQLite ALTER-ADD-FK quirks don't apply.
Idempotent on upgrade (skip if the table already exists, like the F4/G1-G4
and agent_events migrations). Downgrade drops indexes then the table.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f3b5d7c9e1a4'
down_revision: Union[str, None] = 'e1a3c5b7d2f4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    existing_tables = set(sa.inspect(bind).get_table_names())

    if "knowledge_conflicts" not in existing_tables:
        op.create_table(
            "knowledge_conflicts",
            sa.Column("id", sa.String(length=64), nullable=False),
            sa.Column("customer_id", sa.String(length=64), nullable=False),
            sa.Column("fact_a_id", sa.String(length=64), nullable=False),
            sa.Column("fact_b_id", sa.String(length=64), nullable=False),
            sa.Column("crystal_a_id", sa.String(length=64), nullable=True),
            sa.Column("crystal_b_id", sa.String(length=64), nullable=True),
            sa.Column("subject", sa.String(length=256), nullable=True),
            sa.Column("claim_a", sa.Text(), nullable=False),
            sa.Column("claim_b", sa.Text(), nullable=False),
            sa.Column("provenance_a", sa.Text(), nullable=True),
            sa.Column("provenance_b", sa.Text(), nullable=True),
            sa.Column(
                "detector", sa.String(length=64), nullable=False,
                server_default="contradiction_scan",
            ),
            sa.Column(
                "status", sa.String(length=32), nullable=False,
                server_default="open",
            ),
            sa.Column("resolution", sa.String(length=32), nullable=True),
            sa.Column("pair_key", sa.String(length=128), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "ix_knowledge_conflicts_customer_id",
            "knowledge_conflicts",
            ["customer_id"],
        )
        op.create_index(
            "ix_knowledge_conflicts_customer_status",
            "knowledge_conflicts",
            ["customer_id", "status"],
        )
        op.create_index(
            "ux_knowledge_conflicts_pair_key",
            "knowledge_conflicts",
            ["customer_id", "pair_key"],
            unique=True,
        )


def downgrade() -> None:
    op.drop_index(
        "ux_knowledge_conflicts_pair_key", table_name="knowledge_conflicts"
    )
    op.drop_index(
        "ix_knowledge_conflicts_customer_status",
        table_name="knowledge_conflicts",
    )
    op.drop_index(
        "ix_knowledge_conflicts_customer_id", table_name="knowledge_conflicts"
    )
    op.drop_table("knowledge_conflicts")
