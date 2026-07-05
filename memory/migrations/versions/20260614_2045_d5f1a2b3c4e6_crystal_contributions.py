"""crystal_contributions

Revision ID: d5f1a2b3c4e6
Revises: b2e7c9a4f1d3
Create Date: 2026-06-14 20:45:00.000000

Foundation F3 (promotion engine) — contributor provenance + reserved credit
shares. Creates `crystal_contributions`: one row per (merged team crystal,
source crystal) recording WHO contributed (contributor_operator_id) and the
reserved credit share (share_basis_points, summing to 10000 per merged
crystal). The forward-reference to G4's shard ledger — captured at merge
because reconstruct-later is impossible.

CREATE TABLE (not ALTER), so the SQLite ALTER-ADD-FK quirks that the F2
posix-permissions migration worked around do not apply here. Following that
migration's precedent, the columns are PLAIN (no FK constraints): SQLite
doesn't enforce FKs anyway, and `source_crystal_id` deliberately has no FK
because superseded non-survivor crystals are deleted at merge (a constraint
there would dangle). The ORM model (CrystalContributionRow) keeps the FK
declarations on merged_crystal_id / contributor_operator_id for Postgres +
documentation.

Idempotent on upgrade: skip the create if the table already exists — a
local dev DB may have picked it up via store.init()'s create_all before
this migration ran. Downgrade drops the indexes then the table.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd5f1a2b3c4e6'
down_revision: Union[str, None] = 'b2e7c9a4f1d3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    existing_tables = set(sa.inspect(bind).get_table_names())
    if "crystal_contributions" in existing_tables:
        return

    op.create_table(
        "crystal_contributions",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("merged_crystal_id", sa.String(length=64), nullable=False),
        sa.Column(
            "contributor_operator_id", sa.String(length=64), nullable=True
        ),
        sa.Column("source_crystal_id", sa.String(length=64), nullable=False),
        sa.Column(
            "share_basis_points",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=True
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_crystal_contributions_merged",
        "crystal_contributions",
        ["merged_crystal_id"],
    )
    op.create_index(
        "ix_crystal_contributions_contributor",
        "crystal_contributions",
        ["contributor_operator_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_crystal_contributions_contributor",
        table_name="crystal_contributions",
    )
    op.drop_index(
        "ix_crystal_contributions_merged",
        table_name="crystal_contributions",
    )
    op.drop_table("crystal_contributions")
