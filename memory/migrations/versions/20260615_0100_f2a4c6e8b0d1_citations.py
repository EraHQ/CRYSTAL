"""citations

Revision ID: f2a4c6e8b0d1
Revises: e7a2c9d4b1f8
Create Date: 2026-06-15 01:00:00.000000

Growth G1 (trust + the metering rail) — the per-claim citation record.
Creates one table:

  citations — one row per cited claim in a response turn. The proxy's
              post-response step parses [[cc:N]] markers out of the model's
              answer, grounds each against the cited crystal, and records the
              result. `grounded` gates G4 credit (a cited-but-ungrounded span
              is a spurious citation: kept for telemetry, never paid). G4's
              shard ledger dedupes on (interaction, crystal) when it mints
              credit, so this table carries NO uniqueness.

CREATE TABLE (not ALTER), so the SQLite ALTER-ADD-FK quirks don't apply.
Following the F4 precedent, columns are PLAIN (no FK constraints) — SQLite
doesn't enforce FKs and the joins are application-level; the ORM model keeps
the customer_id FK declaration for Postgres + documentation. crystal_id and
query_log_id are soft pointers by design (REPLACE semantics delete crystals;
a FK would dangle).

Idempotent on upgrade: the create is skipped if the table already exists (a
local dev DB may have picked it up via store.init()'s create_all before this
migration ran). Downgrade drops indexes then the table.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f2a4c6e8b0d1'
down_revision: Union[str, None] = 'e7a2c9d4b1f8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    existing_tables = set(sa.inspect(bind).get_table_names())

    if "citations" not in existing_tables:
        op.create_table(
            "citations",
            sa.Column("id", sa.String(length=64), nullable=False),
            sa.Column("customer_id", sa.String(length=64), nullable=False),
            sa.Column("query_log_id", sa.String(length=64), nullable=True),
            sa.Column("crystal_id", sa.String(length=64), nullable=False),
            sa.Column("crystal_version", sa.String(length=64), nullable=True),
            sa.Column(
                "handle", sa.String(length=16), nullable=False, server_default=""
            ),
            sa.Column(
                "claim_span", sa.Text(), nullable=False, server_default=""
            ),
            sa.Column("grounding_score", sa.Float(), nullable=True),
            sa.Column(
                "grounded", sa.Boolean(), nullable=False, server_default=sa.false()
            ),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "ix_citations_customer_created",
            "citations",
            ["customer_id", "created_at"],
        )
        op.create_index(
            "ix_citations_query_log",
            "citations",
            ["query_log_id"],
        )
        op.create_index(
            "ix_citations_crystal",
            "citations",
            ["customer_id", "crystal_id"],
        )


def downgrade() -> None:
    op.drop_index("ix_citations_crystal", table_name="citations")
    op.drop_index("ix_citations_query_log", table_name="citations")
    op.drop_index("ix_citations_customer_created", table_name="citations")
    op.drop_table("citations")
