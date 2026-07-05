"""web_search_logs

Revision ID: d7f9b1c3e5a8
Revises: c6e8a0b2d4f7
Create Date: 2026-07-02 18:00:00.000000

Launch-prep sweep — the web-search interaction log (the goldmine's raw
side). One row per search the system runs; results stored as
title/url/snippet JSON without extracted content; joins to crystallized
knowledge by URL via crystal provenance (source_kind=web_search_result).

CREATE TABLE (not ALTER), so the SQLite ALTER-ADD-FK quirks don't apply.
Following the F4 precedent, columns are PLAIN (no FK constraints) —
customer_id is a soft pointer. Idempotent on upgrade: the create is
skipped if the table already exists (a local dev DB may have picked it up
via store.init()'s create_all before this migration ran). Downgrade drops
the index then the table.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd7f9b1c3e5a8'
down_revision: Union[str, None] = 'c6e8a0b2d4f7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    existing = sa.inspect(bind).get_table_names()
    if "web_search_logs" in existing:
        return
    op.create_table(
        "web_search_logs",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("customer_id", sa.String(length=64), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("n_results", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("results", sa.JSON(), nullable=True),
        sa.Column("origin", sa.String(length=16), nullable=False, server_default="tool"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_web_search_logs_customer_id", "web_search_logs", ["customer_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_web_search_logs_customer_id", table_name="web_search_logs")
    op.drop_table("web_search_logs")
