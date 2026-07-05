"""llm calls (cost accounting)

Revision ID: c9e3a5b7d1f2
Revises: b8d2f4a6c0e1
Create Date: 2026-06-15 03:00:00.000000

Growth G3 (cost accounting + budgets) — the per-call cost ledger. Creates one
table:

  llm_calls — one row per model invocation, emitted by the single
              record_llm_call() choke point. Cost is computed from a per-model
              price table in config (prices move → externalized) and stored as
              INTEGER micro-USD (1e-6 USD; money is never a float).
              Attribution: session + parent_session (rollup) + team + operator
              + origin. Views are GROUP BYs (all-time / daily / weekly per
              agent / operator / team); average = per-agent (D6). Budgets read
              these aggregates and auto-pause via the G2 channel.

CREATE TABLE (not ALTER) so the SQLite ALTER-ADD-FK quirks don't apply.
Following the F4/G1 precedent, columns are PLAIN (no FK constraints) — the ORM
model keeps the customer_id FK for Postgres + docs; session_id /
parent_session_id / operator_id are soft pointers.

Idempotent on upgrade (skip if the table already exists). Downgrade drops
indexes then the table.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c9e3a5b7d1f2'
down_revision: Union[str, None] = 'b8d2f4a6c0e1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    existing_tables = set(sa.inspect(bind).get_table_names())

    if "llm_calls" not in existing_tables:
        op.create_table(
            "llm_calls",
            sa.Column("id", sa.String(length=64), nullable=False),
            sa.Column("customer_id", sa.String(length=64), nullable=False),
            sa.Column("session_id", sa.String(length=64), nullable=True),
            sa.Column(
                "parent_session_id", sa.String(length=64), nullable=True
            ),
            sa.Column("operator_id", sa.String(length=64), nullable=True),
            sa.Column(
                "origin",
                sa.String(length=32),
                nullable=False,
                server_default="interactive",
            ),
            sa.Column("model", sa.String(length=128), nullable=False),
            sa.Column(
                "input_tokens", sa.Integer(), nullable=False, server_default="0"
            ),
            sa.Column(
                "output_tokens", sa.Integer(), nullable=False, server_default="0"
            ),
            sa.Column(
                "cache_creation_tokens",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
            sa.Column(
                "cache_read_tokens",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
            sa.Column(
                "computed_cost_micro_usd",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "ix_llm_calls_team_created",
            "llm_calls",
            ["customer_id", "created_at"],
        )
        op.create_index(
            "ix_llm_calls_session",
            "llm_calls",
            ["session_id"],
        )
        op.create_index(
            "ix_llm_calls_operator_created",
            "llm_calls",
            ["operator_id", "created_at"],
        )


def downgrade() -> None:
    op.drop_index("ix_llm_calls_operator_created", table_name="llm_calls")
    op.drop_index("ix_llm_calls_session", table_name="llm_calls")
    op.drop_index("ix_llm_calls_team_created", table_name="llm_calls")
    op.drop_table("llm_calls")
