"""customers.assumptions_explore (Assumptions funnel F3, Q5=A)

Revision ID: a9b1c3d5e7f0
Revises: f6b8c0d2e4a7
Create Date: 2026-08-07

Per-customer explore toggle for the assumptions funnel. NULL = the
deployment default (explore ON — "assumptions on everything"); False
withholds STRUCTURAL-tier funnel edges (key_adjacent / vector_similar)
at verdict-spend time, so only demand-evidenced pairs cost model calls.
Mirrors the routing_context_window nullable-override shape.
"""
from alembic import op
import sqlalchemy as sa

revision = "a9b1c3d5e7f0"
down_revision = "f6b8c0d2e4a7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "customers",
        sa.Column("assumptions_explore", sa.Boolean(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("customers", "assumptions_explore")
