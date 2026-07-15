"""fact_ledger — immutable bank history (Q6B).

Revision ID: d4e6f8a0b2c3
Revises: c3d5e7f9a1b2
Create Date: 2026-07-15
"""
from alembic import op
import sqlalchemy as sa

revision = "d4e6f8a0b2c3"
down_revision = "c3d5e7f9a1b2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "fact_ledger",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("customer_id", sa.String(length=64), nullable=False),
        sa.Column("crystal_id", sa.String(length=64), nullable=False),
        sa.Column("fact_id", sa.String(length=64), nullable=False),
        sa.Column("op", sa.String(length=32), nullable=False),
        sa.Column("actor", sa.String(length=128), nullable=False,
                  server_default="operator"),
        sa.Column("before_prompt", sa.Text(), nullable=True),
        sa.Column("before_text", sa.Text(), nullable=True),
        sa.Column("after_text", sa.Text(), nullable=True),
        sa.Column("successor_fact_id", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    for col in ("customer_id", "crystal_id", "fact_id", "created_at"):
        op.create_index(f"ix_fact_ledger_{col}", "fact_ledger", [col])


def downgrade() -> None:
    for col in ("created_at", "fact_id", "crystal_id", "customer_id"):
        op.drop_index(f"ix_fact_ledger_{col}", table_name="fact_ledger")
    op.drop_table("fact_ledger")
