"""run_critiques — operator critiques pinned to run anatomy (Q2B).

Revision ID: c3d5e7f9a1b2
Revises: b2c4d6e8f0a1
Create Date: 2026-07-15
"""
from alembic import op
import sqlalchemy as sa

revision = "c3d5e7f9a1b2"
down_revision = "b2c4d6e8f0a1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "run_critiques",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("customer_id", sa.String(length=64), nullable=False),
        sa.Column("trigger_id", sa.String(length=128), nullable=True),
        sa.Column("target_path", sa.String(length=256), nullable=False,
                  server_default="run"),
        sa.Column("author", sa.String(length=128), nullable=False,
                  server_default="operator"),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False,
                  server_default="open"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_run_critiques_run_id", "run_critiques", ["run_id"])
    op.create_index("ix_run_critiques_customer_id", "run_critiques",
                    ["customer_id"])
    op.create_index("ix_run_critiques_trigger_id", "run_critiques",
                    ["trigger_id"])


def downgrade() -> None:
    op.drop_index("ix_run_critiques_trigger_id", table_name="run_critiques")
    op.drop_index("ix_run_critiques_customer_id", table_name="run_critiques")
    op.drop_index("ix_run_critiques_run_id", table_name="run_critiques")
    op.drop_table("run_critiques")
