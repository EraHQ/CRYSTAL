"""task_keys table (Phase 3 G3: task-scoped box credentials)

Revision ID: e6a8b0c2d4f5
Revises: d5f7a9b1c3e4
Create Date: 2026-07-03

One key per disposable task: hash at rest, tenant-bound, budgeted,
expiring, revocable. The box's only credential (ratified G3).
"""
from alembic import op
import sqlalchemy as sa

revision = "e6a8b0c2d4f5"
down_revision = "d5f7a9b1c3e4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "task_keys",
        sa.Column("task_id", sa.String(), primary_key=True),
        sa.Column("key_hash", sa.String(), nullable=False, unique=True),
        sa.Column(
            "customer_id", sa.String(),
            sa.ForeignKey("customers.id"), nullable=False,
        ),
        sa.Column("budget_micro_usd", sa.Integer(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_task_keys_key_hash", "task_keys", ["key_hash"])


def downgrade() -> None:
    op.drop_index("ix_task_keys_key_hash", table_name="task_keys")
    op.drop_table("task_keys")
