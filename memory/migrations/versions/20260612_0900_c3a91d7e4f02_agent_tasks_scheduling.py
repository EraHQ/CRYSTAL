"""agent_tasks scheduling: run_at, recurrence, series lineage

Revision ID: c3a91d7e4f02
Revises: fbe44d2b9519
Create Date: 2026-06-12

Scheduled + recurring background tasks (design locked 2026-06-12):
run_at NULL = ASAP, else the due time (UTC; parsed local at the CLI
boundary). recur_seconds NULL = one-shot, else fixed-rate wall-clock
recurrence anchored at run_at — the daemon enqueues each next
occurrence as a NEW row (parent_task_id lineage), skipping missed
slots. series_failures carries consecutive-failure count across the
series; the daemon parks a series at the cap instead of recurring.
"""
from alembic import op
import sqlalchemy as sa

revision = "c3a91d7e4f02"
down_revision = "fbe44d2b9519"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("agent_tasks", sa.Column("run_at", sa.DateTime(), nullable=True))
    op.add_column("agent_tasks", sa.Column("recur_seconds", sa.Integer(), nullable=True))
    op.add_column("agent_tasks", sa.Column("parent_task_id", sa.String(length=64), nullable=True))
    op.create_index("ix_agent_tasks_parent_task_id", "agent_tasks", ["parent_task_id"])
    op.add_column(
        "agent_tasks",
        sa.Column("series_failures", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("agent_tasks", "series_failures")
    op.drop_index("ix_agent_tasks_parent_task_id", table_name="agent_tasks")
    op.drop_column("agent_tasks", "parent_task_id")
    op.drop_column("agent_tasks", "recur_seconds")
    op.drop_column("agent_tasks", "run_at")
