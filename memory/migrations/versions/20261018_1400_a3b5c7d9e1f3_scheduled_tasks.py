"""Scheduled runs (launch feature, 2026-10-18): the scheduled_tasks table.

Revision ID: a3b5c7d9e1f3
Revises: f2c4d6e8a0b2
Create Date: 2026-10-18 14:00:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "a3b5c7d9e1f3"
down_revision = "f2c4d6e8a0b2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "scheduled_tasks",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("customer_id", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("cadence", sa.String(length=16), nullable=False),
        sa.Column("at_minute", sa.Integer(), nullable=True),
        sa.Column("weekday", sa.Integer(), nullable=True),
        sa.Column("every_hours", sa.Integer(), nullable=True),
        sa.Column("run_once_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("tz", sa.String(length=64), nullable=False, server_default="UTC"),
        sa.Column("next_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="active"),
        sa.Column("series_failures", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_task_id", sa.String(length=64), nullable=True),
        sa.Column("last_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_outcome", sa.String(length=32), nullable=True),
        sa.Column("last_result_summary", sa.Text(), nullable=True),
        sa.Column("last_crystal_id", sa.String(length=64), nullable=True),
        sa.Column("deliver_url", sa.Text(), nullable=True),
        sa.Column("deliver_secret", sa.String(length=80), nullable=True),
        sa.Column("last_delivery", sa.String(length=120), nullable=True),
        sa.Column("last_delivery_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by_operator_id", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_scheduled_tasks_customer_id", "scheduled_tasks", ["customer_id"])
    op.create_index("ix_scheduled_tasks_next_run_at", "scheduled_tasks", ["next_run_at"])
    op.create_index("ix_scheduled_tasks_status", "scheduled_tasks", ["status"])


def downgrade() -> None:
    op.drop_index("ix_scheduled_tasks_status", table_name="scheduled_tasks")
    op.drop_index("ix_scheduled_tasks_next_run_at", table_name="scheduled_tasks")
    op.drop_index("ix_scheduled_tasks_customer_id", table_name="scheduled_tasks")
    op.drop_table("scheduled_tasks")
