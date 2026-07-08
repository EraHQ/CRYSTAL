"""cognition_runs — persisted environment snapshots.

S9 (2026-07-08, docs/INSPECTOR_DATA_AUDIT.md F3): the Cognition
Environments surface has been structurally dead since the api/worker
split (in-memory registry, wrong process). Every lifecycle transition
now upserts a snapshot here; the API serves active + recent completed
runs from this table. summary/detail store the exact wire shapes the
endpoints always served (R3).
"""
from alembic import op
import sqlalchemy as sa

revision = "e2f4a6b8c0d1"
down_revision = "d1e3f5a7b9c0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "cognition_runs",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("customer_id", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False,
                  server_default="created"),
        sa.Column("trigger_type", sa.String(length=64), nullable=True),
        sa.Column("goal_title", sa.Text(), nullable=True),
        sa.Column("summary", sa.JSON(), nullable=True),
        sa.Column("detail", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_cognition_runs_customer_id", "cognition_runs", ["customer_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_cognition_runs_customer_id", table_name="cognition_runs")
    op.drop_table("cognition_runs")
