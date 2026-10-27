"""Goal comments and snapshots (2026-10-27; Q135 A, Q137 A).

goal_comments: what the person said after a check-in; the next check-in
reads the ones it has not seen first. goal_snapshots: a goal's plan when
it was written (step 0) and each time a step was accepted (step n); a
rollback to step n restores snapshot n-1 and undoes the goal's ledgered
acts after it.
"""
from alembic import op
import sqlalchemy as sa

revision = "d7f9b1c3e5a7"
down_revision = "c5e7a9b1d3f5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "goal_comments",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("goal_id", sa.String(64), nullable=False),
        sa.Column("customer_id", sa.String(64), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("after_run_id", sa.String(64), nullable=True),
        sa.Column("seen_by_run_id", sa.String(64), nullable=True),
        sa.Column("seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_goal_comments_goal_id", "goal_comments", ["goal_id"])
    op.create_index("ix_goal_comments_customer_id", "goal_comments", ["customer_id"])
    op.create_table(
        "goal_snapshots",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("goal_id", sa.String(64), nullable=False),
        sa.Column("customer_id", sa.String(64), nullable=False),
        sa.Column("step_n", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("plan", sa.JSON(), nullable=False),
        sa.Column("run_id", sa.String(64), nullable=True),
        sa.Column("rows", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_goal_snapshots_goal_id", "goal_snapshots", ["goal_id"])
    op.create_index("ix_goal_snapshots_customer_id", "goal_snapshots", ["customer_id"])


def downgrade() -> None:
    op.drop_index("ix_goal_snapshots_customer_id", table_name="goal_snapshots")
    op.drop_index("ix_goal_snapshots_goal_id", table_name="goal_snapshots")
    op.drop_table("goal_snapshots")
    op.drop_index("ix_goal_comments_customer_id", table_name="goal_comments")
    op.drop_index("ix_goal_comments_goal_id", table_name="goal_comments")
    op.drop_table("goal_comments")
