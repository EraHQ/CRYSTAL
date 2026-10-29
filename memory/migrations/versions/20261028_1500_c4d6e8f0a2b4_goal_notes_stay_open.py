"""Goal notes stay open until handled (2026-10-28; Q153 A).

goal_comments gains status (open | in_progress | cant | done | dismissed),
always (a standing instruction), report (the last check-in's word on it),
reported_by_run_id, reported_at and item_id (the "needs you" item it
answers). Every existing note starts open, so the next check-in sees what
the person already said and reports on it.
"""
from alembic import op
import sqlalchemy as sa

revision = "c4d6e8f0a2b4"
down_revision = "b3c5d7e9f1a3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("goal_comments", sa.Column("status", sa.String(16), nullable=False, server_default="open"))
    op.add_column("goal_comments", sa.Column("always", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column("goal_comments", sa.Column("report", sa.Text(), nullable=False, server_default=""))
    op.add_column("goal_comments", sa.Column("reported_by_run_id", sa.String(64), nullable=True))
    op.add_column("goal_comments", sa.Column("reported_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("goal_comments", sa.Column("item_id", sa.String(64), nullable=True))


def downgrade() -> None:
    for c in ("item_id", "reported_at", "reported_by_run_id", "report", "always", "status"):
        op.drop_column("goal_comments", c)
