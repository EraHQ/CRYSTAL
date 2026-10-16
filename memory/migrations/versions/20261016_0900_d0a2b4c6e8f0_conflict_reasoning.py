"""knowledge_conflicts.reasoning + recommendation (Console Refine PR 7b, 2026-10-16).

Why the scanner thinks two claims contradict, and what it would do about
it ({resolution, loser, reason}), written by a second small call that
runs only for CONTRADICTS verdicts (scan/contradiction.py
explain_contradiction) and on demand through
POST /admin/api/conflicts/{id}/explain for rows found before this. NULL
on rows written before this column, on dedup_scan rows, and when the
explanation call failed (the conflict is still written).
"""
from alembic import op
import sqlalchemy as sa

revision = "d0a2b4c6e8f0"
down_revision = "c9f1a3b5d7e9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("knowledge_conflicts", sa.Column("reasoning", sa.Text(), nullable=True))
    op.add_column("knowledge_conflicts", sa.Column("recommendation", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("knowledge_conflicts", "recommendation")
    op.drop_column("knowledge_conflicts", "reasoning")
