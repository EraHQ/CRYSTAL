"""The gap gate (2026-10-26; Q134 A).

A gap was opened by seven writers; one checked whether the bank already
answered it, and every check compared only against open gaps, so a
question answered or dismissed yesterday was asked again. Every writer now
goes through cognition/gap_gate.py, which needs on each gap:

  vector              the question's embedding (JSON, like facts.vector):
                      same-question checks against every gap, and new
                      facts checked against open gaps.
  answered_by_fact_id the fact that answers it: the documented edge from a
                      question to its answer (filled_by_crystal_id names
                      only the crystal).
  gate                the gate's record: {checked, verdict, score,
                      candidates, at}.
"""
from alembic import op
import sqlalchemy as sa

revision = "e9b1d3f5a7c9"
down_revision = "b3d5f7a9c1e3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("knowledge_gaps", sa.Column("vector", sa.JSON(), nullable=True))
    op.add_column("knowledge_gaps", sa.Column("answered_by_fact_id", sa.String(length=64), nullable=True))
    op.add_column("knowledge_gaps", sa.Column("gate", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("knowledge_gaps", "gate")
    op.drop_column("knowledge_gaps", "answered_by_fact_id")
    op.drop_column("knowledge_gaps", "vector")
