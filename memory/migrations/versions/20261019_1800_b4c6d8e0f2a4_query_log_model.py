"""query_logs.model (M3 follow-up, 2026-10-19).

Which model answered: the agent run's resolved model id (a Claude id
or an nvidia/ Nemotron id on the Token Factory lane). A tenant who
picked Nemotron for Answers had no way to confirm it ran — the Query
log and the chat trace now say so. NULL on recall-only rows (MCP
tools, proxy) and on rows written before this column.
"""
from alembic import op
import sqlalchemy as sa

revision = "b4c6d8e0f2a4"
down_revision = "a3b5c7d9e1f3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("query_logs", sa.Column("model", sa.String(length=128), nullable=True))


def downgrade() -> None:
    op.drop_column("query_logs", "model")
