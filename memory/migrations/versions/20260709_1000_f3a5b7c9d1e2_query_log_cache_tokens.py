"""query_logs cache token columns.

S12 (2026-07-09, docs/INSPECTOR_DATA_AUDIT.md tail): the agent loop has
accumulated cache_creation/cache_read token totals since C1 (prompt
caching) and returned them per turn — but query_logs had no columns, so
agent rows showed only the NON-cached input delta (prompt_tokens=20 on
a turn that actually read ~9k cached tokens). These columns make the
Logs surface tell the whole caching story. Nullable: rows predating
this migration and non-agent paths simply have no data.
"""
from alembic import op
import sqlalchemy as sa

revision = "f3a5b7c9d1e2"
down_revision = "e2f4a6b8c0d1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "query_logs",
        sa.Column("cache_creation_tokens", sa.Integer(), nullable=True),
    )
    op.add_column(
        "query_logs",
        sa.Column("cache_read_tokens", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("query_logs", "cache_read_tokens")
    op.drop_column("query_logs", "cache_creation_tokens")
