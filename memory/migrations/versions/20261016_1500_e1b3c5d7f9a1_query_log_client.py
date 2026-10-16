"""query_logs.client (Console Refine PR 11, Q65=A, 2026-10-16).

Which tool made the call: the HTTP User-Agent family for API and MCP
callers, "console" for the console's own routes. MCP recall tools now
write query log rows too (they never did), so a tenant's Query log
shows what Claude Desktop, Cursor or ChatGPT asked, and Getting
Started can complete "the first recall from a connected tool". NULL on
rows written before this column.
"""
from alembic import op
import sqlalchemy as sa

revision = "e1b3c5d7f9a1"
down_revision = "d0a2b4c6e8f0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("query_logs", sa.Column("client", sa.String(length=64), nullable=True))
    op.create_index("ix_query_logs_customer_client", "query_logs", ["customer_id", "client"])


def downgrade() -> None:
    op.drop_index("ix_query_logs_customer_client", table_name="query_logs")
    op.drop_column("query_logs", "client")
