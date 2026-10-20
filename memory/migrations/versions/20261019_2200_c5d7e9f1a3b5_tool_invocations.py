"""tool_invocations (Activity trace, 2026-10-19).

One row per tool call from a connected client (every MCP tool): who
called it, what it was asked (shortened arguments), how it ended and how
long it took, with a link to the query_logs row when the call was a
recall. The Activity page's "Tool calls" view reads it.
"""
from alembic import op
import sqlalchemy as sa

revision = "c5d7e9f1a3b5"
down_revision = "b4c6d8e0f2a4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tool_invocations",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("customer_id", sa.String(length=64), nullable=False),
        sa.Column("surface", sa.String(length=16), nullable=False, server_default="mcp"),
        sa.Column("client", sa.String(length=64), nullable=True),
        sa.Column("tool", sa.String(length=64), nullable=False),
        sa.Column("args", sa.JSON(), nullable=True),
        sa.Column("ok", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("code", sa.String(length=64), nullable=True),
        sa.Column("result_summary", sa.Text(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("query_log_id", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_tool_invocations_customer_id", "tool_invocations", ["customer_id"])
    op.create_index("ix_tool_invocations_customer_created", "tool_invocations", ["customer_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_tool_invocations_customer_created", table_name="tool_invocations")
    op.drop_index("ix_tool_invocations_customer_id", table_name="tool_invocations")
    op.drop_table("tool_invocations")
