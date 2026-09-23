"""users.ai_tools + customers.last_mcp_seen_at (T2a onboarding, 2026-09-25)

ai_tools: the environment picker's answer (comma-joined ids) — drives
the connect tabs and the console checklist. last_mcp_seen_at: the
first-contact signal, stamped (throttled) by the MCP auth middleware;
the onboarding connect screen polls it to flip "Listening…" to
"Connected", and NULL simply means no tool has called yet.
"""
from alembic import op
import sqlalchemy as sa

revision = "e5b9c1d3f5a7"
down_revision = "d4f8a0b2c4e6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("users") as batch_op:
        batch_op.add_column(sa.Column("ai_tools", sa.Text(), nullable=True))
    with op.batch_alter_table("customers") as batch_op:
        batch_op.add_column(
            sa.Column("last_mcp_seen_at", sa.DateTime(timezone=True),
                      nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("customers") as batch_op:
        batch_op.drop_column("last_mcp_seen_at")
    with op.batch_alter_table("users") as batch_op:
        batch_op.drop_column("ai_tools")
