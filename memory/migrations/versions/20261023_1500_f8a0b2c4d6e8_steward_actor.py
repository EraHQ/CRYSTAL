"""steward_actions.actor (2026-10-23): who used the controls.

"steward" is the in-app session; "chat:<operator>" is a person or their
AI client acting through the MCP controls. Both land in the same ledger
with the same undo.
"""
from alembic import op
import sqlalchemy as sa

revision = "f8a0b2c4d6e8"
down_revision = "e7f9a1b3c5d7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "steward_actions",
        sa.Column("actor", sa.String(length=64), nullable=False, server_default="steward"),
    )


def downgrade() -> None:
    op.drop_column("steward_actions", "actor")
