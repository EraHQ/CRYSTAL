"""The Steward (Q100 A, 2026-10-23).

customers.steward_enabled: the opt-in switch "Let Crystal run itself".
steward_actions: one row per action the Steward took, with the inverse
that undoes it (24-hour window) and when it was undone.
"""
from alembic import op
import sqlalchemy as sa

revision = "e7f9a1b3c5d7"
down_revision = "d6e8f0a2b4c6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "customers",
        sa.Column("steward_enabled", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_table(
        "steward_actions",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("customer_id", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=True),
        sa.Column("action", sa.String(length=48), nullable=False),
        sa.Column("subject_kind", sa.String(length=32), nullable=False),
        sa.Column("subject_id", sa.String(length=64), nullable=False),
        sa.Column("label", sa.String(length=256), nullable=False, server_default=""),
        sa.Column("reason", sa.Text(), nullable=False, server_default=""),
        sa.Column("inverse", sa.JSON(), nullable=True),
        sa.Column("undoable", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("undone_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_steward_actions_customer_id", "steward_actions", ["customer_id"])
    op.create_index("ix_steward_actions_customer_created", "steward_actions", ["customer_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_steward_actions_customer_created", table_name="steward_actions")
    op.drop_index("ix_steward_actions_customer_id", table_name="steward_actions")
    op.drop_table("steward_actions")
    op.drop_column("customers", "steward_enabled")
