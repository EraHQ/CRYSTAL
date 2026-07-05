"""oauth_states table (F1 OAuth CSRF fix)

Revision ID: d5f7a9b1c3e4
Revises: c4e6f8a0b2d3
Create Date: 2026-07-03

Server-side single-use OAuth state nonces for the Drive connect flow.
The callback only proceeds when the presented state exists here and is
fresh; rows are deleted on consumption.
"""
from alembic import op
import sqlalchemy as sa

revision = "d5f7a9b1c3e4"
down_revision = "c4e6f8a0b2d3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "oauth_states",
        sa.Column("state", sa.String(), primary_key=True),
        sa.Column(
            "customer_id", sa.String(),
            sa.ForeignKey("customers.id"), nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("oauth_states")
