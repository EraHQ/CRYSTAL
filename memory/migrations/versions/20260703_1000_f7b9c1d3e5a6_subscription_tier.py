"""customers.subscription_tier (Phase 3 G6: admission tiers)

Revision ID: f7b9c1d3e5a6
Revises: e6a8b0c2d4f5
Create Date: 2026-07-03

Names the tenant's row in the hosted admission tier table. NULL = the
deployment default tier; self-host ignores it.
"""
from alembic import op
import sqlalchemy as sa

revision = "f7b9c1d3e5a6"
down_revision = "e6a8b0c2d4f5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "customers",
        sa.Column("subscription_tier", sa.String(length=32), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("customers", "subscription_tier")
