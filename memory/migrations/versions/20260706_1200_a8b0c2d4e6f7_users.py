"""users table (Accounts Phase A: hosted-platform identity)

Revision ID: a8b0c2d4e6f7
Revises: f7b9c1d3e5a6
Create Date: 2026-07-06

The IdP-anchored sign-in layer: id = GCP Identity Platform uid; one user
-> one tenant in v1 (customer_id NULL only for platform_admin). Local
default stores create this free via store.init(); the Alembic-managed
dev DB takes this migration.
"""
from alembic import op
import sqlalchemy as sa

revision = "a8b0c2d4e6f7"
down_revision = "f7b9c1d3e5a6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=128), primary_key=True),
        sa.Column("email", sa.String(length=320), nullable=False, unique=True),
        sa.Column(
            "customer_id", sa.String(length=64),
            sa.ForeignKey("customers.id"), nullable=True, index=True,
        ),
        sa.Column(
            "role", sa.String(length=32),
            nullable=False, server_default="owner",
        ),
        sa.Column("industry", sa.String(length=128), nullable=True),
        sa.Column("building", sa.Text(), nullable=True),
        sa.Column("experience", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("users")
