"""General lattices (2026-10-27; Q145, Q146 B).

lattices.scope: private (default) or general. A general lattice's table
lives in the shared general schema of the lattice database; its owner
(customer_id) is the only writer; lattice_subscriptions records who reads
it.
"""
from alembic import op
import sqlalchemy as sa

revision = "a2b4c6d8e0f2"
down_revision = "f1a3c5e7b9d1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("lattices", sa.Column("scope", sa.String(16), nullable=False, server_default="private"))
    op.create_table(
        "lattice_subscriptions",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("customer_id", sa.String(64), nullable=False),
        sa.Column("lattice_id", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("customer_id", "lattice_id", name="ux_lattice_subscriptions_customer_lattice"),
    )
    op.create_index("ix_lattice_subscriptions_customer_id", "lattice_subscriptions", ["customer_id"])
    op.create_index("ix_lattice_subscriptions_lattice_id", "lattice_subscriptions", ["lattice_id"])


def downgrade() -> None:
    op.drop_index("ix_lattice_subscriptions_lattice_id", table_name="lattice_subscriptions")
    op.drop_index("ix_lattice_subscriptions_customer_id", table_name="lattice_subscriptions")
    op.drop_table("lattice_subscriptions")
    op.drop_column("lattices", "scope")
