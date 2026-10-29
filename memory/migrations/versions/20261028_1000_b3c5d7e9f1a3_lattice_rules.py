"""Lattice rules (2026-10-28; Q152).

lattices.rules: optional checks each row of the lattice must meet, as SQL
conditions over its columns (lattices/rules.py). Empty for every existing
lattice, so nothing already stored changes.
"""
from alembic import op
import sqlalchemy as sa

revision = "b3c5d7e9f1a3"
down_revision = "a2b4c6d8e0f2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("lattices", sa.Column("rules", sa.JSON(), nullable=False, server_default="[]"))


def downgrade() -> None:
    op.drop_column("lattices", "rules")
