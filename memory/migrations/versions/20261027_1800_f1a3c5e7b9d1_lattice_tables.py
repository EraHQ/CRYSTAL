"""Lattices as real tables (2026-10-27; Q144 A).

Each lattice gets its own table in a schema that belongs to its customer
(infrastructure/lattice_tables.py). Existing lattices keep their JSON rows
in lattice_rows until first touched; then the app moves them (claimed on
table_name) and fills sql_name, the name the read-only SQL tool uses.
"""
from alembic import op
import sqlalchemy as sa

revision = "f1a3c5e7b9d1"
down_revision = "d7f9b1c3e5a7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("lattices", sa.Column("table_name", sa.String(64), nullable=True))
    op.add_column("lattices", sa.Column("sql_name", sa.String(64), nullable=True))


def downgrade() -> None:
    op.drop_column("lattices", "sql_name")
    op.drop_column("lattices", "table_name")
