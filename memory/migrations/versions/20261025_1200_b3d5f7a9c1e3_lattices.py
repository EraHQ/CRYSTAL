"""Lattices (2026-10-25; Q115 A, Q116 A, Q117 A).

lattices: named tables in Crystal storage, with typed columns (snapshot or
live cells, entity columns to browse by) and the columns that identify a
row. lattice_rows: their rows, with each cell's as-of date, source and,
for live cells, the values it held before. Tenant data: erased with the
memories. Rows count against the plan's lattice row cap, not the fact cap.
"""
from alembic import op
import sqlalchemy as sa

revision = "b3d5f7a9c1e3"
down_revision = "a1c3e5f7b9d1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "lattices",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("customer_id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("columns", sa.JSON(), nullable=False),
        sa.Column("key_columns", sa.JSON(), nullable=False),
        sa.Column("goal_id", sa.String(length=64), nullable=True),
        sa.Column("created_by", sa.String(length=96), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("customer_id", "name", name="ux_lattices_customer_name"),
    )
    op.create_index("ix_lattices_customer_id", "lattices", ["customer_id"])

    op.create_table(
        "lattice_rows",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("lattice_id", sa.String(length=64), nullable=False),
        sa.Column("customer_id", sa.String(length=64), nullable=False),
        sa.Column("row_key", sa.String(length=512), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
        sa.Column("cells", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("lattice_id", "row_key", name="ux_lattice_rows_lattice_key"),
    )
    op.create_index("ix_lattice_rows_lattice_id", "lattice_rows", ["lattice_id"])
    op.create_index("ix_lattice_rows_customer_id", "lattice_rows", ["customer_id"])
    op.create_index("ix_lattice_rows_lattice_updated", "lattice_rows", ["lattice_id", "updated_at"])


def downgrade() -> None:
    op.drop_index("ix_lattice_rows_lattice_updated", table_name="lattice_rows")
    op.drop_index("ix_lattice_rows_customer_id", table_name="lattice_rows")
    op.drop_index("ix_lattice_rows_lattice_id", table_name="lattice_rows")
    op.drop_table("lattice_rows")
    op.drop_index("ix_lattices_customer_id", table_name="lattices")
    op.drop_table("lattices")
