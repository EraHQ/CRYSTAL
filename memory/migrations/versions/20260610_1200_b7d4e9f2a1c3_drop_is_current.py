"""drop is_current from crystals — VS-D3 replace semantics

Revision ID: b7d4e9f2a1c3
Revises: 81cf707a1672
Create Date: 2026-06-10 12:00:00

VS-D3 was relocked on 2026-06-10 from supersede semantics (keep stale
crystals with is_current=False) to REPLACE semantics: a changed source
DELETES its prior crystals and writes fresh ones. No stale crystals
ever exist, so the flag has no consumer — and per R6 a column nothing
reads or writes doesn't stay in the schema.

batch_alter_table so the drop works on SQLite (dev) as well as
Postgres.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "b7d4e9f2a1c3"
down_revision = "81cf707a1672"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("crystals") as batch_op:
        batch_op.drop_column("is_current")


def downgrade() -> None:
    with op.batch_alter_table("crystals") as batch_op:
        batch_op.add_column(sa.Column("is_current", sa.Boolean(), nullable=True))
