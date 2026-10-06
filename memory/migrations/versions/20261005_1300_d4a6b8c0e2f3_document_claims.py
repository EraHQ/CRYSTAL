"""document_uploads.claimed_at + claimed_from (RC-08, 2026-10-05).

A worker that claims a document ('pending' or 'approved' -> 'crystallizing')
and dies leaves the row crystallizing forever; nothing reclaimed it and
the customer's upload never finished. The claim now stamps when it
happened and what status it came from, so a reclaim sweep can return a
stale claim to its prior status for another worker to pick up.
"""
from alembic import op
import sqlalchemy as sa

revision = "d4a6b8c0e2f3"
down_revision = "c3f5a7b9d1e2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("document_uploads") as batch_op:
        batch_op.add_column(sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("claimed_from", sa.String(32), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("document_uploads") as batch_op:
        batch_op.drop_column("claimed_from")
        batch_op.drop_column("claimed_at")
