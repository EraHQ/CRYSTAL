"""customers.deletion_scheduled_at + customers.purge_after (account deletion, 2026-10-03)

Q36=A / Q37=B (ratified 2026-10-03): "Delete my account" schedules an
erasure instead of running it. deletion_scheduled_at is the moment the
owner asked; purge_after is seven days later. Between the two the
account is locked (keys revoked, subscription cancelled, every route but
/v1/me and /v1/me/restore refused) and restorable from the console. The
purge worker erases everything past purge_after. NULL on both = a live
account. "Erase all memories" (Q36=A, immediate) needs no column.
"""
from alembic import op
import sqlalchemy as sa

revision = "a1d3f5b7c9e2"
down_revision = "f6a0c2d4e6b8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("customers") as batch_op:
        batch_op.add_column(
            sa.Column("deletion_scheduled_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch_op.add_column(
            sa.Column("purge_after", sa.DateTime(timezone=True), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("customers") as batch_op:
        batch_op.drop_column("purge_after")
        batch_op.drop_column("deletion_scheduled_at")
