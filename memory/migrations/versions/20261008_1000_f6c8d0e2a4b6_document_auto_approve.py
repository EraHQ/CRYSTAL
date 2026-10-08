"""document_uploads.auto_approve (RC-17, 2026-10-08).

Onboarding's "About you" upload sends auto_crystallize=true; the route
accepted the flag and dropped it, so the first thing a customer uploaded
sat waiting for an approval the wizard never mentioned. The flag is now
stored; when extraction finishes, an auto_approve row goes straight to
'approved' and the write leg runs.
"""
from alembic import op
import sqlalchemy as sa

revision = "f6c8d0e2a4b6"
down_revision = "e5b7c9d1f3a4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("document_uploads") as batch_op:
        batch_op.add_column(sa.Column("auto_approve", sa.Boolean, nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("document_uploads") as batch_op:
        batch_op.drop_column("auto_approve")
