"""crystals.source_document_id (Lockdown PR-4, Q54=A, 2026-10-09).

The upload a crystal was born from, recorded on the crystal at write
time (file crystals and spawned item crystals; a pair that bonds into an
existing crystal leaves that crystal's stamp alone). The forget scrub
matches on this id, not on source_uri: fragment carves (#sheet=,
#msg-window=) and repo:// code crystals never equalled the upload row's
URI, so those uploads were never scrubbed. NULL on rows written before
this column; those fall back to the URI match.
"""
from alembic import op
import sqlalchemy as sa

revision = "b8e0f2a4c6d8"
down_revision = "a7d9e1f3b5c7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "crystals",
        sa.Column("source_document_id", sa.String(64), nullable=True),
    )
    op.create_index(
        "ix_crystals_source_document_id", "crystals", ["source_document_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_crystals_source_document_id", table_name="crystals")
    op.drop_column("crystals", "source_document_id")
