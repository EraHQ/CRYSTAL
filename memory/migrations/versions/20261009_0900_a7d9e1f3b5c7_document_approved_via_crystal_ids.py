"""document_uploads.approved_via + crystal_ids (Lockdown PR-3, 2026-10-09).

approved_via (Q50=A): which kind of credential approved the row —
'console' (a signed-in console session that saw the review surface),
'key' (Key A / an operator key, programmatic), 'auto' (RC-17
auto_crystallize), 'mcp' (memory_ingest). Only 'console' is a curator
verdict: the write leg sets curator_reviewed from this column, so a
programmatic approve keeps the injection quarantine.

crystal_ids (B2-1): the crystal set the write leg actually produced for
this document — file crystals and item crystals — recorded by the server
at write time. Share-source acts on this list, never on ids a client
could have placed in extracted_items. NULL for rows written before this
column; those fall back to provenance resolution filtered by ownership.
"""
from alembic import op
import sqlalchemy as sa

revision = "a7d9e1f3b5c7"
down_revision = "f6c8d0e2a4b6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("document_uploads") as batch_op:
        batch_op.add_column(sa.Column("approved_via", sa.String(16), nullable=True))
        batch_op.add_column(sa.Column("crystal_ids", sa.JSON, nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("document_uploads") as batch_op:
        batch_op.drop_column("crystal_ids")
        batch_op.drop_column("approved_via")
