"""facts.citation — where each fact's knowledge is attributed FROM.

Ingestion Gate A (ratified 2026-07-16): extraction profiles emit a
citation per item (source URL for research reports, clause/scene/
speaker refs for documents). Facts had no column for it —
source_kind is the evidence CLASS and source_doc_id points at the
upload, not the external source. Nullable Text: URLs and document-
internal references both fit, and the web_search_logs URL join the
backlog anticipates becomes possible.

Revision ID: f6a8b0c2d4e6
Revises: e5f7a9b1c3d4
Create Date: 2026-07-16
"""
from alembic import op
import sqlalchemy as sa

revision = "f6a8b0c2d4e6"
down_revision = "e5f7a9b1c3d4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "facts",
        sa.Column("citation", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("facts", "citation")
