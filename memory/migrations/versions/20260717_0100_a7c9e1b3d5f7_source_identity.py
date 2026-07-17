"""Gate D: canonical source identity + fact ordering (C1/C2/C4).

crystals.source_uri — the scheme-qualified LOCATION identity
(upload://, gdrive://, url://, connector://, repo://) that versioning,
the watcher, and supersede-delete key on. source_path stays as the
human-readable raw path; pre-D crystals match by the legacy fallback
and converge on replace.
document_uploads.source_uri + content_hash — the upload's identity pair
(location + sha256 of extracted text).
facts.chunk_index — fact ordering inside file-grain content crystals
(VS-D1: one crystal per source, chunks as ordered facts).

Revision ID: a7c9e1b3d5f7
Revises: f6a8b0c2d4e6
Create Date: 2026-07-16
"""
from alembic import op
import sqlalchemy as sa

revision = "a7c9e1b3d5f7"
down_revision = "f6a8b0c2d4e6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "crystals",
        sa.Column("source_uri", sa.String(512), nullable=True),
    )
    op.create_index(
        "ix_crystals_source_uri", "crystals", ["source_uri"],
    )
    op.add_column(
        "document_uploads",
        sa.Column("source_uri", sa.String(512), nullable=True),
    )
    op.add_column(
        "document_uploads",
        sa.Column("content_hash", sa.String(64), nullable=True),
    )
    op.add_column(
        "facts",
        sa.Column("chunk_index", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("facts", "chunk_index")
    op.drop_column("document_uploads", "content_hash")
    op.drop_column("document_uploads", "source_uri")
    op.drop_index("ix_crystals_source_uri", table_name="crystals")
    op.drop_column("crystals", "source_uri")
