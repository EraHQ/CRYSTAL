"""knowledge_gaps provenance: full_key + triggering_query.

Gap Engine redesign S3 (2026-07-08, docs/GAP_ENGINE_AND_LEARN_REDESIGN.md
P5): a gap carries the complete sparse key when anchored to one (never a
bare Subject) and, when demand-driven, the query that missed. Both
nullable — operator topics have neither; scan-born gaps have a key but
no query.
"""
from alembic import op
import sqlalchemy as sa

revision = "c0d2e4f6a8b9"
down_revision = "b9c1d3e5f7a8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "knowledge_gaps",
        sa.Column("full_key", sa.Text(), nullable=True),
    )
    op.add_column(
        "knowledge_gaps",
        sa.Column("triggering_query", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("knowledge_gaps", "triggering_query")
    op.drop_column("knowledge_gaps", "full_key")
