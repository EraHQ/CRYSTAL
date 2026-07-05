"""P2 scope-on-sources: document_uploads carries scope + owner.

Revision ID: f9b1d3e5a7c0
Revises: e8a0c2d4f6b9
Create Date: 2026-07-03

A document is a SOURCE (ratified 2026-07-02) and carries its own scope;
every crystal born from it inherits the stamps. Both columns are
NULLABLE with no backfill: NULL = legacy row -> team-scoped unowned
crystals, exactly today's behavior, so in-flight uploads and the
downgrade are both safe no-ops.
"""
from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "f9b1d3e5a7c0"
down_revision: Union[str, None] = "e8a0c2d4f6b9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("document_uploads") as batch:
        batch.add_column(sa.Column("scope", sa.String(16), nullable=True))
        batch.add_column(
            sa.Column("owner_operator_id", sa.String(64), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("document_uploads") as batch:
        batch.drop_column("owner_operator_id")
        batch.drop_column("scope")
