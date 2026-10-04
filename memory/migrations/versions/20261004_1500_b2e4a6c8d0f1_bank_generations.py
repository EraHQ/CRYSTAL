"""bank_generations (RC-01, 2026-10-04): the database owns the bank's version.

Every bank write or delete bumps a per-customer integer in one place
(MetadataStore.bank_changed). Every index (the in-memory fact and routing
stores, the Qdrant mirror) remembers the generation it loaded for a
customer and compares it to this table on each search; behind means
reload. That is what lets the API process see what the worker process
wrote, and what makes "two fact indexes drift apart" impossible by
construction instead of by remembering to notify both.
"""
from alembic import op
import sqlalchemy as sa

revision = "b2e4a6c8d0f1"
down_revision = "a1d3f5b7c9e2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "bank_generations",
        sa.Column("customer_id", sa.String(64), primary_key=True),
        sa.Column("generation", sa.Integer, nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("bank_generations")
