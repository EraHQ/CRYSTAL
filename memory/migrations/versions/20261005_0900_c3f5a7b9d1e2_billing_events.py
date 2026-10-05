"""billing_events (RC-13, 2026-10-05): Stripe webhook idempotency and ordering.

Stripe retries deliveries and does not guarantee order. Without a record
of what was processed, a replayed checkout.session.completed re-granted a
tier, and a late subscription.updated could overwrite a newer deleted.
One row per processed event id; the latest `created` per customer is the
ordering watermark.
"""
from alembic import op
import sqlalchemy as sa

revision = "c3f5a7b9d1e2"
down_revision = "b2e4a6c8d0f1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "billing_events",
        sa.Column("event_id", sa.String(128), primary_key=True),
        sa.Column("customer_id", sa.String(64), nullable=True, index=True),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("created", sa.BigInteger, nullable=False),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("billing_events")
