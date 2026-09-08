"""customers.stripe_customer_id (L2-S4=B — the customer-portal join)

Revision ID: d4f8a0b2c4e6
Revises: c3d5e7f9b1a2
Create Date: 2026-09-08

The webhook persists Stripe's customer id at checkout completion so the
console can open Stripe's hosted customer portal (plan management,
invoices, cancellation) without us building any of it. NULL = never paid
(trial / self-host) — the portal route answers 409 and the console shows
the upgrade button instead.
"""
from alembic import op
import sqlalchemy as sa

revision = "d4f8a0b2c4e6"
down_revision = "c3d5e7f9b1a2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("customers") as batch_op:
        batch_op.add_column(
            sa.Column("stripe_customer_id", sa.String(length=64),
                      nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("customers") as batch_op:
        batch_op.drop_column("stripe_customer_id")
