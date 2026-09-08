"""customers.trial_expires_at (L2-S2, Q4=B — the trial that actually expires)

Revision ID: c3d5e7f9b1a2
Revises: b2c4d6e8f0a2
Create Date: 2026-09-07

Signup stamps subscription_tier='trial_29' + trial_expires_at=now+7d
(me.py). Expiry degrades WRITES only — remember/ingest/agent runs pause
with an upgrade pointer while recall and the console stay fully alive
(a memory product never holds memories hostage). NULL trial_expires_at
= not a trial (existing tenants, self-host, paid tiers) and is never
degraded; the S3 billing webhook clears the trial state on payment.
"""
from alembic import op
import sqlalchemy as sa

revision = "c3d5e7f9b1a2"
down_revision = "b2c4d6e8f0a2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("customers") as batch_op:
        batch_op.add_column(
            sa.Column("trial_expires_at", sa.DateTime(timezone=True),
                      nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("customers") as batch_op:
        batch_op.drop_column("trial_expires_at")
