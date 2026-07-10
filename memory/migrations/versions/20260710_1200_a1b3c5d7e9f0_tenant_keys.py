"""tenant_keys — per-tenant wrapped DEKs (envelope layer 1).

P2 of the mature-posture secrets plan (ratified 2026-07-10). Each
tenant's Data Encryption Key is stored ONLY wrapped by the deployment's
KeyWrapper root (Cloud KMS HSM key in cloud; local master key in
self-host). destroy_scheduled_at carries the 24h crypto-shredding
grace window.
"""
from alembic import op
import sqlalchemy as sa

revision = "a1b3c5d7e9f0"
down_revision = "f3a5b7c9d1e2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tenant_keys",
        sa.Column("customer_id", sa.String(64),
                  sa.ForeignKey("customers.id"), primary_key=True),
        sa.Column("dek_wrapped", sa.Text(), nullable=False),
        sa.Column("kek_version", sa.String(256), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("rotated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("destroy_scheduled_at", sa.DateTime(timezone=True),
                  nullable=True),
    )


def downgrade() -> None:
    op.drop_table("tenant_keys")
