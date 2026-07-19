"""Gate M: source_watches — the general watch registration (M-Q1=A).

One table for every watched source, present and future: git repos now,
watch-folders and unified Drive later. scheme + config JSON + last_state
JSON carry any source shape; review_mode implements M-Q3 (auto = born-
quarantine unattended ingest, gated = review queue); encrypted_token
implements M-Q5 (per-watch credential, falls back to env, tokenless for
public sources). Git is the first tenant, not the mold.

Revision ID: b8d0f2a4c6e8
Revises: a7c9e1b3d5f7
Create Date: 2026-07-18
"""
from alembic import op
import sqlalchemy as sa

revision = "b8d0f2a4c6e8"
down_revision = "a7c9e1b3d5f7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "source_watches",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "customer_id", sa.String(64),
            sa.ForeignKey("customers.id"), nullable=False,
        ),
        sa.Column("scheme", sa.String(32), nullable=False),
        sa.Column("source_name", sa.String(256), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("cadence_minutes", sa.Integer(), nullable=False,
                  server_default="15"),
        sa.Column("last_state", sa.JSON(), nullable=True),
        sa.Column("review_mode", sa.String(16), nullable=False,
                  server_default="auto"),
        sa.Column("encrypted_token", sa.Text(), nullable=True),
        sa.Column("status", sa.String(32), nullable=False,
                  server_default="active"),
        sa.Column("last_checked_at", sa.DateTime(timezone=True),
                  nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  nullable=False),
    )
    op.create_index(
        "ix_source_watches_customer_status",
        "source_watches", ["customer_id", "status"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_source_watches_customer_status", table_name="source_watches",
    )
    op.drop_table("source_watches")
