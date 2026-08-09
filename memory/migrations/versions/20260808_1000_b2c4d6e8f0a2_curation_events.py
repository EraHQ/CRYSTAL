"""curation_events (C2 Q3=A — self-curation witness feed)

Revision ID: b2c4d6e8f0a2
Revises: a9b1c3d5e7f0
Create Date: 2026-08-08

Append-only activity feed for self-curation transitions: assumptions
written / approved / invalidated / deleted, gaps filled / reopened —
the watch-drawer pattern (source_watch_events) generalized so nothing
the system does to its own knowledge is silent. subject_id is a soft
pointer (crystal or gap id, deliberately no FK — events outlive their
subjects). Newest-first reads per customer via the composite index.
Feed substrate for the planned activity-drawer UI.
"""
from alembic import op
import sqlalchemy as sa

revision = "b2c4d6e8f0a2"
down_revision = "a9b1c3d5e7f0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "curation_events",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "customer_id",
            sa.String(length=64),
            sa.ForeignKey("customers.id"),
            nullable=False,
            index=True,
        ),
        sa.Column("subject_id", sa.String(length=64), nullable=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column(
            "label",
            sa.String(length=256),
            nullable=False,
            server_default="",
        ),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_curation_events_customer_created",
        "curation_events",
        ["customer_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_curation_events_customer_created",
        table_name="curation_events",
    )
    op.drop_table("curation_events")
