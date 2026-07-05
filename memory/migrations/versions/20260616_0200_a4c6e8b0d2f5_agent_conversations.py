"""agent conversations (CRYS session continuity)

Revision ID: a4c6e8b0d2f5
Revises: f3b5d7c9e1a4
Create Date: 2026-06-16 02:00:00.000000

CRYS session continuity (P5). Creates one table:

  agent_conversations — per-scope conversation transcripts so CRYS resumes
                        context across exit/relaunch. Mode-agnostic:
                        `conversation_key` is the resolved project_dir for the
                        CLI coding mode, a thread id for the future general/web
                        mode (the Inspector chat playground becoming CRYS).
                        DB-backed (not a local file) because the web playground
                        has no client-local store and a local file doesn't
                        follow a user across machines — the store IS the
                        boundary (F4 session precedent). transcript is the raw
                        anthropic `messages` list (JSON), capped by the caller.

ONE ROW PER SCOPE: unique (customer_id, conversation_key) — the CLI reuses the
project_dir key so a relaunch resumes the same rolling conversation (upsert
overwrites); the web mode uses unique thread ids. Soft customer_id (plain
indexed column, no FK) per the recent-table convention.

CREATE TABLE (not ALTER), idempotent on upgrade (skip if the table already
exists, like the agent_events / knowledge_conflicts migrations). Downgrade
drops indexes then the table.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a4c6e8b0d2f5'
down_revision: Union[str, None] = 'f3b5d7c9e1a4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    existing_tables = set(sa.inspect(bind).get_table_names())

    if "agent_conversations" not in existing_tables:
        op.create_table(
            "agent_conversations",
            sa.Column("id", sa.String(length=64), nullable=False),
            sa.Column("customer_id", sa.String(length=64), nullable=False),
            sa.Column("conversation_key", sa.String(length=512), nullable=False),
            sa.Column(
                "mode", sa.String(length=32), nullable=False,
                server_default="coding",
            ),
            sa.Column("title", sa.String(length=256), nullable=True),
            sa.Column("transcript", sa.JSON(), nullable=True),
            sa.Column(
                "turn_count", sa.Integer(), nullable=False, server_default="0"
            ),
            sa.Column("last_summary", sa.Text(), nullable=True),
            sa.Column("meta", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "ix_agent_conversations_customer_id",
            "agent_conversations",
            ["customer_id"],
        )
        op.create_index(
            "ux_agent_conversations_scope",
            "agent_conversations",
            ["customer_id", "conversation_key"],
            unique=True,
        )
        op.create_index(
            "ix_agent_conversations_customer_updated",
            "agent_conversations",
            ["customer_id", "updated_at"],
        )


def downgrade() -> None:
    op.drop_index(
        "ix_agent_conversations_customer_updated",
        table_name="agent_conversations",
    )
    op.drop_index(
        "ux_agent_conversations_scope", table_name="agent_conversations"
    )
    op.drop_index(
        "ix_agent_conversations_customer_id",
        table_name="agent_conversations",
    )
    op.drop_table("agent_conversations")
