"""agent_conversations.model — per-conversation model selection (C6)

Revision ID: b5d7e9c1f3a6
Revises: a4c6e8b0d2f5
Create Date: 2026-06-17 19:00:00.000000

C6 (model selection). Adds one nullable column:

  agent_conversations.model — the controlling model selected for a
                              conversation. The web client's explicit choice is
                              persisted here (last-writer-wins) and reused on
                              later turns from any device; NULL = no selection,
                              fall back to the CC_AGENT_MODEL house default then
                              the built-in DEFAULT_MODEL. Mirrors
                              agent_sessions.model (String(128)).

ADD COLUMN via batch_alter_table so it works on SQLite (dev) as well as
Postgres — the drop_is_current precedent. Idempotent: skip if the column is
already present (a fresh create_all DB built from the current model already has
it, mirroring the agent_events / knowledge_conflicts idempotent guards).
Downgrade drops the column.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b5d7e9c1f3a6'
down_revision: Union[str, None] = 'a4c6e8b0d2f5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    columns = {
        c["name"] for c in sa.inspect(bind).get_columns("agent_conversations")
    }
    if "model" not in columns:
        with op.batch_alter_table("agent_conversations") as batch_op:
            batch_op.add_column(
                sa.Column("model", sa.String(length=128), nullable=True)
            )


def downgrade() -> None:
    bind = op.get_bind()
    columns = {
        c["name"] for c in sa.inspect(bind).get_columns("agent_conversations")
    }
    if "model" in columns:
        with op.batch_alter_table("agent_conversations") as batch_op:
            batch_op.drop_column("model")
