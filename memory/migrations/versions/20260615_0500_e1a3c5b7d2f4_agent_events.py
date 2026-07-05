"""agent events stream (unify CRYS in the Inspector)

Revision ID: e1a3c5b7d2f4
Revises: d0f4b6c8e2a3
Create Date: 2026-06-15 05:00:00.000000

Unify-Agents (make everything CRYS does visible) — the append-only event
stream. Creates one table:

  agent_events — the per-session activity record CRYS writes (turn boundaries,
                 every tool call, delegated subagents, crystals written, gaps
                 opened, daemon queue transitions, cost). Keyed by session_id,
                 ordered by a per-session monotonic `seq`. APPEND-ONLY with a
                 JSON `payload` so new event_types never need a migration;
                 event_type is a free-form String for the same reason. All
                 pointers (session_id, parent_session_id) are PLAIN columns,
                 no FK — the F4 session precedent (events outlive/predate the
                 rows they reference). This is the backbone the "Agents"
                 surface, the unified interaction log, and cost rollups read.

CREATE TABLE (not ALTER) so the SQLite ALTER-ADD-FK quirks don't apply.
Idempotent on upgrade (skip if the table already exists, like the F4/G1-G4
migrations). Downgrade drops indexes then the table.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e1a3c5b7d2f4'
down_revision: Union[str, None] = 'd0f4b6c8e2a3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    existing_tables = set(sa.inspect(bind).get_table_names())

    if "agent_events" not in existing_tables:
        op.create_table(
            "agent_events",
            sa.Column("id", sa.String(length=64), nullable=False),
            sa.Column("session_id", sa.String(length=64), nullable=False),
            sa.Column("team_id", sa.String(length=64), nullable=True),
            sa.Column(
                "seq", sa.Integer(), nullable=False, server_default="0"
            ),
            sa.Column("turn_index", sa.Integer(), nullable=True),
            sa.Column(
                "parent_session_id", sa.String(length=64), nullable=True
            ),
            sa.Column("event_type", sa.String(length=48), nullable=False),
            sa.Column("phase", sa.String(length=24), nullable=True),
            sa.Column(
                "label", sa.Text(), nullable=False, server_default=""
            ),
            sa.Column("payload", sa.JSON(), nullable=True),
            sa.Column("status", sa.String(length=16), nullable=True),
            sa.Column("duration_ms", sa.Integer(), nullable=True),
            sa.Column("tokens_input", sa.Integer(), nullable=True),
            sa.Column("tokens_output", sa.Integer(), nullable=True),
            sa.Column("cost_micro_usd", sa.Integer(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "ix_agent_events_session_seq",
            "agent_events",
            ["session_id", "seq"],
        )
        op.create_index(
            "ix_agent_events_team_created",
            "agent_events",
            ["team_id", "created_at"],
        )
        op.create_index(
            "ix_agent_events_type",
            "agent_events",
            ["event_type"],
        )


def downgrade() -> None:
    op.drop_index("ix_agent_events_type", table_name="agent_events")
    op.drop_index("ix_agent_events_team_created", table_name="agent_events")
    op.drop_index("ix_agent_events_session_seq", table_name="agent_events")
    op.drop_table("agent_events")
