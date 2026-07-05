"""agent_sessions

Revision ID: e7a2c9d4b1f8
Revises: d5f1a2b3c4e6
Create Date: 2026-06-14 21:15:00.000000

Foundation F4 (surface consolidation) — the live session registry. Creates
two tables:

  agent_sessions       — one registered CRYS session per team, with the
                         self-reported status/current_action and the
                         last_heartbeat_at liveness signal (staleness ⇒
                         presumed crashed).
  session_dependencies — the resources a session spawned (mcp_server /
                         subprocess / browser / queued_task / pip_env), so a
                         crashed session's dependencies can be orphaned.

CREATE TABLE (not ALTER), so the SQLite ALTER-ADD-FK quirks the F2 migration
worked around don't apply. Following that precedent, columns are PLAIN (no FK
constraints) — SQLite doesn't enforce FKs and the joins are application-level;
the ORM models keep the FK declarations on team_id / operator_id for Postgres
+ documentation.

Idempotent on upgrade: each table's create is skipped if it already exists (a
local dev DB may have picked them up via store.init()'s create_all before
this migration ran). Downgrade drops indexes then tables.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e7a2c9d4b1f8'
down_revision: Union[str, None] = 'd5f1a2b3c4e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    existing_tables = set(sa.inspect(bind).get_table_names())

    if "agent_sessions" not in existing_tables:
        op.create_table(
            "agent_sessions",
            sa.Column("session_id", sa.String(length=64), nullable=False),
            sa.Column("team_id", sa.String(length=64), nullable=False),
            sa.Column("operator_id", sa.String(length=64), nullable=True),
            sa.Column("host", sa.String(length=256), nullable=True),
            sa.Column("pid", sa.Integer(), nullable=True),
            sa.Column("project_dir", sa.Text(), nullable=True),
            sa.Column("model", sa.String(length=128), nullable=True),
            sa.Column(
                "status",
                sa.String(length=32),
                nullable=False,
                server_default="starting",
            ),
            sa.Column("current_action", sa.Text(), nullable=True),
            sa.Column("awaiting_payload", sa.JSON(), nullable=True),
            sa.Column("parent_session_id", sa.String(length=64), nullable=True),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column(
                "last_heartbeat_at", sa.DateTime(timezone=True), nullable=True
            ),
            sa.Column(
                "cost_usd_cumulative",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
            sa.PrimaryKeyConstraint("session_id"),
        )
        op.create_index(
            "ix_agent_sessions_team",
            "agent_sessions",
            ["team_id", "last_heartbeat_at"],
        )
        op.create_index(
            "ix_agent_sessions_parent_session_id",
            "agent_sessions",
            ["parent_session_id"],
        )

    if "session_dependencies" not in existing_tables:
        op.create_table(
            "session_dependencies",
            sa.Column("dependency_id", sa.String(length=64), nullable=False),
            sa.Column("session_id", sa.String(length=64), nullable=False),
            sa.Column("kind", sa.String(length=32), nullable=False),
            sa.Column(
                "descriptor", sa.Text(), nullable=False, server_default=""
            ),
            sa.Column("pid", sa.Integer(), nullable=True),
            sa.Column(
                "status",
                sa.String(length=32),
                nullable=False,
                server_default="active",
            ),
            sa.Column("spawned_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("dependency_id"),
        )
        op.create_index(
            "ix_session_dependencies_session",
            "session_dependencies",
            ["session_id"],
        )


def downgrade() -> None:
    op.drop_index(
        "ix_session_dependencies_session", table_name="session_dependencies"
    )
    op.drop_table("session_dependencies")
    op.drop_index(
        "ix_agent_sessions_parent_session_id", table_name="agent_sessions"
    )
    op.drop_index("ix_agent_sessions_team", table_name="agent_sessions")
    op.drop_table("agent_sessions")
