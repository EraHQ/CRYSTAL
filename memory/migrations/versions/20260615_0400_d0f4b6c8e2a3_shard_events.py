"""shard events + expert authorizations (marketplace)

Revision ID: d0f4b6c8e2a3
Revises: c9e3a5b7d1f2
Create Date: 2026-06-15 04:00:00.000000

Growth G4 (marketplace: shard ledger + vetting) — the append-only economy
substrate. Creates two tables:

  shard_events — the append-only shard ledger. Metering is CITATIONS (the G1
                 rail): a *cited* crystal (grounding-gated, self-traffic
                 excluded) accrues a shard. Never mutated; corrections are
                 compensating entries (event_type='clawback'). IDEMPOTENT via a
                 UNIQUE index on (interaction_id, crystal_id, event_type) so one
                 interaction can never double-credit a crystal (a credit and its
                 later clawback coexist via the distinct event_type; NULLs are
                 distinct, so spends — all-NULL keys — are unconstrained).
                 INTEGER shard units; balance = sum(shards_credited); spends are
                 negative debits in the same ledger (closed-loop, no cash-out).

  expert_authorizations — the vetting scope registry: an operator authorized to
                 author general crystals in a `general:<domain>`. The minimal
                 substrate the team→general promotion gate checks; reputation /
                 dispute rows are deferred.

CREATE TABLE (not ALTER) so the SQLite ALTER-ADD-FK quirks don't apply.
Columns are PLAIN (no FK constraints) per the F4/G1 precedent — owner_operator_id
/ crystal_id / consuming_team_id / interaction_id are soft pointers; the ORM
keeps the team_id FK on expert_authorizations for Postgres + docs.

Idempotent on upgrade (skip per-table if it already exists). Downgrade drops
indexes then tables.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd0f4b6c8e2a3'
down_revision: Union[str, None] = 'c9e3a5b7d1f2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    existing_tables = set(sa.inspect(bind).get_table_names())

    if "shard_events" not in existing_tables:
        op.create_table(
            "shard_events",
            sa.Column("id", sa.String(length=64), nullable=False),
            sa.Column(
                "owner_operator_id", sa.String(length=64), nullable=True
            ),
            sa.Column("crystal_id", sa.String(length=64), nullable=True),
            sa.Column(
                "consuming_team_id", sa.String(length=64), nullable=True
            ),
            sa.Column("interaction_id", sa.String(length=64), nullable=True),
            sa.Column("event_type", sa.String(length=16), nullable=False),
            sa.Column(
                "signal_type",
                sa.String(length=32),
                nullable=False,
                server_default="citation",
            ),
            sa.Column(
                "raw_weight", sa.Float(), nullable=False, server_default="0"
            ),
            sa.Column(
                "shards_credited",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "ux_shard_events_idempotent",
            "shard_events",
            ["interaction_id", "crystal_id", "event_type"],
            unique=True,
        )
        op.create_index(
            "ix_shard_events_owner",
            "shard_events",
            ["owner_operator_id", "created_at"],
        )
        op.create_index(
            "ix_shard_events_crystal",
            "shard_events",
            ["crystal_id"],
        )

    if "expert_authorizations" not in existing_tables:
        op.create_table(
            "expert_authorizations",
            sa.Column("id", sa.String(length=64), nullable=False),
            sa.Column("operator_id", sa.String(length=64), nullable=False),
            sa.Column("team_id", sa.String(length=64), nullable=False),
            sa.Column("domain", sa.String(length=128), nullable=False),
            sa.Column(
                "status",
                sa.String(length=16),
                nullable=False,
                server_default="active",
            ),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "ux_expert_auth_operator_domain",
            "expert_authorizations",
            ["operator_id", "domain"],
            unique=True,
        )


def downgrade() -> None:
    op.drop_index(
        "ux_expert_auth_operator_domain", table_name="expert_authorizations"
    )
    op.drop_table("expert_authorizations")
    op.drop_index("ix_shard_events_crystal", table_name="shard_events")
    op.drop_index("ix_shard_events_owner", table_name="shard_events")
    op.drop_index("ux_shard_events_idempotent", table_name="shard_events")
    op.drop_table("shard_events")
