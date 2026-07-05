"""crystal_posix_permissions

Revision ID: b2e7c9a4f1d3
Revises: aac586863af5
Create Date: 2026-06-14 17:30:00.000000

Foundation F2 — the crystal as an owned resource. Adds POSIX permission
columns to `crystals`:

  owner_operator_id  — the authoring operator (NULL for legacy /
                       non-operator-authored crystals).
  group_team_id      — the POSIX group, i.e. the owning team (NULL for
                       legacy, where the resolver falls back to
                       customer_id). Distinct from customer_id (the
                       owning TENANT).
  mode               — POSIX mode bits as an int. Only the READ bits are
                       consumed today (retrieval gating in
                       infrastructure/permissions.can_read); write/execute
                       reserved.

PLAIN columns — no inline FK. SQLite can't ALTER-ADD a foreign-key
constraint (Alembic's add_column routes the FK through add_constraint,
which the SQLite dialect rejects with NotImplementedError), and SQLite
doesn't enforce FKs anyway. The ORM model keeps the FK declarations
(owner_operator_id -> operators.id, group_team_id -> customers.id) for
Postgres + documentation; this SQLite dev migration adds bare columns.

Idempotent on upgrade. The first cut of this migration carried inline FKs:
Alembic added the owner_operator_id COLUMN, then raised on its FK
constraint — leaving that column present, group_team_id / mode absent, and
the alembic version un-bumped (a partial apply). Adding only the missing
columns lets `alembic upgrade head` recover without manual surgery and
stays safe to re-run. Downgrade drops the columns in a batch block
(SQLite drop-column safety across versions).
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b2e7c9a4f1d3'
down_revision: Union[str, None] = 'aac586863af5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    existing = {c["name"] for c in sa.inspect(bind).get_columns("crystals")}

    if "owner_operator_id" not in existing:
        op.add_column(
            "crystals",
            sa.Column("owner_operator_id", sa.String(length=64), nullable=True),
        )
    if "group_team_id" not in existing:
        op.add_column(
            "crystals",
            sa.Column("group_team_id", sa.String(length=64), nullable=True),
        )
    if "mode" not in existing:
        op.add_column(
            "crystals",
            sa.Column("mode", sa.Integer(), server_default="416", nullable=False),
        )


def downgrade() -> None:
    with op.batch_alter_table("crystals") as batch_op:
        batch_op.drop_column("mode")
        batch_op.drop_column("group_team_id")
        batch_op.drop_column("owner_operator_id")
