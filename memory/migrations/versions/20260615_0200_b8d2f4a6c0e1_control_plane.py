"""control plane

Revision ID: b8d2f4a6c0e1
Revises: f2a4c6e8b0d1
Create Date: 2026-06-15 02:00:00.000000

Growth G2 (control plane) — the outbound-poll command channel. Creates one
table:

  control_commands — one row per operator command targeting a session: an
                     approval decision (approve/deny an agent's approval
                     gate) or a terminate (session / dependency). The agent
                     POLLS for pending commands on its session and acts;
                     nothing connects inbound (NAT-safe). The decision is
                     SIGNED by the operator's key (signature/nonce/signed_at)
                     and the AGENT verifies it against the pinned public key
                     before acting — the server is a courier that cannot
                     forge. First-wins is a compare-and-set on `status`
                     (pending→consumed); F4 staleness voids a crashed
                     session's pending commands.

CREATE TABLE (not ALTER) so the SQLite ALTER-ADD-FK quirks don't apply.
Following the F4/G1 precedent, columns are PLAIN (no FK constraints) — SQLite
doesn't enforce FKs and the joins are application-level; the ORM model keeps
the customer_id FK declaration for Postgres + documentation. session_id is a
soft pointer (a session row can be swept/removed; a FK would dangle).

Idempotent on upgrade: the create is skipped if the table already exists (a
local dev DB may have picked it up via store.init()'s create_all before this
migration ran). Downgrade drops the index then the table.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b8d2f4a6c0e1'
down_revision: Union[str, None] = 'f2a4c6e8b0d1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    existing_tables = set(sa.inspect(bind).get_table_names())

    if "control_commands" not in existing_tables:
        op.create_table(
            "control_commands",
            sa.Column("id", sa.String(length=64), nullable=False),
            sa.Column("session_id", sa.String(length=64), nullable=False),
            sa.Column("customer_id", sa.String(length=64), nullable=False),
            sa.Column("request_id", sa.String(length=64), nullable=False),
            sa.Column("command_type", sa.String(length=32), nullable=False),
            sa.Column("decision", sa.String(length=16), nullable=True),
            sa.Column("dependency_id", sa.String(length=64), nullable=True),
            sa.Column("payload", sa.JSON(), nullable=True),
            sa.Column("signature", sa.Text(), nullable=True),
            sa.Column("nonce", sa.String(length=128), nullable=True),
            sa.Column("signed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column(
                "issued_by_operator_id", sa.String(length=64), nullable=True
            ),
            sa.Column(
                "status",
                sa.String(length=16),
                nullable=False,
                server_default="pending",
            ),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "ix_control_commands_session_status",
            "control_commands",
            ["session_id", "status"],
        )


def downgrade() -> None:
    op.drop_index(
        "ix_control_commands_session_status", table_name="control_commands"
    )
    op.drop_table("control_commands")
