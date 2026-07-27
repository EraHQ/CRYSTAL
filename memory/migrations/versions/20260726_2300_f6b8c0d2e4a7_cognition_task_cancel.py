"""cognition_tasks: cancel_requested — cooperative cancellation

Revision ID: f6b8c0d2e4a7
Revises: e5a7b9c1d3f6
Create Date: 2026-07-26

One additive column, born from a 40-minute hang (2026-07-26).

Two problems surfaced together; only one of them needed schema.

  NOT this migration — the reclaim half needed no column at all.
  cognition_runs.updated_at has existed since S9 (2026-07-08) with
  onupdate=utcnow, so every one of the engine's seven lifecycle
  transitions has been stamping a heartbeat all along. The defect was
  that nothing READ it: requeue_task 409s on `running` without
  consulting it, and claim_pending_cognition_task filters
  status == 'pending', so a task whose executor was replaced mid-run
  (an api+worker deploy) stayed `running` forever — the one state that
  needed reclaiming was the one state nothing could reclaim. That fix
  is pure logic against data we already had.

  THIS migration — cancel_requested.
  Cooperative cancellation genuinely needs somewhere to put the
  request. The coding agent's queue has had cancel_agent_task since
  the backlog work, but a RUNNING non-recurring task there returns
  `running_uncancelable` because nothing checks between units of
  work. Cognition has step boundaries, so it can honor a request: the
  engine reads this flag at each boundary and stops cleanly rather
  than being killed mid-LLM-call. Operators need it because a run
  visibly going off the rails currently costs full price to watch.

Boolean, NOT NULL, server_default false: existing rows read as
not-cancelled, which is correct. No backfill, no data movement.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f6b8c0d2e4a7'
down_revision: Union[str, None] = 'e5a7b9c1d3f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('cognition_tasks') as batch_op:
        batch_op.add_column(
            sa.Column(
                'cancel_requested',
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            )
        )


def downgrade() -> None:
    with op.batch_alter_table('cognition_tasks') as batch_op:
        batch_op.drop_column('cancel_requested')
