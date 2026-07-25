"""operators: email + user_id — Team v2 (TEAM-Q1=A)

Revision ID: e5a7b9c1d3f6
Revises: d4f6a8b0c2e4
Create Date: 2026-07-24

Operators gain the two account-shaped columns (2026-07-24):

  email    — informational now; the invitation key at the future
             invitations/permissions gate.
  user_id  — the Firebase uid (users.id), linking the agent-facing
             operator to the login-facing user. Nullable: operators
             may exist before their human ever signs in. The default
             admin gets linked in place via the Team card's profile
             edit — login identity and agent identity finally meet.

No FK constraint by choice: users.id is Firebase-owned and the
linkage is advisory until the permissions gate hardens it.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e5a7b9c1d3f6'
down_revision: Union[str, None] = 'd4f6a8b0c2e4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('operators') as batch_op:
        batch_op.add_column(
            sa.Column('email', sa.String(length=320), nullable=True)
        )
        batch_op.add_column(
            sa.Column('user_id', sa.String(length=128), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table('operators') as batch_op:
        batch_op.drop_column('user_id')
        batch_op.drop_column('email')
