"""entities

Revision ID: 67e2858d85c6
Revises: b8d0f2a4c6e8
Create Date: 2026-07-22

Entities layer (design gate 2026-07-22, SESSION_HANDOFF 0c). The
registry that makes people and orgs DETERMINISTICALLY resolvable to
their dedicated crystals — name/alias lookup, never vector similarity.
One pointer mechanism (Q5A): crystal_id here; no column on operators.
Cold table; standalone migration ruled OK at the gate.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '67e2858d85c6'
down_revision: Union[str, None] = 'b8d0f2a4c6e8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'entities',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column(
            'customer_id', sa.String(length=64),
            sa.ForeignKey('customers.id'), nullable=False,
        ),
        # String-backed kind ('person' default) — future kinds land
        # without an Alembic step, matching the codebase convention.
        sa.Column('kind', sa.String(length=32), nullable=False,
                  server_default='person'),
        sa.Column('display_name', sa.String(length=256), nullable=False),
        # JSON list of exact-match alternates ("Maria" -> Maria Lopez).
        sa.Column('aliases', sa.JSON(), nullable=False),
        # The ONE pointer to the entity's dedicated crystal (Q5A).
        # Nullable: creation is lazy so read paths stay side-effect free.
        sa.Column('crystal_id', sa.String(length=64), nullable=True),
        # F1 link when this entity IS an operator; NULL for mentioned
        # third parties. An operator's entity (and crystal) survives
        # suspension by design, matching operators' own posture.
        sa.Column('operator_id', sa.String(length=64),
                  sa.ForeignKey('operators.id'), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_entities_customer_id', 'entities', ['customer_id'],
    )
    op.create_index(
        'ix_entities_operator_id', 'entities', ['operator_id'],
    )


def downgrade() -> None:
    op.drop_index('ix_entities_operator_id', table_name='entities')
    op.drop_index('ix_entities_customer_id', table_name='entities')
    op.drop_table('entities')
