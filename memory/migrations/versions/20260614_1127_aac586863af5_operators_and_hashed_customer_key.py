"""operators_and_hashed_customer_key

Revision ID: aac586863af5
Revises: c3a91d7e4f02
Create Date: 2026-06-14 11:27:24.899281

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'aac586863af5'
down_revision: Union[str, None] = 'c3a91d7e4f02'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- customers.api_key (plaintext) -> api_key_hash (hashed) ---
    # No plaintext at rest. Existing customers are PRESERVED: add the hash
    # column nullable, hash each current key IN PLACE (so a customer keeps
    # authenticating with the key they already hold), then drop the
    # plaintext column and lock the hash column down. SQLite needs a table
    # rebuild for drop-column / NOT-NULL / unique, so those run in a batch
    # block. An empty customers table makes the data loop a no-op.
    from crystal_cache.infrastructure.credentials import hash_api_key

    op.add_column(
        'customers',
        sa.Column('api_key_hash', sa.String(length=128), nullable=True),
    )

    bind = op.get_bind()
    rows = bind.execute(
        sa.text("SELECT id, api_key FROM customers")
    ).mappings().all()
    for row in rows:
        bind.execute(
            sa.text("UPDATE customers SET api_key_hash = :h WHERE id = :id"),
            {"h": hash_api_key(row["api_key"]), "id": row["id"]},
        )

    with op.batch_alter_table('customers') as batch_op:
        batch_op.alter_column('api_key_hash', nullable=False)
        # Batch mode requires a named constraint (unlike op-level, which
        # accepts None). The name is harmless on SQLite and keeps the
        # rebuild deterministic.
        batch_op.create_unique_constraint('uq_customers_api_key_hash', ['api_key_hash'])
        batch_op.drop_column('api_key')

    # --- Operators table (Foundation F1) --- created after customers is
    # finalized so its team_id FK points at the rebuilt customers table.
    op.create_table(
        'operators',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('team_id', sa.String(length=64), nullable=False),
        sa.Column('display_name', sa.String(length=256), nullable=False),
        sa.Column('role', sa.String(length=32), server_default='operator', nullable=False),
        sa.Column('status', sa.String(length=32), server_default='active', nullable=False),
        sa.Column('api_key_hash', sa.String(length=128), nullable=True),
        sa.Column('credential_public_key', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['team_id'], ['customers.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('api_key_hash'),
    )
    op.create_index(op.f('ix_operators_team_id'), 'operators', ['team_id'], unique=False)


def downgrade() -> None:
    # Hashes can't be reversed — the original plaintext keys are NOT
    # recoverable. Re-add api_key as a nullable empty column and drop the
    # hash column (its unique constraint goes with it in the rebuild);
    # any restored customers would need fresh keys issued.
    op.drop_index(op.f('ix_operators_team_id'), table_name='operators')
    op.drop_table('operators')
    with op.batch_alter_table('customers') as batch_op:
        batch_op.add_column(sa.Column('api_key', sa.VARCHAR(length=128), nullable=True))
        batch_op.drop_column('api_key_hash')
