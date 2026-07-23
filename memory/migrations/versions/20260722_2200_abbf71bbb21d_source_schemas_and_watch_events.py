"""source schemas + source watch events

Revision ID: abbf71bbb21d
Revises: 67e2858d85c6
Create Date: 2026-07-22

Gate G slice 1 (design closed 2026-07-22: G-Q1=B record-window
fragments, G-Q2=A status-column review surface, G-Q3=A awaiting_schema
parking, G-Q4=A events table fully wired). Batched per the 2026-07-21
ruling: source_schemas (C5's judgment-once-mechanism-forever registry)
+ source_watch_events (the watcher's durable activity feed) + the
document_uploads.source_schema_hash parking column ride ONE migration.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'abbf71bbb21d'
down_revision: Union[str, None] = '67e2858d85c6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'source_schemas',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column(
            'customer_id', sa.String(length=64),
            sa.ForeignKey('customers.id'), nullable=False,
        ),
        # C5 fingerprint: sha256 over sorted key-paths + JSON types.
        sa.Column('schema_hash', sa.String(length=64), nullable=False),
        # The mapping spec the inference call produced; mechanically
        # executable per record, editable forward (G2/G3).
        sa.Column('mapping', sa.JSON(), nullable=False),
        # G-Q2=A: this column IS the review queue.
        sa.Column('status', sa.String(length=32), nullable=False,
                  server_default='proposed'),
        # Sample records for the proposal preview (rendered THROUGH
        # the mapping in the Inspector — judge the output, not the spec).
        sa.Column('sample', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'customer_id', 'schema_hash', name='uq_source_schemas_cust_hash',
        ),
    )
    op.create_index(
        'ix_source_schemas_customer_id', 'source_schemas', ['customer_id'],
    )

    op.create_table(
        'source_watch_events',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column(
            'customer_id', sa.String(length=64),
            sa.ForeignKey('customers.id'), nullable=False,
        ),
        sa.Column('watch_id', sa.String(length=64), nullable=False),
        # String-backed vocabulary; one home in the sync worker.
        sa.Column('event_type', sa.String(length=64), nullable=False),
        sa.Column('label', sa.String(length=256), nullable=False,
                  server_default=''),
        sa.Column('payload', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_source_watch_events_customer_id',
        'source_watch_events', ['customer_id'],
    )
    op.create_index(
        'ix_source_watch_events_watch_created',
        'source_watch_events', ['watch_id', 'created_at'],
    )

    # G-Q3=A parking: awaiting_schema documents record WHICH shape they
    # wait on, so approval releases them all in one update.
    op.add_column(
        'document_uploads',
        sa.Column('source_schema_hash', sa.String(length=64), nullable=True),
    )
    op.create_index(
        'ix_document_uploads_source_schema_hash',
        'document_uploads', ['source_schema_hash'],
    )


def downgrade() -> None:
    op.drop_index(
        'ix_document_uploads_source_schema_hash',
        table_name='document_uploads',
    )
    op.drop_column('document_uploads', 'source_schema_hash')
    op.drop_index(
        'ix_source_watch_events_watch_created',
        table_name='source_watch_events',
    )
    op.drop_index(
        'ix_source_watch_events_customer_id',
        table_name='source_watch_events',
    )
    op.drop_table('source_watch_events')
    op.drop_index(
        'ix_source_schemas_customer_id', table_name='source_schemas',
    )
    op.drop_table('source_schemas')
