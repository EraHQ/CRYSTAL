"""drop watched_folders + watched_files — DRIVE-Q1=B

Revision ID: d4f6a8b0c2e4
Revises: abbf71bbb21d
Create Date: 2026-07-24

Drive unification (DRIVE-Q1=B, ratified 2026-07-24): a watched Drive
folder is a normal source_watch (scheme=gdrive) synced by
DriveSourceHandler under the one sync loop, so the legacy drive_sync
worker and its two tables retire together. drive_connections and
oauth_states STAY — OAuth credentials remain the substrate the gdrive
handler resolves tokens through per poll.

Downgrade recreates both tables exactly as the baseline
(81cf707a1672) created them.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4f6a8b0c2e4'
down_revision: Union[str, None] = 'abbf71bbb21d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_table('watched_folders')
    op.drop_table('watched_files')


def downgrade() -> None:
    op.create_table('watched_files',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('connection_id', sa.String(length=64), nullable=False),
    sa.Column('customer_id', sa.String(length=64), nullable=False),
    sa.Column('file_id', sa.String(length=256), nullable=False),
    sa.Column('file_name', sa.String(length=512), nullable=False),
    sa.Column('mime_type', sa.String(length=256), nullable=True),
    sa.Column('contains_phi', sa.Boolean(), nullable=False),
    sa.Column('sync_interval_minutes', sa.Integer(), nullable=False),
    sa.Column('last_checked_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('last_modified_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('status', sa.String(length=32), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['connection_id'], ['drive_connections.id'], ),
    sa.ForeignKeyConstraint(['customer_id'], ['customers.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('watched_folders',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('connection_id', sa.String(length=64), nullable=False),
    sa.Column('customer_id', sa.String(length=64), nullable=False),
    sa.Column('folder_id', sa.String(length=256), nullable=False),
    sa.Column('folder_name', sa.String(length=512), nullable=False),
    sa.Column('folder_path', sa.Text(), nullable=True),
    sa.Column('contains_phi', sa.Boolean(), nullable=False),
    sa.Column('sync_interval_minutes', sa.Integer(), nullable=False),
    sa.Column('last_checked_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('last_file_count', sa.Integer(), nullable=True),
    sa.Column('status', sa.String(length=32), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['connection_id'], ['drive_connections.id'], ),
    sa.ForeignKeyConstraint(['customer_id'], ['customers.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
