"""oauth_records (L2-S5a, 2026-09-24): the OAuth authorization server's
single generic table — mcp-SDK pydantic models persisted as JSON with
hot columns (kind, key, client_id, subject, expires_at, revoked_at)
lifted out for lookups. Backs dynamic client registration, PKCE codes,
and the access/refresh token pairs the MCP door will accept in S5c.
"""
from alembic import op
import sqlalchemy as sa

revision = "f6a0c2d4e6b8"
down_revision = "e5b9c1d3f5a7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "oauth_records",
        sa.Column("kind", sa.String(16), primary_key=True),
        sa.Column("key", sa.String(128), primary_key=True),
        sa.Column("client_id", sa.String(128), nullable=True),
        sa.Column("subject", sa.String(64), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("data", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now()),
    )
    op.create_index("ix_oauth_records_subject", "oauth_records", ["subject"])


def downgrade() -> None:
    op.drop_index("ix_oauth_records_subject", table_name="oauth_records")
    op.drop_table("oauth_records")
