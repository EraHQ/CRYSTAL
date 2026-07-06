"""customers.inference_mode + llm_calls.billing (E4 managed inference)

Revision ID: b9c1d3e5f7a8
Revises: a8b0c2d4e6f7
Create Date: 2026-07-06

Accounts Phase B: inference_mode ('byok' back-compat default; 'managed'
= platform-keyed, capped, rebillable) and the per-call billing flag on
the ledger ('managed' = rebill; NULL = byok/internal).
"""
from alembic import op
import sqlalchemy as sa

revision = "b9c1d3e5f7a8"
down_revision = "a8b0c2d4e6f7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "customers",
        sa.Column(
            "inference_mode", sa.String(length=16),
            nullable=False, server_default="byok",
        ),
    )
    op.add_column(
        "llm_calls",
        sa.Column("billing", sa.String(length=16), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("llm_calls", "billing")
    op.drop_column("customers", "inference_mode")
