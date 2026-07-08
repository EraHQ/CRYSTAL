"""spend_budgets table + knowledge_gaps.disposition.

Gap Engine redesign S4 (2026-07-08, docs/GAP_ENGINE_AND_LEARN_REDESIGN.md):
the tenant-owned budget SUBSTRATE (one row = one cap for one spend
function, optionally per-operator; the llm_calls ledger's origin stamps
are the meter) and the gap disposition taxonomy (researchable | workable
| needs_document — cheapest capable actor first). The fill sweep is now
budget-gated per tenant via the 'auto_research' function: manual by
default, ratified B-1.
"""
from alembic import op
import sqlalchemy as sa

revision = "d1e3f5a7b9c0"
down_revision = "c0d2e4f6a8b9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "spend_budgets",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("customer_id", sa.String(length=64), nullable=False),
        sa.Column("function", sa.String(length=64), nullable=False),
        sa.Column("operator_id", sa.String(length=64), nullable=True),
        sa.Column("period", sa.String(length=16), nullable=False,
                  server_default="monthly"),
        sa.Column("cap_micro_usd", sa.BigInteger(), nullable=False,
                  server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint(
            "customer_id", "function", "operator_id",
            name="uq_spend_budgets_scope",
        ),
    )
    op.create_index(
        "ix_spend_budgets_customer_id", "spend_budgets", ["customer_id"]
    )
    op.add_column(
        "knowledge_gaps",
        sa.Column("disposition", sa.String(length=32), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("knowledge_gaps", "disposition")
    op.drop_index("ix_spend_budgets_customer_id", table_name="spend_budgets")
    op.drop_table("spend_budgets")
