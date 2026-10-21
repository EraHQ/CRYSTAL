"""scan_verdicts (convergence scans' memo, 2026-10-21).

One row per (customer, detector, pair of facts) the contradiction or
dedup scan judged and did NOT turn into a knowledge_conflict. The scans
read it before spending a model call, so a pair is judged once per
version of its claim texts instead of once per idle cycle.
"""
from alembic import op
import sqlalchemy as sa

revision = "d6e8f0a2b4c6"
down_revision = "c5d7e9f1a3b5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "scan_verdicts",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("customer_id", sa.String(length=64), nullable=False),
        sa.Column("detector", sa.String(length=32), nullable=False),
        sa.Column("pair_key", sa.String(length=128), nullable=False),
        sa.Column("verdict", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_scan_verdicts_customer_id", "scan_verdicts", ["customer_id"])
    op.create_index(
        "ux_scan_verdicts_customer_detector_pair", "scan_verdicts",
        ["customer_id", "detector", "pair_key"], unique=True,
    )


def downgrade() -> None:
    op.drop_index("ux_scan_verdicts_customer_detector_pair", table_name="scan_verdicts")
    op.drop_index("ix_scan_verdicts_customer_id", table_name="scan_verdicts")
    op.drop_table("scan_verdicts")
