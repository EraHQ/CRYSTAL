"""system_rules table — user-owned judgment automation.

Revision ID: c4e6f8a0b2d3
Revises: b3d5f7a9c1e2
Create Date: 2026-07-03

The generic per-tenant rules table (2026-07-03). Storage is schema-free
(JSON selector/conditions/action) so new rule_types are code + validation,
not migrations; execution is typed per rule_type. First rule_type shipped:
'promotion' (clears the recall gate on background-worker memory).
"""
from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "c4e6f8a0b2d3"
down_revision: Union[str, None] = "b3d5f7a9c1e2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "system_rules",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "customer_id", sa.String(64),
            sa.ForeignKey("customers.id"), nullable=False,
        ),
        sa.Column("rule_type", sa.String(32), nullable=False),
        sa.Column(
            "enabled", sa.Boolean(), nullable=False, server_default="1",
        ),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("selector", sa.JSON(), nullable=False),
        sa.Column("conditions", sa.JSON(), nullable=False),
        sa.Column("action", sa.JSON(), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False, server_default="100"),
        sa.Column("last_fired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("fire_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "idx_system_rules_customer_type", "system_rules",
        ["customer_id", "rule_type"],
    )
    op.create_index(
        "ix_system_rules_rule_type", "system_rules", ["rule_type"],
    )


def downgrade() -> None:
    op.drop_index("ix_system_rules_rule_type", table_name="system_rules")
    op.drop_index("idx_system_rules_customer_type", table_name="system_rules")
    op.drop_table("system_rules")
