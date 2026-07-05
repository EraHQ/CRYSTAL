"""P3 groups: named sub-teams as grant targets.

Revision ID: a1c3e5b7d9f2
Revises: f9b1d3e5a7c0
Create Date: 2026-07-03

Groups (ratified 2026-07-02) are lightweight grant targets: a
crystal_acls row with principal_type 'group' lets every member read the
crystal without touching its POSIX mode. crystal_acls itself needs no
change (String-over-enum vocabulary). Downgrade drops both tables.
"""
from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "a1c3e5b7d9f2"
down_revision: Union[str, None] = "f9b1d3e5a7c0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "groups",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "customer_id", sa.String(64),
            sa.ForeignKey("customers.id"), nullable=False,
        ),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ux_groups_customer_name", "groups",
        ["customer_id", "name"], unique=True,
    )
    op.create_table(
        "group_members",
        sa.Column(
            "group_id", sa.String(64),
            sa.ForeignKey("groups.id"), primary_key=True,
        ),
        sa.Column(
            "operator_id", sa.String(64),
            sa.ForeignKey("operators.id"), primary_key=True,
        ),
    )
    op.create_index(
        "ix_group_members_operator", "group_members", ["operator_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_group_members_operator", table_name="group_members")
    op.drop_table("group_members")
    op.drop_index("ux_groups_customer_name", table_name="groups")
    op.drop_table("groups")
