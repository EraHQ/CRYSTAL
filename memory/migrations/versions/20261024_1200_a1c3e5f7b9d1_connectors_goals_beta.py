"""Connectors, Goals and the Beta list (2026-10-24).

beta_features / beta_invites: private invites to features (Gmail first).
app_connections: a person's apps reached through Nango, with what Crystal
may do with each (watch / remember / act) and the rules for remembering.
watched_items: what the apps showed, kept 7 days.
goals / goal_items: standing directives, their plans, and what they made.
The default beta features are seeded in code (beta.ensure_defaults), not
here, so a fresh test database and production read the same list.
"""
from alembic import op
import sqlalchemy as sa

revision = "a1c3e5f7b9d1"
down_revision = "f8a0b2c4d6e8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "beta_features",
        sa.Column("key", sa.String(length=48), primary_key=True),
        sa.Column("title", sa.String(length=120), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="invite"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_table(
        "beta_invites",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("feature_key", sa.String(length=48), nullable=False),
        sa.Column("customer_id", sa.String(length=64), nullable=True),
        sa.Column("email", sa.String(length=320), nullable=True),
        sa.Column("invited_by", sa.String(length=320), nullable=False, server_default=""),
        sa.Column("note", sa.String(length=256), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_beta_invites_feature_key", "beta_invites", ["feature_key"])
    op.create_index("ix_beta_invites_customer_id", "beta_invites", ["customer_id"])
    op.create_index("ix_beta_invites_email", "beta_invites", ["email"])

    op.create_table(
        "app_connections",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("customer_id", sa.String(length=64), nullable=False),
        sa.Column("integration", sa.String(length=64), nullable=False),
        sa.Column("nango_connection_id", sa.String(length=128), nullable=False),
        sa.Column("account_label", sa.String(length=320), nullable=False, server_default=""),
        sa.Column("status", sa.String(length=24), nullable=False, server_default="active"),
        sa.Column("uses", sa.JSON(), nullable=False),
        sa.Column("remember_rules", sa.JSON(), nullable=False),
        sa.Column("watch_cadence_minutes", sa.Integer(), nullable=False, server_default="15"),
        sa.Column("cursors", sa.JSON(), nullable=False),
        sa.Column("sync_due", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("last_synced_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("created_by_operator_id", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("customer_id", "nango_connection_id", name="ux_app_connections_customer_nango"),
    )
    op.create_index("ix_app_connections_customer_id", "app_connections", ["customer_id"])

    op.create_table(
        "watched_items",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("customer_id", sa.String(length=64), nullable=False),
        sa.Column("connection_id", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("external_id", sa.String(length=256), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False, server_default=""),
        sa.Column("author", sa.String(length=320), nullable=False, server_default=""),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("url", sa.String(length=1024), nullable=True),
        sa.Column("snippet", sa.Text(), nullable=False, server_default=""),
        sa.Column("body", sa.Text(), nullable=False, server_default=""),
        sa.Column("meta", sa.JSON(), nullable=False),
        sa.Column("remembered_document_id", sa.String(length=64), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("connection_id", "kind", "external_id", name="ux_watched_items_conn_kind_ext"),
    )
    op.create_index("ix_watched_items_customer_id", "watched_items", ["customer_id"])
    op.create_index("ix_watched_items_connection_id", "watched_items", ["connection_id"])
    op.create_index("ix_watched_items_customer_occurred", "watched_items", ["customer_id", "occurred_at"])
    op.create_index("ix_watched_items_expires", "watched_items", ["expires_at"])

    op.create_table(
        "goals",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("customer_id", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("directive", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="draft"),
        sa.Column("plan", sa.JSON(), nullable=True),
        sa.Column("plan_model", sa.String(length=128), nullable=True),
        sa.Column("planned_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_task_id", sa.String(length=64), nullable=True),
        sa.Column("last_summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("may_send", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("may_post", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_by_operator_id", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_goals_customer_id", "goals", ["customer_id"])
    op.create_index("ix_goals_next_run_at", "goals", ["next_run_at"])

    op.create_table(
        "goal_items",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("goal_id", sa.String(length=64), nullable=False),
        sa.Column("customer_id", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False, server_default=""),
        sa.Column("body", sa.Text(), nullable=False, server_default=""),
        sa.Column("watched_item_id", sa.String(length=64), nullable=True),
        sa.Column("target", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="open"),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_goal_items_goal_id", "goal_items", ["goal_id"])
    op.create_index("ix_goal_items_customer_id", "goal_items", ["customer_id"])
    op.create_index("ix_goal_items_goal_status", "goal_items", ["goal_id", "status"])


def downgrade() -> None:
    op.drop_index("ix_goal_items_goal_status", table_name="goal_items")
    op.drop_index("ix_goal_items_customer_id", table_name="goal_items")
    op.drop_index("ix_goal_items_goal_id", table_name="goal_items")
    op.drop_table("goal_items")
    op.drop_index("ix_goals_next_run_at", table_name="goals")
    op.drop_index("ix_goals_customer_id", table_name="goals")
    op.drop_table("goals")
    op.drop_index("ix_watched_items_expires", table_name="watched_items")
    op.drop_index("ix_watched_items_customer_occurred", table_name="watched_items")
    op.drop_index("ix_watched_items_connection_id", table_name="watched_items")
    op.drop_index("ix_watched_items_customer_id", table_name="watched_items")
    op.drop_table("watched_items")
    op.drop_index("ix_app_connections_customer_id", table_name="app_connections")
    op.drop_table("app_connections")
    op.drop_index("ix_beta_invites_email", table_name="beta_invites")
    op.drop_index("ix_beta_invites_customer_id", table_name="beta_invites")
    op.drop_index("ix_beta_invites_feature_key", table_name="beta_invites")
    op.drop_table("beta_invites")
    op.drop_table("beta_features")
