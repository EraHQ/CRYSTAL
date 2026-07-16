"""cognition_runs.trigger_id — cycles need runs queryable by trigger.

Cognition cycles (ratified 2026-07-16, Q1B/Q2B/Q3A): a later run of the
same trigger must fetch the prior runs' verdicts. trigger_id previously
lived only inside the detail JSON; this promotes it to an indexed
column. Backfill is Python-side (dialect-proof; rows are few) so the
existing needs_human_review pile joins the loop immediately.

Revision ID: e5f7a9b1c3d4
Revises: d4e6f8a0b2c3
Create Date: 2026-07-16
"""
import json

from alembic import op
import sqlalchemy as sa

revision = "e5f7a9b1c3d4"
down_revision = "d4e6f8a0b2c3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "cognition_runs",
        sa.Column("trigger_id", sa.String(length=64), nullable=True),
    )
    op.create_index(
        "ix_cognition_runs_trigger_id", "cognition_runs", ["trigger_id"]
    )

    # Backfill from the detail JSON (env.to_dict() has always carried
    # trigger_id). Python loop keeps this dialect-proof.
    conn = op.get_bind()
    rows = conn.execute(
        sa.text("SELECT id, detail FROM cognition_runs")
    ).fetchall()
    for row_id, detail in rows:
        if detail is None:
            continue
        try:
            d = detail if isinstance(detail, dict) else json.loads(detail)
        except Exception:  # noqa: BLE001 — malformed detail: skip
            continue
        trig = (d or {}).get("trigger_id") or None
        if trig:
            conn.execute(
                sa.text(
                    "UPDATE cognition_runs SET trigger_id = :t "
                    "WHERE id = :i"
                ),
                {"t": str(trig)[:64], "i": row_id},
            )


def downgrade() -> None:
    op.drop_index("ix_cognition_runs_trigger_id", table_name="cognition_runs")
    op.drop_column("cognition_runs", "trigger_id")
