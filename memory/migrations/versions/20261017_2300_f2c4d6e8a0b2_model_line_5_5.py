"""Q88=A (2026-10-17): managed tenants move to the 5.5 model line.

A MANAGED tenant's stored model id (customers.model_routing_config
JSON, key model_id) that names a previous-generation Anthropic model
is rewritten to its successor, matching control/admission.py's
MANAGED_MODEL_ALIASES. byok tenants are never touched (their key, their
model). Done in Python so it runs on Postgres and SQLite alike.

Revision ID: f2c4d6e8a0b2
Revises: e1b3c5d7f9a1
Create Date: 2026-10-17 23:00:00
"""
from __future__ import annotations

import json

from alembic import op
import sqlalchemy as sa

revision = "f2c4d6e8a0b2"
down_revision = "e1b3c5d7f9a1"
branch_labels = None
depends_on = None

ALIASES = {
    "claude-haiku-4-5": "claude-haiku-5-5",
    "claude-haiku-4-5-20251001": "claude-haiku-5-5",
    "claude-sonnet-5": "claude-sonnet-5-5",
    "claude-sonnet-4-6": "claude-sonnet-5-5",
    "claude-sonnet-4-5-20250929": "claude-sonnet-5-5",
    "claude-opus-5": "claude-opus-5-5",
    "claude-opus-4-8": "claude-opus-5-5",
    "claude-opus-4-7": "claude-opus-5-5",
    "claude-fable-5": "claude-fable-5-1",
}


def _load(raw):
    if raw is None:
        return None
    if isinstance(raw, dict):
        return dict(raw)
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


customers = sa.table(
    "customers",
    sa.column("id", sa.String),
    sa.column("inference_mode", sa.String),
    sa.column("model_routing_config", sa.JSON),
)


def upgrade() -> None:
    bind = op.get_bind()
    rows = bind.execute(
        sa.select(customers.c.id, customers.c.model_routing_config)
        .where(customers.c.inference_mode == "managed")
    ).fetchall()
    for cid, raw in rows:
        cfg = _load(raw)
        if not cfg:
            continue
        target = ALIASES.get(cfg.get("model_id"))
        if not target:
            continue
        cfg["model_id"] = target
        # The JSON type serializes for the dialect (Postgres json, SQLite text).
        bind.execute(
            customers.update().where(customers.c.id == cid).values(model_routing_config=cfg)
        )


def downgrade() -> None:
    # The old ids still exist upstream; nothing to reverse that would be
    # safer than leaving the successor in place.
    pass
