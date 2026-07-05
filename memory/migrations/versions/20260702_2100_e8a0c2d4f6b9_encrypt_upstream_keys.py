"""encrypt_upstream_keys

Revision ID: e8a0c2d4f6b9
Revises: d7f9b1c3e5a8
Create Date: 2026-07-02 21:00:00.000000

Launch-prep security pass — Key B (the customer's upstream provider API
key, stored inside the customers.model_routing_config JSON as
api_key_ref) becomes AES-256-GCM encrypted at rest in the enc:v1
composite format. The store writers encrypt unconditionally from this
release on; this DATA migration encrypts every existing plaintext value
so the strict decrypt-only reader never meets legacy plaintext.

FAILS LOUDLY BY DESIGN: if plaintext keys exist and
CC_TOKEN_ENCRYPTION_KEY is not set, the migration raises instead of
skipping — a deployment must never come out of an upgrade with secrets
still in plaintext. Empty refs and already-encrypted refs are skipped.

Downgrade is a deliberate no-op: decrypting secrets back to plaintext
is a security regression this migration refuses to automate.
"""
from __future__ import annotations

import json
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e8a0c2d4f6b9'
down_revision: Union[str, None] = 'd7f9b1c3e5a8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from crystal_cache.infrastructure.token_crypto import (
        encrypt_secret,
        is_encrypted,
    )

    bind = op.get_bind()
    if "customers" not in sa.inspect(bind).get_table_names():
        return  # fresh DB — nothing to migrate

    rows = bind.execute(
        sa.text("SELECT id, model_routing_config FROM customers")
    ).fetchall()

    pending: list[tuple[str, dict]] = []
    for row_id, cfg in rows:
        config = json.loads(cfg) if isinstance(cfg, str) else (cfg or {})
        ref = config.get("api_key_ref") or ""
        if not ref or is_encrypted(ref):
            continue
        pending.append((row_id, config))

    if not pending:
        return

    # encrypt_secret raises with a clear message when
    # CC_TOKEN_ENCRYPTION_KEY is missing — exactly the fail-loud behavior
    # this migration wants when plaintext exists.
    for row_id, config in pending:
        config["api_key_ref"] = encrypt_secret(config["api_key_ref"])
        bind.execute(
            sa.text(
                "UPDATE customers SET model_routing_config = :cfg "
                "WHERE id = :id"
            ),
            {"cfg": json.dumps(config), "id": row_id},
        )


def downgrade() -> None:
    # Deliberate no-op: automated decryption back to plaintext at rest is a
    # security regression. Encrypted values remain readable by the app as
    # long as CC_TOKEN_ENCRYPTION_KEY is set.
    pass
