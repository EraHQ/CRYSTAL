"""enc:v2 cutover — retire orphaned v1 ciphertext.

P4 (2026-07-10). The census found exactly one enc:v1 Key B row (a test
customer whose ciphertext was orphaned when a --set-env-vars deploy
wiped CC_TOKEN_ENCRYPTION_KEY) and Drive has been disconnected since
2026-06-13, so any drive_connections rows hold equally unrecoverable
tokens. Both are noise: null the refs, truncate the connections. New
secrets are enc:v2 (per-tenant DEK, AAD-bound) from this revision on.

Data migration is Python-side JSON editing (portable across
SQLite/Postgres — the routing config is a JSON column, and
dialect-specific json_set/jsonb_set would fork the code).
"""
import json

import sqlalchemy as sa
from alembic import op

revision = "b2c4d6e8f0a1"
down_revision = "a1b3c5d7e9f0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    rows = conn.execute(sa.text(
        "SELECT id, model_routing_config FROM customers"
    )).fetchall()
    for cid, cfg in rows:
        data = cfg if isinstance(cfg, dict) else json.loads(cfg or "{}")
        ref = (data or {}).get("api_key_ref") or ""
        if ref.startswith("enc:v1:"):
            data["api_key_ref"] = ""
            conn.execute(
                sa.text(
                    "UPDATE customers SET model_routing_config = :cfg "
                    "WHERE id = :cid"
                ),
                {"cfg": json.dumps(data), "cid": cid},
            )
    conn.execute(sa.text("DELETE FROM drive_connections"))


def downgrade() -> None:
    # Data-destructive by design: the nulled refs were unrecoverable
    # ciphertext (wiped key). Nothing to restore.
    pass
