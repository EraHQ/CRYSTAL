"""oauth_records: secrets hashed at rest (Lockdown PR-7, B4-7, 2026-10-13).

Access tokens (cco_), refresh tokens (ccr_) and authorization codes
(ccc_) were stored under their raw value as the primary key and again
inside `data`; a database read yielded live 30-day refresh tokens. From
this revision the store keys those kinds by `hash_api_key(secret)` (the
one-way keyed hash Key A and operator keys use) and the provider keeps
the secret field out of `data`.

This migration converts the rows written before it (the shared helper
`rehash_plaintext_oauth_secrets`, pinned in tests/test_lockdown_pr7.py):
every access / refresh / code row whose key still carries the raw
prefix is re-keyed under its hash and its `data` loses the `token` /
`code` field. Live sessions keep working: the presented token hashes
to the new key. Client rows (public metadata) are untouched.

Downgrade is a no-op: a hash cannot be turned back into the token. The
pre-revision code would simply not find the hashed rows, which revokes
every session in effect (clients re-connect).
"""
from alembic import op

revision = "c9f1a3b5d7e9"
down_revision = "b8e0f2a4c6d8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    from crystal_cache.infrastructure.metadata_store_oauth_ext import (
        rehash_plaintext_oauth_secrets,
    )

    rehash_plaintext_oauth_secrets(op.get_bind())


def downgrade() -> None:
    # Irreversible by construction (see the module docstring).
    pass
