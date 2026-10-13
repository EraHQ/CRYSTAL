"""OAuth record CRUD (L2-S5a, 2026-09-24) — D12 mixin.

Four generic verbs over oauth_records; the provider
(control/oauth_provider.py) owns all (de)serialization with the
mcp-SDK pydantic models, so this layer never learns OAuth semantics.
Codes are DELETED on exchange; tokens are REVOKED, never deleted.

B4-7 (Lockdown PR-7, 2026-10-13): the secret kinds — access tokens,
refresh tokens, authorization codes — are stored under
`hash_api_key(secret)` (HMAC-SHA256 with the deployment pepper, the
same one-way hash Key A and operator keys use), never under the raw
value. A DB read yields no live token. The provider keeps the raw
value out of `data` too; the migration c9f1a3b5d7e9 hashed the rows
written before this. Clients are public metadata and keep their id.
"""
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import func, select

from .credentials import hash_api_key
from .schema import OAuthRecordRow

# The kinds whose key is a bearer secret (hashed at rest).
SECRET_KINDS = frozenset({"access", "refresh", "code"})


def oauth_storage_key(kind: str, key: str) -> str:
    """The primary-key value a record of `kind` is stored under."""
    return hash_api_key(key) if kind in SECRET_KINDS else key


_RAW_PREFIXES = ("cco_", "ccr_", "ccc_")
_SECRET_FIELD = {"access": "token", "refresh": "token", "code": "code"}


def rehash_plaintext_oauth_secrets(conn: Any) -> int:
    """Convert the rows written before B4-7 (sync connection; called by
    migration c9f1a3b5d7e9 and by its pin): every secret-kind row whose
    key still carries the raw prefix is re-keyed under its hash and its
    `data` loses the secret field. Returns the number converted. A row
    already hashed is left alone, so a re-run is harmless."""
    import json

    from sqlalchemy import text

    rows = conn.execute(text(
        "SELECT kind, key, data FROM oauth_records "
        "WHERE kind IN ('access', 'refresh', 'code')"
    )).fetchall()
    converted = 0
    for kind, key, data in rows:
        if not key.startswith(_RAW_PREFIXES):
            continue
        new_data = data
        try:
            payload = json.loads(data)
        except (TypeError, ValueError):
            payload = None
        if isinstance(payload, dict):
            payload.pop(_SECRET_FIELD[kind], None)
            new_data = json.dumps(payload)
        conn.execute(
            text(
                "UPDATE oauth_records SET key = :new_key, data = :new_data "
                "WHERE kind = :kind AND key = :old_key"
            ),
            {"new_key": hash_api_key(key), "new_data": new_data, "kind": kind, "old_key": key},
        )
        converted += 1
    return converted


class OAuthExtensionsMixin:
    async def oauth_put(
        self,
        kind: str,
        key: str,
        data: str,
        client_id: Optional[str] = None,
        subject: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ) -> None:
        async with self.session() as session:  # type: ignore[attr-defined]
            session.add(OAuthRecordRow(
                kind=kind, key=oauth_storage_key(kind, key), data=data,
                client_id=client_id, subject=subject, expires_at_utc=expires_at,
            ))

    async def oauth_get(self, kind: str, key: str) -> Optional[dict[str, Any]]:
        """The live record, or None if absent or revoked. Expiry is the
        caller's check — the SDK models carry their own expires_at. The
        returned `key` is the STORED key (the hash for a secret kind)."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(OAuthRecordRow, (kind, oauth_storage_key(kind, key)))
            if row is None or row.revoked_at_utc is not None:
                return None
            return {"kind": row.kind, "key": row.key, "data": row.data,
                    "client_id": row.client_id, "subject": row.subject}

    async def oauth_delete(self, kind: str, key: str) -> bool:
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(OAuthRecordRow, (kind, oauth_storage_key(kind, key)))
            if row is None:
                return False
            await session.delete(row)
            return True

    async def oauth_revoke(self, kind: str, key: str) -> bool:
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(OAuthRecordRow, (kind, oauth_storage_key(kind, key)))
            if row is None or row.revoked_at_utc is not None:
                return False
            row.revoked_at_utc = datetime.now(timezone.utc)
            return True

    async def oauth_count(self, kind: str) -> int:
        """How many records of `kind` exist (B4-9: the client cap)."""
        async with self.session() as session:  # type: ignore[attr-defined]
            n = await session.scalar(
                select(func.count()).select_from(OAuthRecordRow).where(OAuthRecordRow.kind == kind)
            )
            return int(n or 0)
