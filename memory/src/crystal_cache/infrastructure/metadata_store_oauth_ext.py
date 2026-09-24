"""OAuth record CRUD (L2-S5a, 2026-09-24) — D12 mixin.

Four generic verbs over oauth_records; the provider
(control/oauth_provider.py) owns all (de)serialization with the
mcp-SDK pydantic models, so this layer never learns OAuth semantics.
Codes are DELETED on exchange; tokens are REVOKED, never deleted.
"""
from datetime import datetime, timezone
from typing import Any, Optional

from .schema import OAuthRecordRow


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
                kind=kind, key=key, data=data, client_id=client_id,
                subject=subject, expires_at_utc=expires_at,
            ))

    async def oauth_get(self, kind: str, key: str) -> Optional[dict[str, Any]]:
        """The live record, or None if absent or revoked. Expiry is the
        caller's check — the SDK models carry their own expires_at."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(OAuthRecordRow, (kind, key))
            if row is None or row.revoked_at_utc is not None:
                return None
            return {"kind": row.kind, "key": row.key, "data": row.data,
                    "client_id": row.client_id, "subject": row.subject}

    async def oauth_delete(self, kind: str, key: str) -> bool:
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(OAuthRecordRow, (kind, key))
            if row is None:
                return False
            await session.delete(row)
            return True

    async def oauth_revoke(self, kind: str, key: str) -> bool:
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(OAuthRecordRow, (kind, key))
            if row is None or row.revoked_at_utc is not None:
                return False
            row.revoked_at_utc = datetime.now(timezone.utc)
            return True
