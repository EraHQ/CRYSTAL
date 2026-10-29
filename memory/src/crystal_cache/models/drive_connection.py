"""Drive OAuth connections.

A DriveConnection holds the OAuth credentials for one customer's
Google Drive account. (WatchedFolder / WatchedFile retired 2026-07-24,
DRIVE-Q1=B: a watched Drive folder is a normal source_watch of
scheme=gdrive; DriveSourceHandler resolves tokens through the
connection per poll.)

Refresh tokens are stored encrypted (enc:v2 under the tenant DEK).
The Pydantic model holds the ciphertext + nonce verbatim; decryption
happens at use-time in the drive connector, not at model
construction.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


# 'active' | 'expired' | 'revoked' | 'error'.
# 'expired' is set by the drive connector when the refresh token call
# fails with 401; 'revoked' is set when the user disconnects from the
# customer-facing UI; 'error' is a soft state for non-auth failures
# (rate limit, transient network) that the sync worker retries.
DriveConnectionStatus = Literal["active", "expired", "revoked", "error"]


class DriveConnection(BaseModel):
    """OAuth connection to a customer's Google Drive."""

    id: str
    customer_id: str
    provider: str = "google_drive"
    email: Optional[str] = None

    # AES-256-GCM ciphertext + nonce. Decryption requires
    # CC_TOKEN_ENCRYPTION_KEY env var; see infrastructure/token_crypto.py.
    encrypted_refresh_token: str
    token_nonce: str

    scopes: Optional[str] = None
    status: DriveConnectionStatus = "active"
    last_synced_at: Optional[datetime] = None
    error_message: Optional[str] = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
