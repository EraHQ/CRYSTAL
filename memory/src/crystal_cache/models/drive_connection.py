"""Drive sync — OAuth connections, watched folders, watched files.

Three Pydantic models in one file because they're tightly coupled:
a DriveConnection holds the OAuth credentials for one customer's
Google Drive account; WatchedFolder and WatchedFile track what the
sync worker monitors inside that connection.

Refresh tokens are stored AES-256-GCM encrypted via the token_crypto
module. The Pydantic model holds the ciphertext + nonce verbatim;
decryption happens at use-time in the drive connector, not at model
construction.

PHI tracking: contains_phi is a customer-controlled flag on folder
and file rows. When True, all retrievals from the resource go through
the PHI access log (phi_access_log table; dict-shaped per D1, no
Pydantic model). The flag drives compliance posture, not access
control — ACLs are the access control mechanism.
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

# 'active' | 'paused' | 'error'.
# 'paused' is operator-controlled (the inspector can pause sync on
# a folder without revoking the connection). 'error' is set when the
# sync worker hits a persistent failure on that specific folder/file.
WatchedResourceStatus = Literal["active", "paused", "error"]


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


class WatchedFolder(BaseModel):
    """A Google Drive folder being monitored for new/changed files."""

    id: str
    connection_id: str
    customer_id: str

    folder_id: str
    folder_name: str
    folder_path: Optional[str] = None

    contains_phi: bool = False
    sync_interval_minutes: int = 60

    last_checked_at: Optional[datetime] = None
    last_file_count: Optional[int] = None
    status: WatchedResourceStatus = "active"

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class WatchedFile(BaseModel):
    """A single Google Drive file being monitored for changes.

    Distinct from WatchedFolder because the sync model differs: folders
    are scanned periodically for member changes; files are individually
    re-fetched by modification timestamp. A WatchedFile may belong to a
    WatchedFolder (parent in Drive) or be standalone (file added
    individually via the customer UI).
    """

    id: str
    connection_id: str
    customer_id: str

    file_id: str
    file_name: str
    mime_type: Optional[str] = None

    contains_phi: bool = False
    sync_interval_minutes: int = 60

    last_checked_at: Optional[datetime] = None
    last_modified_at: Optional[datetime] = None
    status: WatchedResourceStatus = "active"

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
