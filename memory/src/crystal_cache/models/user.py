"""User entity — hosted-platform account (Accounts Phase A, 2026-07-06).

The IdP-anchored sign-in identity for the managed platform: `id` is the
GCP Identity Platform uid verbatim. Distinct from Operator (the team-
internal human under Key-A auth): users are how a person SIGNS IN;
operators are how a team organizes people. One user -> one tenant in v1.

Mirrors `UserRow` in `infrastructure/schema.py` 1:1.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


# 'owner' = a tenant account; 'platform_admin' = Era staff (customer_id
# None — the platform root sits above tenants). String-backed Literal per
# codebase convention so future roles land without breaking rows.
UserRole = Literal["owner", "platform_admin"]


class User(BaseModel):
    """An account holder on the hosted platform."""

    id: str                      # GCP Identity Platform uid
    email: str
    customer_id: Optional[str] = None   # None only for platform_admin
    role: UserRole = "owner"

    # Onboarding signal (3-4 fields, one screen — ratified plan).
    industry: Optional[str] = None
    building: Optional[str] = None
    experience: Optional[str] = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
