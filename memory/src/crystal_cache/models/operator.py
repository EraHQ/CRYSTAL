"""Operator entity — Foundation F1 (team identity layer).

An operator is a human user acting under a team (the `customer` row).
F1 of `docs/FOUNDATION_AND_GROWTH.md`: the jump from one API key per
customer to multiple authenticated humans with roles. One shared key
can't carry multiple humans — no per-person attribution, no per-person
revocation, no roles — so operators add an authenticated-human layer
*beneath* the team entity that already owns everything.

Roles (D1, locked 2026-06-13) are the DEFAULT POSIX posture: per-resource
mode bits + named-grant ACLs refine the specifics on top.
  - admin    : root within the team (members, billing, all agents,
               curate/delete/merge team crystals); bypasses mode bits;
               scoped to its own team, never platform-wide.
  - operator : a regular user (run agents, author + use crystals, see
               the team base, manage own sessions).
  - viewer   : a read-only user (audit only).

Mirrors `OperatorRow` in `infrastructure/schema.py` 1:1.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


# Role posture (D1, locked). String-backed Literal so a future role
# (e.g. billing-only) can land without breaking persisted rows — matching
# the codebase's String-over-enum convention for crystal scope/grant.
OperatorRole = Literal["admin", "operator", "viewer"]

# Lifecycle. 'suspended' is denied at the auth boundary WITHOUT deleting
# the row, so an operator's owned crystals + provenance survive a
# suspension.
OperatorStatus = Literal["active", "suspended"]


class Operator(BaseModel):
    """A human user under a team (= customer)."""

    id: str
    # The team this operator belongs to (= customers.id). v1: exactly one
    # team per operator (many-to-many is a future team_memberships table).
    team_id: str
    display_name: str
    role: OperatorRole = "operator"
    status: OperatorStatus = "active"

    # Per-operator scoped credential (D2: per-operator API keys now; full
    # user-auth + tokens later). Stored HASHED — the auth boundary hashes
    # the presented key and looks up by hash; the raw key is never
    # persisted. NULL until a key is issued.
    #
    # Confirmed 2026-06-13: hashed + lookup-by-hash, no plaintext anywhere
    # (security is the priority; no backward-compat constraint). The store
    # hashes via infrastructure/credentials.hash_api_key; the raw key is
    # returned once by create_operator and never persisted.
    api_key_hash: Optional[str] = None

    # Passkey / public-key anchor (D2). The signing identity G2's control
    # plane reuses for end-to-end signed authorization (WebAuthn/passkeys
    # recommended). Provisioned here so the control plane gets it for
    # free; NULL until a passkey is registered.
    credential_public_key: Optional[str] = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
