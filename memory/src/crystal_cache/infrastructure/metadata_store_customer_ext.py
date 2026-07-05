"""Customer-table extension methods — Phase 6.5 P4.1 / CU-8.

Phase 5 added the AuditTablesMixin for the eight audit tables. Phase 3
ported the Customer CRUD methods verbatim from v1. Neither covered
partial JSON updates to `CustomerRow.model_routing_config`, so
`endpoints/customers.py::update_upstream_key` had to use inline
SQLAlchemy — violating the "no SQL outside the store layer" rule.

Phase 6.5 P4.1 closes that gap by adding the missing method here,
using the same mixin-via-setattr pattern as the audit tables (D12).

Why a separate mixin from AuditTablesMixin: scope clarity. The audit
mixin is documented as "the eight audit tables." Adding customer-
extension methods there mixes concerns. A separate file makes the
binding intent visible at module level.

If a future phase adds more `update_customer_X` style partial-JSON
updates (subscription list, retention policy edit, billing config),
they belong here next to `update_customer_upstream_key`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .schema import CustomerRow

logger = structlog.get_logger(__name__)


class CustomerExtensionsMixin:
    """Customer-table partial-update methods bound onto MetadataStore.

    Bound at import time via setattr in `infrastructure/__init__.py`,
    same pattern as AuditTablesMixin (D12).
    """

    async def update_customer_upstream_key(
        self, customer_id: str, new_api_key_ref: str
    ) -> bool:
        """Update the customer's upstream provider API key (Key B).

        The api_key_ref lives inside the JSON
        `CustomerRow.model_routing_config` column rather than as a
        scalar column. SQLAlchemy doesn't have first-class partial-
        JSON-update support across dialects, so we read the dict,
        mutate it, and write it back inside one session.

        Returns True if the row existed and was updated, False if
        no customer matched (caller should 404).

        Per Phase 6.5 P4.1 / CU-8. Replaces inline SQLAlchemy in
        endpoints/customers.py::update_upstream_key.
        """
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(CustomerRow, customer_id)
            if row is None:
                return False
            from .token_crypto import encrypt_secret
            # Mutate a copy and reassign — direct in-place mutation
            # of the JSON dict doesn't reliably mark the column dirty
            # across SQLAlchemy dialect implementations.
            config = dict(row.model_routing_config or {})
            # Key B encrypted at rest UNCONDITIONALLY (launch-prep
            # security pass) — no plaintext fallback.
            config["api_key_ref"] = (
                encrypt_secret(new_api_key_ref) if new_api_key_ref
                else new_api_key_ref
            )
            row.model_routing_config = config
            return True

    async def rotate_encrypted_secrets(self) -> dict[str, int]:
        """E3 key-rotation walk (2026-07-03): re-encrypt every stored secret
        under the CURRENT primary token key. Safe to run repeatedly and
        safe to run when no rotation is pending (already-primary secrets are
        re-encrypted with a fresh nonce, which is harmless). Covers customer
        upstream keys (Key B in model_routing_config) and Drive OAuth
        refresh tokens. Returns per-family counts of rows re-encrypted.

        Operational contract: set the new key as CC_TOKEN_ENCRYPTION_KEY and
        the old one in CC_TOKEN_ENCRYPTION_KEYS_RETIRED, boot, then call
        this. Once it completes cleanly the retired key may be dropped.
        """
        from .token_crypto import (
            decrypt_token, encrypt_token, is_encrypted, rotate_secret,
        )
        from .schema import CustomerRow, DriveConnectionRow

        counts = {"customers": 0, "drive_connections": 0}
        async with self.session() as session:  # type: ignore[attr-defined]
            # Customer upstream keys (composite enc:v1 string in JSON).
            customers = (await session.execute(
                select(CustomerRow)
            )).scalars().all()
            for row in customers:
                config = dict(row.model_routing_config or {})
                ref = config.get("api_key_ref") or ""
                if is_encrypted(ref):
                    config["api_key_ref"] = rotate_secret(ref)
                    row.model_routing_config = config
                    counts["customers"] += 1

            # Drive refresh tokens (two-column ciphertext+nonce).
            conns = (await session.execute(
                select(DriveConnectionRow)
            )).scalars().all()
            for conn in conns:
                try:
                    plaintext = decrypt_token(
                        conn.encrypted_refresh_token, conn.token_nonce,
                    )
                except Exception:  # noqa: BLE001 — leave undecryptable rows
                    continue
                ct_hex, nonce_hex = encrypt_token(plaintext)
                conn.encrypted_refresh_token = ct_hex
                conn.token_nonce = nonce_hex
                counts["drive_connections"] += 1

        return counts

    async def get_customer_shadow_cap_override(
        self, customer_id: str
    ) -> Optional[int]:
        """Return the customer's per-customer shadow cost-cap override.

        Phase 12 (CU-27 / P0.111). Returns the raw `shadow_max_per_day`
        column value: an explicit integer override, or None when the
        customer has no override set (the caller should then fall back
        to the global default). Returns None for a missing customer
        too — the metacognition worker treats both "no override" and
        "no such customer" identically (use the global default).

        Deliberately returns the RAW override rather than resolving the
        global default here: the worker already carries an injectable
        default (`shadow_max_per_day` parameter) for testing, and
        resolving the default in this method would couple the store
        layer to `settings`. Resolution stays in the worker.
        """
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(CustomerRow, customer_id)
            if row is None:
                return None
            return row.shadow_max_per_day

    async def set_customer_shadow_cap(
        self, customer_id: str, cap: Optional[int]
    ) -> bool:
        """Set (or clear) the customer's per-customer shadow cost cap.

        Phase 12 (CU-27 / P0.111). Pass an integer to cap this
        customer's shadow critiques per rolling 24h window; pass None
        to clear the override (revert to the global default).

        Returns True if the row existed and was updated, False if no
        customer matched (caller should 404). Mirrors
        `update_customer_upstream_key`'s return contract.

        This is the programmatic setter; a future admin endpoint can
        call it. Negative caps are rejected (a cap below zero is
        meaningless — use 0 to disable shadowing for the customer).
        """
        if cap is not None and cap < 0:
            raise ValueError(
                f"shadow_max_per_day must be >= 0 or None; got {cap}"
            )
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(CustomerRow, customer_id)
            if row is None:
                return False
            row.shadow_max_per_day = cap
            return True
