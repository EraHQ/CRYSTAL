"""Tenant DEK store surface — envelope layer 1 (P2, 2026-07-10).

Owns the tenant_keys rows and the ONLY in-memory home plaintext DEKs
ever have: a module-level TTL cache (10 minutes, ratified) bounding
both KMS unwrap cost and the exposure window. R9: the SQL lives here.

Lifecycle:
  get_or_create_tenant_dek  first use lazily generates + wraps + inserts;
                            thereafter unwraps (through the cache).
                            REFUSES once a destroy is scheduled — a
                            shredding tenant's secrets go dark
                            immediately, not at the deadline.
  schedule / cancel / destroy  the ratified 24h grace: schedule sets the
                            deadline (reads refuse at once), cancel
                            clears it inside the window, destroy_due
                            hard-deletes expired rows (crypto-shredding:
                            ciphertext without a DEK is noise).
  rewrap_tenant_deks        KEK rotation verb: unwrap each DEK, re-wrap
                            under the CURRENT root, stamp kek_version.
                            Touches kilobytes, never data.
"""
from __future__ import annotations

import secrets as _secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import structlog

from .key_wrapper import KeyWrapperError, get_key_wrapper
from .schema import TenantKeyRow

logger = structlog.get_logger(__name__)

_DEK_CACHE_TTL_SECONDS = 600  # 10 minutes, ratified 2026-07-10
_DESTROY_GRACE = timedelta(hours=24)

# customer_id -> (plaintext DEK, monotonic expiry)
_dek_cache: dict[str, tuple[bytes, float]] = {}


class TenantKeyUnavailable(RuntimeError):
    """DEK cannot be produced: destroy scheduled/completed, or unwrap
    failed. Message never carries key material."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt: datetime) -> datetime:
    """SQLite strips tzinfo on DateTime(timezone=True) round-trips;
    normalize to UTC-aware before comparing."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def reset_dek_cache() -> None:
    """Test hook + destroy-path invalidation."""
    _dek_cache.clear()


class TenantKeyExtensionsMixin:
    async def get_or_create_tenant_dek(self, customer_id: str) -> bytes:
        """The tenant's plaintext DEK (32 bytes), via cache -> unwrap ->
        lazy create, refusing when a destroy is scheduled."""
        cached = _dek_cache.get(customer_id)
        if cached is not None and cached[1] > time.monotonic():
            return cached[0]

        wrapper = get_key_wrapper()
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(TenantKeyRow, customer_id)
            if row is not None:
                if row.destroy_scheduled_at is not None:
                    raise TenantKeyUnavailable(
                        f"tenant key for {customer_id} is scheduled for "
                        "destruction — secrets are unavailable"
                    )
                try:
                    dek = wrapper.unwrap(row.dek_wrapped)
                except KeyWrapperError as e:
                    raise TenantKeyUnavailable(
                        f"tenant DEK unwrap failed for {customer_id}: {e}"
                    ) from e
            else:
                dek = _secrets.token_bytes(32)
                session.add(TenantKeyRow(
                    customer_id=customer_id,
                    dek_wrapped=wrapper.wrap(dek),
                    kek_version=wrapper.kek_id,
                    created_at=_utcnow(),
                ))
                logger.info("tenant_key.created", customer_id=customer_id,
                            kek_version=wrapper.kek_id)

        _dek_cache[customer_id] = (
            dek, time.monotonic() + _DEK_CACHE_TTL_SECONDS,
        )
        return dek

    async def schedule_tenant_dek_destroy(
        self, customer_id: str
    ) -> Optional[datetime]:
        """Begin the 24h crypto-shredding grace. Reads refuse
        immediately; the row survives (cancelable) until the sweep.
        Returns the deadline, or None when no key exists."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(TenantKeyRow, customer_id)
            if row is None:
                return None
            deadline = _utcnow() + _DESTROY_GRACE
            row.destroy_scheduled_at = deadline
        _dek_cache.pop(customer_id, None)
        logger.info("tenant_key.destroy_scheduled",
                    customer_id=customer_id,
                    deadline=deadline.isoformat())
        return deadline

    async def cancel_tenant_dek_destroy(self, customer_id: str) -> bool:
        """Cancel inside the grace window. True when a schedule was
        cleared."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(TenantKeyRow, customer_id)
            if row is None or row.destroy_scheduled_at is None:
                return False
            row.destroy_scheduled_at = None
        logger.info("tenant_key.destroy_canceled", customer_id=customer_id)
        return True

    async def destroy_tenant_dek(
        self, customer_id: str, *, immediate: bool = False
    ) -> bool:
        """Hard-delete the DEK row — the crypto-shred. Default path is
        the scheduled sweep; immediate=True is the explicit
        legal/security override (ratified). True when a row died."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(TenantKeyRow, customer_id)
            if row is None:
                return False
            if not immediate:
                if (row.destroy_scheduled_at is None
                        or _aware(row.destroy_scheduled_at) > _utcnow()):
                    return False  # not due; the sweep respects the grace
            await session.delete(row)
        _dek_cache.pop(customer_id, None)
        logger.info("tenant_key.destroyed", customer_id=customer_id,
                    immediate=immediate)
        return True

    async def encrypt_tenant_secret(
        self, customer_id: str, family: str, plaintext: str
    ) -> str:
        """enc:v2 under this tenant's DEK (fetching/creating it)."""
        from .token_crypto import encrypt_with_dek
        dek = await self.get_or_create_tenant_dek(customer_id)
        return encrypt_with_dek(dek, customer_id, family, plaintext)

    async def decrypt_tenant_secret(
        self, customer_id: str, family: str, value: str
    ) -> str:
        """Decrypt an enc:v2 value under this tenant's DEK."""
        from .token_crypto import decrypt_with_dek
        dek = await self.get_or_create_tenant_dek(customer_id)
        return decrypt_with_dek(dek, customer_id, family, value)

    async def rewrap_tenant_deks(self) -> dict[str, int]:
        """KEK-rotation walk: re-wrap every DEK under the CURRENT root.

        Safe to run repeatedly (already-current rows re-wrap with a
        fresh nonce, harmless). Skips destroy-scheduled rows — a
        shredding tenant's key needs no new wrapping."""
        from sqlalchemy import select
        wrapper = get_key_wrapper()
        rewrapped = 0
        skipped = 0
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(select(TenantKeyRow))).scalars().all()
            for row in rows:
                if row.destroy_scheduled_at is not None:
                    skipped += 1
                    continue
                dek = wrapper.unwrap(row.dek_wrapped)
                row.dek_wrapped = wrapper.wrap(dek)
                row.kek_version = wrapper.kek_id
                row.rotated_at = _utcnow()
                rewrapped += 1
        reset_dek_cache()
        logger.info("tenant_key.rewrap_walk",
                    rewrapped=rewrapped, skipped=skipped)
        return {"rewrapped": rewrapped, "skipped": skipped}
