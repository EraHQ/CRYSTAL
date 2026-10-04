"""Account purge worker (2026-10-03; Q37=B).

Once an hour: every customer whose purge_after has passed is purged
(bank, account, tenant key, operators and grants, ledger anonymized,
customer row deleted) through MetadataStore.purge_tenant, and the
in-memory vector caches are invalidated for that tenant. The seven-day
grace period lives in the row (purge_after); this loop only honours it.

Runs under the "purge" worker role (CC_WORKER_ROLES; "all" includes it).
One tenant per iteration step, each in its own try, so a failing purge
is logged and retried next hour without blocking the others.
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)

PURGE_INTERVAL_SECONDS = 3600


async def purge_due_accounts(store, *, caches: Optional[list[Any]] = None) -> list[str]:
    """One pass. Returns the customer ids purged. Exposed for the pin and
    for an operator to call by hand."""
    purged: list[str] = []
    for customer_id in await store.list_due_purges():
        try:
            counts = await store.purge_tenant(customer_id)
            for cache in caches or []:
                if hasattr(cache, "invalidate"):
                    try:
                        cache.invalidate(customer_id)
                    except Exception:  # noqa: BLE001
                        pass
            logger.info("purge.account_purged", customer_id=customer_id, counts=counts)
            purged.append(customer_id)
        except Exception:  # noqa: BLE001
            logger.exception("purge.account_failed", customer_id=customer_id)
    return purged


async def run_purge_worker(
    *,
    store,
    shutdown_event: asyncio.Event,
    caches: Optional[list[Any]] = None,
    interval_seconds: int = PURGE_INTERVAL_SECONDS,
) -> None:
    logger.info("purge_worker.started", interval_seconds=interval_seconds)
    while not shutdown_event.is_set():
        try:
            await purge_due_accounts(store, caches=caches)
        except Exception:  # noqa: BLE001
            logger.exception("purge_worker.pass_failed")
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            continue
    logger.info("purge_worker.stopped")
