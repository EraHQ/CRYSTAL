"""P2 (2026-07-10): tenant DEK lifecycle — envelope layer 1."""
import secrets

import pytest

from crystal_cache.infrastructure.key_wrapper import reset_wrapper_cache
from crystal_cache.infrastructure.metadata_store_keys_ext import (
    TenantKeyUnavailable,
    reset_dek_cache,
    _dek_cache,
)

MASTER = secrets.token_hex(32)


@pytest.fixture(autouse=True)
def _local_wrapper(monkeypatch):
    from crystal_cache import config as cfg
    s = cfg.get_settings()
    monkeypatch.setattr(s, "key_wrapper", "local", raising=False)
    monkeypatch.setattr(s, "token_encryption_key", MASTER, raising=False)
    reset_wrapper_cache()
    reset_dek_cache()
    yield
    reset_wrapper_cache()
    reset_dek_cache()


async def test_lazy_create_stable_and_cached(store, customer):
    dek1 = await store.get_or_create_tenant_dek(customer.id)
    assert len(dek1) == 32
    assert customer.id in _dek_cache
    # same key back — cache hit AND (after cache purge) unwrap path
    assert await store.get_or_create_tenant_dek(customer.id) == dek1
    reset_dek_cache()
    assert await store.get_or_create_tenant_dek(customer.id) == dek1


async def test_dek_stored_only_wrapped(store, customer):
    from sqlalchemy import select
    from crystal_cache.infrastructure.schema import TenantKeyRow
    dek = await store.get_or_create_tenant_dek(customer.id)
    async with store.session() as session:
        row = await session.get(TenantKeyRow, customer.id)
        assert row.dek_wrapped.startswith("wrap:v1:local:")
        assert dek.hex() not in row.dek_wrapped
        assert row.kek_version == "local"


async def test_destroy_grace_lifecycle(store, customer):
    dek = await store.get_or_create_tenant_dek(customer.id)
    deadline = await store.schedule_tenant_dek_destroy(customer.id)
    assert deadline is not None
    # reads refuse IMMEDIATELY (cache purged too)
    with pytest.raises(TenantKeyUnavailable):
        await store.get_or_create_tenant_dek(customer.id)
    # sweep respects the grace: not due yet -> no delete
    assert await store.destroy_tenant_dek(customer.id) is False
    # cancel inside the window restores access to the SAME dek
    assert await store.cancel_tenant_dek_destroy(customer.id) is True
    assert await store.get_or_create_tenant_dek(customer.id) == dek


async def test_immediate_destroy_shreds_and_regenerates(store, customer):
    dek_old = await store.get_or_create_tenant_dek(customer.id)
    assert await store.destroy_tenant_dek(customer.id, immediate=True) is True
    # next use births a FRESH dek — the old one (and anything encrypted
    # under it) is permanently unreachable
    dek_new = await store.get_or_create_tenant_dek(customer.id)
    assert dek_new != dek_old


async def test_rewrap_walk_updates_and_preserves(store, customer):
    dek = await store.get_or_create_tenant_dek(customer.id)
    from crystal_cache.infrastructure.schema import TenantKeyRow
    async with store.session() as session:
        row = await session.get(TenantKeyRow, customer.id)
        wrapped_before = row.dek_wrapped
    out = await store.rewrap_tenant_deks()
    assert out["rewrapped"] == 1
    async with store.session() as session:
        row = await session.get(TenantKeyRow, customer.id)
        assert row.dek_wrapped != wrapped_before   # fresh wrap
        assert row.rotated_at is not None
    reset_dek_cache()
    assert await store.get_or_create_tenant_dek(customer.id) == dek
