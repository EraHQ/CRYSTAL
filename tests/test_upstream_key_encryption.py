"""Upstream-key (Key B) at rest — enc:v2 era (P4, 2026-07-10).

Replaces the 2026-07-02 v1 suite wholesale: Key B is now encrypted
under the TENANT's DEK with AAD binding. The invariants this file
pins: writers store only enc:v2; the decrypt-at-use factory round-trips
through the tenant surface; non-v2 non-empty refs are REFUSED at use;
managed mode never touches Key B.
"""
from __future__ import annotations

import pytest

from crystal_cache.execution.upstream_client import get_upstream_client
from crystal_cache.infrastructure.schema import CustomerRow
from crystal_cache.infrastructure.token_crypto import is_v2_encrypted


async def test_create_customer_stores_only_v2(store):
    c = await store.create_customer(
        provider="anthropic",
        model_id="claude-sonnet-4-5-20250929",
        api_key_ref="sk-live-created",
    )
    async with store.session() as session:
        row = await session.get(CustomerRow, c.id)
        stored = (row.model_routing_config or {}).get("api_key_ref", "")
    assert is_v2_encrypted(stored)
    assert "sk-live-created" not in stored
    # returned object mirrors the stored ciphertext
    assert c.model_routing_config.api_key_ref == stored
    # and the tenant DEK row was born alongside
    from crystal_cache.infrastructure.schema import TenantKeyRow
    async with store.session() as session:
        assert await session.get(TenantKeyRow, c.id) is not None


async def test_update_then_use_roundtrip(store, customer):
    await store.update_customer_upstream_key(customer.id, "sk-live-updated")
    fresh = await store.get_customer_by_id(customer.id)
    stored = fresh.model_routing_config.api_key_ref
    assert is_v2_encrypted(stored)
    client = await get_upstream_client(fresh, store)
    assert client._api_key == "sk-live-updated"


async def test_non_v2_ref_is_refused_at_use(store, customer):
    fresh = await store.get_customer_by_id(customer.id)
    fresh.model_routing_config.api_key_ref = "enc:v1:deadbeef:cafe"
    with pytest.raises(RuntimeError, match="not enc:v2"):
        await get_upstream_client(fresh, store)
    fresh.model_routing_config.api_key_ref = "sk-plaintext-never"
    with pytest.raises(RuntimeError, match="not enc:v2"):
        await get_upstream_client(fresh, store)


async def test_empty_ref_passes_through(store, customer):
    fresh = await store.get_customer_by_id(customer.id)
    fresh.model_routing_config.api_key_ref = ""
    fresh.model_routing_config.provider = "self_hosted"
    fresh.model_routing_config.base_url = "http://localhost:8000/v1"
    # must not raise: empty ref is legal (self_hosted substitutes a
    # benign placeholder credential internally)
    client = await get_upstream_client(fresh, store)
    assert client is not None


async def test_managed_mode_ignores_key_b(store, customer, monkeypatch):
    from crystal_cache import config as cfg
    monkeypatch.setattr(
        cfg.get_settings(), "anthropic_api_key", "sk-platform",
        raising=False,
    )
    fresh = await store.get_customer_by_id(customer.id)
    fresh.inference_mode = "managed"
    fresh.model_routing_config.api_key_ref = "enc:v1:orphaned:junk"
    # must NOT raise: Key B is never consulted on the managed path
    client = await get_upstream_client(fresh, store)
    assert client is not None
