"""P3 (2026-07-10): enc:v2 tenant-scoped envelope secrets.

The load-bearing test is the AAD wall: a ciphertext moved across
tenants, across families, or tampered with MUST fail to decrypt.
"""
import secrets

import pytest

from crystal_cache.infrastructure.key_wrapper import reset_wrapper_cache
from crystal_cache.infrastructure.metadata_store_keys_ext import reset_dek_cache
from crystal_cache.infrastructure.token_crypto import (
    ENC_V2_PREFIX,
    decrypt_with_dek,
    encrypt_with_dek,
    is_v2_encrypted,
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


def test_pure_roundtrip_and_format():
    dek = secrets.token_bytes(32)
    ct = encrypt_with_dek(dek, "cus_a", "key_b", "sk-ant-secret")
    assert ct.startswith(ENC_V2_PREFIX)
    assert is_v2_encrypted(ct)
    assert "sk-ant-secret" not in ct
    assert decrypt_with_dek(dek, "cus_a", "key_b", ct) == "sk-ant-secret"


def test_aad_wall_cross_tenant_family_and_tamper():
    dek = secrets.token_bytes(32)
    ct = encrypt_with_dek(dek, "cus_a", "key_b", "sk-ant-secret")
    # wrong tenant — same DEK, moved ciphertext: MUST fail
    with pytest.raises(ValueError):
        decrypt_with_dek(dek, "cus_b", "key_b", ct)
    # wrong family — repurposed ciphertext: MUST fail
    with pytest.raises(ValueError):
        decrypt_with_dek(dek, "cus_a", "drive_oauth", ct)
    # wrong DEK
    with pytest.raises(ValueError):
        decrypt_with_dek(secrets.token_bytes(32), "cus_a", "key_b", ct)
    # tamper: flip one ciphertext hex digit
    head, _, tail = ct.rpartition(":")
    flipped = head + ":" + ("0" if tail[0] != "0" else "1") + tail[1:]
    with pytest.raises(ValueError):
        decrypt_with_dek(dek, "cus_a", "key_b", flipped)


async def test_store_surface_roundtrip_and_isolation(store, customer):
    ct = await store.encrypt_tenant_secret(customer.id, "key_b", "sk-live-1")
    assert await store.decrypt_tenant_secret(
        customer.id, "key_b", ct) == "sk-live-1"
    # ANOTHER tenant's store surface cannot read it: different DEK
    # entirely, and the AAD names the owner besides.
    other = await store.create_customer(
        provider="anthropic",
        model_id="claude-sonnet-4-5-20250929",
        api_key_ref="sk-test-other",
    )
    with pytest.raises(ValueError):
        await store.decrypt_tenant_secret(other.id, "key_b", ct)


async def test_shred_makes_ciphertext_permanent_noise(store, customer):
    ct = await store.encrypt_tenant_secret(customer.id, "key_b", "gone")
    await store.destroy_tenant_dek(customer.id, immediate=True)
    # fresh DEK on next use — the old ciphertext can never decrypt again
    with pytest.raises(ValueError):
        await store.decrypt_tenant_secret(customer.id, "key_b", ct)
