"""KEK rotation — the rewrap walk (P4, 2026-07-10).

Replaces the v1 rotation-walk suite: rotation is now two verbs.
`rewrap_tenant_deks` (KEK version bump: re-wrap every DEK under the
current root — kilobytes, never data) is pinned here; per-tenant
DEK rekey is a later, rarer operation. Secrets encrypted before a
rewrap MUST decrypt after it: the DEK is unchanged, only its wrapping.
"""
from __future__ import annotations

import secrets

import pytest

from crystal_cache.infrastructure.key_wrapper import (
    LocalMasterKeyWrapper,
    reset_wrapper_cache,
)
from crystal_cache.infrastructure.metadata_store_keys_ext import (
    reset_dek_cache,
)


@pytest.fixture(autouse=True)
def _fresh_caches():
    reset_wrapper_cache()
    reset_dek_cache()
    yield
    reset_wrapper_cache()
    reset_dek_cache()


async def test_secrets_survive_kek_rotation(store, customer, monkeypatch):
    from crystal_cache import config as cfg
    from crystal_cache.infrastructure import key_wrapper as kw

    s = cfg.get_settings()
    old_master = secrets.token_hex(32)
    new_master = secrets.token_hex(32)
    monkeypatch.setattr(s, "key_wrapper", "local", raising=False)
    monkeypatch.setattr(s, "token_encryption_key", old_master, raising=False)
    reset_wrapper_cache()
    # the customer fixture may have birthed a DEK under conftest's env
    # key before this test's monkeypatch — shred it so the DEK below is
    # deterministically wrapped under old_master.
    await store.destroy_tenant_dek(customer.id, immediate=True)

    ct = await store.encrypt_tenant_secret(customer.id, "key_b", "sk-survive")

    # rotate the ROOT: new master key becomes the wrapper
    monkeypatch.setattr(s, "token_encryption_key", new_master, raising=False)
    reset_wrapper_cache()
    reset_dek_cache()

    # before the rewrap, the stored DEK is wrapped under the OLD root —
    # unwrap under the new one must fail (this is why the walk exists)
    from crystal_cache.infrastructure.metadata_store_keys_ext import (
        TenantKeyUnavailable,
    )
    with pytest.raises(TenantKeyUnavailable):
        await store.get_or_create_tenant_dek(customer.id)

    # transition wrapper: old root unwraps, new root wraps — exactly the
    # dual-key window a real rotation runs the walk inside
    class TransitionWrapper(LocalMasterKeyWrapper):
        kek_id = "local"
        def __init__(self):
            super().__init__(new_master)
            self._old = LocalMasterKeyWrapper(old_master)
        def unwrap(self, wrapped: str) -> bytes:
            try:
                return super().unwrap(wrapped)
            except Exception:
                return self._old.unwrap(wrapped)

    monkeypatch.setattr(kw, "_cached_wrapper", TransitionWrapper())
    reset_dek_cache()
    out = await store.rewrap_tenant_deks()
    assert out["rewrapped"] >= 1

    # steady state: ONLY the new root configured — everything works
    monkeypatch.setattr(kw, "_cached_wrapper", None)
    reset_wrapper_cache()
    reset_dek_cache()
    assert await store.decrypt_tenant_secret(
        customer.id, "key_b", ct) == "sk-survive"
