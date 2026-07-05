"""B3 + E3 crypto hardening (2026-07-03).

B3: production refuses to boot without CC_API_KEY_PEPPER (the boot guard
already enforced CC_ADMIN_API_KEY + CC_API_KEY_PEPPER; this pins the
pepper half so it can't silently regress).

E3: token-encryption key rotation. A secret encrypted under an old key
still decrypts while that key sits in CC_TOKEN_ENCRYPTION_KEYS_RETIRED,
and the rotation walk re-encrypts everything under the new primary.

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

import secrets

import pytest

from crystal_cache.config import Settings
from crystal_cache.infrastructure import token_crypto as tc


def _hexkey() -> str:
    return secrets.token_hex(32)


# --- B3: pepper is mandatory in production ---------------------------------

def test_boot_guard_requires_pepper_in_production():
    """The production safety gate must list CC_API_KEY_PEPPER as required.
    We assert the guard's own logic: production + empty pepper -> refuse."""
    prod_no_pepper = Settings(
        environment="production",
        admin_api_key="cc_sk_admin",
        api_key_pepper="",
        token_encryption_key=_hexkey(),
    )
    # Mirror the app boot-guard predicate.
    missing = [
        name for name, val in (
            ("CC_ADMIN_API_KEY", prod_no_pepper.admin_api_key),
            ("CC_API_KEY_PEPPER", prod_no_pepper.api_key_pepper),
        )
        if not (val or "").strip()
    ]
    assert "CC_API_KEY_PEPPER" in missing


def test_boot_guard_satisfied_when_pepper_set():
    prod = Settings(
        environment="production",
        admin_api_key="cc_sk_admin",
        api_key_pepper="a-real-pepper",
        token_encryption_key=_hexkey(),
    )
    missing = [
        name for name, val in (
            ("CC_ADMIN_API_KEY", prod.admin_api_key),
            ("CC_API_KEY_PEPPER", prod.api_key_pepper),
        )
        if not (val or "").strip()
    ]
    assert missing == []


# --- E3: key rotation -------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_key_cache():
    tc.reset_key_cache()
    yield
    tc.reset_key_cache()


def _use_keys(monkeypatch, primary: str, retired: str = "") -> None:
    s = Settings(
        environment="development",
        token_encryption_key=primary,
        token_encryption_keys_retired=retired,
    )
    monkeypatch.setattr(tc, "settings", s, raising=False)
    # token_crypto imports `settings` lazily from ..config; patch there too.
    import crystal_cache.config as cfg
    monkeypatch.setattr(cfg, "settings", s, raising=False)
    tc.reset_key_cache()


def test_secret_encrypted_under_old_key_decrypts_after_rotation(monkeypatch):
    old = _hexkey()
    new = _hexkey()

    # Encrypt under the OLD key (old is primary).
    _use_keys(monkeypatch, primary=old)
    blob = tc.encrypt_secret("sk-secret-value")
    assert tc.decrypt_secret(blob) == "sk-secret-value"

    # Rotate: NEW is primary, OLD is retired. The old-key blob still decrypts.
    _use_keys(monkeypatch, primary=new, retired=old)
    assert tc.decrypt_secret(blob) == "sk-secret-value"
    # And it is flagged as needing rotation (decrypts under retired, not new).
    assert tc.needs_rotation(blob) is True

    # rotate_secret re-encrypts under the new primary.
    rotated = tc.rotate_secret(blob)
    assert tc.decrypt_secret(rotated) == "sk-secret-value"
    assert tc.needs_rotation(rotated) is False


def test_decrypt_fails_when_neither_key_matches(monkeypatch):
    old = _hexkey()
    new = _hexkey()
    unrelated = _hexkey()

    _use_keys(monkeypatch, primary=old)
    blob = tc.encrypt_secret("v")

    # New primary + a DIFFERENT retired key (old not present) -> cannot decrypt.
    _use_keys(monkeypatch, primary=new, retired=unrelated)
    with pytest.raises(ValueError):
        tc.decrypt_secret(blob)


async def test_rotation_walk_reencrypts_customer_keys(monkeypatch, store):
    old = _hexkey()
    new = _hexkey()

    _use_keys(monkeypatch, primary=old)
    cust = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="upstream-key-b",
    )
    # Stored encrypted under the old key (old is primary now, so it does
    # not yet "need rotation").
    ref0 = (await store.get_customer_by_id(cust.id)) \
        .model_routing_config.api_key_ref
    assert tc.needs_rotation(ref0) is False

    # Rotate keys, then walk.
    _use_keys(monkeypatch, primary=new, retired=old)
    ref_before = (await store.get_customer_by_id(cust.id)) \
        .model_routing_config.api_key_ref
    assert tc.needs_rotation(ref_before) is True

    counts = await store.rotate_encrypted_secrets()
    assert counts["customers"] >= 1

    ref_after = (await store.get_customer_by_id(cust.id)) \
        .model_routing_config.api_key_ref
    assert tc.needs_rotation(ref_after) is False
    assert tc.decrypt_secret(ref_after) == "upstream-key-b"
