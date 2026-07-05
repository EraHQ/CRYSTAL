"""Unit tests for infrastructure/credentials.py (Foundation F1).

Pure functions, no DB. These lock the no-plaintext contract: keys are
high-entropy and unique, the stored hash is deterministic (so it can be
indexed and matched directly at auth time), the hash is not the raw key,
and verify is correct and constant-time-safe.
"""
from __future__ import annotations

from crystal_cache.infrastructure.credentials import (
    generate_api_key,
    hash_api_key,
    verify_api_key,
)


def test_generate_api_key_is_prefixed_and_unique():
    a = generate_api_key()
    b = generate_api_key()
    assert a.startswith("cc_sk_")
    assert b.startswith("cc_sk_")
    assert a != b
    # High-entropy body (token_hex(32) -> 64 hex chars after the prefix).
    assert len(a) >= len("cc_sk_") + 64


def test_hash_is_deterministic_and_distinguishing():
    raw = generate_api_key()
    assert hash_api_key(raw) == hash_api_key(raw)
    assert hash_api_key(raw) != hash_api_key(generate_api_key())


def test_hash_is_not_the_raw_key():
    # No plaintext leakage: the stored hash must not equal the raw key.
    raw = generate_api_key()
    assert hash_api_key(raw) != raw


def test_hash_ignores_surrounding_whitespace():
    raw = generate_api_key()
    assert hash_api_key(f"  {raw}  ") == hash_api_key(raw)


def test_verify_round_trip():
    raw = generate_api_key()
    stored = hash_api_key(raw)
    assert verify_api_key(raw, stored) is True
    assert verify_api_key(generate_api_key(), stored) is False


def test_verify_rejects_empty_inputs():
    raw = generate_api_key()
    stored = hash_api_key(raw)
    assert verify_api_key("", stored) is False
    assert verify_api_key(raw, "") is False
    assert verify_api_key("", "") is False
