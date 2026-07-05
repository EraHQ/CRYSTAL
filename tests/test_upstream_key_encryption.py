"""Upstream-key encryption at rest (launch-prep security pass, 2026-07-02).

Key B (customers.model_routing_config.api_key_ref) is AES-256-GCM
encrypted UNCONDITIONALLY at both store writers, decrypted only at
get_upstream_client, and legacy plaintext is refused with a
run-the-migration error. conftest sets a test CC_TOKEN_ENCRYPTION_KEY
process-wide.

R14 note: these assertions are verified by `pytest`; they describe
expected behavior and have not yet been run at authoring time.
"""
from __future__ import annotations

import pytest

from crystal_cache.execution.upstream_client import (
    AnthropicClient,
    get_upstream_client,
)
from crystal_cache.infrastructure.token_crypto import (
    decrypt_secret,
    encrypt_secret,
    is_encrypted,
)
from crystal_cache.models import Customer, ModelRoutingConfig


def test_secret_roundtrip_and_prefix():
    enc = encrypt_secret("sk-super-secret")
    assert is_encrypted(enc)
    assert enc.startswith("enc:v1:")
    assert "sk-super-secret" not in enc
    assert decrypt_secret(enc) == "sk-super-secret"
    # Idempotence guard: encrypting an encrypted value doesn't double-wrap.
    assert encrypt_secret(enc) == enc


def test_decrypt_refuses_plaintext():
    with pytest.raises(ValueError, match="alembic upgrade head"):
        decrypt_secret("sk-raw-plaintext")


async def test_create_customer_stores_encrypted_ref(store):
    customer = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="sk-upstream-b",
    )
    stored = customer.model_routing_config.api_key_ref
    assert is_encrypted(stored)
    assert decrypt_secret(stored) == "sk-upstream-b"

    # And the row read back carries the encrypted form too.
    fetched = await store.get_customer_by_id(customer.id)
    assert is_encrypted(fetched.model_routing_config.api_key_ref)


async def test_update_upstream_key_encrypts(store):
    customer = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="sk-old",
    )
    ok = await store.update_customer_upstream_key(customer.id, "sk-new")
    assert ok

    fetched = await store.get_customer_by_id(customer.id)
    stored = fetched.model_routing_config.api_key_ref
    assert is_encrypted(stored)
    assert decrypt_secret(stored) == "sk-new"


async def test_get_upstream_client_decrypts(store):
    customer = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="sk-live-key",
    )
    fetched = await store.get_customer_by_id(customer.id)

    client = get_upstream_client(fetched)
    assert isinstance(client, AnthropicClient)
    # The client gets the DECRYPTED key — never the composite form.
    assert client._api_key == "sk-live-key"


def test_get_upstream_client_refuses_legacy_plaintext():
    customer = Customer(
        id="cus_legacy",
        api_key=None,
        model_routing_config=ModelRoutingConfig(
            provider="anthropic", model_id="m", api_key_ref="sk-plaintext",
        ),
    )
    with pytest.raises(RuntimeError, match="PLAINTEXT"):
        get_upstream_client(customer)


def test_empty_ref_passes_through_for_self_hosted():
    customer = Customer(
        id="cus_sh",
        api_key=None,
        model_routing_config=ModelRoutingConfig(
            provider="self_hosted", model_id="qwen", api_key_ref="",
            base_url="http://localhost:8001/v1",
        ),
    )
    client = get_upstream_client(customer)
    # Empty ref passes the strict gate; SelfHostedClient substitutes its
    # local placeholder for endpoints that don't check keys.
    assert client._api_key == "sk-local"
