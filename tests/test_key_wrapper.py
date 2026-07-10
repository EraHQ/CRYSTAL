"""P1 (2026-07-10): KeyWrapper seam contract tests.

LocalMasterKeyWrapper is exercised fully here. GcpKmsWrapper gets a
LIVE smoke at P6 (mocked-KMS unit theater proves nothing about IAM,
resource names, or API shapes — the parts that actually fail).
Structural failure modes (missing config/dep) ARE unit-testable and
are pinned below.
"""
import secrets

import pytest

from crystal_cache.infrastructure.key_wrapper import (
    GcpKmsWrapper,
    KeyWrapperError,
    LocalMasterKeyWrapper,
    get_key_wrapper,
    reset_wrapper_cache,
)

KEY_A = secrets.token_hex(32)
KEY_B = secrets.token_hex(32)


def test_local_roundtrip_and_format():
    w = LocalMasterKeyWrapper(KEY_A)
    dek = secrets.token_bytes(32)
    wrapped = w.wrap(dek)
    assert wrapped.startswith("wrap:v1:local:")
    assert dek.hex() not in wrapped          # never the DEK in the clear
    assert w.unwrap(wrapped) == dek
    # fresh nonce every wrap — same DEK, different ciphertexts
    assert w.wrap(dek) != wrapped


def test_local_wrong_master_key_fails_closed():
    w1 = LocalMasterKeyWrapper(KEY_A)
    w2 = LocalMasterKeyWrapper(KEY_B)
    wrapped = w1.wrap(secrets.token_bytes(32))
    with pytest.raises(KeyWrapperError):
        w2.unwrap(wrapped)


def test_dek_size_enforced_both_directions():
    w = LocalMasterKeyWrapper(KEY_A)
    with pytest.raises(KeyWrapperError):
        w.wrap(secrets.token_bytes(16))
    with pytest.raises(KeyWrapperError):
        w.unwrap("wrap:v1:local:deadbeef:deadbeef")


def test_wrapper_id_mismatch_named_loudly():
    w = LocalMasterKeyWrapper(KEY_A)
    with pytest.raises(KeyWrapperError, match="gcp_kms.*local|local.*gcp_kms"):
        w.unwrap("wrap:v1:gcp_kms:AAAA")
    with pytest.raises(KeyWrapperError, match="prefix"):
        w.unwrap("enc:v1:not-a-wrapped-dek")


def test_local_requires_64_hex():
    with pytest.raises(KeyWrapperError, match="64-hex"):
        LocalMasterKeyWrapper("short")
    with pytest.raises(KeyWrapperError):
        LocalMasterKeyWrapper("z" * 64)  # not hex


def test_gcp_wrapper_requires_resource():
    with pytest.raises(KeyWrapperError, match="CC_KMS_KEY_RESOURCE"):
        GcpKmsWrapper("")


def test_factory_selects_local_and_caches(monkeypatch):
    reset_wrapper_cache()
    from crystal_cache import config as cfg
    s = cfg.get_settings()
    monkeypatch.setattr(s, "key_wrapper", "local", raising=False)
    monkeypatch.setattr(s, "token_encryption_key", KEY_A, raising=False)
    w = get_key_wrapper()
    assert isinstance(w, LocalMasterKeyWrapper)
    assert get_key_wrapper() is w  # cached
    reset_wrapper_cache()


def test_factory_rejects_unknown_wrapper(monkeypatch):
    reset_wrapper_cache()
    from crystal_cache import config as cfg
    s = cfg.get_settings()
    monkeypatch.setattr(s, "key_wrapper", "vault", raising=False)
    with pytest.raises(KeyWrapperError, match="unknown"):
        get_key_wrapper()
    reset_wrapper_cache()
