"""KeyWrapper — the envelope-encryption root seam (P1, 2026-07-10).

Mature-posture secrets architecture (ratified 2026-07-10):

    Layer 0  ROOT (this module): a Key-Encryption-Key that wraps DEKs
             and NOTHING else. Two implementations behind one seam —
             GcpKmsWrapper (cloud: HSM-backed KMS key, non-exportable,
             wrap/unwrap are API calls; the root never exists in our
             process memory) and LocalMasterKeyWrapper (self-host: a
             64-hex master key from CC_TOKEN_ENCRYPTION_KEY wrapping
             DEKs with the identical envelope structure). Same
             per-tenant architecture either way; the hosting-shaped
             open-core boundary holds.
    Layer 1  per-tenant DEKs (P2: tenant_keys table) — random 32-byte
             keys stored ONLY wrapped by this seam.
    Layer 2  secrets (P3: enc:v2) — AES-256-GCM under the tenant DEK
             with AAD = "{customer_id}:{family}", cryptographically
             binding every ciphertext to its owner and purpose.

Wrapped-DEK wire format (stored in tenant_keys.dek_wrapped):

    wrap:v1:local:{nonce_hex}:{ct_hex}     LocalMasterKeyWrapper
    wrap:v1:gcp_kms:{ciphertext_b64}       GcpKmsWrapper

The prefix names the wrapper that produced it, so a misconfigured
deployment fails loudly ("wrapped under gcp_kms but CC_KEY_WRAPPER is
local") instead of feeding one wrapper's output to the other.
"""
from __future__ import annotations

import base64
import os
from abc import ABC, abstractmethod

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_PREFIX = "wrap:v1:"


class KeyWrapperError(RuntimeError):
    """Raised for configuration and format failures. NEVER carries key
    material in its message."""


class KeyWrapper(ABC):
    """wrap/unwrap a 32-byte DEK. Implementations hold the root."""

    #: short identifier stored alongside wrapped DEKs (tenant_keys.kek_version)
    kek_id: str

    @abstractmethod
    def wrap(self, dek: bytes) -> str:
        """32-byte DEK -> prefixed wrapped string."""

    @abstractmethod
    def unwrap(self, wrapped: str) -> bytes:
        """Prefixed wrapped string -> 32-byte DEK."""


def _require_dek(dek: bytes) -> None:
    if not isinstance(dek, (bytes, bytearray)) or len(dek) != 32:
        raise KeyWrapperError("DEK must be exactly 32 bytes")


class LocalMasterKeyWrapper(KeyWrapper):
    """Self-host root: AES-256-GCM under a 64-hex master key.

    Reuses CC_TOKEN_ENCRYPTION_KEY as the root so a self-host operator
    configures ONE secret. The master key wraps DEKs only — tenant
    data is never encrypted under it directly (that's the DEK's job),
    so rotating the master re-wraps kilobytes, never the data.
    """

    kek_id = "local"

    def __init__(self, hex_key: str) -> None:
        hex_key = (hex_key or "").strip()
        if len(hex_key) != 64:
            raise KeyWrapperError(
                "LocalMasterKeyWrapper requires a 64-hex-character "
                "(32-byte) master key — generate one with: "
                'python -c "import secrets; print(secrets.token_hex(32))"'
            )
        try:
            self._key = bytes.fromhex(hex_key)
        except ValueError as e:
            raise KeyWrapperError(
                "master key is not valid hex"
            ) from e

    def wrap(self, dek: bytes) -> str:
        _require_dek(dek)
        nonce = os.urandom(12)
        ct = AESGCM(self._key).encrypt(nonce, bytes(dek), b"dek:v1")
        return f"{_PREFIX}local:{nonce.hex()}:{ct.hex()}"

    def unwrap(self, wrapped: str) -> bytes:
        body = _split(wrapped, expect="local")
        nonce_hex, _, ct_hex = body.partition(":")
        try:
            nonce = bytes.fromhex(nonce_hex)
            ct = bytes.fromhex(ct_hex)
        except ValueError as e:
            raise KeyWrapperError("malformed local-wrapped DEK") from e
        try:
            dek = AESGCM(self._key).decrypt(nonce, ct, b"dek:v1")
        except Exception as e:  # noqa: BLE001 — normalize crypto errors
            raise KeyWrapperError(
                "DEK unwrap failed — wrong master key or corrupted value"
            ) from e
        _require_dek(dek)
        return dek


class GcpKmsWrapper(KeyWrapper):
    """Cloud root: Google Cloud KMS symmetric key (HSM protection level
    per the 2026-07-10 ratification). wrap/unwrap are KMS API calls —
    the KEK is non-exportable and never exists in process memory. Every
    unwrap lands in Cloud Audit Logs.

    Lazy client import: google-cloud-kms is a cloud-deploy dependency,
    not a self-host one.
    """

    def __init__(self, key_resource: str) -> None:
        key_resource = (key_resource or "").strip()
        if not key_resource:
            raise KeyWrapperError(
                "GcpKmsWrapper requires CC_KMS_KEY_RESOURCE "
                "(projects/.../locations/.../keyRings/.../cryptoKeys/...)"
            )
        try:
            from google.cloud import kms  # noqa: PLC0415 — lazy by design
        except ImportError as e:
            raise KeyWrapperError(
                "google-cloud-kms is not installed — required when "
                "CC_KEY_WRAPPER=gcp_kms"
            ) from e
        self._client = kms.KeyManagementServiceClient()
        self._key_resource = key_resource
        self.kek_id = key_resource

    def wrap(self, dek: bytes) -> str:
        _require_dek(dek)
        resp = self._client.encrypt(
            request={"name": self._key_resource, "plaintext": bytes(dek)}
        )
        return f"{_PREFIX}gcp_kms:" + base64.b64encode(resp.ciphertext).decode(
            "ascii"
        )

    def unwrap(self, wrapped: str) -> bytes:
        body = _split(wrapped, expect="gcp_kms")
        try:
            ciphertext = base64.b64decode(body, validate=True)
        except Exception as e:  # noqa: BLE001
            raise KeyWrapperError("malformed gcp_kms-wrapped DEK") from e
        resp = self._client.decrypt(
            request={"name": self._key_resource, "ciphertext": ciphertext}
        )
        dek = bytes(resp.plaintext)
        _require_dek(dek)
        return dek


def _split(wrapped: str, *, expect: str) -> str:
    """Validate prefix + wrapper id; return the payload body."""
    if not (wrapped or "").startswith(_PREFIX):
        raise KeyWrapperError("value is not a wrapped DEK (missing prefix)")
    rest = wrapped[len(_PREFIX):]
    wrapper_id, _, body = rest.partition(":")
    if wrapper_id != expect:
        raise KeyWrapperError(
            f"DEK was wrapped under '{wrapper_id}' but this deployment's "
            f"wrapper is '{expect}' — CC_KEY_WRAPPER mismatch"
        )
    if not body:
        raise KeyWrapperError("malformed wrapped DEK (empty body)")
    return body


_cached_wrapper: KeyWrapper | None = None


def get_key_wrapper() -> KeyWrapper:
    """The deployment's configured wrapper (cached).

    CC_KEY_WRAPPER: "gcp_kms" | "local". Unset resolves to "local"
    WHEN a master key is configured (self-host ergonomics); production
    deployments set it explicitly — the P5 boot gate enforces that
    wrapper config is present in production.
    """
    global _cached_wrapper
    if _cached_wrapper is not None:
        return _cached_wrapper

    from ..config import get_settings
    settings = get_settings()
    choice = (getattr(settings, "key_wrapper", "") or "").strip().lower()
    if choice == "gcp_kms":
        _cached_wrapper = GcpKmsWrapper(
            getattr(settings, "kms_key_resource", "") or ""
        )
    elif choice in ("local", ""):
        _cached_wrapper = LocalMasterKeyWrapper(
            getattr(settings, "token_encryption_key", "") or ""
        )
    else:
        raise KeyWrapperError(
            f"unknown CC_KEY_WRAPPER value {choice!r} — "
            "expected 'gcp_kms' or 'local'"
        )
    return _cached_wrapper


def reset_wrapper_cache() -> None:
    """Test hook: drop the cached wrapper so monkeypatched settings
    take effect."""
    global _cached_wrapper
    _cached_wrapper = None
