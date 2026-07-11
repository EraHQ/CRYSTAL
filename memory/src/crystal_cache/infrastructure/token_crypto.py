"""Tenant-scoped secret encryption — enc:v2 (P3/P4, 2026-07-10).

Envelope layer 2 of the mature-posture architecture: AES-256-GCM under
the TENANT's Data Encryption Key with AAD = "{customer_id}:{family}".
DEK management (layer 1) lives in metadata_store_keys_ext; the root
(layer 0) behind infrastructure/key_wrapper.

v1 (enc:v1: one global CC_TOKEN_ENCRYPTION_KEY over all tenants) was
DELETED at the 2026-07-10 cutover. The production census found exactly
one v1 row — a test customer whose ciphertext was already orphaned by
a wiped env var — so no legacy-decrypt path ships. The
CC_TOKEN_ENCRYPTION_KEY setting survives with ONE meaning: the
LocalMasterKeyWrapper's root (self-host deployments).
"""
from __future__ import annotations

import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# ---------------------------------------------------------------------------
# enc:v2 — tenant-scoped envelope secrets (P3, 2026-07-10)
# ---------------------------------------------------------------------------
# AES-256-GCM under the TENANT's DEK (envelope layer 2), with
# AAD = "{customer_id}:{family}" baked into the GCM tag: a ciphertext
# copied into another tenant's row — or another purpose's column —
# FAILS to decrypt. Cross-tenant splicing is cryptographically
# impossible, not merely policy-forbidden.
#
# These are PURE functions over an explicit 32-byte DEK so they test
# without a store; the async store surface that fetches the DEK lives
# in metadata_store_keys_ext (encrypt_tenant_secret /
# decrypt_tenant_secret). Families in use: "key_b" (provider keys),
# "drive_oauth" (refresh tokens).
#
# v1 (enc:v1, single global key) is DELETED in P4 when its two callers
# are rewired — the 2026-07-10 census found exactly one orphaned v1 row
# in production (key already unrecoverable), so no legacy-decrypt path
# ships.

ENC_V2_PREFIX = "enc:v2:"


def _v2_aad(customer_id: str, family: str) -> bytes:
    if not customer_id or not family:
        raise ValueError("customer_id and family are required for enc:v2")
    return f"{customer_id}:{family}".encode("utf-8")


def encrypt_with_dek(
    dek: bytes, customer_id: str, family: str, plaintext: str
) -> str:
    """Tenant-scoped encrypt: enc:v2:{nonce_hex}:{ct_hex}."""
    if not isinstance(dek, (bytes, bytearray)) or len(dek) != 32:
        raise ValueError("DEK must be exactly 32 bytes")
    if plaintext is None:
        raise ValueError("plaintext must not be None")
    nonce = os.urandom(12)
    ct = AESGCM(bytes(dek)).encrypt(
        nonce, plaintext.encode("utf-8"), _v2_aad(customer_id, family)
    )
    return f"{ENC_V2_PREFIX}{nonce.hex()}:{ct.hex()}"


def decrypt_with_dek(
    dek: bytes, customer_id: str, family: str, value: str
) -> str:
    """Tenant-scoped decrypt. Raises ValueError on wrong tenant, wrong
    family, wrong DEK, tampering, or malformed input — one normalized
    error class, no key material in messages."""
    if not (value or "").startswith(ENC_V2_PREFIX):
        raise ValueError("value is not in the enc:v2 composite format")
    body = value[len(ENC_V2_PREFIX):]
    nonce_hex, _, ct_hex = body.partition(":")
    try:
        nonce = bytes.fromhex(nonce_hex)
        ct = bytes.fromhex(ct_hex)
    except ValueError as e:
        raise ValueError(f"malformed enc:v2 secret: {e}") from e
    try:
        pt = AESGCM(bytes(dek)).decrypt(
            nonce, ct, _v2_aad(customer_id, family)
        )
    except Exception as e:  # noqa: BLE001 — normalize AEAD failures
        raise ValueError(
            "enc:v2 decrypt failed — wrong tenant, family, or key, "
            "or the ciphertext was tampered with"
        ) from e
    return pt.decode("utf-8")


def is_v2_encrypted(value: str) -> bool:
    return (value or "").startswith(ENC_V2_PREFIX)
