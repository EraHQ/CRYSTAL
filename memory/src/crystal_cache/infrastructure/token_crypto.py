"""AES-256-GCM encryption for secrets at rest.

Serves two secret families: Google Drive OAuth refresh tokens (the
original two-column ciphertext+nonce shape) and customer upstream API
keys — Key B — via the composite-string helpers below (launch-prep
security pass, 2026-07-02: upstream keys are encrypted UNCONDITIONALLY;
plaintext at rest is never written and never read back).

The encryption key is loaded from CC_TOKEN_ENCRYPTION_KEY and must be a
32-byte hex string (64 hex characters). Any deployment that stores
upstream keys REQUIRES it — writers fail loudly when it is missing
rather than falling back to plaintext.

Generate a key: python -c "import secrets; print(secrets.token_hex(32))"
"""
from __future__ import annotations

import os
import logging
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

_KEY_ENV = "CC_TOKEN_ENCRYPTION_KEY"
_cached_key: bytes | None = None
_cached_keyring: list[bytes] | None = None


def _parse_hex_key(hex_key: str, *, source: str) -> bytes:
    """Validate and decode one 64-hex-char (32-byte) key."""
    if len(hex_key) != 64:
        raise RuntimeError(
            f"{source} must be exactly 64 hex characters (32 bytes). "
            f"Got {len(hex_key)} characters."
        )
    return bytes.fromhex(hex_key)


def _get_key() -> bytes:
    """The PRIMARY key — used for all encryption and rotation targets."""
    global _cached_key
    if _cached_key is not None:
        return _cached_key

    from ..config import settings
    hex_key = settings.token_encryption_key or ""
    if not hex_key:
        raise RuntimeError(
            "CC_TOKEN_ENCRYPTION_KEY not set. Generate one with: "
            "python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    _cached_key = _parse_hex_key(hex_key, source="CC_TOKEN_ENCRYPTION_KEY")
    return _cached_key


def _get_keyring() -> list[bytes]:
    """All keys usable for DECRYPTION, primary first, then any retired keys
    from CC_TOKEN_ENCRYPTION_KEYS_RETIRED (comma-separated 64-hex strings).

    Key rotation (E3, 2026-07-03): to rotate, generate a new key, move the
    old CC_TOKEN_ENCRYPTION_KEY value into CC_TOKEN_ENCRYPTION_KEYS_RETIRED,
    set the new key as CC_TOKEN_ENCRYPTION_KEY, boot, then run the rotation
    walk (rotate_secret / the admin re-encrypt command) to re-encrypt every
    stored secret under the new primary. Once the walk completes, the old
    key can be dropped from the retired list. Decryption tries the primary
    first (the common path) then each retired key, so already-rotated and
    not-yet-rotated rows both decrypt during the transition.
    """
    global _cached_keyring
    if _cached_keyring is not None:
        return _cached_keyring

    ring = [_get_key()]
    from ..config import settings
    retired_raw = getattr(settings, "token_encryption_keys_retired", "") or ""
    for i, chunk in enumerate(retired_raw.split(",")):
        hexk = chunk.strip()
        if not hexk:
            continue
        ring.append(_parse_hex_key(
            hexk, source=f"CC_TOKEN_ENCRYPTION_KEYS_RETIRED[{i}]",
        ))
    _cached_keyring = ring
    return _cached_keyring


def encrypt_token(plaintext: str) -> tuple[str, str]:
    """Encrypt a token string. Returns (ciphertext_hex, nonce_hex)."""
    key = _get_key()
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)  # 96-bit nonce for GCM
    ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
    return ciphertext.hex(), nonce.hex()


def decrypt_token(ciphertext_hex: str, nonce_hex: str) -> str:
    """Decrypt a token. Returns the plaintext string.

    Tries every key in the keyring (primary, then retired) so a secret
    encrypted under a pre-rotation key still decrypts during a rotation
    transition. Raises the last error if no key succeeds.
    """
    nonce = bytes.fromhex(nonce_hex)
    ciphertext = bytes.fromhex(ciphertext_hex)
    last_err: Exception | None = None
    for key in _get_keyring():
        try:
            plaintext = AESGCM(key).decrypt(nonce, ciphertext, None)
            return plaintext.decode("utf-8")
        except Exception as e:  # noqa: BLE001 — try the next key
            last_err = e
    raise ValueError(
        "token could not be decrypted with any configured key "
        "(primary or retired)"
    ) from last_err


# ---------------------------------------------------------------------------
# Composite-string secrets (single-column storage, e.g. Key B inside the
# customers.model_routing_config JSON). Format: enc:v1:{nonce_hex}:{ct_hex}.
# The prefix makes encrypted-vs-legacy unambiguous; v1 = AES-256-GCM with a
# random 96-bit nonce.
# ---------------------------------------------------------------------------

_SECRET_PREFIX = "enc:v1:"


def is_encrypted(value: str) -> bool:
    """True when the value carries the composite-encryption prefix."""
    return bool(value) and value.startswith(_SECRET_PREFIX)


def encrypt_secret(plaintext: str) -> str:
    """Encrypt to the composite single-column format.

    Requires CC_TOKEN_ENCRYPTION_KEY (fails loudly when missing — there
    is deliberately no plaintext fallback). Idempotence guard: an
    already-encrypted value is returned unchanged rather than
    double-wrapped.
    """
    if is_encrypted(plaintext):
        return plaintext
    ciphertext_hex, nonce_hex = encrypt_token(plaintext)
    return f"{_SECRET_PREFIX}{nonce_hex}:{ciphertext_hex}"


def decrypt_secret(value: str) -> str:
    """Decrypt a composite-format secret.

    Raises ValueError on a non-prefixed value — legacy plaintext must be
    migrated (alembic upgrade head runs the encrypt_upstream_keys data
    migration), never silently accepted.
    """
    if not is_encrypted(value):
        raise ValueError(
            "value is not in the enc:v1 composite format — legacy plaintext "
            "secrets must be migrated: run alembic upgrade head with "
            "CC_TOKEN_ENCRYPTION_KEY set"
        )
    try:
        _, _, rest = value.partition(_SECRET_PREFIX)
        nonce_hex, _, ciphertext_hex = rest.partition(":")
        return decrypt_token(ciphertext_hex, nonce_hex)
    except ValueError:
        raise
    except Exception as e:  # noqa: BLE001 — normalize crypto/parse errors
        raise ValueError(f"malformed enc:v1 secret: {e}") from e


def reset_key_cache() -> None:
    """Test hook: drop the cached key(s) so a monkeypatched
    CC_TOKEN_ENCRYPTION_KEY / retired list takes effect."""
    global _cached_key, _cached_keyring
    _cached_key = None
    _cached_keyring = None


def needs_rotation(value: str) -> bool:
    """True when a composite secret decrypts under a RETIRED key but not the
    primary — i.e. it still needs re-encrypting under the current primary.
    Used by the rotation walk to find rows to re-encrypt. A value that
    decrypts under the primary (or isn't encrypted) returns False."""
    if not is_encrypted(value):
        return False
    _, _, rest = value.partition(_SECRET_PREFIX)
    nonce_hex, _, ct_hex = rest.partition(":")
    try:
        nonce = bytes.fromhex(nonce_hex)
        ct = bytes.fromhex(ct_hex)
    except ValueError:
        return False
    # Primary succeeds → no rotation needed.
    try:
        AESGCM(_get_key()).decrypt(nonce, ct, None)
        return False
    except Exception:  # noqa: BLE001
        pass
    # Primary failed; if ANY retired key works, it needs rotation.
    for key in _get_keyring()[1:]:
        try:
            AESGCM(key).decrypt(nonce, ct, None)
            return True
        except Exception:  # noqa: BLE001
            continue
    return False


def rotate_secret(value: str) -> str:
    """Re-encrypt a composite secret under the current PRIMARY key.

    Decrypts with the keyring (handles retired keys) then re-encrypts with
    the primary. Idempotent for already-primary values (decrypt+re-encrypt
    yields an equivalent secret with a fresh nonce). Non-encrypted input is
    returned unchanged — legacy plaintext is the migration's job, not the
    rotation's.
    """
    if not is_encrypted(value):
        return value
    plaintext = decrypt_secret(value)  # keyring-aware
    # Force a fresh primary-key encryption (bypass the idempotence guard,
    # which would otherwise return the old ciphertext unchanged).
    ciphertext_hex, nonce_hex = encrypt_token(plaintext)
    return f"{_SECRET_PREFIX}{nonce_hex}:{ciphertext_hex}"
