"""API-key credentials — generation, hashing, verification (Foundation F1).

Security posture (2026-06-13: "no plaintext; security is the largest
concern"): raw API keys are NEVER persisted. Only a one-way hash is
stored. Auth hashes the presented key and looks it up by that hash.

API keys are high-entropy random tokens, unlike passwords — so a fast
keyed hash is the right tool. bcrypt/argon2 buy nothing against a
256-bit random secret and would break the O(1) indexed lookup auth
needs. We use HMAC-SHA256 with a server-side pepper when one is
configured (defense-in-depth: a stolen DB can't be brute-forced for
keys without the pepper), falling back to plain SHA-256 when no pepper
is set. Both are deterministic, so the stored hash is directly
indexable.

Set CC_API_KEY_PEPPER in any real deployment. The empty-pepper SHA-256
fallback exists only so dev works with zero config; it is still
no-plaintext (the raw key never lands in the DB), just without the
stolen-DB defense the pepper adds.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

from ..config import get_settings

# Marks a Crystal Cache secret key. Operators and customers share the
# scheme; the raw key is shown once at creation and then only its hash
# is kept.
_KEY_PREFIX = "cc_sk_"


def generate_api_key() -> str:
    """Return a fresh high-entropy API key (the RAW key, shown once)."""
    return f"{_KEY_PREFIX}{secrets.token_hex(32)}"


def hash_api_key(raw_key: str) -> str:
    """One-way, deterministic hash of a raw API key for storage + lookup.

    HMAC-SHA256(pepper, key) when a pepper is configured, else
    SHA-256(key). Deterministic either way, so the result is safe to
    store in a unique-indexed column and match on directly.
    """
    raw = (raw_key or "").strip().encode("utf-8")
    pepper = (getattr(get_settings(), "api_key_pepper", "") or "").strip()
    if pepper:
        return hmac.new(pepper.encode("utf-8"), raw, hashlib.sha256).hexdigest()
    return hashlib.sha256(raw).hexdigest()


def verify_api_key(raw_key: str, stored_hash: str) -> bool:
    """Constant-time check that raw_key hashes to stored_hash.

    The auth path looks operators up BY hash (one indexed query), so it
    doesn't need this; verify is here for callers that already hold a
    row and want to confirm a presented key against it.
    """
    if not raw_key or not stored_hash:
        return False
    return hmac.compare_digest(hash_api_key(raw_key), stored_hash)
