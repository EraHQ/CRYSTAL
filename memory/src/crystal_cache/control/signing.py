"""Signed authorization for control-plane decisions (Growth G2). PURE crypto.

The load-bearing security property: a control decision (approve/deny a
dangerous action, terminate a session) is **signed by the operator's key**, and
the AGENT verifies that signature against the operator's pinned public key
before acting. The server persists + relays the signed blob but cannot forge a
decision — it is a courier. This is what "P2P encryption" in the G2 design
actually means: end-to-end signed authorization, not peer-to-peer transport
(TLS is the transport floor underneath).

Signing primitive (R10 sub-decision, dev): **Ed25519 detached signatures** over
a canonical JSON payload. Ed25519 is small, fast, and dependency-light. The
**recommended production anchor is WebAuthn/passkeys** (the F1 credential
anchor) — hardware-backed, phishing- and replay-resistant, no key material to
manage; `verify_decision` is the seam a WebAuthn-assertion verifier slots into
without changing callers. The dev path uses a raw Ed25519 public key
(base64, 32 bytes) stored as the operator's pinned key.

**Fail-closed everywhere.** A missing crypto library, a missing/malformed key,
a bad signature, or any unexpected error → `verify_decision` returns False
(deny). Verification never raises and never defaults to allow. The agent also
checks the nonce is unused (replay) and the timestamp is fresh
(`is_timestamp_fresh`) before acting; a decision with no valid signature is
rejected, and no decision at all times out to deny.

This module is import-safe without the `cryptography` package (the symbol just
becomes unavailable and verification fails closed), so it never breaks app
boot or the test suite; the signing/verifying tests `importorskip` it.
"""
from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from typing import Any, Optional, Union

import structlog

logger = structlog.get_logger(__name__)

# No-decision timeout → DENY (never approve). The agent enforces this while
# blocking at an approval gate; config (control_decision_timeout_seconds)
# overrides. Defined here so the agent + server agree on the default.
DECISION_TIMEOUT_SECONDS = 300

# Ed25519 is optional at import time — fail closed if absent.
try:  # pragma: no cover - exercised by environments with/without the lib
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    _HAVE_CRYPTO = True
except Exception:  # noqa: BLE001 - any import failure means "no crypto"
    _HAVE_CRYPTO = False


def _coerce_timestamp(timestamp: Union[str, int, float, datetime]) -> str:
    """Render a timestamp into the stable string used inside the signed
    payload. A datetime → ISO8601; everything else → str(). The operator and
    agent must canonicalize identically, so this is the single definition."""
    if isinstance(timestamp, datetime):
        return timestamp.astimezone(timezone.utc).isoformat()
    return str(timestamp)


def canonical_payload(
    *,
    session_id: str,
    request_id: str,
    decision: str,
    nonce: str,
    timestamp: Union[str, int, float, datetime],
) -> bytes:
    """The exact bytes that are signed and verified.

    Deterministic JSON (sorted keys, no insignificant whitespace) over the
    five authorization fields, UTF-8 encoded. Both the operator's signer and
    the agent's verifier MUST build the payload through this function so the
    bytes match exactly.
    """
    obj: dict[str, Any] = {
        "session_id": session_id,
        "request_id": request_id,
        "decision": decision,
        "nonce": nonce,
        "timestamp": _coerce_timestamp(timestamp),
    }
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def verify_decision(
    public_key_b64: Optional[str],
    *,
    session_id: str,
    request_id: str,
    decision: str,
    nonce: str,
    timestamp: Union[str, int, float, datetime],
    signature_b64: Optional[str],
) -> bool:
    """Verify an operator's signature over a control decision. FAIL-CLOSED.

    Returns True only when the `cryptography` library is present, the pinned
    public key and the signature are well-formed base64, and the signature
    verifies over the canonical payload. Any other condition — no crypto lib,
    missing key, missing signature, malformed input, bad signature, unexpected
    error — returns False. Never raises; never defaults to allow.
    """
    if not _HAVE_CRYPTO:
        logger.warning("control_signing.no_crypto_lib", verify="fail_closed")
        return False
    if not public_key_b64 or not signature_b64:
        return False
    try:
        pub_raw = base64.b64decode(public_key_b64, validate=True)
        sig = base64.b64decode(signature_b64, validate=True)
        public_key = Ed25519PublicKey.from_public_bytes(pub_raw)
        message = canonical_payload(
            session_id=session_id,
            request_id=request_id,
            decision=decision,
            nonce=nonce,
            timestamp=timestamp,
        )
        public_key.verify(sig, message)  # raises InvalidSignature on mismatch
        return True
    except InvalidSignature:
        logger.warning(
            "control_signing.invalid_signature",
            session_id=session_id,
            request_id=request_id,
        )
        return False
    except Exception as e:  # noqa: BLE001 - malformed key/sig, etc.
        logger.warning(
            "control_signing.verify_error",
            error=str(e),
            error_type=type(e).__name__,
        )
        return False


def is_timestamp_fresh(
    timestamp: Union[str, int, float, datetime],
    *,
    now: Optional[datetime] = None,
    max_age_seconds: int = DECISION_TIMEOUT_SECONDS,
) -> bool:
    """True iff `timestamp` is within `max_age_seconds` of `now` (replay /
    staleness guard the agent applies alongside the nonce check). A timestamp
    that can't be parsed is treated as stale (fail-closed). Accepts ISO8601
    strings, epoch seconds, or a datetime; a small future skew is tolerated."""
    now = now or datetime.now(timezone.utc)
    try:
        if isinstance(timestamp, datetime):
            ts = timestamp
        elif isinstance(timestamp, (int, float)):
            ts = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
        else:
            ts = datetime.fromisoformat(str(timestamp))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
    except (ValueError, OSError, OverflowError):
        return False
    age = (now - ts).total_seconds()
    # Allow a modest clock skew into the future; reject anything older than
    # the window.
    return -30.0 <= age <= float(max_age_seconds)


def sign_decision_dev(
    private_key_b64: str,
    *,
    session_id: str,
    request_id: str,
    decision: str,
    nonce: str,
    timestamp: Union[str, int, float, datetime],
) -> str:
    """DEV/TEST ONLY signer — produce a base64 Ed25519 signature.

    In production the operator's authenticator (WebAuthn/passkey) signs the
    challenge; the private key never leaves the device and this function is not
    used. It exists so dev tooling + the test suite can mint a valid signature
    to exercise `verify_decision`. Raises if the crypto library is unavailable
    (tests `importorskip("cryptography")`).
    """
    if not _HAVE_CRYPTO:
        raise RuntimeError("cryptography library not available")
    priv_raw = base64.b64decode(private_key_b64, validate=True)
    private_key = Ed25519PrivateKey.from_private_bytes(priv_raw)
    message = canonical_payload(
        session_id=session_id,
        request_id=request_id,
        decision=decision,
        nonce=nonce,
        timestamp=timestamp,
    )
    return base64.b64encode(private_key.sign(message)).decode("ascii")


def generate_keypair_dev() -> tuple[str, str]:
    """DEV/TEST ONLY — return (private_key_b64, public_key_b64) raw Ed25519.

    Production keys come from the operator's authenticator at F1 enrollment;
    this mints a throwaway pair for tooling + tests. Raises without the crypto
    library."""
    if not _HAVE_CRYPTO:
        raise RuntimeError("cryptography library not available")
    from cryptography.hazmat.primitives import serialization

    private_key = Ed25519PrivateKey.generate()
    priv_raw = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return (
        base64.b64encode(priv_raw).decode("ascii"),
        base64.b64encode(pub_raw).decode("ascii"),
    )
