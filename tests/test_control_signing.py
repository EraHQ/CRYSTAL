"""Growth G2 — signed authorization (control/signing.py).

The load-bearing security property: a control decision is signed by the
operator's key and the AGENT verifies it before acting; a tampered or
wrongly-keyed decision is rejected, and missing/malformed inputs fail closed
(deny). These exercise the Ed25519 dev path; production uses WebAuthn through
the same verify seam.

`importorskip("cryptography")` so the suite stays green where the optional
crypto library is absent — verify_decision still fails closed there, but a
sign/verify roundtrip needs the library to run at all.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("cryptography")

from crystal_cache.control.signing import (  # noqa: E402
    generate_keypair_dev,
    is_timestamp_fresh,
    sign_decision_dev,
    verify_decision,
)

_FIELDS = dict(
    session_id="sess_1",
    request_id="req_1",
    decision="approve",
    nonce="nonce-abc",
    timestamp="2026-06-15T02:00:00+00:00",
)


def test_sign_verify_roundtrip():
    priv, pub = generate_keypair_dev()
    sig = sign_decision_dev(priv, **_FIELDS)
    assert verify_decision(pub, signature_b64=sig, **_FIELDS) is True


def test_tampered_decision_rejected():
    priv, pub = generate_keypair_dev()
    sig = sign_decision_dev(priv, **_FIELDS)
    tampered = {**_FIELDS, "decision": "deny"}  # flip approve→deny
    assert verify_decision(pub, signature_b64=sig, **tampered) is False


def test_tampered_request_id_rejected():
    priv, pub = generate_keypair_dev()
    sig = sign_decision_dev(priv, **_FIELDS)
    tampered = {**_FIELDS, "request_id": "req_2"}
    assert verify_decision(pub, signature_b64=sig, **tampered) is False


def test_wrong_key_rejected():
    priv, _pub = generate_keypair_dev()
    _priv2, pub2 = generate_keypair_dev()
    sig = sign_decision_dev(priv, **_FIELDS)
    assert verify_decision(pub2, signature_b64=sig, **_FIELDS) is False


def test_missing_key_or_signature_fails_closed():
    priv, pub = generate_keypair_dev()
    sig = sign_decision_dev(priv, **_FIELDS)
    assert verify_decision(None, signature_b64=sig, **_FIELDS) is False
    assert verify_decision(pub, signature_b64=None, **_FIELDS) is False


def test_malformed_inputs_fail_closed():
    assert (
        verify_decision("!!notbase64!!", signature_b64="!!notbase64!!", **_FIELDS)
        is False
    )


def test_timestamp_freshness():
    now = datetime(2026, 6, 15, 2, 0, 0, tzinfo=timezone.utc)
    assert is_timestamp_fresh(now.isoformat(), now=now, max_age_seconds=300) is True
    old = (now - timedelta(seconds=600)).isoformat()
    assert is_timestamp_fresh(old, now=now, max_age_seconds=300) is False
    # Unparseable timestamp is treated as stale (fail-closed).
    assert is_timestamp_fresh("not-a-timestamp", now=now) is False
