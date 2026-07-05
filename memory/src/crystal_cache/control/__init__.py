"""Control plane (Growth G2) — the control package.

The outbound-poll command channel's server-side half. The agent always
initiates (heartbeats + polls); nothing connects inbound (NAT-safe). The
persistence + claim state machine lives in
infrastructure/metadata_store_control_ext.py; the HTTP surface in
endpoints/control.py; this package holds the signed-authorization primitive
(signing.py).

**Signed authorization, not P2P transport.** The property wanted is that no
middleman — not even our own server — can authorize a dangerous action. TLS is
the transport floor; on top of it the operator's decision is *signed by the
operator's key* and the AGENT verifies it before acting. The server is a
courier that cannot forge.
"""
from .signing import (
    DECISION_TIMEOUT_SECONDS,
    canonical_payload,
    is_timestamp_fresh,
    verify_decision,
)

__all__ = [
    "DECISION_TIMEOUT_SECONDS",
    "canonical_payload",
    "is_timestamp_fresh",
    "verify_decision",
]
