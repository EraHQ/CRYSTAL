"""Audit (e) stage 1.6 — agent-side sequence-id inference (must-port #9,
Q5=A verbatim, ratified 2026-08-26).

The agent endpoint resolves sequence_id metadata → header → server-inferred
(seq_{sha256(customer_id \\x00 first_user_text)[:16]}), exactly the proxy's
recipe — so one conversation lands ONE id whichever surface serves a turn.
The inference is what kills the S2-246 chain: no more sequence-less agent
turns, so no more empty citation-credit idempotency keys.

Includes the cross-surface parity pin: while chat_proxy still exists, its
`_resolve_sequence_id` and the agent's must return IDENTICAL ids for the
same inputs. When Stage 3 deletes the proxy, that one test goes with it —
the recipe assertions below keep pinning the digest independently.

asyncio_mode=auto (all sync here, but consistent with the suite).
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

from crystal_cache.endpoints.agent import _resolve_sequence_id


def _req(headers: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(headers=headers or {})


def _body(metadata: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(metadata=metadata)


_MSGS = [{"role": "user", "content": "first question"}]


def _expected(customer_id: str, first_user_text: str) -> str:
    digest = hashlib.sha256(
        f"{customer_id}\x00{first_user_text}".encode("utf-8")
    ).hexdigest()
    return f"seq_{digest[:16]}"


def test_metadata_wins_over_header_and_inference():
    out = _resolve_sequence_id(
        request=_req({"x-sequence-id": "seq_header"}),
        body=_body({"sequence_id": "  seq_meta  "}),
        customer_id="cus_1", messages=_MSGS,
    )
    assert out == "seq_meta"


def test_header_second():
    out = _resolve_sequence_id(
        request=_req({"x-sequence-id": "seq_header"}),
        body=_body(None),
        customer_id="cus_1", messages=_MSGS,
    )
    assert out == "seq_header"


def test_inference_third_and_deterministic():
    out = _resolve_sequence_id(
        request=_req(), body=_body(None),
        customer_id="cus_1", messages=_MSGS,
    )
    assert out == _expected("cus_1", "first question")
    assert out.startswith("seq_")
    # Same conversation, later turn (history grew): SAME id — the first
    # user message is the stable anchor.
    longer = _MSGS + [
        {"role": "assistant", "content": "an answer"},
        {"role": "user", "content": "a follow-up"},
    ]
    again = _resolve_sequence_id(
        request=_req(), body=_body(None),
        customer_id="cus_1", messages=longer,
    )
    assert again == out


def test_inference_scopes_by_customer():
    a = _resolve_sequence_id(
        request=_req(), body=_body(None),
        customer_id="cus_a", messages=_MSGS,
    )
    b = _resolve_sequence_id(
        request=_req(), body=_body(None),
        customer_id="cus_b", messages=_MSGS,
    )
    assert a != b


def test_multi_block_user_content_joins_text_parts():
    msgs = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "part one "},
            {"type": "tool_result", "content": "ignored"},
            {"type": "text", "text": "part two"},
        ],
    }]
    out = _resolve_sequence_id(
        request=_req(), body=_body(None),
        customer_id="cus_1", messages=msgs,
    )
    assert out == _expected("cus_1", "part one part two")


def test_no_user_message_returns_none():
    out = _resolve_sequence_id(
        request=_req(), body=_body(None),
        customer_id="cus_1",
        messages=[{"role": "assistant", "content": "orphan"}],
    )
    assert out is None


def test_cross_surface_parity_with_the_proxy():
    """One conversation, one id, either surface (Q5=A). Dies with the proxy
    at Stage 3; the digest pins above outlive it."""
    from crystal_cache.endpoints.chat_proxy import (
        _resolve_sequence_id as proxy_resolve,
    )

    for msgs in (
        _MSGS,
        [{"role": "user", "content": [
            {"type": "text", "text": "blocky opener"},
        ]}],
    ):
        agent_id = _resolve_sequence_id(
            request=_req(), body=_body(None),
            customer_id="cus_parity", messages=msgs,
        )
        proxy_id = proxy_resolve(
            request=_req(), body=_body(None),
            customer_id="cus_parity", messages=msgs,
        )
        assert agent_id == proxy_id
