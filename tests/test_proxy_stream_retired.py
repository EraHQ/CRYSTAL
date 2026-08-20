"""Streaming is refused on /v1/chat/completions (2026-08-19 dead-code sweep).

WHY THIS TEST EXISTS
--------------------
`_stream_chat_completion` wrote a QueryLog and emitted NO
`record_model_call`. Because `enforce_managed_budget`
(`control/admission.py`) refuses on a SUM over `llm_calls`, `stream: true`
was a **caller-selectable opt-out** from the managed monthly cap, from the
task-key budget, and from every /v1/cost/* view. P0.57 also suppressed MCR
on streaming, so those turns produced no trace either. The suite had no
streaming test at all, which is exactly why that survived.

The fix is refusal rather than metering: this surface is being retired in
favour of POST /v1/agent/messages, which streams over SSE (shipped
2026-07-21) and runs the same `finalize_agent_turn` on both delivery
shapes.

This test pins the refusal so the branch cannot be quietly reintroduced —
and it must fail loudly (400, not 500), before any pipeline work, so no
retrieval, no upstream call and no ledger row can happen on the way.

Direct-call convention, per the rest of the endpoint suite; asyncio_mode=auto.
"""
from __future__ import annotations

import pytest

from crystal_cache.endpoints.chat_proxy import run_chat_completion
from crystal_cache.ingress.errors import InvalidRequestError
from crystal_cache.ingress.schema import ChatCompletionRequest
from crystal_cache.models.customer import Customer, ModelRoutingConfig


class _ExplodingRequest:
    """Any attribute access is a test failure.

    The guard must fire before `request.app.state` is touched, so a real
    Request is not merely unnecessary — reaching for one proves the guard
    ran too late.
    """

    def __getattr__(self, name: str):  # pragma: no cover - defensive
        raise AssertionError(
            f"request.{name} was accessed — the stream guard must refuse "
            f"before any pipeline work"
        )


def _customer() -> Customer:
    return Customer(
        id="cus_stream_guard",
        model_routing_config=ModelRoutingConfig(
            provider="anthropic",
            model_id="claude-sonnet-4-5-20250929",
            api_key_ref="enc:v2:irrelevant",
        ),
    )


async def test_streaming_is_refused_with_400(store):
    body = ChatCompletionRequest(
        model="claude-sonnet-4-5-20250929",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    with pytest.raises(InvalidRequestError) as exc:
        await run_chat_completion(
            body=body,
            request=_ExplodingRequest(),
            customer=_customer(),
            store=store,
        )
    err = exc.value
    assert err.http_status == 400
    assert err.error_type == "invalid_request_error"
    assert err.param == "stream"
    assert err.code == "stream_retired"
    # The message must point at the surviving surface, not just say "no".
    assert "/v1/agent/messages" in err.message


async def test_streaming_generators_are_gone():
    """The deleted generators must not come back by copy-paste.

    Pins absence rather than behavior on purpose: if either name reappears
    on this module, someone has restored an unmetered spend path.
    """
    from crystal_cache.endpoints import chat_proxy

    assert not hasattr(chat_proxy, "_stream_chat_completion")
    assert not hasattr(chat_proxy, "_stream_cache_hit")


async def test_non_streaming_request_passes_the_guard(store):
    """The guard is narrow: `stream` unset must not be refused.

    The pipeline cannot complete here (no app.state, no real upstream
    credential), so this asserts on the *kind* of failure rather than on
    success: whatever goes wrong downstream, it must not be the stream
    refusal. That is precisely the property that would break if someone
    widened the guard.
    """
    body = ChatCompletionRequest(
        model="claude-sonnet-4-5-20250929",
        messages=[{"role": "user", "content": "hi"}],
    )
    with pytest.raises(Exception) as exc:
        await run_chat_completion(
            body=body,
            request=_ExplodingRequest(),
            customer=_customer(),
            store=store,
        )
    assert getattr(exc.value, "code", None) != "stream_retired"
