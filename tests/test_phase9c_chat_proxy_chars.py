"""Phase 9C characterization + integration tests for chat_proxy.

Per P0.55 + P0.61: seven tests covering the chat_proxy MCR integration.
Tests P1-P4 characterize current chat_proxy behavior BEFORE the
emission integration would affect them; tests P5-P7 verify the NEW
Phase 9C behavior (MCR emission + handle_signals mcr_enabled=True flip).

Test scope (P0.61):
  CHARACTERIZATION (pin down current behavior):
    P1: cache-hit path returns OpenAI-shaped response + writes QueryLog
    P2: upstream-served path forwards upstream response + writes QueryLog
    P3: crystal tool injection happens only on non-streaming requests
    P4: handle_signals is called once after crystal tool calls extract

  INTEGRATION (Phase 9C new behavior):
    P5: cache-hit calls emit_mcr_artifacts with skip_self_critique=True
    P6: upstream-served calls emit_mcr_artifacts with full self-critique
    P7: handle_signals receives mcr_enabled=True; push_gap produces MCR

Phase 8 / 8.5 / 9A / 9B test files are NOT touched.

Per P0.57: streaming MCR is deferred to a future phase. Phase 9C does
NOT test the streaming path.

Testing approach: chat_completions is called as an async function with
its FastAPI dependencies pre-resolved (customer, store passed
directly). The Request parameter is a small FakeRequest carrying just
the bits chat_completions actually reads (`request.headers.get`,
`request.app.state.*`). The upstream LLM is replaced via monkeypatch
on `endpoints.chat_proxy.get_upstream_client`. The retrieval pipeline
is replaced via monkeypatch on `endpoints.chat_proxy.retrieve_and_inject`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest

from crystal_cache.endpoints import chat_proxy as chat_proxy_module
from crystal_cache.endpoints.chat_proxy import chat_completions
from crystal_cache.llm import reset_llm_client, set_llm_client
from crystal_cache.execution.upstream_client import UpstreamResponse
from crystal_cache.ingress.schema import ChatCompletionRequest, ChatMessage
from crystal_cache.retrieval.pipeline import RetrievalOutcome


# ---------------------------------------------------------------------------
# Test fakes — minimal surface for chat_proxy dependencies
# ---------------------------------------------------------------------------

@dataclass
class _FakeAppState:
    """Mirrors what chat_proxy reads via `request.app.state.*`."""
    vector_store: Any = None
    vector_index: Any = None
    prompt_encoder: Any = None
    decomposer: Any = None
    dsl_config_store: Any = None
    decoder_loader: Any = None
    fact_vector_store: Any = None
    shadow_evaluator: Any = None
    mem0: Any = None


@dataclass
class _FakeApp:
    state: _FakeAppState = field(default_factory=_FakeAppState)


@dataclass
class _FakeRequest:
    """Minimal Request-shape stand-in. chat_proxy reads:
        - request.headers.get("x-sequence-id")
        - request.app.state.*
    Both surfaces are populated here.
    """
    _headers: dict[str, str] = field(default_factory=dict)
    app: _FakeApp = field(default_factory=_FakeApp)

    @property
    def headers(self) -> dict[str, str]:
        return self._headers


class _FakeUpstreamClient:
    """Fake of execution.upstream_client.UpstreamClient.

    chat_proxy calls client.complete(messages, model, ...). We script
    one or more responses and record the call args for assertion.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._scripted: list[UpstreamResponse] = []

    def script(self, response: UpstreamResponse) -> None:
        self._scripted.append(response)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> UpstreamResponse:
        self.calls.append({
            "messages": list(messages),
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "kwargs": dict(kwargs),
        })
        if not self._scripted:
            raise AssertionError(
                f"_FakeUpstreamClient.complete called "
                f"{len(self.calls)} time(s), no more scripted responses."
            )
        return self._scripted.pop(0)


def _make_upstream_response(
    *,
    assistant_text: str,
    crystal_tool_calls: Optional[list[dict[str, Any]]] = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> UpstreamResponse:
    """Construct an UpstreamResponse shaped like the real provider clients."""
    message: dict[str, Any] = {
        "role": "assistant",
        "content": assistant_text,
    }
    if crystal_tool_calls:
        message["tool_calls"] = crystal_tool_calls

    openai_format = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "claude-sonnet-4-5-20250929",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

    return UpstreamResponse(
        openai_format=openai_format,
        latency_ms=10,
        provider="fake",
        model_id="claude-sonnet-4-5-20250929",
        assistant_text=assistant_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


async def _make_request_body(
    *,
    model: str = "claude-sonnet-4-5-20250929",
    user_text: str = "What's the deadline?",
    stream: bool = False,
    sequence_id: Optional[str] = None,
) -> ChatCompletionRequest:
    """Build a minimal ChatCompletionRequest for tests."""
    metadata: Optional[dict[str, Any]] = None
    if sequence_id is not None:
        metadata = {"sequence_id": sequence_id}
    return ChatCompletionRequest(
        model=model,
        messages=[
            ChatMessage(role="user", content=user_text),
        ],
        stream=stream,
        metadata=metadata,  # type: ignore[arg-type]
    )


def _patch_retrieve_and_inject(
    monkeypatch: pytest.MonkeyPatch,
    *,
    outcome: RetrievalOutcome,
) -> None:
    """Replace chat_proxy's `retrieve_and_inject` with an async stub."""
    async def _fake(*args: Any, **kwargs: Any) -> RetrievalOutcome:
        return outcome

    monkeypatch.setattr(chat_proxy_module, "retrieve_and_inject", _fake)


def _patch_upstream_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    client: _FakeUpstreamClient,
) -> None:
    """Replace chat_proxy's `get_upstream_client` factory."""

    async def _fake_factory(customer: Any, store: Any = None) -> _FakeUpstreamClient:
        return client

    monkeypatch.setattr(chat_proxy_module, "get_upstream_client", _fake_factory)


def _empty_outcome(messages: list[dict[str, Any]]) -> RetrievalOutcome:
    """Build a 'no match' RetrievalOutcome."""
    return RetrievalOutcome(
        messages=messages,
        match_type="none",
        injection_method="none",
        matched_crystal_ids=[],
        top_score=0.0,
    )


def _cache_hit_outcome(
    messages: list[dict[str, Any]],
    *,
    answer: str = "Cached answer.",
    crystal_id: str = "cry_cache_test",
) -> RetrievalOutcome:
    """Build a cache-hit RetrievalOutcome."""
    return RetrievalOutcome(
        messages=messages,
        match_type="high",
        injection_method="cache_hit",
        matched_crystal_ids=[crystal_id],
        top_score=0.95,
        cache_hit_response=answer,
        cache_hit_crystal_id=crystal_id,
        routing_top1=0.95,
        routing_top2=0.30,
    )


# ===========================================================================
# CHARACTERIZATION TESTS (P1-P4)
# ===========================================================================


# ---------------------------------------------------------------------------
# P1 — Cache-hit path returns OpenAI-shaped response + writes QueryLog
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_p1_cache_hit_returns_openai_shape_and_writes_query_log(
    customer: Any,
    store: Any,
    monkeypatch: pytest.MonkeyPatch,
):
    """Cache-hit path: returns OpenAI-shaped JSON response, writes a
    QueryLog row with upstream_call_made=False and
    injection_method='cache_hit'.
    """
    body = await _make_request_body(user_text="What's the deadline?")

    outcome = _cache_hit_outcome(
        [m.model_dump(exclude_none=True) for m in body.messages],
        answer="The deadline is April 1, 2027.",
        crystal_id="cry_p1",
    )
    _patch_retrieve_and_inject(monkeypatch, outcome=outcome)

    fake_client = _FakeUpstreamClient()
    _patch_upstream_client(monkeypatch, client=fake_client)

    response = await chat_completions(
        body=body,
        request=_FakeRequest(),
        customer=customer,
        store=store,
    )

    payload = json.loads(response.body)

    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert payload["choices"][0]["message"]["content"] == (
        "The deadline is April 1, 2027."
    )
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["usage"]["prompt_tokens"] == 0
    assert payload["usage"]["completion_tokens"] == 0
    assert payload["id"].startswith("chatcmpl-cache-cry_p1")

    # Upstream was NOT called.
    assert fake_client.calls == []

    # QueryLog row was written. The store's helper returns
    # (total_count, rows).
    total, logs = await store.list_query_logs_for_customer(
        customer_id=customer.id,
        limit=10,
    )
    assert total == 1
    assert len(logs) == 1
    log = logs[0]
    assert log.upstream_call_made is False
    assert log.injection_method == "cache_hit"
    assert log.routed_crystal_id == "cry_p1"


# ---------------------------------------------------------------------------
# P2 — Upstream-served path forwards upstream response + writes QueryLog
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_p2_upstream_served_forwards_response_and_writes_query_log(
    customer: Any,
    store: Any,
    monkeypatch: pytest.MonkeyPatch,
):
    """Upstream-served path: calls the upstream client, forwards its
    response, writes a QueryLog with upstream_call_made=True.
    """
    body = await _make_request_body(user_text="What is 2+2?")

    outcome = _empty_outcome(
        [m.model_dump(exclude_none=True) for m in body.messages],
    )
    _patch_retrieve_and_inject(monkeypatch, outcome=outcome)

    fake_client = _FakeUpstreamClient()
    fake_client.script(_make_upstream_response(
        assistant_text="2 + 2 equals 4.",
    ))
    _patch_upstream_client(monkeypatch, client=fake_client)

    response = await chat_completions(
        body=body,
        request=_FakeRequest(),
        customer=customer,
        store=store,
    )

    payload = json.loads(response.body)
    assert payload["choices"][0]["message"]["content"] == "2 + 2 equals 4."
    assert payload["usage"]["prompt_tokens"] == 100
    assert payload["usage"]["completion_tokens"] == 50

    # Upstream was called exactly once.
    assert len(fake_client.calls) == 1
    assert fake_client.calls[0]["model"] == "claude-sonnet-4-5-20250929"

    total, logs = await store.list_query_logs_for_customer(
        customer_id=customer.id,
        limit=10,
    )
    assert total == 1
    log = logs[0]
    assert log.upstream_call_made is True
    assert log.injection_method == "none"


# ---------------------------------------------------------------------------
# P3 — Crystal tool injection happens only when fact_vector_store present
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_p3_crystal_tools_injected_when_fact_store_present(
    customer: Any,
    store: Any,
    fact_vector_store: Any,
    monkeypatch: pytest.MonkeyPatch,
):
    """The crystal push/pull tools are injected when app.state
    .fact_vector_store is configured (and the request is non-streaming).
    """
    body = await _make_request_body(user_text="Tell me something.")

    outcome = _empty_outcome(
        [m.model_dump(exclude_none=True) for m in body.messages],
    )
    _patch_retrieve_and_inject(monkeypatch, outcome=outcome)

    fake_client = _FakeUpstreamClient()
    fake_client.script(_make_upstream_response(assistant_text="Sure."))
    _patch_upstream_client(monkeypatch, client=fake_client)

    req = _FakeRequest(
        app=_FakeApp(state=_FakeAppState(fact_vector_store=fact_vector_store)),
    )

    await chat_completions(
        body=body,
        request=req,
        customer=customer,
        store=store,
    )

    assert len(fake_client.calls) == 1
    tools = fake_client.calls[0]["kwargs"].get("tools") or []
    tool_names = [t.get("function", {}).get("name", "") for t in tools]
    assert "crystal_push_gap" in tool_names
    assert "crystal_push_correct" in tool_names
    assert "crystal_push_store" in tool_names


# ---------------------------------------------------------------------------
# P4 — handle_signals is called once after crystal tool calls extract
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_p4_handle_signals_called_once_with_crystal_calls(
    customer: Any,
    store: Any,
    fact_vector_store: Any,
    monkeypatch: pytest.MonkeyPatch,
):
    """When the upstream LLM emits a crystal tool call, the proxy
    extracts it and calls handle_signals once with the parsed signals.
    """
    body = await _make_request_body(user_text="What's in the bank?")

    outcome = _empty_outcome(
        [m.model_dump(exclude_none=True) for m in body.messages],
    )
    _patch_retrieve_and_inject(monkeypatch, outcome=outcome)

    gap_tool_call = {
        "id": "tc_gap_p4",
        "type": "function",
        "function": {
            "name": "crystal_push_gap",
            "arguments": json.dumps({
                "domain": "test",
                "subject": "p4",
                "missing": "info about p4",
            }),
        },
    }
    fake_client = _FakeUpstreamClient()
    fake_client.script(_make_upstream_response(
        assistant_text="I don't know.",
        crystal_tool_calls=[gap_tool_call],
    ))
    _patch_upstream_client(monkeypatch, client=fake_client)

    call_count = {"n": 0}
    captured_kwargs: dict[str, Any] = {}
    original_handle = chat_proxy_module.handle_signals

    async def _spy_handle_signals(
        signals: Any,
        customer_id: str,
        store: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        call_count["n"] += 1
        captured_kwargs.update(kwargs)
        return await original_handle(signals, customer_id, store, **kwargs)

    monkeypatch.setattr(chat_proxy_module, "handle_signals", _spy_handle_signals)

    req = _FakeRequest(
        app=_FakeApp(state=_FakeAppState(fact_vector_store=fact_vector_store)),
    )

    await chat_completions(
        body=body,
        request=req,
        customer=customer,
        store=store,
    )

    assert call_count["n"] == 1
    assert "conversation_context" in captured_kwargs


# ===========================================================================
# INTEGRATION TESTS (P5-P7) — Phase 9C new behavior
# ===========================================================================


# ---------------------------------------------------------------------------
# P5 — Cache-hit emits trace without self-critique
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_p5_cache_hit_emits_trace_without_self_critique(
    customer: Any,
    store: Any,
    monkeypatch: pytest.MonkeyPatch,
):
    """Per P0.58: cache-hit path persists a ReasoningTrace via
    emit_mcr_artifacts(skip_self_critique=True). No Critique row
    is written.
    """
    body = await _make_request_body(
        user_text="What's the deadline?",
        sequence_id="seq_p5",
    )

    outcome = _cache_hit_outcome(
        [m.model_dump(exclude_none=True) for m in body.messages],
        answer="April 1, 2027.",
        crystal_id="cry_p5",
    )
    _patch_retrieve_and_inject(monkeypatch, outcome=outcome)
    _patch_upstream_client(monkeypatch, client=_FakeUpstreamClient())

    await chat_completions(
        body=body,
        request=_FakeRequest(),
        customer=customer,
        store=store,
    )

    # No critique on cache hit.
    critiques = await store.list_critiques_for_sequence(
        customer_id=customer.id,
        sequence_id="seq_p5",
    )
    assert critiques == []

    # Trace landed via the soft-join key.
    traces = await store.list_traces_for_sequence(
        customer_id=customer.id,
        sequence_id="seq_p5",
    )
    assert len(traces) == 1
    trace = traces[0]
    assert trace.sequence_id == "seq_p5"
    assert "cry_p5" in trace.crystals_used
    # The trailing final_text event carries the served answer.
    final_event = trace.events[-1]
    assert final_event.get("type") == "final_text"
    assert "April 1" in final_event.get("text", "")


# ---------------------------------------------------------------------------
# P6 — Upstream-served calls emit_mcr_artifacts with full self-critique
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_p6_upstream_served_emits_trace_and_critique(
    customer: Any,
    store: Any,
    monkeypatch: pytest.MonkeyPatch,
):
    """Per P0.55 + P0.60: upstream-served path calls emit_mcr_artifacts
    with full self-critique. A trace persists; a critique persists.
    """
    body = await _make_request_body(
        user_text="Tell me about the project.",
        sequence_id="seq_p6",
    )

    outcome = _empty_outcome(
        [m.model_dump(exclude_none=True) for m in body.messages],
    )
    _patch_retrieve_and_inject(monkeypatch, outcome=outcome)

    fake_client = _FakeUpstreamClient()
    fake_client.script(_make_upstream_response(
        assistant_text="The project is on schedule.",
    ))
    _patch_upstream_client(monkeypatch, client=fake_client)

    # Drive the self-critique through the seam: inject a scripted fake
    # LLM client. is_ready() is True so chat_proxy does NOT skip
    # self-critique, and complete() returns this canned critique.
    from fakes import FakeAnthropic
    fake_anthropic_instance = FakeAnthropic()
    fake_anthropic_instance.script_text(json.dumps({
        "observations": [
            {
                "type": "gap_papered_over",
                "text": "Response was vague about specifics.",
                "confidence": 0.7,
                "anchors": [],
            }
        ],
        "action_items": [],
        "summary_text": "Generic response.",
    }))
    set_llm_client(fake_anthropic_instance)

    try:
        await chat_completions(
            body=body,
            request=_FakeRequest(),
            customer=customer,
            store=store,
        )
    finally:
        reset_llm_client()

    # A trace landed.
    traces = await store.list_traces_for_sequence(
        customer_id=customer.id,
        sequence_id="seq_p6",
    )
    assert len(traces) == 1
    trace = traces[0]
    assert trace.sequence_id == "seq_p6"

    # A critique landed.
    critiques = await store.list_critiques_for_sequence(
        customer_id=customer.id,
        sequence_id="seq_p6",
    )
    assert len(critiques) == 1
    critique = critiques[0]
    assert critique.critic_role == "agent_self"
    assert len(critique.observations) == 1
    assert critique.observations[0]["type"] == "gap_papered_over"
    assert critique.summary_text == "Generic response."


# ---------------------------------------------------------------------------
# P7 — handle_signals receives mcr_enabled=True; push_gap produces MCR
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_p7_handle_signals_mcr_enabled_flips_to_true(
    customer: Any,
    store: Any,
    fact_vector_store: Any,
    monkeypatch: pytest.MonkeyPatch,
):
    """Per P0.59: the chat_proxy now passes mcr_enabled=True to
    handle_signals, activating Phase 9B's BD-3 + BD-11 writes in
    production.
    """
    body = await _make_request_body(
        user_text="What is the obscure thing?",
        sequence_id="seq_p7",
    )

    outcome = _empty_outcome(
        [m.model_dump(exclude_none=True) for m in body.messages],
    )
    _patch_retrieve_and_inject(monkeypatch, outcome=outcome)

    gap_tool_call = {
        "id": "tc_gap_p7",
        "type": "function",
        "function": {
            "name": "crystal_push_gap",
            "arguments": json.dumps({
                "domain": "engineering",
                "subject": "ci runner",
                "missing": "which CI runner this project uses",
            }),
        },
    }
    fake_client = _FakeUpstreamClient()
    fake_client.script(_make_upstream_response(
        assistant_text="I don't know.",
        crystal_tool_calls=[gap_tool_call],
    ))
    _patch_upstream_client(monkeypatch, client=fake_client)

    # Inject a not-ready seam so the proxy's emit_mcr_artifacts passes
    # skip_self_critique=True. This isolates the BD-11 signal-handler-side
    # MCR write as the only Critique row in the sequence.
    class _NotReadyClient:
        def is_ready(self) -> bool:
            return False

    set_llm_client(_NotReadyClient())

    req = _FakeRequest(
        app=_FakeApp(state=_FakeAppState(fact_vector_store=fact_vector_store)),
    )

    try:
        await chat_completions(
            body=body,
            request=req,
            customer=customer,
            store=store,
        )
    finally:
        reset_llm_client()

    # Expect exactly 1 critique (from Phase 9B's gap-MCR write);
    # the proxy's emit_mcr_artifacts skipped self-critique due to
    # empty api_key.
    critiques = await store.list_critiques_for_sequence(
        customer_id=customer.id,
        sequence_id="seq_p7",
    )
    assert len(critiques) == 1
    critique = critiques[0]
    assert critique.critic_role == "agent_self"
    assert critique.observations[0]["type"] == "gap_papered_over"
    assert critique.critic_model == "claude-sonnet-4-5-20250929"

    # The ActionItem(gap_declaration) carries conversation_context.
    items = await store.list_action_items_for_critique(critique.id)
    assert len(items) == 1
    item = items[0]
    assert item.action_type == "gap_declaration"
    assert item.content["want"] == "which CI runner this project uses"
    assert item.content["why_needed"] == "ci runner"
    assert "obscure thing" in item.content["conversation_context"]


# ===========================================================================
# PHASE 11.5 CHARS GAPS (P8-P9) — CU-22 scoped backfill per P0.104
# ===========================================================================

# Why these two specifically:
#   The Phase 9C chars (P1-P7) cover the emission paths but leave
#   sequence_id derivation untested. sequence_id is the
#   conversation-tracking primitive; a regression here would scramble
#   MCR cross-trace correlation (the metacognitive layer would see
#   each turn as a separate conversation). Per P0.97, chat_proxy stays
#   permanently — so investing in chars for its sequence handling is
#   defensible. Streaming chars (P0.57-deferred) remain out of scope
#   per P0.106's design-only decision on CU-21.


# ---------------------------------------------------------------------------
# P8 — sequence_id derivation from x-sequence-id header (CU-22)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_p8_sequence_id_from_x_sequence_id_header(
    customer: Any,
    store: Any,
    monkeypatch: pytest.MonkeyPatch,
):
    """CU-22 / P0.104 — when the request carries the
    `x-sequence-id` header, the chat_proxy uses its value as the
    sequence_id on the persisted ReasoningTrace, OVERRIDING any
    metadata-supplied or hash-derived id.

    Verifies that the soft-join key for MCR cross-trace correlation
    honors the header. Regressions here would silently scramble
    sequence ids across turns when the caller is explicitly tracking
    a conversation.
    """
    body = await _make_request_body(user_text="Test header path")

    outcome = _empty_outcome(
        [m.model_dump(exclude_none=True) for m in body.messages],
    )
    _patch_retrieve_and_inject(monkeypatch, outcome=outcome)

    fake_client = _FakeUpstreamClient()
    fake_client.script(_make_upstream_response(assistant_text="OK."))
    _patch_upstream_client(monkeypatch, client=fake_client)

    # Suppress self-critique by leaving the api key empty — isolates
    # the trace's sequence_id assertion from Anthropic mocking.
    monkeypatch.setattr(
        chat_proxy_module.settings,
        "anthropic_api_key",
        "",
    )

    # Explicit sequence id via the header.
    explicit_seq_id = "seq_p8_from_header_abc123"
    req = _FakeRequest(_headers={"x-sequence-id": explicit_seq_id})

    await chat_completions(
        body=body,
        request=req,
        customer=customer,
        store=store,
    )

    # Trace's sequence_id matches the header value verbatim.
    traces = await store.list_traces_for_sequence(
        customer_id=customer.id,
        sequence_id=explicit_seq_id,
    )
    assert len(traces) == 1, (
        f"Expected exactly 1 trace under sequence_id={explicit_seq_id!r}; "
        f"got {len(traces)} — sequence_id derivation may have ignored the "
        f"x-sequence-id header."
    )
    assert traces[0].sequence_id == explicit_seq_id


# ---------------------------------------------------------------------------
# P9 — multi-turn conversation preserves sequence_id (CU-22)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_p9_multi_turn_preserves_sequence_id(
    customer: Any,
    store: Any,
    monkeypatch: pytest.MonkeyPatch,
):
    """CU-22 / P0.104 — in the absence of an x-sequence-id header or
    metadata.sequence_id, the chat_proxy derives sequence_id from the
    first user message. Two requests carrying the same first user
    message — a real multi-turn conversation — must share a
    sequence_id, so the metacognitive layer can correlate the turns.
    A request with a different first user message must produce a
    DIFFERENT sequence_id.

    This is the conversation-tracking primitive's stability test. A
    regression would either (a) tag every turn of one conversation as
    a separate sequence (breaks cross-trace correlation), or (b) tag
    different conversations as the same sequence (incorrect
    correlation).
    """
    from crystal_cache.ingress.schema import ChatCompletionRequest, ChatMessage

    # No header, no metadata.sequence_id — force hash-of-first-message
    # derivation.
    req = _FakeRequest()

    # Empty outcome + scripted no-op response for every call.
    def _outcome_for(messages: list[dict[str, Any]]) -> RetrievalOutcome:
        return _empty_outcome(messages)

    fake_client = _FakeUpstreamClient()
    fake_client.script(_make_upstream_response(assistant_text="R1."))
    fake_client.script(_make_upstream_response(assistant_text="R2."))
    fake_client.script(_make_upstream_response(assistant_text="R3."))
    _patch_upstream_client(monkeypatch, client=fake_client)
    monkeypatch.setattr(
        chat_proxy_module.settings,
        "anthropic_api_key",
        "",
    )

    # Turn 1 of conversation A: first user message is "hello there".
    body_t1 = ChatCompletionRequest(
        model="claude-sonnet-4-5-20250929",
        messages=[ChatMessage(role="user", content="hello there")],
        stream=False,
    )
    _patch_retrieve_and_inject(
        monkeypatch,
        outcome=_outcome_for(
            [m.model_dump(exclude_none=True) for m in body_t1.messages]
        ),
    )
    await chat_completions(
        body=body_t1, request=req, customer=customer, store=store,
    )

    # Turn 2 of the SAME conversation: first user message still
    # "hello there" but more turns follow.
    body_t2 = ChatCompletionRequest(
        model="claude-sonnet-4-5-20250929",
        messages=[
            ChatMessage(role="user", content="hello there"),
            ChatMessage(role="assistant", content="R1."),
            ChatMessage(role="user", content="how are you"),
        ],
        stream=False,
    )
    _patch_retrieve_and_inject(
        monkeypatch,
        outcome=_outcome_for(
            [m.model_dump(exclude_none=True) for m in body_t2.messages]
        ),
    )
    await chat_completions(
        body=body_t2, request=req, customer=customer, store=store,
    )

    # Turn 1 of a DIFFERENT conversation: different first user message.
    body_diff = ChatCompletionRequest(
        model="claude-sonnet-4-5-20250929",
        messages=[ChatMessage(role="user", content="goodbye world")],
        stream=False,
    )
    _patch_retrieve_and_inject(
        monkeypatch,
        outcome=_outcome_for(
            [m.model_dump(exclude_none=True) for m in body_diff.messages]
        ),
    )
    await chat_completions(
        body=body_diff, request=req, customer=customer, store=store,
    )

    # Get all traces for this customer. Filter by hash-derived
    # sequence_id via the list_traces helper isn't possible without
    # knowing the hash; instead, fetch all critiques (none) and pull
    # sequence ids off the traces directly via a per-trace lookup.
    # The customer's traces are listable via list_traces_for_sequence
    # only when we know the sequence_id. So: read raw via the
    # ReasoningTraceRow table.
    from crystal_cache.infrastructure.schema import ReasoningTraceRow
    from sqlalchemy import select

    async with store.session() as session:
        stmt = (
            select(ReasoningTraceRow)
            .where(ReasoningTraceRow.customer_id == customer.id)
            .order_by(ReasoningTraceRow.created_at.asc())
        )
        rows = (await session.execute(stmt)).scalars().all()
        trace_seq_ids = [r.sequence_id for r in rows]

    assert len(trace_seq_ids) == 3, (
        f"Expected 3 traces from 3 requests; got {len(trace_seq_ids)}."
    )

    # Turn 1 and Turn 2 of conversation A share a sequence_id.
    assert trace_seq_ids[0] == trace_seq_ids[1], (
        f"Multi-turn sequence_id derivation broke: turn 1 and turn 2 "
        f"of same conversation got different sequence_ids: "
        f"{trace_seq_ids[0]!r} vs {trace_seq_ids[1]!r}."
    )

    # The different conversation gets a different sequence_id.
    assert trace_seq_ids[2] != trace_seq_ids[0], (
        f"Distinct-conversation sequence_id derivation broke: "
        f"different first messages produced the same sequence_id: "
        f"{trace_seq_ids[2]!r}."
    )


# ===========================================================================
# PHASE 12 CHARS GAPS (P10-P11) — CU-22 tool-call routing per P0.113
# ===========================================================================

# Why these two specifically:
#   P1-P9 cover the emission + sequence paths but leave the
#   crystal-vs-customer tool-call SPLIT under-tested. The proxy strips
#   crystal tool calls from the response while forwarding the
#   customer's own tool calls unchanged (the split lives in
#   extract_crystal_tool_calls + the CRYSTAL_TOOL_NAMES stripping
#   block). A regression there would either leak crystal tools to the
#   customer (confusing/garbage tool names the customer never defined)
#   or swallow the customer's real tool calls (breaking their agent
#   loop). P10 pins the coexistence contract; P11 pins resilience to
#   malformed tool_use blocks (the LLM occasionally emits structurally
#   incomplete or non-JSON tool calls — these must never 500 the
#   proxy). Streaming tool-call paths remain out of scope (P0.106 /
#   CU-21).


# ---------------------------------------------------------------------------
# P10 — customer tool calls coexist with crystal tool calls (CU-22)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_p10_customer_tool_calls_survive_crystal_strip(
    customer: Any,
    store: Any,
    fact_vector_store: Any,
    monkeypatch: pytest.MonkeyPatch,
):
    """CU-22 / P0.113 — when the upstream LLM emits BOTH a crystal
    tool call and a customer-defined tool call in the same response,
    the proxy strips the crystal call (handled internally) and
    FORWARDS the customer call unchanged in the final response.

    Regression guard: a break here either leaks crystal tools to the
    customer (tool names they never defined) or eats the customer's
    real tool call (breaking their tool-use loop).
    """
    body = await _make_request_body(
        user_text="What's the weather, and flag what you don't know?",
    )

    outcome = _empty_outcome(
        [m.model_dump(exclude_none=True) for m in body.messages],
    )
    _patch_retrieve_and_inject(monkeypatch, outcome=outcome)

    crystal_call = {
        "id": "tc_gap_p10",
        "type": "function",
        "function": {
            "name": "crystal_push_gap",
            "arguments": json.dumps({
                "domain": "test",
                "subject": "p10",
                "missing": "info about p10",
            }),
        },
    }
    customer_call = {
        "id": "tc_weather_p10",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": json.dumps({"location": "Boston"}),
        },
    }
    fake_client = _FakeUpstreamClient()
    fake_client.script(_make_upstream_response(
        assistant_text="",
        crystal_tool_calls=[crystal_call, customer_call],
        finish_reason="tool_calls",
    ))
    _patch_upstream_client(monkeypatch, client=fake_client)

    # Skip self-critique (empty api key) to isolate the tool-call
    # split from Anthropic mocking.
    monkeypatch.setattr(
        chat_proxy_module.settings,
        "anthropic_api_key",
        "",
    )

    req = _FakeRequest(
        app=_FakeApp(state=_FakeAppState(fact_vector_store=fact_vector_store)),
    )

    response = await chat_completions(
        body=body,
        request=req,
        customer=customer,
        store=store,
    )

    payload = json.loads(response.body)
    msg = payload["choices"][0]["message"]
    tool_calls = msg.get("tool_calls", [])
    names = [tc.get("function", {}).get("name", "") for tc in tool_calls]

    # Customer tool call forwarded unchanged.
    assert "get_weather" in names
    # Crystal tool call stripped from the customer-facing response.
    assert "crystal_push_gap" not in names
    # Exactly the one customer call survives.
    assert len(tool_calls) == 1
    assert tool_calls[0]["id"] == "tc_weather_p10"


# ---------------------------------------------------------------------------
# P11 — malformed tool_use blocks don't crash the proxy (CU-22)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_p11_malformed_tool_calls_do_not_crash(
    customer: Any,
    store: Any,
    fact_vector_store: Any,
    monkeypatch: pytest.MonkeyPatch,
):
    """CU-22 / P0.113 — structurally malformed tool_use blocks (a
    crystal call with non-JSON arguments, plus a block missing its
    `function` entirely) must NOT raise out of the proxy. The request
    completes, returns an OpenAI-shaped response, and the QueryLog
    still lands.

    The proxy's tool-processing block is wrapped in a fail-safe
    try/except; this test pins that contract so a future refactor
    can't turn a malformed LLM tool call into a customer-facing 500.
    """
    body = await _make_request_body(user_text="Trigger a malformed call.")

    outcome = _empty_outcome(
        [m.model_dump(exclude_none=True) for m in body.messages],
    )
    _patch_retrieve_and_inject(monkeypatch, outcome=outcome)

    # A crystal call whose arguments are not valid JSON.
    malformed_args_call = {
        "id": "tc_badjson_p11",
        "type": "function",
        "function": {
            "name": "crystal_push_gap",
            "arguments": "{not valid json at all",
        },
    }
    # A block missing the `function` key entirely.
    missing_function_call = {
        "id": "tc_nofn_p11",
        "type": "function",
    }
    fake_client = _FakeUpstreamClient()
    fake_client.script(_make_upstream_response(
        assistant_text="Here is my best effort.",
        crystal_tool_calls=[malformed_args_call, missing_function_call],
        finish_reason="tool_calls",
    ))
    _patch_upstream_client(monkeypatch, client=fake_client)

    monkeypatch.setattr(
        chat_proxy_module.settings,
        "anthropic_api_key",
        "",
    )

    req = _FakeRequest(
        app=_FakeApp(state=_FakeAppState(fact_vector_store=fact_vector_store)),
    )

    # The call must not raise.
    response = await chat_completions(
        body=body,
        request=req,
        customer=customer,
        store=store,
    )

    # Response is OpenAI-shaped with a message.
    payload = json.loads(response.body)
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["role"] == "assistant"

    # The request completed far enough to write its QueryLog.
    total, logs = await store.list_query_logs_for_customer(
        customer_id=customer.id,
        limit=10,
    )
    assert total == 1
    assert logs[0].upstream_call_made is True
