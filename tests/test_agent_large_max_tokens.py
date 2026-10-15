"""PR-8 / Q42 (lockdown 2026-10-13): the SDK's non-streaming guard is REAL,
and the agent's routing is what carries a Scale caller past it.

test_q42_streaming_above_ceiling pins the agent's CHOICE against a fake
seam. This file drives the installed anthropic SDK through the real
LLMClient with only the HTTP layer replaced (httpx.MockTransport), so
the guard that NON_STREAMING_MAX_TOKENS describes is the guard that
fires:

- a non-streaming call one token over the ceiling is refused by the SDK
  before a request exists (ValueError; the wire is never reached);
- a non-streaming call AT the ceiling reaches the wire;
- stream_messages at Scale's 32,768 reaches the wire with stream=true
  and returns the assembled final message;
- the bare Agent on a NON-streaming request at 32,768, wired to the real
  client, lands on the streaming wire with one complete answer — and
  with its routing switched off the same request dies in the SDK, which
  is what makes the routing load-bearing rather than decorative.

The client is the real anthropic.Anthropic with max_retries=0 and the
SDK's own DEFAULT_TIMEOUT: the guard runs only when
client.timeout == DEFAULT_TIMEOUT (a float silently switches it off),
and _sdk_limits() is what production passes, so the pin reads it.
"""
import json

import pytest

anthropic = pytest.importorskip("anthropic")
# anthropic 1.2.0 is built on httpx 2.x, published as the `httpx2`
# distribution (verified: anthropic/_base_client.py imports httpx2 and
# refuses an httpx 1.x Client with a TypeError). The transport handed to
# the SDK must come from the same package.
httpx = pytest.importorskip("httpx2")

from crystal_cache.agent import agent as agent_module
from crystal_cache.agent.agent import Agent
from crystal_cache.llm.client import (
    NON_STREAMING_MAX_TOKENS,
    LLMClient,
    _sdk_limits,
)

MODEL = "claude-sonnet-4-6"
_MSGS = [{"role": "user", "content": "q"}]


def _message_json(text: str = "ok") -> dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stream_body(text: str = "streamed") -> bytes:
    start = dict(_message_json(""), content=[], stop_reason=None)
    return "".join([
        _sse("message_start", {"type": "message_start", "message": start}),
        _sse("content_block_start", {
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""},
        }),
        _sse("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": text[:3]},
        }),
        _sse("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": text[3:]},
        }),
        _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 5},
        }),
        _sse("message_stop", {"type": "message_stop"}),
    ]).encode()


class _Wire:
    """Records every request the SDK actually sends; answers like the API."""

    def __init__(self) -> None:
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        if body.get("stream"):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_stream_body(),
            )
        return httpx.Response(200, json=_message_json())


def _client(wire: _Wire) -> LLMClient:
    llm = LLMClient(
        provider="anthropic",
        api_key="test-key",
        base_url=None,
        model_small=None,
        model_large=MODEL,
        model_frontier=None,
    )
    # The real SDK client, production's own limits, only the transport
    # swapped. check_installed_sdk() is skipped because the lazy slot
    # is filled; the SDK itself is what is under test here.
    limits = _sdk_limits()
    llm._anthropic_client = anthropic.Anthropic(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(wire)),
        max_retries=0,
        timeout=limits["timeout"],
    )
    assert llm._anthropic_client.timeout == anthropic.DEFAULT_TIMEOUT
    return llm


def _bare_agent(llm: LLMClient, max_tokens: int) -> Agent:
    a = Agent.__new__(Agent)
    a.llm = llm
    a.max_tokens = max_tokens
    a.model = MODEL
    a.temperature = None
    a.thinking = None
    a.stream_tokens = False   # a non-streaming request
    a.emit = None
    a.system_tail = None
    return a


def test_production_limits_keep_the_guard_switched_on():
    # The guard only runs under the SDK's DEFAULT_TIMEOUT object. A float
    # (even 600.0) would silently disable it; _sdk_limits is what the
    # production constructor passes.
    assert _sdk_limits()["timeout"] == anthropic.DEFAULT_TIMEOUT
    assert 600.0 != anthropic.DEFAULT_TIMEOUT


def test_the_sdk_refuses_one_token_over_the_ceiling_without_streaming():
    wire = _Wire()
    llm = _client(wire)
    with pytest.raises(ValueError, match="Streaming is required"):
        llm.complete_messages(
            system="s", messages=_MSGS,
            max_tokens=NON_STREAMING_MAX_TOKENS + 1, model=MODEL,
        )
    assert wire.requests == []


def test_a_non_streaming_call_at_the_ceiling_reaches_the_wire():
    wire = _Wire()
    llm = _client(wire)
    resp = llm.complete_messages(
        system="s", messages=_MSGS,
        max_tokens=NON_STREAMING_MAX_TOKENS, model=MODEL,
    )
    assert len(wire.requests) == 1
    sent = wire.requests[0]
    assert sent["max_tokens"] == NON_STREAMING_MAX_TOKENS
    assert sent["model"] == MODEL
    assert not sent.get("stream")
    assert resp.content[0].text == "ok"
    assert resp.stop_reason == "end_turn"


def test_stream_messages_at_the_scale_cap_streams_on_the_wire():
    wire = _Wire()
    llm = _client(wire)
    chunks: list[str] = []
    final = llm.stream_messages(
        system="s", messages=_MSGS, max_tokens=32_768, model=MODEL,
        on_text=chunks.append,
    )
    assert len(wire.requests) == 1
    sent = wire.requests[0]
    assert sent["stream"] is True
    assert sent["max_tokens"] == 32_768
    assert "".join(chunks) == "streamed"
    assert final.content[0].text == "streamed"
    assert final.stop_reason == "end_turn"
    assert final.usage.output_tokens == 5


@pytest.mark.asyncio
async def test_a_non_streaming_agent_request_at_the_scale_cap_is_served():
    wire = _Wire()
    agent = _bare_agent(_client(wire), 32_768)
    out = await agent._call_model(system="s", messages=list(_MSGS), tools=[])
    assert out.content[0].text == "streamed"
    assert out.stop_reason == "end_turn"
    assert [(r["max_tokens"], r.get("stream")) for r in wire.requests] == [
        (32_768, True),
    ]


@pytest.mark.asyncio
async def test_without_the_routing_the_same_request_dies_in_the_sdk(monkeypatch):
    # Raise the agent's idea of the ceiling above the request so it takes
    # the plain complete_messages path; the SDK's own guard then refuses
    # it before the wire. This is the failure the routing exists to avoid.
    monkeypatch.setattr(agent_module, "NON_STREAMING_MAX_TOKENS", 10**9)
    wire = _Wire()
    agent = _bare_agent(_client(wire), 32_768)
    with pytest.raises(ValueError, match="Streaming is required"):
        await agent._call_model(system="s", messages=list(_MSGS), tools=[])
    assert wire.requests == []
