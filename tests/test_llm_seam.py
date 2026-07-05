"""B1 + Slice 3: seam-level behavior — usage metering and structured output.

complete() still returns bare text (unchanged for every migrated call site);
complete_detailed() additionally reports the resolved model and normalized
token counts so cost sites can meter provider-neutrally. json_schema requests
structured output with a provider-specific wire mapping. The underlying SDK
client is injected directly (the lazy _anthropic_client / _http_client
attributes), so these run without a network call.
"""
from __future__ import annotations

from types import SimpleNamespace

from crystal_cache.llm.client import LLMClient, LLMResult


class _FakeMessages:
    """Anthropic-shaped messages API that returns a canned response."""

    def __init__(self, resp):
        self._resp = resp
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._resp


def _anthropic_client_with(text, usage):
    client = LLMClient(
        provider="anthropic",
        api_key="k",
        base_url=None,
        model_small="m-small",
        model_large=None,
        model_frontier=None,
    )
    resp = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=usage,
    )
    # Inject the lazy SDK client so _get_anthropic() returns our fake.
    client._anthropic_client = SimpleNamespace(messages=_FakeMessages(resp))
    return client


def test_complete_detailed_extracts_anthropic_usage():
    usage = SimpleNamespace(
        input_tokens=11,
        output_tokens=7,
        cache_read_input_tokens=3,
        cache_creation_input_tokens=2,
    )
    client = _anthropic_client_with("hello world", usage)

    result = client.complete_detailed(
        system=None,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=50,
        tier="small",
    )

    assert isinstance(result, LLMResult)
    assert result.text == "hello world"
    # model is the RESOLVED model, not the tier name.
    assert result.model == "m-small"
    assert result.input_tokens == 11
    assert result.output_tokens == 7
    assert result.cache_read_tokens == 3
    assert result.cache_creation_tokens == 2


def test_complete_returns_text_only():
    """complete() delegates to complete_detailed() and yields just the text."""
    usage = SimpleNamespace(input_tokens=1, output_tokens=1)
    client = _anthropic_client_with("just text", usage)

    out = client.complete(
        system=None,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=50,
        tier="small",
    )

    assert out == "just text"


def test_complete_detailed_tolerates_missing_usage():
    """A response with no usage object yields None token counts, not an error."""
    client = _anthropic_client_with("no usage here", None)

    result = client.complete_detailed(
        system=None,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=50,
        tier="small",
    )

    assert result.text == "no usage here"
    assert result.model == "m-small"
    assert result.input_tokens is None
    assert result.output_tokens is None
    assert result.cache_read_tokens is None
    assert result.cache_creation_tokens is None


_SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string"}}}


def test_json_schema_maps_to_anthropic_output_config():
    client = _anthropic_client_with('{"verdict": "ok"}', None)

    out = client.complete(
        system=None,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=50,
        tier="small",
        json_schema=_SCHEMA,
    )

    assert out == '{"verdict": "ok"}'
    sent = client._anthropic_client.messages.last_kwargs
    assert sent["output_config"] == {
        "format": {"type": "json_schema", "schema": _SCHEMA}
    }


def test_no_json_schema_sends_no_output_config():
    client = _anthropic_client_with("plain", None)

    client.complete(
        system=None,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=50,
        tier="small",
    )

    sent = client._anthropic_client.messages.last_kwargs
    assert "output_config" not in sent


class _FakeHttp:
    """httpx-shaped client capturing the POSTed body."""

    def __init__(self, text):
        self._text = text
        self.last_json = None

    def post(self, url, *, json, headers):
        self.last_json = json
        text = self._text

        class _Resp:
            @staticmethod
            def raise_for_status():
                return None

            @staticmethod
            def json():
                return {
                    "choices": [{"message": {"content": text}}],
                    "usage": {},
                }

        return _Resp()


def test_json_schema_maps_to_openai_response_format():
    client = LLMClient(
        provider="openai",
        api_key="k",
        base_url="http://localhost:9999/v1",
        model_small="m-small",
        model_large=None,
        model_frontier=None,
    )
    fake_http = _FakeHttp('{"verdict": "ok"}')
    client._http_client = fake_http

    out = client.complete(
        system=None,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=50,
        tier="small",
        json_schema=_SCHEMA,
    )

    assert out == '{"verdict": "ok"}'
    assert fake_http.last_json["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": _SCHEMA},
    }


# ---------------------------------------------------------------------------
# Slice 5b: complete_messages — the Anthropic-shaped agent-loop method
# ---------------------------------------------------------------------------

_AGENT_TOOLS = [{
    "name": "key_scan",
    "description": "Scan sparse keys.",
    "input_schema": {"type": "object", "properties": {}},
    "cache_control": {"type": "ephemeral"},
}]


def test_complete_messages_anthropic_is_verbatim_passthrough():
    client = _anthropic_client_with("unused", None)
    system_blocks = [{
        "type": "text", "text": "You are CRYS.",
        "cache_control": {"type": "ephemeral"},
    }]
    messages = [{"role": "user", "content": "hi"}]

    out = client.complete_messages(
        system=system_blocks,
        messages=messages,
        tools=_AGENT_TOOLS,
        max_tokens=8192,
        model="claude-sonnet-4-6",
    )

    # The RAW response object comes back untouched.
    assert out is client._anthropic_client.messages._resp
    sent = client._anthropic_client.messages.last_kwargs
    # Anthropic-only decoration passes through verbatim on this path.
    assert sent["system"] is system_blocks
    assert sent["messages"] is messages
    assert sent["tools"] is _AGENT_TOOLS
    assert sent["model"] == "claude-sonnet-4-6"
    assert sent["max_tokens"] == 8192
    assert "temperature" not in sent  # loop parity: API default sampling


class _FakeHttpPayload:
    """httpx-shaped client returning an arbitrary completion payload."""

    def __init__(self, payload):
        self._payload = payload
        self.last_json = None

    def post(self, url, *, json, headers):
        self.last_json = json
        payload = self._payload

        class _Resp:
            @staticmethod
            def raise_for_status():
                return None

            @staticmethod
            def json():
                return payload

        return _Resp()


def _openai_client() -> LLMClient:
    return LLMClient(
        provider="openai",
        api_key="k",
        base_url="http://localhost:9999/v1",
        model_small="m-small",
        model_large="m-large",
        model_frontier=None,
    )


def test_complete_messages_openai_translates_and_shims():
    client = _openai_client()
    fake_http = _FakeHttpPayload({
        "model": "m-large",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "key_scan", "arguments": '{"q": "x"}'},
            }]},
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    })
    client._http_client = fake_http

    out = client.complete_messages(
        system="You are CRYS.",
        messages=[{"role": "user", "content": "count"}],
        tools=_AGENT_TOOLS,
        max_tokens=8192,
    )

    body = fake_http.last_json
    # tier="large" default resolved to the configured openai model.
    assert body["model"] == "m-large"
    assert body["messages"][0] == {"role": "system", "content": "You are CRYS."}
    assert body["tools"][0]["function"]["name"] == "key_scan"
    assert "cache_control" not in str(body)
    assert "temperature" not in body
    # Shim duck-types the SDK response for the loop.
    assert out.stop_reason == "tool_use"
    assert out.content[0]["input"] == {"q": "x"}
    assert out.usage.input_tokens == 10


def test_complete_messages_openai_omits_tools_when_none():
    client = _openai_client()
    fake_http = _FakeHttpPayload({
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "done"},
            "finish_reason": "stop",
        }],
        "usage": {},
    })
    client._http_client = fake_http

    out = client.complete_messages(
        system="s",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
    )

    assert "tools" not in fake_http.last_json
    assert out.stop_reason == "end_turn"
    assert out.content == [{"type": "text", "text": "done"}]


def test_provider_property_exposed():
    assert _openai_client().provider == "openai"
    assert _anthropic_client_with("x", None).provider == "anthropic"
