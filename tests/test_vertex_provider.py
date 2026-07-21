"""Vertex LLM provider (2026-07-03, GCP consolidation).

provider="vertex" = Claude-on-Vertex: the SAME Anthropic Messages API,
served from Vertex AI, authenticated via GCP Application Default
Credentials — so the entire existing anthropic wire path (completion
shape, json_schema mapping, usage extraction, complete_messages
passthrough) is reused verbatim and only the SDK client constructor and
model resolution differ. Proven here:
  - vertex rides the anthropic completion path (fake-injected client);
  - per-tier models are REQUIRED (no universal defaults — Vertex ids are
    region/version-flavored) with an actionable error naming the env var;
  - construction fails loud without project + region;
  - the unknown-provider error names all three providers.

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from crystal_cache.llm.client import LLMClient


class _FakeMessages:
    def __init__(self, resp):
        self._resp = resp
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._resp


def _vertex_client_with(text, **model_kw):
    client = LLMClient(
        provider="vertex",
        api_key=None,                       # ADC — no key on this provider
        base_url=None,
        model_small=model_kw.get("model_small"),
        model_large=model_kw.get("model_large"),
        model_frontier=model_kw.get("model_frontier"),
        vertex_project="era-proj",
        vertex_region="us-east5",
    )
    resp = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=10, output_tokens=5,
            cache_creation_input_tokens=None, cache_read_input_tokens=None,
        ),
    )
    client._anthropic_client = SimpleNamespace(messages=_FakeMessages(resp))
    return client


def test_vertex_rides_the_anthropic_wire_path():
    client = _vertex_client_with(
        "hi from vertex", model_small="claude-haiku-4-5@20251001",
    )
    result = client.complete_detailed(
        system="s", messages=[{"role": "user", "content": "u"}],
        max_tokens=32, tier="small",
    )
    assert result.text == "hi from vertex"
    assert result.model == "claude-haiku-4-5@20251001"
    assert result.input_tokens == 10 and result.output_tokens == 5
    sent = client._anthropic_client.messages.last_kwargs
    assert sent["model"] == "claude-haiku-4-5@20251001"
    # Cost slice 1a (2026-07-21): the shared wire path wraps string
    # systems as cached blocks — Anthropic-on-Vertex supports the same
    # cache_control form, so vertex callers get prompt caching too.
    assert sent["system"] == [{
        "type": "text", "text": "s",
        "cache_control": {"type": "ephemeral"},
    }]


def test_vertex_requires_per_tier_models():
    client = _vertex_client_with("x")  # no models configured
    with pytest.raises(ValueError, match="CC_LLM_MODEL_FRONTIER"):
        client.complete(
            system=None, messages=[{"role": "user", "content": "u"}],
            max_tokens=8, tier="frontier",
        )


def test_vertex_requires_project_and_region():
    client = LLMClient(
        provider="vertex", api_key=None, base_url=None,
        model_small="m", model_large=None, model_frontier=None,
        vertex_project=None, vertex_region=None,
    )
    with pytest.raises(ValueError, match="CC_VERTEX_PROJECT"):
        client._get_anthropic()


def test_unknown_provider_names_all_three():
    client = LLMClient(
        provider="nonsense", api_key=None, base_url=None,
        model_small="m", model_large=None, model_frontier=None,
    )
    with pytest.raises(ValueError, match="anthropic.*vertex.*openai"):
        client.complete_detailed(
            system=None, messages=[{"role": "user", "content": "u"}],
            max_tokens=8, tier="small",
        )


def test_vertex_complete_messages_passthrough():
    """The agent-loop path: vertex passes Anthropic-shaped messages + tools
    straight through, same as the anthropic provider."""
    client = _vertex_client_with("t", model_large="claude-sonnet-4-6@x")
    tools = [{"name": "t1", "description": "d", "input_schema": {}}]
    client.complete_messages(
        system="sys", messages=[{"role": "user", "content": "u"}],
        tools=tools, max_tokens=16, tier="large",
    )
    sent = client._anthropic_client.messages.last_kwargs
    assert sent["tools"] == tools
    # RAW path: verbatim by contract — the agent owns its own cache
    # decoration; the seam only wraps on the complete() workers' lane.
    assert sent["system"] == "sys"
    assert sent["model"] == "claude-sonnet-4-6@x"
