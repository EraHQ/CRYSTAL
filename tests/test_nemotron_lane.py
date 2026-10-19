"""M3 (2026-10-19): NVIDIA Nemotron on Nebius Token Factory as a second
model family (Q74 A, Q75 B, Q76 A), without touching the Anthropic
defaults.

- lanes: nvidia/... ids are the "nemotron" family; the small tier runs on
  the configured light id; the lane exists only while the key is set.
- client: get_llm_client() returns the Token Factory singleton inside a
  background lane whose pick is Nemotron (and the lane is on); the
  Anthropic singleton ignores a Nemotron pick; the Nemotron client
  ignores an Anthropic pick; get_llm_client_for_customer routes a managed
  tenant's Nemotron ANSWERS pick; byok is untouched.
- admission: Nemotron ids are allowed on every managed tier only while
  the lane is on; the Anthropic tier lists are unchanged.
- openai path: read timeout scales with max_tokens; 429 honours
  Retry-After then succeeds; inline <think> is stripped; json_schema is
  strict-cleaned and repeated in the prompt; a 500 raises.
- pricing: the Nemotron rows meter (nonzero) by case-insensitive prefix.
"""
from __future__ import annotations

import json

import httpx
import pytest

from crystal_cache.control import admission
from crystal_cache.cost.pricing import compute_cost_micro_usd
from crystal_cache.llm import client as client_mod
from crystal_cache.llm import lanes
from crystal_cache.llm.client import LLMClient, get_llm_client, get_llm_client_for_customer, get_nemotron_client, reset_llm_client

SUPER = "nvidia/nemotron-3-super-120b-a12b"
LIGHT = "nvidia/Nemotron-3_5-Lightning"


class _Cfg:
    def __init__(self, model_id, background_model_id=None, api_key_ref="", provider="anthropic", base_url=None):
        self.model_id = model_id
        self.background_model_id = background_model_id
        self.api_key_ref = api_key_ref
        self.provider = provider
        self.base_url = base_url


class _Customer:
    def __init__(self, model_id="claude-sonnet-5-5", background=None, tier=None, mode="managed"):
        self.id = "cus_n"
        self.model_routing_config = _Cfg(model_id, background)
        self.subscription_tier = tier
        self.inference_mode = mode


@pytest.fixture
def lane_on(monkeypatch):
    from crystal_cache import config

    monkeypatch.setattr(config.settings, "nebius_api_key", "nb-test-key")
    monkeypatch.setattr(config.settings, "nemotron_model_small", LIGHT)
    monkeypatch.setattr(config.settings, "nemotron_model_large", SUPER)
    monkeypatch.setattr(config.settings, "nemotron_model_frontier", SUPER)
    monkeypatch.setattr(config.settings, "anthropic_api_key", "sk-ant-test")
    reset_llm_client()
    yield
    reset_llm_client()


@pytest.fixture
def lane_off(monkeypatch):
    from crystal_cache import config

    monkeypatch.setattr(config.settings, "nebius_api_key", None)
    reset_llm_client()
    yield
    reset_llm_client()


def test_family_and_light_model(lane_on):
    assert lanes.family_of(SUPER) == "nemotron" and lanes.family_of("NVIDIA/x") == "nemotron"
    assert lanes.family_of("claude-opus-5-5") == "claude" and lanes.family_of("gpt-4o") is None
    assert lanes.nemotron_enabled() and lanes.nemotron_models() == (LIGHT, SUPER)
    with lanes.tenant_lane(_Customer(background=SUPER), "background"):
        assert lanes.lane_model_for_tier("large") == SUPER
        assert lanes.lane_model_for_tier("frontier") == SUPER
        assert lanes.lane_model_for_tier("small") == LIGHT
        assert lanes.lane_pick() == SUPER
    assert lanes.lane_pick() is None


def test_lane_off_means_no_models(lane_off):
    assert not lanes.nemotron_enabled() and lanes.nemotron_models() == ()


def test_get_llm_client_switches_provider_inside_a_nemotron_lane(lane_on):
    base = get_llm_client()
    assert base.provider == "anthropic" and base.family == "claude"
    nem = get_nemotron_client()
    assert nem.provider == "openai" and nem.family == "nemotron" and nem.is_ready()
    with lanes.tenant_lane(_Customer(background=SUPER), "background"):
        assert get_llm_client() is nem
        assert nem._resolve_model("large", None) == SUPER
        assert nem._resolve_model("small", None) == LIGHT
        assert nem._resolve_model("large", "claude-opus-5-5") == "claude-opus-5-5"  # explicit always wins
        # The Anthropic singleton, captured earlier, ignores the foreign pick.
        assert base._resolve_model("large", None) == "claude-sonnet-5-5"
    with lanes.tenant_lane(_Customer(background="claude-opus-5-5"), "background"):
        assert get_llm_client() is base
        assert nem._resolve_model("large", None) == SUPER  # a Claude pick never reaches the Nemotron client
    assert get_llm_client() is base


def test_get_llm_client_stays_anthropic_when_the_lane_is_off(lane_off):
    base = get_llm_client()
    with lanes.tenant_lane(_Customer(background=SUPER), "background"):
        assert get_llm_client() is base
        assert base._resolve_model("large", None) == "claude-sonnet-5-5"


@pytest.mark.asyncio
async def test_answers_pick_routes_the_agent_turn(lane_on):
    nem = get_nemotron_client()
    assert await get_llm_client_for_customer(_Customer(model_id=SUPER), store=None) is nem
    assert await get_llm_client_for_customer(_Customer(model_id="claude-sonnet-5-5"), store=None) is get_llm_client()
    # byok is untouched by the lane: no key on file still fails loud.
    with pytest.raises(RuntimeError):
        await get_llm_client_for_customer(_Customer(model_id=SUPER, mode="byok"), store=None)


def test_admission_allows_nemotron_only_while_the_lane_is_on(lane_on, monkeypatch):
    assert admission.allowed_models_for("free") == admission.MANAGED_MODELS_FREE + (LIGHT, SUPER)
    assert admission.allowed_models_for("starter_29") == admission.MANAGED_MODELS_ALL + (LIGHT, SUPER)
    admission.enforce_managed_model(_Customer(tier=None), SUPER)
    admission.enforce_managed_model(_Customer(tier="starter_29"), LIGHT)
    with pytest.raises(admission.PlanWallError):
        admission.enforce_managed_model(_Customer(tier=None), "claude-opus-5-5")
    with pytest.raises(admission.PlanWallError):
        admission.enforce_managed_model(_Customer(tier=None), "nvidia/not-on-the-platform")
    from crystal_cache import config

    monkeypatch.setattr(config.settings, "nebius_api_key", "")
    assert admission.allowed_models_for("free") == admission.MANAGED_MODELS_FREE
    with pytest.raises(admission.PlanWallError):
        admission.enforce_managed_model(_Customer(tier=None), SUPER)
    # The tier lists themselves never changed.
    assert admission.MANAGED_MODELS_ALL == ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1")


def _nemotron_with_transport(handler):
    c = LLMClient(provider="openai", api_key="nb", base_url="https://api.tokenfactory.nebius.com/v1",
                  model_small=LIGHT, model_large=SUPER, model_frontier=SUPER, family="nemotron")
    c._http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return c


def test_openai_path_timeout_retry_think_and_schema(monkeypatch):
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: None)
    seen = {"calls": [], "timeouts": []}

    def handler(request):
        body = json.loads(request.content)
        seen["calls"].append(body)
        seen["timeouts"].append(request.extensions.get("timeout"))
        if len(seen["calls"]) == 1:
            return httpx.Response(429, headers={"retry-after": "1"}, json={"error": "slow down"})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "<think>let me reason</think>{\"a\": 1, \"b\": \"x\"}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7},
        })
    c = _nemotron_with_transport(handler)
    schema = {"type": "object", "properties": {"a": {"type": "integer", "minimum": 0}, "b": {"type": "string", "maxLength": 5}}, "required": ["a", "b"]}
    r = c.complete_detailed(system="You judge.", messages=[{"role": "user", "content": "go"}], max_tokens=16_000, tier="large", json_schema=schema)
    assert r.text == '{"a": 1, "b": "x"}' and r.model == SUPER and r.stop_reason == "end_turn"
    assert r.input_tokens == 12 and r.output_tokens == 7
    assert len(seen["calls"]) == 2  # one 429, one success
    body = seen["calls"][-1]
    assert body["model"] == SUPER and body["max_tokens"] == 16_000
    wire_schema = body["response_format"]["json_schema"]["schema"]
    assert wire_schema.get("additionalProperties") is False
    assert "minimum" not in wire_schema["properties"]["a"] and "maxLength" not in wire_schema["properties"]["b"]
    assert body["messages"][0]["role"] == "system" and "matches this JSON schema exactly" in body["messages"][0]["content"]
    assert body["messages"][0]["content"].startswith("You judge.")
    read = seen["timeouts"][-1]["read"]
    assert read == min(900.0, 60.0 + 16_000 / 25.0)
    assert client_mod._openai_read_timeout(0) == 60.0 and client_mod._openai_read_timeout(100) == 64.0
    assert client_mod._openai_read_timeout(10 ** 9) == 900.0


def test_openai_path_gives_up_after_three_and_raises_on_500(monkeypatch):
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: None)
    n = {"calls": 0}

    def always_429(request):
        n["calls"] += 1
        return httpx.Response(429, headers={"retry-after": "900"}, json={})
    c = _nemotron_with_transport(always_429)
    with pytest.raises(httpx.HTTPStatusError):
        c.complete(system=None, messages=[{"role": "user", "content": "go"}], max_tokens=10, tier="small")
    assert n["calls"] == 3
    assert client_mod._retry_after_seconds(httpx.Response(429, headers={"retry-after": "900"}), 0) == 30.0
    assert client_mod._retry_after_seconds(httpx.Response(429), 2) == 4.0

    def boom(request):
        return httpx.Response(500, json={"error": "x"})
    with pytest.raises(httpx.HTTPStatusError):
        _nemotron_with_transport(boom).complete(system=None, messages=[{"role": "user", "content": "go"}], max_tokens=10, tier="small")


def test_agent_path_strips_reasoning(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "<think>hmm</think>Hello there"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })
    c = _nemotron_with_transport(handler)
    resp = c.complete_messages(system="s", messages=[{"role": "user", "content": "hi"}], max_tokens=50, tier="large")
    texts = [b.get("text") if isinstance(b, dict) else getattr(b, "text", None) for b in resp.content]
    assert texts == ["Hello there"]
    assert client_mod.strip_reasoning("<THINK>a\nb</think>  answer") == "answer"
    assert client_mod.strip_reasoning(None) == ""


def test_nemotron_rows_meter_nonzero():
    assert compute_cost_micro_usd(SUPER, input_tokens=1_000_000, output_tokens=1_000_000) == 1_200_000
    assert compute_cost_micro_usd(LIGHT, input_tokens=1_000_000, output_tokens=0) == 60_000
    assert compute_cost_micro_usd("nvidia/Nemotron-3-Ultra-550b-a55b", input_tokens=0, output_tokens=1_000_000) == 3_000_000


def test_vertex_is_ready_with_project_and_region():
    v = LLMClient(provider="vertex", api_key=None, base_url=None, model_small="a", model_large="b", model_frontier="c",
                  vertex_project="p", vertex_region="us-east5")
    assert v.is_ready() and v.family == "claude"
    assert not LLMClient(provider="vertex", api_key=None, base_url=None, model_small="a", model_large="b", model_frontier="c").is_ready()
