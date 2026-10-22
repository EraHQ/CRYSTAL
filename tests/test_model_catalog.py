"""Hackathon item 1 (2026-10-22): any text model Token Factory lists is a
pick, metered at its listed price, probed for json_schema and tools; item
3: the frontier tier lifts a Nemotron pick to Ultra; item 5: background
calls carry service_tier flex. The listing is a recorded payload; nothing
here touches the network (conftest keeps the catalog offline)."""
from __future__ import annotations

import json

import httpx
import pytest

from crystal_cache.cost.pricing import RUNTIME_PRICE_TABLE, compute_cost_micro_usd, resolve_model_price
from crystal_cache.llm import catalog, lanes
from crystal_cache.llm.client import LLMClient, reset_llm_client

LIGHT = "nvidia/Nemotron-3_5-Lightning"
SUPER = "nvidia/nemotron-3-super-120b-a12b"
ULTRA = "nvidia/Nemotron-3-Ultra-550b-a55b"
DEEPSEEK = "deepseek-ai/DeepSeek-V4-Pro"
VISION = "nvidia/Nemotron-3.5-Super-VL"


def _payload(unit: str = "per_mtok") -> dict:
    # One listing, three ways the price strings could be written.
    scale = {"per_mtok": 1.0, "per_ktok": 1e-3, "per_token": 1e-6}[unit]

    def p(usd_per_mtok: float) -> str:
        return f"{usd_per_mtok * scale:.12f}".rstrip("0")

    def row(mid, ctx, pin, pout, modality="text->text", rpm=300, tpm=200_000):
        return {"id": mid, "object": "model", "name": mid.split("/")[-1], "context_length": ctx,
                "architecture": {"modality": modality, "tokenizer": "Other"},
                "pricing": {"prompt": p(pin), "completion": p(pout), "image": "0", "request": "0"},
                "per_request_limits": {"requests_per_minute": rpm, "tokens_per_minute": tpm, "burst_ratio": 2}}
    return {"object": "list", "data": [
        row(LIGHT, 1_048_576, 0.06, 0.24),
        row(SUPER, 262_144, 0.30, 0.90),
        row(ULTRA, 1_048_576, 1.00, 3.00),
        row(DEEPSEEK, 1_000_000, 1.20, 2.40),
        row(SUPER + "-fast", 262_144, 0.60, 1.80),
        row(VISION, 1_048_576, 0.30, 0.90, modality="image+text->text"),
        row("Qwen/Qwen3-Embedding-8B", 40_960, 0.01, 0.0, modality="text->embedding"),
    ]}


@pytest.fixture
def lane_on(monkeypatch):
    from crystal_cache import config

    monkeypatch.setattr(config.settings, "nebius_api_key", "nb-test-key")
    monkeypatch.setattr(config.settings, "nemotron_model_small", LIGHT)
    monkeypatch.setattr(config.settings, "nemotron_model_large", SUPER)
    monkeypatch.setattr(config.settings, "nemotron_model_frontier", ULTRA)
    monkeypatch.setattr(config.settings, "anthropic_api_key", "sk-ant-test")
    reset_llm_client()
    yield
    reset_llm_client()


def _serve(monkeypatch, payload):
    catalog.reset_for_tests()
    monkeypatch.setattr(catalog, "_fetch_payload", lambda: payload)


@pytest.mark.parametrize("unit", ["per_mtok", "per_ktok", "per_token"])
def test_prices_calibrate_against_the_verified_anchor_whatever_the_unit(unit):
    models = {m.id: m for m in catalog.parse_catalog(_payload(unit))}
    assert models[LIGHT].input_micro_per_mtok == 60_000 and models[LIGHT].output_micro_per_mtok == 240_000
    assert models[SUPER].input_micro_per_mtok == 300_000 and models[ULTRA].output_micro_per_mtok == 3_000_000
    assert models[DEEPSEEK].input_micro_per_mtok == 1_200_000
    assert models[LIGHT].band == "light" and models[SUPER].band == "standard" and models[ULTRA].band == "heavy"
    assert models[SUPER + "-fast"].is_fast and not models[SUPER].is_fast
    assert models[VISION].is_text is False and models["Qwen/Qwen3-Embedding-8B"].is_text is False


def test_no_anchor_means_no_prices_and_a_warning(caplog):
    payload = _payload()
    payload["data"] = [m for m in payload["data"] if m["id"] != LIGHT]
    models = catalog.parse_catalog(payload)
    assert models and all(m.input_micro_per_mtok is None for m in models)
    assert all(m.band == "standard" for m in models)
    assert "price_unit_unknown" in caplog.text
    assert catalog.price_rows(models) == {}


def test_catalog_feeds_the_family_the_allow_list_and_the_ledger(lane_on, monkeypatch):
    _serve(monkeypatch, _payload())
    ids = catalog.catalog_ids()
    assert DEEPSEEK.lower() in ids and VISION.lower() in ids
    # Any listed id is the Token Factory family; an unlisted one is not.
    assert lanes.family_of(DEEPSEEK) == "nemotron" and lanes.family_of("gpt-4o") is None
    assert lanes.family_of("claude-opus-5-5") == "claude"
    # Text models only join the pickable list, configured ids first, no duplicates.
    picks = lanes.nemotron_models()
    assert picks[:3] == (LIGHT, SUPER, ULTRA)
    assert DEEPSEEK in picks and SUPER + "-fast" in picks and VISION not in picks
    assert len(picks) == len(set(p.lower() for p in picks))
    from crystal_cache.control import admission
    assert DEEPSEEK in admission.allowed_models_for("free")
    # The ledger meters the pick at its listed price, exact id first.
    assert DEEPSEEK.lower() in RUNTIME_PRICE_TABLE
    assert resolve_model_price(DEEPSEEK).input_micro_per_mtok == 1_200_000
    assert compute_cost_micro_usd(DEEPSEEK, input_tokens=1_000_000, output_tokens=0) == 1_200_000
    # A static row (lower-cased stem) still serves ids the listing does not carry.
    assert resolve_model_price("nvidia/nemotron-3-super-120b-a12b").input_micro_per_mtok == 300_000


def test_catalog_is_empty_when_the_lane_is_off(monkeypatch):
    from crystal_cache import config

    monkeypatch.setattr(config.settings, "nebius_api_key", None)
    _serve(monkeypatch, _payload())
    assert catalog.fetch_catalog() == () and lanes.family_of(DEEPSEEK) is None


def test_outage_serves_stale_and_never_raises(lane_on, monkeypatch):
    _serve(monkeypatch, _payload())
    assert len(catalog.fetch_catalog()) == 7

    def _down():
        raise RuntimeError("502")
    monkeypatch.setattr(catalog, "_fetch_payload", _down)
    assert len(catalog.fetch_catalog(force=True)) == 7  # the stale copy stays
    # A process whose first read fails has an empty catalog and keeps the configured ids.
    catalog.reset_for_tests()
    assert catalog.fetch_catalog() == () and lanes.nemotron_models() == (LIGHT, SUPER, ULTRA)


def test_frontier_lifts_a_nemotron_pick_to_ultra_and_leaves_other_vendors_alone(lane_on, monkeypatch):
    _serve(monkeypatch, _payload())

    class _Cfg:
        def __init__(self, bg):
            self.model_id = "claude-sonnet-5-5"; self.background_model_id = bg; self.model_caps = {}

    class _C:
        def __init__(self, bg):
            self.id = "cus_f"; self.model_routing_config = _Cfg(bg); self.inference_mode = "managed"
    with lanes.tenant_lane(_C(SUPER), "background"):
        assert lanes.lane_model_for_tier("small") == LIGHT
        assert lanes.lane_model_for_tier("large") == SUPER
        assert lanes.lane_model_for_tier("frontier") == ULTRA
    with lanes.tenant_lane(_C(DEEPSEEK), "background"):
        assert lanes.lane_model_for_tier("small") == LIGHT
        assert lanes.lane_model_for_tier("large") == DEEPSEEK
        assert lanes.lane_model_for_tier("frontier") == DEEPSEEK
    with lanes.tenant_lane(_C("claude-opus-5-5"), "background"):
        assert lanes.lane_model_for_tier("frontier") == "claude-opus-5-5"
    # The cognition validator asks for the frontier under a Nemotron pick only.
    from crystal_cache.cognition.roles import _validator_tier
    with lanes.tenant_lane(_C(SUPER), "background"):
        assert _validator_tier() == "frontier"
    with lanes.tenant_lane(_C("claude-opus-5-5"), "background"):
        assert _validator_tier() == "large"
    assert _validator_tier() == "large"


class _Scripted:
    """A client whose raw_chat answers from a script, one entry per call."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.bodies = []

    def raw_chat(self, body, *, max_tokens):
        self.bodies.append(body)
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a


def _json_reply(text):
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}


def _tool_reply(name):
    return {"choices": [{"message": {"content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": name, "arguments": "{\"word\": \"hello\"}"}}]}, "finish_reason": "tool_calls"}]}


def test_probe_reads_both_answers_and_records_failures():
    good = _Scripted([_json_reply('{"ok": true}'), _tool_reply("ping")])
    caps = catalog.probe_model(good, SUPER)
    assert caps.json_schema and caps.tools and caps.error is None
    assert good.bodies[0]["response_format"]["type"] == "json_schema" and good.bodies[0]["reasoning_effort"] == "none"
    assert good.bodies[1]["tool_choice"]["function"]["name"] == "ping" and good.bodies[0]["model"] == SUPER
    rec = caps.as_dict()
    assert rec["json_schema"] is True and rec["tools"] is True and rec["probed_at"]
    # The server rejects response_format (a 400, recorded as the error) and
    # the model answers in prose instead of a call (a failed flag, no error).
    bad = _Scripted([httpx.HTTPStatusError("400", request=None, response=httpx.Response(400)), _json_reply("I would call ping.")])
    caps = catalog.probe_model(bad, DEEPSEEK)
    assert not caps.json_schema and not caps.tools and caps.error == "json_schema: 400"
    # Both calls failing at transport keeps both error texts.
    down = _Scripted([RuntimeError("timeout a"), RuntimeError("timeout b")])
    caps = catalog.probe_model(down, DEEPSEEK)
    assert not caps.json_schema and not caps.tools and caps.error == "json_schema: timeout a; tools: timeout b"
    # Valid JSON of the wrong shape is not a pass.
    assert not catalog.probe_model(_Scripted([_json_reply('{"fine": 1}'), _tool_reply("other")]), SUPER).json_schema


def _nemotron(handler, bodies):
    c = LLMClient(provider="openai", api_key="k", base_url="https://tf.test/v1",
                  model_small=LIGHT, model_large=SUPER, model_frontier=ULTRA, family="nemotron")
    c._http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return c


def test_seam_drops_to_json_object_for_a_model_without_json_schema_and_rides_flex_in_background(lane_on, monkeypatch):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"a": "x"}'}, "finish_reason": "stop"}], "usage": {}})
    c = _nemotron(handler, bodies)
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}

    class _Cfg:
        model_id = "claude-sonnet-5-5"; background_model_id = DEEPSEEK
        model_caps = {DEEPSEEK.lower(): {"json_schema": False, "tools": True}, SUPER.lower(): {"json_schema": True, "tools": True}}

    class _C:
        id = "cus_s"; model_routing_config = _Cfg(); inference_mode = "managed"
    _serve(monkeypatch, _payload())
    # Outside any lane: json_schema on the wire, no service tier.
    c.complete(system="s", messages=[{"role": "user", "content": "q"}], max_tokens=50, tier="large", json_schema=schema)
    assert bodies[-1]["response_format"]["type"] == "json_schema" and "service_tier" not in bodies[-1]
    with lanes.tenant_lane(_C(), "background"):
        assert lanes.lane_caps(DEEPSEEK) == {"json_schema": False, "tools": True}
        assert lanes.lane_caps("nobody/unknown") is None
        c.complete(system="s", messages=[{"role": "user", "content": "q"}], max_tokens=50, tier="large", json_schema=schema)
        assert bodies[-1]["model"] == DEEPSEEK
        assert bodies[-1]["response_format"] == {"type": "json_object"}
        assert "matches this JSON schema exactly" in bodies[-1]["messages"][0]["content"]
        assert bodies[-1]["service_tier"] == "flex"
        # A probed-good model keeps json_schema; the explicit model wins over the pick.
        c.complete(system="s", messages=[{"role": "user", "content": "q"}], max_tokens=50, tier="large", json_schema=schema, model=SUPER)
        assert bodies[-1]["response_format"]["type"] == "json_schema" and bodies[-1]["service_tier"] == "flex"
    with lanes.tenant_lane(_C(), "answers"):
        c.complete(system="s", messages=[{"role": "user", "content": "q"}], max_tokens=50, tier="large")
        assert "service_tier" not in bodies[-1]
    from crystal_cache import config
    monkeypatch.setattr(config.settings, "nebius_background_service_tier", "")
    with lanes.tenant_lane(_C(), "background"):
        c.complete(system="s", messages=[{"role": "user", "content": "q"}], max_tokens=50, tier="large")
        assert "service_tier" not in bodies[-1]
    # A byok openai client (no family) never carries the tier.
    byok = LLMClient(provider="openai", api_key="k", base_url="https://x.test/v1", model_small="s", model_large="l", model_frontier="f")
    byok._http_client = httpx.Client(transport=httpx.MockTransport(handler))
    with lanes.tenant_lane(_C(), "background"):
        byok.complete(system=None, messages=[{"role": "user", "content": "q"}], max_tokens=50, tier="large")
        assert "service_tier" not in bodies[-1]


@pytest.mark.asyncio
async def test_pick_routes_probe_store_and_refuse(lane_on, monkeypatch, store):
    try:
        from tests.test_accounts_phase_c import _AuthedReq, _use
    except ModuleNotFoundError:
        from test_accounts_phase_c import _AuthedReq, _use
    from fastapi import HTTPException
    from crystal_cache.endpoints.customers import model_catalog, update_background_model, update_model

    _use(monkeypatch)
    _serve(monkeypatch, _payload())
    probes = []

    def fake_probe(client, model_id):
        probes.append(model_id)
        if model_id == DEEPSEEK:
            return catalog.ModelCaps(json_schema=False, tools=False, error="tools: prose")
        return catalog.ModelCaps(json_schema=True, tools=True)
    monkeypatch.setattr(catalog, "probe_model", fake_probe)
    c = await store.create_customer(provider="anthropic", model_id="claude-sonnet-5-5", api_key_ref="")
    await store.set_customer_inference_mode(c.id, "managed")
    bearer = f"Bearer {c.api_key}"

    # An answers pick that fails the tools probe is refused with a code; nothing stored.
    with pytest.raises(HTTPException) as e:
        await update_model(c.id, _AuthedReq(bearer, body={"model_id": DEEPSEEK}), store)
    assert e.value.status_code == 400 and e.value.detail["error"]["code"] == "model_no_tools"
    got = await store.get_customer_by_id(c.id)
    assert got.model_routing_config.model_id == "claude-sonnet-5-5" and got.model_routing_config.model_caps == {}
    # The same model is fine for background: stored with its probe record.
    r = await update_background_model(c.id, _AuthedReq(bearer, body={"model_id": DEEPSEEK}), store)
    body = json.loads(r.body)
    assert body["background_model_id"] == DEEPSEEK and body["caps"]["json_schema"] is False
    got = await store.get_customer_by_id(c.id)
    assert got.model_routing_config.model_caps[DEEPSEEK.lower()]["tools"] is False
    # Super passes both and becomes the answers model.
    r = await update_model(c.id, _AuthedReq(bearer, body={"model_id": SUPER}), store)
    assert json.loads(r.body)["caps"]["tools"] is True
    got = await store.get_customer_by_id(c.id)
    assert got.model_routing_config.model_id == SUPER and got.model_routing_config.model_caps[SUPER.lower()]["json_schema"] is True
    # An Anthropic pick is never probed.
    await update_model(c.id, _AuthedReq(bearer, body={"model_id": "claude-haiku-5-5"}), store)
    assert probes == [DEEPSEEK, DEEPSEEK, SUPER]
    # The lane reads the stored record.
    with lanes.tenant_lane(got, "background"):
        assert lanes.lane_caps(DEEPSEEK)["json_schema"] is False
    # The picker's read: text models only, NVIDIA first, bands and the tenant's own probe records, no dollars.
    r = await model_catalog(c.id, _AuthedReq(bearer), store)
    body = json.loads(r.body)
    assert body["enabled"] is True
    ids = [m["id"] for m in body["models"]]
    assert set(ids[:4]) == {SUPER, SUPER + "-fast", ULTRA, LIGHT} and ids[4:] == [DEEPSEEK]
    assert VISION not in ids and "Qwen/Qwen3-Embedding-8B" not in ids
    ds = next(m for m in body["models"] if m["id"] == DEEPSEEK)
    assert ds["band"] == "heavy" and ds["context_length"] == 1_000_000 and ds["caps"]["tools"] is False and ds["nvidia"] is False
    assert not any("price" in k or "usd" in k or "micro" in k for m in body["models"] for k in m)
