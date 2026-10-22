"""Hackathon item 2 (Q96 A, 2026-10-22): a tenant's own Nebius Token Factory
key. Saved with provider "nebius" after one /v1/models check; the answers
client and every background lane run on it (tenant_background binds the
client, get_llm_client() returns it); ledger rows made on it are stamped
byok so the platform's capacity doors leave them out. Nothing here touches
the network."""
from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from crystal_cache.cost import emit as emit_mod
from crystal_cache.llm import catalog, lanes
from crystal_cache.llm.client import get_llm_client, get_llm_client_for_customer, reset_llm_client

LIGHT = "nvidia/Nemotron-3_5-Lightning"
SUPER = "nvidia/nemotron-3-super-120b-a12b"
ULTRA = "nvidia/Nemotron-3-Ultra-550b-a55b"


@pytest.fixture
def lane_on(monkeypatch):
    from crystal_cache import config

    monkeypatch.setattr(config.settings, "nebius_api_key", "nb-platform-key")
    monkeypatch.setattr(config.settings, "nemotron_model_small", LIGHT)
    monkeypatch.setattr(config.settings, "nemotron_model_large", SUPER)
    monkeypatch.setattr(config.settings, "nemotron_model_frontier", ULTRA)
    monkeypatch.setattr(config.settings, "anthropic_api_key", "sk-ant-test")
    reset_llm_client()
    yield
    reset_llm_client()


def _authed():
    try:
        from tests.test_accounts_phase_c import _AuthedReq, _use
    except ModuleNotFoundError:
        from test_accounts_phase_c import _AuthedReq, _use
    return _AuthedReq, _use


@pytest.mark.asyncio
async def test_saving_a_nebius_key_checks_it_and_moves_the_picks_onto_the_lane(lane_on, monkeypatch, store):
    _AuthedReq, _use = _authed()
    from crystal_cache.endpoints.customers import update_inference_mode, update_upstream_key

    _use(monkeypatch)
    c = await store.create_customer(provider="anthropic", model_id="claude-sonnet-5-5", api_key_ref="")
    await store.set_customer_background_model(c.id, "claude-haiku-5-5")
    bearer = f"Bearer {c.api_key}"
    seen = []

    def fake_verify(key):
        seen.append(key)
        if key == "bad-key":
            raise RuntimeError("401")
        return 28
    monkeypatch.setattr(catalog, "verify_key", fake_verify)

    # A refused key is not stored, and the refusal carries a code.
    with pytest.raises(HTTPException) as e:
        await update_upstream_key(c.id, _AuthedReq(bearer, body={"api_key_ref": "bad-key", "provider": "nebius"}), store)
    assert e.value.status_code == 400 and e.value.detail["error"]["code"] == "provider_key_rejected"
    assert (await store.get_customer_by_id(c.id)).model_routing_config.api_key_ref == ""
    # An unknown provider and a base_url outside self_hosted are refused too.
    with pytest.raises(HTTPException) as e:
        await update_upstream_key(c.id, _AuthedReq(bearer, body={"api_key_ref": "k", "provider": "groq"}), store)
    assert e.value.detail["error"]["code"] == "provider_unknown"
    with pytest.raises(HTTPException) as e:
        await update_upstream_key(c.id, _AuthedReq(bearer, body={"api_key_ref": "k", "provider": "nebius", "base_url": "https://x"}), store)
    assert e.value.detail["error"]["code"] == "base_url_invalid"
    # A good key: stored encrypted, provider recorded, the Anthropic picks moved onto the lane.
    r = await update_upstream_key(c.id, _AuthedReq(bearer, body={"api_key_ref": "nb-tenant-key", "provider": "nebius"}), store)
    body = json.loads(r.body)
    assert body["provider"] == "nebius" and body["models_seen"] == 28
    assert body["moved"] == {"model_id": SUPER, "background_model_id": None}
    assert seen == ["bad-key", "nb-tenant-key"]
    got = await store.get_customer_by_id(c.id)
    cfg = got.model_routing_config
    assert cfg.provider == "nebius" and cfg.api_key_ref.startswith("enc:") and cfg.base_url is None
    assert cfg.model_id == SUPER and cfg.background_model_id is None
    # The key without a provider keeps the provider (a plain re-paste).
    await update_upstream_key(c.id, _AuthedReq(bearer, body={"api_key_ref": "nb-tenant-key-2"}), store)
    assert (await store.get_customer_by_id(c.id)).model_routing_config.provider == "nebius"
    # Flip to byok now that a key is on file.
    await update_inference_mode(c.id, _AuthedReq(bearer, body={"inference_mode": "byok"}), store)
    got = await store.get_customer_by_id(c.id)
    assert got.inference_mode == "byok"

    # The answers client is built from the tenant's key on Token Factory, Nemotron family.
    client = await get_llm_client_for_customer(got, store)
    assert client.provider == "openai" and client.family == "nemotron"
    assert client._base_url.startswith("https://api.tokenfactory.nebius.com")
    assert client._api_key == "nb-tenant-key-2"
    assert client._resolve_model("large", None) == SUPER and client._resolve_model("small", None) == LIGHT
    assert client._resolve_model("frontier", None) == ULTRA

    # Background: tenant_background binds that client; get_llm_client() returns it inside, not outside.
    platform = get_llm_client()
    assert platform.family == "claude"
    async with lanes.tenant_background(got, store):
        bound = get_llm_client()
        assert bound is not platform and bound.family == "nemotron" and bound._api_key == "nb-tenant-key-2"
        assert lanes.current_client() is bound
        # A ledger row written inside is stamped byok without the caller saying so.
        emitted = []

        class _St:
            async def record_llm_call(self, customer_id, **kw):
                emitted.append(kw); return {"id": "x", **kw}

        class _S:
            enable_cost_accounting = True; llm_price_table_overrides = None
        monkeypatch.setattr(emit_mod, "get_settings", lambda: _S())
        await emit_mod.record_model_call(store=_St(), customer_id=got.id, model=SUPER, origin="contradiction_scan",
                                         input_tokens=10, output_tokens=5)
        assert emitted[-1]["billing"] == "byok"
        # An explicit stamp is never overwritten.
        await emit_mod.record_model_call(store=_St(), customer_id=got.id, model=SUPER, origin="x",
                                         input_tokens=1, output_tokens=1, billing="managed")
        assert emitted[-1]["billing"] == "managed"
    assert lanes.current_client() is None and get_llm_client() is platform
    # Outside any tenant's lane the stamp is left to the caller (None), as before.
    emitted.clear()
    await emit_mod.record_model_call(store=_St(), customer_id=got.id, model=SUPER, origin="x", input_tokens=1, output_tokens=1)
    assert emitted[-1]["billing"] is None


@pytest.mark.asyncio
async def test_managed_and_keyless_tenants_keep_the_platform_clients(lane_on, monkeypatch, store):
    _AuthedReq, _use = _authed()
    _use(monkeypatch)
    managed = await store.create_customer(provider="anthropic", model_id="claude-sonnet-5-5", api_key_ref="")
    await store.set_customer_inference_mode(managed.id, "managed")
    managed = await store.get_customer_by_id(managed.id)
    assert await lanes.tenant_client_for(managed, store) is None
    # byok with no key: None, and the lane still works on the platform map (logged, never raised).
    keyless = await store.create_customer(provider="anthropic", model_id="claude-sonnet-5-5", api_key_ref="")
    keyless = await store.get_customer_by_id(keyless.id)
    assert keyless.inference_mode == "byok" and await lanes.tenant_client_for(keyless, store) is None
    platform = get_llm_client()
    async with lanes.tenant_background(keyless, store):
        assert get_llm_client() is platform
    # byok on another provider: that provider's client, not Token Factory.
    other = await store.create_customer(provider="openai", model_id="gpt-4o", api_key_ref="")
    await store.update_customer_upstream_key(other.id, "sk-openai", provider="openai")
    other = await store.get_customer_by_id(other.id)
    async with lanes.tenant_background(other, store):
        bound = get_llm_client()
        assert bound.provider == "openai" and bound.family is None and bound._resolve_model("large", None) == "gpt-4o"
    # A byok Token Factory tenant's pick is probed on THEIR client.
    from crystal_cache.endpoints.customers import update_background_model
    nebius = await store.create_customer(provider="anthropic", model_id="claude-sonnet-5-5", api_key_ref="")
    await store.update_customer_upstream_key(nebius.id, "nb-key", provider="nebius")
    probed = []

    def fake_probe(client, model_id):
        probed.append((client._api_key, model_id))
        return catalog.ModelCaps(json_schema=True, tools=True)
    monkeypatch.setattr(catalog, "probe_model", fake_probe)
    r = await update_background_model(nebius.id, _AuthedReq(f"Bearer {nebius.api_key}", body={"model_id": LIGHT}), store)
    assert json.loads(r.body)["caps"]["tools"] is True and probed == [("nb-key", LIGHT)]
