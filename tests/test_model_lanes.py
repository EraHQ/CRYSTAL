"""Q74=A / Q87 (2026-10-17): two model picks per tenant, read through a
lane context by the shared client.

- lanes.py: inside tenant_lane(customer, "background") a tier resolves
  to the tenant's background pick (family light model for small);
  answers / infra lanes and an unbound context fall through; the
  context is restored on exit.
- LLMClient._resolve_model: explicit model wins, then the lane, then
  the per-tier override, then the platform default; only the Anthropic
  singleton honors an Anthropic pick.
- PATCH /v1/customers/{id}/background_model: plan-checked like the
  answers model, legacy ids stored as their successor, empty clears;
  GET /v1/customers/{id} reports it.
- Every background-curation site binds the tenant; the critics (infra)
  do not.
"""
from __future__ import annotations

import inspect

import pytest
from fastapi import HTTPException

from crystal_cache.llm import lanes
from crystal_cache.llm.client import LLMClient


class _Cfg:
    def __init__(self, model_id, background_model_id=None):
        self.model_id = model_id
        self.background_model_id = background_model_id


class _Customer:
    def __init__(self, model_id="claude-sonnet-5-5", background=None, tier=None, mode="managed"):
        self.id = "cus_l"
        self.model_routing_config = _Cfg(model_id, background)
        self.subscription_tier = tier
        self.inference_mode = mode


def test_lane_resolution_and_restore():
    assert lanes.lane_model_for_tier("large") is None
    opus = _Customer(background="claude-opus-5-5")
    with lanes.tenant_lane(opus, "background"):
        assert lanes.current_lane() == "background"
        assert lanes.lane_model_for_tier("large") == "claude-opus-5-5"
        assert lanes.lane_model_for_tier("frontier") == "claude-opus-5-5"
        assert lanes.lane_model_for_tier("small") == "claude-haiku-5-5"
        with lanes.tenant_lane(opus, "infra"):
            assert lanes.lane_model_for_tier("large") is None
        assert lanes.lane_model_for_tier("large") == "claude-opus-5-5"
    assert lanes.current_lane() == "infra" and lanes.current_tenant() is None
    # No pick, answers lane, unknown family: fall through.
    with lanes.tenant_lane(_Customer(), "background"):
        assert lanes.lane_model_for_tier("large") is None
    with lanes.tenant_lane(opus, "answers"):
        assert lanes.lane_model_for_tier("large") is None
    # M3: a Nemotron pick resolves in its own family (the provider switch
    # is the client's job; tests/test_nemotron_lane.py). An unknown family
    # still falls through.
    with lanes.tenant_lane(_Customer(background="nvidia/nemotron-x"), "background"):
        assert lanes.lane_model_for_tier("large") == "nvidia/nemotron-x"
    with lanes.tenant_lane(_Customer(background="gemini-2.5-pro"), "background"):
        assert lanes.lane_model_for_tier("large") is None
    with pytest.raises(ValueError):
        with lanes.tenant_lane(opus, "sideways"):
            pass


def test_client_resolution_order():
    c = LLMClient(provider="anthropic", api_key="k", base_url=None,
                  model_small=None, model_large="claude-sonnet-5-5", model_frontier=None)
    assert c._resolve_model("large", None) == "claude-sonnet-5-5"
    with lanes.tenant_lane(_Customer(background="claude-opus-5-5"), "background"):
        assert c._resolve_model("large", None) == "claude-opus-5-5"
        assert c._resolve_model("small", None) == "claude-haiku-5-5"
        assert c._resolve_model("large", "claude-fable-5-1") == "claude-fable-5-1"
    assert c._resolve_model("large", None) == "claude-sonnet-5-5"
    # A non-Anthropic singleton never takes an Anthropic pick.
    o = LLMClient(provider="openai", api_key="k", base_url="https://x.test/v1",
                  model_small="s", model_large="l", model_frontier="f")
    with lanes.tenant_lane(_Customer(background="claude-opus-5-5"), "background"):
        assert o._resolve_model("large", None) == "l"


@pytest.mark.asyncio
async def test_background_model_route_and_read(monkeypatch, store):
    try:
        from tests.test_accounts_phase_c import _AuthedReq, _use
    except ModuleNotFoundError:
        from test_accounts_phase_c import _AuthedReq, _use
    from crystal_cache.endpoints.customers import get_customer, update_background_model

    _use(monkeypatch)
    c = await store.create_customer(provider="anthropic", model_id="claude-sonnet-5-5", api_key_ref="")
    await store.set_customer_inference_mode(c.id, "managed")
    bearer = f"Bearer {c.api_key}"

    # Free plan: Opus refused, Sonnet accepted, a legacy id stored as its successor.
    with pytest.raises(HTTPException) as e:
        await update_background_model(c.id, _AuthedReq(bearer, body={"model_id": "claude-opus-5-5"}), store)
    assert e.value.status_code == 400
    await update_background_model(c.id, _AuthedReq(bearer, body={"model_id": "claude-haiku-4-5"}), store)
    got = await store.get_customer_by_id(c.id)
    assert got.model_routing_config.background_model_id == "claude-haiku-5-5"
    read = await get_customer(c.id, _AuthedReq(bearer), store)
    assert read.background_model_id == "claude-haiku-5-5" and read.model_id == "claude-sonnet-5-5"
    # The lane reads what was stored.
    with lanes.tenant_lane(got, "background"):
        assert lanes.lane_model_for_tier("large") == "claude-haiku-5-5"
    # Empty clears it.
    await update_background_model(c.id, _AuthedReq(bearer, body={"model_id": ""}), store)
    got = await store.get_customer_by_id(c.id)
    assert got.model_routing_config.background_model_id is None
    assert (await get_customer(c.id, _AuthedReq(bearer), store)).background_model_id is None


def test_background_sites_bind_the_tenant_and_infra_sites_do_not():
    from crystal_cache.agent import mcr_emitter
    from crystal_cache.endpoints import admin
    from crystal_cache.workers import assumptions, cognition, crystallization

    bound = {
        cognition._run_contradiction_scan: 'tenant_lane(c, "background")',
        cognition._run_dedup_scan: 'tenant_lane(c, "background")',
        cognition._run_gap_discovery: 'tenant_lane(c, "background")',
        cognition._run_outbound_review: 'tenant_lane(c, "background")',
        assumptions._run_one_cycle: 'tenant_lane(customer, "background")',
        crystallization.crystallize_document: 'tenant_lane(owner, "background")',
        admin.admin_explain_conflict: 'tenant_lane(owner, "background")',
    }
    for fn, needle in bound.items():
        assert needle in inspect.getsource(fn), fn.__name__
    src = inspect.getsource(cognition)
    assert 'tenant_lane(_tenant, "background")' in src and 'tenant_lane(customer, "background")' in src
    # The critics are infra: no tenant binding anywhere in the emitter.
    assert "tenant_lane" not in inspect.getsource(mcr_emitter)
