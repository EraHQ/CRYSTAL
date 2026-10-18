"""Q88=A (2026-10-17): the platform runs on the 5.5 line.

- Every id the console offers and the lane map uses has a price row,
  taken from Anthropic's pricing page the same day (a card with no row
  would meter at zero).
- The infra tier map is Haiku 5.5 / Sonnet 5.5 / Opus 5.5.
- A legacy id (what a tenant stored before the move) is read as its
  successor by the plan check, and the tier clamp rewrites it on touch
  instead of knocking the tenant to the default.
- The data migration rewrites MANAGED rows and leaves byok rows alone.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

from crystal_cache.control import admission
from crystal_cache.cost.pricing import DEFAULT_PRICE_TABLE, resolve_model_price
from crystal_cache.llm.client import _ANTHROPIC_TIER_DEFAULTS


def test_every_offered_model_has_a_price_row_and_the_tiers_moved():
    for model in admission.MANAGED_MODELS_ALL + tuple(_ANTHROPIC_TIER_DEFAULTS.values()):
        assert model in DEFAULT_PRICE_TABLE, model
    assert _ANTHROPIC_TIER_DEFAULTS == {
        "small": "claude-haiku-5-5", "large": "claude-sonnet-5-5", "frontier": "claude-opus-5-5",
    }
    # The official numbers (USD per MTok -> micro): Haiku 5.5 is the
    # up-to-100k band; Fable 5.1's cache read is $0.25.
    assert DEFAULT_PRICE_TABLE["claude-haiku-5-5"].input_micro_per_mtok == 100_000
    assert DEFAULT_PRICE_TABLE["claude-haiku-5-5"].output_micro_per_mtok == 500_000
    assert DEFAULT_PRICE_TABLE["claude-opus-5-5"].input_micro_per_mtok == 4_000_000
    assert DEFAULT_PRICE_TABLE["claude-fable-5-1"].cache_read_micro_per_mtok == 250_000
    # Exact rows win over the family prefix (fable-5-1 is not fable-5's row).
    assert resolve_model_price("claude-fable-5-1") is DEFAULT_PRICE_TABLE["claude-fable-5-1"]
    assert resolve_model_price("claude-sonnet-5-5") is DEFAULT_PRICE_TABLE["claude-sonnet-5-5"]


def test_five_series_models_never_receive_sampling_parameters():
    """Haiku 5.5's migration guide: temperature other than 1, any top_k,
    or any top_p other than 0.99 is a 400. The small tier moved to Haiku
    5.5 with v156, so the seam must drop the 0.0 the JSON lanes pin."""
    from crystal_cache.llm.client import _anthropic_sampling_kwargs

    for model in _ANTHROPIC_TIER_DEFAULTS.values():
        assert _anthropic_sampling_kwargs(model, 0.0) == {}, model
    for model in admission.MANAGED_MODELS_ALL + ("claude-haiku-5-5-20260301", "claude-opus-4-8"):
        assert _anthropic_sampling_kwargs(model, 0.0) == {}, model
    # The 4.x sampling models keep the explicit value.
    assert _anthropic_sampling_kwargs("claude-haiku-4-5-20251001", 0.0) == {"extra_body": {"temperature": 0.0}}
    assert _anthropic_sampling_kwargs("claude-sonnet-4-6", 0.2) == {"extra_body": {"temperature": 0.2}}


def test_legacy_ids_read_as_their_successor():
    assert admission.canonical_managed_model("claude-sonnet-5") == "claude-sonnet-5-5"
    assert admission.canonical_managed_model("claude-opus-4-8") == "claude-opus-5-5"
    assert admission.canonical_managed_model("claude-haiku-5-5") == "claude-haiku-5-5"
    assert admission.canonical_managed_model(None) is None
    for legacy in admission.MANAGED_MODEL_ALIASES.values():
        assert legacy in admission.MANAGED_MODELS_ALL


class _Cfg:
    def __init__(self, model_id):
        self.model_id = model_id


class _Customer:
    def __init__(self, model_id, tier=None, mode="managed"):
        self.id = "cus_t"
        self.model_routing_config = _Cfg(model_id)
        self.subscription_tier = tier
        self.inference_mode = mode


def test_plan_check_accepts_a_legacy_id_in_plan_and_refuses_one_out_of_plan():
    admission.enforce_managed_model(_Customer("claude-sonnet-5"), "claude-sonnet-5")  # free: Sonnet
    with pytest.raises(admission.PlanWallError) as e:
        admission.enforce_managed_model(_Customer("claude-opus-4-8"), "claude-opus-4-8")  # free: no Opus
    assert e.value.code == "model_not_in_plan"
    admission.enforce_managed_model(_Customer("claude-opus-4-8", tier="starter_29"), "claude-opus-4-8")


@pytest.mark.asyncio
async def test_clamp_rewrites_a_legacy_id_instead_of_defaulting_it():
    calls = []

    class _Store:
        async def set_customer_model(self, cid, model):
            calls.append((cid, model))

    # Starter tenant still storing Opus 4.8: in plan via its successor -> rewritten, not defaulted.
    out = await admission.clamp_stored_model_to_tier(_Store(), _Customer("claude-opus-4-8", tier="starter_29"))
    assert out == "claude-opus-5-5" and calls == [("cus_t", "claude-opus-5-5")]
    # Free tenant storing Opus 4.8: out of plan -> the managed default.
    calls.clear()
    out = await admission.clamp_stored_model_to_tier(_Store(), _Customer("claude-opus-4-8"))
    assert out == admission.MANAGED_DEFAULT_MODEL == "claude-sonnet-5-5"
    # Already current: untouched.
    calls.clear()
    assert await admission.clamp_stored_model_to_tier(_Store(), _Customer("claude-sonnet-5-5")) is None
    assert calls == []


@pytest.mark.asyncio
async def test_migration_rewrites_managed_rows_only(store):
    managed = await store.create_customer(provider="anthropic", model_id="claude-sonnet-5", api_key_ref="")
    await store.set_customer_inference_mode(managed.id, "managed")
    byok = await store.create_customer(provider="anthropic", model_id="claude-sonnet-5", api_key_ref="sk-own")
    path = pathlib.Path(__file__).resolve().parents[1] / "memory" / "migrations" / "versions" / "20261017_2300_f2c4d6e8a0b2_model_line_5_5.py"
    spec = importlib.util.spec_from_file_location("mig_f2c4d6e8a0b2", path)
    mig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mig)

    async with store._engine.begin() as conn:
        def _run(sync_conn):
            from alembic.migration import MigrationContext
            from alembic.operations import Operations
            ctx = MigrationContext.configure(sync_conn)
            with Operations.context(ctx):
                mig.upgrade()
        await conn.run_sync(_run)
    assert (await store.get_customer_by_id(managed.id)).model_routing_config.model_id == "claude-sonnet-5-5"
    assert (await store.get_customer_by_id(byok.id)).model_routing_config.model_id == "claude-sonnet-5"
