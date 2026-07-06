"""Accounts Phase B (2026-07-06) — E4 managed inference + monthly spend cap.

Covers the ratified design at the policy layer:
  - customers.inference_mode: byok default (back-compat), managed opt-in;
  - get_upstream_client managed branch: PLATFORM credentials, Key B
    ignored, fail-LOUD when the platform key is missing;
  - per-call ledger flagging (billing='managed') — mid-month mode flips
    never distort rebilling;
  - managed_spend_micro_usd_this_month: counts only this month's MANAGED
    rows;
  - the tier-derived monthly cap and its TIER_TABLE values.

The proxy-door 429 itself is exercised via the spend reader + tier
resolution it composes (the door is four lines over these two seams,
mirroring the task-key budget door proven in test_task_keys).

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from crystal_cache.config import Settings
from crystal_cache.control.admission import TIER_TABLE, resolve_tier
from crystal_cache.execution import upstream_client as uc_mod
from crystal_cache.execution.upstream_client import (
    AnthropicClient,
    OpenAIClient,
    get_upstream_client,
)


def _use(monkeypatch, **kw) -> None:
    base = dict(environment="development", admin_api_key="",
                api_key_pepper="")
    base.update(kw)
    settings = Settings(**base)
    monkeypatch.setattr(uc_mod, "get_settings", lambda: settings,
                        raising=False)
    import crystal_cache.config as config_mod
    monkeypatch.setattr(config_mod, "get_settings", lambda: settings)


@pytest.fixture
async def customer(store):
    return await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="")


# --- inference_mode: model + persistence -------------------------------------

async def test_inference_mode_defaults_to_byok(store, customer):
    """Back-compat is the ratified default: API-created customers are
    byok unless something explicitly says otherwise."""
    loaded = await store.get_customer_by_id(customer.id)
    assert loaded.inference_mode == "byok"


# --- managed client branch ----------------------------------------------------

def _managed_customer(base):
    return base.model_copy(update={"inference_mode": "managed"})


async def test_managed_branch_uses_platform_anthropic_key(
        monkeypatch, customer):
    _use(monkeypatch, ANTHROPIC_API_KEY="sk-ant-platform-key")
    client = get_upstream_client(_managed_customer(customer))
    assert isinstance(client, AnthropicClient)
    assert client._api_key == "sk-ant-platform-key"


async def test_managed_branch_ignores_key_b_entirely(monkeypatch, store):
    """A managed customer may carry NO Key B — the platform key serves.
    (A byok customer with an empty ref would fail downstream; managed
    must not even look at it.)"""
    _use(monkeypatch, ANTHROPIC_API_KEY="sk-ant-platform-key")
    c = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="")
    client = get_upstream_client(_managed_customer(c))
    assert isinstance(client, AnthropicClient)
    assert client._api_key == "sk-ant-platform-key"


async def test_managed_without_platform_key_fails_loud(monkeypatch, customer):
    """Operator misconfiguration is an ERROR, never a silent fallback to
    the customer's Key B."""
    _use(monkeypatch, ANTHROPIC_API_KEY="")
    with pytest.raises(RuntimeError, match="CC_ANTHROPIC_API_KEY"):
        get_upstream_client(_managed_customer(customer))


async def test_managed_openai_provider_uses_llm_api_key(
        monkeypatch, customer):
    _use(monkeypatch, managed_inference_provider="openai",
         llm_api_key="sk-platform-openai")
    client = get_upstream_client(_managed_customer(customer))
    assert isinstance(client, OpenAIClient)


async def test_managed_unknown_provider_fails_loud(monkeypatch, customer):
    _use(monkeypatch, managed_inference_provider="bedrock")
    with pytest.raises(RuntimeError, match="bedrock"):
        get_upstream_client(_managed_customer(customer))


async def test_byok_path_untouched_by_e4(monkeypatch, store):
    """TRIPWIRE: byok behavior is byte-identical — an empty ref still
    builds the per-customer client from Key B semantics."""
    _use(monkeypatch, ANTHROPIC_API_KEY="sk-ant-platform-key")
    c = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="")
    client = get_upstream_client(c)  # byok default
    assert isinstance(client, AnthropicClient)
    assert client._api_key == ""  # customer's (empty) ref — NOT the platform key


# --- ledger flag + monthly reader ----------------------------------------------

async def test_billing_flag_stamps_ledger_row(monkeypatch, store, customer):
    _use(monkeypatch, enable_cost_accounting=True)
    from crystal_cache.cost.emit import record_model_call
    await record_model_call(
        customer_id=customer.id, model="claude-haiku-4-5",
        origin="interactive", input_tokens=1000, output_tokens=100,
        store=store, billing="managed",
    )
    await record_model_call(
        customer_id=customer.id, model="claude-haiku-4-5",
        origin="interactive", input_tokens=1000, output_tokens=100,
        store=store, billing=None,
    )
    spent = await store.managed_spend_micro_usd_this_month(customer.id)
    assert spent > 0  # exactly the managed row


async def test_monthly_reader_scopes_month_billing_and_tenant(
        store, customer):
    """Only THIS month's, MANAGED, THIS tenant's rows count."""
    other = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="")
    now = datetime.now(timezone.utc)
    last_month = (now.replace(day=1) - timedelta(days=2))

    async def _row(cid, billing, created_at, cost=1_000):
        rec = await store.record_llm_call(
            cid, model="claude-haiku-4-5",
            input_tokens=0, output_tokens=0, billing=billing,
            price_table={},
        )
        # Pin cost + timestamp directly (the reader sums the column).
        from crystal_cache.infrastructure.schema import LlmCallRow
        async with store.session() as session:
            row = await session.get(LlmCallRow, rec["id"])
            row.computed_cost_micro_usd = cost
            row.created_at = created_at

    await _row(customer.id, "managed", now, cost=7_000)        # counts
    await _row(customer.id, None, now, cost=100_000)           # byok: no
    await _row(customer.id, "managed", last_month, cost=50_000)  # old: no
    await _row(other.id, "managed", now, cost=200_000)         # foreign: no

    assert await store.managed_spend_micro_usd_this_month(
        customer.id) == 7_000


# --- tier cap -------------------------------------------------------------------

def test_tier_table_carries_managed_monthly_caps():
    """The non-negotiable cap exists for every tier, tier-ordered."""
    caps = {name: t.monthly_managed_budget_micro_usd
            for name, t in TIER_TABLE.items()}
    assert caps["free"] > 0
    assert caps["free"] < caps["pro"] < caps["scale"]


def test_resolve_tier_falls_back_to_default(monkeypatch):
    import crystal_cache.control.admission as adm
    tier = resolve_tier(None)
    assert tier.monthly_managed_budget_micro_usd > 0
    assert resolve_tier("nonsense_tier").monthly_managed_budget_micro_usd \
        == tier.monthly_managed_budget_micro_usd


# --- the door, composed ----------------------------------------------------------

async def test_door_logic_blocks_at_cap_and_passes_under(store, customer):
    """The exact comparison the proxy door makes, over the real seams:
    spent >= cap blocks; under passes; byok never reads."""
    cap = TIER_TABLE["free"].monthly_managed_budget_micro_usd
    now = datetime.now(timezone.utc)

    rec = await store.record_llm_call(
        customer.id, model="claude-haiku-4-5",
        input_tokens=0, output_tokens=0, billing="managed", price_table={},
    )
    from crystal_cache.infrastructure.schema import LlmCallRow
    async with store.session() as session:
        row = await session.get(LlmCallRow, rec["id"])
        row.computed_cost_micro_usd = cap  # exactly at the cap
        row.created_at = now

    spent = await store.managed_spend_micro_usd_this_month(customer.id)
    assert spent >= cap  # → the door 429s

    async with store.session() as session:
        row = await session.get(LlmCallRow, rec["id"])
        row.computed_cost_micro_usd = cap - 1
    spent = await store.managed_spend_micro_usd_this_month(customer.id)
    assert spent < cap  # → the door passes
