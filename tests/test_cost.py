"""Growth G3 — cost accounting (cost/pricing.py + CostExtensionsMixin).

Pure integer-micro-USD cost math, then the record→aggregate path: every call
emits one row with the right attribution + cost; rollups sum per session /
operator; average is per-agent (D6). Money is never a float. Direct against
the in-memory store fixture; asyncio_mode=auto.
"""
from __future__ import annotations

from crystal_cache.cost.pricing import (
    DEFAULT_PRICE_TABLE,
    compute_cost_micro_usd,
    price_table_from_settings,
)

_SONNET = "claude-sonnet-4-6"


# --- pure pricing math -----------------------------------------------------

def test_compute_cost_known_model():
    # Sonnet placeholder rates: $3/Mtok in, $15/Mtok out → micro-USD/Mtok
    # 3_000_000 / 15_000_000. One Mtok of each = 3_000_000 + 15_000_000.
    cost = compute_cost_micro_usd(
        _SONNET,
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        price_table=DEFAULT_PRICE_TABLE,
    )
    assert cost == 18_000_000
    assert isinstance(cost, int)


def test_unknown_model_costs_zero():
    assert compute_cost_micro_usd("no-such-model", input_tokens=1000) == 0


def test_negative_tokens_clamped_to_zero():
    assert compute_cost_micro_usd(_SONNET, input_tokens=-50, output_tokens=0) == 0


def test_price_override_merges_over_defaults():
    table = price_table_from_settings(
        {"my-model": {"input": 1_000_000, "output": 2_000_000}}
    )
    # Defaults preserved.
    assert _SONNET in table
    # Override usable.
    cost = compute_cost_micro_usd(
        "my-model", input_tokens=1_000_000, output_tokens=0, price_table=table
    )
    assert cost == 1_000_000


def test_malformed_override_skipped():
    # A bad price entry is skipped (defaults win) rather than raising.
    table = price_table_from_settings({"bad": {"input": "oops"}})
    assert "bad" not in table
    assert _SONNET in table


# --- record + aggregate ----------------------------------------------------

async def test_record_and_total(store, customer):
    await store.record_llm_call(
        customer.id, model=_SONNET, input_tokens=1_000_000, session_id="s1",
        origin="interactive",
    )
    await store.record_llm_call(
        customer.id, model=_SONNET, output_tokens=1_000_000, session_id="s1",
    )
    totals = await store.cost_totals_for_team(customer.id)
    assert totals["call_count"] == 2
    assert totals["cost_micro_usd"] == 3_000_000 + 15_000_000


async def test_cost_by_session_and_average_per_agent(store, customer):
    await store.record_llm_call(
        customer.id, model=_SONNET, input_tokens=1_000_000, session_id="sA",
    )
    await store.record_llm_call(
        customer.id, model=_SONNET, input_tokens=1_000_000, session_id="sB",
    )
    rows = await store.cost_by_session(customer.id)
    assert {r["session_id"] for r in rows} == {"sA", "sB"}
    # Total 6_000_000 micro-USD over 2 distinct sessions → 3_000_000 (D6).
    assert await store.average_cost_per_agent(customer.id) == 3_000_000


async def test_unknown_model_records_zero_cost(store, customer):
    row = await store.record_llm_call(
        customer.id, model="mystery-model", input_tokens=10, output_tokens=10,
    )
    assert row["computed_cost_micro_usd"] == 0


async def test_timeseries_buckets(store, customer):
    await store.record_llm_call(
        customer.id, model=_SONNET, input_tokens=1_000_000, session_id="sT",
    )
    series = await store.cost_timeseries(customer.id, bucket="day", days=30)
    assert len(series) == 1
    assert series[0]["call_count"] == 1
    assert series[0]["cost_micro_usd"] == 3_000_000


async def test_session_and_team_cost_reads(store, customer):
    from datetime import datetime, timedelta, timezone

    await store.record_llm_call(
        customer.id, model=_SONNET, input_tokens=1_000_000, session_id="sX",
    )
    assert await store.session_cost_micro_usd("sX") == 3_000_000
    since = datetime.now(timezone.utc) - timedelta(days=1)
    assert await store.team_cost_since(customer.id, since) == 3_000_000
