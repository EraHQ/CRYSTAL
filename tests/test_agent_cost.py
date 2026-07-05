"""C0 — agent-surface cost parity (endpoints/agent.py::record_agent_llm_cost).

The proxy emits one record_llm_call row per turn (Growth G3); the agent
surface did not, leaving CRYS runs invisible to the cost ledger and the
Inspector's spend views. These verify the wiring helper: it records one row
with origin="agent", the run's tokens, session_id=sequence_id, and a non-zero
computed cost; respects the enable_cost_accounting flag; forwards cache tokens
(forward-compat for C1); and is fail-safe. Direct against the in-memory store
fixture; asyncio_mode=auto.
"""
from __future__ import annotations

from crystal_cache.config import settings
from crystal_cache.endpoints.agent import record_agent_llm_cost

# CRYS's default turn model — priced in DEFAULT_PRICE_TABLE, so cost > 0.
_MODEL = "claude-sonnet-4-5-20250929"


def _result(**over: object) -> dict:
    base = {"model": _MODEL, "prompt_tokens": 1000, "completion_tokens": 200}
    base.update(over)
    return base


async def test_records_agent_cost_row(store, customer, monkeypatch):
    monkeypatch.setattr(settings, "enable_cost_accounting", True)
    row = await record_agent_llm_cost(
        store=store,
        customer_id=customer.id,
        result=_result(),
        sequence_id="seq_abc",
    )
    assert row is not None
    assert row["origin"] == "agent"
    assert row["input_tokens"] == 1000
    assert row["output_tokens"] == 200
    assert row["session_id"] == "seq_abc"
    assert row["operator_id"] is None
    assert row["computed_cost_micro_usd"] > 0

    totals = await store.cost_totals_for_team(customer.id)
    assert totals["call_count"] == 1
    assert totals["input_tokens"] == 1000
    assert totals["output_tokens"] == 200


async def test_session_id_groups_per_agent(store, customer, monkeypatch):
    # sequence_id is the agent unit the per-agent rollups group by.
    monkeypatch.setattr(settings, "enable_cost_accounting", True)
    await record_agent_llm_cost(
        store=store,
        customer_id=customer.id,
        result=_result(),
        sequence_id="conv_1",
    )
    rows = await store.cost_by_session(customer.id)
    assert [r["session_id"] for r in rows] == ["conv_1"]


async def test_disabled_flag_records_nothing(store, customer, monkeypatch):
    monkeypatch.setattr(settings, "enable_cost_accounting", False)
    row = await record_agent_llm_cost(
        store=store,
        customer_id=customer.id,
        result=_result(),
        sequence_id="seq_x",
    )
    assert row is None
    totals = await store.cost_totals_for_team(customer.id)
    assert totals["call_count"] == 0


async def test_cache_tokens_forwarded_when_present(store, customer, monkeypatch):
    # Forward-compat for C1: cache-token fields on the result are recorded.
    monkeypatch.setattr(settings, "enable_cost_accounting", True)
    row = await record_agent_llm_cost(
        store=store,
        customer_id=customer.id,
        result=_result(cache_creation_tokens=500, cache_read_tokens=4000),
        sequence_id=None,
    )
    assert row is not None
    assert row["cache_creation_tokens"] == 500
    assert row["cache_read_tokens"] == 4000


async def test_missing_tokens_default_to_zero(store, customer, monkeypatch):
    monkeypatch.setattr(settings, "enable_cost_accounting", True)
    row = await record_agent_llm_cost(
        store=store,
        customer_id=customer.id,
        result={"model": _MODEL},  # no token keys at all
        sequence_id=None,
    )
    assert row is not None
    assert row["input_tokens"] == 0
    assert row["output_tokens"] == 0
    assert row["computed_cost_micro_usd"] == 0
