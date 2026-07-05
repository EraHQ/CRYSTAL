"""Growth G3 — cost API (endpoints/cost.py) tests.

Direct-call convention (principal injected). Cost visibility is role-gated to
operator+ at the auth layer; the aggregation math is covered in test_cost.py.
These focus on endpoint wiring: shape of the responses, team-scoping
isolation, and the bucket validation guard.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from crystal_cache.endpoints.cost import (
    cost_by_operator,
    cost_by_session,
    cost_summary,
    cost_timeseries,
)

_SONNET = "claude-sonnet-4-6"


async def test_cost_summary_endpoint(store, customer):
    await store.record_llm_call(
        customer.id, model=_SONNET, input_tokens=1_000_000, session_id="s1",
    )
    resp = await cost_summary(principal=(customer, None), store=store)
    s = resp["summary"]
    assert s["cost_micro_usd"] == 3_000_000
    assert s["call_count"] == 1
    assert "average_cost_micro_usd_per_agent" in s


async def test_cost_summary_team_isolated(store, customer):
    other = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="x",
    )
    await store.record_llm_call(
        other.id, model=_SONNET, input_tokens=1_000_000, session_id="sO",
    )
    # The other team's call must not count toward this team's summary.
    resp = await cost_summary(principal=(customer, None), store=store)
    assert resp["summary"]["call_count"] == 0


async def test_cost_by_session_endpoint(store, customer):
    await store.record_llm_call(
        customer.id, model=_SONNET, input_tokens=1_000_000, session_id="sA",
    )
    resp = await cost_by_session(principal=(customer, None), store=store)
    assert any(r["session_id"] == "sA" for r in resp["sessions"])


async def test_cost_by_operator_endpoint(store, customer):
    op, _ = await store.create_operator(
        team_id=customer.id, display_name="Ada",
    )
    await store.record_llm_call(
        customer.id, model=_SONNET, input_tokens=1_000_000,
        session_id="sB", operator_id=op.id,
    )
    resp = await cost_by_operator(principal=(customer, None), store=store)
    assert any(r["operator_id"] == op.id for r in resp["operators"])


async def test_cost_timeseries_endpoint(store, customer):
    await store.record_llm_call(
        customer.id, model=_SONNET, input_tokens=1_000_000, session_id="sT",
    )
    resp = await cost_timeseries(
        principal=(customer, None), store=store, bucket="day", days=30,
    )
    assert resp["bucket"] == "day"
    assert len(resp["series"]) == 1
    assert resp["series"][0]["cost_micro_usd"] == 3_000_000


async def test_cost_timeseries_rejects_bad_bucket(store, customer):
    with pytest.raises(HTTPException) as exc:
        await cost_timeseries(
            principal=(customer, None), store=store, bucket="month", days=30,
        )
    assert exc.value.status_code == 400
