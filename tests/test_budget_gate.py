"""Cost slice 1c (2026-07-21): the daily background-spend gate.
Ledger sum + threshold + cache semantics; the agent lane never
consults it by construction."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from crystal_cache.workers import budget as budget_mod
from crystal_cache.workers.budget import (
    llm_budget_exhausted,
    reset_budget_cache,
)


@pytest.fixture(autouse=True)
def _fresh_cache():
    reset_budget_cache()
    yield
    reset_budget_cache()


@pytest.mark.asyncio
async def test_no_budget_configured_never_gates(store, monkeypatch):
    from crystal_cache import config
    monkeypatch.setattr(
        config.get_settings(), "daily_llm_budget_usd", None,
    )
    assert not await llm_budget_exhausted(store)


@pytest.mark.asyncio
async def test_gate_trips_on_ledger_total(store, customer, monkeypatch):
    from crystal_cache import config
    monkeypatch.setattr(
        config.get_settings(), "daily_llm_budget_usd", 1.0,  # $1/day
    )
    # DEFAULT_PRICE_TABLE: haiku input = $1/MTok -> 400k tokens = $0.40.
    await store.record_llm_call(
        customer_id=customer.id, origin="cognition",
        model="claude-haiku-4-5", input_tokens=400_000, output_tokens=0,
    )
    assert not await llm_budget_exhausted(store)

    # +700k tokens = +$0.70 -> $1.10 total; bypass the 60s cache.
    await store.record_llm_call(
        customer_id=customer.id, origin="cognition",
        model="claude-haiku-4-5", input_tokens=700_000, output_tokens=0,
    )
    reset_budget_cache()
    assert await llm_budget_exhausted(store)


@pytest.mark.asyncio
async def test_sum_is_cutoff_scoped(store, customer):
    # haiku default $1/MTok -> 300k tokens = $0.30.
    await store.record_llm_call(
        customer_id=customer.id, origin="agent",
        model="claude-haiku-4-5", input_tokens=300_000, output_tokens=0,
    )
    now = datetime.now(timezone.utc)
    assert await store.sum_llm_cost_since_micro(
        now - timedelta(hours=1)
    ) >= 300_000  # $0.30 at $3/MTok
    assert await store.sum_llm_cost_since_micro(
        now + timedelta(hours=1)
    ) == 0


@pytest.mark.asyncio
async def test_per_customer_gate_scopes_by_customer(store, customer, monkeypatch):
    """Rails for per-plan limits: customer A exhausted, customer B
    unaffected — the resolver (budget_for_customer) is the single
    switch point to dynamic plans."""
    from crystal_cache import config
    from crystal_cache.workers.budget import customer_llm_budget_exhausted
    monkeypatch.setattr(
        config.get_settings(),
        "daily_llm_budget_per_customer_usd", 1.0,
    )
    other = await store.create_customer(
        provider="anthropic", model_id="claude-haiku-4-5",
        api_key_ref="test-key-other",
    )

    # Customer A: $1.10 spent -> exhausted.
    await store.record_llm_call(
        customer_id=customer.id, origin="cognition",
        model="claude-haiku-4-5", input_tokens=1_100_000, output_tokens=0,
    )
    assert await customer_llm_budget_exhausted(store, customer.id)
    # Customer B: nothing spent -> proceeds.
    assert not await customer_llm_budget_exhausted(store, other.id)


@pytest.mark.asyncio
async def test_release_document_to_pending(store, customer):
    doc = await store.create_document_upload(
        customer.id, "b.py", "def b(): pass",
    )
    claimed = await store.claim_pending_documents_batch(limit=5)
    assert any(d.id == doc.id for d in claimed)
    await store.release_document_to_pending(doc.id)
    again = await store.claim_pending_documents_batch(limit=5)
    assert any(d.id == doc.id for d in again)


@pytest.mark.asyncio
async def test_spend_by_family_aggregation(store, customer):
    """1d: the Inspector panel's feed — grouped by origin, costliest
    first, customer-scopable."""
    await store.record_llm_call(
        customer_id=customer.id, origin="cognition",
        model="claude-haiku-4-5", input_tokens=2_000_000, output_tokens=0,
    )
    await store.record_llm_call(
        customer_id=customer.id, origin="ingest",
        model="claude-haiku-4-5", input_tokens=1_000_000, output_tokens=0,
    )
    from datetime import datetime, timedelta, timezone
    cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
    fams = await store.sum_llm_cost_by_origin_since(
        cutoff, customer_id=customer.id,
    )
    assert [f["origin"] for f in fams] == ["cognition", "ingest"]
    assert fams[0]["cost_micro_usd"] == 2_000_000   # $2 at $1/MTok
    assert fams[0]["calls"] == 1
