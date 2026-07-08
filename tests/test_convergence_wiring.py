"""Never-Idle Convergence — P4 wiring (endpoints + cognition-worker Phase 3).

Direct-call convention for endpoints (store injected), matching test_cost_api.
Covers the /conflicts + /backlog reads, the on-demand /conflicts/scan endpoint
(503 without a provider; success via an injected seam fake), and the worker's
idle-cycle scan helper (round-robin across customers, per-cycle + daily budget).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException

from crystal_cache.endpoints.admin import (
    admin_list_backlog,
    admin_list_conflicts,
    admin_scan_conflicts,
)
from crystal_cache.infrastructure.schema import CrystalRow, FactRow
from crystal_cache.llm import reset_llm_client, set_llm_client
from crystal_cache.workers.cognition import _run_contradiction_scan

from fakes import NotReadyLLM

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _FakeContradicts:
    """Seam-shaped client whose discriminator always says CONTRADICTS."""

    def is_ready(self) -> bool:
        return True

    def complete(self, **kwargs):
        return "CONTRADICTS"


async def _seed_pair(store, customer_id, *, crystal_id, claim_a, claim_b):
    async with store.session() as s:
        s.add(CrystalRow(
            id=crystal_id, customer_id=customer_id,
            crystal_type="customer:legacy", summary_vector=[],
        ))
        s.add(FactRow(
            id=f"{crystal_id}_a", crystal_id=crystal_id, pair_type="question_answer",
            prompt_text="", claim_text=claim_a, source_kind="model_reasoning",
            vector=[], created_at=_T0,
        ))
        s.add(FactRow(
            id=f"{crystal_id}_b", crystal_id=crystal_id, pair_type="question_answer",
            prompt_text="", claim_text=claim_b, source_kind="model_reasoning",
            vector=[], created_at=_T0 + timedelta(minutes=1),
        ))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def _admin_req():
    """Unpinned request stub (platform-admin shape) for direct endpoint
    calls — the 2026-07-07 tenant sweep added a `request` param whose
    tenant_pin, when set, force-scopes results."""
    from types import SimpleNamespace
    return SimpleNamespace(state=SimpleNamespace(tenant_pin=None))


async def test_conflicts_endpoint_lists_open(store, customer):
    await store.create_knowledge_conflict(
        customer.id, fact_a_id="f1", fact_b_id="f2",
        claim_a="rate 120", claim_b="rate 95", pair_key="pk1", subject="Rate",
    )
    resp = await admin_list_conflicts(request=_admin_req(), store=store, customer_id=customer.id)
    assert resp["count"] == 1
    assert resp["conflicts"][0]["subject"] == "Rate"
    # model_dump(mode="json") → created_at serialized to a string.
    assert isinstance(resp["conflicts"][0]["created_at"], str)


async def test_backlog_endpoint_shape(store, customer):
    await store.create_knowledge_conflict(
        customer.id, fact_a_id="f1", fact_b_id="f2",
        claim_a="a", claim_b="b", pair_key="pk1",
    )
    resp = await admin_list_backlog(request=_admin_req(), store=store, customer_id=customer.id)
    assert resp["count"] == 1
    item = resp["items"][0]
    assert item["kind"] == "conflict"
    assert set(item) == {"kind", "id", "subject", "status", "priority_score", "created_at"}


async def test_scan_endpoint_503_without_provider(store, customer):
    # Force "no provider" regardless of the test environment's env vars.
    set_llm_client(NotReadyLLM())
    try:
        with pytest.raises(HTTPException) as ei:
            await admin_scan_conflicts(store=store, customer_id=customer.id)
    finally:
        reset_llm_client()
    assert ei.value.status_code == 503


async def test_scan_endpoint_success(store, customer):
    await _seed_pair(
        store, customer.id, crystal_id="cA",
        claim_a="The rate is $120", claim_b="The rate is $95",
    )
    set_llm_client(_FakeContradicts())
    try:
        resp = await admin_scan_conflicts(store=store, customer_id=customer.id)
    finally:
        reset_llm_client()
    assert resp["scan"]["conflicts_found"] == 1
    # The endpoint surfaced an open conflict the list endpoint can read back.
    listed = await admin_list_conflicts(request=_admin_req(), store=store, customer_id=customer.id)
    assert listed["count"] == 1


# ---------------------------------------------------------------------------
# Cognition worker idle Phase 3 helper
# ---------------------------------------------------------------------------

async def test_scan_helper_round_robins_customers(store, customer):
    other = await store.create_customer(
        provider="anthropic", model_id="claude-sonnet-4-5-20250929",
        api_key_ref="sk-test-other",
    )
    await _seed_pair(store, customer.id, crystal_id="c1", claim_a="x 1", claim_b="x 2")
    await _seed_pair(store, other.id, crystal_id="c2", claim_a="y 1", claim_b="y 2")

    fake = _FakeContradicts()
    scan_state = {"day": None, "calls_today": 0, "cust_offset": 0}

    # One customer per cycle → two cycles cover both customers.
    set_llm_client(fake)
    try:
        await _run_contradiction_scan(
            store=store, scan_state=scan_state,
            customers_per_cycle=1, max_candidate_pairs=200,
            max_calls_per_cycle=50, max_calls_per_day=500,
        )
        await _run_contradiction_scan(
            store=store, scan_state=scan_state,
            customers_per_cycle=1, max_candidate_pairs=200,
            max_calls_per_cycle=50, max_calls_per_day=500,
        )
    finally:
        reset_llm_client()

    mine = await store.list_knowledge_conflicts(customer.id)
    theirs = await store.list_knowledge_conflicts(other.id)
    assert len(mine) == 1
    assert len(theirs) == 1
    assert scan_state["calls_today"] == 2


async def test_scan_helper_daily_cap_blocks(store, customer):
    await _seed_pair(store, customer.id, crystal_id="c1", claim_a="x 1", claim_b="x 2")
    fake = _FakeContradicts()
    scan_state = {"day": None, "calls_today": 0, "cust_offset": 0}

    set_llm_client(fake)
    try:
        spent = await _run_contradiction_scan(
            store=store, scan_state=scan_state,
            customers_per_cycle=3, max_candidate_pairs=200,
            max_calls_per_cycle=50, max_calls_per_day=0,  # daily cap exhausted
        )
    finally:
        reset_llm_client()
    assert spent == 0
    assert await store.list_knowledge_conflicts(customer.id) == []


async def test_scan_helper_per_cycle_budget(store, customer):
    # One crystal, 3 facts → 3 candidate pairs; cap calls at 1.
    async with store.session() as s:
        s.add(CrystalRow(
            id="cBig", customer_id=customer.id,
            crystal_type="customer:legacy", summary_vector=[],
        ))
        for i in range(3):
            s.add(FactRow(
                id=f"bf{i}", crystal_id="cBig", pair_type="question_answer",
                prompt_text="", claim_text=f"claim {i}", source_kind="model_reasoning",
                vector=[], created_at=_T0 + timedelta(minutes=i),
            ))

    fake = _FakeContradicts()
    scan_state = {"day": None, "calls_today": 0, "cust_offset": 0}
    set_llm_client(fake)
    try:
        spent = await _run_contradiction_scan(
            store=store, scan_state=scan_state,
            customers_per_cycle=1, max_candidate_pairs=200,
            max_calls_per_cycle=1, max_calls_per_day=500,
        )
    finally:
        reset_llm_client()
    assert spent == 1
    assert scan_state["calls_today"] == 1
