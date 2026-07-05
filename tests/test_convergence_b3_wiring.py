"""Never-Idle Convergence — P5/B3 worker wiring (dedup + gap-discovery passes).

The generators themselves are covered by test_dedup_scan / test_gap_discovery;
this file covers their cognition-worker idle Phase-3 helpers
(_run_dedup_scan / _run_gap_discovery): each surfaces and counts correctly,
and all three convergence scans share ONE per-UTC-day call ceiling
(scan_state['calls_today']), so enabling more generators bounds — never
multiplies — the daily cost.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from crystal_cache.infrastructure.schema import CrystalRow, FactRow
from crystal_cache.llm import reset_llm_client, set_llm_client
from crystal_cache.workers.cognition import (
    _run_contradiction_scan,
    _run_dedup_scan,
    _run_gap_discovery,
    _utc_day,
)

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _FakeContradicts:
    def is_ready(self) -> bool:
        return True

    def complete(self, **kwargs):
        return "CONTRADICTS"


class _FakeDuplicates:
    def is_ready(self) -> bool:
        return True

    def complete(self, **kwargs):
        return "DUPLICATE"


class _FakeGap:
    def is_ready(self) -> bool:
        return True

    def complete(self, **kwargs):
        return "What is the out-of-network value?"


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


async def _seed_subject(store, customer_id, *, crystal_id, subject, domain):
    async with store.session() as s:
        s.add(CrystalRow(
            id=crystal_id, customer_id=customer_id,
            crystal_type="customer:legacy", summary_vector=[],
        ))
        s.add(FactRow(
            id=f"{crystal_id}_1", crystal_id=crystal_id, pair_type="question_answer",
            prompt_text=f"Topic|a|{subject}|{domain}", claim_text="fact one",
            source_kind="model_reasoning", vector=[], created_at=_T0,
        ))
        s.add(FactRow(
            id=f"{crystal_id}_2", crystal_id=crystal_id, pair_type="question_answer",
            prompt_text=f"Topic|b|{subject}|{domain}", claim_text="fact two",
            source_kind="model_reasoning", vector=[], created_at=_T0 + timedelta(minutes=1),
        ))


# ---------------------------------------------------------------------------
# Dedup pass
# ---------------------------------------------------------------------------

async def test_dedup_helper_surfaces_and_counts(store, customer):
    await _seed_pair(store, customer.id, crystal_id="cA",
                     claim_a="The office opens at 9am",
                     claim_b="We open at nine in the morning")
    fake = _FakeDuplicates()
    scan_state = {"day": None, "calls_today": 0, "cust_offset": 0}

    set_llm_client(fake)
    try:
        spent = await _run_dedup_scan(
            store=store, scan_state=scan_state,
            customers_per_cycle=1, max_candidate_pairs=200,
            max_calls_per_cycle=50, max_calls_per_day=500,
        )
    finally:
        reset_llm_client()
    assert spent == 1
    rows = await store.list_knowledge_conflicts(customer.id)
    assert len(rows) == 1
    assert rows[0].detector == "dedup_scan"
    assert scan_state["calls_today"] == 1


# ---------------------------------------------------------------------------
# Gap-discovery pass
# ---------------------------------------------------------------------------

async def test_gap_helper_surfaces_and_counts(store, customer):
    await _seed_subject(store, customer.id, crystal_id="cA",
                        subject="Deductible", domain="Health")
    fake = _FakeGap()
    scan_state = {"day": None, "calls_today": 0, "cust_offset": 0}

    set_llm_client(fake)
    try:
        spent = await _run_gap_discovery(
            store=store, scan_state=scan_state,
            customers_per_cycle=1, max_subjects_per_cycle=20, max_calls_per_day=500,
        )
    finally:
        reset_llm_client()
    assert spent == 1
    gaps = await store.list_knowledge_gaps(customer.id, status="open")
    assert len(gaps) == 1
    assert gaps[0].source == "gap_discovery"
    assert gaps[0].subject == "Deductible"
    assert scan_state["calls_today"] == 1


async def test_gap_respects_shared_daily_ceiling(store, customer):
    # Daily ceiling already spent (same UTC day) → gap pass spends nothing.
    await _seed_subject(store, customer.id, crystal_id="cA",
                        subject="Deductible", domain="Health")
    fake = _FakeGap()
    scan_state = {"day": _utc_day(), "calls_today": 1, "gap_offset": 0}

    set_llm_client(fake)
    try:
        spent = await _run_gap_discovery(
            store=store, scan_state=scan_state,
            customers_per_cycle=1, max_subjects_per_cycle=20, max_calls_per_day=1,
        )
    finally:
        reset_llm_client()
    assert spent == 0
    assert await store.list_knowledge_gaps(customer.id, status="open") == []


# ---------------------------------------------------------------------------
# Shared daily ceiling across generators
# ---------------------------------------------------------------------------

async def test_shared_daily_ceiling_across_generators(store, customer):
    """The contradiction scan spends the day's whole 1-call ceiling; the dedup
    pass then sees zero remaining and spends nothing — one shared total."""
    await _seed_pair(store, customer.id, crystal_id="cA",
                     claim_a="rate is 120", claim_b="rate is 95")

    scan_state = {"day": None, "calls_today": 0, "cust_offset": 0}

    try:
        set_llm_client(_FakeContradicts())
        spent_contra = await _run_contradiction_scan(
            store=store, scan_state=scan_state,
            customers_per_cycle=1, max_candidate_pairs=200,
            max_calls_per_cycle=50, max_calls_per_day=1,
        )
        assert spent_contra == 1
        assert scan_state["calls_today"] == 1

        # Same scan_state, same daily ceiling of 1 → already exhausted.
        set_llm_client(_FakeDuplicates())
        spent_dedup = await _run_dedup_scan(
            store=store, scan_state=scan_state,
            customers_per_cycle=1, max_candidate_pairs=200,
            max_calls_per_cycle=50, max_calls_per_day=1,
        )
    finally:
        reset_llm_client()
    assert spent_dedup == 0

    # Exactly the one contradiction row; the dedup pass wrote nothing.
    rows = await store.list_knowledge_conflicts(customer.id)
    assert len(rows) == 1
    assert rows[0].detector == "contradiction_scan"
