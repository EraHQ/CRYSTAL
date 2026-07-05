"""Phase 10B tests — scheduler worker + critic calibration tracking.

Per P0.83: 8 tests covering mixin queries (SC1-SC2), calibration
(SC3-SC4), cost cap (SC5), worker integration (SC6-SC7), and
cold-start (SC8).
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from crystal_cache.metacognition import (
    compute_alignment_and_synthesis_for_trace,
    update_calibrations_from_synthesis,
)
from crystal_cache.llm import reset_llm_client, set_llm_client
from crystal_cache.workers.metacognition import (
    _run_one_cycle,
    _shadow_pass,
    _synthesis_pass,
    run_metacognition_worker,
)


# ---------------------------------------------------------------------------
# SC1 — list_traces_needing_shadow_review returns only agent_self-only traces
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sc1_list_traces_needing_shadow_review(store, customer):
    """Three traces: agent_self only, agent_self+shadow, neither.
    Only the first should appear in the eligibility scan."""
    # Trace A: agent_self critique only → ELIGIBLE
    trace_a = await store.create_reasoning_trace(
        customer_id=customer.id,
        sequence_id="seq_sc1_a",
        events=[],
    )
    await store.create_critique(
        customer_id=customer.id,
        critic_role="agent_self",
        critic_model="haiku",
        trace_id=trace_a.id,
    )

    # Trace B: both agent_self AND shadow → NOT eligible
    trace_b = await store.create_reasoning_trace(
        customer_id=customer.id,
        sequence_id="seq_sc1_b",
        events=[],
    )
    await store.create_critique(
        customer_id=customer.id,
        critic_role="agent_self",
        critic_model="haiku",
        trace_id=trace_b.id,
    )
    await store.create_critique(
        customer_id=customer.id,
        critic_role="shadow",
        critic_model="opus",
        trace_id=trace_b.id,
    )

    # Trace C: no critiques → NOT eligible (no agent_self)
    trace_c = await store.create_reasoning_trace(
        customer_id=customer.id,
        sequence_id="seq_sc1_c",
        events=[],
    )

    eligible = await store.list_traces_needing_shadow_review(
        customer_id=customer.id,
    )
    ids = {t.id for t in eligible}
    assert trace_a.id in ids
    assert trace_b.id not in ids
    assert trace_c.id not in ids


# ---------------------------------------------------------------------------
# SC2 — list_traces_needing_synthesis respects settling + skips synthesized
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sc2_list_traces_needing_synthesis(store, customer):
    """Trace with critique and no synthesis is eligible (settling=0).
    A trace already-synthesized is NOT eligible. Settling guard
    (large settling_seconds) suppresses fresh traces.
    """
    # Trace A: has critique, no synthesis → eligible when settling=0
    trace_a = await store.create_reasoning_trace(
        customer_id=customer.id, events=[],
    )
    await store.create_critique(
        customer_id=customer.id,
        critic_role="agent_self",
        critic_model="haiku",
        trace_id=trace_a.id,
    )

    # Trace B: has critique AND synthesis → NEVER eligible
    trace_b = await store.create_reasoning_trace(
        customer_id=customer.id, events=[],
    )
    await store.create_critique(
        customer_id=customer.id,
        critic_role="agent_self",
        critic_model="haiku",
        trace_id=trace_b.id,
    )
    await store.create_critique_synthesis(
        customer_id=customer.id,
        trace_id=trace_b.id,
    )

    # With settling=0, trace_a should appear.
    eligible_no_settling = await store.list_traces_needing_synthesis(
        customer_id=customer.id,
        settling_seconds=0,
    )
    ids = {t.id for t in eligible_no_settling}
    assert trace_a.id in ids
    assert trace_b.id not in ids

    # With huge settling (1 hour), fresh traces are too young.
    eligible_with_settling = await store.list_traces_needing_synthesis(
        customer_id=customer.id,
        settling_seconds=3600,
    )
    # trace_a was created seconds ago → outside the settled window.
    assert trace_a.id not in {t.id for t in eligible_with_settling}


# ---------------------------------------------------------------------------
# SC3 — upsert_critic_calibration creates then updates the same row
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sc3_upsert_critic_calibration(store, customer):
    """First call INSERTs; second call with the same identity UPDATEs."""
    # Insert
    cal_1 = await store.upsert_critic_calibration(
        customer_id=customer.id,
        critic_role="agent_self",
        critic_model="haiku",
        promoted_delta=2,
        deferred_delta=1,
    )
    assert cal_1.promoted_count == 2
    assert cal_1.deferred_count == 1
    assert cal_1.total_proposals == 3
    assert cal_1.dropped_count == 0

    # Update via second call with same identity
    cal_2 = await store.upsert_critic_calibration(
        customer_id=customer.id,
        critic_role="agent_self",
        critic_model="haiku",
        promoted_delta=1,
    )
    assert cal_2.id == cal_1.id  # same row
    assert cal_2.promoted_count == 3  # 2 + 1
    assert cal_2.deferred_count == 1  # unchanged
    assert cal_2.total_proposals == 4

    # Different identity → new row
    cal_other = await store.upsert_critic_calibration(
        customer_id=customer.id,
        critic_role="shadow",
        critic_model="opus",
        promoted_delta=5,
    )
    assert cal_other.id != cal_1.id
    assert cal_other.promoted_count == 5

    # list_calibrations_for_customer returns both, ordered by
    # total_proposals desc.
    listed = await store.list_calibrations_for_customer(customer.id)
    assert len(listed) == 2
    assert listed[0].total_proposals >= listed[1].total_proposals


# ---------------------------------------------------------------------------
# SC4 — engine auto-updates calibration after synthesis (P0.79 integration)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sc4_engine_auto_updates_calibration(store, customer):
    """Reuse a Phase 10A M6-style fixture and verify that the engine
    populates calibration rows for both critic identities.
    """
    trace = await store.create_reasoning_trace(
        customer_id=customer.id, events=[],
    )
    crit_agent = await store.create_critique(
        customer_id=customer.id,
        critic_role="agent_self",
        critic_model="claude-haiku-4-5-20251001",
        trace_id=trace.id,
    )
    agent_shared = await store.create_action_item(
        critique_id=crit_agent.id,
        customer_id=customer.id,
        action_type="research_task",
        content={"topic": "shared topic"},
    )
    agent_solo = await store.create_action_item(
        critique_id=crit_agent.id,
        customer_id=customer.id,
        action_type="gap_declaration",
        content={"want": "agent solo wants"},
    )

    crit_shadow = await store.create_critique(
        customer_id=customer.id,
        critic_role="shadow",
        critic_model="claude-opus-4-7",
        trace_id=trace.id,
    )
    shadow_shared = await store.create_action_item(
        critique_id=crit_shadow.id,
        customer_id=customer.id,
        action_type="research_task",
        content={"topic": "shared topic"},
    )
    shadow_solo = await store.create_action_item(
        critique_id=crit_shadow.id,
        customer_id=customer.id,
        action_type="evidence_gathering",
        content={"topic": "shadow solo gathering"},
    )

    result = await compute_alignment_and_synthesis_for_trace(
        store=store,
        trace_id=trace.id,
    )
    assert result["reason"] == "synthesized"
    assert result.get("calibration_critics_updated") == 2

    # Verify calibration counts per the M6 expected decisions:
    # agent_self: agent_shared promoted (same_action 2-critic) + agent_solo promoted (divergent agent_self)
    # shadow:    shadow_shared promoted (same_action 2-critic) + shadow_solo deferred (divergent shadow)
    cal_agent = await store.get_critic_calibration(
        customer_id=customer.id,
        critic_role="agent_self",
        critic_model="claude-haiku-4-5-20251001",
    )
    assert cal_agent is not None
    assert cal_agent.promoted_count == 2
    assert cal_agent.deferred_count == 0
    assert cal_agent.total_proposals == 2

    cal_shadow = await store.get_critic_calibration(
        customer_id=customer.id,
        critic_role="shadow",
        critic_model="claude-opus-4-7",
    )
    assert cal_shadow is not None
    assert cal_shadow.promoted_count == 1
    assert cal_shadow.deferred_count == 1
    assert cal_shadow.total_proposals == 2


# ---------------------------------------------------------------------------
# SC5 — shadow cost cap blocks new shadow calls when at limit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sc5_cost_cap_blocks_shadow(store, customer, fake_anthropic):
    """Seed N shadow critiques in the last 24h. Then attempt a shadow
    pass with cap=N. Expected: skipped_cost_cap >= 1, shadowed == 0.
    """
    cap = 3

    # Seed N shadow critiques (no traces required for the count query;
    # critic_role='shadow' + recent created_at is all that matters).
    for i in range(cap):
        await store.create_critique(
            customer_id=customer.id,
            critic_role="shadow",
            critic_model="opus",
        )

    # Create one trace + agent_self critique that's now eligible for
    # shadow review.
    target = await store.create_reasoning_trace(
        customer_id=customer.id,
        events=[{"type": "final_text", "text": "test response"}],
    )
    await store.create_critique(
        customer_id=customer.id,
        critic_role="agent_self",
        critic_model="haiku",
        trace_id=target.id,
    )

    # Run the shadow pass with the test cap. The shadow critique routes
    # through the seam; inject the fake so the gate is ready.
    set_llm_client(fake_anthropic)
    try:
        result = await _shadow_pass(
            store=store,
            shadow_max_per_day=cap,
        )
    finally:
        reset_llm_client()

    assert result["skipped_cost_cap"] >= 1
    assert result["shadowed"] == 0  # capped, never ran the LLM


# ---------------------------------------------------------------------------
# SC6 — single worker cycle: shadow + synthesis both fire on eligible trace
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sc6_single_cycle_integration(store, customer, fake_anthropic):
    """Seed an eligible trace and run _run_one_cycle. Both passes
    should do work: a shadow critique gets attached AND a synthesis
    row is written.
    """
    trace = await store.create_reasoning_trace(
        customer_id=customer.id,
        events=[{"type": "final_text", "text": "test response"}],
    )
    agent_critique = await store.create_critique(
        customer_id=customer.id,
        critic_role="agent_self",
        critic_model="claude-haiku-4-5-20251001",
        trace_id=trace.id,
        observations=[
            {"type": "assumption_identified", "text": "obs",
             "confidence": 0.5, "anchors": []}
        ],
        summary_text="agent summary",
    )
    await store.create_action_item(
        critique_id=agent_critique.id,
        customer_id=customer.id,
        action_type="research_task",
        content={"topic": "test"},
    )

    # Script the fake anthropic to return a valid shadow critique JSON.
    fake_anthropic.script_text(json.dumps({
        "observations": [],
        "action_items": [],
        "summary_text": "agent's reasoning looks sound",
    }))

    set_llm_client(fake_anthropic)
    try:
        result = await _run_one_cycle(
            store=store,
            shadow_max_per_day=100,
            settling_seconds=0,
        )
    finally:
        reset_llm_client()

    # Pass 1: shadow review ran on the trace.
    assert result["shadow_shadowed"] == 1
    # Pass 2: synthesis ran (the trace now has at least the agent_self
    # critique; with settling=0 the un-synthesized trace was picked up).
    assert result["synthesis_synthesized"] == 1

    # Verify both artifacts exist.
    critiques = await store.list_critiques_for_trace(trace.id)
    assert any(c.critic_role == "shadow" for c in critiques)
    syntheses = await store.list_syntheses_for_trace(trace.id)
    assert len(syntheses) >= 1


# ---------------------------------------------------------------------------
# SC7 — worker shutdown clean exit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sc7_worker_shutdown_clean_exit(
    store, fake_anthropic, monkeypatch,
):
    """Worker started as asyncio.Task, shutdown_event triggered, task
    completes cleanly without exception.
    """
    monkeypatch.setenv("CC_METACOG_WORKER_INTERVAL_SECONDS", "1")

    shutdown = asyncio.Event()
    set_llm_client(fake_anthropic)
    try:
        task = asyncio.create_task(
            run_metacognition_worker(
                store=store,
                shutdown_event=shutdown,
            )
        )

        # Let it spin briefly so it enters its loop.
        await asyncio.sleep(0.05)

        # Trigger shutdown.
        shutdown.set()

        # Worker should complete within a few seconds.
        await asyncio.wait_for(task, timeout=5.0)
    finally:
        reset_llm_client()
    assert task.done()
    assert not task.cancelled()
    # If the loop raised, awaiting would re-raise; reaching here means
    # clean exit.


# ---------------------------------------------------------------------------
# SC8 — cold-start: first synthesis for an unseen critic creates a row
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sc8_cold_start_creates_calibration_row(store, customer):
    """No prior calibration rows exist. Run synthesis once. A fresh
    calibration row is created for the previously-unseen critic
    identity with counters reflecting the synthesis decisions.
    """
    # Confirm no prior rows.
    prior = await store.list_calibrations_for_customer(customer.id)
    assert prior == []

    # Seed a minimal trace + agent_self critique + one item.
    trace = await store.create_reasoning_trace(
        customer_id=customer.id, events=[],
    )
    critique = await store.create_critique(
        customer_id=customer.id,
        critic_role="agent_self",
        critic_model="unseen-model-v1",
        trace_id=trace.id,
    )
    await store.create_action_item(
        critique_id=critique.id,
        customer_id=customer.id,
        action_type="research_task",
        content={"topic": "cold-start test"},
    )

    # Run synthesis.
    await compute_alignment_and_synthesis_for_trace(
        store=store, trace_id=trace.id,
    )

    # Calibration row was created.
    cal = await store.get_critic_calibration(
        customer_id=customer.id,
        critic_role="agent_self",
        critic_model="unseen-model-v1",
    )
    assert cal is not None
    # The single item was divergent agent_self → promoted per P0.74.
    assert cal.promoted_count == 1
    assert cal.total_proposals == 1
    assert cal.last_synthesis_at is not None
