"""Phase 8.5 smoke tests for MCR schema + mixin CRUD.

Per the locked Phase 8.5 decisions (P0.34–P0.40):

  Test M1: reasoning_trace round-trip — create → get → list
  Test M2: critique round-trip — create → list by trace → list by sequence
  Test M3: action_item round-trip — create → list by critique →
           list by status → update status (lifecycle transition)
  Test M4: critic_role calibration scan — list_critiques_by_role
  Test M5: action_item lifecycle — pending → promoted with
           metacog_decision_at auto-default

All tests use the in-memory SQLite + store fixture established in
Phase 8 conftest. No new fixtures needed; the existing `customer`
fixture is reused.

R14 note: every assertion below corresponds to a runtime check
performed by pytest. The Phase 8.5 close-out in the ledger records
the pytest output that backs the "passes" claims.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest


# ===========================================================================
# Test M1 — Reasoning trace round-trip
# ===========================================================================

@pytest.mark.asyncio
async def test_reasoning_trace_create_get_list(
    customer: Any,
    store: Any,
):
    """A reasoning trace created via the mixin must round-trip
    through `get_reasoning_trace` and appear in
    `list_traces_for_sequence`. The five aggregate JSON columns
    (crystals_used, tool_calls, inferences, borders_crossed,
    gaps_felt) and the events list must all survive the round-trip
    unchanged.
    """
    # Sample trace content shaped to match the per-entry schemas
    # documented in the ReasoningTraceRow class docstring.
    events = [
        {"type": "tool_call", "tool_name": "knowledge_search",
         "input": {"query": "alpha"}, "output_snippet": "no matches"},
        {"type": "inference", "claim": "the user is asking about X",
         "basis": "explicit phrasing"},
    ]
    crystals_used = ["cry_aaa", "cry_bbb"]
    tool_calls = [
        {"tool_name": "knowledge_search", "input": {"query": "alpha"},
         "output": {"matched_fact_ids": []}, "role": "load_bearing"},
    ]
    inferences = [
        {"claim": "no prior knowledge on this topic",
         "basis": "empty knowledge_search", "confidence": 0.9},
    ]
    borders_crossed = [
        {"claim": "X is probably true",
         "agent_confidence": 0.6, "flagged_by_agent": True},
    ]
    gaps_felt = [
        {"want": "verified data on X", "why_needed": "to answer reliably"},
    ]

    trace = await store.create_reasoning_trace(
        customer_id=customer.id,
        events=events,
        sequence_id="seq_test_001",
        turn_index=0,
        crystals_used=crystals_used,
        tool_calls=tool_calls,
        inferences=inferences,
        borders_crossed=borders_crossed,
        gaps_felt=gaps_felt,
    )

    assert trace.id
    assert trace.customer_id == customer.id
    assert trace.sequence_id == "seq_test_001"
    assert trace.turn_index == 0
    assert trace.events == events
    assert trace.crystals_used == crystals_used
    assert trace.tool_calls == tool_calls
    assert trace.inferences == inferences
    assert trace.borders_crossed == borders_crossed
    assert trace.gaps_felt == gaps_felt

    # get_reasoning_trace returns the same content.
    fetched = await store.get_reasoning_trace(trace.id)
    assert fetched is not None
    assert fetched.id == trace.id
    assert fetched.events == events
    assert fetched.borders_crossed == borders_crossed

    # list_traces_for_sequence finds it.
    listed = await store.list_traces_for_sequence(
        customer_id=customer.id,
        sequence_id="seq_test_001",
    )
    assert len(listed) == 1
    assert listed[0].id == trace.id

    # get_reasoning_trace on a missing id returns None.
    missing = await store.get_reasoning_trace("trace_does_not_exist")
    assert missing is None


# ===========================================================================
# Test M2 — Critique round-trip + soft-join scans
# ===========================================================================

@pytest.mark.asyncio
async def test_critique_round_trip_by_trace_and_sequence(
    customer: Any,
    store: Any,
):
    """A critique created with both trace_id and (sequence_id,
    turn_index) populated must be findable through either lookup
    path. The 8-type observation taxonomy must survive the JSON
    round-trip.
    """
    # Create the trace first so we have a trace_id to point at.
    trace = await store.create_reasoning_trace(
        customer_id=customer.id,
        sequence_id="seq_test_002",
        turn_index=0,
    )

    observations = [
        {"type": "border_crossing_unflagged",
         "text": "agent claimed X with no flag", "confidence": 0.8,
         "anchors": [{"event_index": 1}]},
        {"type": "gap_papered_over",
         "text": "agent glossed over the missing data",
         "confidence": 0.7, "anchors": []},
    ]

    critique = await store.create_critique(
        customer_id=customer.id,
        critic_role="agent_self",
        critic_model="claude-sonnet-4-5-20250929",
        trace_id=trace.id,
        sequence_id="seq_test_002",
        turn_index=0,
        observations=observations,
        summary_text="The agent should have flagged its uncertainty.",
    )

    assert critique.id
    assert critique.critic_role == "agent_self"
    assert critique.critic_model == "claude-sonnet-4-5-20250929"
    assert critique.observations == observations
    assert critique.summary_text == "The agent should have flagged its uncertainty."
    assert critique.total_action_items == 0

    # Lookup by trace_id.
    by_trace = await store.list_critiques_for_trace(trace.id)
    assert len(by_trace) == 1
    assert by_trace[0].id == critique.id
    assert by_trace[0].observations == observations

    # Lookup by (customer_id, sequence_id) without turn_index.
    by_seq = await store.list_critiques_for_sequence(
        customer_id=customer.id,
        sequence_id="seq_test_002",
    )
    assert len(by_seq) == 1
    assert by_seq[0].id == critique.id

    # Lookup by (customer_id, sequence_id, turn_index).
    by_seq_turn = await store.list_critiques_for_sequence(
        customer_id=customer.id,
        sequence_id="seq_test_002",
        turn_index=0,
    )
    assert len(by_seq_turn) == 1
    assert by_seq_turn[0].id == critique.id

    # Critique against a different turn doesn't match the turn-narrow
    # query but DOES match the sequence-wide query.
    other_critique = await store.create_critique(
        customer_id=customer.id,
        critic_role="agent_self",
        critic_model="claude-sonnet-4-5-20250929",
        sequence_id="seq_test_002",
        turn_index=1,
    )
    by_seq_all = await store.list_critiques_for_sequence(
        customer_id=customer.id,
        sequence_id="seq_test_002",
    )
    assert {c.id for c in by_seq_all} == {critique.id, other_critique.id}
    by_seq_turn_0 = await store.list_critiques_for_sequence(
        customer_id=customer.id,
        sequence_id="seq_test_002",
        turn_index=0,
    )
    assert [c.id for c in by_seq_turn_0] == [critique.id]


# ===========================================================================
# Test M3 — Action item round-trip + FK + lifecycle
# ===========================================================================

@pytest.mark.asyncio
async def test_action_item_round_trip_and_lifecycle(
    customer: Any,
    store: Any,
):
    """An action item linked to a critique via FK must round-trip,
    appear in `list_action_items_for_critique` AND
    `list_action_items_by_status`, and accept lifecycle transitions
    through `update_action_item_status`.
    """
    critique = await store.create_critique(
        customer_id=customer.id,
        critic_role="shadow",
        critic_model="claude-opus-4-5-20251201",
    )

    item = await store.create_action_item(
        critique_id=critique.id,
        customer_id=customer.id,
        action_type="research_task",
        content={"topic": "the rate of methane emissions from cattle",
                 "scope": "broad", "why_needed": "agent flagged a gap"},
        critic_confidence=0.85,
    )

    assert item.id
    assert item.critique_id == critique.id
    assert item.customer_id == customer.id
    assert item.action_type == "research_task"
    assert item.content["topic"] == "the rate of methane emissions from cattle"
    assert item.critic_confidence == 0.85
    assert item.status == "pending"
    assert item.metacog_decision_at is None
    assert item.acted_artifact_id is None

    # list_action_items_for_critique.
    by_critique = await store.list_action_items_for_critique(critique.id)
    assert len(by_critique) == 1
    assert by_critique[0].id == item.id

    # list_action_items_by_status — pending matches.
    pending = await store.list_action_items_by_status(
        customer_id=customer.id, status="pending",
    )
    assert len(pending) == 1
    assert pending[0].id == item.id

    # Narrow by action_type.
    pending_research = await store.list_action_items_by_status(
        customer_id=customer.id, status="pending",
        action_type="research_task",
    )
    assert len(pending_research) == 1
    pending_verify = await store.list_action_items_by_status(
        customer_id=customer.id, status="pending",
        action_type="verification_task",
    )
    assert pending_verify == []

    # Transition pending → promoted. metacog_decision_at should
    # auto-default to now since the caller didn't supply one.
    promoted = await store.update_action_item_status(
        action_item_id=item.id,
        status="promoted",
    )
    assert promoted is not None
    assert promoted.status == "promoted"
    assert promoted.metacog_decision_at is not None
    # Sanity: should be within a minute of now.
    now = datetime.now(timezone.utc)
    assert abs((now - promoted.metacog_decision_at).total_seconds()) < 60

    # The item no longer appears in pending.
    pending_after = await store.list_action_items_by_status(
        customer_id=customer.id, status="pending",
    )
    assert pending_after == []
    promoted_list = await store.list_action_items_by_status(
        customer_id=customer.id, status="promoted",
    )
    assert len(promoted_list) == 1
    assert promoted_list[0].id == item.id

    # Transition promoted → acted with an acted_artifact_id.
    acted = await store.update_action_item_status(
        action_item_id=item.id,
        status="acted",
        acted_artifact_id="cog_task_aaa",
    )
    assert acted is not None
    assert acted.status == "acted"
    assert acted.acted_artifact_id == "cog_task_aaa"

    # update on a non-existent id returns None.
    missing = await store.update_action_item_status(
        action_item_id="ai_does_not_exist",
        status="dropped",
    )
    assert missing is None


# ===========================================================================
# Test M4 — Critic calibration scan (list_critiques_by_role)
# ===========================================================================

@pytest.mark.asyncio
async def test_list_critiques_by_role_supports_calibration_scan(
    customer: Any,
    store: Any,
):
    """`list_critiques_by_role` is the read path for Phase 10's
    calibration loop ("how reliable has this critic been?").
    Verify it filters by critic_role, supports an optional `since`
    timestamp, and respects `limit`.
    """
    # Create 3 agent_self critiques and 2 shadow critiques.
    for i in range(3):
        await store.create_critique(
            customer_id=customer.id,
            critic_role="agent_self",
            critic_model="claude-sonnet-4-5-20250929",
            sequence_id=f"seq_cal_{i}",
            turn_index=0,
        )
    for i in range(2):
        await store.create_critique(
            customer_id=customer.id,
            critic_role="shadow",
            critic_model="claude-opus-4-5-20251201",
            sequence_id=f"seq_cal_{i}",
            turn_index=0,
        )

    # Filter by role.
    self_critiques = await store.list_critiques_by_role(
        customer_id=customer.id, critic_role="agent_self",
    )
    assert len(self_critiques) == 3
    for c in self_critiques:
        assert c.critic_role == "agent_self"

    shadow_critiques = await store.list_critiques_by_role(
        customer_id=customer.id, critic_role="shadow",
    )
    assert len(shadow_critiques) == 2
    for c in shadow_critiques:
        assert c.critic_role == "shadow"
        assert c.critic_model == "claude-opus-4-5-20251201"

    # `since` filters older critiques out. Use a future timestamp
    # to confirm filtering happens — should return zero rows.
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    none_since_future = await store.list_critiques_by_role(
        customer_id=customer.id, critic_role="agent_self",
        since=future,
    )
    assert none_since_future == []

    # `limit` caps the result.
    limited = await store.list_critiques_by_role(
        customer_id=customer.id, critic_role="agent_self",
        limit=2,
    )
    assert len(limited) == 2

    # A specialist role with no critiques returns empty (proving
    # the literal-type-allow-extension contract works).
    specialist = await store.list_critiques_by_role(
        customer_id=customer.id, critic_role="specialist",
    )
    assert specialist == []


# ===========================================================================
# Test M5 — Soft join across critique → trace
# ===========================================================================

@pytest.mark.asyncio
async def test_critique_resolves_to_trace_via_soft_join(
    customer: Any,
    store: Any,
):
    """The soft-join key (customer_id, sequence_id, turn_index)
    must let a critique written BEFORE its trace exists still resolve
    to the trace via `list_critiques_for_sequence` once the trace
    lands. This is the Phase 9 open Q2 path — agent self-critique
    may write before the trace finishes streaming.
    """
    # Critique writes FIRST, no trace_id, only the soft-join key.
    critique = await store.create_critique(
        customer_id=customer.id,
        critic_role="agent_self",
        critic_model="claude-sonnet-4-5-20250929",
        sequence_id="seq_softjoin_001",
        turn_index=0,
        observations=[{"type": "assumption_identified",
                       "text": "...", "confidence": 0.5,
                       "anchors": []}],
    )
    assert critique.trace_id is None
    assert critique.sequence_id == "seq_softjoin_001"

    # Trace writes LATER.
    trace = await store.create_reasoning_trace(
        customer_id=customer.id,
        sequence_id="seq_softjoin_001",
        turn_index=0,
    )

    # Both should be findable via the sequence lookup independently.
    critiques = await store.list_critiques_for_sequence(
        customer_id=customer.id, sequence_id="seq_softjoin_001",
    )
    traces = await store.list_traces_for_sequence(
        customer_id=customer.id, sequence_id="seq_softjoin_001",
    )
    assert len(critiques) == 1
    assert len(traces) == 1
    assert critiques[0].id == critique.id
    assert traces[0].id == trace.id

    # list_critiques_for_trace using the trace's id does NOT match
    # because the critique's trace_id was never updated. This is
    # the documented behavior — sequence-key lookup is the
    # primary join when trace_id wasn't set.
    by_trace_id = await store.list_critiques_for_trace(trace.id)
    assert by_trace_id == []
