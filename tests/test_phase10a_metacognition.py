"""Phase 10A tests for the metacognitive layer.

Per P0.75: 7 tests covering schema/mixin (M1–M3), algorithm (M4–M5),
end-to-end integration (M6), and idempotency (M7).

Test scope:
  M1: create_item_alignment + get_alignment_for_item round-trip
  M2: create_critique_synthesis + list_syntheses_for_trace round-trip
  M3: list_alignments_for_trace returns oldest-first
  M4: classify_pair covers all 4 alignment classes including
      edit_proposal contradiction
  M5: synthesize_for_trace promotion rules verified per P0.74
  M6: compute_alignment_and_synthesis_for_trace end-to-end with
      4 items across 2 critics → 4 alignments + 1 synthesis +
      correct status transitions
  M7: calling compute twice doesn't double-process
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from crystal_cache.metacognition import (
    classify_pair,
    compute_alignment_and_synthesis_for_trace,
    synthesize_for_trace,
)
from crystal_cache.metacognition.synthesis import (
    RATIONALE_AGENT_SELF_SOLO,
    RATIONALE_BOTH_CRITICS_AGREED,
    RATIONALE_CRITICS_CONTRADICTED,
    RATIONALE_SHADOW_SOLO,
    RATIONALE_SUBSTRATE_DEFERRED,
)
from crystal_cache.models.action_item import ActionItem
from crystal_cache.models.critique import Critique
from crystal_cache.models.item_alignment import ItemAlignment


# ---------------------------------------------------------------------------
# Helpers — synthetic Pydantic objects for the pure-function tests
# ---------------------------------------------------------------------------

def _make_item(
    item_id: str,
    critique_id: str,
    action_type: str,
    content: dict[str, Any],
    *,
    customer_id: str = "cus_test",
    status: str = "pending",
) -> ActionItem:
    return ActionItem(
        id=item_id,
        critique_id=critique_id,
        customer_id=customer_id,
        action_type=action_type,
        content=content,
        status=status,
    )


def _make_critique(
    critique_id: str,
    critic_role: str,
    *,
    customer_id: str = "cus_test",
) -> Critique:
    return Critique(
        id=critique_id,
        customer_id=customer_id,
        critic_role=critic_role,  # type: ignore[arg-type]
        critic_model="test-model",
        observations=[],
        summary_text=None,
        total_action_items=0,
    )


def _make_alignment(
    item_id: str,
    alignment_class: str,
    paired_ids: list[str],
    *,
    customer_id: str = "cus_test",
) -> ItemAlignment:
    return ItemAlignment(
        id=f"al_{item_id}",
        customer_id=customer_id,
        trace_id="trace_test",
        focus_item_id=item_id,
        alignment_class=alignment_class,  # type: ignore[arg-type]
        paired_item_ids=paired_ids,
        confidence=1.0,
        computed_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# M1 — create_item_alignment + get_alignment_for_item round-trip
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_m1_create_and_get_item_alignment(store: Any, customer: Any):
    """Schema + mixin smoke test: alignment row round-trips correctly."""
    # Seed: a trace + critique + action_item to point at.
    trace = await store.create_reasoning_trace(
        customer_id=customer.id,
        sequence_id="seq_m1",
        events=[],
    )
    critique = await store.create_critique(
        customer_id=customer.id,
        critic_role="agent_self",
        critic_model="test-model",
        trace_id=trace.id,
    )
    item = await store.create_action_item(
        critique_id=critique.id,
        customer_id=customer.id,
        action_type="research_task",
        content={"topic": "test topic"},
    )

    alignment = await store.create_item_alignment(
        customer_id=customer.id,
        focus_item_id=item.id,
        alignment_class="same_action",
        trace_id=trace.id,
        paired_item_ids=["other_item_id"],
        confidence=1.0,
    )

    assert alignment.id
    assert alignment.alignment_class == "same_action"
    assert alignment.paired_item_ids == ["other_item_id"]
    assert alignment.trace_id == trace.id

    # Round-trip via get_alignment_for_item.
    fetched = await store.get_alignment_for_item(item.id)
    assert fetched is not None
    assert fetched.id == alignment.id
    assert fetched.alignment_class == "same_action"
    assert fetched.paired_item_ids == ["other_item_id"]


# ---------------------------------------------------------------------------
# M2 — create_critique_synthesis + list_syntheses_for_trace round-trip
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_m2_create_and_list_critique_synthesis(
    store: Any, customer: Any
):
    """Schema + mixin smoke test: synthesis row round-trips and lists."""
    trace = await store.create_reasoning_trace(
        customer_id=customer.id,
        sequence_id="seq_m2",
        events=[],
    )

    synth = await store.create_critique_synthesis(
        customer_id=customer.id,
        trace_id=trace.id,
        promoted_item_ids=["item_a", "item_b"],
        deferred_item_ids=["item_c"],
        dropped_item_ids=[],
        promotion_rationales={"item_a": "test rationale a"},
    )

    assert synth.id
    assert synth.promoted_item_ids == ["item_a", "item_b"]
    assert synth.deferred_item_ids == ["item_c"]
    assert synth.dropped_item_ids == []
    assert synth.promotion_rationales == {"item_a": "test rationale a"}

    # Round-trip via list_syntheses_for_trace.
    syntheses = await store.list_syntheses_for_trace(trace.id)
    assert len(syntheses) == 1
    assert syntheses[0].id == synth.id


# ---------------------------------------------------------------------------
# M3 — list_alignments_for_trace returns oldest-first
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_m3_list_alignments_oldest_first(store: Any, customer: Any):
    """list_alignments_for_trace orders by computed_at ascending."""
    trace = await store.create_reasoning_trace(
        customer_id=customer.id,
        sequence_id="seq_m3",
        events=[],
    )
    critique = await store.create_critique(
        customer_id=customer.id,
        critic_role="agent_self",
        critic_model="test-model",
        trace_id=trace.id,
    )

    # Seed 3 action items + 3 alignments. Order of creation IS the
    # expected order in the result.
    item_ids: list[str] = []
    for i in range(3):
        item = await store.create_action_item(
            critique_id=critique.id,
            customer_id=customer.id,
            action_type="research_task",
            content={"topic": f"topic {i}"},
        )
        item_ids.append(item.id)

    alignment_ids: list[str] = []
    for item_id in item_ids:
        alignment = await store.create_item_alignment(
            customer_id=customer.id,
            focus_item_id=item_id,
            alignment_class="divergent_action",
            trace_id=trace.id,
        )
        alignment_ids.append(alignment.id)

    listed = await store.list_alignments_for_trace(trace.id)
    assert len(listed) == 3
    assert [a.id for a in listed] == alignment_ids


# ---------------------------------------------------------------------------
# M4 — classify_pair covers all 4 alignment classes
# ---------------------------------------------------------------------------

def test_m4_classify_pair_all_classes():
    """The v1 classifier handles each of the 4 alignment classes."""
    # same_action: same action_type, same canonical key.
    a1 = _make_item("a1", "c1", "research_task", {"topic": "fiscal year deadlines"})
    a2 = _make_item("a2", "c2", "research_task", {"topic": "fiscal year deadlines"})
    assert classify_pair(a1, a2) == "same_action"

    # similar_action: same type, similar-but-not-identical key.
    b1 = _make_item("b1", "c1", "research_task", {"topic": "fiscal year deadline 2025"})
    b2 = _make_item("b2", "c2", "research_task", {"topic": "deadline for fiscal year 2024"})
    assert classify_pair(b1, b2) == "similar_action"

    # divergent_action: different action_type.
    c1 = _make_item("c1", "c1", "research_task", {"topic": "abc"})
    c2 = _make_item("c2", "c2", "verification_task", {"crystal_id": "cry_x"})
    assert classify_pair(c1, c2) == "divergent_action"

    # divergent_action: same type, no key overlap.
    d1 = _make_item("d1", "c1", "research_task", {"topic": "weather patterns"})
    d2 = _make_item("d2", "c2", "research_task", {"topic": "stock prices"})
    assert classify_pair(d1, d2) == "divergent_action"

    # contradictory_action: two edit_proposal with same crystal_id but
    # different proposed_change.
    e1 = _make_item(
        "e1", "c1", "edit_proposal",
        {"crystal_id": "cry_x", "proposed_change": "raise threshold to 0.7"},
    )
    e2 = _make_item(
        "e2", "c2", "edit_proposal",
        {"crystal_id": "cry_x", "proposed_change": "lower threshold to 0.3"},
    )
    assert classify_pair(e1, e2) == "contradictory_action"

    # Order symmetry: classify_pair(a, b) == classify_pair(b, a).
    assert classify_pair(a2, a1) == "same_action"
    assert classify_pair(e2, e1) == "contradictory_action"


# ---------------------------------------------------------------------------
# M5 — synthesize_for_trace promotion rules
# ---------------------------------------------------------------------------

def test_m5_synthesis_rules():
    """The v1 synthesis policy applies each rule correctly."""
    critiques = {
        "crit_agent": _make_critique("crit_agent", "agent_self"),
        "crit_shadow": _make_critique("crit_shadow", "shadow"),
    }

    # Build five items exercising each rule.
    # Item 1: substrate_observation → always defer.
    item_substrate = _make_item(
        "i_substrate", "crit_agent", "substrate_observation",
        {"subsystem": "retrieval", "complaint": "x"},
    )
    # Item 2: same_action with 2 critics → promote.
    item_same = _make_item(
        "i_same", "crit_agent", "research_task", {"topic": "x"},
    )
    # Item 3: contradictory → defer.
    item_contra = _make_item(
        "i_contra", "crit_agent", "edit_proposal",
        {"crystal_id": "cry_x", "proposed_change": "raise"},
    )
    # Item 4: divergent agent_self → promote.
    item_div_agent = _make_item(
        "i_div_agent", "crit_agent", "research_task", {"topic": "agent-only"},
    )
    # Item 5: divergent shadow → defer.
    item_div_shadow = _make_item(
        "i_div_shadow", "crit_shadow", "research_task",
        {"topic": "shadow-only"},
    )

    alignments = {
        "i_same": _make_alignment("i_same", "same_action", ["other"]),
        "i_contra": _make_alignment("i_contra", "contradictory_action", ["other"]),
        "i_div_agent": _make_alignment("i_div_agent", "divergent_action", []),
        "i_div_shadow": _make_alignment("i_div_shadow", "divergent_action", []),
        # substrate_observation skipped before alignment is checked.
    }

    promoted, deferred, dropped, rationales = synthesize_for_trace(
        pending_items=[
            item_substrate,
            item_same,
            item_contra,
            item_div_agent,
            item_div_shadow,
        ],
        critiques_by_id=critiques,
        alignments_by_focus_id=alignments,
    )

    # substrate_observation → deferred.
    assert "i_substrate" in deferred
    assert rationales["i_substrate"] == RATIONALE_SUBSTRATE_DEFERRED
    # same_action 2-critic → promoted.
    assert "i_same" in promoted
    assert rationales["i_same"] == RATIONALE_BOTH_CRITICS_AGREED
    # contradictory → deferred.
    assert "i_contra" in deferred
    assert rationales["i_contra"] == RATIONALE_CRITICS_CONTRADICTED
    # divergent agent_self → promoted.
    assert "i_div_agent" in promoted
    assert rationales["i_div_agent"] == RATIONALE_AGENT_SELF_SOLO
    # divergent shadow → deferred.
    assert "i_div_shadow" in deferred
    assert rationales["i_div_shadow"] == RATIONALE_SHADOW_SOLO

    # v1 produces no dropped items.
    assert dropped == []


# ---------------------------------------------------------------------------
# M6 — end-to-end integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_m6_compute_alignment_and_synthesis_end_to_end(
    store: Any, customer: Any
):
    """Seed a trace with agent_self (2 items) + shadow (2 items), one
    same as agent_self's, one different. Expected: 4 alignment rows +
    1 synthesis row + correct status transitions.
    """
    trace = await store.create_reasoning_trace(
        customer_id=customer.id,
        sequence_id="seq_m6",
        events=[],
    )

    # Agent_self critique with 2 items.
    crit_agent = await store.create_critique(
        customer_id=customer.id,
        critic_role="agent_self",
        critic_model="claude-haiku-4-5-20251001",
        trace_id=trace.id,
    )
    agent_item_shared = await store.create_action_item(
        critique_id=crit_agent.id,
        customer_id=customer.id,
        action_type="research_task",
        content={"topic": "shared topic"},
    )
    agent_item_solo = await store.create_action_item(
        critique_id=crit_agent.id,
        customer_id=customer.id,
        action_type="gap_declaration",
        content={"want": "agent solo wants"},
    )

    # Shadow critique with 2 items: one matching agent's first item,
    # one solo.
    crit_shadow = await store.create_critique(
        customer_id=customer.id,
        critic_role="shadow",
        critic_model="claude-opus-4-7",
        trace_id=trace.id,
    )
    shadow_item_shared = await store.create_action_item(
        critique_id=crit_shadow.id,
        customer_id=customer.id,
        action_type="research_task",
        content={"topic": "shared topic"},  # Same canonical key as agent's.
    )
    shadow_item_solo = await store.create_action_item(
        critique_id=crit_shadow.id,
        customer_id=customer.id,
        action_type="evidence_gathering",
        content={"topic": "shadow solo gathering"},
    )

    # Run the engine.
    result = await compute_alignment_and_synthesis_for_trace(
        store=store,
        trace_id=trace.id,
    )

    # Expected: 4 alignments computed (one per pending item).
    assert result["reason"] == "synthesized"
    assert len(result["alignment_ids"]) == 4
    assert result["synthesis_id"] is not None
    assert result["skipped_already_decided"] == 0

    # Expected promotion decisions:
    #   - agent_item_shared: same_action (paired with shadow_item_shared) → promote
    #   - agent_item_solo: divergent agent_self → promote (Rule 5a)
    #   - shadow_item_shared: same_action → promote
    #   - shadow_item_solo: divergent shadow → defer (Rule 5b)
    assert result["promoted_count"] == 3
    assert result["deferred_count"] == 1
    assert result["dropped_count"] == 0

    # Verify the synthesis row contents.
    syntheses = await store.list_syntheses_for_trace(trace.id)
    assert len(syntheses) == 1
    synth = syntheses[0]
    assert agent_item_shared.id in synth.promoted_item_ids
    assert shadow_item_shared.id in synth.promoted_item_ids
    assert agent_item_solo.id in synth.promoted_item_ids
    assert shadow_item_solo.id in synth.deferred_item_ids

    # Verify status transitions on the action items.
    refreshed_shared_agent = (
        await store.list_action_items_for_critique(crit_agent.id)
    )
    statuses = {it.id: it.status for it in refreshed_shared_agent}
    assert statuses[agent_item_shared.id] == "promoted"
    assert statuses[agent_item_solo.id] == "promoted"

    refreshed_shadow = (
        await store.list_action_items_for_critique(crit_shadow.id)
    )
    statuses_s = {it.id: it.status for it in refreshed_shadow}
    assert statuses_s[shadow_item_shared.id] == "promoted"
    assert statuses_s[shadow_item_solo.id] == "deferred"

    # Verify the 4 alignment rows have the right classes.
    alignments = await store.list_alignments_for_trace(trace.id)
    assert len(alignments) == 4
    by_focus = {a.focus_item_id: a for a in alignments}
    assert by_focus[agent_item_shared.id].alignment_class == "same_action"
    assert by_focus[shadow_item_shared.id].alignment_class == "same_action"
    assert by_focus[agent_item_solo.id].alignment_class == "divergent_action"
    assert by_focus[shadow_item_solo.id].alignment_class == "divergent_action"


# ---------------------------------------------------------------------------
# M7 — idempotency
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_m7_compute_twice_does_not_double_process(
    store: Any, customer: Any
):
    """Calling compute_alignment_and_synthesis_for_trace twice doesn't
    re-process already-decided items. Per P0.74: items whose status
    is non-pending are skipped on subsequent runs.
    """
    trace = await store.create_reasoning_trace(
        customer_id=customer.id,
        sequence_id="seq_m7",
        events=[],
    )
    crit = await store.create_critique(
        customer_id=customer.id,
        critic_role="agent_self",
        critic_model="test-model",
        trace_id=trace.id,
    )
    item = await store.create_action_item(
        critique_id=crit.id,
        customer_id=customer.id,
        action_type="research_task",
        content={"topic": "test"},
    )

    # First run: item is pending → gets processed.
    first = await compute_alignment_and_synthesis_for_trace(
        store=store, trace_id=trace.id,
    )
    assert first["reason"] == "synthesized"
    assert first["promoted_count"] + first["deferred_count"] == 1
    assert first["skipped_already_decided"] == 0

    # Second run: item is now non-pending → skipped. A second synthesis
    # row IS created (audit trail) but with empty buckets.
    second = await compute_alignment_and_synthesis_for_trace(
        store=store, trace_id=trace.id,
    )
    assert second["reason"] == "all_items_already_decided"
    assert second["skipped_already_decided"] == 1
    assert second["promoted_count"] == 0
    assert second["deferred_count"] == 0
    # Synthesis row still gets created for the audit trail.
    assert second["synthesis_id"] is not None
    assert second["synthesis_id"] != first["synthesis_id"]

    # And there should be TWO synthesis rows total.
    syntheses = await store.list_syntheses_for_trace(trace.id)
    assert len(syntheses) == 2
