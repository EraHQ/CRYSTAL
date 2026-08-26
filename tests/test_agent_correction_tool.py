"""Audit (e) stage 1.9 — the propose_correction tool + finalize-side MCR pairs.

The migration target the retirement doc names for
`test_phase9b_signal_handler_mcr.py`'s correction pins. Seven groups per
the ratified gate (Q4=A, 2026-08-26):

  1. S2 migration, inverted where the port demands it: tool call →
     emitter walk → Critique(source_contradiction) + ActionItem
     (edit_proposal) with the NEW content {crystal_id, proposed_change,
     rationale}, anchor carrying crystal_id + stored_value — and
     critique.trace_id == trace.id, the deliberate inversion of the old
     test's `trace_id is None` assert (the S2-214 non-reproduction pin).
  2. S5 adaptation: two propose_correction calls + non-correction tools
     in one turn → exactly two pairs, nothing from the other tools.
  3. Exit criterion 6 end-to-end: the persisted item resolves through
     alignment._canonical_key — the assertion the proxy version could
     never make — and two proposals for the same crystal with different
     changes classify contradictory_action.
  4. Q3 placement pin: skip_self_critique=True still persists the pair
     (trace-yes, Haiku-no, corrections-yes).
  5. Validation refusals produce no MCR: foreign crystal, missing
     crystal, empty proposed_change, empty rationale → error dict from
     the tool, and the walk writes zero rows.
  6. Q1 rider pin: trace-persist failure → no pairs, no trace_id=None
     fallback rows.
  7. Return-shape honesty: correction_critique_ids /
     correction_action_item_ids stay distinct from the Haiku
     critique_id.

Payload note (retirement finding 1b): the proxy's edit_proposal content
({key, old_value, new_value}) was invisible to the metacognitive
classifier — _canonical_key returned "" and the contradiction rule never
fired. Group 3 pins that the ported shape actually feeds the classifier.
"""
from __future__ import annotations

from typing import Any

import pytest

from crystal_cache.agent.mcr_emitter import emit_mcr_artifacts
from crystal_cache.agent.tools.curation import propose_correction
from crystal_cache.agent.tools.retrievers import set_tool_state
from crystal_cache.metacognition.alignment import _canonical_key, classify_pair
from crystal_cache.models.crystal import Crystal


AGENT_MODEL = "claude-sonnet-4-5-20250929"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _CleanCritiqueClient:
    """Fake upstream client for the Haiku self-critique call.

    Returns the clean-critique JSON as a plain string — run_self_critique
    accepts a str result directly, so no usage stamping is attempted.
    """

    def complete_detailed(self, **kwargs: Any) -> str:
        return (
            '{"observations": [], "action_items": [], '
            '"summary_text": "clean"}'
        )


async def _mk_crystal(
    store: Any,
    customer_id: str,
    cid: str = "cry_corr_1",
    summary: str = "Project deadline is March 15, 2027.",
) -> str:
    await store.upsert_crystal(Crystal(
        id=cid,
        customer_id=customer_id,
        summary_vector=[0.0, 0.0, 0.0, 0.0],
        summary_text=summary,
    ))
    return cid


def _entry(
    tool_input: dict[str, Any],
    output: Any,
    *,
    is_error: bool = False,
    tool_name: str = "propose_correction",
) -> dict[str, Any]:
    """One tool_calls_log entry, shaped exactly as Agent.run appends them."""
    return {
        "iteration": 1,
        "tool_name": tool_name,
        "tool_use_id": "tu_test",
        "input": tool_input,
        "output": output,
        "is_error": is_error,
        "duration_ms": 3,
    }


def _agent_result(
    tool_calls: list[dict[str, Any]],
    model: str = AGENT_MODEL,
) -> dict[str, Any]:
    """Minimal Agent.run()-shaped result dict."""
    return {
        "id": "chatcmpl-agent-test",
        "model": model,
        "messages": [],
        "final_text": "Noted — the stored value looks outdated.",
        "stop_reason": "end_turn",
        "iterations": 1,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "cache_creation_tokens": None,
        "cache_read_tokens": None,
        "tool_calls": tool_calls,
        "duration_ms": 42,
    }


async def _call_tool(
    tool_state: dict[str, Any],
    customer_id: str,
    **kwargs: Any,
) -> dict[str, Any]:
    set_tool_state(tool_state)
    return await propose_correction(customer_id=customer_id, **kwargs)


# ===========================================================================
# Group 1 — S2 migration, inverted: the pair lands with a REAL trace_id
# ===========================================================================

@pytest.mark.asyncio
async def test_correction_pair_persists_with_real_trace_id(
    customer: Any, store: Any, tool_state: dict[str, Any],
):
    """The full lane: real tool call → its logged output → emitter walk →
    Critique(source_contradiction) + ActionItem(edit_proposal) carrying the
    trace's id. `critique.trace_id == trace.id` is the deliberate inversion
    of the proxy test's `trace_id is None` — the S2-214 non-reproduction pin.
    Runs the default path (skip_self_critique=False) with a fake clean
    Haiku critique so both critic lanes coexist.
    """
    cid = await _mk_crystal(store, customer.id)
    tool_input = {
        "crystal_id": cid,
        "proposed_change": "Project deadline is April 1, 2027.",
        "rationale": "User stated the new deadline explicitly this turn.",
        "disputed_claim": "deadline of March 15",
    }
    output = await _call_tool(tool_state, customer.id, **tool_input)
    assert output["proposed"] is True
    assert output["crystal_id"] == cid
    # Q3=A: the snippet is the crystal's ACTUAL stored value.
    assert "March 15" in output["stored_value"]

    result = _agent_result([_entry(tool_input, output)])
    out = await emit_mcr_artifacts(
        store=store,
        customer_id=customer.id,
        user_query="update the deadline",
        agent_result=result,
        anthropic_client=_CleanCritiqueClient(),
        sequence_id="seq_g1",
        turn_index=0,
    )

    assert out["trace_id"] is not None
    assert len(out["correction_critique_ids"]) == 1
    assert len(out["correction_action_item_ids"]) == 1

    critiques = await store.list_critiques_for_sequence(
        customer_id=customer.id, sequence_id="seq_g1", turn_index=0,
    )
    corr = [c for c in critiques if c.id in out["correction_critique_ids"]]
    assert len(corr) == 1
    critique = corr[0]
    # The inversion: born linked, never orphaned.
    assert critique.trace_id == out["trace_id"]
    assert critique.critic_role == "agent_self"
    # P0.52: the critic is the agent model that produced the signal.
    assert critique.critic_model == AGENT_MODEL
    assert critique.total_action_items == 1
    assert len(critique.observations) == 1
    obs = critique.observations[0]
    assert obs["type"] == "source_contradiction"
    assert obs["confidence"] == 0.8
    assert cid in obs["text"]
    assert len(obs["anchors"]) == 1
    anchor = obs["anchors"][0]
    assert anchor["crystal_id"] == cid
    assert "March 15" in anchor["stored_value"]
    assert anchor["disputed_claim"] == "deadline of March 15"

    items = await store.list_action_items_for_critique(critique.id)
    assert len(items) == 1
    item = items[0]
    assert item.action_type == "edit_proposal"
    assert item.status == "pending"
    assert item.critic_confidence == 0.8
    # Finding 1b: the agent's canonical edit_proposal shape.
    assert item.content["crystal_id"] == cid
    assert item.content["proposed_change"] == (
        "Project deadline is April 1, 2027."
    )
    assert item.content["rationale"] == (
        "User stated the new deadline explicitly this turn."
    )
    assert item.content["disputed_claim"] == "deadline of March 15"


# ===========================================================================
# Group 2 — S5 adaptation: mixed batch produces exactly the right pairs
# ===========================================================================

@pytest.mark.asyncio
async def test_mixed_tool_batch_produces_exactly_two_pairs(
    customer: Any, store: Any, tool_state: dict[str, Any],
):
    """Two successful propose_correction calls + non-correction tool
    entries in one turn → exactly two pairs, nothing from the others.
    The agent-shaped successor of the proxy suite's S5 mixed-batch count
    pin.
    """
    cid_a = await _mk_crystal(store, customer.id, "cry_mix_a", "Value A is 1.")
    cid_b = await _mk_crystal(store, customer.id, "cry_mix_b", "Value B is 2.")

    input_a = {
        "crystal_id": cid_a,
        "proposed_change": "Value A is 10.",
        "rationale": "Source doc revised this morning.",
    }
    input_b = {
        "crystal_id": cid_b,
        "proposed_change": "Value B is 20.",
        "rationale": "Directly contradicted by the user's paste.",
    }
    out_a = await _call_tool(tool_state, customer.id, **input_a)
    out_b = await _call_tool(tool_state, customer.id, **input_b)
    assert out_a["proposed"] is True and out_b["proposed"] is True

    result = _agent_result([
        _entry(
            {"query": "value a"},
            {"results": [], "count": 0},
            tool_name="crystal_search",
        ),
        _entry(input_a, out_a),
        _entry(
            {"question": "q", "disposition": "researchable"},
            {"recorded": True, "gap_id": "gap_x"},
            tool_name="record_gap",
        ),
        _entry(input_b, out_b),
    ])
    out = await emit_mcr_artifacts(
        store=store,
        customer_id=customer.id,
        user_query="check the values",
        agent_result=result,
        sequence_id="seq_g2",
        turn_index=0,
        skip_self_critique=True,
    )

    assert len(out["correction_critique_ids"]) == 2
    assert len(out["correction_action_item_ids"]) == 2
    critiques = await store.list_critiques_for_sequence(
        customer_id=customer.id, sequence_id="seq_g2",
    )
    # Only the two correction critiques exist (Haiku skipped).
    assert len(critiques) == 2
    assert {c.observations[0]["anchors"][0]["crystal_id"] for c in critiques} \
        == {cid_a, cid_b}


# ===========================================================================
# Group 3 — Exit criterion 6: the payload feeds the alignment classifier
# ===========================================================================

@pytest.mark.asyncio
async def test_persisted_item_resolves_through_canonical_key_and_contradicts(
    customer: Any, store: Any, tool_state: dict[str, Any],
):
    """The assertion the proxy version could never make: the persisted
    ActionItem resolves through alignment._canonical_key, and two
    proposals for the same crystal with different changes classify as
    contradictory_action.
    """
    cid = await _mk_crystal(store, customer.id, "cry_align_1", "X is red.")
    input_1 = {
        "crystal_id": cid,
        "proposed_change": "X is blue.",
        "rationale": "The user said blue.",
    }
    input_2 = {
        "crystal_id": cid,
        "proposed_change": "X is green.",
        "rationale": "The linked spec says green.",
    }
    out_1 = await _call_tool(tool_state, customer.id, **input_1)
    out_2 = await _call_tool(tool_state, customer.id, **input_2)

    out = await emit_mcr_artifacts(
        store=store,
        customer_id=customer.id,
        user_query="what color is X",
        agent_result=_agent_result([
            _entry(input_1, out_1), _entry(input_2, out_2),
        ]),
        sequence_id="seq_g3",
        turn_index=0,
        skip_self_critique=True,
    )
    assert len(out["correction_critique_ids"]) == 2

    items = []
    for critique_id in out["correction_critique_ids"]:
        items.extend(await store.list_action_items_for_critique(critique_id))
    assert len(items) == 2

    # _canonical_key(edit_proposal) → content["crystal_id"], normalized.
    for item in items:
        assert _canonical_key(item) == cid.lower()

    # Same crystal, different proposed_change → contradictory_action —
    # the contradiction rule the proxy's payload could never trigger.
    assert classify_pair(items[0], items[1]) == "contradictory_action"
    assert classify_pair(items[1], items[0]) == "contradictory_action"


# ===========================================================================
# Group 4 — Q3 placement pin: skip_self_critique never suppresses the pair
# ===========================================================================

@pytest.mark.asyncio
async def test_skip_self_critique_still_persists_correction_pair(
    customer: Any, store: Any, tool_state: dict[str, Any],
):
    """Trace-yes, Haiku-no, corrections-yes: the walk runs before the
    skip_self_critique early-return because the pair is deterministic and
    costs zero model calls — cost control must not suppress an explicit
    self-correction.
    """
    cid = await _mk_crystal(store, customer.id, "cry_skip_1")
    tool_input = {
        "crystal_id": cid,
        "proposed_change": "Project deadline is April 1, 2027.",
        "rationale": "Confirmed in this conversation.",
    }
    output = await _call_tool(tool_state, customer.id, **tool_input)

    out = await emit_mcr_artifacts(
        store=store,
        customer_id=customer.id,
        user_query="deadline?",
        agent_result=_agent_result([_entry(tool_input, output)]),
        sequence_id="seq_g4",
        turn_index=0,
        skip_self_critique=True,
    )
    assert out["trace_id"] is not None
    assert out["critique_id"] is None  # Haiku lane skipped
    assert len(out["correction_critique_ids"]) == 1
    assert len(out["correction_action_item_ids"]) == 1
    critiques = await store.list_critiques_for_sequence(
        customer_id=customer.id, sequence_id="seq_g4",
    )
    assert len(critiques) == 1
    assert critiques[0].trace_id == out["trace_id"]


# ===========================================================================
# Group 5 — Validation refusals produce no MCR rows
# ===========================================================================

@pytest.mark.asyncio
async def test_validation_refusals_error_and_write_nothing(
    customer: Any, store: Any, tool_state: dict[str, Any],
):
    """Foreign crystal, missing crystal, empty proposed_change, empty
    rationale → error dict from the tool; and a turn whose log carries
    only refusals (plus an is_error entry) writes zero correction rows.
    """
    # A crystal owned by ANOTHER tenant — must read as not-found (no
    # existence oracle).
    other = await store.create_customer(
        provider="anthropic",
        model_id=AGENT_MODEL,
        api_key_ref="sk-test-other-tenant",
    )
    foreign_cid = await _mk_crystal(store, other.id, "cry_foreign_1")

    refusals: list[dict[str, Any]] = []

    out_foreign = await _call_tool(
        tool_state, customer.id,
        crystal_id=foreign_cid,
        proposed_change="new value",
        rationale="because",
    )
    assert out_foreign["proposed"] is False
    assert "not found in this tenant's bank" in out_foreign["error"]
    refusals.append((
        {"crystal_id": foreign_cid, "proposed_change": "new value",
         "rationale": "because"},
        out_foreign,
    ))

    out_missing = await _call_tool(
        tool_state, customer.id,
        crystal_id="cry_does_not_exist",
        proposed_change="new value",
        rationale="because",
    )
    assert out_missing["proposed"] is False
    assert "not found in this tenant's bank" in out_missing["error"]
    refusals.append((
        {"crystal_id": "cry_does_not_exist", "proposed_change": "new value",
         "rationale": "because"},
        out_missing,
    ))

    cid = await _mk_crystal(store, customer.id, "cry_valid_1")
    out_no_change = await _call_tool(
        tool_state, customer.id,
        crystal_id=cid, proposed_change="   ", rationale="because",
    )
    assert out_no_change["proposed"] is False
    assert "proposed_change" in out_no_change["error"]
    refusals.append((
        {"crystal_id": cid, "proposed_change": "   ", "rationale": "because"},
        out_no_change,
    ))

    out_no_rationale = await _call_tool(
        tool_state, customer.id,
        crystal_id=cid, proposed_change="new value", rationale="",
    )
    assert out_no_rationale["proposed"] is False
    assert "rationale" in out_no_rationale["error"]
    refusals.append((
        {"crystal_id": cid, "proposed_change": "new value", "rationale": ""},
        out_no_rationale,
    ))

    entries = [_entry(inp, outp) for inp, outp in refusals]
    # A dispatch-level error entry (string output, is_error=True) too.
    entries.append(_entry(
        {"crystal_id": cid}, "boom: dispatch failed", is_error=True,
    ))

    out = await emit_mcr_artifacts(
        store=store,
        customer_id=customer.id,
        user_query="try corrections",
        agent_result=_agent_result(entries),
        sequence_id="seq_g5",
        turn_index=0,
        skip_self_critique=True,
    )
    assert out["trace_id"] is not None
    assert out["correction_critique_ids"] == []
    assert out["correction_action_item_ids"] == []
    critiques = await store.list_critiques_for_sequence(
        customer_id=customer.id, sequence_id="seq_g5",
    )
    assert critiques == []


# ===========================================================================
# Group 6 — Q1 rider: trace-persist failure loses the pair, no fallback
# ===========================================================================

class _ExplodingTraceStore:
    """Delegates everything; create_reasoning_trace raises. The Q4-cited
    exploding-store proof pattern (the (d) suite's precedent): the failure
    fires before ANY correction write, and nothing lands trace_id=None.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def create_reasoning_trace(self, **kwargs: Any):
        raise RuntimeError("trace store exploded")


@pytest.mark.asyncio
async def test_trace_failure_loses_pair_without_orphan_fallback(
    customer: Any, store: Any, tool_state: dict[str, Any],
):
    cid = await _mk_crystal(store, customer.id, "cry_boom_1")
    tool_input = {
        "crystal_id": cid,
        "proposed_change": "new value",
        "rationale": "because the source changed",
    }
    output = await _call_tool(tool_state, customer.id, **tool_input)
    assert output["proposed"] is True

    out = await emit_mcr_artifacts(
        store=_ExplodingTraceStore(store),
        customer_id=customer.id,
        user_query="correct it",
        agent_result=_agent_result([_entry(tool_input, output)]),
        sequence_id="seq_g6",
        turn_index=0,
        skip_self_critique=True,
    )
    assert out["trace_id"] is None
    assert out["correction_critique_ids"] == []
    assert out["correction_action_item_ids"] == []
    # No trace_id=None fallback rows — the store holds NOTHING for the
    # sequence. (Queried on the real store; the wrapper delegated reads.)
    critiques = await store.list_critiques_for_sequence(
        customer_id=customer.id, sequence_id="seq_g6",
    )
    assert critiques == []


# ===========================================================================
# Group 7 — Return-shape honesty: the two critic lanes stay unconflated
# ===========================================================================

@pytest.mark.asyncio
async def test_correction_ids_distinct_from_haiku_critique_id(
    customer: Any, store: Any, tool_state: dict[str, Any],
):
    cid = await _mk_crystal(store, customer.id, "cry_lane_1")
    tool_input = {
        "crystal_id": cid,
        "proposed_change": "new value",
        "rationale": "the stored one is stale",
    }
    output = await _call_tool(tool_state, customer.id, **tool_input)

    result = _agent_result([_entry(tool_input, output)])

    # Under skip: the Haiku lane yields None while the correction lane
    # still carries ids.
    out_skip = await emit_mcr_artifacts(
        store=store,
        customer_id=customer.id,
        user_query="q",
        agent_result=result,
        sequence_id="seq_g7a",
        turn_index=0,
        skip_self_critique=True,
    )
    assert out_skip["critique_id"] is None
    assert len(out_skip["correction_critique_ids"]) == 1

    # With the Haiku lane live: both exist and never share an id.
    out_full = await emit_mcr_artifacts(
        store=store,
        customer_id=customer.id,
        user_query="q",
        agent_result=result,
        anthropic_client=_CleanCritiqueClient(),
        sequence_id="seq_g7b",
        turn_index=0,
    )
    assert out_full["critique_id"] is not None
    assert len(out_full["correction_critique_ids"]) == 1
    assert out_full["critique_id"] not in out_full["correction_critique_ids"]
