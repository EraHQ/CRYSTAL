"""Phase 9A smoke tests for MCR trace + self-critique emission.

Per P0.47: six tests covering the agent-endpoint trace+critique
emission path. The tests call `emit_mcr_artifacts(...)` directly
rather than going through the endpoint; the endpoint composition is
straightforward dispatch and lives in Phase 9B/9C work.

Test scope (P0.47):
  T1: trace round-trip after a no-tool-call agent response
  T2: trace round-trip after a tool-using agent response
      (knowledge_search returns facts → trace.crystals_used populated)
  T3: self-critique with mocked Haiku → Critique row written with
      parsed observations
  T4: self-critique LLM call fails → empty critique still written +
      warning logged (no exception raised)
  T5: self-critique with action_items → ActionItem rows written and
      FK-linked
  T6: trace.tool_calls reflects the full tool_calls_log shape

Phase 8 / 8.5 test files are NOT touched.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from crystal_cache.agent.mcr_emitter import emit_mcr_artifacts


# ---------------------------------------------------------------------------
# Helpers — build agent_result dicts shaped like Agent.run() returns
# ---------------------------------------------------------------------------

def _make_agent_result(
    *,
    final_text: str = "Here is your answer.",
    tool_calls: list[dict[str, Any]] | None = None,
    stop_reason: str = "end_turn",
    iterations: int = 1,
) -> dict[str, Any]:
    """Build a result dict shaped like Agent.run() returns.

    The fields the emitter actually reads: final_text, tool_calls,
    stop_reason. Other fields are included for shape-completeness
    so accidental dependencies surface immediately.
    """
    return {
        "id": "chatcmpl-agent-test",
        "model": "claude-sonnet-4-5-20250929",
        "messages": [],
        "final_text": final_text,
        "stop_reason": stop_reason,
        "iterations": iterations,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "tool_calls": tool_calls or [],
    }


def _empty_critique_json() -> str:
    """Render a self-critique JSON response with no observations.

    Matches the Phase 9A self-critique prompt's expected shape per
    P0.45 — used by tests that focus on trace persistence rather
    than critique content.
    """
    return json.dumps({
        "observations": [],
        "action_items": [],
        "summary_text": "Agent reasoning was clean.",
    })


# ===========================================================================
# Test T1 — Trace round-trip after a no-tool-call agent response
# ===========================================================================

@pytest.mark.asyncio
async def test_trace_round_trip_no_tool_calls(
    customer: Any,
    store: Any,
    fake_anthropic: Any,
):
    """A trace persists when the agent answered without tool calls.

    Verifies the deterministic-extraction path (P0.46) handles the
    empty-tool-call case: events contains only the trailing
    final_text event; the five aggregate columns are all empty
    lists; the trace round-trips via get_reasoning_trace.
    """
    fake_anthropic.script_text(_empty_critique_json())

    agent_result = _make_agent_result(
        final_text="The deadline is April 1, 2027.",
        tool_calls=[],
    )

    result = await emit_mcr_artifacts(
        store=store,
        customer_id=customer.id,
        user_query="When is the deadline?",
        agent_result=agent_result,
        anthropic_client=fake_anthropic,
        sequence_id="seq_t1",
        turn_index=0,
    )

    assert result["trace_id"] is not None
    assert result["critique_id"] is not None
    assert result["action_item_ids"] == []

    # Round-trip the trace.
    trace = await store.get_reasoning_trace(result["trace_id"])
    assert trace is not None
    assert trace.customer_id == customer.id
    assert trace.sequence_id == "seq_t1"
    assert trace.turn_index == 0
    assert trace.crystals_used == []
    assert trace.tool_calls == []
    assert trace.inferences == []
    assert trace.borders_crossed == []
    assert trace.gaps_felt == []
    # events list contains exactly the final_text trailing event.
    assert len(trace.events) == 1
    assert trace.events[0]["type"] == "final_text"
    assert trace.events[0]["text"] == "The deadline is April 1, 2027."
    assert trace.events[0]["stop_reason"] == "end_turn"


# ===========================================================================
# Test T2 — Trace after a tool-using agent response
# ===========================================================================

@pytest.mark.asyncio
async def test_trace_round_trip_with_retrieval_tool_calls(
    customer: Any,
    store: Any,
    fake_anthropic: Any,
):
    """A trace's crystals_used reflects facts returned by retrieval tools.

    Verifies _extract_crystals_used pulls fact_ids deterministically
    from knowledge_search / content_search outputs. Multiple
    retrieval calls dedupe their fact_ids in the trace.
    """
    fake_anthropic.script_text(_empty_critique_json())

    tool_calls = [
        {
            "iteration": 1,
            "tool_name": "knowledge_search",
            "tool_use_id": "tu_001",
            "input": {"query": "deadline"},
            "output": {
                "matched_fact_ids": ["fact_alpha", "fact_beta"],
                "injection_text": "...",
            },
            "is_error": False,
        },
        {
            "iteration": 2,
            "tool_name": "content_search",
            "tool_use_id": "tu_002",
            "input": {"query": "April"},
            "output": {
                # Overlap with knowledge_search — should dedupe.
                "matched_fact_ids": ["fact_beta", "fact_gamma"],
                "injection_text": "...",
            },
            "is_error": False,
        },
        {
            "iteration": 3,
            "tool_name": "llm_invoke",
            "tool_use_id": "tu_003",
            "input": {"prompt": "Summarize"},
            "output": {"text": "Summary"},
            "is_error": False,
        },
    ]
    agent_result = _make_agent_result(
        final_text="Based on the bank: deadline is April 1, 2027.",
        tool_calls=tool_calls,
        iterations=4,
    )

    result = await emit_mcr_artifacts(
        store=store,
        customer_id=customer.id,
        user_query="Tell me about the April deadline.",
        agent_result=agent_result,
        anthropic_client=fake_anthropic,
        sequence_id="seq_t2",
        turn_index=1,
    )

    assert result["trace_id"] is not None

    trace = await store.get_reasoning_trace(result["trace_id"])
    assert trace is not None

    # crystals_used: deduped union from knowledge_search +
    # content_search. llm_invoke is not in _RETRIEVAL_TOOL_NAMES so
    # its output is ignored.
    assert trace.crystals_used == ["fact_alpha", "fact_beta", "fact_gamma"]
    # tool_calls preserves the full log (all three entries).
    assert len(trace.tool_calls) == 3
    assert [tc["tool_name"] for tc in trace.tool_calls] == [
        "knowledge_search", "content_search", "llm_invoke",
    ]


# ===========================================================================
# Test T3 — Self-critique with parsed observations
# ===========================================================================

@pytest.mark.asyncio
async def test_self_critique_parses_observations(
    customer: Any,
    store: Any,
    fake_anthropic: Any,
):
    """When Haiku returns a valid JSON critique, observations land
    on the Critique row with their type / text / confidence intact.
    """
    critique_json = json.dumps({
        "observations": [
            {
                "type": "gap_papered_over",
                "text": "Agent stated the deadline without citing a source.",
                "confidence": 0.85,
                "anchors": [],
            },
            {
                "type": "border_crossing_unflagged",
                "text": "Inferred 'this year' without checking.",
                "confidence": 0.6,
                "anchors": [],
            },
        ],
        "action_items": [],
        "summary_text": "Agent should have flagged its uncertainty.",
    })
    fake_anthropic.script_text(critique_json)

    agent_result = _make_agent_result(
        final_text="The deadline is April 1.",
        tool_calls=[],
    )

    result = await emit_mcr_artifacts(
        store=store,
        customer_id=customer.id,
        user_query="When is the deadline?",
        agent_result=agent_result,
        anthropic_client=fake_anthropic,
        sequence_id="seq_t3",
        turn_index=0,
    )

    assert result["critique_id"] is not None

    # Look up the critique via list_critiques_for_trace.
    critiques = await store.list_critiques_for_trace(result["trace_id"])
    assert len(critiques) == 1
    critique = critiques[0]
    assert critique.critic_role == "agent_self"
    # The model id should match settings.reflection_model.
    assert critique.critic_model == "claude-haiku-4-5-20251001"
    assert critique.summary_text == "Agent should have flagged its uncertainty."
    assert len(critique.observations) == 2
    assert critique.observations[0]["type"] == "gap_papered_over"
    assert critique.observations[0]["confidence"] == 0.85
    assert critique.observations[1]["type"] == "border_crossing_unflagged"
    assert critique.total_action_items == 0


# ===========================================================================
# Test T4 — Self-critique LLM failure leaves trace intact
# ===========================================================================

@pytest.mark.asyncio
async def test_self_critique_call_failure_persists_empty_critique(
    customer: Any,
    store: Any,
):
    """When the self-critique LLM call raises, the emitter logs and
    persists an empty critique with the failure noted in summary_text.

    The agent's response is unaffected — emit_mcr_artifacts NEVER
    raises per P0.44.
    """

    class _RaisingClient:
        """Seam-shaped client whose complete raises."""
        def complete(self, **kwargs: Any) -> str:
            raise RuntimeError("simulated transport error")

    bad_client = _RaisingClient()

    agent_result = _make_agent_result(
        final_text="Some answer.",
        tool_calls=[],
    )

    # Must NOT raise.
    result = await emit_mcr_artifacts(
        store=store,
        customer_id=customer.id,
        user_query="Some question.",
        agent_result=agent_result,
        anthropic_client=bad_client,
        sequence_id="seq_t4",
        turn_index=0,
    )

    # Trace persists.
    assert result["trace_id"] is not None
    trace = await store.get_reasoning_trace(result["trace_id"])
    assert trace is not None

    # Critique persists too (the failure is captured in summary_text).
    assert result["critique_id"] is not None
    critiques = await store.list_critiques_for_trace(result["trace_id"])
    assert len(critiques) == 1
    critique = critiques[0]
    # Observations are empty when the call failed.
    assert critique.observations == []
    # The failure is captured in the summary so it's queryable.
    assert critique.summary_text is not None
    assert "self-critique call failed" in critique.summary_text
    assert "RuntimeError" in critique.summary_text
    # No action items.
    assert critique.total_action_items == 0
    assert result["action_item_ids"] == []


# ===========================================================================
# Test T5 — Self-critique with action_items → ActionItem rows FK-linked
# ===========================================================================

@pytest.mark.asyncio
async def test_self_critique_action_items_persisted_and_fk_linked(
    customer: Any,
    store: Any,
    fake_anthropic: Any,
):
    """When the self-critique returns action_items, each one becomes
    an ActionItem row linked to the parent Critique via critique_id.
    The Critique's total_action_items count matches the number of
    items actually written.
    """
    critique_json = json.dumps({
        "observations": [
            {
                "type": "gap_papered_over",
                "text": "Agent gave a vague answer.",
                "confidence": 0.7,
                "anchors": [],
            }
        ],
        "action_items": [
            {
                "action_type": "research_task",
                "content": {
                    "topic": "the project's actual deadline policy",
                    "scope": "narrow",
                    "why_needed": "agent guessed",
                },
                "critic_confidence": 0.8,
            },
            {
                "action_type": "verification_task",
                "content": {
                    "crystal_id": "cry_deadline",
                    "claim_to_verify": "deadline is April 1",
                },
                "critic_confidence": 0.6,
            },
        ],
        "summary_text": "Two follow-ups proposed.",
    })
    fake_anthropic.script_text(critique_json)

    agent_result = _make_agent_result(
        final_text="The deadline is around April.",
        tool_calls=[],
    )

    result = await emit_mcr_artifacts(
        store=store,
        customer_id=customer.id,
        user_query="When is the project deadline?",
        agent_result=agent_result,
        anthropic_client=fake_anthropic,
        sequence_id="seq_t5",
        turn_index=0,
    )

    assert result["trace_id"] is not None
    assert result["critique_id"] is not None
    assert len(result["action_item_ids"]) == 2

    # Verify the critique's denormalized count matches.
    critiques = await store.list_critiques_for_trace(result["trace_id"])
    assert len(critiques) == 1
    critique = critiques[0]
    assert critique.total_action_items == 2

    # Look up the action items via the FK lookup.
    items = await store.list_action_items_for_critique(critique.id)
    assert len(items) == 2

    types = sorted(it.action_type for it in items)
    assert types == ["research_task", "verification_task"]

    # Each item carries the correct content payload.
    research = next(it for it in items if it.action_type == "research_task")
    assert research.content["topic"] == "the project's actual deadline policy"
    assert research.content["scope"] == "narrow"
    assert research.critic_confidence == 0.8
    assert research.critique_id == critique.id
    assert research.customer_id == customer.id
    assert research.status == "pending"

    verify = next(it for it in items if it.action_type == "verification_task")
    assert verify.content["crystal_id"] == "cry_deadline"
    assert verify.content["claim_to_verify"] == "deadline is April 1"
    assert verify.critic_confidence == 0.6


# ===========================================================================
# Test T6 — Full tool_calls_log + gaps_felt extraction
# ===========================================================================

@pytest.mark.asyncio
async def test_trace_tool_calls_full_shape_and_gaps_felt(
    customer: Any,
    store: Any,
    fake_anthropic: Any,
):
    """trace.tool_calls preserves the full log entry shape and
    trace.gaps_felt extracts the right fields from crystal_push_gap
    tool calls.

    Covers two pieces P0.46 promises:
      1. tool_calls is a faithful mirror of agent_result["tool_calls"]
         (every key + every value preserved).
      2. gaps_felt entries are {want, why_needed} dicts populated
         from crystal_push_gap input args (per the
         _extract_gaps_felt mapping).
    """
    fake_anthropic.script_text(_empty_critique_json())

    tool_calls = [
        {
            "iteration": 1,
            "tool_name": "knowledge_search",
            "tool_use_id": "tu_a",
            "input": {"query": "the obscure thing"},
            "output": {"matched_fact_ids": [], "injection_text": ""},
            "is_error": False,
        },
        {
            "iteration": 2,
            "tool_name": "crystal_push_gap",
            "tool_use_id": "tu_b",
            "input": {
                "domain": "engineering",
                "subject": "build pipeline",
                "missing": "which CI runner is used",
            },
            "output": "Gap recorded: which CI runner is used",
            "is_error": False,
        },
        {
            "iteration": 3,
            "tool_name": "crystal_push_gap",
            "tool_use_id": "tu_c",
            "input": {
                "domain": "engineering",
                "subject": "deployment",
                "missing": "what host is prod on",
            },
            "output": "Gap recorded: what host is prod on",
            "is_error": False,
        },
        {
            "iteration": 4,
            "tool_name": "crystal_push_gap",
            "tool_use_id": "tu_d",
            "input": {
                # Errored gap call should be filtered out.
                "domain": "x",
                "subject": "y",
                "missing": "z",
            },
            "output": "Error: something",
            "is_error": True,
        },
    ]
    agent_result = _make_agent_result(
        final_text="I don't have that information.",
        tool_calls=tool_calls,
        iterations=5,
    )

    result = await emit_mcr_artifacts(
        store=store,
        customer_id=customer.id,
        user_query="Tell me about the build pipeline.",
        agent_result=agent_result,
        anthropic_client=fake_anthropic,
        sequence_id="seq_t6",
        turn_index=0,
    )

    trace = await store.get_reasoning_trace(result["trace_id"])
    assert trace is not None

    # tool_calls: faithful mirror — 4 entries with all keys
    # preserved.
    assert len(trace.tool_calls) == 4
    assert trace.tool_calls[0] == tool_calls[0]
    assert trace.tool_calls[1] == tool_calls[1]
    assert trace.tool_calls[2] == tool_calls[2]
    assert trace.tool_calls[3] == tool_calls[3]

    # gaps_felt: only the two NON-errored crystal_push_gap calls
    # contribute. Mapping per _extract_gaps_felt:
    #   want = args["missing"]
    #   why_needed = args["subject"]
    assert len(trace.gaps_felt) == 2
    assert trace.gaps_felt[0] == {
        "want": "which CI runner is used",
        "why_needed": "build pipeline",
    }
    assert trace.gaps_felt[1] == {
        "want": "what host is prod on",
        "why_needed": "deployment",
    }

    # crystals_used stays empty: knowledge_search returned no
    # fact_ids; the other tools don't contribute.
    assert trace.crystals_used == []

    # events list mirrors tool_calls plus trailing final_text.
    assert len(trace.events) == 5  # 4 tool_calls + 1 final_text
    assert trace.events[-1]["type"] == "final_text"
    assert trace.events[-1]["text"] == "I don't have that information."
