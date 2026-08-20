"""FIX 4 regression tests — the agent surface's feedback link.

`finalize_agent_turn` wrote query_logs rows WITHOUT turn_index, so
`find_query_log_by_sequence` (which matches sequence_id == X AND
turn_index == <int>) never resolved an agent turn: thumbs up/down
returned 200, learned nothing, logged nothing. FIX 4 persists
turn_index on the QueryLog write — the caller's explicit value when
given, otherwise derived via `store.next_turn_index(customer_id,
sequence_id)` (the same primitive the chat proxy uses).

Three pins:
  1. Explicit turn_index → the row carries it → the feedback lookup
     (store method called directly) finds the turn.
  2. turn_index=None + a sequence_id → an index is derived and
     persisted (not NULL).
  3. Characterization: two finalizes on one sequence get distinct,
     increasing indexes.

Fixtures from conftest: store (in-memory SQLite MetadataStore),
customer. Cost, citations, and MCR are exercised elsewhere
(test_turn_finalize) — these tests use a minimal Agent.run-shaped
result with no tool_calls and skip the self-critique call, keeping
the focus on the QueryLog write.
"""
from __future__ import annotations

from typing import Any

from crystal_cache.agent.turn_finalize import finalize_agent_turn


def _result(run_id: str = "run_fb") -> dict[str, Any]:
    """Minimal Agent.run()-shaped result: no tool_calls (no citations
    to ground), real token counts so the cost step is exercised."""
    return {
        "id": run_id,
        "model": "claude-sonnet-4-5-20250929",
        "final_text": "The shipping window is 30 days.",
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "tool_calls": [],
    }


async def _finalize(store, customer, *, sequence_id, turn_index, query="q?"):
    return await finalize_agent_turn(
        store=store,
        encoder=object(),  # no tool_calls → grounding never runs
        customer=customer,
        anthropic_client=None,
        result=_result(),
        user_query=query,
        sequence_id=sequence_id,
        turn_index=turn_index,
        skip_self_critique=True,
    )


async def test_explicit_turn_index_lands_and_feedback_lookup_finds_it(
    store, customer,
):
    """CRYS passes a real turn_index — the row must carry it, and the
    feedback endpoint's lookup (called directly) must resolve it."""
    await _finalize(
        store, customer, sequence_id="seq_explicit", turn_index=3,
        query="what is the shipping window?",
    )

    found = await store.find_query_log_by_sequence(
        customer_id=customer.id,
        sequence_id="seq_explicit",
        turn_index=3,
    )
    assert found is not None
    assert found.turn_index == 3
    assert found.query_text == "what is the shipping window?"
    assert found.injection_method == "agent_tools"


async def test_none_turn_index_is_derived_from_sequence(
    store, customer,
):
    """The stateless HTTP agent endpoint passes turn_index=None. With a
    sequence_id present, the index must be derived (first turn → 0) and
    persisted — the pre-FIX-4 behavior left it NULL forever."""
    await _finalize(
        store, customer, sequence_id="seq_derived", turn_index=None,
    )

    log = await store.get_last_query_log_for_sequence(
        customer.id, "seq_derived",
    )
    assert log is not None
    assert log.turn_index is not None
    assert log.turn_index == 0

    # And the feedback-side lookup resolves it.
    found = await store.find_query_log_by_sequence(
        customer_id=customer.id,
        sequence_id="seq_derived",
        turn_index=0,
    )
    assert found is not None


async def test_two_finalizes_on_one_sequence_get_increasing_indexes(
    store, customer,
):
    """Characterization: successive turns on the same sequence derive
    distinct, increasing indexes (next_turn_index counts prior rows)."""
    await _finalize(
        store, customer, sequence_id="seq_multi", turn_index=None,
        query="first turn",
    )
    await _finalize(
        store, customer, sequence_id="seq_multi", turn_index=None,
        query="second turn",
    )

    first = await store.find_query_log_by_sequence(
        customer_id=customer.id, sequence_id="seq_multi", turn_index=0,
    )
    second = await store.find_query_log_by_sequence(
        customer_id=customer.id, sequence_id="seq_multi", turn_index=1,
    )
    assert first is not None and first.query_text == "first turn"
    assert second is not None and second.query_text == "second turn"
    assert first.turn_index < second.turn_index
