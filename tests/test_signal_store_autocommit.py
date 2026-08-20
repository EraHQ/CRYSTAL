"""FIX 3 regression tests — the crystal_push_store auto-commit path.

The self-curation write path never worked: `_handle_store` referenced
`vector_index` without having it in scope (handle_signals accepted the
kwarg but never forwarded it), so every high-confidence push_store
raised NameError inside the per-signal try/except, silently dropping
the knowledge AND skipping the review-queue fallback. The old failure
surfaced (when anyone looked) as a tool_result reading:

    Error processing crystal_push_store: name 'vector_index' is not defined

Three pins:
  1. confidence >= 0.9 actually lands a Crystal + Fact via
     add_pair_for_customer (real in-memory store + the conftest
     semantic encoder stub).
  2. confidence in the medium band goes to the push review queue.
  3. Regression: with a vector_index kwarg passed to handle_signals,
     NO tool_result carries an error string mentioning vector_index.

Fixtures come from conftest: store (in-memory SQLite MetadataStore),
customer, semantic_encoder_stub, vector_store, vector_index.
"""
from __future__ import annotations

import json
from typing import Any

from crystal_cache.retrieval.v3_push_pull import ParsedSignals
from crystal_cache.retrieval.v3_signal_handler import handle_signals


def _store_call(
    key: str,
    value: str,
    confidence: float,
    tool_call_id: str = "tc_store",
) -> dict[str, Any]:
    """One crystal_push_store tool_call shaped like the upstream LLM emits."""
    return {
        "id": tool_call_id,
        "function": {
            "name": "crystal_push_store",
            "arguments": json.dumps(
                {"key": key, "value": value, "confidence": confidence}
            ),
        },
    }


def _signals(*tool_calls: dict[str, Any]) -> ParsedSignals:
    """Wrap raw tool_call dicts as ParsedSignals (same pattern as the
    Phase 9B signal-handler tests): Pass 1 re-parses raw_tool_calls,
    the per-type lists only need to make has_signals return True."""
    sig = ParsedSignals()
    sig.raw_tool_calls = list(tool_calls)
    sig.push_stores = [{} for _ in tool_calls]
    return sig


async def test_high_confidence_store_autocommits_a_fact(
    store, customer, semantic_encoder_stub, vector_store, vector_index,
):
    """confidence >= 0.9 must land a Crystal + Fact via
    add_pair_for_customer — the path that raised NameError before FIX 3."""
    signals = _signals(_store_call(
        "shipping|returns|window", "Returns accepted within 30 days", 0.95,
        tool_call_id="tc_hi",
    ))

    stats = await handle_signals(
        signals,
        customer_id=customer.id,
        store=store,
        encoder=semantic_encoder_stub,
        vector_store=vector_store,
        vector_index=vector_index,
    )

    assert stats["errors"] == 0
    assert stats["auto_committed"] == 1
    assert stats["queued_for_review"] == 0

    # The tool_result reports success, not an error.
    [result] = stats["tool_results"]
    assert result["tool_call_id"] == "tc_hi"
    assert result["content"].startswith("Stored:")

    # The fact actually landed in the customer's bank.
    crystals = await store.list_crystals_for_customer(customer.id)
    assert len(crystals) == 1
    facts = await store.list_facts_for_crystal(crystals[0].id)
    assert len(facts) == 1
    assert facts[0].claim_text == "Returns accepted within 30 days"

    # And nothing leaked into the review queue.
    assert await store.list_push_review_items(customer.id) == []


async def test_medium_confidence_store_goes_to_review_queue(
    store, customer, semantic_encoder_stub, vector_store, vector_index,
):
    """confidence in [0.5, 0.9) queues for human review, no crystal write."""
    signals = _signals(_store_call(
        "shipping|returns|window", "Maybe 45 days for members?", 0.7,
        tool_call_id="tc_med",
    ))

    stats = await handle_signals(
        signals,
        customer_id=customer.id,
        store=store,
        encoder=semantic_encoder_stub,
        vector_store=vector_store,
        vector_index=vector_index,
    )

    assert stats["errors"] == 0
    assert stats["auto_committed"] == 0
    assert stats["queued_for_review"] == 1

    [result] = stats["tool_results"]
    assert result["content"].startswith("Queued for review:")

    items = await store.list_push_review_items(customer.id)
    assert len(items) == 1
    assert items[0].status == "pending"
    assert items[0].value == "Maybe 45 days for members?"

    # No crystal was written for the medium band.
    assert await store.list_crystals_for_customer(customer.id) == []


async def test_vector_index_kwarg_produces_no_vector_index_error(
    store, customer, semantic_encoder_stub, vector_store, vector_index,
):
    """Regression pin for the NameError. Before FIX 3, this exact call
    shape (handle_signals with a vector_index kwarg + a >= 0.9 store)
    produced a tool_result containing:
        "Error processing crystal_push_store: name 'vector_index' is not defined"
    No tool_result may mention vector_index in any error string now."""
    signals = _signals(
        _store_call("a|b", "high-band value", 0.95, tool_call_id="tc_1"),
        _store_call("c|d", "medium-band value", 0.6, tool_call_id="tc_2"),
    )

    stats = await handle_signals(
        signals,
        customer_id=customer.id,
        store=store,
        encoder=semantic_encoder_stub,
        vector_store=vector_store,
        vector_index=vector_index,
    )

    assert stats["errors"] == 0
    for result in stats["tool_results"]:
        assert "vector_index" not in result["content"]
        assert not result["content"].startswith("Error processing")
