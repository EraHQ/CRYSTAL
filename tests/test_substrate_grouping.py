"""Substrate review grouping — CU-30, was PRD-6 (2026-07-02).

`group_substrate_observations` rolls deferred substrate observations up by
subsystem implicated with frequency + severity histograms, most-frequent
first, missing fields under "(unspecified)".

R14 note: these assertions are verified by `pytest`; they describe
expected behavior and have not yet been run at authoring time.
"""
from __future__ import annotations

from typing import Any

from crystal_cache.metacognition import group_substrate_observations


async def _seed_observation(
    store: Any, customer_id: str, *, content: dict, defer: bool = True,
):
    trace = await store.create_reasoning_trace(
        customer_id=customer_id,
        sequence_id="seq_grp",
        turn_index=0,
        events=[],
    )
    critique = await store.create_critique(
        customer_id=customer_id,
        critic_role="agent_self",
        critic_model="haiku",
        trace_id=trace.id,
        summary_text="substrate complaint",
        observations=[
            {"type": "substrate_complaint", "text": "x",
             "confidence": 0.7, "anchors": []}
        ],
    )
    item = await store.create_action_item(
        critique_id=critique.id,
        customer_id=customer_id,
        action_type="substrate_observation",
        content=content,
    )
    if defer:
        await store.update_action_item_status(item.id, status="deferred")
    return item


async def test_groups_by_subsystem_ordered_by_frequency(store, customer):
    await _seed_observation(store, customer.id, content={
        "subsystem": "retrieval", "complaint": "old retrieval gripe",
        "severity": "low",
    })
    await _seed_observation(store, customer.id, content={
        "subsystem": "tools", "complaint": "shell output truncated",
        "severity": "medium",
    })
    await _seed_observation(store, customer.id, content={
        "subsystem": "retrieval", "complaint": "newest retrieval gripe",
        "severity": "high",
    })

    groups = await group_substrate_observations(store, customer_id=customer.id)

    assert [g.subsystem for g in groups] == ["retrieval", "tools"]
    retrieval = groups[0]
    assert retrieval.count == 2
    assert retrieval.severities == {"low": 1, "high": 1}
    assert retrieval.latest_complaint == "newest retrieval gripe"
    assert len(retrieval.item_ids) == 2
    assert groups[1].count == 1


async def test_missing_fields_roll_up_under_unspecified(store, customer):
    await _seed_observation(store, customer.id, content={
        "complaint": "no subsystem named",
    })

    groups = await group_substrate_observations(store, customer_id=customer.id)

    assert len(groups) == 1
    assert groups[0].subsystem == "(unspecified)"
    assert groups[0].severities == {"(unspecified)": 1}


async def test_pending_items_are_excluded(store, customer):
    """Grouping inherits the lister's deferred-only filter."""
    await _seed_observation(store, customer.id, defer=False, content={
        "subsystem": "retrieval", "complaint": "still pending",
    })

    groups = await group_substrate_observations(store, customer_id=customer.id)

    assert groups == []
