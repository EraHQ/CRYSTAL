"""CONF-R (2026-07-23): conflict-as-retrieval-signal — the read path
from the idle machinery to answer time.

Motivating incident: the bank held BOTH launch dates and an open
conflict card pairing them, yet the agent retrieved only the stale
date and reasoned confidently on half of a known disagreement. These
pin the plumbing that prevents that: open conflicts travel with
retrieval, carrying the other side's claim.
"""

from __future__ import annotations

import pytest

from crystal_cache.agent.tools.retrievers import _apply_tier_signal
from crystal_cache.retrieval.tier_signal import conflict_note


async def _seed_conflict(store, customer_id, *, fact_a="fact_a1",
                         fact_b="fact_b1", claim_a="The launch is Sep 15",
                         claim_b="The launch is Sep 22", pair_key="pk_1"):
    return await store.create_knowledge_conflict(
        customer_id,
        fact_a_id=fact_a, fact_b_id=fact_b,
        claim_a=claim_a, claim_b=claim_b,
        pair_key=pair_key,
    )


@pytest.mark.asyncio
async def test_open_conflicts_for_facts_maps_both_directions(store, customer):
    await _seed_conflict(store, customer.id)

    # Asking from side A yields B's claim; from side B yields A's.
    from_a = await store.open_conflicts_for_facts(customer.id, ["fact_a1"])
    assert from_a["fact_a1"][0]["counterpart_claim"] == "The launch is Sep 22"
    from_b = await store.open_conflicts_for_facts(customer.id, ["fact_b1"])
    assert from_b["fact_b1"][0]["counterpart_claim"] == "The launch is Sep 15"

    # Unrelated ids and empty input stay silent.
    assert await store.open_conflicts_for_facts(customer.id, ["fact_zz"]) == {}
    assert await store.open_conflicts_for_facts(customer.id, []) == {}


@pytest.mark.asyncio
async def test_resolved_conflicts_stop_travelling(store, customer):
    from datetime import datetime, timezone

    c = await _seed_conflict(store, customer.id)
    assert await store.open_conflicts_for_facts(customer.id, ["fact_a1"])

    await store.apply_conflict_resolution(
        c.id, resolution="dismissed",
        resolved_at=datetime.now(timezone.utc),
    )
    # Resolution clears the warning — the arc completes.
    assert await store.open_conflicts_for_facts(customer.id, ["fact_a1"]) == {}


def test_conflict_note_rendering():
    assert conflict_note({}) is None

    single = conflict_note({
        "fact_1": [{"counterpart_claim": "The launch is September 22",
                    "detector": "contradiction_scan"}],
    })
    assert "CONTESTED: 1 retrieved fact is party" in single
    assert "September 22" in single
    assert "ask the user to confirm" in single

    long = conflict_note({
        "f": [{"counterpart_claim": "x" * 400, "detector": "d"}],
    })
    assert "\u2026" in long  # 240-char truncation

    many = conflict_note({
        f"f{i}": [{"counterpart_claim": f"claim {i}", "detector": "d"}]
        for i in range(5)
    })
    assert "+2 more open conflict" in many  # cap at 3 shown


@pytest.mark.asyncio
async def test_apply_tier_signal_carries_conflict_note(store, customer):
    await _seed_conflict(store, customer.id)
    payload = await _apply_tier_signal(store, customer.id, {
        "matched_crystal_ids": [],
        "matched_fact_ids": ["fact_a1"],
    })
    assert payload["contested_facts"]["fact_a1"]
    assert "CONTESTED" in payload["conflict_note"]
    assert "Sep 22" in payload["conflict_note"]

    # Uncontested retrieval carries no note.
    clean = await _apply_tier_signal(store, customer.id, {
        "matched_crystal_ids": [],
        "matched_fact_ids": ["fact_other"],
    })
    assert clean["conflict_note"] is None
    assert clean["contested_facts"] == {}


@pytest.mark.asyncio
async def test_conflict_signal_failure_never_breaks_retrieval():
    class _BrokenStore:
        async def open_conflicts_for_facts(self, *a, **k):
            raise RuntimeError("db down")

    payload = await _apply_tier_signal(_BrokenStore(), "cus_x", {
        "matched_crystal_ids": [],
        "matched_fact_ids": ["fact_1"],
    })
    assert payload["conflict_note"] is None
    assert payload["contested_facts"] == {}
