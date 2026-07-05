"""Tests for headline_facts_for_crystals (P3 bank readability).

The list route attaches each crystal's representative sparse key + claim
+ source_kind so the inspector renders a human breadcrumb/title and
classifies the crystal. The method returns the EARLIEST fact per crystal
(created_at asc = write order), selects only text columns (never the
768-dim vector), and omits crystals that have no facts.

In-memory store per test (conftest `store` fixture). asyncio_mode=auto,
so plain `async def`. Facts are inserted directly with controlled
created_at values so the earliest-wins ordering is deterministic (two
add_pair_to_crystal calls could land sub-millisecond apart).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from crystal_cache.infrastructure.schema import CrystalRow, FactRow

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


async def _seed(store) -> None:
    async with store.session() as s:
        s.add(CrystalRow(id="crys_a", customer_id="cus_1", crystal_type="reflection", summary_vector=[]))
        s.add(CrystalRow(id="crys_b", customer_id="cus_1", crystal_type="customer:legacy", summary_vector=[]))
        # crys_a has TWO facts; the earlier-created one must be the headline,
        # even though it is added to the session second (ordering is by
        # created_at, not insertion order).
        s.add(FactRow(
            id="f_a_late", crystal_id="crys_a", pair_type="question_answer",
            prompt_text="Reflections|Game|Later Lesson", claim_text="the later one",
            source_kind="model_reasoning", vector=[], created_at=_T0 + timedelta(minutes=5),
        ))
        s.add(FactRow(
            id="f_a_first", crystal_id="crys_a", pair_type="question_answer",
            prompt_text="Reflections|Game|First Lesson", claim_text="the earliest one",
            source_kind="model_reasoning", vector=[], created_at=_T0,
        ))
        # crys_b: a single content-chunk fact.
        s.add(FactRow(
            id="f_b", crystal_id="crys_b", pair_type="content_chunk",
            prompt_text="Code|game.js|update", claim_text="function update() {}",
            source_kind="document_chunk", vector=[], created_at=_T0 + timedelta(minutes=1),
        ))


async def test_headline_returns_earliest_fact_per_crystal(store):
    await _seed(store)
    out = await store.headline_facts_for_crystals(["crys_a", "crys_b"])

    assert out["crys_a"]["key"] == "Reflections|Game|First Lesson"  # earliest wins
    assert out["crys_a"]["claim"] == "the earliest one"
    assert out["crys_a"]["source_kind"] == "model_reasoning"

    assert out["crys_b"]["key"] == "Code|game.js|update"
    assert out["crys_b"]["source_kind"] == "document_chunk"


async def test_headline_omits_crystals_without_facts(store):
    await _seed(store)
    async with store.session() as s:
        s.add(CrystalRow(id="crys_empty", customer_id="cus_1", crystal_type="customer:legacy", summary_vector=[]))

    out = await store.headline_facts_for_crystals(["crys_a", "crys_empty", "crys_missing"])
    # A crystal with no facts (and an id that doesn't exist) is simply absent.
    assert set(out.keys()) == {"crys_a"}


async def test_headline_empty_input_returns_empty(store):
    assert await store.headline_facts_for_crystals([]) == {}
