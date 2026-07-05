"""Topic seeding (BACKLOG §3 remainder, 2026-07-02).

Thin crystals and the operator topic list write knowledge_gaps rows the
Phase-2 fill sweep consumes — no model calls, idempotent against open
gaps, flood-guarded by the open-gap cap.

R14 note: these assertions are verified by `pytest`; they describe
expected behavior and have not yet been run at authoring time.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from crystal_cache.infrastructure.schema import CrystalRow, FactRow
from crystal_cache.scan import run_topic_seeding
from crystal_cache.scan.topic_seeding import parse_topics

_T0 = datetime(2026, 7, 1, tzinfo=timezone.utc)


async def _seed_crystal_with_facts(
    store: Any, customer_id: str, cid: str, n_facts: int,
    *, key: str = "Doc|p1|Widgets|Test", tier: str = "neutral",
):
    async with store.session() as s:
        s.add(CrystalRow(
            id=cid, customer_id=customer_id,
            crystal_type="customer:legacy", summary_vector=[],
            quality_tier=tier,
        ))
    for i in range(n_facts):
        async with store.session() as s:
            s.add(FactRow(
                id=f"{cid}_f{i}", crystal_id=cid,
                pair_type="question_answer",
                prompt_text=key, claim_text=f"claim {i}",
                source_kind="model_reasoning", vector=[],
                created_at=_T0,
            ))


async def test_thin_crystal_seeds_a_gap(store, customer):
    await _seed_crystal_with_facts(store, customer.id, "c_thin", 1)

    result = await run_topic_seeding(store=store, customer_id=customer.id)

    assert result.seeded_thin == 1
    gaps = await store.list_knowledge_gaps(customer.id, status="open")
    assert len(gaps) == 1
    assert gaps[0].source == "thin_crystal_seed"
    assert gaps[0].subject == "Widgets"
    assert "thin" in gaps[0].missing


async def test_thick_crystal_does_not_seed(store, customer):
    await _seed_crystal_with_facts(store, customer.id, "c_thick", 5)

    result = await run_topic_seeding(store=store, customer_id=customer.id)

    assert result.seeded_thin == 0
    assert await store.count_knowledge_gaps(customer.id, status="open") == 0


async def test_blacklisted_thin_crystal_never_seeds(store, customer):
    await _seed_crystal_with_facts(
        store, customer.id, "c_bl", 1, tier="blacklist",
    )

    result = await run_topic_seeding(store=store, customer_id=customer.id)

    assert result.seeded_thin == 0


async def test_idempotent_against_open_gaps(store, customer):
    await _seed_crystal_with_facts(store, customer.id, "c_thin2", 1)

    first = await run_topic_seeding(store=store, customer_id=customer.id)
    second = await run_topic_seeding(store=store, customer_id=customer.id)

    assert first.seeded_thin == 1
    assert second.seeded_thin == 0
    assert second.skipped_existing == 1
    assert await store.count_knowledge_gaps(customer.id, status="open") == 1


async def test_operator_topics_seed_and_dedupe(store, customer):
    result = await run_topic_seeding(
        store=store, customer_id=customer.id,
        topics=["Kubernetes networking", "SQLite WAL mode"],
    )
    assert result.seeded_topics == 2

    again = await run_topic_seeding(
        store=store, customer_id=customer.id,
        topics=["kubernetes networking"],  # case-insensitive dedupe
    )
    assert again.seeded_topics == 0
    assert again.skipped_existing == 1

    gaps = await store.list_knowledge_gaps(customer.id, status="open")
    assert {g.source for g in gaps} == {"topic_spec"}
    assert len(gaps) == 2


async def test_flood_guard_skips_the_pass(store, customer):
    for i in range(3):
        await store.create_knowledge_gap(
            customer.id, domain=None, subject=f"s{i}",
            missing="pre-existing", source="gap_discovery",
        )
    await _seed_crystal_with_facts(store, customer.id, "c_thin3", 1)

    result = await run_topic_seeding(
        store=store, customer_id=customer.id, open_gap_cap=3,
    )

    assert result.flood_guarded is True
    assert result.seeded_thin == 0
    assert await store.count_knowledge_gaps(customer.id, status="open") == 3


def test_parse_topics():
    assert parse_topics("") == []
    assert parse_topics(" a, b ,, A ,c") == ["a", "b", "c"]
