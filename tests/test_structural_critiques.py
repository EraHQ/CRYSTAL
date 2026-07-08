"""S6 — structural critics + the shadow dollar cap (2026-07-08).

The substrate channel reaches every part of the system that affects
outcomes (Anthony, 2026-07-08). This suite covers the first structural
writer — the blob-fact ingestion detector — and the spend_budgets gate
on the shadow critic (MCR §11 Q10's hard cap, closed by S4's substrate).
"""
from __future__ import annotations

from crystal_cache.infrastructure.schema import CrystalRow, FactRow
from crystal_cache.metacognition.structural import (
    run_structural_ingestion_scan,
)


async def _seed_crystal(store, customer_id, cid, claim, n_facts=1):
    async with store.session() as s:
        s.add(CrystalRow(
            id=cid, customer_id=customer_id,
            crystal_type="customer:legacy", summary_vector=[],
            quality_tier="neutral",
        ))
    for i in range(n_facts):
        async with store.session() as s:
            s.add(FactRow(
                id=f"{cid}_f{i}", crystal_id=cid,
                pair_type="entity_attribute",
                prompt_text=f"Company|Era HQ|Overview{i}|Test",
                claim_text=claim,
                source_kind="model_reasoning", vector=[],
            ))


async def test_blob_fact_files_a_structural_critique(store, customer):
    """The erahq incident, correctly channeled: a single-fact crystal
    with a jumbo claim becomes a substrate_observation blaming
    INGESTION — not a knowledge gap addressed to the human."""
    await _seed_crystal(store, customer.id, "crys_blob", "x" * 1200)
    await _seed_crystal(store, customer.id, "crys_ok", "Raleigh, NC")
    await _seed_crystal(  # multi-fact crystals are extraction-shaped: fine
        store, customer.id, "crys_multi", "y" * 1200, n_facts=3)

    out = await run_structural_ingestion_scan(store=store)
    assert out["found"] == 1 and out["filed"] == 1

    items = await store.list_substrate_action_items(customer_id=customer.id)
    assert len(items) == 1
    content = items[0].content
    assert content["subsystem"] == "ingestion"
    assert content["crystal_id"] == "crys_blob"
    assert content["claim_chars"] == 1200
    critique = await store.get_critique(items[0].critique_id)
    assert critique.critic_role == "structural"
    assert critique.critic_model == "store-signal"

    # No gap rows were created — the judgment lives in the critique
    # channel now (S1 stands).
    assert await store.count_knowledge_gaps(customer.id, status="open") == 0


async def test_structural_scan_is_idempotent(store, customer):
    await _seed_crystal(store, customer.id, "crys_blob2", "z" * 1000)

    first = await run_structural_ingestion_scan(store=store)
    second = await run_structural_ingestion_scan(store=store)

    assert first["filed"] == 1
    assert second["filed"] == 0 and second["skipped_existing"] == 1
    items = await store.list_substrate_action_items(customer_id=customer.id)
    assert len(items) == 1


async def test_shadow_budget_row_gates_by_dollars(store, customer):
    """No row = allowed (live behavior preserved). A zero-cap row = the
    tenant turned the shadow off. A funded row meters the
    origin='shadow_critic' ledger."""
    from crystal_cache.control.admission import function_budget_allows

    # No row: the worker's gate treats budget=None as allowed.
    assert await store.get_spend_budget(
        customer.id, function="shadow_critic") is None

    # Zero-cap row: off.
    await store.upsert_spend_budget(
        customer.id, function="shadow_critic", cap_micro_usd=0)
    assert await function_budget_allows(
        store, customer, "shadow_critic", origin="shadow_critic") is False

    # Funded row: on until the shadow-stamped ledger reaches the cap.
    await store.upsert_spend_budget(
        customer.id, function="shadow_critic", cap_micro_usd=50)
    assert await function_budget_allows(
        store, customer, "shadow_critic", origin="shadow_critic") is True
    rec = await store.record_llm_call(
        customer.id, model="claude-opus-4-8",
        input_tokens=0, output_tokens=0, origin="shadow_critic",
        price_table={},
    )
    from crystal_cache.infrastructure.schema import LlmCallRow
    async with store.session() as session:
        row = await session.get(LlmCallRow, rec["id"])
        row.computed_cost_micro_usd = 50
    assert await function_budget_allows(
        store, customer, "shadow_critic", origin="shadow_critic") is False
