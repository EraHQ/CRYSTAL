"""Assumptions slice 4 — lifecycle mechanics (Q3=B, capture-at-delete).

Exercises parent-death invalidation end-to-end against the in-memory
store: delete_crystal's capture-at-delete (blacklist + audit tag +
recall_gated stays True, in the SAME transaction as the parent
delete), the primary-parent FK NULL, both-parents-die accumulating
both audit tags, the pre-existing parent_crystal_id FK repair for
NON-assumption children (spawn lineage), tenancy-scoped delete
refusal producing zero side effects, and sweep_orphaned_assumptions
covering out-of-band deaths (dangling edges — reachable on SQLite/
surgery; the chain FKs make it unreachable via SQL deletes on
Postgres) with idempotent reruns.

R14 note: verified by `pytest`; the same assertions ran green in the
container rig at authoring time (2026-08-05).
"""
from __future__ import annotations

from sqlalchemy import select

from crystal_cache.infrastructure.schema import (
    CrystalChainRow, CrystalRow, FactRow,
)


async def _seed_crystal(store, crystal_id, customer_id, *, tier="neutral"):
    async with store.session() as s:
        s.add(CrystalRow(
            id=crystal_id, customer_id=customer_id,
            crystal_type="customer:legacy", summary_vector=[],
            summary_text=f"summary of {crystal_id}",
            quality_tier=tier,
        ))


async def _make_assumption(store, customer, encoder,
                           a="cr_a", b="cr_b"):
    await _seed_crystal(store, a, customer.id)
    await _seed_crystal(store, b, customer.id)
    written = await store.create_assumption_crystal(
        customer.id,
        statement=f"bridge over {a}+{b}",
        subject="Bridged subject",
        parent_a_id=a,
        parent_b_id=b,
        confidence=0.8,
        encoder=encoder,
    )
    return written["crystal_id"]


async def _get_row(store, crystal_id):
    async with store.session() as s:
        return await s.get(CrystalRow, crystal_id)


async def _edges_from(store, crystal_id):
    async with store.session() as s:
        return list((await s.execute(
            select(CrystalChainRow).where(
                CrystalChainRow.source_crystal_id == crystal_id
            )
        )).scalars().all())


# ---------------------------------------------------------------------------
# Capture-at-delete
# ---------------------------------------------------------------------------

async def test_parent_delete_invalidates_assumption(
    store, customer, semantic_encoder_stub,
):
    asm_id = await _make_assumption(store, customer,
                                    semantic_encoder_stub)

    assert await store.delete_crystal("cr_b", customer.id) is True

    row = await _get_row(store, asm_id)
    assert row.quality_tier == "blacklist"
    assert "assumption_invalidated:parent:cr_b" in row.diagnostic_tags
    assert bool(row.recall_gated) is True            # stays gated
    assert row.parent_crystal_id == "cr_a"           # primary untouched
    # D2 removed only the edges touching the dead parent.
    remaining = await _edges_from(store, asm_id)
    assert [e.target_crystal_id for e in remaining] == ["cr_a"]


async def test_primary_parent_delete_nulls_fk_and_second_death_accumulates(
    store, customer, semantic_encoder_stub,
):
    asm_id = await _make_assumption(store, customer,
                                    semantic_encoder_stub)

    assert await store.delete_crystal("cr_a", customer.id) is True
    row = await _get_row(store, asm_id)
    assert row.quality_tier == "blacklist"
    assert row.parent_crystal_id is None             # FK cleared
    assert "assumption_invalidated:parent:cr_a" in row.diagnostic_tags

    # The second parent dying is a second, distinct audit entry.
    assert await store.delete_crystal("cr_b", customer.id) is True
    row = await _get_row(store, asm_id)
    assert "assumption_invalidated:parent:cr_a" in row.diagnostic_tags
    assert "assumption_invalidated:parent:cr_b" in row.diagnostic_tags
    assert await _edges_from(store, asm_id) == []


async def test_spawned_child_fk_repair_without_invalidation(
    store, customer, semantic_encoder_stub,
):
    """A NON-assumption child (spawn lineage) gets its dangling FK
    cleared and NOTHING else — no blacklist, no tag. The pre-existing
    Postgres FK hole, fixed for all children."""
    await _seed_crystal(store, "cr_parent", customer.id)
    async with store.session() as s:
        s.add(CrystalRow(
            id="cr_child", customer_id=customer.id,
            crystal_type="customer:legacy", summary_vector=[],
            summary_text="spawned sibling", quality_tier="neutral",
            parent_crystal_id="cr_parent", build_method="spawned",
        ))

    assert await store.delete_crystal("cr_parent", customer.id) is True

    child = await _get_row(store, "cr_child")
    assert child.parent_crystal_id is None
    assert child.quality_tier == "neutral"           # untouched
    assert list(child.diagnostic_tags or []) == []   # no audit tag


async def test_tenancy_mismatch_leaves_assumption_untouched(
    store, customer, semantic_encoder_stub,
):
    asm_id = await _make_assumption(store, customer,
                                    semantic_encoder_stub)
    foreign = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="sk-other",
    )

    assert await store.delete_crystal("cr_b", foreign.id) is False

    row = await _get_row(store, asm_id)
    assert row.quality_tier == "quarantine"          # no side effects
    assert list(row.diagnostic_tags or []) == [
        "assumption_confidence:0.80",
    ]
    assert len(await _edges_from(store, asm_id)) == 2


# ---------------------------------------------------------------------------
# Out-of-band sweep
# ---------------------------------------------------------------------------

async def _raw_delete_leaving_chains(store, crystal_id):
    """Out-of-band death: the crystal row and its facts vanish but the
    chain edges survive — unreachable via SQL deletes on Postgres (the
    chain FKs block it), reachable on SQLite/dev and via FK-disabled
    surgery."""
    async with store.session() as s:
        for f in (await s.execute(
            select(FactRow).where(FactRow.crystal_id == crystal_id)
        )).scalars().all():
            await s.delete(f)
        row = await s.get(CrystalRow, crystal_id)
        await s.delete(row)


async def test_sweep_invalidates_out_of_band_death(
    store, customer, semantic_encoder_stub,
):
    asm_id = await _make_assumption(store, customer,
                                    semantic_encoder_stub)
    await _raw_delete_leaving_chains(store, "cr_b")

    swept = await store.sweep_orphaned_assumptions(limit=50)
    assert swept == 1

    row = await _get_row(store, asm_id)
    assert row.quality_tier == "blacklist"
    assert "assumption_invalidated:parent:cr_b" in row.diagnostic_tags
    assert bool(row.recall_gated) is True
    # The dangling edge is gone; the live-parent edge remains.
    remaining = await _edges_from(store, asm_id)
    assert [e.target_crystal_id for e in remaining] == ["cr_a"]

    # Idempotent: the swept edge no longer exists.
    assert await store.sweep_orphaned_assumptions(limit=50) == 0


async def test_sweep_noop_on_healthy_bank(
    store, customer, semantic_encoder_stub,
):
    asm_id = await _make_assumption(store, customer,
                                    semantic_encoder_stub)

    assert await store.sweep_orphaned_assumptions(limit=50) == 0

    row = await _get_row(store, asm_id)
    assert row.quality_tier == "quarantine"          # untouched
    assert len(await _edges_from(store, asm_id)) == 2
