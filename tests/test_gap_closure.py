"""C2 — gap closure + the reopen witness (ratified 2026-08-08:
Q1=A close on approve, Q2=A reopen on invalidation/deletion, Q3=A
curation_events activity feed).

Store-level coverage of the three ratified behaviors: the guarded
close (`close_gap_for_approved_assumption`), both reopen paths inside
`delete_crystal` (capture-at-delete invalidation; curator delete of
the filling assumption), and the witness feed
(`record_curation_event` / `list_curation_events` + the emitters).
The approve endpoint's wiring adds one store call + best-effort emits
on top of `set_crystal_recall_gate`, which slice-5's admin tests
already pin.
"""
from __future__ import annotations


from crystal_cache.infrastructure.schema import CrystalRow


async def _seed_crystal(store, crystal_id, customer_id, *, summary=None):
    async with store.session() as s:
        s.add(CrystalRow(
            id=crystal_id, customer_id=customer_id,
            crystal_type="customer:legacy", summary_vector=[],
            summary_text=summary,
        ))


async def _seed_gap(store, customer_id, *, missing="Deploy risk unknown"):
    gap = await store.create_knowledge_gap(
        customer_id,
        domain=None,
        subject="Deploy risk windows",
        missing=missing,
    )
    return gap.id


async def _seed_gap_assumption(store, customer, encoder, gap_id):
    """Two parents + a gap-seeded assumption via the production path."""
    await _seed_crystal(store, "cr_a", customer.id, summary="Parent A")
    await _seed_crystal(store, "cr_b", customer.id, summary="Parent B")
    result = await store.create_assumption_crystal(
        customer.id,
        statement="Bridging inference for the gap",
        subject="Deploy risk windows",
        parent_a_id="cr_a",
        parent_b_id="cr_b",
        confidence=0.8,
        encoder=encoder,
        gap_id=gap_id,
    )
    return result["crystal_id"]


async def _gap(store, gap_id):
    return await store.get_knowledge_gap(gap_id)


# ---------------------------------------------------------------------------
# Q1=A — approval closes the seeding gap (guarded)
# ---------------------------------------------------------------------------

async def test_close_on_approve_fills_open_gap(
    store, customer, semantic_encoder_stub,
):
    gap_id = await _seed_gap(store, customer.id)
    asm_id = await _seed_gap_assumption(
        store, customer, semantic_encoder_stub, gap_id,
    )

    closed = await store.close_gap_for_approved_assumption(
        asm_id, customer.id,
    )
    assert closed == gap_id
    gap = await _gap(store, gap_id)
    assert gap.status == "filled"
    assert gap.filled_by_crystal_id == asm_id
    assert gap.resolved_at is not None


async def test_close_guards(
    store, customer, semantic_encoder_stub,
):
    gap_id = await _seed_gap(store, customer.id)
    asm_id = await _seed_gap_assumption(
        store, customer, semantic_encoder_stub, gap_id,
    )

    # Foreign tenant: no close.
    foreign = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="sk-other",
    )
    assert await store.close_gap_for_approved_assumption(
        asm_id, foreign.id,
    ) is None

    # Gap already filled by something else: never overwritten.
    from datetime import datetime, timezone
    await store.mark_knowledge_gap_filled(
        gap_id,
        filled_by_crystal_id="cr_other",
        resolved_at=datetime.now(timezone.utc),
    )
    assert await store.close_gap_for_approved_assumption(
        asm_id, customer.id,
    ) is None
    gap = await _gap(store, gap_id)
    assert gap.filled_by_crystal_id == "cr_other"

    # An assumption without gap provenance: no-op.
    no_gap_asm = await store.create_assumption_crystal(
        customer.id,
        statement="Chainless bridge",
        subject="Other subject",
        parent_a_id="cr_a",
        parent_b_id="cr_b",
        confidence=0.7,
        encoder=semantic_encoder_stub,
        gap_id=None,
    )
    assert await store.close_gap_for_approved_assumption(
        no_gap_asm["crystal_id"], customer.id,
    ) is None


# ---------------------------------------------------------------------------
# Q2=A — reopen on invalidation / deletion (guarded, witnessed)
# ---------------------------------------------------------------------------

async def test_parent_death_reopens_gap_and_witnesses(
    store, customer, semantic_encoder_stub,
):
    gap_id = await _seed_gap(store, customer.id)
    asm_id = await _seed_gap_assumption(
        store, customer, semantic_encoder_stub, gap_id,
    )
    assert await store.close_gap_for_approved_assumption(
        asm_id, customer.id,
    ) == gap_id

    # Parent dies -> capture-at-delete invalidates the assumption AND
    # reopens the gap it filled.
    assert await store.delete_crystal("cr_a", customer.id) is True

    gap = await _gap(store, gap_id)
    assert gap.status == "open"
    assert gap.filled_by_crystal_id is None
    assert gap.resolved_at is None

    events = await store.list_curation_events(customer.id)
    types = [e["event_type"] for e in events]
    assert "assumption_invalidated" in types
    assert "gap_reopened" in types
    reopen = next(e for e in events if e["event_type"] == "gap_reopened")
    assert reopen["subject_id"] == gap_id
    assert reopen["payload"]["was_filled_by"] == asm_id


async def test_deleting_filling_assumption_reopens_gap(
    store, customer, semantic_encoder_stub,
):
    gap_id = await _seed_gap(store, customer.id)
    asm_id = await _seed_gap_assumption(
        store, customer, semantic_encoder_stub, gap_id,
    )
    assert await store.close_gap_for_approved_assumption(
        asm_id, customer.id,
    ) == gap_id

    assert await store.delete_crystal(asm_id, customer.id) is True

    gap = await _gap(store, gap_id)
    assert gap.status == "open"
    assert gap.filled_by_crystal_id is None

    events = await store.list_curation_events(customer.id)
    types = [e["event_type"] for e in events]
    assert "assumption_deleted" in types
    assert "gap_reopened" in types


async def test_no_reopen_when_gap_refilled_by_other(
    store, customer, semantic_encoder_stub,
):
    """A gap since re-filled by something else is untouched by the
    death of the assumption that once filled it."""
    gap_id = await _seed_gap(store, customer.id)
    asm_id = await _seed_gap_assumption(
        store, customer, semantic_encoder_stub, gap_id,
    )
    assert await store.close_gap_for_approved_assumption(
        asm_id, customer.id,
    ) == gap_id

    # Re-fill by another crystal (manual operator resolution path).
    from datetime import datetime, timezone
    await store.mark_knowledge_gap_filled(
        gap_id,
        filled_by_crystal_id="cr_other",
        resolved_at=datetime.now(timezone.utc),
    )

    assert await store.delete_crystal(asm_id, customer.id) is True
    gap = await _gap(store, gap_id)
    assert gap.status == "filled"
    assert gap.filled_by_crystal_id == "cr_other"


# ---------------------------------------------------------------------------
# Q3=A — the witness feed itself
# ---------------------------------------------------------------------------

async def test_assumption_written_witness(
    store, customer, semantic_encoder_stub,
):
    gap_id = await _seed_gap(store, customer.id)
    asm_id = await _seed_gap_assumption(
        store, customer, semantic_encoder_stub, gap_id,
    )
    events = await store.list_curation_events(customer.id)
    written = [e for e in events if e["event_type"] == "assumption_written"]
    assert len(written) == 1
    assert written[0]["subject_id"] == asm_id
    assert written[0]["payload"]["gap_id"] == gap_id


async def test_feed_roundtrip_tenant_scoped_newest_first(
    store, customer,
):
    foreign = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="sk-other",
    )
    for i in range(3):
        await store.record_curation_event(
            customer.id,
            event_type="assumption_written",
            label=f"event {i}",
            subject_id=f"cr_{i}",
        )
    await store.record_curation_event(
        foreign.id, event_type="gap_filled", label="foreign",
    )

    events = await store.list_curation_events(customer.id)
    assert len(events) == 3
    assert all(e["event_type"] == "assumption_written" for e in events)
    # Newest first.
    assert [e["label"] for e in events] == ["event 2", "event 1", "event 0"]

    assert len(await store.list_curation_events(foreign.id)) == 1
