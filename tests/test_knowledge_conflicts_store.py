"""Never-Idle Convergence — knowledge_conflicts store surface
(ConflictExtensionsMixin).

The contradiction-scan generator's write target. These cover the P1
contract: idempotent create on (customer_id, pair_key), status filtering,
the resolve/dismiss transitions, and the D4 don't-reopen guarantee
(a terminal pair_key still reads as existing, so the scan never
re-surfaces it).
"""
from __future__ import annotations

from datetime import datetime, timezone


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def test_create_and_list_conflict(store, customer):
    conflict = await store.create_knowledge_conflict(
        customer.id,
        fact_a_id="fact_a",
        fact_b_id="fact_b",
        claim_a="The contract rate is $120/hr.",
        claim_b="The contract rate is $95/hr.",
        pair_key="pk_rate_001",
        crystal_a_id="cryst_1",
        crystal_b_id="cryst_2",
        subject="Contract|Rate",
        provenance_a="document @ msa_2024.pdf",
        provenance_b="document @ rate_card.xlsx",
    )
    assert conflict.id.startswith("kc_")
    assert conflict.status == "open"
    assert conflict.resolution is None
    assert conflict.detector == "contradiction_scan"

    rows = await store.list_knowledge_conflicts(customer.id)
    assert len(rows) == 1
    got = rows[0]
    assert got.fact_a_id == "fact_a"
    assert got.fact_b_id == "fact_b"
    assert got.claim_a == "The contract rate is $120/hr."
    assert got.claim_b == "The contract rate is $95/hr."
    assert got.subject == "Contract|Rate"
    assert got.provenance_a == "document @ msa_2024.pdf"
    assert got.pair_key == "pk_rate_001"


async def test_create_is_idempotent_on_pair_key(store, customer):
    first = await store.create_knowledge_conflict(
        customer.id,
        fact_a_id="fa",
        fact_b_id="fb",
        claim_a="A",
        claim_b="B",
        pair_key="pk_dupe",
    )
    # Re-running the scan over the unchanged bank: same pair_key.
    second = await store.create_knowledge_conflict(
        customer.id,
        fact_a_id="fa",
        fact_b_id="fb",
        claim_a="A",
        claim_b="B",
        pair_key="pk_dupe",
    )
    assert first.id == second.id
    rows = await store.list_knowledge_conflicts(customer.id)
    assert len(rows) == 1


async def test_idempotent_create_does_not_reset_status(store, customer):
    created = await store.create_knowledge_conflict(
        customer.id,
        fact_a_id="fa",
        fact_b_id="fb",
        claim_a="A",
        claim_b="B",
        pair_key="pk_keep",
    )
    await store.mark_knowledge_conflict_dismissed(
        created.id, resolved_at=_utcnow()
    )
    # A later scan hitting the same pair must NOT reopen the dismissed row.
    again = await store.create_knowledge_conflict(
        customer.id,
        fact_a_id="fa",
        fact_b_id="fb",
        claim_a="A",
        claim_b="B",
        pair_key="pk_keep",
    )
    assert again.id == created.id
    assert again.status == "dismissed"


async def test_changed_claim_new_pair_key_is_separate(store, customer):
    await store.create_knowledge_conflict(
        customer.id,
        fact_a_id="fa",
        fact_b_id="fb",
        claim_a="old claim",
        claim_b="B",
        pair_key="pk_v1",
    )
    # A fact whose claim changed → generator computes a different pair_key.
    await store.create_knowledge_conflict(
        customer.id,
        fact_a_id="fa",
        fact_b_id="fb",
        claim_a="new claim",
        claim_b="B",
        pair_key="pk_v2",
    )
    rows = await store.list_knowledge_conflicts(customer.id)
    assert len(rows) == 2


async def test_knowledge_conflict_exists(store, customer):
    assert (
        await store.knowledge_conflict_exists(customer.id, pair_key="pk_x")
        is False
    )
    await store.create_knowledge_conflict(
        customer.id,
        fact_a_id="fa",
        fact_b_id="fb",
        claim_a="A",
        claim_b="B",
        pair_key="pk_x",
    )
    assert (
        await store.knowledge_conflict_exists(customer.id, pair_key="pk_x")
        is True
    )


async def test_exists_true_after_dismiss_dont_reopen(store, customer):
    """D4: a terminal (dismissed) pair still reads as existing, so the
    generator's pre-check skips it and never re-surfaces it."""
    created = await store.create_knowledge_conflict(
        customer.id,
        fact_a_id="fa",
        fact_b_id="fb",
        claim_a="A",
        claim_b="B",
        pair_key="pk_term",
    )
    await store.mark_knowledge_conflict_dismissed(
        created.id, resolved_at=_utcnow()
    )
    assert (
        await store.knowledge_conflict_exists(customer.id, pair_key="pk_term")
        is True
    )


async def test_status_filter_and_transitions(store, customer):
    a = await store.create_knowledge_conflict(
        customer.id, fact_a_id="a1", fact_b_id="a2",
        claim_a="A1", claim_b="A2", pair_key="pk_a",
    )
    await store.create_knowledge_conflict(
        customer.id, fact_a_id="b1", fact_b_id="b2",
        claim_a="B1", claim_b="B2", pair_key="pk_b",
    )

    await store.mark_knowledge_conflict_resolved(
        a.id, resolution="qualified", resolved_at=_utcnow()
    )

    open_rows = await store.list_knowledge_conflicts(customer.id, status="open")
    assert len(open_rows) == 1
    assert open_rows[0].pair_key == "pk_b"

    resolved_rows = await store.list_knowledge_conflicts(
        customer.id, status="resolved"
    )
    assert len(resolved_rows) == 1
    assert resolved_rows[0].resolution == "qualified"
    assert resolved_rows[0].resolved_at is not None


async def test_count_open_conflicts(store, customer):
    a = await store.create_knowledge_conflict(
        customer.id, fact_a_id="a1", fact_b_id="a2",
        claim_a="A1", claim_b="A2", pair_key="pk_c1",
    )
    await store.create_knowledge_conflict(
        customer.id, fact_a_id="b1", fact_b_id="b2",
        claim_a="B1", claim_b="B2", pair_key="pk_c2",
    )
    assert await store.count_knowledge_conflicts(customer.id) == 2

    await store.mark_knowledge_conflict_dismissed(a.id, resolved_at=_utcnow())
    assert await store.count_knowledge_conflicts(customer.id) == 1
    assert (
        await store.count_knowledge_conflicts(customer.id, status="dismissed")
        == 1
    )


async def test_conflicts_are_tenant_scoped(store, customer):
    other = await store.create_customer(
        provider="anthropic",
        model_id="claude-sonnet-4-5-20250929",
        api_key_ref="sk-test-other",
    )
    await store.create_knowledge_conflict(
        customer.id, fact_a_id="a1", fact_b_id="a2",
        claim_a="A1", claim_b="A2", pair_key="pk_shared",
    )
    # Same pair_key under a DIFFERENT customer is a distinct conflict
    # (uniqueness is scoped to customer_id).
    await store.create_knowledge_conflict(
        other.id, fact_a_id="a1", fact_b_id="a2",
        claim_a="A1", claim_b="A2", pair_key="pk_shared",
    )
    assert len(await store.list_knowledge_conflicts(customer.id)) == 1
    assert len(await store.list_knowledge_conflicts(other.id)) == 1
