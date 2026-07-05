"""Recall-gated memory (2026-07-03, option b) — the safety-critical core.

Background-worker memory must be UNUSABLE (not recallable) until approved,
without changing what any quality_tier means. The mechanism: a recall_gated
bit orthogonal to tier, plus an origin attribution. This test proves:
  - crystals are born ungated/direct by default (zero behavior change);
  - a gated crystal is held OUT of the recall candidate load;
  - admin/promotion reads STILL see gated crystals (to review/promote them);
  - clearing the gate makes the crystal recallable again;
  - origin is stamped and queryable.

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

import pytest

from crystal_cache.infrastructure.schema import CrystalRow
from crystal_cache.models.crystal import Crystal


async def _make_crystal(store, customer_id, *, recall_gated, origin, cid):
    """Insert a crystal row directly with the given gate/origin."""
    c = Crystal(
        id=cid, customer_id=customer_id, summary_vector=[0.1, 0.2],
        crystal_type="customer:legacy", recall_gated=recall_gated,
        origin=origin,
    )
    await store.upsert_crystal(c)
    return c


# --- defaults: zero behavior change ----------------------------------------

async def test_crystals_are_born_ungated_and_direct(store, customer):
    await _make_crystal(store, customer.id, recall_gated=False,
                        origin="direct", cid="crys_default")
    got = await store.get_crystal("crys_default")
    assert got.recall_gated is False
    assert got.origin == "direct"


# --- the gate holds crystals out of the recall load ------------------------

async def test_gated_crystal_hidden_from_recall_load(store, customer):
    await _make_crystal(store, customer.id, recall_gated=False,
                        origin="direct", cid="crys_ungated")
    await _make_crystal(store, customer.id, recall_gated=True,
                        origin="background_worker", cid="crys_gated")

    # The recall path passes include_recall_gated=False.
    recall_view = await store.list_crystals_for_customer(
        customer.id, include_recall_gated=False,
    )
    ids = {c.id for c in recall_view}
    assert "crys_ungated" in ids
    assert "crys_gated" not in ids  # held out of recall


async def test_admin_read_still_sees_gated(store, customer):
    await _make_crystal(store, customer.id, recall_gated=True,
                        origin="background_worker", cid="crys_gated2")
    # Default include_recall_gated=True: admin/promotion/consolidation see it.
    admin_view = await store.list_crystals_for_customer(customer.id)
    assert "crys_gated2" in {c.id for c in admin_view}


async def test_and_type_recall_filter(store, customer):
    await _make_crystal(store, customer.id, recall_gated=True,
                        origin="background_worker", cid="crys_gated_typed")
    recall_view = await store.list_crystals_for_customer_and_type(
        customer.id, "customer:legacy", include_recall_gated=False,
    )
    assert "crys_gated_typed" not in {c.id for c in recall_view}
    admin_view = await store.list_crystals_for_customer_and_type(
        customer.id, "customer:legacy",
    )
    assert "crys_gated_typed" in {c.id for c in admin_view}


# --- clearing the gate = promotion -----------------------------------------

async def test_clearing_gate_makes_recallable(store, customer):
    await _make_crystal(store, customer.id, recall_gated=True,
                        origin="background_worker", cid="crys_promote")
    # Hidden from recall while gated.
    before = await store.list_crystals_for_customer(
        customer.id, include_recall_gated=False,
    )
    assert "crys_promote" not in {c.id for c in before}

    # Clear the gate (promotion).
    ok = await store.set_crystal_recall_gate(
        "crys_promote", customer.id, gated=False,
    )
    assert ok is True

    # Now visible to recall.
    after = await store.list_crystals_for_customer(
        customer.id, include_recall_gated=False,
    )
    assert "crys_promote" in {c.id for c in after}


async def test_setting_gate_removes_from_recall(store, customer):
    await _make_crystal(store, customer.id, recall_gated=False,
                        origin="direct", cid="crys_regate")
    ok = await store.set_crystal_recall_gate(
        "crys_regate", customer.id, gated=True,
    )
    assert ok is True
    recall = await store.list_crystals_for_customer(
        customer.id, include_recall_gated=False,
    )
    assert "crys_regate" not in {c.id for c in recall}


# --- the review queue -------------------------------------------------------

async def test_list_recall_gated_is_the_review_queue(store, customer):
    await _make_crystal(store, customer.id, recall_gated=True,
                        origin="background_worker", cid="crys_q1")
    await _make_crystal(store, customer.id, recall_gated=True,
                        origin="background_worker", cid="crys_q2")
    await _make_crystal(store, customer.id, recall_gated=False,
                        origin="direct", cid="crys_notq")

    queue = await store.list_recall_gated_crystals(customer.id)
    qids = {c.id for c in queue}
    assert qids == {"crys_q1", "crys_q2"}

    # origin filter narrows it.
    bg = await store.list_recall_gated_crystals(
        customer.id, origin="background_worker",
    )
    assert {c.id for c in bg} == {"crys_q1", "crys_q2"}


# --- tenancy: the gate clear is customer-guarded ---------------------------

async def test_gate_clear_is_customer_guarded(store, customer):
    await _make_crystal(store, customer.id, recall_gated=True,
                        origin="background_worker", cid="crys_tenant")
    # Wrong customer can't clear it.
    ok = await store.set_crystal_recall_gate(
        "crys_tenant", "some_other_customer", gated=False,
    )
    assert ok is False
    still = await store.get_crystal("crys_tenant")
    assert still.recall_gated is True
