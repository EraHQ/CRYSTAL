"""Foundation F3 — promotion engine (detect → curate → merge up) tests.

Exercises the operator→team rung: independent operators holding
near-identical private crystals surface as one promotion candidate; a merge
produces a single team crystal carrying both operators' provenance +
reserved shares, with the originals cleanly superseded.

Direct-call convention (see test_operators_api): the engine is driven
against the in-memory store fixture; the HTTP layer is the F3 API chunk, not
here. asyncio_mode=auto (pyproject) — async tests need no marker.
"""
from __future__ import annotations

import uuid

import pytest

from crystal_cache.maintenance.promotion_service import (
    PromotionError,
    PromotionService,
    TOTAL_SHARE_BASIS_POINTS,
)
from crystal_cache.models import Crystal


async def _make_operator(store, customer, name, role="operator"):
    op, _raw = await store.create_operator(
        team_id=customer.id, display_name=name, role=role,
    )
    return op


async def _make_private_crystal(
    store,
    customer,
    operator,
    routing_vector,
    *,
    fact_count=1,
    summary_text=None,
):
    """Create an operator-private crystal with a controlled routing vector."""
    crystal = Crystal(
        id=f"crys_{uuid.uuid4().hex[:16]}",
        customer_id=customer.id,
        summary_vector=[],
        routing_vector=routing_vector,
        owner_operator_id=operator.id,
        group_team_id=customer.id,
        mode=0o600,
        fact_count=fact_count,
        summary_text=summary_text,
        crystal_type="customer:legacy",
    )
    await store.upsert_crystal(crystal)
    return crystal


# ---------------------------------------------------------------------------
# detect
# ---------------------------------------------------------------------------

async def test_detect_clusters_two_operators_near_identical(store, customer):
    ada = await _make_operator(store, customer, "Ada")
    grace = await _make_operator(store, customer, "Grace")
    a = await _make_private_crystal(store, customer, ada, [1.0, 0.0, 0.0])
    b = await _make_private_crystal(store, customer, grace, [0.99, 0.01, 0.0])

    svc = PromotionService(store)
    candidates = await svc.detect_candidates(customer.id, threshold=0.95)

    assert len(candidates) == 1
    cand = candidates[0]
    assert set(cand.crystal_ids) == {a.id, b.id}
    assert set(cand.operator_ids) == {ada.id, grace.id}
    assert cand.size == 2
    assert cand.mean_similarity >= 0.95


async def test_detect_ignores_single_operator_duplicate(store, customer):
    ada = await _make_operator(store, customer, "Ada")
    # Two near-identical crystals but ONE operator → not a promotion.
    await _make_private_crystal(store, customer, ada, [1.0, 0.0, 0.0])
    await _make_private_crystal(store, customer, ada, [1.0, 0.0, 0.0])

    svc = PromotionService(store)
    assert await svc.detect_candidates(customer.id, threshold=0.95) == []


async def test_detect_ignores_dissimilar(store, customer):
    ada = await _make_operator(store, customer, "Ada")
    grace = await _make_operator(store, customer, "Grace")
    await _make_private_crystal(store, customer, ada, [1.0, 0.0, 0.0])
    await _make_private_crystal(store, customer, grace, [0.0, 1.0, 0.0])

    svc = PromotionService(store)
    assert await svc.detect_candidates(customer.id, threshold=0.95) == []


# ---------------------------------------------------------------------------
# merge — the F3 done-when
# ---------------------------------------------------------------------------

async def test_merge_supersedes_and_records_provenance(store, customer):
    ada = await _make_operator(store, customer, "Ada")
    grace = await _make_operator(store, customer, "Grace")
    # Ada's crystal is richer (survivor); Grace's is superseded.
    a = await _make_private_crystal(
        store, customer, ada, [1.0, 0.0, 0.0], fact_count=5,
    )
    b = await _make_private_crystal(
        store, customer, grace, [1.0, 0.0, 0.0], fact_count=2,
    )

    svc = PromotionService(store)
    result = await svc.merge(customer.id, [a.id, b.id])

    # Survivor = richer crystal (Ada's), now team-owned.
    assert result.merged_crystal_id == a.id
    assert result.superseded_crystal_ids == [b.id]

    survivor = await store.get_crystal(a.id)
    assert survivor is not None
    assert survivor.owner_operator_id is None        # team-owned (chgrp)
    assert survivor.group_team_id == customer.id      # grouped to team
    assert survivor.mode == 0o640                     # group-readable (chmod)
    assert survivor.customer_id == customer.id        # tenancy unchanged

    # Non-survivor cleanly superseded.
    assert await store.get_crystal(b.id) is None

    # Provenance: one row per source, both operators, shares sum to 10000.
    prov = await store.list_promotion_contributions(a.id)
    assert len(prov) == 2
    by_source = {p["source_crystal_id"]: p for p in prov}
    assert by_source[a.id]["contributor_operator_id"] == ada.id
    assert by_source[b.id]["contributor_operator_id"] == grace.id
    assert sum(p["share_basis_points"] for p in prov) == TOTAL_SHARE_BASIS_POINTS


async def test_merge_rejects_single_operator(store, customer):
    ada = await _make_operator(store, customer, "Ada")
    a = await _make_private_crystal(store, customer, ada, [1.0, 0.0, 0.0])
    b = await _make_private_crystal(store, customer, ada, [1.0, 0.0, 0.0])

    svc = PromotionService(store)
    with pytest.raises(PromotionError):
        await svc.merge(customer.id, [a.id, b.id])


async def test_merge_rejects_cross_team(store, customer):
    ada = await _make_operator(store, customer, "Ada")
    a = await _make_private_crystal(store, customer, ada, [1.0, 0.0, 0.0])

    other = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="x",
    )
    other_op, _ = await store.create_operator(
        team_id=other.id, display_name="Zed",
    )
    foreign = Crystal(
        id=f"crys_{uuid.uuid4().hex[:16]}",
        customer_id=other.id,
        summary_vector=[],
        routing_vector=[1.0, 0.0, 0.0],
        owner_operator_id=other_op.id,
        group_team_id=other.id,
        mode=0o600,
        crystal_type="customer:legacy",
    )
    await store.upsert_crystal(foreign)

    svc = PromotionService(store)
    with pytest.raises(PromotionError):
        await svc.merge(customer.id, [a.id, foreign.id])


async def test_merge_rejects_non_operator_owned(store, customer):
    ada = await _make_operator(store, customer, "Ada")
    a = await _make_private_crystal(store, customer, ada, [1.0, 0.0, 0.0])
    # A team-owned (unowned) crystal — not promotable.
    team_crystal = Crystal(
        id=f"crys_{uuid.uuid4().hex[:16]}",
        customer_id=customer.id,
        summary_vector=[],
        routing_vector=[1.0, 0.0, 0.0],
        owner_operator_id=None,
        group_team_id=customer.id,
        mode=0o640,
        crystal_type="customer:legacy",
    )
    await store.upsert_crystal(team_crystal)

    svc = PromotionService(store)
    with pytest.raises(PromotionError):
        await svc.merge(customer.id, [a.id, team_crystal.id])


def test_equal_shares_sum():
    for n in (1, 2, 3, 4, 7):
        shares = PromotionService._equal_shares(n)
        assert len(shares) == n
        assert sum(shares) == TOTAL_SHARE_BASIS_POINTS
