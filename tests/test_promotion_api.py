"""Foundation F3 — promotion API (endpoints/promotion.py) tests.

The admin-gated HTTP surface over the promotion engine: GET candidates
(detect live) and POST merge. Direct-call convention with a FakeRequest
carrying app.state.{vector_store,fact_vector_store}; the engine's own
behavior is covered in test_promotion_engine.py, so these focus on the
endpoint wiring — principal → team scoping, response shape, store
invalidation path, and the 400 mapping on PromotionError.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi import HTTPException

from crystal_cache.endpoints.promotion import (
    MergeRequest,
    list_promotion_candidates,
    merge_promotion_candidate,
)
from crystal_cache.models import Crystal


@dataclass
class _FakeState:
    vector_store: Any = None
    fact_vector_store: Any = None


@dataclass
class _FakeApp:
    state: _FakeState = field(default_factory=_FakeState)


@dataclass
class _FakeRequest:
    app: _FakeApp = field(default_factory=_FakeApp)


async def _make_operator(store, customer, name, role="operator"):
    op, _raw = await store.create_operator(
        team_id=customer.id, display_name=name, role=role,
    )
    return op


async def _make_private_crystal(
    store, customer, operator, routing_vector, *, fact_count=1,
):
    crystal = Crystal(
        id=f"crys_{uuid.uuid4().hex[:16]}",
        customer_id=customer.id,
        summary_vector=[],
        routing_vector=routing_vector,
        owner_operator_id=operator.id,
        group_team_id=customer.id,
        mode=0o600,
        fact_count=fact_count,
        crystal_type="customer:legacy",
    )
    await store.upsert_crystal(crystal)
    return crystal


async def test_candidates_endpoint_returns_clusters(store, customer):
    ada = await _make_operator(store, customer, "Ada")
    grace = await _make_operator(store, customer, "Grace")
    a = await _make_private_crystal(store, customer, ada, [1.0, 0.0, 0.0])
    b = await _make_private_crystal(store, customer, grace, [1.0, 0.0, 0.0])

    resp = await list_promotion_candidates(
        request=_FakeRequest(),
        principal=(customer, None),
        store=store,
        threshold=0.95,
    )
    assert len(resp.candidates) == 1
    cand = resp.candidates[0]
    assert set(cand.crystal_ids) == {a.id, b.id}
    assert cand.size == 2


async def test_merge_endpoint_promotes_and_returns_result(
    store, customer, vector_store, fact_vector_store,
):
    ada = await _make_operator(store, customer, "Ada")
    grace = await _make_operator(store, customer, "Grace")
    a = await _make_private_crystal(
        store, customer, ada, [1.0, 0.0, 0.0], fact_count=5,
    )
    b = await _make_private_crystal(
        store, customer, grace, [1.0, 0.0, 0.0], fact_count=2,
    )

    resp = await merge_promotion_candidate(
        body=MergeRequest(source_crystal_ids=[a.id, b.id]),
        request=_FakeRequest(_FakeApp(_FakeState(
            vector_store=vector_store,
            fact_vector_store=fact_vector_store,
        ))),
        principal=(customer, None),
        store=store,
    )
    assert resp.merged_crystal_id == a.id
    assert resp.superseded_crystal_ids == [b.id]
    assert sum(c["share_basis_points"] for c in resp.contributions) == 10000

    # Survivor is team-owned; non-survivor gone.
    survivor = await store.get_crystal(a.id)
    assert survivor.owner_operator_id is None
    assert survivor.group_team_id == customer.id
    assert survivor.mode == 0o640
    assert await store.get_crystal(b.id) is None


async def test_merge_endpoint_rejects_invalid_with_400(store, customer):
    ada = await _make_operator(store, customer, "Ada")
    # A single operator's two crystals → engine raises PromotionError,
    # which the endpoint maps to HTTP 400. The default FakeRequest's
    # None stores are never touched (the engine validates first).
    a = await _make_private_crystal(store, customer, ada, [1.0, 0.0, 0.0])
    b = await _make_private_crystal(store, customer, ada, [1.0, 0.0, 0.0])

    with pytest.raises(HTTPException) as exc:
        await merge_promotion_candidate(
            body=MergeRequest(source_crystal_ids=[a.id, b.id]),
            request=_FakeRequest(),
            principal=(customer, None),
            store=store,
        )
    assert exc.value.status_code == 400
