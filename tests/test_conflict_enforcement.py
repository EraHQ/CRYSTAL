"""Conflict enforcement — the Conflicts page's human decision takes effect.

apply_conflict_resolution's superseded/blacklisted verbs set the losing
fact's grating_strength to 0.0 ("deactivate it from retrieval"). These
tests pin the enforcement half of that promise (2026-08-20):

  * the fact-lane loaders (list_all_facts_for_customer → FactVectorStore)
    and the navigation key-scan (list_facts_by_key_prefix) EXCLUDE
    grating-0 facts, so a settled loser stops surfacing;
  * list_facts_for_crystal keeps its admin default (deactivated facts
    stay visible in the Bank browser) but hides them when retrieval
    hydration passes include_deactivated=False;
  * a NULL grating (hypothetical legacy row — the live schema is NOT
    NULL, so the arm is defensive) reads as ACTIVE, never as deactivated;
  * the admin resolve endpoint invalidates the per-customer fact index
    on superseded/blacklisted (so the change lands on the NEXT query),
    and does NOT invalidate on dismissed/qualified (no fact effect).
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from sqlalchemy import text

from crystal_cache.endpoints.admin import (
    ResolveConflictRequest,
    admin_resolve_conflict,
)
from crystal_cache.infrastructure.schema import CrystalRow, FactRow

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
_NOW = datetime(2026, 8, 20, tzinfo=timezone.utc)

# Orthogonal-ish 4-dim vectors: the query points at the loser so the
# only way it drops out of the results is the grating filter, not rank.
_VEC_LOSER = [1.0, 0.0, 0.0, 0.0]
_VEC_WINNER = [0.6, 0.8, 0.0, 0.0]
_QUERY = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


async def _seed_conflict(store, customer_id):
    """One crystal, two vectored facts with sparse keys, one open conflict."""
    async with store.session() as s:
        s.add(CrystalRow(
            id="cE", customer_id=customer_id,
            crystal_type="customer:legacy", summary_vector=[],
        ))
        s.add(FactRow(
            id="f_lose", crystal_id="cE", pair_type="question_answer",
            prompt_text="Rate|Old", claim_text="rate is 120",
            source_kind="model_reasoning", vector=_VEC_LOSER,
            grating_strength=1.0, created_at=_T0,
        ))
        s.add(FactRow(
            id="f_win", crystal_id="cE", pair_type="question_answer",
            prompt_text="Rate|New", claim_text="rate is 95",
            source_kind="model_reasoning", vector=_VEC_WINNER,
            grating_strength=1.0, created_at=_T0,
        ))
    return await store.create_knowledge_conflict(
        customer_id, fact_a_id="f_lose", fact_b_id="f_win",
        claim_a="rate is 120", claim_b="rate is 95", pair_key="pk-enf",
        crystal_a_id="cE", crystal_b_id="cE", subject="Rate",
    )


# --- 1. end-to-end store level: resolution removes the loser from the lanes


async def test_superseded_loser_leaves_fact_lane_and_key_scan(
    store, customer, fact_vector_store,
):
    c = await _seed_conflict(store, customer.id)

    # Before resolution: both facts on both lanes.
    before = await fact_vector_store.search(customer.id, _QUERY, k=10)
    assert {r[0] for r in before} == {"f_lose", "f_win"}
    keys = await store.list_facts_by_key_prefix(customer.id, key_prefix="Rate|")
    assert {f.id for f in keys} == {"f_lose", "f_win"}

    await store.apply_conflict_resolution(
        c.id, resolution="superseded", loser="a", resolved_at=_NOW,
    )
    # The per-customer fact cache is stale until invalidated — this is
    # what the endpoint does; here we exercise the store-level loop.
    fact_vector_store.invalidate(customer.id)

    after = await fact_vector_store.search(customer.id, _QUERY, k=10)
    assert {r[0] for r in after} == {"f_win"}      # loser gone, winner stays
    keys = await store.list_facts_by_key_prefix(customer.id, key_prefix="Rate|")
    assert {f.id for f in keys} == {"f_win"}       # key-scan lane too
    # And the loader every backend builds from excludes it as well.
    all_facts = await store.list_all_facts_for_customer(customer.id)
    assert {f.id for f in all_facts} == {"f_win"}


# --- 2. admin visibility preserved; retrieval hydration hides


async def test_list_facts_for_crystal_visibility_split(store, customer):
    c = await _seed_conflict(store, customer.id)
    await store.apply_conflict_resolution(
        c.id, resolution="superseded", loser="a", resolved_at=_NOW,
    )
    # Default (admin/Bank browser): deactivated fact still visible.
    admin_view = await store.list_facts_for_crystal("cE")
    assert {f.id for f in admin_view} == {"f_lose", "f_win"}
    explicit = await store.list_facts_for_crystal("cE", include_deactivated=True)
    assert {f.id for f in explicit} == {"f_lose", "f_win"}
    # Retrieval hydration: hidden.
    retrieval_view = await store.list_facts_for_crystal(
        "cE", include_deactivated=False,
    )
    assert {f.id for f in retrieval_view} == {"f_win"}


# --- 3. NULL grating (legacy) is treated as ACTIVE


async def test_null_grating_reads_as_active(store, customer):
    """The live schema declares grating_strength NOT NULL, so a NULL row
    cannot exist under it — the IS NULL arm is defensive. To prove the
    semantics anyway, rebuild the throwaway test table without NOT NULL
    constraints, plant a NULL, and assert every lane treats it as active."""
    await _seed_conflict(store, customer.id)
    async with store.session() as s:
        ddl = (await s.execute(text(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='facts'"
        ))).scalar_one()
        await s.execute(text("ALTER TABLE facts RENAME TO facts_nn_bak"))
        await s.execute(text(ddl.replace("NOT NULL", "")))
        await s.execute(text("INSERT INTO facts SELECT * FROM facts_nn_bak"))
        await s.execute(text("DROP TABLE facts_nn_bak"))
        await s.execute(text(
            "UPDATE facts SET grating_strength = NULL WHERE id = 'f_lose'"
        ))

    all_facts = await store.list_all_facts_for_customer(customer.id)
    assert {f.id for f in all_facts} == {"f_lose", "f_win"}
    # The row→model copy maps the NULL to fully active, not a falsy 0.
    legacy = next(f for f in all_facts if f.id == "f_lose")
    assert legacy.grating_strength == 1.0
    keys = await store.list_facts_by_key_prefix(customer.id, key_prefix="Rate|")
    assert {f.id for f in keys} == {"f_lose", "f_win"}
    hydrated = await store.list_facts_for_crystal(
        "cE", include_deactivated=False,
    )
    assert {f.id for f in hydrated} == {"f_lose", "f_win"}


# --- 4. endpoint invalidates the fact index on deactivating verbs only


class _SpyIndex:
    def __init__(self):
        self.invalidated: list[str] = []

    def invalidate(self, customer_id: str) -> None:
        self.invalidated.append(customer_id)


class _FakeRequest:
    """Request stub carrying app.state.vector_index (the seam the
    endpoint invalidates through) and no tenant pin."""

    def __init__(self, spy):
        class _State:
            tenant_pin = None
        class _AppState:
            pass
        class _App:
            pass
        self.state = _State()
        self.app = _App()
        self.app.state = _AppState()
        self.app.state.vector_index = spy


async def test_resolve_endpoint_invalidates_on_superseded(store, customer):
    c = await _seed_conflict(store, customer.id)
    spy = _SpyIndex()
    resp = await admin_resolve_conflict(
        request=_FakeRequest(spy),
        conflict_id=c.id,
        body=ResolveConflictRequest(resolution="superseded", loser="a"),
        store=store,
    )
    assert resp["conflict"]["resolution"] == "superseded"
    assert spy.invalidated == [customer.id]


async def test_resolve_endpoint_no_invalidate_on_dismissed_or_qualified(
    store, customer,
):
    c = await _seed_conflict(store, customer.id)
    spy = _SpyIndex()
    await admin_resolve_conflict(
        request=_FakeRequest(spy),
        conflict_id=c.id,
        body=ResolveConflictRequest(resolution="dismissed"),
        store=store,
    )
    assert spy.invalidated == []

    c2 = await store.create_knowledge_conflict(
        customer.id, fact_a_id="f_lose", fact_b_id="f_win",
        claim_a="a", claim_b="b", pair_key="pk-enf-2",
        crystal_a_id="cE", crystal_b_id="cE", subject="Rate",
    )
    await admin_resolve_conflict(
        request=_FakeRequest(spy),
        conflict_id=c2.id,
        body=ResolveConflictRequest(resolution="qualified"),
        store=store,
    )
    assert spy.invalidated == []
