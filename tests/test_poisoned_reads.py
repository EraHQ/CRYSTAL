"""Poisoned reads — the writer/Literal drift class (fixed 2026-08-20).

The defect class: a writer emits a string outside a Pydantic Literal;
the row persists fine; every subsequent read that hydrates the table
through the model raises ValidationError — an entire feature dies
silently. Two live instances were fixed:

1. Quality tier: the admin tier PATCH route allowlisted 'verified',
   which is not in QualityTier (models/crystal.py — the model value is
   'whitelist'). The route now derives its allowlist from
   typing.get_args(QualityTier); the console dropdown / TIER_FILL were
   re-synced to the model vocabulary.

   DATA REPAIR for rows already poisoned by the old route (there is no
   defensive coercion in _crystal_from_row, by design — validation
   lives in the model):

       UPDATE crystals SET quality_tier = 'whitelist'
       WHERE quality_tier = 'verified';

2. Gap literals: the code legitimately writes disposition
   'cycles_exhausted' (workers/cognition.py fill sweep), status
   'retrying' / 'needs_operator' (metadata_store_agent_ext), and
   source 'agent_run' (create_agent_gap) — none were in the Literals.
   They are now; the store writers validate against get_args() and
   raise ValueError on drift; and the cognition worker's idle-phase
   sequence runs each phase through _run_idle_phase so one poisoned
   row kills one phase, not the whole idle pass.

Frontend note (no test infra): grep-verified that no frontend file
still carries a tier value outside the model vocabulary ('verified' /
'established' removed from BankBrowser.tsx and Constellation.tsx).

R14 note: verified by pytest.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import get_args

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from crystal_cache.endpoints.admin import admin_set_crystal_tier
from crystal_cache.infrastructure.schema import CrystalRow, KnowledgeGapRow
from crystal_cache.models.crystal import QualityTier
from crystal_cache.models.knowledge_gap import GapDisposition, GapStatus
from crystal_cache.workers.cognition import _run_idle_phase


class _Req:
    """FakeRequest for the direct-call endpoint convention
    (test_assumptions_admin.py precedent) + async body."""

    def __init__(self, body=None, tenant_pin=None):
        self._body = body or {}
        self.state = SimpleNamespace()
        if tenant_pin is not None:
            self.state.tenant_pin = tenant_pin

    async def json(self):
        return self._body


async def _seed_crystal(store, crystal_id, customer_id):
    async with store.session() as s:
        s.add(CrystalRow(
            id=crystal_id, customer_id=customer_id,
            crystal_type="customer:legacy", summary_vector=[],
            summary_text="poison probe", quality_tier="neutral",
        ))


# ---------------------------------------------------------------------------
# 1. Tier route: model-derived allowlist
# ---------------------------------------------------------------------------

async def test_tier_route_rejects_value_outside_literal(store, customer):
    await _seed_crystal(store, "cr_tier_1", customer.id)
    with pytest.raises(HTTPException) as exc:
        await admin_set_crystal_tier(
            _Req({"tier": "verified"}), "cr_tier_1", store,
        )
    assert exc.value.status_code == 422
    # The 422 names the allowed (model) values — and only those.
    for v in get_args(QualityTier):
        assert v in exc.value.detail
    assert "verified" not in exc.value.detail
    # The crystal is untouched — and still hydrates.
    c = await store.get_crystal("cr_tier_1")
    assert c.quality_tier == "neutral"


async def test_tier_route_accepts_every_model_value(store, customer):
    await _seed_crystal(store, "cr_tier_2", customer.id)
    for v in get_args(QualityTier):
        resp = await admin_set_crystal_tier(
            _Req({"tier": v}), "cr_tier_2", store,
        )
        assert resp.status_code == 200
        # The original kill vector: the read after the write.
        c = await store.get_crystal("cr_tier_2")
        assert c.quality_tier == v


# ---------------------------------------------------------------------------
# 2. Write-side gap validation
# ---------------------------------------------------------------------------

async def _gap(store, customer_id):
    return await store.create_knowledge_gap(
        customer_id, domain=None, subject="s", missing="what is missing",
    )


async def test_update_disposition_rejects_unknown_value(store, customer):
    gap = await _gap(store, customer.id)
    with pytest.raises(ValueError):
        await store.update_knowledge_gap_disposition(gap.id, "exhausted")


async def test_update_disposition_accepts_cycles_exhausted(store, customer):
    gap = await _gap(store, customer.id)
    await store.update_knowledge_gap_disposition(gap.id, "cycles_exhausted")
    got = await store.get_knowledge_gap(gap.id)
    assert got.disposition == "cycles_exhausted"


async def test_create_knowledge_gap_rejects_unknown_source(store, customer):
    with pytest.raises(ValueError):
        await store.create_knowledge_gap(
            customer.id, domain=None, subject="s", missing="m",
            source="not_a_source",
        )


async def test_resolve_agent_gap_rejects_unknown_status(store, customer):
    g = await store.create_agent_gap(
        customer.id, task="t", task_id="tid", branch="b",
        failing_tail="tail", project_dir="/p",
    )
    with pytest.raises(ValueError):
        await store.resolve_agent_gap(g["id"], status="abandoned")
    assert await store.resolve_agent_gap(g["id"], status="needs_operator")


# ---------------------------------------------------------------------------
# 3. The original kill vector: cycles_exhausted must round-trip
# ---------------------------------------------------------------------------

async def test_cycles_exhausted_round_trips_through_list(store, customer):
    gap = await _gap(store, customer.id)
    await store.update_knowledge_gap_disposition(gap.id, "cycles_exhausted")
    gaps = await store.list_knowledge_gaps(customer.id, limit=10)
    assert [g.disposition for g in gaps if g.id == gap.id] == \
        ["cycles_exhausted"]
    # Sanity: the value is a member of the Literal, not a coincidence.
    assert "cycles_exhausted" in get_args(GapDisposition)


async def test_agent_statuses_round_trip_through_list(store, customer):
    """'retrying' / 'needs_operator' rows (agent gaps) must hydrate."""
    await store.create_agent_gap(
        customer.id, task="t", task_id="tid", branch="b",
        failing_tail="tail", project_dir="/p",
    )
    claimed = await store.claim_next_open_agent_gap()   # open -> retrying
    assert claimed is not None
    gaps = await store.list_knowledge_gaps(customer.id, limit=10)
    assert any(g.status == "retrying" for g in gaps)
    assert {"retrying", "needs_operator"} <= set(get_args(GapStatus))


# ---------------------------------------------------------------------------
# 4. Blast radius: a poisoned row kills one phase, not the idle pass
# ---------------------------------------------------------------------------

async def _poison_gap_row(store, customer_id):
    async with store.session() as s:
        s.add(KnowledgeGapRow(
            id="gap_poisoned", customer_id=customer_id,
            missing="m", status="zombie",   # outside GapStatus, on purpose
        ))


async def test_poisoned_row_makes_list_raise(store, customer):
    await _poison_gap_row(store, customer.id)
    with pytest.raises(ValidationError):
        await store.list_knowledge_gaps(customer.id, limit=10)


async def test_idle_phase_wrapper_contains_the_blast(store, customer):
    """The worker's per-phase fence (_run_idle_phase): phase A reads the
    poisoned table and dies; phase B still runs."""
    await _poison_gap_row(store, customer.id)
    ran: list[str] = []

    async def phase_a():
        await store.list_knowledge_gaps(customer.id, limit=10)  # raises
        ran.append("a")

    async def phase_b():
        ran.append("b")

    await _run_idle_phase("phase_a", phase_a())
    await _run_idle_phase("phase_b", phase_b())
    assert ran == ["b"]
