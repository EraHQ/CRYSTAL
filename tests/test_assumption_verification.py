"""C4 — assumption verification spending (ratified 2026-08-11:
Q1=A recall-driven spawn + D manual button, Q2=B own budget function
+ ledger origin, Q3=A substrate-pure goal with no verdict writeback).

Covers the spawn trigger (approved + quarantine/neutral + grounded
citations >= knob + no open conflicts + untagged), the durable tag
idempotence, the per-cycle cap, the S4 budget door's manual-by-default
posture for the new function, env.origin threading, and the manual
Verify endpoint (user-commanded lane, gated-assumption allowed,
invalidated refused).

R14 note: verified by `pytest`.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from crystal_cache.cognition.models import CognitionEnvironment
from crystal_cache.control.admission import function_budget_allows
from crystal_cache.endpoints.admin import admin_verify_assumption
from crystal_cache.infrastructure.schema import CitationRow, CrystalRow
from crystal_cache.scan.verification import (
    VERIFICATION_TASK_TYPE,
    has_verification_tag,
    run_assumption_verification_scan,
    verification_goal,
)


def _request(tenant_pin=None):
    state = SimpleNamespace()
    if tenant_pin is not None:
        state.tenant_pin = tenant_pin
    return SimpleNamespace(state=state)


async def _seed_crystal(store, crystal_id, customer_id, *, summary):
    async with store.session() as s:
        s.add(CrystalRow(
            id=crystal_id, customer_id=customer_id,
            crystal_type="customer:legacy", summary_vector=[],
            summary_text=summary, quality_tier="neutral",
        ))


async def _make_assumption(store, customer, encoder, a, b, statement):
    await _seed_crystal(store, a, customer.id, summary=f"about {a}")
    await _seed_crystal(store, b, customer.id, summary=f"about {b}")
    written = await store.create_assumption_crystal(
        customer.id,
        statement=statement,
        subject="Bridged subject",
        parent_a_id=a,
        parent_b_id=b,
        confidence=0.8,
        encoder=encoder,
    )
    return written["crystal_id"]


async def _seed_grounded_citations(store, customer_id, crystal_id, n):
    async with store.session() as s:
        for _ in range(n):
            s.add(CitationRow(
                id=f"cit_{uuid.uuid4().hex[:12]}",
                customer_id=customer_id,
                crystal_id=crystal_id,
                grounded=True,
            ))


async def _approve(store, customer_id, crystal_id):
    await store.set_crystal_recall_gate(crystal_id, customer_id, False)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_verification_goal_neutral_framing():
    goal = verification_goal("Friday deploys carry elevated risk")
    assert "CONFIRMS OR REFUTES" in goal
    assert "Friday deploys carry elevated risk" in goal
    # Q3=A: refutation framed as equally valuable — no confirmation bias.
    assert "Refutation is exactly as valuable as confirmation" in goal


def test_has_verification_tag():
    assert has_verification_tag(["verification_task:cog_1"]) is True
    assert has_verification_tag(["assumption_confidence:0.8"]) is False
    assert has_verification_tag(None) is False


def test_env_origin_default_and_override():
    # Q2=B: default keeps every existing caller's ledger rows unchanged;
    # the verification lane overrides.
    assert CognitionEnvironment().origin == "cognition"
    assert (
        CognitionEnvironment(origin="assumption_verification").origin
        == "assumption_verification"
    )


# ---------------------------------------------------------------------------
# The spawn trigger (Q1=A)
# ---------------------------------------------------------------------------

async def test_scan_spawns_for_influential_approved_assumption(
    store, customer, semantic_encoder_stub,
):
    asm_id = await _make_assumption(
        store, customer, semantic_encoder_stub, "cr_a", "cr_b",
        "Friday deploys carry elevated risk",
    )
    await _approve(store, customer.id, asm_id)
    await _seed_grounded_citations(store, customer.id, asm_id, 3)

    result = await run_assumption_verification_scan(
        store=store, customer_id=customer.id,
        min_recalls=3, per_cycle=2,
    )
    assert result.tasks_spawned == 1

    tasks = await store.list_cognition_tasks(customer.id)
    task = next(
        t for t in tasks if t.task_type == VERIFICATION_TASK_TYPE
    )
    assert task.payload["assumption_crystal_id"] == asm_id
    assert task.payload["statement"] == (
        "Friday deploys carry elevated risk"
    )
    assert "CONFIRMS OR REFUTES" in task.payload["topic"]

    # Durable spawn record on the crystal itself.
    rows = await store.list_assumption_crystals(customer.id)
    row = next(r for r in rows if r["id"] == asm_id)
    assert has_verification_tag(row["diagnostic_tags"])

    # The witness (C2 activity feed).
    events = await store.list_curation_events(customer.id)
    spawned = [
        e for e in events if e["event_type"] == "verification_spawned"
    ]
    assert len(spawned) == 1
    assert spawned[0]["subject_id"] == asm_id
    assert spawned[0]["payload"]["grounded_citations"] == 3


async def test_scan_skips_gated_underweight_and_whitelist(
    store, customer, semantic_encoder_stub,
):
    # Gated (never approved): even with citations, not in play.
    gated = await _make_assumption(
        store, customer, semantic_encoder_stub, "g1", "g2", "gated one",
    )
    await _seed_grounded_citations(store, customer.id, gated, 3)
    # Approved but below the influence threshold.
    light = await _make_assumption(
        store, customer, semantic_encoder_stub, "l1", "l2", "light one",
    )
    await _approve(store, customer.id, light)
    await _seed_grounded_citations(store, customer.id, light, 2)
    # Approved + cited but already whitelist: the passive loop finished.
    done = await _make_assumption(
        store, customer, semantic_encoder_stub, "w1", "w2", "done one",
    )
    await _approve(store, customer.id, done)
    await _seed_grounded_citations(store, customer.id, done, 3)
    await store.set_crystal_quality_tier(done, customer.id, "whitelist")

    result = await run_assumption_verification_scan(
        store=store, customer_id=customer.id,
        min_recalls=3, per_cycle=5,
    )
    assert result.tasks_spawned == 0
    assert (
        await store.list_cognition_tasks(customer.id)
    ) == []


async def test_scan_skips_open_conflict(
    store, customer, semantic_encoder_stub,
):
    """A disputed assumption is already in the invalidation path —
    verification spend would re-buy what the conflict surface knows."""
    asm_id = await _make_assumption(
        store, customer, semantic_encoder_stub, "c1", "c2",
        "conflicted one",
    )
    await _approve(store, customer.id, asm_id)
    await _seed_grounded_citations(store, customer.id, asm_id, 3)
    await store.create_knowledge_conflict(
        customer.id,
        fact_a_id="fa", fact_b_id="fb",
        claim_a="x", claim_b="not x",
        pair_key="fa|fb",
        crystal_a_id=asm_id, crystal_b_id="c1",
        subject=None, provenance_a=None, provenance_b=None,
    )

    result = await run_assumption_verification_scan(
        store=store, customer_id=customer.id,
        min_recalls=3, per_cycle=5,
    )
    assert result.tasks_spawned == 0
    assert result.skipped_conflicted == 1


async def test_scan_is_idempotent_via_tag(
    store, customer, semantic_encoder_stub,
):
    asm_id = await _make_assumption(
        store, customer, semantic_encoder_stub, "i1", "i2", "idem one",
    )
    await _approve(store, customer.id, asm_id)
    await _seed_grounded_citations(store, customer.id, asm_id, 3)

    first = await run_assumption_verification_scan(
        store=store, customer_id=customer.id, min_recalls=3, per_cycle=5,
    )
    second = await run_assumption_verification_scan(
        store=store, customer_id=customer.id, min_recalls=3, per_cycle=5,
    )
    assert first.tasks_spawned == 1
    assert second.tasks_spawned == 0
    assert second.skipped_tagged == 1
    tasks = [
        t for t in await store.list_cognition_tasks(customer.id)
        if t.task_type == VERIFICATION_TASK_TYPE
    ]
    assert len(tasks) == 1


async def test_scan_respects_per_cycle_cap(
    store, customer, semantic_encoder_stub,
):
    for i in range(2):
        asm = await _make_assumption(
            store, customer, semantic_encoder_stub,
            f"p{i}a", f"p{i}b", f"capped {i}",
        )
        await _approve(store, customer.id, asm)
        await _seed_grounded_citations(store, customer.id, asm, 3)

    result = await run_assumption_verification_scan(
        store=store, customer_id=customer.id, min_recalls=3, per_cycle=1,
    )
    assert result.tasks_spawned == 1


# ---------------------------------------------------------------------------
# The budget door (Q2=B) — manual-by-default posture
# ---------------------------------------------------------------------------

async def test_budget_door_manual_by_default(store, customer):
    # No budget row + the zero default cap = the function is OFF.
    assert await function_budget_allows(
        store, customer, "assumption_verification",
        origin="assumption_verification", default_cap_micro_usd=0,
    ) is False
    # A tenant budget row switches the autonomous lane on.
    await store.upsert_spend_budget(
        customer.id,
        function="assumption_verification",
        cap_micro_usd=1_000_000,
    )
    assert await function_budget_allows(
        store, customer, "assumption_verification",
        origin="assumption_verification", default_cap_micro_usd=0,
    ) is True


# ---------------------------------------------------------------------------
# The manual Verify endpoint (Q1=D, user-commanded lane)
# ---------------------------------------------------------------------------

async def test_manual_verify_endpoint(
    store, customer, semantic_encoder_stub,
):
    """Manual verify works on a still-GATED assumption ('I don't trust
    this — go check' is a legitimate curator move) and needs no budget
    row (user-commanded lane)."""
    asm_id = await _make_assumption(
        store, customer, semantic_encoder_stub, "m1", "m2", "manual one",
    )

    body = await admin_verify_assumption(_request(), asm_id, store)
    assert body["crystal_id"] == asm_id
    assert body["task_id"]

    task = await store.get_cognition_task(body["task_id"])
    assert task.task_type == VERIFICATION_TASK_TYPE
    assert task.payload["assumption_crystal_id"] == asm_id

    rows = await store.list_assumption_crystals(customer.id)
    row = next(r for r in rows if r["id"] == asm_id)
    assert has_verification_tag(row["diagnostic_tags"])

    events = await store.list_curation_events(customer.id)
    spawned = [
        e for e in events if e["event_type"] == "verification_spawned"
    ]
    assert len(spawned) == 1
    assert spawned[0]["payload"]["manual"] is True


async def test_manual_verify_refuses_non_assumption_and_invalidated(
    store, customer, semantic_encoder_stub,
):
    await _seed_crystal(store, "plain", customer.id, summary="a doc")
    with pytest.raises(HTTPException) as e1:
        await admin_verify_assumption(_request(), "plain", store)
    assert e1.value.status_code == 422

    asm_id = await _make_assumption(
        store, customer, semantic_encoder_stub, "x1", "x2", "dead one",
    )
    await store.set_crystal_quality_tier(asm_id, customer.id, "blacklist")
    with pytest.raises(HTTPException) as e2:
        await admin_verify_assumption(_request(), asm_id, store)
    assert e2.value.status_code == 422
