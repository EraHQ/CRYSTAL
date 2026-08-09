"""Assumptions slice 5 — the Inspector admin surface
(endpoints/admin.py: admin_list_assumptions / admin_approve_assumption).

Direct-call convention (the test_promotion_api.py precedent) with a
FakeRequest carrying request.state.tenant_pin: the list read's field
shape, tag parsing (confidence / gap provenance / invalidated
parents), live-parent hydration via chains, the tenant-pin override,
approve clearing the recall gate while leaving the tier untouched,
and approve refusing invalidated (blacklist) and non-assumption
crystals. The store read itself is covered here too via the endpoint
path; the delete action reuses the long-standing crystal DELETE
route covered by test_crystal_delete.py.

R14 note: verified by `pytest`; the store-read shape ran green in the
container rig at authoring time (2026-08-05).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from crystal_cache.endpoints.admin import (
    admin_approve_assumption,
    admin_list_assumptions,
)
from crystal_cache.infrastructure.schema import CrystalRow


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


async def _make_assumption(store, customer, encoder, a="cr_a", b="cr_b"):
    await _seed_crystal(store, a, customer.id, summary=f"about {a}")
    await _seed_crystal(store, b, customer.id, summary=f"about {b}")
    written = await store.create_assumption_crystal(
        customer.id,
        statement=f"bridge over {a}+{b}",
        subject="Bridged subject",
        parent_a_id=a,
        parent_b_id=b,
        confidence=0.8,
        encoder=encoder,
        gap_id="gap_seed_1",
    )
    return written["crystal_id"]


async def test_list_shape_parents_and_parsed_tags(
    store, customer, semantic_encoder_stub,
):
    asm_id = await _make_assumption(store, customer,
                                    semantic_encoder_stub)

    body = await admin_list_assumptions(
        _request(), store, customer_id=customer.id,
    )
    assert body["count"] == 1
    item = body["assumptions"][0]
    assert item["id"] == asm_id
    assert item["statement"] == "bridge over cr_a+cr_b"
    assert item["quality_tier"] == "quarantine"
    assert item["recall_gated"] is True
    assert item["confidence"] == 0.8
    assert item["gap_id"] == "gap_seed_1"
    assert item["invalidated_parents"] == []
    assert sorted(p["id"] for p in item["parents"]) == ["cr_a", "cr_b"]
    assert all(p["summary_text"] for p in item["parents"])


async def test_list_shows_invalidation_state(
    store, customer, semantic_encoder_stub,
):
    await _make_assumption(store, customer, semantic_encoder_stub)
    assert await store.delete_crystal("cr_b", customer.id) is True

    body = await admin_list_assumptions(
        _request(), store, customer_id=customer.id,
    )
    item = body["assumptions"][0]
    assert item["quality_tier"] == "blacklist"
    assert item["invalidated_parents"] == ["cr_b"]
    # capture-at-delete removed the dead edge: only the live parent
    # hydrates.
    assert [p["id"] for p in item["parents"]] == ["cr_a"]


async def test_tenant_pin_overrides_customer_param(
    store, customer, semantic_encoder_stub,
):
    await _make_assumption(store, customer, semantic_encoder_stub)
    foreign = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="sk-other",
    )
    # A pinned tenant asking for someone else's list gets THEIR OWN.
    body = await admin_list_assumptions(
        _request(tenant_pin=foreign.id), store, customer_id=customer.id,
    )
    assert body["count"] == 0


async def test_approve_clears_gate_and_leaves_tier(
    store, customer, semantic_encoder_stub,
):
    asm_id = await _make_assumption(store, customer,
                                    semantic_encoder_stub)

    out = await admin_approve_assumption(_request(), asm_id, store)
    # C2 Q1=A (2026-08-08): the approve response gained gap_filled —
    # the seeding-gap closure result (None here: this fixture's gap_id
    # has no real gap row, so the guarded close correctly declines).
    assert out == {
        "crystal_id": asm_id,
        "recall_gated": False,
        "gap_filled": None,
    }

    crystal = await store.get_crystal(asm_id)
    assert crystal.recall_gated is False
    assert crystal.quality_tier == "quarantine"     # tier untouched


async def test_approve_refuses_invalidated(
    store, customer, semantic_encoder_stub,
):
    asm_id = await _make_assumption(store, customer,
                                    semantic_encoder_stub)
    await store.delete_crystal("cr_a", customer.id)

    with pytest.raises(HTTPException) as exc:
        await admin_approve_assumption(_request(), asm_id, store)
    assert exc.value.status_code == 422
    crystal = await store.get_crystal(asm_id)
    assert crystal.recall_gated is True             # gate untouched


async def test_approve_refuses_non_assumption_and_foreign(
    store, customer, semantic_encoder_stub,
):
    await _seed_crystal(store, "cr_plain", customer.id, summary="plain")
    with pytest.raises(HTTPException) as exc:
        await admin_approve_assumption(_request(), "cr_plain", store)
    assert exc.value.status_code == 422

    asm_id = await _make_assumption(store, customer,
                                    semantic_encoder_stub)
    foreign = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="sk-other",
    )
    # Pinned to the foreign tenant: the same 404 as missing — never an
    # existence oracle.
    with pytest.raises(HTTPException) as exc:
        await admin_approve_assumption(
            _request(tenant_pin=foreign.id), asm_id, store,
        )
    assert exc.value.status_code == 404
