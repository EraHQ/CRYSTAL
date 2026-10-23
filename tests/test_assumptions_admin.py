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
    # 2026-10-21: every parent carries a label and its claims in text.
    assert sorted(p["label"] for p in item["parents"]) == ["about cr_a", "about cr_b"]
    assert all(p["claims"] == [] for p in item["parents"])


async def test_parent_without_a_summary_is_named_by_its_file_and_quoted(
    store, customer, semantic_encoder_stub,
):
    """A bonded document cluster has no summary; the chip used to show
    a bare crystal id. Now: the source file (or the first key) as the
    label, and up to three claims the page can quote."""
    from crystal_cache.infrastructure.schema import FactRow

    await _seed_crystal(store, "cr_doc", customer.id, summary=None)
    await _seed_crystal(store, "cr_b", customer.id, summary="about cr_b")
    async with store.session() as s:
        row = await s.get(CrystalRow, "cr_doc")
        row.source_path = "Kerning_Code/adr/0002-pro-price-15.md"
        s.add(FactRow(
            id="f_chunk", crystal_id="cr_doc", pair_type="content_chunk",
            prompt_text="Docs|ADR 0002", claim_text="# ADR 0002 ...", source_kind="document_chunk",
            vector=[],
        ))
        for i in range(4):
            s.add(FactRow(
                id=f"f_{i}", crystal_id="cr_doc", pair_type="question_answer",
                prompt_text=f"ADR 0002|Pricing|point {i}", claim_text=f"claim number {i}",
                source_kind="document_chunk", vector=[],
            ))
    await store.create_assumption_crystal(
        customer.id, statement="bridge", subject="Pricing", parent_a_id="cr_doc",
        parent_b_id="cr_b", confidence=0.7, encoder=semantic_encoder_stub,
    )
    body = await admin_list_assumptions(_request(), store, customer_id=customer.id)
    doc = next(p for p in body["assumptions"][0]["parents"] if p["id"] == "cr_doc")
    assert doc["label"] == "Kerning_Code/adr/0002-pro-price-15.md"
    assert len(doc["claims"]) == 3
    assert all(c.startswith("claim number ") for c in doc["claims"])
    assert "# ADR 0002" not in " ".join(doc["claims"])


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
    # C2 Q1=A (2026-08-08): the approve response gained gap_filled, the
    # seeding-gap closure result (None here: this fixture's gap_id has no
    # real gap row, so the guarded close correctly declines).
    # v110 (Q26=C, 2026-10-01): it also reports what the approval
    # superseded in its respond thread; a lone assumption supersedes
    # nothing.
    assert out == {
        "crystal_id": asm_id,
        "recall_gated": False,
        # Q99 C (2026-10-23): the approve act reports the standing it
        # left; a two-parent assumption keeps its tier (a signal).
        "quality_tier": "quarantine",
        "kind": "assumption",
        "gap_filled": None,
        "superseded": [],
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
