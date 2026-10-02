"""v110 respond-on-assumptions pins (Q24=A Sonnet, Q25=A thread state on
the crystals and chains, Q26=C supersede ancestors and replaces_parent
siblings, Q27=A delete one; ratified 2026-10-01).

The model is a fake with exactly the real client's method (complete_detailed
returning an LLMResult); the store, the chains and the tags are real.
"""
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from crystal_cache.infrastructure.metadata_store_assumption_ext import parse_assumption_tags
from crystal_cache.infrastructure.schema import CrystalRow, FactRow
from crystal_cache.llm import reset_llm_client, set_llm_client
from crystal_cache.llm.client import LLMResult


class _FakeRespondLLM:
    """Same surface as LLMClient for this path: is_ready + complete_detailed."""

    def __init__(self, proposals_by_call):
        self._queue = list(proposals_by_call)
        self.calls = []

    def is_ready(self):
        return True

    def complete_detailed(self, **kwargs):
        self.calls.append(kwargs)
        proposals = self._queue.pop(0)
        return LLMResult(
            text=json.dumps({"assumptions": proposals}), model="fake-sonnet",
            input_tokens=100, output_tokens=50,
        )


def _req(customer_id, encoder):
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(prompt_encoder=encoder)),
        state=SimpleNamespace(tenant_pin=customer_id),
    )


async def _seed_evidence(store, customer_id):
    async with store.session() as s:
        s.add(CrystalRow(id="ev_a", customer_id=customer_id, summary_vector=[],
                         summary_text="Invoices are sent on the 1st"))
        s.add(FactRow(id="evf_a", crystal_id="ev_a", claim_text="Invoices go out on the 1st"))
        s.add(CrystalRow(id="ev_b", customer_id=customer_id, summary_vector=[],
                         summary_text="Net-30 terms apply"))
        s.add(FactRow(id="evf_b", crystal_id="ev_b", claim_text="Payment terms are net 30"))


async def _root_assumption(store, customer_id, encoder):
    out = await store.create_assumption_crystal(
        customer_id, statement="Payments are due on the 31st", subject="billing",
        parent_a_id="ev_a", parent_b_id="ev_b", confidence=0.6, encoder=encoder,
    )
    return out["crystal_id"]


def test_parse_tags_round_trips_thread_state():
    parsed = parse_assumption_tags([
        "assumption_confidence:0.70", "assumption_thread:asm_root",
        "assumption_responds_to:asm_prev", "assumption_iteration:2",
        "assumption_replaces_parent:1", "assumption_superseded_by:asm_win",
    ])
    assert parsed["confidence"] == 0.7
    assert parsed["thread_id"] == "asm_root"
    assert parsed["responds_to"] == "asm_prev"
    assert parsed["iteration"] == 2
    assert parsed["replaces_parent"] is True
    assert parsed["superseded_by"] == "asm_win"
    assert parse_assumption_tags([])["replaces_parent"] is False


@pytest.mark.asyncio
async def test_respond_creates_revisions_in_a_thread(store, customer, semantic_encoder_stub):
    from crystal_cache.endpoints import admin as admin_mod

    await _seed_evidence(store, customer.id)
    root = await _root_assumption(store, customer.id, semantic_encoder_stub)
    fake = _FakeRespondLLM([[
        {"statement": "Payments are due 30 days after the invoice on the 1st",
         "subject": "billing", "confidence": 0.85,
         "rationale": "Operator clarified net-30 from the invoice date", "replaces_parent": True},
        {"statement": "Late payments accrue a 2% fee", "subject": "billing",
         "confidence": 0.6, "rationale": "Operator mentioned a late fee", "replaces_parent": False},
    ]])
    set_llm_client(fake)
    try:
        out = await admin_mod.admin_respond_assumption(
            _req(customer.id, semantic_encoder_stub), root,
            admin_mod.AssumptionRespondRequest(text="Due 30 days after the 1st; 2% late fee."),
            store,
        )
    finally:
        reset_llm_client()
    assert out["root_id"] == root
    assert len(out["created"]) == 2
    assert out["count"] == 3
    # The model saw the assumption, the evidence facts, and the response.
    sent = json.loads(fake.calls[0]["messages"][0]["content"])
    assert sent["current_assumption"]["statement"] == "Payments are due on the 31st"
    assert {e["id"] for e in sent["evidence"]} == {"ev_a", "ev_b"}
    assert sent["operator_response"].startswith("Due 30 days")
    assert fake.calls[0]["tier"] == "large"  # Q24=A

    thread = out["thread"]
    assert thread[0]["id"] == root
    revisions = thread[1:]
    assert all(r["thread_id"] == root for r in revisions)
    assert all(r["responds_to"] == root for r in revisions)
    assert all(r["iteration"] == 1 for r in revisions)
    assert all(r["operator_response"].startswith("Due 30 days") for r in revisions)
    assert all(r["recall_gated"] for r in revisions)  # still assumptions
    assert sorted(r["replaces_parent"] for r in revisions) == [False, True]
    # Lineage: each revision chains to the same evidence and to the root.
    for r in revisions:
        targets = {c.target_crystal_id for c in await store.list_chains_from_source(r["id"])}
        assert targets == {"ev_a", "ev_b", root}
    # Metered.
    from datetime import datetime, timezone

    rows = await store.cost_by_origin(customer.id, since=datetime(2000, 1, 1, tzinfo=timezone.utc))
    assert any(r["origin"] == "assumption_respond" for r in rows)


@pytest.mark.asyncio
async def test_approve_supersedes_ancestors_and_same_claim_siblings_only(
    store, customer, semantic_encoder_stub
):
    """Q26=C: accept A2 (a rewrite of A, which rewrote the root).
    Superseded: A and the root. Not superseded: B (an additional claim
    the first response implied) and C (a sibling that was additional)."""
    from crystal_cache.endpoints import admin as admin_mod

    await _seed_evidence(store, customer.id)
    root = await _root_assumption(store, customer.id, semantic_encoder_stub)
    fake = _FakeRespondLLM([
        [
            {"statement": "A: due net-30 from the 1st", "subject": "billing",
             "confidence": 0.8, "rationale": "r", "replaces_parent": True},
            {"statement": "B: 2% late fee", "subject": "billing",
             "confidence": 0.6, "rationale": "r", "replaces_parent": False},
        ],
        [
            {"statement": "A2: due on the 31st of the invoice month", "subject": "billing",
             "confidence": 0.9, "rationale": "r", "replaces_parent": True},
            {"statement": "C: invoices are emailed", "subject": "billing",
             "confidence": 0.5, "rationale": "r", "replaces_parent": False},
        ],
    ])
    set_llm_client(fake)
    try:
        first = await admin_mod.admin_respond_assumption(
            _req(customer.id, semantic_encoder_stub), root,
            admin_mod.AssumptionRespondRequest(text="net 30 and a late fee"), store,
        )
        a_id = next(m["id"] for m in first["thread"] if m["statement"].startswith("A:"))
        b_id = next(m["id"] for m in first["thread"] if m["statement"].startswith("B:"))
        second = await admin_mod.admin_respond_assumption(
            _req(customer.id, semantic_encoder_stub), a_id,
            admin_mod.AssumptionRespondRequest(text="actually the 31st"), store,
        )
        a2_id = next(m["id"] for m in second["thread"] if m["statement"].startswith("A2:"))
        c_id = next(m["id"] for m in second["thread"] if m["statement"].startswith("C:"))
    finally:
        reset_llm_client()
    assert next(m for m in second["thread"] if m["id"] == a2_id)["iteration"] == 2

    out = await admin_mod.admin_approve_assumption(
        _req(customer.id, semantic_encoder_stub), a2_id, store,
    )
    assert set(out["superseded"]) == {a_id, root}

    thread = await store.list_assumption_thread(customer.id, root)
    state = {m["id"]: m for m in thread}
    assert state[a2_id]["recall_gated"] is False  # accepted
    for sid in (a_id, root):
        assert state[sid]["quality_tier"] == "blacklist"
        assert state[sid]["superseded_by"] == a2_id
    for live in (b_id, c_id):
        assert state[live]["quality_tier"] != "blacklist"
        assert state[live]["superseded_by"] is None
        assert state[live]["recall_gated"] is True


@pytest.mark.asyncio
async def test_respond_refuses_superseded_and_non_assumptions(store, customer, semantic_encoder_stub):
    from crystal_cache.endpoints import admin as admin_mod

    await _seed_evidence(store, customer.id)
    root = await _root_assumption(store, customer.id, semantic_encoder_stub)
    await store.supersede_assumptions(customer.id, [root], by_id="asm_other")
    set_llm_client(_FakeRespondLLM([]))
    try:
        with pytest.raises(HTTPException) as e:
            await admin_mod.admin_respond_assumption(
                _req(customer.id, semantic_encoder_stub), root,
                admin_mod.AssumptionRespondRequest(text="x"), store,
            )
        assert e.value.status_code == 422
        with pytest.raises(HTTPException) as e:
            await admin_mod.admin_respond_assumption(
                _req(customer.id, semantic_encoder_stub), "ev_a",
                admin_mod.AssumptionRespondRequest(text="x"), store,
            )
        assert e.value.status_code == 422
    finally:
        reset_llm_client()


@pytest.mark.asyncio
async def test_thread_read_and_cross_tenant_404(store, customer, semantic_encoder_stub):
    from crystal_cache.endpoints import admin as admin_mod

    await _seed_evidence(store, customer.id)
    root = await _root_assumption(store, customer.id, semantic_encoder_stub)
    out = await admin_mod.admin_assumption_thread(_req(customer.id, semantic_encoder_stub), root, store)
    assert out["root_id"] == root and out["count"] == 1
    with pytest.raises(HTTPException) as e:
        await admin_mod.admin_assumption_thread(_req("cus_stranger", semantic_encoder_stub), root, store)
    assert e.value.status_code == 404
