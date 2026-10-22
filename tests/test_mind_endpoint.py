"""The Mind tab's read (2026-10-21): regions and positions from the facts'
own vectors, every kind of thought the system has, what it did in the
window, and how hard each process worked. One route, one read."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from crystal_cache.infrastructure.schema import CrystalRow, FactRow
from crystal_cache.mind.layout import layout_facts
from crystal_cache.models.query_log import QueryLog

_T0 = datetime.now(timezone.utc) - timedelta(hours=2)


def _app(store, encoder, vector_store, fact_vector_store):
    try:
        from tests.test_endpoint_smoke import _build_app
    except ModuleNotFoundError:
        from test_endpoint_smoke import _build_app
    return _build_app(store, encoder, vector_store, fact_vector_store)


def _vec(group: int, i: int) -> list[float]:
    rng = np.random.default_rng(group * 100 + i)
    v = np.zeros(12); v[group * 4:(group + 1) * 4] = 1.0
    return (v + rng.normal(0, 0.15, 12)).tolist()


def test_layout_finds_regions_names_them_by_key_and_is_deterministic():
    vecs, keys = [], []
    for g, name in enumerate(["Pricing", "Launch", "Customers"]):
        for i in range(20):
            vecs.append(_vec(g, i)); keys.append(f"{name}|item {i}|detail")
    vecs.append([]); keys.append("Empty|x")  # a fact without a vector still gets a place
    a = layout_facts(vecs, keys, k=3)
    b = layout_facts(vecs, keys, k=3)
    assert a.positions.shape == (61, 3)
    assert np.allclose(a.positions, b.positions) and (a.region_of == b.region_of).all()
    names = sorted(r["name"] for r in a.regions)
    assert names == ["Customers", "Launch", "Pricing"]
    # Each real group lands in one region.
    for g in range(3):
        assert len(set(a.region_of[g * 20:(g + 1) * 20].tolist())) == 1
    # Regions are pushed apart.
    c = np.array([r["center"] for r in a.regions])
    for i in range(3):
        for j in range(i + 1, 3):
            assert np.linalg.norm(c[i] - c[j]) >= 27.9
    assert layout_facts([], []).positions.shape == (0, 3)
    one = layout_facts([[1.0, 2.0, 3.0, 4.0]], ["Solo|a"])
    assert one.regions[0]["name"] == "Solo" and one.regions[0]["count"] == 1


async def _seed(store, customer_id, doc_id):
    async with store.session() as s:
        for cid, tier, path in [("cP", "whitelist", "Kerning: pricing"), ("cL", "quarantine", "Founder journal"), ("cC", "neutral", None)]:
            s.add(CrystalRow(id=cid, customer_id=customer_id, crystal_type="customer:legacy", summary_vector=[],
                             quality_tier=tier, source_path=path, summary_text=None))
        s.add(CrystalRow(id="asm", customer_id=customer_id, crystal_type="assumption", summary_vector=[],
                         quality_tier="quarantine", summary_text="bridge"))
        for g, (cid, name) in enumerate([("cP", "Pricing"), ("cL", "Launch"), ("cC", "Customers")]):
            for i in range(8):
                s.add(FactRow(id=f"f_{cid}_{i}", crystal_id=cid, pair_type="question_answer",
                              prompt_text=f"{name}|item {i}", claim_text=f"{name} claim {i}",
                              source_kind="document_chunk", source_doc_id=doc_id if g == 0 else None,
                              vector=_vec(g, i), hit_count=i, created_at=_T0 + timedelta(minutes=i)))
        s.add(FactRow(id="chunk", crystal_id="cP", pair_type="content_chunk", prompt_text="Docs|Pricing",
                      claim_text="## Pricing ...", source_kind="document_chunk", vector=_vec(0, 99), created_at=_T0))
        s.add(FactRow(id="asm_fact", crystal_id="asm", pair_type="question_answer", prompt_text="Assumptions|x",
                      claim_text="bridge", source_kind="agent_inferred", vector=_vec(1, 99), created_at=_T0))


@pytest.mark.asyncio
async def test_mind_route_serves_the_scene_in_one_read(store, customer, semantic_encoder_stub, vector_store, fact_vector_store):
    from httpx import ASGITransport, AsyncClient

    doc = await store.create_document_upload(customer.id, "Kerning: product brief", "text")
    await _seed(store, customer.id, doc.id)
    conflict = await store.create_knowledge_conflict(
        customer.id, fact_a_id="f_cP_0", fact_b_id="f_cP_1", claim_a="Pricing claim 0", claim_b="Pricing claim 1",
        pair_key="pk1", crystal_a_id="cP", crystal_b_id="cP", subject="Pricing",
        reasoning="two prices", recommendation={"resolution": "superseded", "loser": "a", "reason": "newer"},
    )
    await store.create_knowledge_conflict(
        customer.id, fact_a_id="f_cC_0", fact_b_id="f_cC_1", claim_a="a", claim_b="b", pair_key="pk2",
        crystal_a_id="cC", crystal_b_id="cC", detector="dedup_scan",
    )
    await store.create_assumption_crystal(
        customer.id, statement="Pennant will outgrow three seats", subject="Pennant", parent_a_id="cP",
        parent_b_id="cC", confidence=0.74, encoder=semantic_encoder_stub,
        extra_tags=["assumption_rationale:A says x; B says y; so z"],
    )
    await store.create_knowledge_gap(customer.id, domain="Pricing", subject="Pricing", missing="EU VAT?", full_key="Pricing|VAT")
    task = await store.create_cognition_task(customer.id, task_type="research", payload={"topic": "token format"})
    await store.write_query_log(QueryLog(
        id="q1", customer_id=customer.id, query_text="what does Pro cost?", match_type="high",
        injection_method="agent_tools", matched_facts=["f_cP_2", "f_cP_3"], client="Claude Desktop",
        model="nvidia/nemotron-3-super-120b-a12b", latency_ms=900, timestamp=datetime.now(timezone.utc) - timedelta(minutes=30),
    ))
    await store.record_llm_call(customer.id, model="m", input_tokens=100, output_tokens=20, origin="scan_contradiction")
    await store.record_llm_call(customer.id, model="m", input_tokens=300, output_tokens=40, origin="agent")
    await store.record_curation_event(customer.id, event_type="conflict_found", subject_id=conflict.id, label="Contradiction found - Pricing")

    app = _app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as cl:
        r = await cl.get(f"/admin/api/customers/{customer.id}/mind")
        assert r.status_code == 200, r.text[:400]
        body = r.json()

    # Claims only: the passage and the assumption's own fact stay out.
    ids = {f["id"] for f in body["facts"]}
    assert "chunk" not in ids and "asm_fact" not in ids and len(ids) == 24
    assert body["totals"]["facts"] == 24 and body["totals"]["regions"] == 3
    names = sorted(r["name"] for r in body["regions"])
    assert names == ["Customers", "Launch", "Pricing"]
    pricing = next(r for r in body["regions"] if r["name"] == "Pricing")
    assert pricing["facts"] == 8 and pricing["trusted"] == 8 and pricing["clusters"] == 1
    f0 = next(f for f in body["facts"] if f["id"] == "f_cP_0")
    assert f0["tier"] == "trusted" and f0["region"] == pricing["id"] and len(f0["pos"]) == 3
    # The brief fed Pricing.
    assert body["sources"] == [{"id": doc.id, "label": "Kerning: product brief", "region": pricing["id"], "facts": 8,
                                "learned_at": body["sources"][0]["learned_at"]}]
    # Thoughts: both kinds of conflict carry their sides' positions; the assumption its parents and rationale; the gap its region.
    kinds = sorted(c["kind"] for c in body["conflicts"])
    assert kinds == ["contradiction", "duplicate"]
    c = next(c for c in body["conflicts"] if c["kind"] == "contradiction")
    assert c["a"]["pos"] and c["b"]["pos"] and c["reasoning"] == "two prices" and c["recommendation"]["loser"] == "a"
    a = body["assumptions"][0]
    assert a["confidence"] == 0.74 and a["rationale"].startswith("A says x") and a["state"] == "waiting"
    assert sorted(p["crystal_id"] for p in a["parents"]) == ["cC", "cP"] and all(p["pos"] for p in a["parents"])
    assert body["gaps"][0]["missing"] == "EU VAT?" and body["gaps"][0]["region"] == pricing["id"]
    # What it did: the run, the recall with the facts it touched, the event, the lanes.
    assert body["runs"][0]["id"] == task.id and body["runs"][0]["title"] == "token format"
    assert body["recalls"][0]["question"] == "what does Pro cost?" and body["recalls"][0]["region"] == pricing["id"]
    assert body["recalls"][0]["touched"] == ["f_cP_2", "f_cP_3"]
    assert body["events"][0]["type"] == "conflict_found"
    lanes = {l["name"]: l for l in body["lanes"]}
    assert sum(h["calls"] for h in lanes["Contradictions"]["hours"]) == 1
    assert sum(h["tokens"] for h in lanes["Recalls"]["hours"]) == 340
    assert "cost_micro_usd" not in r.text


@pytest.mark.asyncio
async def test_mind_route_on_an_empty_bank(store, customer, semantic_encoder_stub, vector_store, fact_vector_store):
    from httpx import ASGITransport, AsyncClient

    app = _app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as cl:
        r = await cl.get(f"/admin/api/customers/{customer.id}/mind", params={"hours": 6})
    assert r.status_code == 200
    body = r.json()
    assert body["facts"] == [] and body["regions"] == [] and body["window"]["hours"] == 6
    assert all(len(l["hours"]) == 6 for l in body["lanes"])
