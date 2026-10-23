"""Q99 C + the first approval rule (2026-10-23): consequences as knowledge.

Anthony: "an implication and an assumption are the same thing... tie the
consequences between the new assumption and old facts. We also tie it to
every other fact it checked. Knowing something doesn't contradict is
knowledge in itself." A consequence the checker kept becomes an
assumption crystal of kind "consequence" with ONE parent fact and a typed
edge to every fact it was judged against; it enters recall at the first
consistent_with edge (once its parent can be recalled) and is approved by
a person, a rule, or the parent. The rule: "a fact needs enough
consistent_with edges to auto approve."
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException
from sqlalchemy import select

from crystal_cache.infrastructure.metadata_store_assumption_ext import (
    CHECK_EDGE_TYPES, assumption_state, parse_assumption_tags,
)
from crystal_cache.infrastructure.schema import CrystalEdgeRow, CrystalRow, FactRow
from crystal_cache.retrieval.tier_signal import assumption_note
from crystal_cache.scan import implication as mod
from crystal_cache.system_rules import RuleValidationError, validate_rule
from crystal_cache.system_rules.model import SystemRule
from crystal_cache.system_rules.store import create_rule, run_approval_rules
from fakes import SeamFake

_T0 = datetime.now(timezone.utc) - timedelta(hours=1)
D = 8


def _vec(*on: int) -> list[float]:
    v = np.zeros(D, dtype=np.float32)
    for i in on:
        v[i] = 1.0
    return v.tolist()


class _Encoder:
    """The oxygen consequence lands on BOTH the water fact (which contradicts
    it) and the steam fact (which agrees); the pricing consequence lands on
    the invoice fact (agrees)."""

    table = {
        "No liquid substance contains oxygen": _vec(3),
        "Pro is billed every month": _vec(5),
    }

    def encode_native(self, text: str) -> np.ndarray:
        return np.asarray(self.table.get(text, _vec(7)), dtype=np.float32)

    def encode(self, text: str) -> np.ndarray:
        return self.encode_native(text)

    def fingerprint(self) -> str:
        return "stub"


class _Scripted(SeamFake):
    def __init__(self):
        self.judges = []

    def complete(self, **kwargs):
        system = kwargs.get("system") or ""
        user = kwargs["messages"][0]["content"]
        if system.startswith("You read ONE stored fact"):
            if "Oxygen only exists" in user:
                return json.dumps({"consequences": ["No liquid substance contains oxygen"], "questions": []})
            if "Pro is $15" in user:
                return json.dumps({"consequences": ["Pro is billed every month"], "questions": []})
            return json.dumps({"consequences": [], "questions": []})
        if system.startswith("You grade whether each proposed CONSEQUENCE"):
            grades = []
            for line in user.split("CONSEQUENCES:")[-1].strip().splitlines():
                grades.append({"consequence": line.split(". ", 1)[-1].strip(), "grade": "FOLLOWS"})
            return json.dumps({"grades": grades})
        self.judges.append(user)
        if "Water is made of" in user:
            return json.dumps({"reason": "Water is a liquid that contains oxygen.", "verdict": "CONTRADICTS"})
        if "Steam is" in user:
            return json.dumps({"reason": "Steam is a gas.", "verdict": "CONSISTENT"})
        if "invoice" in user.lower():
            return json.dumps({"reason": "A monthly invoice is a monthly charge.", "verdict": "CONSISTENT"})
        return json.dumps({"reason": "No bearing.", "verdict": "UNRELATED"})


async def _seed(store, customer_id, *, parent_tier="neutral"):
    async with store.session() as s:
        s.add(CrystalRow(id="c_chem", customer_id=customer_id, crystal_type="customer:legacy", summary_vector=[], quality_tier="neutral"))
        s.add(CrystalRow(id="c_oxy", customer_id=customer_id, crystal_type="customer:legacy", summary_vector=[], quality_tier=parent_tier))
        s.add(CrystalRow(id="c_price", customer_id=customer_id, crystal_type="customer:legacy", summary_vector=[], quality_tier="whitelist"))
        s.add(FactRow(id="f_water", crystal_id="c_chem", pair_type="question_answer", prompt_text="Chemistry|Water|Composition",
                      claim_text="Water is made of two hydrogen and one oxygen.", source_kind="document_chunk",
                      source_doc_id="doc_chem", vector=_vec(3), created_at=_T0))
        s.add(FactRow(id="f_steam", crystal_id="c_chem", pair_type="question_answer", prompt_text="Chemistry|Steam|State",
                      claim_text="Steam is water as a gas.", source_kind="document_chunk",
                      source_doc_id="doc_chem2", vector=_vec(3, 4), created_at=_T0))
        s.add(FactRow(id="f_invoice", crystal_id="c_price", pair_type="question_answer", prompt_text="Pricing|Pro|Invoice",
                      claim_text="Pro customers get an invoice every month.", source_kind="document_chunk",
                      source_doc_id="doc_inv", vector=_vec(5), created_at=_T0))
        s.add(FactRow(id="f_oxygen", crystal_id="c_oxy", pair_type="question_answer", prompt_text="Chemistry|Oxygen|States",
                      claim_text="Oxygen only exists as a gas.", source_kind="model_reasoning",
                      vector=_vec(1), created_at=_T0 + timedelta(minutes=2)))
        s.add(FactRow(id="f_pro", crystal_id="c_price", pair_type="question_answer", prompt_text="Pricing|Pro|Price",
                      claim_text="Pro is $15 since Oct 1.", source_kind="document_chunk",
                      source_doc_id="doc_price", vector=_vec(2), created_at=_T0 + timedelta(minutes=3)))


async def _edges(store, cid):
    async with store.session() as s:
        rows = (await s.execute(select(CrystalEdgeRow).where(
            (CrystalEdgeRow.crystal_a_id == cid) | (CrystalEdgeRow.crystal_b_id == cid)))).scalars().all()
        return sorted((r.edge_type, r.crystal_a_id if r.crystal_a_id != cid else r.crystal_b_id) for r in rows)


async def _scan(store, customer, llm=None, **kw):
    """One fact per call, the counters summed. The rig's in-memory SQLite
    hands every session the same connection, so two probes' writes
    interleave and one probe's rollback can take the other's rows with it
    (Windows and CI both lost a fact row and a tag that way). A single
    probe per call keeps every write sequential; what this file pins is
    the bookkeeping. Concurrency is tests/test_implication_scan.py's."""
    llm = llm or _Scripted()
    total = None
    while True:
        res = await mod.scan_by_implication(store=store, encoder=_Encoder(), customer_id=customer.id,
                                            llm_client=llm, max_facts=1, hits_per_fact=3, min_score=0.45, **kw)
        if total is None:
            total = res
        else:
            for f in ("facts_probed", "checks", "dropped", "searches", "hits", "conflicts_found", "memoized",
                      "consequences_written", "edges_written", "entered_recall"):
                setattr(total, f, getattr(total, f) + getattr(res, f))
            total.skipped_probed = res.skipped_probed
            total.budget_exhausted = res.budget_exhausted
        if res.facts_probed == 0:
            break
    return total, llm


@pytest.mark.asyncio
async def test_a_kept_consequence_becomes_knowledge_tied_to_its_parent_and_every_check(store, customer):
    await _seed(store, customer.id)
    res, llm = await _scan(store, customer)
    assert res.consequences_written == 2 and res.edges_written == 3 and res.conflicts_found == 1
    rows = {r["statement"]: r for r in await store.list_assumption_crystals(customer.id)}
    oxy = rows["No liquid substance contains oxygen"]
    assert oxy["kind"] == "consequence" and oxy["build_method"] == "consequence"
    tags = parse_assumption_tags(oxy["diagnostic_tags"])
    assert tags["parent_fact_id"] == "f_oxygen"
    assert sorted((c["fact_id"], c["verdict"]) for c in tags["checks"]) == [("f_steam", "CONSISTENT"), ("f_water", "CONTRADICTS")]
    assert tags["consistent_edges"] == 1 and tags["contradicted_edges"] == 1
    # One parent, by chain and by parent_crystal_id; typed edges in crystal_edges.
    chains = await store.list_chains_from_source(oxy["id"])
    assert [c.target_crystal_id for c in chains] == ["c_oxy"] and oxy["parent_crystal_id"] == "c_oxy"
    assert await _edges(store, oxy["id"]) == [("consistent_with", "c_chem"), ("contradicted_by", "c_chem")]
    assert await store.count_consequence_edges(oxy["id"]) == {"consistent_with": 1, "unrelated_to": 0, "contradicted_by": 1}
    # The consequence's own fact is filed under Consequences|<subject>.
    facts = await store.list_facts_for_crystal(oxy["id"])
    assert len(facts) == 1 and facts[0].prompt_text == "Consequences|Chemistry" and facts[0].source_kind == "agent_inferred"
    # Standing: one agreeing fact opened the gate (the parent is recallable), but nobody approved it.
    assert oxy["recall_gated"] is False and oxy["quality_tier"] == "quarantine" and assumption_state(oxy) == "waiting"
    assert res.entered_recall == 2
    pro = rows["Pro is billed every month"]
    assert parse_assumption_tags(pro["diagnostic_tags"])["checks"] == [{"fact_id": "f_invoice", "verdict": "CONSISTENT"}]
    # The feed saw it kept and entering recall; the conflict row is still there.
    events = [e["event_type"] for e in await store.list_curation_events(customer.id, limit=50)]
    assert events.count("consequence_written") == 2 and events.count("consequence_recallable") == 2 and "conflict_found" in events
    conflicts = await store.list_knowledge_conflicts(customer.id, status="open")
    assert len(conflicts) == 1 and {conflicts[0].fact_a_id, conflicts[0].fact_b_id} == {"f_oxygen", "f_water"}
    # A second scan probes nothing new (memo) and never probes a consequence's own fact.
    again, _ = await _scan(store, customer, llm)
    assert again.facts_considered == 5 and again.facts_probed == 0 and again.consequences_written == 0
    # A re-probe of the same fact would not double the consequence.
    assert await store.find_consequence_crystal(customer.id, parent_fact_id="f_oxygen", statement="No liquid substance contains oxygen") == oxy["id"]


@pytest.mark.asyncio
async def test_the_gate_waits_for_a_gated_parent(store, customer):
    await _seed(store, customer.id)
    await store.set_crystal_recall_gate("c_oxy", customer.id, True)
    res, _ = await _scan(store, customer)
    rows = {r["statement"]: r for r in await store.list_assumption_crystals(customer.id)}
    oxy = rows["No liquid substance contains oxygen"]
    assert oxy["recall_gated"] is True and res.entered_recall == 1  # only the pricing one opened
    # Once the parent is let in, the next consistent check opens the consequence too.
    await store.set_crystal_recall_gate("c_oxy", customer.id, False)
    out = await store.add_consequence_check(customer.id, oxy["id"], fact_id="f_steam", fact_crystal_id="c_chem", verdict="CONSISTENT")
    assert out == {"edge_type": "consistent_with", "recallable": True, "opened": True}
    with pytest.raises(ValueError):
        await store.add_consequence_check(customer.id, oxy["id"], fact_id="f_steam", fact_crystal_id="c_chem", verdict="MAYBE")


@pytest.mark.asyncio
async def test_approval_takes_the_parents_standing_by_a_person_or_a_rule(store, customer):
    await _seed(store, customer.id)
    await _scan(store, customer)
    rows = {r["statement"]: r for r in await store.list_assumption_crystals(customer.id)}
    pro, oxy = rows["Pro is billed every month"], rows["No liquid substance contains oxygen"]
    assert {r["id"] for r in await store.list_unapproved_assumptions(customer.id)} == {pro["id"], oxy["id"]}

    # A person approves the oxygen one: it takes its parent's tier (neutral) and remembers who.
    from crystal_cache.endpoints.admin import admin_approve_assumption
    out = await admin_approve_assumption(SimpleNamespace(state=SimpleNamespace(tenant_pin=customer.id)), oxy["id"], store)
    assert out["kind"] == "consequence" and out["quality_tier"] == "neutral" and out["recall_gated"] is False
    got = next(r for r in await store.list_assumption_crystals(customer.id) if r["id"] == oxy["id"])
    assert assumption_state(got) == "approved" and parse_assumption_tags(got["diagnostic_tags"])["approved_by"] == "human"

    # No rule: nothing else is approved by the system.
    assert await run_approval_rules(store, customer.id) == {"approved": 0, "rules_fired": 0}
    # A rule that wants two agreeing facts does not fire on one.
    strict = await create_rule(store, customer.id, "approval", "Two must agree",
                               selector={"kind": "consequence"}, conditions={"min_consistent_edges": 2}, action={"approve": True})
    assert await run_approval_rules(store, customer.id) == {"approved": 0, "rules_fired": 0}
    # One agreeing fact is enough for this rule: the pricing consequence takes its whitelisted parent's standing.
    loose = await create_rule(store, customer.id, "approval", "One agrees",
                              selector={"kind": "consequence"}, conditions={"min_consistent_edges": 1}, action={"approve": True})
    assert await run_approval_rules(store, customer.id) == {"approved": 1, "rules_fired": 1}
    got = next(r for r in await store.list_assumption_crystals(customer.id) if r["id"] == pro["id"])
    assert assumption_state(got) == "approved" and got["quality_tier"] == "whitelist"
    assert parse_assumption_tags(got["diagnostic_tags"])["approved_by"] == f"rule:{loose.id}"
    labels = [e["label"] for e in await store.list_curation_events(customer.id, limit=50) if e["event_type"] == "assumption_approved"]
    assert any(l.startswith("Approved by rule One agrees - Pro is billed") for l in labels)
    # Audited on the rule; idempotent on the next pass.
    from crystal_cache.system_rules.store import list_rules
    fired = {r.id: r.fire_count for r in await list_rules(store, customer.id)}
    assert fired == {strict.id: 0, loose.id: 1}
    assert await run_approval_rules(store, customer.id) == {"approved": 0, "rules_fired": 0}


@pytest.mark.asyncio
async def test_a_contradicted_consequence_is_never_auto_approved(store, customer):
    await _seed(store, customer.id)
    await _scan(store, customer)
    await create_rule(store, customer.id, "approval", "One agrees",
                      selector={"kind": "consequence"}, conditions={"min_consistent_edges": 1}, action={"approve": True})
    out = await run_approval_rules(store, customer.id)
    assert out["approved"] == 1  # pricing only; the oxygen one has a contradicted_by edge and an open conflict
    states = {r["statement"]: assumption_state(r) for r in await store.list_assumption_crystals(customer.id)}
    assert states == {"Pro is billed every month": "approved", "No liquid substance contains oxygen": "waiting"}


def test_the_approval_rule_is_strict():
    def rule(**over):
        base = dict(id="r", customer_id="c", rule_type="approval", name="n",
                    selector={"kind": "consequence"}, conditions={"min_consistent_edges": 2}, action={"approve": True})
        base.update(over)
        return SystemRule(**base)
    validate_rule(rule())
    validate_rule(rule(conditions={"min_consistent_edges": 1, "no_contradicted_edges": False}))
    for bad in (
        dict(selector={"kind": "fact"}),
        dict(selector={}),
        dict(selector={"kind": "consequence", "origin": "x"}),
        dict(conditions={"min_consistent_edges": 0}),
        dict(conditions={"min_consistent_edges": True}),
        dict(conditions={"min_consistent": 2}),
        dict(conditions={}),
        dict(action={"approve": False}),
        dict(action={"approve": True, "set_tier": "trusted"}),
    ):
        with pytest.raises(RuleValidationError):
            validate_rule(rule(**bad))


@pytest.mark.asyncio
async def test_rules_routes_are_strict_tenant_pathed_settings(monkeypatch, store):
    try:
        from tests.test_accounts_phase_c import _AuthedReq, _use
    except ModuleNotFoundError:
        from test_accounts_phase_c import _AuthedReq, _use
    from crystal_cache.endpoints.customers import (
        create_customer_rule, delete_customer_rule, list_customer_rules, update_customer_rule,
    )
    _use(monkeypatch)
    c = await store.create_customer(provider="anthropic", model_id="m", api_key_ref="")
    bearer = f"Bearer {c.api_key}"
    with pytest.raises(HTTPException) as e:
        await create_customer_rule(c.id, _AuthedReq(bearer, body={"rule_type": "approval", "name": "x", "selector": {"kind": "consequence"},
                                                                "conditions": {"min_consistent_edges": 2, "typo": 1}, "action": {"approve": True}}), store)
    assert e.value.status_code == 400 and e.value.detail["error"]["code"] == "rule_invalid" and "typo" in e.value.detail["error"]["message"]
    r = await create_customer_rule(c.id, _AuthedReq(bearer, body={"rule_type": "approval", "name": "Two agree", "selector": {"kind": "consequence"},
                                                                 "conditions": {"min_consistent_edges": 2}, "action": {"approve": True}}), store)
    rule = json.loads(r.body)["rule"]
    assert r.status_code == 201 and rule["rule_type"] == "approval" and rule["enabled"] and rule["fire_count"] == 0
    # Q101 (2026-10-23): the list folds the older shape into a curation rule
    # (same id, name and switch) and carries the sentence.
    body = json.loads((await list_customer_rules(c.id, _AuthedReq(bearer), store)).body)
    assert body["count"] == 1 and body["rules"][0]["id"] == rule["id"] and body["rules"][0]["rule_type"] == "curation"
    assert body["rules"][0]["text"] == "Approve a consequence when 2 stated facts agree with it and nothing contradicts it."
    r = await update_customer_rule(c.id, rule["id"], _AuthedReq(bearer, body={"enabled": False, "conditions": {"min_agreeing_facts": 3}}), store)
    assert json.loads(r.body)["rule"]["enabled"] is False and json.loads(r.body)["rule"]["conditions"] == {"min_agreeing_facts": 3}
    with pytest.raises(HTTPException) as e:
        await update_customer_rule(c.id, rule["id"], _AuthedReq(bearer, body={"conditions": {"min_consistent_edges": 3}}), store)
    assert e.value.detail["error"]["code"] == "rule_invalid"
    # Another tenant's key cannot see or touch it.
    other = await store.create_customer(provider="anthropic", model_id="m", api_key_ref="")
    with pytest.raises(HTTPException) as e:
        await list_customer_rules(c.id, _AuthedReq(f"Bearer {other.api_key}"), store)
    assert e.value.status_code in (401, 403, 404)  # never an existence oracle
    assert json.loads((await delete_customer_rule(c.id, rule["id"], _AuthedReq(bearer), store)).body) == {"deleted": True, "rule_id": rule["id"]}
    with pytest.raises(HTTPException) as e:
        await delete_customer_rule(c.id, rule["id"], _AuthedReq(bearer), store)
    assert e.value.status_code == 404


@pytest.mark.asyncio
async def test_check_edges_never_become_pairing_candidates(store, customer):
    from crystal_cache.scan.assumptions import run_assumptions_scan
    await _seed(store, customer.id)
    await _scan(store, customer)
    calls = []

    class _Never(SeamFake):
        def complete(self, **kwargs):
            calls.append(kwargs)
            return json.dumps({"assumption_exists": False})
    edges = await store.list_candidate_edges(customer.id)
    assert {e["edge_type"] for e in edges} <= set(CHECK_EDGE_TYPES.values()) and edges
    res = await run_assumptions_scan(store=store, slm_client=_Never(), customer_id=customer.id, encoder=_Encoder())
    assert res.pairs_evaluated == 0 and calls == []


@pytest.mark.asyncio
async def test_the_console_and_the_model_both_see_derived_from(store, customer):
    from crystal_cache.endpoints.admin import admin_list_assumptions
    await _seed(store, customer.id)
    await _scan(store, customer)
    body = await admin_list_assumptions(SimpleNamespace(state=SimpleNamespace()), store, customer_id=customer.id)
    items = {i["statement"]: i for i in body["assumptions"]}
    oxy = items["No liquid substance contains oxygen"]
    assert oxy["kind"] == "consequence" and oxy["state"] == "waiting" and oxy["recall_gated"] is False
    assert oxy["parent_fact"] == {"id": "f_oxygen", "claim": "Oxygen only exists as a gas.", "key": "Chemistry|Oxygen|States", "crystal_id": "c_oxy"}
    assert sorted((c["fact_id"], c["verdict"], c["claim"][:5]) for c in oxy["checks"]) == [("f_steam", "CONSISTENT", "Steam"), ("f_water", "CONTRADICTS", "Water")]
    assert [p["id"] for p in oxy["parents"]] == ["c_oxy"]
    # The retrieval framing: DERIVED from the parent, with how much agreed.
    ann = await store.list_assumption_annotations(customer.id, [oxy["id"], items["Pro is billed every month"]["id"], "c_chem"])
    assert ann[oxy["id"]]["kind"] == "consequence" and ann[oxy["id"]]["parent_claim"] == "Oxygen only exists as a gas."
    note = assumption_note(ann)
    assert 'DERIVED from "Oxygen only exists as a gas."; consistent with 1 stated fact here' in note
    assert 'DERIVED from "Pro is $15 since Oct 1."; consistent with 1 stated fact here' in note
    assert "say \"derived from\"" in note


@pytest.mark.asyncio
async def test_the_mind_draws_a_consequence_as_a_ring_with_threads(store, customer, semantic_encoder_stub, vector_store, fact_vector_store):
    from httpx import ASGITransport, AsyncClient
    try:
        from tests.test_endpoint_smoke import _build_app
    except ModuleNotFoundError:
        from test_endpoint_smoke import _build_app
    await _seed(store, customer.id)
    await _scan(store, customer)
    app = _build_app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as cl:
        r = await cl.get(f"/admin/api/customers/{customer.id}/mind")
        assert r.status_code == 200, r.text[:400]
        body = r.json()
    cons = {c["statement"]: c for c in body["consequences"]}
    oxy = cons["No liquid substance contains oxygen"]
    assert oxy["parent_fact_id"] == "f_oxygen" and oxy["pos"] and oxy["region"] is not None and oxy["state"] == "waiting" and oxy["recallable"]
    assert sorted((c["fact_id"], c["verdict"]) for c in oxy["checks"]) == [("f_steam", "CONSISTENT"), ("f_water", "CONTRADICTS")]
    assert all(c["pos"] for c in oxy["checks"])
    assert body["totals"]["consequences"] == 2 and body["assumptions"] == []
    # The consequence's own fact is not a dot in the bank.
    assert not any(f["claim"].startswith("No liquid substance") for f in body["facts"])
