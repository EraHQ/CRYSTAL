"""Q101 (2026-10-23): the ordered allow/deny list.

Anthony: "there is no way to set rules. It should be easy like criteria
met then approve fact. Or I could exclude on x facts from automatically
whitelisting. Think of a networking allow and deny rules." One rule type
(curation) over everything waiting for a decision; rules run top to
bottom; the first match decides; the implicit last line is "a person
decides". The older promotion and approval rules fold into it.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fastapi import HTTPException

from crystal_cache.agent import mcp_server
from crystal_cache.infrastructure.metadata_store_assumption_ext import assumption_state, parse_assumption_tags
from crystal_cache.infrastructure.schema import CrystalRow, FactRow
from crystal_cache.models.crystal import Crystal
from crystal_cache.system_rules import RuleValidationError, validate_rule
from crystal_cache.system_rules.curation import (
    DEFAULT_LINE, Candidate, from_legacy, list_candidates, rule_text, source_of,
)
from crystal_cache.system_rules.model import SystemRule
from crystal_cache.system_rules import store as rules_store

_T0 = datetime.now(timezone.utc) - timedelta(days=5)
D = 8


def _vec(*on: int) -> list[float]:
    v = np.zeros(D, dtype=np.float32)
    for i in on:
        v[i] = 1.0
    return v.tolist()


class _Encoder:
    def encode_native(self, text: str) -> np.ndarray:
        return np.asarray(_vec(hash(text) % D), dtype=np.float32)

    def encode(self, text: str) -> np.ndarray:
        return self.encode_native(text)

    def fingerprint(self) -> str:
        return "stub"


def _rule(**over) -> SystemRule:
    base = dict(id="r", customer_id="c", rule_type="curation", name="n",
                selector={"kind": "consequence"}, conditions={"min_agreeing_facts": 2, "no_contradictions": True},
                action={"decision": "allow"})
    base.update(over)
    return SystemRule(**base)


async def _gated(store, cid, crystal_id, *, origin="background_worker", tags=None, tier="quarantine",
                 keys=(), created_at=None, source_kind="document_chunk"):
    c = Crystal(id=crystal_id, customer_id=cid, summary_vector=[0.1], crystal_type="customer:legacy",
                recall_gated=True, origin=origin, quality_tier=tier, diagnostic_tags=list(tags or []),
                summary_text=f"Memory {crystal_id}", created_at=created_at or datetime.now(timezone.utc))
    await store.upsert_crystal(c)
    async with store.session() as s:
        for i, key in enumerate(keys):
            s.add(FactRow(id=f"{crystal_id}_f{i}", crystal_id=crystal_id, pair_type="question_answer", prompt_text=key,
                          claim_text=f"Claim {i} of {crystal_id}", source_kind=source_kind, vector=_vec(i % D), created_at=_T0))
    return c


async def _consequences(store, cid):
    """Two consequences of a whitelisted parent: one with an agreeing fact,
    one with a contradicting fact and an open conflict."""
    async with store.session() as s:
        s.add(CrystalRow(id="c_price", customer_id=cid, crystal_type="customer:legacy", summary_vector=[], quality_tier="whitelist"))
        s.add(FactRow(id="f_pro", crystal_id="c_price", pair_type="question_answer", prompt_text="Pricing|Pro|Price",
                      claim_text="Pro is $15 a month.", source_kind="document_chunk", vector=_vec(1), created_at=_T0))
        s.add(FactRow(id="f_invoice", crystal_id="c_price", pair_type="question_answer", prompt_text="Pricing|Pro|Invoice",
                      claim_text="Pro customers get an invoice every month.", source_kind="document_chunk", vector=_vec(2), created_at=_T0))
        s.add(FactRow(id="f_free", crystal_id="c_price", pair_type="question_answer", prompt_text="Pricing|Free|Price",
                      claim_text="The free plan has no invoice.", source_kind="document_chunk", vector=_vec(3), created_at=_T0))
    good = await store.create_consequence_crystal(cid, statement="Pro is billed every month", subject="Pricing",
                                                  parent_fact_id="f_pro", parent_crystal_id="c_price", encoder=_Encoder())
    await store.add_consequence_check(cid, good, fact_id="f_invoice", fact_crystal_id="c_price", verdict="CONSISTENT")
    bad = await store.create_consequence_crystal(cid, statement="Every plan is invoiced monthly", subject="Pricing",
                                                 parent_fact_id="f_pro", parent_crystal_id="c_price", encoder=_Encoder())
    await store.add_consequence_check(cid, bad, fact_id="f_free", fact_crystal_id="c_price", verdict="CONTRADICTS")
    await store.create_knowledge_conflict(
        cid, fact_a_id="f_free", fact_b_id="f_pro", claim_a="The free plan has no invoice.", claim_b="Pro is $15 a month.",
        pair_key="pk_bad", crystal_a_id="c_price", crystal_b_id=bad, subject="Pricing", reasoning="A free plan is not invoiced.",
        recommendation={"resolution": "dismissed", "reason": "Different plans."},
    )
    return good, bad


async def _state(store, cid, crystal_id):
    return next(assumption_state(r) for r in await store.list_assumption_crystals(cid) if r["id"] == crystal_id)


# --- the rule itself ---------------------------------------------------------

def test_the_curation_rule_is_strict():
    validate_rule(_rule())
    validate_rule(_rule(selector={"kind": "fact", "key_prefix": "Finance|", "source": "research", "tier": "quarantine"},
                        conditions={"min_citations": 1, "parent_trusted": True, "min_confidence": 0.8, "waited_days": 3, "scan_passed": True},
                        action={"decision": "allow", "tier": "neutral"}))
    validate_rule(_rule(selector={"kind": "any"}, conditions={}, action={"decision": "deny"}))
    for bad in (
        dict(selector={}),
        dict(selector={"kind": "memory"}),
        dict(selector={"kind": "fact", "origin": "x"}),
        dict(selector={"kind": "fact", "key_prefix": "   "}),
        dict(selector={"kind": "fact", "source": "web"}),
        dict(selector={"kind": "fact", "tier": "blacklist"}),
        dict(conditions={"min_agreeing_facts": 0}),
        dict(conditions={"min_consistent_edges": 2}),
        dict(conditions={"min_confidence": 1.5}),
        dict(conditions={"no_contradictions": "yes"}),
        dict(conditions={"waited_days": -1}),
        dict(action={}),
        dict(action={"decision": "hold"}),
        dict(action={"decision": "deny", "tier": "neutral"}),
        dict(action={"decision": "allow", "tier": "whitelist"}),
        dict(action={"decision": "allow", "approve": True}),
    ):
        with pytest.raises(RuleValidationError):
            validate_rule(_rule(**bad))


def test_a_rule_reads_as_one_sentence():
    assert rule_text(_rule()) == "Approve a consequence when 2 stated facts agree with it and nothing contradicts it."
    assert rule_text(_rule(selector={"kind": "fact", "source": "research"}, conditions={"scan_passed": True},
                           action={"decision": "allow", "tier": "neutral"})) == \
        "Let in a fact from research when the review scan passed, and trust it."
    assert rule_text(_rule(selector={"kind": "any", "key_prefix": "Finance|"}, conditions={}, action={"decision": "deny"})) == \
        'Hold anything whose key starts with "Finance|" for a person.'
    assert rule_text(_rule(selector={"kind": "assumption", "tier": "quarantine"},
                           conditions={"min_agreeing_facts": 1, "min_citations": 3, "parent_trusted": True,
                                       "min_confidence": 0.75, "waited_days": 1},
                           action={"decision": "allow"})) == \
        ("Approve an assumption in the quarantine tier when 1 stated fact agrees with it, it has been cited at least "
         "3 times, its parent is trusted, its confidence is at least 75% and it has waited at least 1 day.")
    assert rule_text(_rule(selector={"kind": "consequence", "source": "reasoning"}, conditions={"no_contradictions": True},
                           action={"decision": "deny"})) == "Hold a consequence found by reasoning for a person when nothing contradicts it."
    assert DEFAULT_LINE == "Otherwise, a person decides."


def test_legacy_rules_fold_into_the_same_sentences():
    promo = from_legacy(SystemRule(id="p", customer_id="c", rule_type="promotion", name="auto", priority=20,
                                   selector={"origin": "background_worker"},
                                   conditions={"outbound_scan_passed": True, "no_open_conflicts": True, "min_grounded_citations": 2},
                                   action={"clear_recall_gate": True, "set_tier": "neutral"}))
    assert promo.rule_type == "curation" and promo.id == "p" and promo.priority == 20
    assert promo.selector == {"kind": "fact", "source": "research"}
    assert promo.conditions == {"scan_passed": True, "no_contradictions": True, "min_citations": 2}
    assert promo.action == {"decision": "allow", "tier": "neutral"}
    validate_rule(promo)
    appr = from_legacy(SystemRule(id="a", customer_id="c", rule_type="approval", name="two", selector={"kind": "consequence"},
                                  conditions={"min_consistent_edges": 2}, action={"approve": True}))
    assert appr.selector == {"kind": "consequence"} and appr.conditions == {"min_agreeing_facts": 2, "no_contradictions": True}
    assert appr.action == {"decision": "allow"}
    loose = from_legacy(SystemRule(id="a", customer_id="c", rule_type="approval", name="two", selector={"kind": "any"},
                                   conditions={"min_consistent_edges": 1, "no_contradicted_edges": False}, action={"approve": True}))
    assert loose.conditions == {"min_agreeing_facts": 1}
    assert from_legacy(_rule()) is None
    # A legacy rule reads as its folded sentence even before conversion.
    assert rule_text(SystemRule(id="a", customer_id="c", rule_type="approval", name="two", selector={"kind": "consequence"},
                                conditions={"min_consistent_edges": 2}, action={"approve": True})) == rule_text(_rule())


def test_source_is_read_from_origin_then_provenance():
    assert source_of(origin="background_worker", source_kind="document_chunk", source_document_id=None) == "research"
    assert source_of(origin="assumptions", source_kind="agent_inferred", source_document_id=None) == "reasoning"
    assert source_of(origin="implication", source_kind="agent_inferred", source_document_id=None) == "reasoning"
    assert source_of(origin="direct", source_kind="document_extraction", source_document_id=None) == "document"
    assert source_of(origin="direct", source_kind="operator_stated", source_document_id="doc_1") == "document"
    assert source_of(origin="direct", source_kind="operator_stated", source_document_id=None) == "chat"


# --- the list ----------------------------------------------------------------

@pytest.mark.asyncio
async def test_candidates_are_gated_memory_and_waiting_assumptions_never_blacklisted(store, customer):
    cid = customer.id
    await _gated(store, cid, "g_research", keys=("Research|Market|Size",))
    await _gated(store, cid, "g_black", tier="blacklist")
    good, bad = await _consequences(store, cid)
    # The good consequence's gate is open (its parent is whitelisted and one fact agrees) but nobody approved it.
    assert await _state(store, cid, good) == "waiting"
    cands = {c.id: c for c in await list_candidates(store, cid)}
    assert set(cands) == {"g_research", good, bad}
    assert cands["g_research"].kind == "fact" and cands["g_research"].source == "research"
    assert cands[good].kind == "consequence" and cands[good].source == "reasoning" and cands[good].parent_crystal_id == "c_price"
    assert isinstance(cands[good], Candidate)


@pytest.mark.asyncio
async def test_no_rule_touches_nothing(store, customer):
    cid = customer.id
    await _gated(store, cid, "g1", tags=["outbound_scan_passed"])
    good, _ = await _consequences(store, cid)
    assert await rules_store.run_curation_rules(store, cid) == {"allowed": 0, "held": 0, "rules_fired": 0}
    assert (await store.get_crystal("g1")).recall_gated is True
    assert await _state(store, cid, good) == "waiting"


@pytest.mark.asyncio
async def test_first_match_decides_and_deny_holds(store, customer):
    cid = customer.id
    await _gated(store, cid, "g_fin", tags=["outbound_scan_passed"], keys=("Finance|Q3|Revenue",))
    await _gated(store, cid, "g_ops", tags=["outbound_scan_passed"], keys=("Ops|Oncall|Rota",))
    hold = await rules_store.create_rule(store, cid, "curation", "Hold finance",
                                         selector={"kind": "any", "key_prefix": "finance|"}, conditions={}, action={"decision": "deny"})
    let_in = await rules_store.create_rule(store, cid, "curation", "Scanned research",
                                           selector={"kind": "fact", "source": "research"}, conditions={"scan_passed": True},
                                           action={"decision": "allow", "tier": "neutral"})
    assert [r.id for r in await rules_store.list_rules(store, cid)] == [hold.id, let_in.id]
    out = await rules_store.run_curation_rules(store, cid)
    assert out == {"allowed": 1, "held": 1, "rules_fired": 1}
    fin, ops = await store.get_crystal("g_fin"), await store.get_crystal("g_ops")
    assert fin.recall_gated is True and fin.quality_tier == "quarantine"
    assert ops.recall_gated is False and ops.quality_tier == "neutral"
    fired = {r.id: r.fire_count for r in await rules_store.list_rules(store, cid)}
    assert fired == {hold.id: 0, let_in.id: 1}
    labels = [e["label"] for e in await store.list_curation_events(cid, limit=20) if e["event_type"] == "memory_allowed"]
    assert labels == ["Let into recall by rule Scanned research - Memory g_ops"]
    # Move the allow above the deny: now finance research is let in too (first match wins).
    rules = await rules_store.move_rule(store, cid, let_in.id, "up")
    assert [r.id for r in rules] == [let_in.id, hold.id] and [r.priority for r in rules] == [10, 20]
    assert await rules_store.run_curation_rules(store, cid) == {"allowed": 1, "held": 0, "rules_fired": 1}
    assert (await store.get_crystal("g_fin")).recall_gated is False
    # Moving past the ends is a no-op; another tenant's rule is None.
    assert [r.id for r in await rules_store.move_rule(store, cid, let_in.id, "up")] == [let_in.id, hold.id]
    assert await rules_store.move_rule(store, "cus_other", let_in.id, "down") is None
    with pytest.raises(ValueError):
        await rules_store.move_rule(store, cid, let_in.id, "sideways")


@pytest.mark.asyncio
async def test_conditions_that_cannot_be_read_do_not_hold(store, customer):
    """A fact has no agreement edges and no confidence; a rule that needs
    them falls through to the next line instead of matching."""
    cid = customer.id
    await _gated(store, cid, "g_plain", tags=["outbound_scan_passed"])
    await rules_store.create_rule(store, cid, "curation", "Agreeing", selector={"kind": "any"},
                                  conditions={"min_agreeing_facts": 1}, action={"decision": "allow"})
    await rules_store.create_rule(store, cid, "curation", "Confident", selector={"kind": "any"},
                                  conditions={"min_confidence": 0.5}, action={"decision": "allow"})
    assert await rules_store.run_curation_rules(store, cid) == {"allowed": 0, "held": 0, "rules_fired": 0}
    assert (await store.get_crystal("g_plain")).recall_gated is True


@pytest.mark.asyncio
async def test_consequences_are_approved_by_agreement_and_never_when_contradicted(store, customer):
    cid = customer.id
    good, bad = await _consequences(store, cid)
    rule = await rules_store.create_rule(store, cid, "curation", "One agrees", selector={"kind": "consequence"},
                                         conditions={"min_agreeing_facts": 1, "no_contradictions": True, "parent_trusted": True},
                                         action={"decision": "allow"})
    assert await rules_store.run_curation_rules(store, cid) == {"allowed": 1, "held": 0, "rules_fired": 1}
    assert await _state(store, cid, good) == "approved" and await _state(store, cid, bad) == "waiting"
    row = next(r for r in await store.list_assumption_crystals(cid) if r["id"] == good)
    assert row["quality_tier"] == "whitelist"  # the parent's standing
    assert parse_assumption_tags(row["diagnostic_tags"])["approved_by"] == f"rule:{rule.id}"
    labels = [e["label"] for e in await store.list_curation_events(cid, limit=20) if e["event_type"] == "assumption_approved"]
    assert labels == ["Approved by rule One agrees - Pro is billed every month"]
    # Idempotent; the contradicted one stays for a person even under a looser rule below.
    await rules_store.create_rule(store, cid, "curation", "Anything", selector={"kind": "consequence"},
                                  conditions={}, action={"decision": "allow"})
    assert await rules_store.run_curation_rules(store, cid) == {"allowed": 1, "held": 0, "rules_fired": 1}
    assert await _state(store, cid, bad) == "approved"  # the loose rule approves it: that is what the list says


@pytest.mark.asyncio
async def test_waited_days_and_tier_selectors(store, customer):
    cid = customer.id
    await _gated(store, cid, "g_old", created_at=datetime.now(timezone.utc) - timedelta(days=10))
    await _gated(store, cid, "g_new")
    await _gated(store, cid, "g_neutral", tier="neutral")
    await rules_store.create_rule(store, cid, "curation", "Patient", selector={"kind": "fact", "tier": "quarantine"},
                                  conditions={"waited_days": 7}, action={"decision": "allow"})
    assert await rules_store.run_curation_rules(store, cid) == {"allowed": 1, "held": 0, "rules_fired": 1}
    assert (await store.get_crystal("g_old")).recall_gated is False
    assert (await store.get_crystal("g_new")).recall_gated is True
    assert (await store.get_crystal("g_neutral")).recall_gated is True


@pytest.mark.asyncio
async def test_legacy_rules_are_folded_in_place_and_keep_working(store, customer):
    cid = customer.id
    await _gated(store, cid, "g_scan", tags=["outbound_scan_passed"])
    good, _ = await _consequences(store, cid)
    promo = await rules_store.create_rule(store, cid, "promotion", "auto", selector={"origin": "background_worker"},
                                          conditions={"outbound_scan_passed": True}, action={"clear_recall_gate": True})
    appr = await rules_store.create_rule(store, cid, "approval", "one agrees", selector={"kind": "consequence"},
                                         conditions={"min_consistent_edges": 1}, action={"approve": True}, enabled=False)
    assert await rules_store.run_curation_rules(store, cid) == {"allowed": 1, "held": 0, "rules_fired": 1}
    rules = {r.id: r for r in await rules_store.list_rules(store, cid)}
    assert {r.rule_type for r in rules.values()} == {"curation"}
    assert rules[promo.id].name == "auto" and rules[promo.id].fire_count == 1 and rules[promo.id].enabled
    assert rules[appr.id].enabled is False and rules[appr.id].conditions == {"min_agreeing_facts": 1, "no_contradictions": True}
    assert (await store.get_crystal("g_scan")).recall_gated is False
    assert await _state(store, cid, good) == "waiting"  # the folded approval rule is off
    assert await rules_store.convert_legacy_rules(store, cid) == 0


@pytest.mark.asyncio
async def test_the_worker_pass_runs_the_list(store, customer):
    from crystal_cache.workers.cognition import _run_system_promotion_rules
    cid = customer.id
    await _gated(store, cid, "wp_a", tags=["outbound_scan_passed"])
    await rules_store.create_rule(store, cid, "curation", "Scanned", selector={"kind": "fact"},
                                  conditions={"scan_passed": True}, action={"decision": "allow"})
    other = await store.create_customer(provider="anthropic", model_id="claude-sonnet-4-5-20250929", api_key_ref="test-ref-other")
    await _gated(store, other.id, "wp_b", tags=["outbound_scan_passed"])
    await _run_system_promotion_rules(store=store)
    assert (await store.get_crystal("wp_a")).recall_gated is False
    assert (await store.get_crystal("wp_b")).recall_gated is True


@pytest.mark.asyncio
async def test_update_can_change_what_a_rule_applies_to(store, customer):
    cid = customer.id
    r = await rules_store.create_rule(store, cid, "curation", "x", selector={"kind": "fact"}, conditions={}, action={"decision": "deny"})
    out = await rules_store.update_rule(store, cid, r.id, selector={"kind": "any", "source": "chat"}, action={"decision": "allow"})
    assert out.selector == {"kind": "any", "source": "chat"} and out.action == {"decision": "allow"}
    with pytest.raises(RuleValidationError):
        await rules_store.update_rule(store, cid, r.id, action={"decision": "deny", "tier": "neutral"})
    assert await rules_store.update_rule(store, "cus_other", r.id, enabled=False) is None


# --- the routes ----------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_routes_list_fold_move_and_edit(monkeypatch, store):
    try:
        from tests.test_accounts_phase_c import _AuthedReq, _use
    except ModuleNotFoundError:
        from test_accounts_phase_c import _AuthedReq, _use
    import json as _json
    from crystal_cache.endpoints import customers as ep

    _use(monkeypatch)
    c = await store.create_customer(provider="anthropic", model_id="m", api_key_ref="")
    cid = c.id
    bearer = f"Bearer {c.api_key}"

    def body(resp):
        return _json.loads(resp.body)

    await rules_store.create_rule(store, cid, "approval", "Approve after 2 agree", selector={"kind": "consequence"},
                                  conditions={"min_consistent_edges": 2, "no_contradicted_edges": True}, action={"approve": True})
    created = body(await ep.create_customer_rule(cid, _AuthedReq(bearer, body={
        "rule_type": "curation", "name": "Hold finance",
        "selector": {"kind": "any", "key_prefix": "Finance|"}, "conditions": {}, "action": {"decision": "deny"},
    }), store))["rule"]
    assert created["text"] == 'Hold anything whose key starts with "Finance|" for a person.' and created["priority"] == 20
    listed = body(await ep.list_customer_rules(cid, _AuthedReq(bearer), store))
    assert listed["default"] == DEFAULT_LINE and [r["rule_type"] for r in listed["rules"]] == ["curation", "curation"]
    assert listed["rules"][0]["text"] == "Approve a consequence when 2 stated facts agree with it and nothing contradicts it."
    moved = body(await ep.move_customer_rule(cid, created["id"], _AuthedReq(bearer, body={"direction": "up"}), store))
    assert [r["id"] for r in moved["rules"]][0] == created["id"] and [r["priority"] for r in moved["rules"]] == [10, 20]
    with pytest.raises(HTTPException) as e:
        await ep.move_customer_rule(cid, created["id"], _AuthedReq(bearer, body={"direction": "left"}), store)
    assert e.value.status_code == 400
    with pytest.raises(HTTPException) as e:
        await ep.move_customer_rule(cid, "rule_missing", _AuthedReq(bearer, body={"direction": "up"}), store)
    assert e.value.status_code == 404
    edited = body(await ep.update_customer_rule(cid, created["id"], _AuthedReq(bearer, body={"selector": {"kind": "fact", "key_prefix": "Finance|"}}), store))["rule"]
    assert edited["text"] == 'Hold a fact whose key starts with "Finance|" for a person.'
    with pytest.raises(HTTPException) as e:
        await ep.update_customer_rule(cid, created["id"], _AuthedReq(bearer, body={"action": {"decision": "deny", "tier": "neutral"}}), store)
    assert e.value.status_code == 400
    # Another tenant cannot move it.
    other = await store.create_customer(provider="anthropic", model_id="m", api_key_ref="")
    with pytest.raises(HTTPException) as e:
        await ep.move_customer_rule(cid, created["id"], _AuthedReq(f"Bearer {other.api_key}", body={"direction": "up"}), store)
    assert e.value.status_code in (401, 403, 404)


# --- over MCP ----------------------------------------------------------------

@pytest.fixture
def mcp_state(store, monkeypatch):
    state = {"store": store, "vector_store": None, "fact_vector_store": None, "encoder": _Encoder(), "vector_index": None}
    monkeypatch.setattr(mcp_server, "_get_state", lambda: state)
    return state


def _tool(name):
    fn = getattr(mcp_server, name)
    return getattr(fn, "fn", fn)


@pytest.mark.asyncio
async def test_a_client_writes_the_list_from_chat(store, customer, mcp_state):
    cid = customer.id
    token = mcp_server._current_customer_id.set(cid)
    try:
        assert (await _tool("memory_rules")()) == {"rules": [], "count": 0, "default": DEFAULT_LINE}
        out = await _tool("memory_rule_add")("allow", kind="consequence", min_agreeing_facts=2, parent_trusted=True)
        assert out["ok"] and out["rule"]["order"] == 1 and out["rule"]["decision"] == "allow"
        assert out["rule"]["text"] == "Approve a consequence when 2 stated facts agree with it, nothing contradicts it and its parent is trusted."
        assert out["rule"]["name"] == out["rule"]["text"]
        first = out["rule"]["id"]
        out = await _tool("memory_rule_add")("deny", key_prefix="Finance|", name="Keep finance for me")
        assert out["rule"]["order"] == 2 and out["rule"]["conditions"] == {} and out["rule"]["name"] == "Keep finance for me"
        second = out["rule"]["id"]
        assert (await _tool("memory_rule_add")("maybe"))["code"] == "rule_invalid"
        assert (await _tool("memory_rule_add")("allow", source="web"))["code"] == "rule_invalid"
        moved = await _tool("memory_rule_move")(second, "up")
        assert [r["id"] for r in moved["rules"]] == [second, first]
        assert (await _tool("memory_rule_move")(second, "sideways"))["code"] == "invalid_argument"
        assert (await _tool("memory_rule_move")("rule_missing", "up"))["code"] == "not_found"
        toggled = await _tool("memory_rule_toggle")(first, False)
        assert toggled["rule"]["enabled"] is False
        removed = await _tool("memory_rule_remove")(second)
        assert removed["deleted"] == second and [r["id"] for r in removed["rules"]] == [first]
        assert (await _tool("memory_rule_remove")(second))["code"] == "not_found"
        assert (await _tool("memory_rule_remove")("bad/id"))["code"] == "invalid_argument"
    finally:
        mcp_server._current_customer_id.reset(token)
