"""The controls over MCP (2026-10-23). Anthony: "I want ALL THE CONTROLS.
Give me the mcp route." Every curation act the console offers is a tool
an AI client can use, sharing the in-app Steward's implementation
(cognition/controls.py), ledger (actor chat:<operator>), and 24-hour undo.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from crystal_cache.agent import mcp_server
from crystal_cache.infrastructure.metadata_store_assumption_ext import assumption_state, parse_assumption_tags
from crystal_cache.infrastructure.schema import CrystalRow, FactRow

_T0 = datetime.now(timezone.utc) - timedelta(days=2)
D = 8

CONTROL_TOOLS = {
    "memory_assumptions", "memory_actions", "memory_decide_conflict", "memory_reopen_conflict",
    "memory_approve_assumption", "memory_reject_assumption", "memory_verify_assumption",
    "memory_research_gap", "memory_dismiss_gap", "memory_reopen_gap", "memory_answer_gap",
    "memory_edit_fact", "memory_retire_fact", "memory_set_tier", "memory_reprobe_fact",
    "memory_undo", "memory_steward",
}


def _tool(name):
    fn = getattr(mcp_server, name)
    return getattr(fn, "fn", fn)


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


@pytest.fixture
def mcp_state(store, monkeypatch):
    from crystal_cache.infrastructure.fact_vector_store import FactVectorStore
    from crystal_cache.infrastructure.vector_store import VectorStore

    state = {"store": store, "vector_store": VectorStore(store), "fact_vector_store": FactVectorStore(store),
             "encoder": _Encoder(), "vector_index": None}
    monkeypatch.setattr(mcp_server, "_get_state", lambda: state)
    return state


async def _seed(store, cid):
    async with store.session() as s:
        s.add(CrystalRow(id="c_price", customer_id=cid, crystal_type="customer:legacy", summary_vector=[], quality_tier="whitelist", summary_text="Pricing"))
        s.add(CrystalRow(id="c_old", customer_id=cid, crystal_type="customer:legacy", summary_vector=[], quality_tier="neutral", summary_text="Old pricing"))
        s.add(FactRow(id="f_new", crystal_id="c_price", pair_type="question_answer", prompt_text="Pricing|Pro|Price",
                      claim_text="Pro is $15 a month since October 1, 2026.", source_kind="document_chunk", vector=_vec(1), created_at=_T0 + timedelta(days=1)))
        s.add(FactRow(id="f_old", crystal_id="c_old", pair_type="question_answer", prompt_text="Pricing|Pro|Price",
                      claim_text="Pro costs $12 a month.", source_kind="document_chunk", vector=_vec(1, 2), created_at=_T0))
        s.add(FactRow(id="f_invoice", crystal_id="c_price", pair_type="question_answer", prompt_text="Pricing|Pro|Invoice",
                      claim_text="Pro customers get an invoice every month.", source_kind="document_chunk", vector=_vec(3), created_at=_T0))
    conflict = await store.create_knowledge_conflict(
        cid, fact_a_id="f_old", fact_b_id="f_new", claim_a="Pro costs $12 a month.", claim_b="Pro is $15 a month since October 1, 2026.",
        pair_key="pk_price", crystal_a_id="c_old", crystal_b_id="c_price", subject="Pricing",
        reasoning="Two prices for one plan.", recommendation={"resolution": "superseded", "loser": "a", "reason": "B is dated."},
    )
    good = await store.create_consequence_crystal(cid, statement="Pro is billed every month", subject="Pricing",
                                                  parent_fact_id="f_new", parent_crystal_id="c_price", encoder=_Encoder())
    await store.add_consequence_check(cid, good, fact_id="f_invoice", fact_crystal_id="c_price", verdict="CONSISTENT")
    gap = await store.create_knowledge_gap(cid, domain="Pricing", subject="VAT", missing="Does Kerning charge EU VAT?", full_key="Pricing|VAT")
    return {"conflict": conflict.id, "good": good, "gap": gap.id}


@pytest.mark.asyncio
async def test_the_controls_are_on_the_full_toolset_and_annotated():
    names = {t.name for t in await mcp_server.mcp.list_tools()}
    assert CONTROL_TOOLS <= names, CONTROL_TOOLS - names
    for name in CONTROL_TOOLS:
        assert name in mcp_server._TOOL_ANNOTATIONS
    assert mcp_server._TOOL_ANNOTATIONS["memory_retire_fact"].destructiveHint is True
    assert mcp_server._TOOL_ANNOTATIONS["memory_assumptions"].readOnlyHint is True


@pytest.mark.asyncio
async def test_a_client_curates_the_bank_and_every_act_is_ledgered_and_undoable(store, customer, mcp_state):
    cid = customer.id
    ids = await _seed(store, cid)
    token = mcp_server._current_customer_id.set(cid)
    try:
        # Read first: the queue as a reviewer sees it.
        q = await _tool("memory_assumptions")(state="waiting", kind="consequence")
        assert q["count"] == 1 and q["assumptions"][0]["derived_from"].startswith("Pro is $15") and q["assumptions"][0]["checks"][0]["verdict"] == "CONSISTENT"
        # Reasons are required; ids are checked.
        assert (await _tool("memory_decide_conflict")(ids["conflict"], "superseded", "short", loser="a"))["code"] == "reason_required"
        assert (await _tool("memory_decide_conflict")("nope/bad", "superseded", "The newer claim is dated and wins."))["code"] == "invalid_argument"
        # Settle the conflict; approve the consequence; dismiss the gap; edit a fact; move a tier.
        out = await _tool("memory_decide_conflict")(ids["conflict"], "superseded", "The $15 price is dated October 2026 and newer.", loser="a")
        assert out["ok"] and out["retired_claim"] == "Pro costs $12 a month."
        assert (await store.get_fact("f_old")).grating_strength == 0.0
        out = await _tool("memory_approve_assumption")(ids["good"], "The invoice fact agrees and nothing contradicts it.")
        assert out["ok"] and out["quality_tier"] == "whitelist"
        out = await _tool("memory_dismiss_gap")(ids["gap"], "VAT is handled by the billing provider; not needed here.")
        assert out["ok"] and (await store.get_knowledge_gap(ids["gap"])).status == "closed"
        out = await _tool("memory_edit_fact")("c_price", "f_invoice", "Pro customers get an invoice on the first of every month.", "The invoice date was missing.")
        assert out["ok"] and out["superseded"] == "f_invoice" and out["fact_id"] != "f_invoice"
        new_fact_id = out["fact_id"]
        assert (await store.get_fact(new_fact_id)).claim_text.endswith("first of every month.") and await store.get_fact("f_invoice") is None
        out = await _tool("memory_set_tier")("c_old", "quarantine", "The old pricing cluster is stale.")
        assert out["ok"] and out["was"] == "neutral" and (await store.get_crystal("c_old")).quality_tier == "quarantine"
        # The ledger says who (chat), what, why, and that each can be undone.
        ledger = (await _tool("memory_actions")())["actions"]
        assert [a["action"] for a in ledger] == ["set_tier", "edit_fact", "dismiss_gap", "approve_assumption", "decide_conflict"]
        assert all(a["actor"] == "chat" and a["undoable"] and a["reason"] for a in ledger)
        by = {a["action"]: a for a in ledger}
        # Undo every one of them; the bank is as it was.
        assert (await _tool("memory_undo")(by["edit_fact"]["id"]))["undone"]
        restored = [f for f in await store.list_facts_for_crystal("c_price") if f.prompt_text == "Pricing|Pro|Invoice"]
        assert len(restored) == 1 and restored[0].claim_text == "Pro customers get an invoice every month."
        assert (await _tool("memory_undo")(by["set_tier"]["id"]))["undone"] and (await store.get_crystal("c_old")).quality_tier == "neutral"
        assert (await _tool("memory_undo")(by["decide_conflict"]["id"]))["undone"]
        assert (await store.get_knowledge_conflict(ids["conflict"])).status == "open" and (await store.get_fact("f_old")).grating_strength == 1.0
        assert (await _tool("memory_undo")(by["approve_assumption"]["id"]))["undone"]
        good = next(r for r in await store.list_assumption_crystals(cid) if r["id"] == ids["good"])
        assert assumption_state(good) == "waiting" and parse_assumption_tags(good["diagnostic_tags"])["approved_by"] is None
        assert (await _tool("memory_undo")(by["dismiss_gap"]["id"]))["undone"] and (await store.get_knowledge_gap(ids["gap"])).status == "open"
        assert (await _tool("memory_undo")(by["dismiss_gap"]["id"]))["code"] == "already_undone"
        events = [e["event_type"] for e in await store.list_curation_events(cid, limit=60)]
        assert events.count("control_action") == 5 and events.count("steward_undone") == 5
    finally:
        mcp_server._current_customer_id.reset(token)


@pytest.mark.asyncio
async def test_answer_research_verify_retire_and_the_steward_switch(store, customer, mcp_state):
    cid = customer.id
    ids = await _seed(store, cid)
    token = mcp_server._current_customer_id.set(cid)
    try:
        out = await _tool("memory_answer_gap")(ids["gap"], "Kerning charges EU VAT at the customer's local rate.", "Known from the billing setup.")
        assert out["ok"] and out["key"] == "Pricing|VAT"
        assert (await store.get_knowledge_gap(ids["gap"])).status == "filled"
        written = await store.get_fact(out["fact_id"])
        assert written.source_kind == "operator_stated" and written.claim_text.startswith("Kerning charges EU VAT")
        assert (await _tool("memory_undo")(out["action_id"]))["undone"]
        assert (await store.get_knowledge_gap(ids["gap"])).status == "open" and await store.get_crystal(out["crystal_id"]) is None
        # Research and verification queue runs; undo cancels them.
        out = await _tool("memory_research_gap")(ids["gap"], "Public tax rules; the web answers this.")
        assert out["ok"] and (await store.get_cognition_task(out["task_id"])).task_type == "research"
        assert (await _tool("memory_research_gap")(ids["gap"], "Asking again for the same gap now."))["code"] == "already_researching"
        assert (await _tool("memory_undo")(out["action_id"]))["undone"] and (await store.get_cognition_task(out["task_id"])).status == "cancelled"
        out = await _tool("memory_verify_assumption")(ids["good"], "Check the billing cadence against the docs.")
        assert out["ok"] and (await store.get_cognition_task(out["task_id"])).task_type == "assumption_verification"
        assert (await _tool("memory_verify_assumption")(ids["good"], "Trying to queue it a second time now."))["code"] == "already_verifying"
        assert (await _tool("memory_undo")(out["action_id"]))["undone"]
        assert not any(str(t).startswith("verification_task:") for t in (await store.get_crystal(ids["good"])).diagnostic_tags)
        # Retire is permanent and says so; the ledger keeps the text.
        out = await _tool("memory_retire_fact")("c_old", "f_old", "The old price is wrong and misleads recall.")
        assert out["ok"] and out["permanent"] and await store.get_fact("f_old") is None
        assert (await _tool("memory_undo")(out["action_id"]))["code"] == "not_undoable"
        # The in-app Steward switch, read and flipped from chat.
        st = await _tool("memory_steward")()
        assert st["enabled"] is False
        st = await _tool("memory_steward")(enabled=True)
        assert st["enabled"] is True and (await store.get_customer_by_id(cid)).steward_enabled is True
        # A blacklisted resolution through chat is allowed (the in-app Steward may not).
        out = await _tool("memory_decide_conflict")(ids["conflict"], "blacklisted", "The $12 claim must never come back.", loser="a")
        assert out["ok"] and out["resolution"] == "blacklisted"
        assert (await _tool("memory_reopen_conflict")(ids["conflict"], "Reconsidering; a person asked."))["ok"]
        assert (await store.get_knowledge_conflict(ids["conflict"])).status == "open"
    finally:
        mcp_server._current_customer_id.reset(token)


@pytest.mark.asyncio
async def test_the_in_app_steward_shares_the_implementation(store, customer):
    from crystal_cache.cognition import steward as mod

    ids = await _seed(store, customer.id)
    session = mod.StewardSession(store=store, customer=customer, run_id="cog_x", encoder=_Encoder(), fact_store=None, max_actions=6)
    tool = {t.name: t for t in mod.build_steward_registry(session).list_for_context("agent")}
    out = await tool["decide_conflict"].impl(customer_id=customer.id, conflict_id=ids["conflict"], resolution="blacklisted", loser="a", reason="Trying a blacklist from the Steward.")
    assert out["code"] == "bad_resolution"
    out = await tool["dismiss_gap"].impl(customer_id=customer.id, gap_id=ids["gap"], reason="Not worth the research budget today.")
    assert out["ok"] and session.actions[0]["action"] == "dismiss_gap" and session.actions[0]["subject_id"] == ids["gap"]
    ledger = await store.list_steward_actions(customer.id)
    assert ledger[0]["actor"] == "steward" and ledger[0]["label"].startswith("Steward dismissed a gap")
