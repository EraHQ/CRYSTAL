"""The Steward (Q100 A, 2026-10-23): "let a model have the hands."

One session, scripted end to end: the agent orients (bank_status,
recent_actions), reads the queue, settles a conflict the bank itself
decides, approves the consequence the facts agree with and is refused
the contradicted one, queues research for a researchable gap, dismisses
a trivial one, and writes its brief. Every action lands in the ledger
with its inverse and on the curation feed; the run shows in Activity;
the spend is metered under origin steward. Then a person undoes each
action and the bank is as it was. The worker pass runs one session per
enabled tenant, caps the day, and asks for a refill once at the wall.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

from crystal_cache.cognition import steward as mod
from crystal_cache.infrastructure.metadata_store_assumption_ext import assumption_state, parse_assumption_tags
from crystal_cache.infrastructure.schema import CrystalRow, FactRow
from fakes import FakeAnthropic

_T0 = datetime.now(timezone.utc) - timedelta(days=2)
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


async def _seed(store, customer):
    cid = customer.id
    async with store.session() as s:
        s.add(CrystalRow(id="c_price", customer_id=cid, crystal_type="customer:legacy", summary_vector=[], quality_tier="whitelist"))
        s.add(CrystalRow(id="c_old", customer_id=cid, crystal_type="customer:legacy", summary_vector=[], quality_tier="neutral"))
        s.add(FactRow(id="f_new", crystal_id="c_price", pair_type="question_answer", prompt_text="Pricing|Pro|Price",
                      claim_text="Pro is $15 a month since October 1, 2026.", source_kind="document_chunk",
                      source_doc_id="doc_new", vector=_vec(1), created_at=_T0 + timedelta(days=1)))
        s.add(FactRow(id="f_old", crystal_id="c_old", pair_type="question_answer", prompt_text="Pricing|Pro|Price",
                      claim_text="Pro costs $12 a month.", source_kind="document_chunk",
                      source_doc_id="doc_old", vector=_vec(1, 2), created_at=_T0))
        s.add(FactRow(id="f_invoice", crystal_id="c_price", pair_type="question_answer", prompt_text="Pricing|Pro|Invoice",
                      claim_text="Pro customers get an invoice every month.", source_kind="document_chunk",
                      source_doc_id="doc_new", vector=_vec(3), created_at=_T0))
        s.add(FactRow(id="f_refund", crystal_id="c_price", pair_type="question_answer", prompt_text="Pricing|Refunds",
                      claim_text="Refunds are never offered.", source_kind="document_chunk",
                      source_doc_id="doc_new", vector=_vec(4), created_at=_T0))
    conflict = await store.create_knowledge_conflict(
        cid, fact_a_id="f_old", fact_b_id="f_new", claim_a="Pro costs $12 a month.", claim_b="Pro is $15 a month since October 1, 2026.",
        pair_key="pk_price", crystal_a_id="c_old", crystal_b_id="c_price", subject="Pricing",
        reasoning="Two prices for one plan.", recommendation={"resolution": "superseded", "loser": "a", "reason": "B is dated."},
    )
    enc = _Encoder()
    good = await store.create_consequence_crystal(cid, statement="Pro is billed every month", subject="Pricing",
                                                  parent_fact_id="f_new", parent_crystal_id="c_price", encoder=enc)
    await store.add_consequence_check(cid, good, fact_id="f_invoice", fact_crystal_id="c_price", verdict="CONSISTENT")
    bad = await store.create_consequence_crystal(cid, statement="Pro customers can get a refund", subject="Pricing",
                                                 parent_fact_id="f_new", parent_crystal_id="c_price", encoder=enc)
    await store.add_consequence_check(cid, bad, fact_id="f_refund", fact_crystal_id="c_price", verdict="CONTRADICTS")
    g1 = await store.create_knowledge_gap(cid, domain="Pricing", subject="VAT", missing="Does Kerning charge EU VAT?", full_key="Pricing|VAT")
    await store.update_knowledge_gap_disposition(g1.id, "researchable")
    g2 = await store.create_knowledge_gap(cid, domain="Office", subject="Coffee", missing="Which coffee does the office buy?", full_key="Office|Coffee")
    return {"conflict": conflict.id, "good": good, "bad": bad, "g1": g1.id, "g2": g2.id}


def _script(anth: FakeAnthropic, ids: dict) -> None:
    """The model's plan for the session, one tool call per turn."""
    anth.script_tool_use("bank_status", {}, "tu_01")
    anth.script_tool_use("recent_actions", {}, "tu_02")
    anth.script_tool_use("open_conflicts", {"limit": 5}, "tu_03")
    anth.script_tool_use("decide_conflict", {"conflict_id": ids["conflict"], "resolution": "superseded", "loser": "a",
                                             "reason": "The $15 price is dated October 2026 and comes from the newer document; $12 is the old price."}, "tu_04")
    anth.script_tool_use("waiting_assumptions", {"kind": "consequence"}, "tu_05")
    anth.script_tool_use("approve_assumption", {"crystal_id": ids["good"], "reason": "The invoice fact agrees and nothing contradicts it."}, "tu_06")
    anth.script_tool_use("approve_assumption", {"crystal_id": ids["bad"], "reason": "Trying the contradicted one; the tool should refuse."}, "tu_07")
    anth.script_tool_use("open_gaps", {}, "tu_08")
    anth.script_tool_use("research_gap", {"gap_id": ids["g1"], "reason": "EU VAT rules are public; a web search answers this."}, "tu_09")
    anth.script_tool_use("dismiss_gap", {"gap_id": ids["g2"], "reason": "Office coffee is trivia the bank does not need."}, "tu_10")
    anth.script_text("Did: settled the Pro price conflict (the $15 claim is newer and dated); approved the monthly-billing consequence; "
                     "queued research on EU VAT; dismissed the coffee gap.\nLeft alone: the refund consequence, a fact contradicts it.\nNext: nothing until research lands.")


@pytest.mark.asyncio
async def test_one_session_works_the_queue_and_everything_is_recorded(store, customer):
    ids = await _seed(store, customer)
    anth = FakeAnthropic()
    _script(anth, ids)
    out = await mod.run_steward_session(store=store, customer=customer, encoder=_Encoder(), fact_store=None, llm=anth, session_number=1)
    assert out["ok"] and out["actions"] == 4 and out["calls"] == 10 and out["stop_reason"] == "end_turn"
    assert out["brief"].startswith("Did: settled the Pro price conflict")
    # The model saw only the Steward's tools, in the Steward's charter.
    first = anth.calls[0]
    names = sorted(t["name"] for t in first["tools"])
    assert names == sorted(["bank_status", "recent_actions", "open_conflicts", "waiting_assumptions", "open_gaps", "crystal_search", "crystal_key_scan",
                            "decide_conflict", "approve_assumption", "reject_assumption", "research_gap", "dismiss_gap", "reprobe_fact"])
    assert "You are the Steward" in first["system"][0]["text"] and "undoable for 24 hours" in first["system"][0]["text"]
    # The tool results the model read: a refusal for the contradicted consequence, with a code.
    def result_of(i: int) -> dict:
        msg = anth.calls[i + 1]["messages"][-1]["content"][0]
        return json.loads(msg["content"]) if isinstance(msg["content"], str) else msg["content"]
    assert result_of(0)["open_conflicts"] == 1 and result_of(0)["waiting_consequences"] == 2 and result_of(0)["actions_left_this_session"] == 6
    assert result_of(2)["conflicts"][0]["a"]["scope"].startswith("filed under Pricing > Pro > Price")
    assert result_of(3)["ok"] and result_of(3)["retired_claim"] == "Pro costs $12 a month."
    assert result_of(4)["assumptions"][0]["derived_from"].startswith("Pro is $15")
    assert result_of(5)["ok"] and result_of(5)["quality_tier"] == "whitelist"
    assert result_of(6) == {"ok": False, "code": "contradicted", "error": "A stated fact contradicts it (or a conflict is open on it); a person decides this one."}
    assert result_of(8)["ok"] and result_of(8)["task_id"].startswith("cog_")
    assert result_of(9)["ok"]
    # The bank moved exactly as the tools said.
    c = await store.get_knowledge_conflict(ids["conflict"])
    assert c.status == "resolved" and c.resolution == "superseded"
    assert (await store.get_fact("f_old")).grating_strength == 0.0
    good = next(r for r in await store.list_assumption_crystals(customer.id) if r["id"] == ids["good"])
    assert assumption_state(good) == "approved" and good["quality_tier"] == "whitelist"
    assert parse_assumption_tags(good["diagnostic_tags"])["approved_by"] == "steward"
    bad = next(r for r in await store.list_assumption_crystals(customer.id) if r["id"] == ids["bad"])
    assert assumption_state(bad) == "waiting"
    assert (await store.get_knowledge_gap(ids["g2"])).status == "closed"
    tasks = [t for t in await store.list_cognition_tasks(customer.id) if t.task_type == "research"]
    assert len(tasks) == 1 and tasks[0].payload["gap_id"] == ids["g1"] and tasks[0].payload["actor"] == "steward"
    # The ledger: four actions, each with the inverse that undoes it.
    actions = await store.list_steward_actions(customer.id)
    assert [a["action"] for a in actions] == ["dismiss_gap", "research_gap", "approve_assumption", "decide_conflict"]
    assert all(a["undoable"] and a["reason"] for a in actions)
    assert actions[-1]["inverse"] == {"reopen_conflict": ids["conflict"], "resolution": "superseded", "loser": "a"}
    assert actions[2]["inverse"] == {"quality_tier": "quarantine", "recall_gated": False, "gap_filled": None}
    assert actions[1]["inverse"]["cancel_task"] == tasks[0].id
    # The run is a cognition task of type steward, complete, with the brief and the trace.
    run = await store.get_cognition_task(out["run_id"])
    assert run.task_type == "steward" and run.status == "complete" and run.result["brief"] == out["brief"]
    assert run.result["actions"][0]["action"] == "decide_conflict" and run.result["calls"] == 10
    assert run.payload["budget"] == {"tool_calls": 14, "actions": 6}
    # Metered once, origin steward; the witness feed saw every step.
    calls = await store.list_llm_calls(customer.id)
    assert len(calls) == 1 and calls[0]["origin"] == "steward" and calls[0]["session_id"] == out["run_id"] and calls[0]["input_tokens"] > 0
    events = [e["event_type"] for e in await store.list_curation_events(customer.id, limit=50)]
    assert events.count("steward_action") == 4 and "steward_session" in events and "assumption_approved" in events


@pytest.mark.asyncio
async def test_a_person_undoes_each_action_within_the_window(store, customer):
    ids = await _seed(store, customer)
    anth = FakeAnthropic()
    _script(anth, ids)
    out = await mod.run_steward_session(store=store, customer=customer, encoder=_Encoder(), fact_store=None, llm=anth)
    by_action = {a["action"]: a for a in await store.list_steward_actions(customer.id)}

    r = await mod.undo_steward_action(store, customer.id, by_action["decide_conflict"]["id"])
    assert r["undone"] and r["action"]["undone_at"] and not r["action"]["undoable"]
    assert (await store.get_knowledge_conflict(ids["conflict"])).status == "open"
    assert (await store.get_fact("f_old")).grating_strength == 1.0

    r = await mod.undo_steward_action(store, customer.id, by_action["approve_assumption"]["id"])
    assert r["undone"]
    good = next(x for x in await store.list_assumption_crystals(customer.id) if x["id"] == ids["good"])
    assert assumption_state(good) == "waiting" and good["quality_tier"] == "quarantine" and good["recall_gated"] is False
    assert parse_assumption_tags(good["diagnostic_tags"])["approved_by"] is None

    r = await mod.undo_steward_action(store, customer.id, by_action["research_gap"]["id"])
    assert r["undone"]
    task = await store.get_cognition_task(by_action["research_gap"]["inverse"]["cancel_task"])
    assert task.status == "cancelled"

    r = await mod.undo_steward_action(store, customer.id, by_action["dismiss_gap"]["id"])
    assert r["undone"] and (await store.get_knowledge_gap(ids["g2"])).status == "open"

    # Twice is refused; a foreign id is not found; the feed saw the undos.
    assert (await mod.undo_steward_action(store, customer.id, by_action["dismiss_gap"]["id"]))["code"] == "already_undone"
    assert (await mod.undo_steward_action(store, "cus_other", by_action["dismiss_gap"]["id"]))["code"] == "not_found"
    events = [e["event_type"] for e in await store.list_curation_events(customer.id, limit=60)]
    assert events.count("steward_undone") == 4 and "conflict_reopened" in events
    # The next session sees what was undone and the ledger says so.
    recent = await store.list_steward_actions(customer.id)
    assert all(a["undone_at"] for a in recent)
    assert out["run_id"]


@pytest.mark.asyncio
async def test_the_window_closes_after_a_day_and_reprobe_has_no_undo(store, customer):
    await _seed(store, customer)
    row = await store.record_steward_action(customer.id, run_id=None, action="dismiss_gap", subject_kind="gap", subject_id="g",
                                            label="x", reason="y", inverse={"reopen_gap": "g"})
    async with store.session() as s:
        from crystal_cache.infrastructure.schema import StewardActionRow
        r = await s.get(StewardActionRow, row["id"])
        r.created_at = datetime.now(timezone.utc) - timedelta(hours=25)
    assert (await mod.undo_steward_action(store, customer.id, row["id"]))["code"] == "undo_expired"
    probe = await store.record_steward_action(customer.id, run_id=None, action="reprobe_fact", subject_kind="fact", subject_id="f_new",
                                              label="x", reason="y", inverse=None, undoable=False)
    assert (await mod.undo_steward_action(store, customer.id, probe["id"]))["code"] == "not_undoable"


@pytest.mark.asyncio
async def test_the_tools_refuse_what_the_steward_may_not_do(store, customer):
    ids = await _seed(store, customer)
    session = mod.StewardSession(store=store, customer=customer, run_id="cog_x", encoder=_Encoder(), fact_store=None, max_actions=1)
    reg = mod.build_steward_registry(session)
    tool = {t.name: t for t in reg.list_for_context("agent")}
    cid = customer.id
    # No reason, no action.
    assert (await tool["dismiss_gap"].impl(customer_id=cid, gap_id=ids["g2"], reason="ok"))["code"] == "reason_required"
    # superseded needs the loser; a settled conflict cannot be settled again.
    assert (await tool["decide_conflict"].impl(customer_id=cid, conflict_id=ids["conflict"], resolution="superseded", reason="The newer claim wins here."))["code"] == "loser_required"
    assert (await tool["decide_conflict"].impl(customer_id=cid, conflict_id=ids["conflict"], resolution="qualified", reason="Both are true in different years."))["ok"]
    assert (await tool["decide_conflict"].impl(customer_id=cid, conflict_id=ids["conflict"], resolution="dismissed", reason="Trying to settle it twice now."))["code"] == "session_action_cap"
    session.max_actions = 6
    assert (await tool["decide_conflict"].impl(customer_id=cid, conflict_id=ids["conflict"], resolution="dismissed", reason="Trying to settle it twice now."))["code"] == "not_open"
    # Another tenant's things do not exist; a gap already queued is not queued twice.
    assert (await tool["approve_assumption"].impl(customer_id="cus_other", crystal_id=ids["good"], reason="Not my bank, should fail."))["code"] == "not_found"
    assert (await tool["research_gap"].impl(customer_id=cid, gap_id=ids["g1"], reason="Public rules; the web answers this."))["ok"]
    assert (await tool["research_gap"].impl(customer_id=cid, gap_id=ids["g1"], reason="Asking again for the same gap now."))["code"] == "already_researching"
    # reprobe drops the probe memo so the implication scan looks again.
    await store.record_scan_verdicts(cid, detector="implication_probe", verdicts=[("f_new", "PROBED")])
    out = await tool["reprobe_fact"].impl(customer_id=cid, fact_id="f_new", reason="New facts landed around it this week.")
    assert out["ok"] and out["was_probed"] is True
    assert await store.recorded_scan_verdict_keys(cid, detector="implication_probe", pair_keys=["f_new"]) == set()
    assert (await store.list_steward_actions(cid))[0]["undoable"] is False


@pytest.mark.asyncio
async def test_the_worker_runs_one_session_per_enabled_tenant_and_asks_for_a_refill_at_the_wall(store, customer, monkeypatch):
    from crystal_cache.llm.client import reset_llm_client, set_llm_client
    from crystal_cache.workers import cognition as worker

    ids = await _seed(store, customer)
    other = await store.create_customer(provider="anthropic", model_id="m", api_key_ref="")
    anth = FakeAnthropic()
    _script(anth, ids)
    set_llm_client(anth)
    try:
        state = {"day": None, "sessions": {}, "paused": {}}
        # Nobody opted in: nothing runs, nothing is spent.
        assert await worker._run_steward(store=store, encoder=_Encoder(), fact_store=None, steward_state=state, sessions_per_day=2) == 0
        assert anth.calls == []
        await store.set_customer_steward_enabled(customer.id, True)
        assert (await store.get_customer_by_id(customer.id)).steward_enabled is True
        assert await worker._run_steward(store=store, encoder=_Encoder(), fact_store=None, steward_state=state, sessions_per_day=2) == 1
        assert state["sessions"] == {customer.id: 1} and len(anth.calls) == 11
        # Second session: the model only orients and stops (nothing left to do safely).
        anth.script_tool_use("bank_status", {}, "tu_a")
        anth.script_text("Did: nothing.\nLeft alone: the refund consequence.\nNext: nothing.")
        assert await worker._run_steward(store=store, encoder=_Encoder(), fact_store=None, steward_state=state, sessions_per_day=2) == 1
        # Third: the day's sessions are used; the refill ask is written once.
        assert await worker._run_steward(store=store, encoder=_Encoder(), fact_store=None, steward_state=state, sessions_per_day=2) == 0
        assert await worker._run_steward(store=store, encoder=_Encoder(), fact_store=None, steward_state=state, sessions_per_day=2) == 0
        paused = [e for e in await store.list_curation_events(customer.id, limit=60) if e["event_type"] == "steward_paused"]
        assert len(paused) == 1 and paused[0]["payload"]["code"] == "session_cap"
        # A new day: the plan's door shuts instead (managed tenant out of capacity).
        from crystal_cache.control.admission import PlanWallError
        from crystal_cache.ingress import auth as auth_mod

        async def shut(customer_, store_, *, spend=False, incoming=0):
            raise PlanWallError("daily_capacity", "Daily AI capacity is used up for this plan.")
        monkeypatch.setattr(auth_mod, "require_write_capacity", shut)
        state2 = {"day": None, "sessions": {}, "paused": {}}
        assert await worker._run_steward(store=store, encoder=_Encoder(), fact_store=None, steward_state=state2, sessions_per_day=2) == 0
        paused = [e for e in await store.list_curation_events(customer.id, limit=60) if e["event_type"] == "steward_paused"]
        assert len(paused) == 2 and paused[0]["payload"]["code"] == "daily_capacity"
        assert other.id not in state2["paused"]
    finally:
        reset_llm_client()


@pytest.mark.asyncio
async def test_the_console_read_the_switch_and_the_undo_route(store, customer):
    from crystal_cache.endpoints.steward import get_steward, set_steward, undo_steward

    ids = await _seed(store, customer)
    anth = FakeAnthropic()
    _script(anth, ids)
    await mod.run_steward_session(store=store, customer=customer, encoder=_Encoder(), fact_store=None, llm=anth)
    req = SimpleNamespace(state=SimpleNamespace())
    body = json.loads((await get_steward(customer.id, req, store)).body)
    assert body["enabled"] is False and body["platform_enabled"] is True and body["paused"] is None
    assert body["today"] == {"sessions": 1, "actions": 4, "sessions_per_day": 12}
    assert body["latest"]["brief"].startswith("Did:") and body["latest"]["actions"] == 4 and body["latest"]["status"] == "complete"
    assert len(body["actions"]) == 4 and body["actions"][0]["undoable"] is True

    class _Req:
        def __init__(self, b): self._b = b; self.state = SimpleNamespace()
        async def json(self): return self._b
    with pytest.raises(HTTPException) as e:
        await set_steward(customer.id, _Req({"enabled": "yes"}), store)
    assert e.value.status_code == 400
    assert json.loads((await set_steward(customer.id, _Req({"enabled": True}), store)).body) == {"customer_id": customer.id, "enabled": True}
    assert json.loads((await get_steward(customer.id, req, store)).body)["enabled"] is True

    aid = body["actions"][0]["id"]
    out = json.loads((await undo_steward(customer.id, aid, req, store)).body)
    assert out["undone"] and out["action"]["undone_at"]
    with pytest.raises(HTTPException) as e:
        await undo_steward(customer.id, aid, req, store)
    assert e.value.status_code == 409 and e.value.detail["error"]["code"] == "already_undone"
    with pytest.raises(HTTPException) as e:
        await undo_steward(customer.id, "stw_nope", req, store)
    assert e.value.status_code == 404


@pytest.mark.asyncio
async def test_an_orphaned_steward_run_never_runs_as_research(store, customer):
    from crystal_cache.workers.cognition import _process_pending_tasks

    task = await store.create_cognition_task(customer.id, task_type="steward", payload={"goal": "improve this bank"})
    n = await _process_pending_tasks(store=store, fact_vector_store=None, encoder=None, max_tasks=5)
    assert n == 1
    after = await store.get_cognition_task(task.id)
    assert after.status == "failed" and "orphaned" in (after.error_message or "")
