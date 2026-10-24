"""Q102 (2026-10-23): the scans stop making their own noise.

Stewarding the demo bank by hand found the queues mostly self-made: about
40 of 82 open conflicts were two facts that did not disagree, about 30 were
the same fact twice, consequences overreached ('always exactly $30'), came
back after being retired, and 'agreed' with unrelated facts; 48 of 76 gaps
asked what the bank already said or what a function does with empty input.
Each fix is pinned here against the case that exposed it.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from crystal_cache.cognition import controls
from crystal_cache.infrastructure.metadata_store_assumption_ext import normalize_statement
from crystal_cache.infrastructure.schema import CrystalRow, FactRow
from crystal_cache.scan import discover_gaps, scan_for_duplicates
from crystal_cache.scan import implication as imp
from crystal_cache.scan.dedup import merge_loser
from fakes import SeamFake

_T0 = datetime.now(timezone.utc) - timedelta(hours=2)
D = 8


def _vec(*on: int, w: float = 1.0) -> list[float]:
    v = np.zeros(D, dtype=np.float32)
    for i in on:
        v[i] = w
    return v.tolist()


# --- the prompts say the new rules ------------------------------------------

def test_the_prompts_forbid_overreach_and_mean_support_by_consistent():
    assert "absolutes" not in imp.DERIVE_SYSTEM and "NEVER stronger than the fact" in imp.DERIVE_SYSTEM
    assert "OVERREACHES" in imp.CHECK_SYSTEM
    assert imp.CHECK_SCHEMA["properties"]["grades"]["items"]["properties"]["grade"]["enum"] == [
        "FOLLOWS", "SPECULATES", "OVERREACHES", "DOESNT"]
    assert "SUPPORTS the consequence" in imp.BRIDGE_SYSTEM
    assert "When in doubt, answer UNRELATED." in imp.BRIDGE_SYSTEM
    assert "answer CONSISTENT" not in imp.BRIDGE_SYSTEM


# --- duplicates ----------------------------------------------------------------

def test_merge_loser_keeps_the_fuller_copy_and_never_guesses():
    # One copy's words all sit in the other: it goes.
    assert merge_loser("Exports keep working after a customer cancels.",
                       "Exports keep working after a customer cancels: an export is a file, not a session.")[0] == "a"
    assert merge_loser("A workspace holds brands. The share link (in progress) gives read access to one token set at a URL without an account.",
                       "The share link (in progress) gives read access to one token set at a URL without an account.")[0] == "b"
    # Same words: the shorter text goes.
    assert merge_loser("Any deploy after noon on a Friday is prohibited",
                       "Any deploy after noon on a Friday is prohibited.")[0] == "a"
    # A derived copy loses to a stated one, whatever the words.
    assert merge_loser("No calls are scheduled on Wednesdays", "Wednesday is no-calls day for deep work.",
                       key_a="Consequences|Founder", key_b="Founder|Rhythm|Wednesday")[0] == "a"
    # Different scope: left for a person.
    assert merge_loser("No deployment can be initiated on Friday after 12:00 PM",
                       "No deployment can start after 12:00 PM") is None
    assert merge_loser("Pro is $15 a month", "Pro is $12 a month") is None


class _Dup(SeamFake):
    def complete(self, **kwargs):
        system = kwargs.get("system") or ""
        if "Explain" in system:
            return json.dumps({"why": "Both say exports outlive cancellation.", "resolution": "superseded",
                               "loser": "a", "reason": "B is fuller."})
        return "DUPLICATE"


async def _seed_two(store, cid, claim_a, claim_b, key_a="Kerning|Exports|Cancel", key_b="Kerning|Exports|After"):
    async with store.session() as s:
        s.add(CrystalRow(id="cA", customer_id=cid, crystal_type="customer:legacy", summary_vector=[]))
        s.add(FactRow(id="f1", crystal_id="cA", pair_type="question_answer", prompt_text=key_a, claim_text=claim_a,
                      source_kind="document_chunk", vector=[], created_at=_T0))
        s.add(FactRow(id="f2", crystal_id="cA", pair_type="question_answer", prompt_text=key_b, claim_text=claim_b,
                      source_kind="document_chunk", vector=[], created_at=_T0 + timedelta(minutes=1)))


@pytest.mark.asyncio
async def test_a_clear_duplicate_is_merged_by_crystal_on_the_ledger_with_undo(store, customer):
    cid = customer.id
    await _seed_two(store, cid, "Exports keep working after a customer cancels.",
                    "Exports keep working after a customer cancels: an export is a file, not a session.")
    res = await scan_for_duplicates(store=store, slm_client=_Dup(), customer_id=cid)
    assert res.duplicates_found == 1 and res.auto_merged == 1
    assert await store.list_knowledge_conflicts(cid, status="open") == []
    assert (await store.get_fact("f1")).grating_strength == 0.0  # the thinner copy left recall
    ledger = await store.list_steward_actions(cid, limit=5)
    assert len(ledger) == 1 and ledger[0]["actor"] == "system" and ledger[0]["undoable"]
    assert ledger[0]["label"].startswith("Crystal settled a conflict")
    assert "adds nothing" in ledger[0]["reason"]
    events = [e["event_type"] for e in await store.list_curation_events(cid, limit=20)]
    assert "auto_action" in events
    # A person can take it back.
    out = await controls.undo_action(store, cid, ledger[0]["id"], actor="chat")
    assert out.get("undone") and len(await store.list_knowledge_conflicts(cid, status="open")) == 1


@pytest.mark.asyncio
async def test_a_duplicate_with_different_scope_stays_open(store, customer):
    cid = customer.id
    await _seed_two(store, cid, "No deployment can be initiated on Friday after 12:00 PM",
                    "No deployment can start after 12:00 PM")
    res = await scan_for_duplicates(store=store, slm_client=_Dup(), customer_id=cid)
    assert res.duplicates_found == 1 and res.auto_merged == 0
    assert len(await store.list_knowledge_conflicts(cid, status="open")) == 1


def test_crystal_is_the_system_actor():
    assert controls.who("system") == "Crystal" and controls.who("steward") == "Steward" and controls.who("chat:op_1") == "Chat"


# --- a duplicate never blocks an approval --------------------------------------

class _Enc:
    def encode_native(self, text):
        return np.asarray(_vec(7), dtype=np.float32)

    def encode(self, text):
        return self.encode_native(text)

    def fingerprint(self):
        return "stub"


@pytest.mark.asyncio
async def test_an_open_duplicate_does_not_block_approval_but_a_contradiction_does(store, customer):
    cid = customer.id
    async with store.session() as s:
        s.add(CrystalRow(id="c_rhythm", customer_id=cid, crystal_type="customer:legacy", summary_vector=[], quality_tier="neutral"))
        s.add(FactRow(id="f_wed", crystal_id="c_rhythm", pair_type="question_answer", prompt_text="Founder|Rhythm|Wednesday",
                      claim_text="Wednesday is no-calls day for deep work.", source_kind="document_chunk", vector=_vec(1), created_at=_T0))
    asm = await store.create_consequence_crystal(cid, statement="Wednesday is reserved for deep work", subject="Founder",
                                                 parent_fact_id="f_wed", parent_crystal_id="c_rhythm", encoder=_Enc())
    await store.create_knowledge_conflict(cid, fact_a_id="f_wed", fact_b_id="f_x", claim_a="a", claim_b="b", pair_key="pk_dup",
                                          crystal_a_id="c_rhythm", crystal_b_id=asm, subject="Founder", detector="dedup_scan")
    assert await store.count_open_conflicts_for_crystal(cid, asm) == 1
    assert await store.count_open_contradictions_for_crystal(cid, asm) == 0
    out = await controls.approve_assumption(store, cid, crystal_id=asm, reason="Restates the stated Wednesday rule.", actor="chat")
    assert out["ok"]
    other = await store.create_consequence_crystal(cid, statement="Calls happen on Wednesdays", subject="Founder",
                                                   parent_fact_id="f_wed", parent_crystal_id="c_rhythm", encoder=_Enc())
    await store.create_knowledge_conflict(cid, fact_a_id="f_wed", fact_b_id="f_y", claim_a="a", claim_b="b", pair_key="pk_con",
                                          crystal_a_id="c_rhythm", crystal_b_id=other, subject="Founder", detector="implication_scan")
    out = await controls.approve_assumption(store, cid, crystal_id=other, reason="Testing the contradiction gate.", actor="chat")
    assert out["code"] == "contradicted"


# --- consequences are written once, and never as restatements ------------------

class _Encoder:
    """Two facts say the same Wednesday rule (as two documents did); a third
    is unrelated. 'No calls are scheduled on Wednesdays' sits near (0.7)
    the second fact; 'Fridays end the week' is a restatement of the third."""
    table = {
        "No calls are scheduled on Wednesdays": _vec(3, 6),
        "Fridays end the week": _vec(5),
    }

    def encode_native(self, text):
        return np.asarray(self.table.get(text, _vec(7)), dtype=np.float32)

    def encode(self, text):
        return self.encode_native(text)

    def fingerprint(self):
        return "stub"


class _Scripted(SeamFake):
    def complete(self, **kwargs):
        system = kwargs.get("system") or ""
        user = kwargs["messages"][0]["content"]
        if system.startswith("You read ONE stored fact"):
            if "Wednesday" in user:
                return json.dumps({"consequences": ["No calls are scheduled on Wednesdays"], "questions": []})
            if "Friday" in user:
                return json.dumps({"consequences": ["Fridays end the week"], "questions": []})
            return json.dumps({"consequences": [], "questions": []})
        if system.startswith("You grade whether each proposed CONSEQUENCE"):
            line = user.split("CONSEQUENCES:")[-1].strip().split(". ", 1)[-1]
            return json.dumps({"grades": [{"consequence": line, "grade": "FOLLOWS"}]})
        return json.dumps({"reason": "No bearing.", "verdict": "UNRELATED"})


async def _seed_rhythm(store, cid):
    async with store.session() as s:
        s.add(CrystalRow(id="c_r1", customer_id=cid, crystal_type="customer:legacy", summary_vector=[], quality_tier="neutral"))
        s.add(CrystalRow(id="c_r2", customer_id=cid, crystal_type="customer:legacy", summary_vector=[], quality_tier="neutral"))
        s.add(FactRow(id="f_wed1", crystal_id="c_r1", pair_type="question_answer", prompt_text="Rhythm|Wednesday",
                      claim_text="Wednesday: no calls, deep work.", source_kind="document_chunk", source_doc_id="d1",
                      vector=_vec(1), created_at=_T0))
        s.add(FactRow(id="f_wed2", crystal_id="c_r2", pair_type="question_answer", prompt_text="Operating rhythm|Wednesday",
                      claim_text="Wednesday is designated as no calls day.", source_kind="document_chunk", source_doc_id="d2",
                      vector=_vec(3), created_at=_T0 + timedelta(minutes=1)))
        s.add(FactRow(id="f_fri", crystal_id="c_r2", pair_type="question_answer", prompt_text="Operating rhythm|Friday",
                      claim_text="Friday is ship or cut.", source_kind="document_chunk", source_doc_id="d3",
                      vector=_vec(5), created_at=_T0 + timedelta(minutes=2)))


async def _scan_all(store, cid):
    total_written = total_skipped = 0
    while True:
        res = await imp.scan_by_implication(store=store, encoder=_Encoder(), customer_id=cid, llm_client=_Scripted(),
                                            max_facts=1, hits_per_fact=3, min_score=0.45)
        total_written += res.consequences_written
        total_skipped += res.consequences_skipped
        if res.facts_probed == 0:
            return total_written, total_skipped


@pytest.mark.asyncio
async def test_the_same_consequence_from_two_facts_is_written_once_and_a_restatement_never(store, customer):
    cid = customer.id
    await _seed_rhythm(store, cid)
    written, skipped = await _scan_all(store, cid)
    rows = await store.list_assumption_crystals(cid)
    assert [r["statement"] for r in rows] == ["No calls are scheduled on Wednesdays"]  # once, from either Wednesday fact
    assert written == 1 and skipped == 2  # the second Wednesday copy, and the Friday restatement


@pytest.mark.asyncio
async def test_a_rejected_consequence_is_never_written_again(store, customer):
    cid = customer.id
    await _seed_rhythm(store, cid)
    gone = await store.create_consequence_crystal(cid, statement="No calls are scheduled on Wednesdays.", subject="Rhythm",
                                                  parent_fact_id="f_old", parent_crystal_id="c_r1", encoder=_Encoder())
    await controls.reject_assumption(store, cid, crystal_id=gone, reason="Rejected by a person earlier today.", actor="chat")
    written, _ = await _scan_all(store, cid)
    assert written == 0
    assert normalize_statement("No calls are scheduled on Wednesdays.") == normalize_statement("no calls are  scheduled on wednesdays")
    assert [r["id"] for r in await store.list_assumption_crystals(cid)] == [gone]


# --- gaps ----------------------------------------------------------------------

class _Gaps(SeamFake):
    def __init__(self, answered: bool):
        self.answered = answered
        self.calls = 0

    def complete(self, **kwargs):
        self.calls += 1
        system = kwargs.get("system") or ""
        if system.startswith("You decide whether a knowledge base already answers"):
            return "ANSWERED" if self.answered else "OPEN"
        return "What is the founder's name?"


async def _seed_gap_bank(store, cid, *, code=False):
    head = "Code|src/a.ts" if code else "Founder"
    async with store.session() as s:
        s.add(CrystalRow(id="cG", customer_id=cid, crystal_type="customer:legacy", summary_vector=[]))
        s.add(FactRow(id="g1", crystal_id="cG", pair_type="question_answer", prompt_text=f"{head}|Profile|Role",
                      claim_text="Runs product, code, support and sales alone.", source_kind="document_chunk",
                      vector=_vec(1), created_at=_T0))
        s.add(FactRow(id="g2", crystal_id="cG", pair_type="question_answer", prompt_text=f"{head}|Profile|Base",
                      claim_text="Based in Portland, Oregon.", source_kind="document_chunk",
                      vector=_vec(2), created_at=_T0 + timedelta(minutes=1)))
        s.add(FactRow(id="g3", crystal_id="cG", pair_type="question_answer", prompt_text="Who am I|Name",
                      claim_text="Dana Reyes, solo founder of Kerning.", source_kind="document_chunk",
                      vector=_vec(7), created_at=_T0 + timedelta(minutes=2)))


@pytest.mark.asyncio
async def test_a_gap_the_bank_answers_elsewhere_is_not_opened(store, customer):
    await _seed_gap_bank(store, customer.id)
    res = await discover_gaps(store=store, slm_client=_Gaps(answered=True), customer_id=customer.id, encoder=_Enc())
    assert res.gaps_found == 0 and res.already_answered == 1 and res.answer_checks == 1
    assert await store.list_knowledge_gaps(customer.id, status="open") == []


@pytest.mark.asyncio
async def test_a_real_gap_still_opens_and_without_an_encoder_nothing_changes(store, customer):
    await _seed_gap_bank(store, customer.id)
    res = await discover_gaps(store=store, slm_client=_Gaps(answered=False), customer_id=customer.id, encoder=_Enc())
    assert res.gaps_found == 1 and res.already_answered == 0 and res.answer_checks == 1


@pytest.mark.asyncio
async def test_code_only_subjects_are_never_asked(store, customer):
    await _seed_gap_bank(store, customer.id, code=True)
    fake = _Gaps(answered=False)
    res = await discover_gaps(store=store, slm_client=fake, customer_id=customer.id)
    assert res.code_subjects_skipped >= 1 and res.gaps_found == 0


def test_the_gap_prompt_rules_out_edge_cases_and_undecided_things():
    from crystal_cache.scan.gap_discovery import GAP_DISCOVERY_SYSTEM
    assert "empty, invalid, null" in GAP_DISCOVERY_SYSTEM and "not decided" in GAP_DISCOVERY_SYSTEM


# --- Q103 (2026-10-23): a second opinion, an absolute floor, meaning dedupe ----

def test_an_absolute_the_fact_never_said_is_caught_by_rule():
    from crystal_cache.scan.implication import overreaches
    assert overreaches("The change count will always be two for a rename",
                       "the display was corrected while the underlying change count remains two") == "always"
    assert overreaches("No other function orchestrates the conversion", "The main entry point that orchestrates it") == "no other"
    assert overreaches("No liquid substance contains oxygen", "Oxygen only exists as a gas.") is None
    assert overreaches("Oxygen is only ever a gas", "Oxygen only exists as a gas.") is None  # the fact says 'only'
    assert overreaches("Pro is billed every month", "Pro is $15 a month") is None


class _Bridge(SeamFake):
    """The light judge always says CONTRADICTS; the large model says
    whatever `large` is. Records the tiers it was asked at."""

    def __init__(self, large: str):
        self.large = large
        self.tiers = []

    def complete(self, **kwargs):
        system = kwargs.get("system") or ""
        user = kwargs["messages"][0]["content"]
        self.tiers.append(kwargs.get("tier"))
        if system.startswith("You read ONE stored fact"):
            if "Launch" in user:
                return json.dumps({"consequences": ["The product has not launched publicly"], "questions": []})
            return json.dumps({"consequences": [], "questions": []})
        if system.startswith("You grade whether each proposed CONSEQUENCE"):
            line = user.split("CONSEQUENCES:")[-1].strip().split(". ", 1)[-1]
            return json.dumps({"grades": [{"consequence": line, "grade": "FOLLOWS"}]})
        if system.startswith("You are the second opinion"):
            assert kwargs.get("tier") == "large"
            return json.dumps({"reason": "A plan for November next to a record of no launch so far.", "verdict": self.large})
        return json.dumps({"reason": "B says it launches.", "verdict": "CONTRADICTS"})


class _LaunchEnc:
    table = {"The product has not launched publicly": _vec(3, 6)}

    def encode_native(self, text):
        return np.asarray(self.table.get(text, _vec(7)), dtype=np.float32)

    def encode(self, text):
        return self.encode_native(text)

    def fingerprint(self):
        return "stub"


async def _seed_launch(store, cid):
    async with store.session() as s:
        s.add(CrystalRow(id="c_l", customer_id=cid, crystal_type="customer:legacy", summary_vector=[], quality_tier="neutral"))
        s.add(FactRow(id="f_growth", crystal_id="c_l", pair_type="question_answer", prompt_text="Metrics|Growth",
                      claim_text="Launch: none yet; growth is word of mouth.", source_kind="document_chunk", source_doc_id="d1",
                      vector=_vec(1), created_at=_T0))
        s.add(FactRow(id="f_date", crystal_id="c_l", pair_type="question_answer", prompt_text="Plans|Date",
                      claim_text="The product launches on Thursday, November 12.", source_kind="document_chunk", source_doc_id="d2",
                      vector=_vec(3), created_at=_T0 - timedelta(minutes=5)))


@pytest.mark.asyncio
async def test_a_contradiction_the_large_model_does_not_uphold_is_never_written(store, customer):
    cid = customer.id
    await _seed_launch(store, cid)
    llm = _Bridge(large="CONSISTENT")
    res = await imp.scan_by_implication(store=store, encoder=_LaunchEnc(), customer_id=cid, llm_client=llm,
                                        max_facts=1, hits_per_fact=3, min_score=0.45)
    assert res.hits == 1 and res.confirms == 1 and res.overruled == 1 and res.conflicts_found == 0
    assert res.calls == res.facts_probed + res.checks + res.hits + 1
    assert await store.list_knowledge_conflicts(cid, status="open") == []
    assert "large" in llm.tiers and llm.tiers.count("large") == 1
    # The consequence carries the second opinion's verdict, not the first.
    row = (await store.list_assumption_crystals(cid))[0]
    from crystal_cache.infrastructure.metadata_store_assumption_ext import parse_assumption_tags
    assert [c["verdict"] for c in parse_assumption_tags(row["diagnostic_tags"])["checks"]] == ["CONSISTENT"]


@pytest.mark.asyncio
async def test_a_contradiction_both_models_see_is_written_with_the_second_reason(store, customer):
    cid = customer.id
    await _seed_launch(store, cid)
    res = await imp.scan_by_implication(store=store, encoder=_LaunchEnc(), customer_id=cid, llm_client=_Bridge(large="CONTRADICTS"),
                                        max_facts=1, hits_per_fact=3, min_score=0.45)
    assert res.conflicts_found == 1 and res.overruled == 0
    c = (await store.list_knowledge_conflicts(cid, status="open"))[0]
    assert c.detector == "implication_scan" and "A plan for November" in (c.reasoning or "")


@pytest.mark.asyncio
async def test_no_second_opinion_means_no_conflict_and_a_retry(store, customer):
    cid = customer.id
    await _seed_launch(store, cid)

    class _Down(_Bridge):
        def complete(self, **kwargs):
            if (kwargs.get("system") or "").startswith("You are the second opinion"):
                raise RuntimeError("large tier down")
            return super().complete(**kwargs)

    res = await imp.scan_by_implication(store=store, encoder=_LaunchEnc(), customer_id=cid, llm_client=_Down(large="CONTRADICTS"),
                                        max_facts=1, hits_per_fact=3, min_score=0.45)
    assert res.conflicts_found == 0 and res.confirms == 0
    from crystal_cache.scan.contradiction import _memoized_keys, _pair_key
    a, b = await store.get_fact("f_growth"), await store.get_fact("f_date")
    assert await _memoized_keys(store, cid, detector=imp.DETECTOR, pair_keys=[_pair_key(a, b)]) == set()


class _QEnc:
    """The three ADR rewordings share a direction; the VAT question does not."""
    table = {
        "What is the new color format adopted this year?": _vec(2, 3),
        "What is the new color format specification?": _vec(2, 3, w=0.9),
        "What does Kerning charge for VAT in the EU?": _vec(5),
    }

    def encode_native(self, text):
        return np.asarray(self.table.get(text, _vec(7)), dtype=np.float32)

    def encode(self, text):
        return self.encode_native(text)

    def fingerprint(self):
        return "stub"


@pytest.mark.asyncio
async def test_a_reworded_open_gap_is_not_opened_again(store, customer):
    cid = customer.id
    await store.create_knowledge_gap(cid, domain="ADR 0006", subject="ADR 0006",
                                     missing="What is the new color format adopted this year?", full_key="ADR 0006")
    await _seed_gap_bank(store, cid)

    class _Ask(SeamFake):
        def complete(self, **kwargs):
            if (kwargs.get("system") or "").startswith("You decide whether"):
                return "OPEN"
            return "What is the new color format specification?"

    res = await discover_gaps(store=store, slm_client=_Ask(), customer_id=cid, encoder=_QEnc())
    assert res.gaps_found == 0 and res.deduped >= 1
    assert len(await store.list_knowledge_gaps(cid, status="open")) == 1
