"""Q98 B (2026-10-22): implication probing. Two facts that contradict only
through one step of reasoning and share no meaning in embedding space
never meet the pairwise judge; this scan derives each new fact's
consequences and search questions, searches the bank with them, and
judges the hits across the bridge. The light model is scripted and the
encoder is a stub whose vectors are chosen so that the QUESTION, not the
fact, lands on the water fact."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from crystal_cache.infrastructure.schema import CrystalRow, FactRow
from crystal_cache.scan import implication as mod
from fakes import SeamFake

_T0 = datetime.now(timezone.utc) - timedelta(hours=1)
D = 8


def _vec(*on: int) -> list[float]:
    v = np.zeros(D, dtype=np.float32)
    for i in on:
        v[i] = 1.0
    return v.tolist()


class _Encoder:
    """Texts map to fixed directions: the oxygen fact and the water fact are
    orthogonal (no similarity), the oxygen question points at water."""

    table = {
        "Which compounds contain oxygen?": _vec(3),
        "No liquid substance contains oxygen": _vec(3, 5),
        "What is the launch date?": _vec(6),
    }

    def encode_native(self, text: str) -> np.ndarray:
        return np.asarray(self.table.get(text, _vec(7)), dtype=np.float32)


class _Scripted(SeamFake):
    """The derive call answers from a table keyed by the fact; the bridge
    judge answers CONTRADICTS when the water fact is B."""

    def __init__(self):
        self.derives = []
        self.checks = []
        self.judges = []

    def complete(self, **kwargs):
        system = kwargs.get("system") or ""
        user = kwargs["messages"][0]["content"]
        if system.startswith("You read ONE stored fact"):
            self.derives.append(user)
            if "Oxygen only exists" in user:
                return json.dumps({"consequences": ["No liquid substance contains oxygen"],
                                   "questions": ["Which compounds contain oxygen?"]})
            if "launch" in user.lower():
                return json.dumps({"consequences": [], "questions": ["What is the launch date?"]})
            if "Pro is $18" in user:
                # The derive model invents a plan the fact never named.
                return json.dumps({"consequences": ["Pro is a recurring monthly charge", "The Team plan costs more than Pro"],
                                   "questions": []})
            return json.dumps({"consequences": [], "questions": []})
        if system.startswith("You grade whether each proposed CONSEQUENCE"):
            self.checks.append(user)
            grades = []
            for line in user.split("CONSEQUENCES:")[-1].strip().splitlines():
                text = line.split(". ", 1)[-1].strip()
                grade = "SPECULATES" if "Team plan" in text else "FOLLOWS"
                grades.append({"consequence": text, "grade": grade})
            return json.dumps({"grades": grades})
        self.judges.append(user)
        if "Water is made of" in user and "CONSEQUENCE OF A: No liquid substance contains oxygen" in user:
            return json.dumps({"reason": "Water is a liquid that contains oxygen, which the consequence rules out.", "verdict": "CONTRADICTS"})
        return json.dumps({"reason": "No bearing.", "verdict": "UNRELATED"})


async def _seed(store, customer_id):
    async with store.session() as s:
        s.add(CrystalRow(id="cx", customer_id=customer_id, crystal_type="customer:legacy", summary_vector=[]))
        s.add(CrystalRow(id="cy", customer_id=customer_id, crystal_type="customer:legacy", summary_vector=[]))
        s.add(FactRow(id="f_water", crystal_id="cx", pair_type="question_answer", prompt_text="Chemistry|Water|Composition",
                      claim_text="Water is made of two hydrogen and one oxygen.", source_kind="document_chunk",
                      source_doc_id="doc_chem", vector=_vec(3), created_at=_T0))
        s.add(FactRow(id="f_date", crystal_id="cy", pair_type="question_answer", prompt_text="Kerning|Launch|Date",
                      claim_text="Launch day is November 12.", source_kind="document_chunk",
                      source_doc_id="doc_journal", vector=_vec(6), created_at=_T0 + timedelta(minutes=1)))
        s.add(FactRow(id="f_chunk", crystal_id="cy", pair_type="content_chunk", prompt_text="",
                      claim_text="a passage", source_kind="document_chunk", vector=_vec(0), created_at=_T0))
        s.add(FactRow(id="f_oxygen", crystal_id="cy", pair_type="question_answer", prompt_text="Chemistry|Oxygen|States",
                      claim_text="Oxygen only exists as a gas.", source_kind="model_reasoning",
                      vector=_vec(1), created_at=_T0 + timedelta(minutes=2)))


@pytest.mark.asyncio
async def test_a_consequence_finds_what_similarity_never_would(store, customer):
    await _seed(store, customer.id)
    llm = _Scripted()
    res = await mod.scan_by_implication(store=store, encoder=_Encoder(), customer_id=customer.id,
                                        llm_client=llm, max_facts=10, hits_per_fact=3, min_score=0.45)
    # Three usable facts (the chunk is never a fact), each probed once.
    assert res.facts_considered == 3 and res.facts_probed == 3 and res.skipped_probed == 0
    assert len(llm.derives) == 3 and any("filed under Chemistry > Oxygen > States" in d for d in llm.derives)
    # The checker graded the one consequence FOLLOWS (one check call, nothing dropped).
    assert res.checks == 1 and res.dropped == 0 and len(llm.checks) == 1 and "No liquid substance" in llm.checks[0]
    # The oxygen fact's consequence and question both land on water; the launch
    # question lands on nothing new (its only hit is the fact itself, excluded).
    assert res.searches == 3 and res.hits == 1 and res.conflicts_found == 1
    assert res.calls == res.facts_probed + res.checks + res.hits == 5
    judge = llm.judges[0]
    assert "FACT A (filed under Chemistry > Oxygen > States" in judge
    assert "CONSEQUENCE OF A: No liquid substance contains oxygen" in judge
    assert "FACT B (filed under Chemistry > Water > Composition" in judge and "Water is made of" in judge
    rows = await store.list_knowledge_conflicts(customer.id, status="open")
    assert len(rows) == 1
    c = rows[0]
    assert c.detector == "implication_scan" and {c.fact_a_id, c.fact_b_id} == {"f_oxygen", "f_water"}
    assert c.reasoning.startswith("A implies: No liquid substance contains oxygen") and "Water is a liquid" in c.reasoning
    assert c.subject == "Chemistry" and c.recommendation is None
    # The second run probes nothing (memo per fact) and spends nothing.
    again = await mod.scan_by_implication(store=store, encoder=_Encoder(), customer_id=customer.id, llm_client=llm, max_facts=10)
    assert again.facts_probed == 0 and again.skipped_probed == 3 and again.hits == 0 and len(llm.derives) == 3


@pytest.mark.asyncio
async def test_budget_memo_and_fail_safes(store, customer):
    await _seed(store, customer.id)
    llm = _Scripted()
    # Budget: one fact per run, newest first; the rest wait for the next cycle.
    res = await mod.scan_by_implication(store=store, encoder=_Encoder(), customer_id=customer.id, llm_client=llm, max_facts=1)
    assert res.facts_probed == 1 and res.budget_exhausted and "Oxygen only exists" in llm.derives[0]
    assert res.conflicts_found == 1
    res = await mod.scan_by_implication(store=store, encoder=_Encoder(), customer_id=customer.id, llm_client=llm, max_facts=5)
    assert res.facts_probed == 2 and res.skipped_probed == 1 and not res.budget_exhausted
    # A judged non-contradiction is memoized per pair; an existing conflict is never re-opened.
    assert (await store.list_knowledge_conflicts(customer.id, status="open")) and res.conflicts_found == 0
    assert res.calls == res.facts_probed + res.hits

    class _Broken(SeamFake):
        def complete(self, **kwargs):
            raise RuntimeError("upstream down")
    # A derive failure probes nothing and leaves the fact for next time; no exception escapes.
    async with store.session() as s:
        s.add(FactRow(id="f_new", crystal_id="cy", pair_type="question_answer", prompt_text="Kerning|Pricing|Pro",
                      claim_text="Pro is $18 a month.", source_kind="model_reasoning", vector=_vec(2), created_at=_T0 + timedelta(minutes=3)))
    res = await mod.scan_by_implication(store=store, encoder=_Encoder(), customer_id=customer.id, llm_client=_Broken(), max_facts=5)
    assert res.facts_probed == 1 and res.searches == 0 and res.hits == 0
    # The failed derive did not mark the fact probed: the next cycle tries it again,
    # and the checker drops the invented "Team plan" consequence before any search.
    res = await mod.scan_by_implication(store=store, encoder=_Encoder(), customer_id=customer.id, llm_client=llm, max_facts=5)
    assert res.facts_probed == 1 and res.skipped_probed == 3 and "Pro is $18" in llm.derives[-1]
    assert res.checks == 1 and res.dropped == 1 and res.searches == 1  # only the FOLLOWS consequence was searched
    assert "Team plan" in llm.checks[-1]


@pytest.mark.asyncio
async def test_a_failed_check_searches_nothing(store, customer):
    """An ungraded consequence is never searched: the checker's failure keeps
    the invented ones out at the cost of one cycle."""
    await _seed(store, customer.id)

    class _NoCheck(_Scripted):
        def complete(self, **kwargs):
            if (kwargs.get("system") or "").startswith("You grade whether"):
                raise RuntimeError("checker down")
            return super().complete(**kwargs)
    llm = _NoCheck()
    res = await mod.scan_by_implication(store=store, encoder=_Encoder(), customer_id=customer.id, llm_client=llm, max_facts=1)
    assert res.facts_probed == 1 and res.checks == 1 and res.dropped == 1
    # The question still searches (a search alone never writes a row), with no consequence to bridge.
    assert res.searches == 1 and res.hits == 1 and res.conflicts_found == 0
    assert "(A rules out the answer to: Which compounds contain oxygen?)" in llm.judges[-1]


@pytest.mark.asyncio
async def test_unknown_detector_is_refused_at_the_write(store, customer):
    """The poisoned-read class: a detector the reader's Literal does not
    carry is refused by the writer (implication_scan is carried)."""
    with pytest.raises(ValueError, match="unknown conflict detector"):
        await store.create_knowledge_conflict(customer.id, fact_a_id="x", fact_b_id="y", claim_a="a", claim_b="b",
                                              pair_key="k1", detector="staleness_scan")
    c = await store.create_knowledge_conflict(customer.id, fact_a_id="x", fact_b_id="y", claim_a="a", claim_b="b",
                                              pair_key="k2", detector="implication_scan")
    assert c.detector == "implication_scan"


def test_index_excludes_self_and_same_document():
    class F:
        def __init__(self, fid, vec, doc=None):
            self.id = fid; self.vector = vec; self.source_doc_id = doc; self.claim_text = fid; self.pair_type = "question_answer"
    idx = mod._Index([F("a", _vec(3), "d1"), F("b", _vec(3), "d1"), F("c", _vec(3), "d2"), F("d", _vec(5))])
    hits = idx.search(np.asarray(_vec(3), dtype=np.float32), k=5, exclude_id="a", exclude_doc="d1", min_score=0.45)
    assert [f.id for f, _ in hits] == ["c"]
    assert idx.search(np.asarray(_vec(3) + [0.0], dtype=np.float32), k=5, exclude_id="a", exclude_doc=None, min_score=0.1) == []
