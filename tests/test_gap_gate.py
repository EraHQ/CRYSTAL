"""The gap gate (Q134 A, 2026-10-26).

Stewarding the demo bank closed 16 of 32 gaps by hand: questions the bank
already answered ("What is the name of the product?" with Kerning named
throughout), the same ADR question three times, a question dismissed and
asked again. Seven writers opened gaps; one checked the bank, and every
check compared only against open gaps. Every writer now goes through
cognition/gap_gate.py.
"""
from __future__ import annotations

import ast
import pathlib
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from crystal_cache.cognition import gap_gate as gg
from crystal_cache.infrastructure.schema import CrystalRow, FactRow
from fakes import SeamFake

_T0 = datetime.now(timezone.utc) - timedelta(hours=1)
D = 8


def _vec(*on: int) -> np.ndarray:
    v = np.zeros(D, dtype=np.float32)
    for i in on:
        v[i] = 1.0
    return v


class _Enc:
    """Each text maps to a direction; unknown texts point nowhere useful."""

    def __init__(self, table: dict[str, np.ndarray]):
        self.table = table
        self.seen: list[str] = []

    def encode_native(self, text):
        self.seen.append(text)
        return self.table.get(text, _vec(7))

    def encode(self, text):
        return self.encode_native(text)

    def fingerprint(self):
        return "stub"


class _Index:
    """search_facts over (fact_id, crystal_id, vector)."""

    def __init__(self, facts: list[tuple[str, str, np.ndarray]]):
        self.facts = facts
        self.queries = 0

    async def search_facts(self, *, customer_id, query_vector, k=10, with_keys=False, **_kw):
        self.queries += 1
        q = np.asarray(query_vector, dtype=np.float32)
        out = []
        for fid, cid, v in self.facts:
            n = float(np.linalg.norm(q) * np.linalg.norm(v)) or 1.0
            out.append((fid, cid, "question_answer", float(q @ v) / n, None))
        return sorted(out, key=lambda t: -t[3])[:k]


class _Judge(SeamFake):
    def __init__(self, pick: int):
        self.pick = pick
        self.calls = 0

    def complete(self, **kwargs):
        self.calls += 1
        assert kwargs.get("tier") == "large"
        return f'{{"answer": {self.pick}, "why": "test"}}'


async def _seed(store, cid):
    """Ids carry the tenant so the same test runs on a shared Postgres."""
    t = cid[-8:]
    K, FN, FL = f"cK_{t}", f"f_name_{t}", f"f_launch_{t}"
    async with store.session() as s:
        s.add(CrystalRow(id=K, customer_id=cid, crystal_type="customer:legacy", summary_vector=[]))
        s.add(FactRow(id=FN, crystal_id=K, pair_type="question_answer", prompt_text="Kerning|Overview",
                      claim_text="Kerning turns a brand's style guide into design tokens.",
                      source_kind="document_chunk", vector=_vec(1).tolist(), created_at=_T0))
        s.add(FactRow(id=FL, crystal_id=K, pair_type="question_answer", prompt_text="Launch|Date",
                      claim_text="Kerning launches on November 12, 2026.",
                      source_kind="document_chunk", vector=_vec(2).tolist(), created_at=_T0))
    return K, FN, FL


def test_every_gap_writer_goes_through_the_gate():
    src = pathlib.Path(gg.__file__).resolve().parents[1]
    callers = []
    for path in src.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "create_knowledge_gap":
                callers.append(path.relative_to(src).as_posix())
    assert callers == ["cognition/gap_gate.py"], callers


def test_the_probes_reach_the_answer_not_the_question():
    assert gg.content_phrase("What is the name of the product?") == "name product"
    assert gg.key_phrase("Consequences|What is Kerning, what does my product do", None) == \
        "Consequences What is Kerning, what does my product do"
    # Response-path gaps carry boilerplate in `missing`; the demand is the question.
    assert gg.question_of("the bank may lack coverage for this query", "What is the VAT rule?",
                          "uncited_answer") == "What is the VAT rule?"
    # A run's leftover question is `missing`; its triggering_query is the run's goal.
    assert gg.question_of("License of LTX Desktop", "Video toolchain report", "run_residual") == "License of LTX Desktop"
    assert gg.question_of("Operator research topic: SQLite WAL mode. Gather ...", None, "topic_spec",
                          "SQLite WAL mode") == "SQLite WAL mode"


@pytest.mark.asyncio
async def test_a_question_answered_or_dismissed_before_is_not_asked_again(store, customer):
    cid = customer.id
    first = await gg.open_gap(store, cid, missing="What is the new color format adopted this year?",
                              source="gap_discovery")
    assert first.opened
    await store.set_knowledge_gap_status(first.id, cid, "closed")  # dismissed
    again = await gg.open_gap(store, cid, missing="What is the new color format adopted this year?",
                              source="gap_discovery")
    assert again.verdict == "duplicate" and again.existing_gap_id == first.id and again.detail["status"] == "closed"
    # Two uncited answers with the same boilerplate but different demands are two gaps.
    boiler = "An answer was produced ... the bank may lack coverage for this query."
    a = await gg.open_gap(store, cid, missing=boiler, triggering_query="Who reviews pull requests?",
                          source="uncited_answer", check_answer=False)
    b = await gg.open_gap(store, cid, missing=boiler, triggering_query="What does the Pro plan cost?",
                          source="uncited_answer", check_answer=False)
    assert a.opened and b.opened
    assert len(await store.list_knowledge_gaps(cid, status="open")) == 2


@pytest.mark.asyncio
async def test_a_question_close_in_meaning_is_the_same_gap(store, customer):
    cid = customer.id
    enc = _Enc({"What is the new color format adopted this year?": _vec(3),
                "Which color notation does the 2025 token spec require?": _vec(3) * 0.98 + _vec(4) * 0.05})
    idx = _Index([])
    first = await gg.open_gap(store, cid, missing="What is the new color format adopted this year?",
                              encoder=enc, vector_index=idx)
    assert first.opened and first.gap.id
    rows = await store.list_gate_gaps(cid)
    assert rows[0]["vector"] is not None  # stored for the next comparison
    dup = await gg.open_gap(store, cid, missing="Which color notation does the 2025 token spec require?",
                            encoder=enc, vector_index=idx)
    assert dup.verdict == "duplicate"


@pytest.mark.asyncio
async def test_an_answered_question_is_born_filled_with_its_answer(store, customer):
    cid = customer.id
    K, FN, FL = await _seed(store, cid)
    # The question points nowhere; its key ("Kerning ...") finds the fact.
    enc = _Enc({"What is the name of the product?": _vec(6), "name product": _vec(6),
                "Consequences What is Kerning, what does my product do": _vec(1)})
    idx = _Index([(FN, K, _vec(1)), (FL, K, _vec(2))])
    judge = _Judge(pick=0)
    res = await gg.open_gap(store, cid, missing="What is the name of the product?", source="gap_discovery",
                            full_key="Consequences|What is Kerning, what does my product do",
                            encoder=enc, vector_index=idx, llm=judge)
    assert res.verdict == "answered" and not res.opened and res.fact_id == FN and judge.calls == 1
    assert "Consequences What is Kerning, what does my product do" in enc.seen
    assert await store.list_knowledge_gaps(cid, status="open") == []
    filled = await store.list_knowledge_gaps(cid, status="filled")
    assert filled[0].answered_by_fact_id == FN and filled[0].filled_by_crystal_id == K
    assert filled[0].gate["verdict"] == "answered"
    # Asking again finds the settled question, not a new gap.
    again = await gg.open_gap(store, cid, missing="What is the name of the product?", encoder=enc,
                              vector_index=idx, llm=judge)
    assert again.verdict == "duplicate" and judge.calls == 1


@pytest.mark.asyncio
async def test_nothing_near_means_no_model_call(store, customer):
    cid = customer.id
    K, FN, FL = await _seed(store, cid)
    enc = _Enc({})  # everything encodes to _vec(7): nowhere near any fact
    judge = _Judge(pick=0)
    res = await gg.open_gap(store, cid, missing="What is the EU VAT rule for agencies?", encoder=enc,
                            vector_index=_Index([(FN, K, _vec(1))]), llm=judge)
    assert res.opened and judge.calls == 0 and res.gap.gate["verdict"] == "open"


@pytest.mark.asyncio
async def test_the_sweep_fills_a_gap_when_a_fact_answers_it_and_pays_only_for_new_facts(store, customer):
    cid = customer.id
    K, FN, FL = await _seed(store, cid)
    enc = _Enc({"When does Kerning launch?": _vec(2)})
    # A response-path gap: checked later, not inline.
    res = await gg.open_gap(store, cid, missing="the bank may lack coverage for this query",
                            triggering_query="When does Kerning launch?", source="uncited_answer",
                            check_answer=False, encoder=enc)
    assert res.opened and res.gap.gate["reason"] == "deferred"
    no = _Judge(pick=-1)
    out = await gg.recheck_open_gaps(store, cid, encoder=enc, vector_index=_Index([(FN, K, _vec(2) + _vec(1))]),
                                     llm=no)
    assert out["checked"] == 1 and out["filled"] == 0 and no.calls == 1
    # Same candidates next pass: no model call.
    out = await gg.recheck_open_gaps(store, cid, encoder=enc, vector_index=_Index([(FN, K, _vec(2) + _vec(1))]),
                                     llm=no)
    assert no.calls == 1 and out["calls"] == 0
    # A new fact arrives that answers it.
    yes = _Judge(pick=0)
    idx = _Index([(FL, K, _vec(2)), (FN, K, _vec(2) + _vec(1))])
    out = await gg.recheck_open_gaps(store, cid, encoder=enc, vector_index=idx, llm=yes)
    assert out["filled"] == 1 and yes.calls == 1
    gap = (await store.list_knowledge_gaps(cid, status="filled"))[0]
    assert gap.answered_by_fact_id == FL


@pytest.mark.asyncio
async def test_the_record_gap_tool_says_where_the_answer_is(store, customer, tool_state, monkeypatch):
    from crystal_cache.agent.tools.curation import record_gap
    from crystal_cache.agent.tools.retrievers import set_tool_state
    set_tool_state(tool_state)
    cid = customer.id
    K, FN, FL = await _seed(store, cid)
    enc = _Enc({"When does Kerning launch?": _vec(2)})
    monkeypatch.setattr(gg, "_RUNTIME", {"encoder": enc, "vector_index": _Index([(FL, K, _vec(2))])})
    from crystal_cache.llm.client import reset_llm_client, set_llm_client
    set_llm_client(_Judge(pick=0))
    try:
        out = await record_gap(cid, question="When does Kerning launch?", disposition="researchable")
    finally:
        reset_llm_client()
    assert out["recorded"] is False and out["code"] == "already_answered" and out["fact_id"] == FL
    assert out["answer"] == "Kerning launches on November 12, 2026."
    monkeypatch.setattr(gg, "_RUNTIME", {"encoder": None, "vector_index": None})
    first = await record_gap(cid, question="Who signs off on refunds?", disposition="researchable")
    dup = await record_gap(cid, question="Who signs off on refunds?", disposition="researchable")
    assert first["recorded"] is True and dup["code"] == "gap_exists" and dup["gap_id"] == first["gap_id"]
