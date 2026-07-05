"""P3 — agent citations (Growth G1/G4 on the agent surface, CC-D11 = B).

Grounding-based implicit credit: the agent surfaces crystals via its retrieval
tools (no [[cc:N]] markers), so we ground each surfaced crystal against the
final answer and record/credit the grounded ones. Covers the new answer-level
grounding primitive (ground_sources_against_answer) and the endpoint helper's
orchestration (ground_agent_citations): recording, the G4 credit path, the G1c
uncited-answer gap dual, and the no-op / fail-safe gates.

R14 note: verified by pytest; describes expected behavior, not yet run at
authoring time.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from crystal_cache import config
from crystal_cache.endpoints.agent import ground_agent_citations
from crystal_cache.retrieval.citations import CitationSource
from crystal_cache.retrieval.citation_grounding import (
    ground_sources_against_answer,
)


# A controlled encoder: encode_native(text) -> a unit vector on axis 0 if the
# text contains "MATCH", else axis 1 (orthogonal). Records calls so a test can
# assert the answer is encoded once. Deterministic, no model.
class _AxisEncoder:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.dim = 8

    def encode_native(self, text: str) -> np.ndarray:
        self.calls.append(text)
        v = np.zeros(self.dim, dtype=np.float32)
        v[0 if "MATCH" in text else 1] = 1.0
        return v


def _src(handle: str, crystal_id: str) -> CitationSource:
    return CitationSource(handle=handle, crystal_id=crystal_id)


# ---------------------------------------------------------------------------
# 1. ground_sources_against_answer (the Option-B primitive)
# ---------------------------------------------------------------------------

async def test_parallel_source_grounds():
    enc = _AxisEncoder()
    out = await ground_sources_against_answer(
        enc, "MATCH the answer", [(_src("1", "cry_A"), "MATCH the source")],
    )
    assert len(out) == 1
    assert out[0]["grounded"] is True
    assert out[0]["grounding_score"] > 0.99   # parallel -> cosine ~1
    assert out[0]["claim_span"] == ""         # no markers => no span


async def test_orthogonal_source_does_not_ground():
    enc = _AxisEncoder()
    out = await ground_sources_against_answer(
        enc, "MATCH the answer", [(_src("1", "cry_A"), "unrelated source")],
    )
    assert out[0]["grounded"] is False
    assert out[0]["grounding_score"] < 0.25


async def test_empty_answer_grounds_nothing():
    enc = _AxisEncoder()
    out = await ground_sources_against_answer(
        enc, "", [(_src("1", "cry_A"), "MATCH source")],
    )
    assert out[0]["grounded"] is False
    assert enc.calls == []   # short-circuits before encoding anything


async def test_empty_source_text_not_grounded():
    enc = _AxisEncoder()
    out = await ground_sources_against_answer(
        enc, "MATCH the answer", [(_src("1", "cry_A"), "")],
    )
    assert out[0]["grounded"] is False


async def test_threshold_is_respected():
    enc = _AxisEncoder()
    pair = [(_src("1", "cry_A"), "MATCH source")]   # parallel -> cosine ~1
    high = await ground_sources_against_answer(enc, "MATCH ans", pair, threshold=0.9)
    impossible = await ground_sources_against_answer(enc, "MATCH ans", pair, threshold=1.01)
    assert high[0]["grounded"] is True
    assert impossible[0]["grounded"] is False


async def test_answer_encoded_once_across_sources():
    enc = _AxisEncoder()
    await ground_sources_against_answer(
        enc, "MATCH the answer",
        [(_src("1", "cry_A"), "MATCH a"), (_src("2", "cry_B"), "MATCH b")],
    )
    # The answer is encoded ONCE and reused; each source is encoded too.
    assert enc.calls.count("MATCH the answer") == 1
    assert "MATCH a" in enc.calls and "MATCH b" in enc.calls


# ---------------------------------------------------------------------------
# 2. ground_agent_citations (the endpoint orchestration)
# ---------------------------------------------------------------------------

_GROUNDING_PATH = (
    "crystal_cache.retrieval.citation_grounding.ground_sources_against_answer"
)

_LONG_ANSWER = (
    "This is a substantive answer that comfortably exceeds the uncited-gap "
    "minimum character threshold so the G1c dual can fire when nothing grounds."
)


def _result_surfacing(
    crystal_id: str, *, final_text: str = _LONG_ANSWER, run_id: str = "run_1"
) -> dict:
    return {
        "id": run_id,
        "final_text": final_text,
        "tool_calls": [
            {"name": "knowledge_search", "output": {
                "matched_crystal_ids": [crystal_id],
                "matched_fact_ids": ["f1"],
            }},
        ],
    }


def _patch_grounding(monkeypatch, results):
    async def _fake(encoder, answer_text, sources, *, threshold=0.25):
        return results
    monkeypatch.setattr(_GROUNDING_PATH, _fake)


async def test_grounded_citation_is_recorded(store, customer, monkeypatch):
    monkeypatch.setattr(config.settings, "enable_citations", True)
    monkeypatch.setattr(config.settings, "enable_marketplace_metering", False)
    _patch_grounding(monkeypatch, [
        {"source": _src("1", "cry_A"), "claim_span": "",
         "grounding_score": 0.9, "grounded": True},
    ])
    await ground_agent_citations(
        store=store, encoder=object(), customer=customer,
        result=_result_surfacing("cry_A"), user_query="q", sequence_id="s1",
    )
    rows = await store.list_citations_for_crystal(
        customer.id, "cry_A", grounded_only=False,
    )
    assert len(rows) == 1
    assert rows[0]["grounded"] is True
    assert rows[0]["crystal_id"] == "cry_A"


async def test_ungrounded_substantive_answer_fires_gap(store, customer, monkeypatch):
    monkeypatch.setattr(config.settings, "enable_citations", True)
    monkeypatch.setattr(config.settings, "enable_marketplace_metering", False)
    _patch_grounding(monkeypatch, [
        {"source": _src("1", "cry_A"), "claim_span": "",
         "grounding_score": 0.05, "grounded": False},
    ])
    gap_calls = []

    async def _gap_spy(customer_id, **kwargs):
        gap_calls.append((customer_id, kwargs))
    monkeypatch.setattr(store, "create_knowledge_gap", _gap_spy)

    await ground_agent_citations(
        store=store, encoder=object(), customer=customer,
        result=_result_surfacing("cry_A"), user_query="how do I X?",
        sequence_id="s1",
    )
    # The ungrounded citation is still recorded (telemetry)...
    rows = await store.list_citations_for_crystal(
        customer.id, "cry_A", grounded_only=False,
    )
    assert len(rows) == 1 and rows[0]["grounded"] is False
    # ...and the coverage-gap dual fired.
    assert len(gap_calls) == 1
    assert gap_calls[0][1].get("source") == "uncited_answer"


async def test_grounded_marketplace_crystal_is_credited(store, customer, monkeypatch):
    monkeypatch.setattr(config.settings, "enable_citations", True)
    monkeypatch.setattr(config.settings, "enable_marketplace_metering", True)
    _patch_grounding(monkeypatch, [
        {"source": _src("1", "cry_A"), "claim_span": "",
         "grounding_score": 0.9, "grounded": True},
    ])

    async def _fake_get_crystal(cid):
        return SimpleNamespace(
            owner_operator_id="op1", group_team_id="team1",
            crystal_type="general", customer_id="cust_owner",
        )
    monkeypatch.setattr(store, "get_crystal", _fake_get_crystal)

    credit_calls = []

    async def _credit_spy(**kwargs):
        credit_calls.append(kwargs)
    monkeypatch.setattr(store, "record_citation_credit", _credit_spy)

    await ground_agent_citations(
        store=store, encoder=object(), customer=customer,
        result=_result_surfacing("cry_A", run_id="run_42"),
        user_query="q", sequence_id="s1",
    )
    assert len(credit_calls) == 1
    c = credit_calls[0]
    assert c["crystal_id"] == "cry_A"
    assert c["consuming_team_id"] == customer.id
    assert c["interaction_id"] == "run_42"   # the agent run id
    assert c["raw_weight"] == 1.0


async def test_no_surfaced_crystals_is_noop(store, customer, monkeypatch):
    monkeypatch.setattr(config.settings, "enable_citations", True)
    called = {"n": 0}

    async def _fake(*a, **k):
        called["n"] += 1
        return []
    monkeypatch.setattr(_GROUNDING_PATH, _fake)

    # A non-retrieval tool output carries no matched_crystal_ids.
    result = {"id": "r", "final_text": _LONG_ANSWER, "tool_calls": [
        {"name": "llm_invoke", "output": {"answer": "no matched ids here"}},
    ]}
    await ground_agent_citations(
        store=store, encoder=object(), customer=customer,
        result=result, user_query="q", sequence_id="s1",
    )
    assert called["n"] == 0   # grounding never runs when nothing was surfaced
    assert await store.list_citations_for_crystal(
        customer.id, "cry_A", grounded_only=False,
    ) == []


async def test_disabled_flag_is_noop(store, customer, monkeypatch):
    monkeypatch.setattr(config.settings, "enable_citations", False)
    called = {"n": 0}

    async def _fake(*a, **k):
        called["n"] += 1
        return []
    monkeypatch.setattr(_GROUNDING_PATH, _fake)

    await ground_agent_citations(
        store=store, encoder=object(), customer=customer,
        result=_result_surfacing("cry_A"), user_query="q", sequence_id="s1",
    )
    assert called["n"] == 0


async def test_failsafe_when_record_raises(store, customer, monkeypatch):
    monkeypatch.setattr(config.settings, "enable_citations", True)
    _patch_grounding(monkeypatch, [
        {"source": _src("1", "cry_A"), "claim_span": "",
         "grounding_score": 0.9, "grounded": True},
    ])

    async def _boom(*a, **k):
        raise RuntimeError("db down")
    monkeypatch.setattr(store, "record_citations", _boom)

    # Must not raise — citation processing never breaks the response.
    await ground_agent_citations(
        store=store, encoder=object(), customer=customer,
        result=_result_surfacing("cry_A"), user_query="q", sequence_id="s1",
    )


async def test_configured_threshold_is_passed_to_grounding(store, customer, monkeypatch):
    monkeypatch.setattr(config.settings, "enable_citations", True)
    monkeypatch.setattr(config.settings, "enable_marketplace_metering", False)
    monkeypatch.setattr(config.settings, "agent_citation_grounding_threshold", 0.42)
    seen = {}

    async def _fake(encoder, answer_text, sources, *, threshold=0.25):
        seen["threshold"] = threshold
        return [
            {"source": s, "claim_span": "", "grounding_score": 0.9, "grounded": True}
            for s, _ in sources
        ]
    monkeypatch.setattr(_GROUNDING_PATH, _fake)

    await ground_agent_citations(
        store=store, encoder=object(), customer=customer,
        result=_result_surfacing("cry_A"), user_query="q", sequence_id="s1",
    )
    # CC-D13: the helper reads the agent-level threshold from settings and
    # passes it through (not the proxy's claim-span 0.25 default).
    assert seen["threshold"] == 0.42
