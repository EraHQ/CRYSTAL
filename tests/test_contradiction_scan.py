"""Never-Idle Convergence — contradiction-scan generator
(crystal_cache.scan.contradiction).

Two layers:
  * pure helpers (no I/O) — sparse-key Subject parsing, idempotence pair_key,
    bounded candidate enumeration (within-crystal ∪ same-Subject-cross-crystal);
  * scan_for_contradictions against the in-memory store with a bespoke
    discriminator fake — CONTRADICTS→row / else→no-row, idempotent re-run,
    the budget cap, the candidate cap, no-provider no-op, and the fail-safe
    ERROR path.

The fake mirrors the seam surface the generator actually calls
(`client.complete(...) -> str` verdict) and is verdict-by-
claims so tests don't depend on enumeration order. Budget/eval counts are
asserted on the returned ScanResult (deterministic), not the fake's call
counter (which the semaphore touches across threads).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from crystal_cache.infrastructure.schema import CrystalRow, FactRow
from crystal_cache.scan import ScanResult, scan_for_contradictions
from crystal_cache.scan.contradiction import (
    _enumerate_candidate_pairs,
    _pair_key,
)
from crystal_cache.llm import reset_llm_client, set_llm_client

from fakes import NotReadyLLM, SeamFake

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Bespoke discriminator fake
# ---------------------------------------------------------------------------

class FakeDiscriminator(SeamFake):
    """Seam-shaped client (complete(...) -> str) returning a one-word verdict.

    rules: list of (needle_a, needle_b, verdict). On each call the user
    message is scanned; the first rule whose BOTH needles appear (case-
    insensitive) wins. Default verdict otherwise. `raise_on_call=True`
    exercises the generator's fail-safe ERROR path.
    """

    def __init__(self, rules=None, default="UNRELATED", *, raise_on_call=False):
        self.rules = rules or []
        self.default = default
        self.raise_on_call = raise_on_call
        self.calls = 0

    def complete(self, **kwargs):
        self.calls += 1
        if self.raise_on_call:
            raise RuntimeError("simulated upstream failure")
        content = kwargs["messages"][0]["content"].upper()
        for needle_a, needle_b, verdict in self.rules:
            if needle_a.upper() in content and needle_b.upper() in content:
                return verdict
        return self.default


def _fact(fid, crystal_id, claim, key=""):
    """A Fact-shaped namespace for the pure-helper tests."""
    return SimpleNamespace(
        id=fid,
        crystal_id=crystal_id,
        claim_text=claim,
        prompt_text=key,
        source_kind="model_reasoning",
        source_doc_id=None,
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_pair_key_is_order_independent_and_claim_sensitive():
    a = _fact("fa", "c1", "claim one")
    b = _fact("fb", "c1", "claim two")
    assert _pair_key(a, b) == _pair_key(b, a)  # symmetric
    # A changed claim → a different key (so the pair is re-evaluated).
    a2 = _fact("fa", "c1", "claim one CHANGED")
    assert _pair_key(a2, b) != _pair_key(a, b)


def test_enumerate_within_crystal_pairs():
    facts = [
        _fact("f1", "cA", "one"),
        _fact("f2", "cA", "two"),
        _fact("f3", "cA", "three"),
    ]
    pairs = _enumerate_candidate_pairs(facts, max_pairs=100)
    # C(3,2) = 3 within-crystal pairs.
    assert len(pairs) == 3


def test_enumerate_shared_segment_across_crystals_rarity_excluded():
    """C3 Q1=A: cross-crystal candidates come from shared RARE segments
    at any position; namespace-scale segments (here 'Contract'/'Legal',
    carried by the whole population) never form groups alone. The old
    positional Subject-slot test this replaces predates the segment
    model — helper-level coverage lives in test_segment_scan.py."""
    facts = [
        _fact("f1", "cA", "rate is 120", "Contract|x|Rate|Legal"),
        _fact("f2", "cB", "rate is 95", "Contract|y|Rate|Legal"),
        _fact("f3", "cC", "unrelated", "Contract|z|Parties|Legal"),
    ] + [
        # Filler population pushes 'Contract'/'Legal' past the
        # exclusion threshold; each filler key's middle segment is
        # unique so filler forms no groups of its own.
        _fact(f"ff{i}", f"cF{i}", f"filler {i}", f"Contract|filler {i}|Legal")
        for i in range(17)
    ]
    pairs = _enumerate_candidate_pairs(facts, max_pairs=100)
    ids = {frozenset((a.id, b.id)) for a, b in pairs}
    assert ids == {frozenset(("f1", "f2"))}  # 'Rate' — the rare meeting point


def test_enumerate_respects_cap():
    facts = [_fact(f"f{i}", "cA", f"claim {i}") for i in range(6)]  # C(6,2)=15
    pairs = _enumerate_candidate_pairs(facts, max_pairs=4)
    assert len(pairs) == 4


def test_enumerate_skips_empty_claims():
    facts = [
        _fact("f1", "cA", "real"),
        _fact("f2", "cA", "   "),   # blank → excluded
        _fact("f3", "cA", ""),       # empty → excluded
    ]
    assert _enumerate_candidate_pairs(facts, max_pairs=100) == []


# ---------------------------------------------------------------------------
# Integration — scan_for_contradictions against the store
# ---------------------------------------------------------------------------

async def _seed_crystal(store, crystal_id, customer_id):
    async with store.session() as s:
        s.add(CrystalRow(
            id=crystal_id, customer_id=customer_id,
            crystal_type="customer:legacy", summary_vector=[],
        ))


async def _seed_fact(store, *, fid, crystal_id, claim, key="", offset_min=0,
                     source_doc_id=None):
    async with store.session() as s:
        s.add(FactRow(
            id=fid, crystal_id=crystal_id, pair_type="question_answer",
            prompt_text=key, claim_text=claim, source_kind="model_reasoning",
            source_doc_id=source_doc_id,
            vector=[], created_at=_T0 + timedelta(minutes=offset_min),
        ))


async def test_same_source_document_pairs_are_excluded(store, customer):
    """CF-Q1=A (2026-07-23): a document cannot contradict itself via its
    own extraction — same-source pairs never reach the discriminator.
    The same claims from DIFFERENT documents remain fully eligible."""
    await _seed_crystal(store, "cA", customer.id)
    await _seed_fact(store, fid="f1", crystal_id="cA",
                     claim="The launch is September 15",
                     key="Brief|date|Launch|Ops", source_doc_id="doc_1")
    await _seed_fact(store, fid="f2", crystal_id="cA",
                     claim="The launch is September 22",
                     key="Brief|date2|Launch|Ops", offset_min=1,
                     source_doc_id="doc_1")

    fake = FakeDiscriminator(default="CONTRADICTS")  # trips if consulted
    result = await scan_for_contradictions(
        store=store, slm_client=fake, customer_id=customer.id,
    )
    assert result.conflicts_found == 0
    assert result.pairs_evaluated == 0

    # Move one side to another document: the pair is live again.
    await _seed_fact(store, fid="f3", crystal_id="cA",
                     claim="The launch is September 22nd per the meeting",
                     key="Meeting|date|Launch|Ops", offset_min=2,
                     source_doc_id="doc_2")
    fake2 = FakeDiscriminator(
        rules=[("SEPTEMBER 15", "SEPTEMBER 22ND", "CONTRADICTS")],
    )
    result2 = await scan_for_contradictions(
        store=store, slm_client=fake2, customer_id=customer.id,
    )
    assert result2.conflicts_found == 1


async def test_contradicts_creates_open_conflict(store, customer):
    await _seed_crystal(store, "cA", customer.id)
    await _seed_fact(store, fid="f1", crystal_id="cA",
                     claim="The contract rate is $120/hr", key="Legal|Contract|Rate|Section x")
    await _seed_fact(store, fid="f2", crystal_id="cA",
                     claim="The contract rate is $95/hr", key="Legal|Contract|Rate|Section y",
                     offset_min=1)

    fake = FakeDiscriminator(rules=[("$120/hr", "$95/hr", "CONTRADICTS")])
    result = await scan_for_contradictions(
        store=store, slm_client=fake, customer_id=customer.id,
    )

    assert result.conflicts_found == 1
    assert result.pairs_evaluated == 1
    rows = await store.list_knowledge_conflicts(customer.id, status="open")
    assert len(rows) == 1
    assert rows[0].subject == "Rate"
    assert {rows[0].fact_a_id, rows[0].fact_b_id} == {"f1", "f2"}
    assert rows[0].provenance_a == "model_reasoning"


async def test_consistent_creates_no_conflict(store, customer):
    await _seed_crystal(store, "cA", customer.id)
    await _seed_fact(store, fid="f1", crystal_id="cA", claim="Opens at 9am")
    await _seed_fact(store, fid="f2", crystal_id="cA", claim="Staff arrive by nine", offset_min=1)

    fake = FakeDiscriminator(default="CONSISTENT")
    result = await scan_for_contradictions(
        store=store, slm_client=fake, customer_id=customer.id,
    )
    assert result.conflicts_found == 0
    assert await store.list_knowledge_conflicts(customer.id) == []


async def test_rescan_is_idempotent(store, customer):
    await _seed_crystal(store, "cA", customer.id)
    await _seed_fact(store, fid="f1", crystal_id="cA", claim="rate 120")
    await _seed_fact(store, fid="f2", crystal_id="cA", claim="rate 95", offset_min=1)
    fake = FakeDiscriminator(rules=[("RATE 120", "RATE 95", "CONTRADICTS")])

    first = await scan_for_contradictions(
        store=store, slm_client=fake, customer_id=customer.id,
    )
    assert first.conflicts_found == 1

    # Re-run over the unchanged bank: the pair_key already exists → skipped,
    # no new discriminator call, no duplicate row.
    second = await scan_for_contradictions(
        store=store, slm_client=fake, customer_id=customer.id,
    )
    assert second.pairs_evaluated == 0
    assert second.skipped_existing == 1
    assert second.conflicts_found == 0
    assert len(await store.list_knowledge_conflicts(customer.id)) == 1


async def test_budget_caps_discriminator_calls(store, customer):
    await _seed_crystal(store, "cA", customer.id)
    for i in range(3):  # C(3,2) = 3 candidate pairs
        await _seed_fact(store, fid=f"f{i}", crystal_id="cA",
                         claim=f"claim number {i}", offset_min=i)

    fake = FakeDiscriminator(default="CONTRADICTS")
    result = await scan_for_contradictions(
        store=store, slm_client=fake, customer_id=customer.id,
        max_discriminator_calls=1,
    )
    assert result.candidate_pairs == 3
    assert result.pairs_evaluated == 1
    assert result.budget_exhausted is True
    assert result.conflicts_found == 1


async def test_candidate_cap_bounds_enumeration(store, customer):
    await _seed_crystal(store, "cA", customer.id)
    for i in range(5):  # C(5,2) = 10 possible
        await _seed_fact(store, fid=f"f{i}", crystal_id="cA",
                         claim=f"claim {i}", offset_min=i)

    fake = FakeDiscriminator(default="UNRELATED")
    result = await scan_for_contradictions(
        store=store, slm_client=fake, customer_id=customer.id,
        max_candidate_pairs=2,
    )
    assert result.candidate_pairs == 2


async def test_cross_crystal_same_subject_contradiction(store, customer):
    await _seed_crystal(store, "cA", customer.id)
    await _seed_crystal(store, "cB", customer.id)
    await _seed_fact(store, fid="f1", crystal_id="cA",
                     claim="Deductible is $500", key="Health|Policy|Deductible|Section A")
    await _seed_fact(store, fid="f2", crystal_id="cB",
                     claim="Deductible is $1500", key="Health|Policy|Deductible|Section B",
                     offset_min=1)

    fake = FakeDiscriminator(rules=[("$500", "$1500", "CONTRADICTS")])
    result = await scan_for_contradictions(
        store=store, slm_client=fake, customer_id=customer.id,
    )
    assert result.conflicts_found == 1
    rows = await store.list_knowledge_conflicts(customer.id)
    assert {rows[0].crystal_a_id, rows[0].crystal_b_id} == {"cA", "cB"}


async def test_none_client_is_noop(store, customer):
    """A None override with a not-ready seam is a no-op (NotReadyLLM is
    injected so the test never depends on real environment keys)."""
    await _seed_crystal(store, "cA", customer.id)
    await _seed_fact(store, fid="f1", crystal_id="cA", claim="a")
    await _seed_fact(store, fid="f2", crystal_id="cA", claim="b", offset_min=1)

    set_llm_client(NotReadyLLM())
    try:
        result = await scan_for_contradictions(
            store=store, slm_client=None, customer_id=customer.id,
        )
    finally:
        reset_llm_client()
    assert isinstance(result, ScanResult)
    assert result.conflicts_found == 0
    assert result.pairs_evaluated == 0
    assert await store.list_knowledge_conflicts(customer.id) == []


async def test_discriminator_error_writes_nothing(store, customer):
    await _seed_crystal(store, "cA", customer.id)
    await _seed_fact(store, fid="f1", crystal_id="cA", claim="a")
    await _seed_fact(store, fid="f2", crystal_id="cA", claim="b", offset_min=1)

    fake = FakeDiscriminator(raise_on_call=True)
    result = await scan_for_contradictions(
        store=store, slm_client=fake, customer_id=customer.id,
    )
    # The pair was evaluated (a call was attempted) but the error verdict
    # never writes a conflict.
    assert result.pairs_evaluated == 1
    assert result.conflicts_found == 0
    assert await store.list_knowledge_conflicts(customer.id) == []


async def test_scan_is_own_facts_only(store, customer):
    """D8: another customer's contradicting facts are not scanned."""
    other = await store.create_customer(
        provider="anthropic", model_id="claude-sonnet-4-5-20250929",
        api_key_ref="sk-test-other",
    )
    await _seed_crystal(store, "cOther", other.id)
    await _seed_fact(store, fid="o1", crystal_id="cOther", claim="rate 120")
    await _seed_fact(store, fid="o2", crystal_id="cOther", claim="rate 95", offset_min=1)

    fake = FakeDiscriminator(default="CONTRADICTS")
    result = await scan_for_contradictions(
        store=store, slm_client=fake, customer_id=customer.id,
    )
    # `customer` has no facts of its own → nothing scanned, nothing found.
    assert result.facts_scanned == 0
    assert result.conflicts_found == 0


# ---------------------------------------------------------------------------
# 2026-10-21: neighbor candidates, shared cap, chunk exclusion, verdict memo,
# reasoning judge
# ---------------------------------------------------------------------------

from crystal_cache.scan.contradiction import (  # noqa: E402
    VERDICT_SCHEMA, _neighbor_pairs, parse_verdict,
)


def _vfact(fid, crystal_id, claim, vector, key="", doc=None):
    return SimpleNamespace(
        id=fid, crystal_id=crystal_id, claim_text=claim, prompt_text=key,
        source_kind="model_reasoning", source_doc_id=doc, vector=vector,
        pair_type="question_answer",
    )


def test_parse_verdict_accepts_the_object_and_the_word():
    assert parse_verdict('{"reason": "two prices", "verdict": "CONTRADICTS"}') == "CONTRADICTS"
    assert parse_verdict('{"reason": "same thing", "verdict": "consistent"}') == "CONSISTENT"
    assert parse_verdict("UNRELATED") == "UNRELATED"
    assert parse_verdict("CONTRADICTS - the rates differ") == "CONTRADICTS"
    assert parse_verdict('{"reason": "x", "verdict": "MAYBE"}') == "UNKNOWN"
    assert parse_verdict("") == "UNKNOWN"
    assert parse_verdict(None) == "UNKNOWN"


def test_neighbor_pairs_cross_crystals_by_vector_above_the_floor():
    """The source that pairs 'Product launch | November 5' with 'Launch on
    Product Hunt | November 12': no shared key segment, near vectors."""
    a = _vfact("f1", "cA", "Launch is November 5", [1.0, 0.0, 0.0],
               key="Founder journal|September|Product launch")
    b = _vfact("f2", "cB", "Launch is November 12", [0.95, 0.3, 0.0],
               key="ADR 0004|Launch on Product Hunt|Decision")
    c = _vfact("f3", "cC", "The database costs $25", [0.0, 0.0, 1.0],
               key="Stack|Database|Cost")
    d = _vfact("f4", "cA", "Launch slipped", [0.9, 0.4, 0.0])  # same crystal as a
    pairs = _neighbor_pairs([a, b, c, d], k=5, min_score=0.70)
    ids = [frozenset((x.id, y.id)) for x, y in pairs]
    assert frozenset(("f1", "f2")) in ids
    assert frozenset(("f1", "f3")) not in ids          # below the floor
    assert frozenset(("f1", "f4")) not in ids          # same crystal: not this source
    assert frozenset(("f2", "f4")) in ids
    # Strongest first.
    assert ids[0] == frozenset(("f1", "f2")) or ids[0] == frozenset(("f2", "f4"))
    # No vectors, or k=0: no neighbor pairs, no error.
    assert _neighbor_pairs([_fact("x", "c1", "one"), _fact("y", "c2", "two")], k=5, min_score=0.7) == []
    assert _neighbor_pairs([a, b], k=0, min_score=0.7) == []


def test_enumerate_takes_the_sources_in_turns_so_the_cap_is_shared():
    """One document-heavy crystal (15 within pairs) must not fill a cap of
    two on its own: the cross-crystal segment pair gets its turn."""
    big = [_fact(f"f{i}", "cBig", f"claim {i}") for i in range(6)]
    seg = [
        _fact("s1", "cS1", "deductible 500", "Plan alpha|Deductible"),
        _fact("s2", "cS2", "deductible 1500", "Plan beta|Deductible"),
    ]
    pairs = _enumerate_candidate_pairs(big + seg, max_pairs=2, neighbor_k=0)
    ids = {frozenset((a.id, b.id)) for a, b in pairs}
    assert len(pairs) == 2
    assert frozenset(("s1", "s2")) in ids


def test_enumerate_refuses_same_source_pairs_before_the_cap():
    """Same-document pairs used to be filtered AFTER the cap and could eat
    all of it; now they never count."""
    same = [_vfact(f"d{i}", "cA", f"claim {i}", [], doc="doc_1") for i in range(5)]
    other = _vfact("o1", "cA", "claim from elsewhere", [], doc="doc_2")
    pairs = _enumerate_candidate_pairs(
        same + [other], max_pairs=3, skip_same_source=True, neighbor_k=0,
    )
    assert len(pairs) == 3
    assert all("o1" in (a.id, b.id) for a, b in pairs)
    # Without the switch (the dedup scan's call) same-document pairs stay.
    assert len(_enumerate_candidate_pairs(same, max_pairs=100, neighbor_k=0)) == 10


async def test_content_chunks_never_enter_the_contradiction_scan(store, customer):
    await _seed_crystal(store, "cA", customer.id)
    await _seed_crystal(store, "cB", customer.id)
    async with store.session() as s:
        s.add(FactRow(
            id="chunk", crystal_id="cA", pair_type="content_chunk",
            prompt_text="Document|Passage 1", source_kind="document_chunk",
            claim_text="Pro is $12 a month. The launch is November 5.",
            source_doc_id="doc_1", vector=[], created_at=_T0,
        ))
    await _seed_fact(store, fid="f1", crystal_id="cB", claim="Pro is $15 a month",
                     key="Pricing|Pro", offset_min=1, source_doc_id="doc_2")

    fake = FakeDiscriminator(default="CONTRADICTS")  # trips if consulted
    result = await scan_for_contradictions(
        store=store, slm_client=fake, customer_id=customer.id,
    )
    assert result.facts_scanned == 1
    assert result.pairs_evaluated == 0
    assert result.conflicts_found == 0


async def test_consistent_verdict_is_remembered_and_never_rejudged(store, customer):
    await _seed_crystal(store, "cA", customer.id)
    await _seed_fact(store, fid="f1", crystal_id="cA", claim="Opens at 9am")
    await _seed_fact(store, fid="f2", crystal_id="cA", claim="Staff arrive by nine", offset_min=1)

    fake = FakeDiscriminator(default="CONSISTENT")
    first = await scan_for_contradictions(
        store=store, slm_client=fake, customer_id=customer.id,
    )
    assert first.pairs_evaluated == 1
    assert first.memoized == 1
    assert fake.calls == 1

    second = await scan_for_contradictions(
        store=store, slm_client=fake, customer_id=customer.id,
    )
    assert second.pairs_evaluated == 0
    assert second.skipped_existing == 1
    assert fake.calls == 1  # no new model call
    assert await store.list_knowledge_conflicts(customer.id) == []

    # UNKNOWN is not remembered: the pair is judged again next cycle.
    await _seed_fact(store, fid="f3", crystal_id="cA", claim="Closes at 5pm", offset_min=2)
    odd = FakeDiscriminator(default="I cannot tell")
    third = await scan_for_contradictions(
        store=store, slm_client=odd, customer_id=customer.id,
    )
    assert third.pairs_evaluated == 2 and third.memoized == 0
    fourth = await scan_for_contradictions(
        store=store, slm_client=odd, customer_id=customer.id,
    )
    assert fourth.pairs_evaluated == 2


async def test_judge_is_asked_for_reasoning_then_verdict_by_schema(store, customer):
    await _seed_crystal(store, "cA", customer.id)
    await _seed_fact(store, fid="f1", crystal_id="cA", claim="The rate is $120/hr")
    await _seed_fact(store, fid="f2", crystal_id="cA", claim="The rate is $95/hr", offset_min=1)

    class Recording(SeamFake):
        def __init__(self):
            self.sent = []

        def complete(self, **kwargs):
            self.sent.append(kwargs)
            if kwargs.get("json_schema") is VERDICT_SCHEMA:
                return '{"reason": "Two different rates for the same contract.", "verdict": "CONTRADICTS"}'
            return '{"why": "x", "resolution": "superseded", "loser": "a", "reason": "y"}'

    fake = Recording()
    result = await scan_for_contradictions(
        store=store, slm_client=fake, customer_id=customer.id,
    )
    assert result.conflicts_found == 1
    verdict_calls = [k for k in fake.sent if k.get("json_schema") is VERDICT_SCHEMA]
    assert len(verdict_calls) == 1
    assert verdict_calls[0]["max_tokens"] >= 100
    assert verdict_calls[0]["tier"] == "small"
