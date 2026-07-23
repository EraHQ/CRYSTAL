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
    _subject_of,
)
from crystal_cache.llm import reset_llm_client, set_llm_client

from fakes import NotReadyLLM

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Bespoke discriminator fake
# ---------------------------------------------------------------------------

class FakeDiscriminator:
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

def test_subject_of_parses_sparse_key():
    assert _subject_of(_fact("f", "c", "x", "Contract|Rate|Hourly|Legal")) == "Hourly"
    # Free text (no pipes) → no subject.
    assert _subject_of(_fact("f", "c", "x", "just some free text")) is None
    # Too few segments → no subject.
    assert _subject_of(_fact("f", "c", "x", "Source|Locator")) is None


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


def test_enumerate_same_subject_across_crystals():
    facts = [
        _fact("f1", "cA", "rate is 120", "Contract|x|Rate|Legal"),
        _fact("f2", "cB", "rate is 95", "Contract|y|Rate|Legal"),
        # Different subject, different crystal → NOT a candidate.
        _fact("f3", "cC", "unrelated", "Contract|z|Parties|Legal"),
    ]
    pairs = _enumerate_candidate_pairs(facts, max_pairs=100)
    ids = {frozenset((a.id, b.id)) for a, b in pairs}
    assert frozenset(("f1", "f2")) in ids  # same Subject, different crystals
    assert frozenset(("f1", "f3")) not in ids
    assert frozenset(("f2", "f3")) not in ids


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
                     claim="The contract rate is $120/hr", key="Contract|x|Rate|Legal")
    await _seed_fact(store, fid="f2", crystal_id="cA",
                     claim="The contract rate is $95/hr", key="Contract|y|Rate|Legal",
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
                     claim="Deductible is $500", key="Policy|secA|Deductible|Health")
    await _seed_fact(store, fid="f2", crystal_id="cB",
                     claim="Deductible is $1500", key="Policy|secB|Deductible|Health",
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
