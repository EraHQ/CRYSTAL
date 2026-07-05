"""Never-Idle Convergence — gap-discovery generator
(crystal_cache.scan.gap_discovery).

Exercises discover_gaps against the in-memory store with a generator fake:
a non-NONE answer writes a knowledge_gap (source='gap_discovery', priority
'low') / NONE writes nothing, the min-facts-per-subject gate, per-subject
idempotence (a subject with an existing discovered gap is skipped), the
budget cap, no-provider no-op, the fail-safe path, and free-text facts
(no Subject) not participating.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from crystal_cache.infrastructure.schema import CrystalRow, FactRow
from crystal_cache.scan import GapScanResult, discover_gaps
from crystal_cache.llm import reset_llm_client, set_llm_client

from fakes import NotReadyLLM

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


class FakeGapGen:
    """Seam-shaped client (complete(...) -> str) returning a subject's missing-question.

    rules: list of (needle, response_text); first rule whose needle appears
    in the user content (case-insensitive) wins, else `default`. Default
    'NONE' means "no gap". raise_on_call=True exercises the fail-safe path."""

    def __init__(self, rules=None, default="NONE", *, raise_on_call=False):
        self.rules = rules or []
        self.default = default
        self.raise_on_call = raise_on_call
        self.calls = 0

    def complete(self, **kwargs):
        self.calls += 1
        if self.raise_on_call:
            raise RuntimeError("simulated upstream failure")
        content = kwargs["messages"][0]["content"].upper()
        for needle, response in self.rules:
            if needle.upper() in content:
                return response
        return self.default


async def _seed_crystal(store, crystal_id, customer_id):
    async with store.session() as s:
        s.add(CrystalRow(
            id=crystal_id, customer_id=customer_id,
            crystal_type="customer:legacy", summary_vector=[],
        ))


async def _seed_fact(store, *, fid, crystal_id, claim, key="", offset_min=0):
    async with store.session() as s:
        s.add(FactRow(
            id=fid, crystal_id=crystal_id, pair_type="question_answer",
            prompt_text=key, claim_text=claim, source_kind="model_reasoning",
            vector=[], created_at=_T0 + timedelta(minutes=offset_min),
        ))


async def test_names_gap_for_subject(store, customer):
    await _seed_crystal(store, "cA", customer.id)
    await _seed_fact(store, fid="f1", crystal_id="cA",
                     claim="In-network deductible is $500",
                     key="Policy|secA|Deductible|Health")
    await _seed_fact(store, fid="f2", crystal_id="cA",
                     claim="Deductible resets each January",
                     key="Policy|secB|Deductible|Health", offset_min=1)

    fake = FakeGapGen(rules=[("DEDUCTIBLE", "What is the out-of-network deductible?")])
    result = await discover_gaps(
        store=store, slm_client=fake, customer_id=customer.id,
    )

    assert result.subjects_seen == 1
    assert result.subjects_evaluated == 1
    assert result.gaps_found == 1
    gaps = await store.list_knowledge_gaps(customer.id, status="open")
    assert len(gaps) == 1
    assert gaps[0].source == "gap_discovery"
    assert gaps[0].subject == "Deductible"
    assert gaps[0].domain == "Health"
    assert gaps[0].priority == "low"
    assert gaps[0].missing == "What is the out-of-network deductible?"


async def test_none_answer_writes_no_gap(store, customer):
    await _seed_crystal(store, "cA", customer.id)
    await _seed_fact(store, fid="f1", crystal_id="cA", claim="fact one",
                     key="Topic|a|Coverage|Domain")
    await _seed_fact(store, fid="f2", crystal_id="cA", claim="fact two",
                     key="Topic|b|Coverage|Domain", offset_min=1)

    fake = FakeGapGen(default="NONE")
    result = await discover_gaps(
        store=store, slm_client=fake, customer_id=customer.id,
    )
    assert result.subjects_evaluated == 1
    assert result.gaps_found == 0
    assert await store.list_knowledge_gaps(customer.id, status="open") == []


async def test_min_facts_gate_excludes_thin_subjects(store, customer):
    # A subject with a single fact is not a candidate (default min is 2).
    await _seed_crystal(store, "cA", customer.id)
    await _seed_fact(store, fid="f1", crystal_id="cA", claim="lonely fact",
                     key="Topic|a|Solo|Domain")

    fake = FakeGapGen(default="What is missing?")
    result = await discover_gaps(
        store=store, slm_client=fake, customer_id=customer.id,
    )
    assert result.subjects_seen == 0
    assert result.subjects_evaluated == 0
    assert result.gaps_found == 0


async def test_subject_with_existing_discovered_gap_is_skipped(store, customer):
    # Pre-seed an open gap_discovery gap for the subject → idempotent skip.
    await store.create_knowledge_gap(
        customer.id, domain="Health", subject="Deductible",
        missing="a previously discovered gap", source="gap_discovery",
    )
    await _seed_crystal(store, "cA", customer.id)
    await _seed_fact(store, fid="f1", crystal_id="cA", claim="ded a",
                     key="Policy|a|Deductible|Health")
    await _seed_fact(store, fid="f2", crystal_id="cA", claim="ded b",
                     key="Policy|b|Deductible|Health", offset_min=1)

    fake = FakeGapGen(rules=[("DEDUCTIBLE", "What is the family deductible?")])
    result = await discover_gaps(
        store=store, slm_client=fake, customer_id=customer.id,
    )
    assert result.subjects_seen == 1
    assert result.skipped_existing == 1
    assert result.subjects_evaluated == 0
    assert result.gaps_found == 0
    # Still just the one pre-existing gap.
    assert len(await store.list_knowledge_gaps(customer.id, status="open")) == 1


async def test_budget_caps_subject_evaluations(store, customer):
    await _seed_crystal(store, "cA", customer.id)
    # Two subjects, two facts each → two candidates; cap at one.
    await _seed_fact(store, fid="a1", crystal_id="cA", claim="a one",
                     key="T|x|SubA|D", offset_min=0)
    await _seed_fact(store, fid="a2", crystal_id="cA", claim="a two",
                     key="T|y|SubA|D", offset_min=1)
    await _seed_fact(store, fid="b1", crystal_id="cA", claim="b one",
                     key="T|x|SubB|D", offset_min=2)
    await _seed_fact(store, fid="b2", crystal_id="cA", claim="b two",
                     key="T|y|SubB|D", offset_min=3)

    fake = FakeGapGen(default="What is missing here?")
    result = await discover_gaps(
        store=store, slm_client=fake, customer_id=customer.id,
        max_subjects=1,
    )
    assert result.subjects_seen == 2
    assert result.subjects_evaluated == 1
    assert result.budget_exhausted is True
    assert result.gaps_found == 1


async def test_none_client_is_noop(store, customer):
    """A None override with a not-ready seam is a no-op (NotReadyLLM is
    injected so the test never depends on real environment keys)."""
    await _seed_crystal(store, "cA", customer.id)
    await _seed_fact(store, fid="f1", crystal_id="cA", claim="x",
                     key="T|a|Sub|D")
    await _seed_fact(store, fid="f2", crystal_id="cA", claim="y",
                     key="T|b|Sub|D", offset_min=1)

    set_llm_client(NotReadyLLM())
    try:
        result = await discover_gaps(
            store=store, slm_client=None, customer_id=customer.id,
        )
    finally:
        reset_llm_client()
    assert isinstance(result, GapScanResult)
    assert result.gaps_found == 0
    assert result.subjects_evaluated == 0
    assert await store.list_knowledge_gaps(customer.id, status="open") == []


async def test_generator_error_writes_nothing(store, customer):
    await _seed_crystal(store, "cA", customer.id)
    await _seed_fact(store, fid="f1", crystal_id="cA", claim="x",
                     key="T|a|Sub|D")
    await _seed_fact(store, fid="f2", crystal_id="cA", claim="y",
                     key="T|b|Sub|D", offset_min=1)

    fake = FakeGapGen(raise_on_call=True)
    result = await discover_gaps(
        store=store, slm_client=fake, customer_id=customer.id,
    )
    assert result.subjects_evaluated == 1
    assert result.gaps_found == 0
    assert await store.list_knowledge_gaps(customer.id, status="open") == []


async def test_free_text_facts_have_no_subject(store, customer):
    # Facts whose prompt_text isn't a pipe-delimited sparse key have no
    # Subject and don't participate in gap discovery.
    await _seed_crystal(store, "cA", customer.id)
    await _seed_fact(store, fid="f1", crystal_id="cA", claim="free text one")
    await _seed_fact(store, fid="f2", crystal_id="cA", claim="free text two",
                     offset_min=1)

    fake = FakeGapGen(default="What is missing?")
    result = await discover_gaps(
        store=store, slm_client=fake, customer_id=customer.id,
    )
    assert result.subjects_seen == 0
    assert result.gaps_found == 0
