"""Never-Idle Convergence — dedup-scan generator
(crystal_cache.scan.dedup).

The dedup scan reuses contradiction.py's candidate enumeration + pair_key +
subject/provenance helpers verbatim, so the pure-helper layer is already
covered by test_contradiction_scan. This file exercises scan_for_duplicates
against the in-memory store with a dedup discriminator fake:
  DUPLICATE → row (detector='dedup_scan') / DISTINCT|UNRELATED → no row,
  idempotent re-run, the budget cap, no-provider no-op, the fail-safe ERROR
  path, and the SHARED KEYSPACE (a pair already recorded by the contradiction
  scan is skipped here — the two generators never double-write one pair).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from crystal_cache.infrastructure.schema import CrystalRow, FactRow
from crystal_cache.scan import DedupScanResult, scan_for_duplicates
from crystal_cache.scan.contradiction import _pair_key
from crystal_cache.llm import reset_llm_client, set_llm_client

from fakes import NotReadyLLM

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


class FakeDedup:
    """Seam-shaped client (complete(...) -> str) returning a one-word dedup verdict.

    rules: list of (needle_a, needle_b, verdict); first rule whose BOTH
    needles appear (case-insensitive) wins, else `default`.
    raise_on_call=True exercises the fail-safe ERROR path."""

    def __init__(self, rules=None, default="DISTINCT", *, raise_on_call=False):
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


async def _seed_crystal(store, crystal_id, customer_id):
    async with store.session() as s:
        s.add(CrystalRow(
            id=crystal_id, customer_id=customer_id,
            crystal_type="customer:legacy", summary_vector=[],
        ))


async def _seed_fact(store, *, fid, crystal_id, claim, key="", offset_min=0,
                     pair_type="question_answer", source_doc_id=None):
    async with store.session() as s:
        s.add(FactRow(
            id=fid, crystal_id=crystal_id, pair_type=pair_type,
            prompt_text=key, claim_text=claim, source_kind="model_reasoning",
            source_doc_id=source_doc_id,
            vector=[], created_at=_T0 + timedelta(minutes=offset_min),
        ))


async def test_content_chunks_never_enter_dedup(store, customer):
    """CF-Q1=A (2026-07-23): a chunk and the fact extracted from it are
    provenance, not duplication — the pair must never reach the
    discriminator, no matter how identical the text."""
    await _seed_crystal(store, "cA", customer.id)
    await _seed_fact(store, fid="f1", crystal_id="cA",
                     claim="WS-112 has a unit cost of $58.00",
                     key="Costs|row|WS-112|Data")
    await _seed_fact(store, fid="f2", crystal_id="cA",
                     claim="| WS-112 | Ceramic table lamp | $58.00 | 50 |",
                     key="Costs|table|WS-112|Data", offset_min=1,
                     pair_type="content_chunk")

    fake = FakeDedup(default="DUPLICATE")  # would fire if ever consulted
    result = await scan_for_duplicates(
        store=store, slm_client=fake, customer_id=customer.id,
    )
    assert fake.calls == 0
    assert result.duplicates_found == 0
    assert result.candidate_pairs == 0


async def test_duplicate_creates_open_conflict(store, customer):
    await _seed_crystal(store, "cA", customer.id)
    await _seed_fact(store, fid="f1", crystal_id="cA",
                     claim="The office opens at 9am", key="Hours|x|Opening|Office")
    await _seed_fact(store, fid="f2", crystal_id="cA",
                     claim="We open at nine in the morning",
                     key="Hours|y|Opening|Office", offset_min=1)

    fake = FakeDedup(rules=[("9AM", "NINE IN THE MORNING", "DUPLICATE")])
    result = await scan_for_duplicates(
        store=store, slm_client=fake, customer_id=customer.id,
    )

    assert result.duplicates_found == 1
    assert result.pairs_evaluated == 1
    rows = await store.list_knowledge_conflicts(customer.id, status="open")
    assert len(rows) == 1
    assert rows[0].detector == "dedup_scan"
    assert rows[0].subject == "Opening"
    assert {rows[0].fact_a_id, rows[0].fact_b_id} == {"f1", "f2"}


async def test_distinct_creates_no_conflict(store, customer):
    # Different values must read as DISTINCT (a conflict, not a duplicate) and
    # write nothing here — the dedup scan does not double-flag contradictions.
    await _seed_crystal(store, "cA", customer.id)
    await _seed_fact(store, fid="f1", crystal_id="cA", claim="The rate is $120/hr")
    await _seed_fact(store, fid="f2", crystal_id="cA", claim="The rate is $95/hr",
                     offset_min=1)

    fake = FakeDedup(default="DISTINCT")
    result = await scan_for_duplicates(
        store=store, slm_client=fake, customer_id=customer.id,
    )
    assert result.duplicates_found == 0
    assert await store.list_knowledge_conflicts(customer.id) == []


async def test_rescan_is_idempotent(store, customer):
    await _seed_crystal(store, "cA", customer.id)
    await _seed_fact(store, fid="f1", crystal_id="cA", claim="opens at nine")
    await _seed_fact(store, fid="f2", crystal_id="cA", claim="open at 9", offset_min=1)
    fake = FakeDedup(rules=[("OPENS AT NINE", "OPEN AT 9", "DUPLICATE")])

    first = await scan_for_duplicates(
        store=store, slm_client=fake, customer_id=customer.id,
    )
    assert first.duplicates_found == 1

    second = await scan_for_duplicates(
        store=store, slm_client=fake, customer_id=customer.id,
    )
    assert second.pairs_evaluated == 0
    assert second.skipped_existing == 1
    assert second.duplicates_found == 0
    assert len(await store.list_knowledge_conflicts(customer.id)) == 1


async def test_budget_caps_discriminator_calls(store, customer):
    await _seed_crystal(store, "cA", customer.id)
    for i in range(3):  # C(3,2) = 3 candidate pairs
        await _seed_fact(store, fid=f"f{i}", crystal_id="cA",
                         claim=f"claim number {i}", offset_min=i)

    fake = FakeDedup(default="DUPLICATE")
    result = await scan_for_duplicates(
        store=store, slm_client=fake, customer_id=customer.id,
        max_discriminator_calls=1,
    )
    assert result.candidate_pairs == 3
    assert result.pairs_evaluated == 1
    assert result.budget_exhausted is True
    assert result.duplicates_found == 1


async def test_none_client_is_noop(store, customer):
    """A None override with a not-ready seam is a no-op (NotReadyLLM is
    injected so the test never depends on real environment keys)."""
    await _seed_crystal(store, "cA", customer.id)
    await _seed_fact(store, fid="f1", crystal_id="cA", claim="a")
    await _seed_fact(store, fid="f2", crystal_id="cA", claim="b", offset_min=1)

    set_llm_client(NotReadyLLM())
    try:
        result = await scan_for_duplicates(
            store=store, slm_client=None, customer_id=customer.id,
        )
    finally:
        reset_llm_client()
    assert isinstance(result, DedupScanResult)
    assert result.duplicates_found == 0
    assert result.pairs_evaluated == 0
    assert await store.list_knowledge_conflicts(customer.id) == []


async def test_discriminator_error_writes_nothing(store, customer):
    await _seed_crystal(store, "cA", customer.id)
    await _seed_fact(store, fid="f1", crystal_id="cA", claim="a")
    await _seed_fact(store, fid="f2", crystal_id="cA", claim="b", offset_min=1)

    fake = FakeDedup(raise_on_call=True)
    result = await scan_for_duplicates(
        store=store, slm_client=fake, customer_id=customer.id,
    )
    assert result.pairs_evaluated == 1
    assert result.duplicates_found == 0
    assert await store.list_knowledge_conflicts(customer.id) == []


async def test_shared_keyspace_skips_pair_already_recorded(store, customer):
    """A pair the contradiction scan already wrote (any detector) is skipped
    by the dedup scan — the two generators never double-write one pair."""
    await _seed_crystal(store, "cA", customer.id)
    await _seed_fact(store, fid="f1", crystal_id="cA", claim="same thing said once")
    await _seed_fact(store, fid="f2", crystal_id="cA", claim="same thing said twice",
                     offset_min=1)

    # Pre-record a conflict for this exact pair (default detector =
    # contradiction_scan) using the SAME pair_key the dedup scan computes.
    pk = _pair_key(
        SimpleNamespace(id="f1", claim_text="same thing said once"),
        SimpleNamespace(id="f2", claim_text="same thing said twice"),
    )
    await store.create_knowledge_conflict(
        customer.id, fact_a_id="f1", fact_b_id="f2",
        claim_a="same thing said once", claim_b="same thing said twice",
        pair_key=pk,
    )

    fake = FakeDedup(default="DUPLICATE")  # would fire if the pair were checked
    result = await scan_for_duplicates(
        store=store, slm_client=fake, customer_id=customer.id,
    )
    assert result.pairs_evaluated == 0
    assert result.skipped_existing == 1
    assert result.duplicates_found == 0
    # Still exactly the one pre-existing row; no dedup duplicate written.
    assert len(await store.list_knowledge_conflicts(customer.id)) == 1
