"""Assumptions slice 1 — inference core + store substrate
(crystal_cache.scan.assumptions + AssumptionExtensionsMixin).

Exercises run_assumptions_scan against the in-memory store with
seam-shaped fakes: the chained-pair read (canonicalization of the
bidirectional two-row shape, tenant scoping, assumption-endpoint
exclusion), the write path's ratified birth fields (type=assumption,
tier=quarantine, recall_gated, origin='assumptions' per Q2=B, TWO
parent chain edges, Q1=B type registration), per-pair idempotence,
the min-confidence write gate, the gap-seeded pairing input, seam
json_schema pass-through + origin-tagged ledger rows (RQ3=B), the
no-provider no-op, the fail-safe path, and the write primitive's
parent guards.

R14 note: these assertions are verified by `pytest`; container
validation of the same behaviors ran green at authoring time
(2026-08-04, /tmp/work3 rig, 10 checks).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from sqlalchemy import select

from crystal_cache.infrastructure.schema import (
    CrystalChainRow, CrystalRow, FactRow, LlmCallRow,
)
from crystal_cache.llm import reset_llm_client, set_llm_client
from crystal_cache.llm.client import LLMResult
from crystal_cache.scan import AssumptionScanResult, run_assumptions_scan
from crystal_cache.scan.assumptions import ASSUMPTION_VERDICT_SCHEMA

from fakes import NotReadyLLM

_T0 = datetime(2026, 8, 1, tzinfo=timezone.utc)

_BRIDGE_VERDICT = {
    "assumption_exists": True,
    "statement": (
        "Friday deploys through Cloud Run carry elevated failure risk"
    ),
    "subject": "Deploy risk windows",
    "confidence": 0.82,
    "reasoning": "A says deploys ride Cloud Run; B says failures spike Fridays",
}

_NO_BRIDGE_VERDICT = {
    "assumption_exists": False,
    "statement": "",
    "subject": "",
    "confidence": 0.0,
    "reasoning": "nothing connects these",
}


class MeteredVerdictFake:
    """Seam-shaped client (complete_detailed -> LLMResult) returning
    scripted structured verdicts. Records call kwargs so tests can
    assert the json_schema pass-through. raise_on_call exercises the
    fail-safe path."""

    def __init__(self, verdicts=None, *, raise_on_call=False):
        self.verdicts = list(verdicts or [])
        self.raise_on_call = raise_on_call
        self.calls: list[dict[str, Any]] = []

    def is_ready(self) -> bool:
        return True

    def complete_detailed(self, **kwargs: Any) -> LLMResult:
        self.calls.append(kwargs)
        if self.raise_on_call:
            raise RuntimeError("simulated upstream failure")
        v = self.verdicts.pop(0) if self.verdicts else _NO_BRIDGE_VERDICT
        return LLMResult(
            text=json.dumps(v),
            model="fake-small-model",
            input_tokens=120,
            output_tokens=40,
        )


async def _seed_crystal(store, crystal_id, customer_id, *, summary=None):
    async with store.session() as s:
        s.add(CrystalRow(
            id=crystal_id, customer_id=customer_id,
            crystal_type="customer:legacy", summary_vector=[],
            summary_text=summary,
        ))


async def _seed_fact(store, *, fid, crystal_id, claim, key="", offset_min=0):
    async with store.session() as s:
        s.add(FactRow(
            id=fid, crystal_id=crystal_id, pair_type="question_answer",
            prompt_text=key, claim_text=claim, source_kind="model_reasoning",
            vector=[], created_at=_T0 + timedelta(minutes=offset_min),
        ))


async def _seed_chain(store, source_id, target_id):
    async with store.session() as s:
        s.add(CrystalChainRow(
            source_crystal_id=source_id, target_crystal_id=target_id,
            direction="source_uses_target",
        ))


async def _seed_chained_pair(store, customer):
    """Two chained crystals in the bidirectional two-row shape."""
    await _seed_crystal(store, "cr_a", customer.id,
                        summary="Deploys go through Cloud Run")
    await _seed_crystal(store, "cr_b", customer.id,
                        summary="Deploy failures spike on Fridays")
    await _seed_fact(store, fid="f_a1", crystal_id="cr_a",
                     claim="All deploys go through Cloud Run",
                     key="Repo | ops.md | Deploys | Ops")
    await _seed_fact(store, fid="f_b1", crystal_id="cr_b",
                     claim="Deploy failures spike on Fridays",
                     key="Repo | incidents.md | Deploys | Ops",
                     offset_min=1)
    await _seed_chain(store, "cr_a", "cr_b")
    await _seed_chain(store, "cr_b", "cr_a")


async def _assumption_rows(store):
    async with store.session() as s:
        stmt = select(CrystalRow).where(
            CrystalRow.crystal_type == "assumption"
        )
        return list((await s.execute(stmt)).scalars().all())


# ---------------------------------------------------------------------------
# Pair read
# ---------------------------------------------------------------------------

async def test_chained_pair_read_canonicalizes_and_scopes(store, customer):
    await _seed_chained_pair(store, customer)
    # Tenancy trap: an edge into a foreign tenant's crystal must not pair.
    foreign = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="sk-other",
    )
    await _seed_crystal(store, "cr_foreign", foreign.id, summary="foreign")
    await _seed_chain(store, "cr_a", "cr_foreign")

    pairs = await store.list_chained_crystal_pairs(customer.id, limit=10)
    # The two-row bidirectional shape collapses to ONE canonical pair;
    # the cross-tenant edge is invisible.
    assert pairs == [("cr_a", "cr_b")]


# ---------------------------------------------------------------------------
# Write path (chained input)
# ---------------------------------------------------------------------------

async def test_scan_writes_assumption_with_birth_fields(
    store, customer, semantic_encoder_stub,
):
    await _seed_chained_pair(store, customer)
    fake = MeteredVerdictFake([_BRIDGE_VERDICT])

    # Defaults path deliberately: limits + threshold resolve from
    # settings (CC_ASSUMPTIONS_*), covering the knob wiring.
    result = await run_assumptions_scan(
        store=store, slm_client=fake, customer_id=customer.id,
        encoder=semantic_encoder_stub,
    )

    assert isinstance(result, AssumptionScanResult)
    assert result.chained_pairs_seen == 1
    assert result.pairs_evaluated == 1
    assert result.assumptions_written == 1

    rows = await _assumption_rows(store)
    assert len(rows) == 1
    row = rows[0]
    assert row.customer_id == customer.id
    assert row.quality_tier == "quarantine"
    assert bool(row.recall_gated) is True
    assert row.origin == "assumptions"          # Q2=B
    assert row.source_kind == "agent_inferred"
    assert row.build_method == "assumption"
    assert row.parent_crystal_id == "cr_a"      # primary parent
    assert row.summary_text == _BRIDGE_VERDICT["statement"]
    assert any(
        t.startswith("assumption_confidence:0.82")
        for t in (row.diagnostic_tags or [])
    )

    async with store.session() as s:
        facts = list((await s.execute(
            select(FactRow).where(FactRow.crystal_id == row.id)
        )).scalars().all())
        assert len(facts) == 1
        assert facts[0].prompt_text == "Assumptions|Deploy risk windows"
        assert facts[0].claim_text == _BRIDGE_VERDICT["statement"]
        edges = list((await s.execute(
            select(CrystalChainRow).where(
                CrystalChainRow.source_crystal_id == row.id
            )
        )).scalars().all())
        assert sorted(e.target_crystal_id for e in edges) == ["cr_a", "cr_b"]
        assert all(e.direction == "source_uses_target" for e in edges)

    # Q1=B: the type registered create-if-missing, customer scope.
    t = await store.get_crystal_type("assumption")
    assert t is not None
    assert t.scope == "customer"


async def test_seam_passes_json_schema_and_meters_origin(
    store, customer, semantic_encoder_stub,
):
    await _seed_chained_pair(store, customer)
    fake = MeteredVerdictFake([_BRIDGE_VERDICT])

    await run_assumptions_scan(
        store=store, slm_client=fake, customer_id=customer.id,
        encoder=semantic_encoder_stub,
        pairs_limit=5, gaps_limit=0, min_confidence=0.6,
    )

    # The structured schema reached the client, with the Anthropic
    # 400-guard intact (additionalProperties: false — the v75/v76
    # lesson).
    assert len(fake.calls) == 1
    assert fake.calls[0]["json_schema"] is ASSUMPTION_VERDICT_SCHEMA
    assert fake.calls[0]["json_schema"]["additionalProperties"] is False

    # RQ3=B: one origin-tagged ledger row per inference call.
    async with store.session() as session:
        rows = list((await session.execute(
            select(LlmCallRow).where(
                LlmCallRow.customer_id == customer.id
            )
        )).scalars().all())
    assert len(rows) == 1
    assert rows[0].origin == "assumptions"
    assert rows[0].model == "fake-small-model"


# ---------------------------------------------------------------------------
# Idempotence + pairing exclusion
# ---------------------------------------------------------------------------

async def test_rescan_skips_existing_assumption(
    store, customer, semantic_encoder_stub,
):
    await _seed_chained_pair(store, customer)
    await run_assumptions_scan(
        store=store, slm_client=MeteredVerdictFake([_BRIDGE_VERDICT]),
        customer_id=customer.id, encoder=semantic_encoder_stub,
        pairs_limit=5, gaps_limit=0, min_confidence=0.6,
    )

    rerun_fake = MeteredVerdictFake([])
    result = await run_assumptions_scan(
        store=store, slm_client=rerun_fake, customer_id=customer.id,
        encoder=semantic_encoder_stub,
        pairs_limit=5, gaps_limit=0, min_confidence=0.6,
    )

    assert result.skipped_existing == 1
    assert result.pairs_evaluated == 0
    assert result.assumptions_written == 0
    assert rerun_fake.calls == []          # no model spend on a rerun
    assert len(await _assumption_rows(store)) == 1


async def test_assumption_endpoints_excluded_from_pairing(
    store, customer, semantic_encoder_stub,
):
    """The written assumption chains to both parents, but those edges
    must never become pairing input — no speculation on speculation."""
    await _seed_chained_pair(store, customer)
    await run_assumptions_scan(
        store=store, slm_client=MeteredVerdictFake([_BRIDGE_VERDICT]),
        customer_id=customer.id, encoder=semantic_encoder_stub,
        pairs_limit=5, gaps_limit=0, min_confidence=0.6,
    )

    pairs = await store.list_chained_crystal_pairs(customer.id, limit=10)
    assert pairs == [("cr_a", "cr_b")]


# ---------------------------------------------------------------------------
# Write gates
# ---------------------------------------------------------------------------

async def test_below_threshold_not_written(
    store, customer, semantic_encoder_stub,
):
    await _seed_chained_pair(store, customer)
    weak = dict(_BRIDGE_VERDICT, confidence=0.3)
    result = await run_assumptions_scan(
        store=store, slm_client=MeteredVerdictFake([weak]),
        customer_id=customer.id, encoder=semantic_encoder_stub,
        pairs_limit=5, gaps_limit=0, min_confidence=0.6,
    )
    assert result.pairs_evaluated == 1
    assert result.below_threshold == 1
    assert result.assumptions_written == 0
    assert await _assumption_rows(store) == []


async def test_no_bridge_verdict_not_written(
    store, customer, semantic_encoder_stub,
):
    await _seed_chained_pair(store, customer)
    result = await run_assumptions_scan(
        store=store, slm_client=MeteredVerdictFake([_NO_BRIDGE_VERDICT]),
        customer_id=customer.id, encoder=semantic_encoder_stub,
        pairs_limit=5, gaps_limit=0, min_confidence=0.6,
    )
    assert result.pairs_evaluated == 1
    assert result.below_threshold == 0     # counted only when a bridge existed
    assert result.assumptions_written == 0
    assert await _assumption_rows(store) == []


async def test_inference_failure_writes_nothing(
    store, customer, semantic_encoder_stub,
):
    await _seed_chained_pair(store, customer)
    result = await run_assumptions_scan(
        store=store,
        slm_client=MeteredVerdictFake(raise_on_call=True),
        customer_id=customer.id, encoder=semantic_encoder_stub,
        pairs_limit=5, gaps_limit=0, min_confidence=0.6,
    )
    assert result.pairs_evaluated == 1
    assert result.assumptions_written == 0
    assert await _assumption_rows(store) == []


# ---------------------------------------------------------------------------
# Gap-seeded pairing
# ---------------------------------------------------------------------------

async def _seed_gap_scenario(store, customer):
    """Facts under one Subject spread across TWO crystals + an open gap
    on that subject. No chain edges — the gap is the only pairing input."""
    await _seed_crystal(store, "cr_a", customer.id,
                        summary="Deploys go through Cloud Run")
    await _seed_crystal(store, "cr_b", customer.id,
                        summary="Deploy failures spike on Fridays")
    await _seed_fact(store, fid="f_a1", crystal_id="cr_a",
                     claim="All deploys go through Cloud Run",
                     key="Repo | ops.md | Deploys | Ops")
    await _seed_fact(store, fid="f_b1", crystal_id="cr_b",
                     claim="Deploy failures spike on Fridays",
                     key="Repo | incidents.md | Deploys | Ops",
                     offset_min=1)
    return await store.create_knowledge_gap(
        customer.id, domain="Ops", subject="Deploys",
        missing="What mitigates Friday deploy risk?",
        priority="low", source="gap_discovery",
    )


async def test_gap_seeded_pairing_writes_with_gap_tag(
    store, customer, semantic_encoder_stub,
):
    gap = await _seed_gap_scenario(store, customer)
    result = await run_assumptions_scan(
        store=store, slm_client=MeteredVerdictFake([_BRIDGE_VERDICT]),
        customer_id=customer.id, encoder=semantic_encoder_stub,
        pairs_limit=0, gaps_limit=3, min_confidence=0.6,
    )
    assert result.chained_pairs_seen == 0
    assert result.gap_pairs_seen == 1
    assert result.assumptions_written == 1

    rows = await _assumption_rows(store)
    assert len(rows) == 1
    assert f"assumption_gap:{gap.id}" in (rows[0].diagnostic_tags or [])


async def test_gap_subject_in_single_crystal_is_skipped(
    store, customer, semantic_encoder_stub,
):
    """A subject whose facts live in ONE crystal has nothing to bridge."""
    await _seed_crystal(store, "cr_a", customer.id, summary="solo")
    await _seed_fact(store, fid="f_a1", crystal_id="cr_a",
                     claim="only home", key="Repo | a.md | Deploys | Ops")
    await _seed_fact(store, fid="f_a2", crystal_id="cr_a",
                     claim="also here", key="Repo | b.md | Deploys | Ops",
                     offset_min=1)
    await store.create_knowledge_gap(
        customer.id, domain="Ops", subject="Deploys",
        missing="anything?", priority="low", source="gap_discovery",
    )
    fake = MeteredVerdictFake([_BRIDGE_VERDICT])
    result = await run_assumptions_scan(
        store=store, slm_client=fake, customer_id=customer.id,
        encoder=semantic_encoder_stub,
        pairs_limit=0, gaps_limit=3, min_confidence=0.6,
    )
    assert result.gap_pairs_seen == 0
    assert result.assumptions_written == 0
    assert fake.calls == []


# ---------------------------------------------------------------------------
# No-provider no-op + write-primitive guards
# ---------------------------------------------------------------------------

async def test_none_client_is_noop(store, customer, semantic_encoder_stub):
    """A None override with a not-ready seam is a no-op (NotReadyLLM is
    injected so the test never depends on real environment keys)."""
    await _seed_chained_pair(store, customer)
    set_llm_client(NotReadyLLM())
    try:
        result = await run_assumptions_scan(
            store=store, slm_client=None, customer_id=customer.id,
            encoder=semantic_encoder_stub,
        )
    finally:
        reset_llm_client()
    assert isinstance(result, AssumptionScanResult)
    assert result.pairs_evaluated == 0
    assert result.assumptions_written == 0
    assert await _assumption_rows(store) == []


async def test_create_assumption_rejects_bad_parents(
    store, customer, semantic_encoder_stub,
):
    await _seed_crystal(store, "cr_a", customer.id, summary="a")
    foreign = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="sk-other",
    )
    await _seed_crystal(store, "cr_foreign", foreign.id, summary="theirs")

    with pytest.raises(ValueError, match="two distinct"):
        await store.create_assumption_crystal(
            customer.id, statement="x", subject="y",
            parent_a_id="cr_a", parent_b_id="cr_a",
            confidence=0.9, encoder=semantic_encoder_stub,
        )
    with pytest.raises(ValueError, match="different tenant"):
        await store.create_assumption_crystal(
            customer.id, statement="x", subject="y",
            parent_a_id="cr_a", parent_b_id="cr_foreign",
            confidence=0.9, encoder=semantic_encoder_stub,
        )
    with pytest.raises(ValueError, match="does not exist"):
        await store.create_assumption_crystal(
            customer.id, statement="x", subject="y",
            parent_a_id="cr_a", parent_b_id="cr_missing",
            confidence=0.9, encoder=semantic_encoder_stub,
        )
    assert await _assumption_rows(store) == []
