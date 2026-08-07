"""Assumptions funnel F1 — the crystal_edges writer
(crystal_cache.scan.pairing_funnel + the assumption_ext funnel reads).

Exercises the funnel against the in-memory store: every tier emits
edges from its recorded signal (co-citation from grounded citations
grouped by answer turn — ungrounded excluded; co-routing from
conversation sequences over routed_crystal_id + matched_facts;
chained and gap_subject from the existing inputs; structural
key_adjacent from shared sparse-key Sources and vector_similar from
stored routing vectors above the cosine floor). Also: the upsert's
composite-PK accumulate + canonical ordering, watermarks preventing
double-counting across passes, the structural rotation walking the
full pair space, assumption-crystal exclusion funnel-wide, and
delete_crystal removing a dying crystal's edges in the same
transaction (both edge columns are FKs — Postgres-fatal otherwise).

R14 note: verified by `pytest`; the same assertions ran green in the
container rig at authoring time (2026-08-06).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from crystal_cache.infrastructure.schema import (
    CitationRow, CrystalChainRow, CrystalEdgeRow, CrystalRow, FactRow,
    KnowledgeGapRow, QueryLogRow,
)
from crystal_cache.scan.pairing_funnel import (
    EDGE_TIER_ORDER, FunnelState, run_pairing_funnel,
)

_T0 = datetime(2026, 8, 1, tzinfo=timezone.utc)


async def _seed_crystal(store, crystal_id, customer_id, *,
                        crystal_type="customer:legacy", vector=None):
    async with store.session() as s:
        s.add(CrystalRow(
            id=crystal_id, customer_id=customer_id,
            crystal_type=crystal_type, summary_vector=[],
            summary_text=f"about {crystal_id}",
            routing_vector=vector,
        ))


async def _seed_fact(store, *, crystal_id, key, offset_min=0):
    async with store.session() as s:
        s.add(FactRow(
            id=f"f_{uuid.uuid4().hex[:12]}", crystal_id=crystal_id,
            pair_type="question_answer", prompt_text=key,
            claim_text=f"claim in {crystal_id}",
            source_kind="model_reasoning", vector=[],
            created_at=_T0 + timedelta(minutes=offset_min),
        ))


async def _seed_citation(store, customer_id, *, turn_id, crystal_id,
                         grounded=True, offset_min=0):
    async with store.session() as s:
        s.add(CitationRow(
            id=f"cit_{uuid.uuid4().hex[:12]}", customer_id=customer_id,
            query_log_id=turn_id, crystal_id=crystal_id,
            handle=1, claim_span="span", grounded=grounded,
            created_at=_T0 + timedelta(minutes=offset_min),
        ))


async def _seed_query_log(store, customer_id, *, sequence_id,
                          routed=None, matched=None, offset_min=0):
    async with store.session() as s:
        s.add(QueryLogRow(
            id=f"ql_{uuid.uuid4().hex[:12]}", customer_id=customer_id,
            query_text="q", query_vector=[], match_type="high",
            sequence_id=sequence_id, routed_crystal_id=routed,
            matched_facts=matched or [],
            timestamp=_T0 + timedelta(minutes=offset_min),
        ))


async def _edges(store, edge_type=None):
    async with store.session() as s:
        stmt = select(CrystalEdgeRow)
        if edge_type:
            stmt = stmt.where(CrystalEdgeRow.edge_type == edge_type)
        return list((await s.execute(stmt)).scalars().all())


async def test_co_citation_edges_grounded_only(store, customer):
    for cid in ("cr_a", "cr_b", "cr_c"):
        await _seed_crystal(store, cid, customer.id)
    # One answer turn cites a+b (grounded) and c (ungrounded).
    await _seed_citation(store, customer.id, turn_id="ql_1",
                         crystal_id="cr_a")
    await _seed_citation(store, customer.id, turn_id="ql_1",
                         crystal_id="cr_b", offset_min=1)
    await _seed_citation(store, customer.id, turn_id="ql_1",
                         crystal_id="cr_c", grounded=False, offset_min=2)
    # A different turn cites a+b again — weight accumulates.
    await _seed_citation(store, customer.id, turn_id="ql_2",
                         crystal_id="cr_a", offset_min=3)
    await _seed_citation(store, customer.id, turn_id="ql_2",
                         crystal_id="cr_b", offset_min=4)

    result = await run_pairing_funnel(
        store=store, customer_id=customer.id, state=FunnelState(),
    )
    assert result.co_cited_edges == 2          # two turn-pairs emitted
    rows = await _edges(store, "co_cited")
    assert len(rows) == 1                      # ONE edge row, accumulated
    row = rows[0]
    assert (row.crystal_a_id, row.crystal_b_id) == ("cr_a", "cr_b")
    assert row.weight == 2.0                   # both turns counted


async def test_co_routing_edges_from_sequence_and_matched_facts(
    store, customer,
):
    for cid in ("cr_a", "cr_b"):
        await _seed_crystal(store, cid, customer.id)
    async with store.session() as s:
        s.add(FactRow(
            id="f_b1", crystal_id="cr_b", pair_type="question_answer",
            prompt_text="Doc | x | T | D", claim_text="c",
            source_kind="model_reasoning", vector=[], created_at=_T0,
        ))
    # Turn 1 routes to cr_a; turn 2 of the SAME conversation retrieves
    # a fact living in cr_b — the conversation touched both.
    await _seed_query_log(store, customer.id, sequence_id="seq_1",
                          routed="cr_a")
    await _seed_query_log(store, customer.id, sequence_id="seq_1",
                          matched=["f_b1"], offset_min=1)

    result = await run_pairing_funnel(
        store=store, customer_id=customer.id, state=FunnelState(),
    )
    assert result.co_routed_edges == 1
    rows = await _edges(store, "co_routed")
    assert len(rows) == 1
    assert (rows[0].crystal_a_id, rows[0].crystal_b_id) == ("cr_a", "cr_b")


async def test_watermarks_prevent_double_counting(store, customer):
    for cid in ("cr_a", "cr_b"):
        await _seed_crystal(store, cid, customer.id)
    await _seed_citation(store, customer.id, turn_id="ql_1",
                         crystal_id="cr_a")
    await _seed_citation(store, customer.id, turn_id="ql_1",
                         crystal_id="cr_b", offset_min=1)

    state = FunnelState()
    await run_pairing_funnel(
        store=store, customer_id=customer.id, state=state,
    )
    # Second pass over the SAME history: watermark excludes it.
    result2 = await run_pairing_funnel(
        store=store, customer_id=customer.id, state=state,
    )
    assert result2.co_cited_edges == 0
    rows = await _edges(store, "co_cited")
    assert rows[0].weight == 1.0               # not double-counted

    # New citations past the watermark DO count.
    await _seed_citation(store, customer.id, turn_id="ql_9",
                         crystal_id="cr_a", offset_min=10)
    await _seed_citation(store, customer.id, turn_id="ql_9",
                         crystal_id="cr_b", offset_min=11)
    result3 = await run_pairing_funnel(
        store=store, customer_id=customer.id, state=state,
    )
    assert result3.co_cited_edges == 1
    rows = await _edges(store, "co_cited")
    assert rows[0].weight == 2.0


async def test_chained_gap_and_structural_tiers(store, customer):
    va = [1.0, 0.0, 0.0]
    vb = [0.9, 0.1, 0.0]        # cosine ~0.994 with va
    vc = [0.0, 0.0, 1.0]        # orthogonal
    await _seed_crystal(store, "cr_a", customer.id, vector=va)
    await _seed_crystal(store, "cr_b", customer.id, vector=vb)
    await _seed_crystal(store, "cr_c", customer.id, vector=vc)
    # Shared Source between a and c (key_adjacent); Subject 'Deploys'
    # spans a and b with an open gap (gap_subject).
    await _seed_fact(store, crystal_id="cr_a",
                     key="Handbook | s1 | Deploys | Ops")
    await _seed_fact(store, crystal_id="cr_b",
                     key="Runbook | s2 | Deploys | Ops", offset_min=1)
    await _seed_fact(store, crystal_id="cr_c",
                     key="Handbook | s9 | Billing | Ops", offset_min=2)
    async with store.session() as s:
        s.add(CrystalChainRow(source_crystal_id="cr_a",
                              target_crystal_id="cr_b",
                              direction="source_uses_target"))
        s.add(KnowledgeGapRow(
            id="gap_1", customer_id=customer.id, domain="Ops",
            subject="Deploys", missing="?", priority="low",
            status="open", source="gap_discovery",
        ))

    result = await run_pairing_funnel(
        store=store, customer_id=customer.id, state=FunnelState(),
        structural_pairs_limit=10,             # covers all 3 pairs
    )
    assert result.chained_edges == 1
    assert result.gap_subject_edges == 1
    assert result.key_adjacent_edges == 1      # a<->c share 'Handbook'
    assert result.vector_similar_edges == 1    # a<->b cosine above floor
    assert result.structural_pairs_examined == 3

    sim_rows = await _edges(store, "vector_similar")
    assert (sim_rows[0].crystal_a_id, sim_rows[0].crystal_b_id) == (
        "cr_a", "cr_b",
    )
    assert sim_rows[0].weight > 0.9
    adj_rows = await _edges(store, "key_adjacent")
    assert (adj_rows[0].crystal_a_id, adj_rows[0].crystal_b_id) == (
        "cr_a", "cr_c",
    )


async def test_structural_rotation_advances(store, customer):
    for cid in ("cr_a", "cr_b", "cr_c"):
        await _seed_crystal(store, cid, customer.id)
    state = FunnelState()
    r1 = await run_pairing_funnel(
        store=store, customer_id=customer.id, state=state,
        structural_pairs_limit=2,
    )
    assert r1.structural_pairs_examined == 2
    assert state.structural_offset == 2
    r2 = await run_pairing_funnel(
        store=store, customer_id=customer.id, state=state,
        structural_pairs_limit=2,
    )
    assert r2.structural_pairs_examined == 2
    assert state.structural_offset == 1        # wrapped (3 pairs total)


async def test_assumption_crystals_excluded_everywhere(
    store, customer, semantic_encoder_stub,
):
    await _seed_crystal(store, "cr_a", customer.id)
    await _seed_crystal(store, "cr_b", customer.id)
    asm = await store.create_assumption_crystal(
        customer.id, statement="s", subject="Subj",
        parent_a_id="cr_a", parent_b_id="cr_b", confidence=0.9,
        encoder=semantic_encoder_stub,
    )
    # A turn co-cites a real crystal AND the assumption.
    await _seed_citation(store, customer.id, turn_id="ql_1",
                         crystal_id="cr_a")
    await _seed_citation(store, customer.id, turn_id="ql_1",
                         crystal_id=asm["crystal_id"], offset_min=1)

    result = await run_pairing_funnel(
        store=store, customer_id=customer.id, state=FunnelState(),
        structural_pairs_limit=50,
    )
    assert result.co_cited_edges == 0
    # The assumption's own chain edges must not leak into the chained
    # tier either (only a->b authored edges would; there are none).
    assert result.chained_edges == 0
    for edge_type in EDGE_TIER_ORDER:
        for row in await _edges(store, edge_type):
            assert asm["crystal_id"] not in (
                row.crystal_a_id, row.crystal_b_id,
            )


async def test_delete_crystal_removes_edges_in_transaction(
    store, customer,
):
    for cid in ("cr_a", "cr_b"):
        await _seed_crystal(store, cid, customer.id)
    await _seed_citation(store, customer.id, turn_id="ql_1",
                         crystal_id="cr_a")
    await _seed_citation(store, customer.id, turn_id="ql_1",
                         crystal_id="cr_b", offset_min=1)
    await run_pairing_funnel(
        store=store, customer_id=customer.id, state=FunnelState(),
    )
    assert len(await _edges(store)) >= 1

    assert await store.delete_crystal("cr_a", customer.id) is True
    for row in await _edges(store):
        assert "cr_a" not in (row.crystal_a_id, row.crystal_b_id)
