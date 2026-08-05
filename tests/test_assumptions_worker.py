"""Assumptions worker slice 2 (crystal_cache.workers.assumptions).

Exercises _run_one_cycle against the in-memory store: the rotating
customer slice (fairness across cycles via the shared state dict),
end-to-end writes through the scan for every customer in the slice,
the seam-not-ready no-op, the empty-bank no-op, and the loop
wrapper's prompt shutdown.

The scan's own behavior (pairing, verdicts, birth fields, ledger) is
covered by tests/test_assumptions_core.py — these tests pin the
WORKER's contract: who gets scanned, when, and that counts aggregate.

R14 note: verified by `pytest`; the same assertions ran green in the
container rig at authoring time (2026-08-05).
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from sqlalchemy import select

from crystal_cache.infrastructure.schema import CrystalChainRow, CrystalRow, FactRow
from crystal_cache.llm import reset_llm_client, set_llm_client
from crystal_cache.llm.client import LLMResult
from crystal_cache.workers.assumptions import (
    _run_one_cycle,
    run_assumptions_worker,
)

from fakes import NotReadyLLM

_BRIDGE_VERDICT = {
    "assumption_exists": True,
    "statement": "These two crystals jointly imply a bridge",
    "subject": "Bridged subject",
    "confidence": 0.9,
    "reasoning": "both sides",
}


class VerdictFake:
    """Seam-shaped client: every call returns the bridge verdict."""

    def __init__(self):
        self.calls = 0

    def is_ready(self) -> bool:
        return True

    def complete_detailed(self, **kwargs: Any) -> LLMResult:
        self.calls += 1
        return LLMResult(
            text=json.dumps(_BRIDGE_VERDICT),
            model="fake-small-model",
            input_tokens=100,
            output_tokens=30,
        )


async def _seed_chained_pair(store, customer_id, prefix):
    a, b = f"{prefix}_a", f"{prefix}_b"
    async with store.session() as s:
        for cid, summary in ((a, "side A"), (b, "side B")):
            s.add(CrystalRow(
                id=cid, customer_id=customer_id,
                crystal_type="customer:legacy", summary_vector=[],
                summary_text=summary,
            ))
        s.add(FactRow(
            id=f"{prefix}_fa", crystal_id=a, pair_type="question_answer",
            prompt_text="Doc | x | Topic | D", claim_text="claim A",
            source_kind="model_reasoning", vector=[],
        ))
        s.add(FactRow(
            id=f"{prefix}_fb", crystal_id=b, pair_type="question_answer",
            prompt_text="Doc | y | Topic | D", claim_text="claim B",
            source_kind="model_reasoning", vector=[],
        ))
        s.add(CrystalChainRow(
            source_crystal_id=a, target_crystal_id=b,
            direction="source_uses_target",
        ))
        await s.commit()


async def _assumption_tenants(store) -> set[str]:
    async with store.session() as s:
        rows = (await s.execute(
            select(CrystalRow.customer_id).where(
                CrystalRow.crystal_type == "assumption"
            )
        )).scalars().all()
        return set(rows)


async def test_cycle_scans_slice_and_aggregates(
    store, customer, semantic_encoder_stub,
):
    other = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="sk-two",
    )
    await _seed_chained_pair(store, customer.id, "c1")
    await _seed_chained_pair(store, other.id, "c2")

    fake = VerdictFake()
    out = await _run_one_cycle(
        store=store, encoder=semantic_encoder_stub,
        customers_per_cycle=2, slm_client=fake,
    )

    assert out["customers_scanned"] == 2
    assert out["assumptions_written"] == 2
    assert out["pairs_evaluated"] == 2
    assert fake.calls == 2
    assert await _assumption_tenants(store) == {customer.id, other.id}


async def test_rotation_advances_across_cycles(
    store, customer, semantic_encoder_stub,
):
    """customers_per_cycle=1 over two customers: two cycles sharing one
    state dict must cover BOTH tenants exactly once each."""
    other = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="sk-two",
    )
    await _seed_chained_pair(store, customer.id, "c1")
    await _seed_chained_pair(store, other.id, "c2")

    state: dict = {}
    fake = VerdictFake()

    out1 = await _run_one_cycle(
        store=store, encoder=semantic_encoder_stub,
        customers_per_cycle=1, slm_client=fake, state=state,
    )
    assert out1["customers_scanned"] == 1
    assert state["cust_offset"] == 1
    tenants_after_1 = await _assumption_tenants(store)
    assert len(tenants_after_1) == 1

    out2 = await _run_one_cycle(
        store=store, encoder=semantic_encoder_stub,
        customers_per_cycle=1, slm_client=fake, state=state,
    )
    assert out2["customers_scanned"] == 1
    assert state["cust_offset"] == 0        # wrapped around
    assert await _assumption_tenants(store) == {customer.id, other.id}

    # A third cycle revisits the first tenant and SKIPS (idempotence
    # composes with rotation — no model spend on a settled slice).
    calls_before = fake.calls
    out3 = await _run_one_cycle(
        store=store, encoder=semantic_encoder_stub,
        customers_per_cycle=1, slm_client=fake, state=state,
    )
    assert out3["skipped_existing"] == 1
    assert out3["assumptions_written"] == 0
    assert fake.calls == calls_before


async def test_not_ready_seam_skips_cycle(
    store, customer, semantic_encoder_stub,
):
    await _seed_chained_pair(store, customer.id, "c1")
    set_llm_client(NotReadyLLM())
    try:
        out = await _run_one_cycle(
            store=store, encoder=semantic_encoder_stub,
            customers_per_cycle=3, slm_client=None,
        )
    finally:
        reset_llm_client()
    assert out["customers_scanned"] == 0
    assert await _assumption_tenants(store) == set()


async def test_empty_bank_is_noop(store, semantic_encoder_stub):
    out = await _run_one_cycle(
        store=store, encoder=semantic_encoder_stub,
        customers_per_cycle=3, slm_client=VerdictFake(),
    )
    assert out == {
        "customers_scanned": 0,
        "pairs_evaluated": 0,
        "assumptions_written": 0,
        "skipped_existing": 0,
    }


async def test_loop_stops_promptly_on_shutdown(
    store, semantic_encoder_stub,
):
    shutdown = asyncio.Event()
    shutdown.set()
    await asyncio.wait_for(
        run_assumptions_worker(
            store=store, encoder=semantic_encoder_stub,
            shutdown_event=shutdown,
        ),
        timeout=5,
    )
