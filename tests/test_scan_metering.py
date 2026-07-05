"""Scan metering (G3 at the seam boundary, 2026-07-02).

The convergence scans emit one origin-tagged llm_calls cost row per
discriminator/generator call when the client exposes complete_detailed;
legacy complete()-only clients run unmetered but functionally identical.

R14 note: these assertions are verified by `pytest`; they describe
expected behavior and have not yet been run at authoring time.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select

from crystal_cache.infrastructure.schema import CrystalRow, FactRow, LlmCallRow
from crystal_cache.llm.client import LLMResult
from crystal_cache.scan import scan_for_contradictions

_T0 = datetime(2026, 7, 1, tzinfo=timezone.utc)


class MeteredFake:
    """Seam-shaped client exposing complete_detailed (the real seam shape)."""

    def __init__(self, verdict: str = "CONSISTENT"):
        self.verdict = verdict
        self.calls = 0

    def is_ready(self) -> bool:
        return True

    def complete_detailed(self, **kwargs: Any) -> LLMResult:
        self.calls += 1
        return LLMResult(
            text=self.verdict,
            model="fake-small-model",
            input_tokens=120,
            output_tokens=4,
        )


async def _seed_pair(store: Any, customer: Any):
    async with store.session() as s:
        s.add(CrystalRow(
            id="c_meter", customer_id=customer.id,
            crystal_type="customer:legacy", summary_vector=[],
        ))
    for i, claim in enumerate(["the sky is blue", "the sky is green"]):
        async with store.session() as s:
            s.add(FactRow(
                id=f"f_meter_{i}", crystal_id="c_meter",
                pair_type="question_answer",
                prompt_text=f"Doc|p{i}|Sky|Test", claim_text=claim,
                source_kind="model_reasoning", vector=[],
                created_at=_T0 + timedelta(minutes=i),
            ))


async def _llm_call_rows(store: Any, customer_id: str):
    async with store.session() as session:
        stmt = select(LlmCallRow).where(LlmCallRow.customer_id == customer_id)
        return list((await session.execute(stmt)).scalars().all())


async def test_contradiction_scan_emits_cost_rows(store, customer):
    await _seed_pair(store, customer)
    fake = MeteredFake()

    await scan_for_contradictions(
        store=store, slm_client=fake, customer_id=customer.id,
    )

    assert fake.calls >= 1
    rows = await _llm_call_rows(store, customer.id)
    assert len(rows) == fake.calls
    assert all(r.origin == "scan_contradiction" for r in rows)
    assert all(r.model == "fake-small-model" for r in rows)
    assert all(r.input_tokens == 120 and r.output_tokens == 4 for r in rows)


async def test_complete_only_clients_run_unmetered(store, customer):
    """Legacy fakes exposing only complete() keep working, with no rows."""
    await _seed_pair(store, customer)

    class LegacyFake:
        def is_ready(self) -> bool:
            return True

        def complete(self, **kwargs: Any) -> str:
            return "CONSISTENT"

    await scan_for_contradictions(
        store=store, slm_client=LegacyFake(), customer_id=customer.id,
    )
    rows = await _llm_call_rows(store, customer.id)
    assert rows == []
