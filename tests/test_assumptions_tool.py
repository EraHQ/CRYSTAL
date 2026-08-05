"""Assumptions slice 3 — the agent `assume` tool
(crystal_cache.agent.tools.curation.assume).

Exercises the tool against the in-memory store with the seam fake:
registration surface (agent-only, customer_id injected), the
ephemeral default (verdict returned, NOTHING written), persist=true
writing the quarantined recall-gated crystal, the agent-judgment
override (persist honored below the worker's min_confidence — gated
birth is the guard, not the number), the tenancy guard BOTH at the
tool (clear error) and at the inference core's hydration (zero model
calls on a foreign parent — no cross-tenant content ever reaches a
prompt), input validation, the no-bridge verdict, and the not-ready
seam.

R14 note: verified by `pytest`; the same assertions ran green in the
container rig at authoring time (2026-08-05).
"""
from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select

from crystal_cache.agent.tool_registry import get_registry, import_all_tools
from crystal_cache.agent.tools.curation import assume
from crystal_cache.agent.tools.retrievers import set_tool_state
from crystal_cache.infrastructure.schema import CrystalRow, FactRow
from crystal_cache.llm import reset_llm_client, set_llm_client
from crystal_cache.llm.client import LLMResult

from fakes import NotReadyLLM

_BRIDGE_VERDICT = {
    "assumption_exists": True,
    "statement": "Friday deploys through Cloud Run carry elevated failure risk",
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


class SeamFake:
    """Injected via set_llm_client: ready, detailed-shape, scripted."""

    def __init__(self, verdicts=None):
        self.verdicts = list(verdicts or [])
        self.calls: list[dict[str, Any]] = []

    def is_ready(self) -> bool:
        return True

    def complete_detailed(self, **kwargs: Any) -> LLMResult:
        self.calls.append(kwargs)
        v = self.verdicts.pop(0) if self.verdicts else _NO_BRIDGE_VERDICT
        return LLMResult(
            text=json.dumps(v),
            model="fake-small-model",
            input_tokens=100,
            output_tokens=30,
        )


async def _seed_crystal(store, crystal_id, customer_id, *, summary,
                        claim=None):
    async with store.session() as s:
        s.add(CrystalRow(
            id=crystal_id, customer_id=customer_id,
            crystal_type="customer:legacy", summary_vector=[],
            summary_text=summary,
        ))
        if claim:
            s.add(FactRow(
                id=f"f_{crystal_id}", crystal_id=crystal_id,
                pair_type="question_answer",
                prompt_text="Doc | x | Topic | D", claim_text=claim,
                source_kind="model_reasoning", vector=[],
            ))
        await s.commit()


async def _seed_pair(store, customer_id):
    await _seed_crystal(store, "cr_a", customer_id,
                        summary="Deploys go through Cloud Run",
                        claim="All deploys go through Cloud Run")
    await _seed_crystal(store, "cr_b", customer_id,
                        summary="Deploy failures spike on Fridays",
                        claim="Deploy failures spike on Fridays")


async def _assumption_rows(store):
    async with store.session() as s:
        return list((await s.execute(
            select(CrystalRow).where(
                CrystalRow.crystal_type == "assumption"
            )
        )).scalars().all())


def test_assume_registered_agent_only():
    import_all_tools()
    tool = get_registry().get("assume")
    assert tool is not None, "assume not registered"
    assert tool.contexts == frozenset({"agent"})
    assert "customer_id" not in tool.parameters_schema["properties"]
    assert set(tool.parameters_schema["required"]) == {
        "crystal_a_id", "crystal_b_id",
    }
    assert (
        tool.parameters_schema["properties"]["persist"]["default"] is False
    )


async def test_ephemeral_default_writes_nothing(
    store, customer, tool_state,
):
    set_tool_state(tool_state)
    await _seed_pair(store, customer.id)
    fake = SeamFake([_BRIDGE_VERDICT])
    set_llm_client(fake)
    try:
        out = await assume(customer.id, "cr_a", "cr_b")
    finally:
        reset_llm_client()

    assert out["assumption_exists"] is True
    assert out["statement"] == _BRIDGE_VERDICT["statement"]
    assert out["subject"] == _BRIDGE_VERDICT["subject"]
    assert out["confidence"] == 0.82
    assert out["persisted"] is False
    assert "crystal_id" not in out
    assert await _assumption_rows(store) == []      # nothing written


async def test_persist_writes_gated_crystal(store, customer, tool_state):
    set_tool_state(tool_state)
    await _seed_pair(store, customer.id)
    set_llm_client(SeamFake([_BRIDGE_VERDICT]))
    try:
        out = await assume(customer.id, "cr_a", "cr_b", persist=True)
    finally:
        reset_llm_client()

    assert out["persisted"] is True
    rows = await _assumption_rows(store)
    assert len(rows) == 1
    row = rows[0]
    assert out["crystal_id"] == row.id
    assert row.quality_tier == "quarantine"
    assert bool(row.recall_gated) is True
    assert row.origin == "assumptions"


async def test_persist_honored_below_worker_threshold(
    store, customer, tool_state,
):
    """Agent judgment overrides the worker's min_confidence — the gated
    birth is the structural guard (mechanism in code, judgment in
    models)."""
    set_tool_state(tool_state)
    await _seed_pair(store, customer.id)
    weak = dict(_BRIDGE_VERDICT, confidence=0.3)
    set_llm_client(SeamFake([weak]))
    try:
        out = await assume(customer.id, "cr_a", "cr_b", persist=True)
    finally:
        reset_llm_client()

    assert out["persisted"] is True
    assert out["confidence"] == 0.3
    rows = await _assumption_rows(store)
    assert len(rows) == 1
    assert bool(rows[0].recall_gated) is True       # the actual guard


async def test_foreign_crystal_refused_before_any_model_call(
    store, customer, tool_state,
):
    """The tenancy guard fires at the tool AND no cross-tenant content
    ever reaches a prompt: zero model calls."""
    set_tool_state(tool_state)
    await _seed_crystal(store, "cr_a", customer.id, summary="mine",
                        claim="my claim")
    foreign = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="sk-other",
    )
    await _seed_crystal(store, "cr_foreign", foreign.id, summary="theirs",
                        claim="their secret claim")

    fake = SeamFake([_BRIDGE_VERDICT])
    set_llm_client(fake)
    try:
        out = await assume(customer.id, "cr_a", "cr_foreign",
                           persist=True)
    finally:
        reset_llm_client()

    assert out["assumption_exists"] is False
    assert out["persisted"] is False
    assert "not found in this tenant's bank" in out["error"]
    assert fake.calls == []                          # nothing hydrated
    assert await _assumption_rows(store) == []


async def test_input_validation(store, customer, tool_state):
    set_tool_state(tool_state)
    out = await assume(customer.id, "", "cr_b")
    assert out["error"] == "both crystal ids are required."
    out = await assume(customer.id, "cr_a", "cr_a")
    assert "DISTINCT" in out["error"]
    assert await _assumption_rows(store) == []


async def test_no_bridge_verdict_is_not_an_error(
    store, customer, tool_state,
):
    set_tool_state(tool_state)
    await _seed_pair(store, customer.id)
    set_llm_client(SeamFake([_NO_BRIDGE_VERDICT]))
    try:
        out = await assume(customer.id, "cr_a", "cr_b", persist=True)
    finally:
        reset_llm_client()

    assert out["assumption_exists"] is False
    assert out["persisted"] is False
    assert "error" not in out
    assert await _assumption_rows(store) == []


async def test_not_ready_seam_returns_error(store, customer, tool_state):
    set_tool_state(tool_state)
    await _seed_pair(store, customer.id)
    set_llm_client(NotReadyLLM())
    try:
        out = await assume(customer.id, "cr_a", "cr_b")
    finally:
        reset_llm_client()
    assert out["assumption_exists"] is False
    assert "no model provider" in out["error"]
