"""S4 — the spend-budget substrate + gap dispositions (2026-07-08).

Budgets are rows, not knobs: one cap per (customer, function[, operator]);
the llm_calls ledger's origin stamps are the meter; auto paths consult
function_budget_allows and skip quietly. Manual by default (B-1): no
auto_research row and a zero config default = the fill sweep never burns
a cent for that tenant.
"""
from __future__ import annotations

import pytest

from crystal_cache.control.admission import function_budget_allows


async def test_budget_upsert_and_resolution_order(store, customer):
    tenant = await store.upsert_spend_budget(
        customer.id, function="auto_research", cap_micro_usd=5_000_000)
    assert tenant.cap_micro_usd == 5_000_000

    # Update-in-place, same scope row.
    again = await store.upsert_spend_budget(
        customer.id, function="auto_research", cap_micro_usd=7_000_000)
    assert again.id == tenant.id and again.cap_micro_usd == 7_000_000

    # Operator narrowing wins when present (F1 team seats).
    op_row = await store.upsert_spend_budget(
        customer.id, function="auto_research", cap_micro_usd=1_000_000,
        operator_id="op_alice")
    got = await store.get_spend_budget(
        customer.id, function="auto_research", operator_id="op_alice")
    assert got.id == op_row.id and got.cap_micro_usd == 1_000_000
    # Unknown operator falls back to the tenant row.
    got = await store.get_spend_budget(
        customer.id, function="auto_research", operator_id="op_bob")
    assert got.id == again.id

    assert len(await store.list_spend_budgets(customer.id)) == 2


async def test_function_budget_allows_manual_by_default(store, customer):
    """B-1 TRIPWIRE: no row + zero default = OFF. The auto sweep spends
    nothing unless the tenant (or self-host config) opts in."""
    assert await function_budget_allows(
        store, customer, "auto_research", origin="cognition",
        default_cap_micro_usd=0,
    ) is False

    # Config default enables without a table row (self-host posture).
    assert await function_budget_allows(
        store, customer, "auto_research", origin="cognition",
        default_cap_micro_usd=1_000_000,
    ) is True

    # Explicit zero-cap row = OFF even with a nonzero default upstream.
    await store.upsert_spend_budget(
        customer.id, function="auto_research", cap_micro_usd=0)
    assert await function_budget_allows(
        store, customer, "auto_research", origin="cognition",
        default_cap_micro_usd=1_000_000,
    ) is False


async def test_function_budget_meter_reads_origin_ledger(store, customer):
    """The ledger IS the meter: cognition-origin spend counts; other
    origins don't."""
    await store.upsert_spend_budget(
        customer.id, function="auto_research", cap_micro_usd=100)

    await store.record_llm_call(
        customer.id, model="claude-haiku-4-5",
        input_tokens=0, output_tokens=0, origin="agent",
        price_table={},
    )
    assert await function_budget_allows(
        store, customer, "auto_research", origin="cognition") is True

    rec = await store.record_llm_call(
        customer.id, model="claude-haiku-4-5",
        input_tokens=0, output_tokens=0, origin="cognition",
        price_table={},
    )
    from crystal_cache.infrastructure.schema import LlmCallRow
    async with store.session() as session:
        row = await session.get(LlmCallRow, rec["id"])
        row.computed_cost_micro_usd = 100  # at cap
    assert await function_budget_allows(
        store, customer, "auto_research", origin="cognition") is False


async def test_gap_disposition_persists_and_classifier_defaults(store, customer):
    from crystal_cache.scan.gap_disposition import classify_gap_disposition

    # Explicit valid disposition wins.
    assert classify_gap_disposition("workable") == "workable"
    # Web tools are registered in this env -> researchable.
    assert classify_gap_disposition() == "researchable"
    assert classify_gap_disposition("nonsense") == "researchable"

    gap = await store.create_knowledge_gap(
        customer.id, domain=None, subject="s", missing="m",
        source="manual", disposition="workable",
    )
    assert gap.disposition == "workable"
    listed = await store.list_knowledge_gaps(customer.id, status="open")
    assert listed[0].disposition == "workable"


async def test_promote_endpoint_owner_and_foreign(monkeypatch, store):
    """Manual Research click: owner Key A promotes; foreign gets the
    uniform 404; the task carries the gap's provenance."""
    from crystal_cache.endpoints.customers import promote_gap_to_research
    try:  # tests/ is a package in some environments, bare rootdir in others
        from tests.test_accounts_phase_c import _AuthedReq, _use
    except ModuleNotFoundError:
        from test_accounts_phase_c import _AuthedReq, _use
    _use(monkeypatch)
    from fastapi import HTTPException

    a = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="enc:v1:x")
    b = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="enc:v1:x")
    gap = await store.create_knowledge_gap(
        a.id, domain=None, subject="s", missing="find the thing",
        source="manual", full_key="Doc|p1|Thing|Test",
        triggering_query="where is the thing",
    )

    with pytest.raises(HTTPException) as e:
        await promote_gap_to_research(
            gap.id, _AuthedReq(f"Bearer {b.api_key}", body={}), store)
    assert e.value.status_code == 404

    import json
    out = await promote_gap_to_research(
        gap.id, _AuthedReq(f"Bearer {a.api_key}", body={}), store)
    payload = json.loads(out.body)
    tasks = await store.list_cognition_tasks(a.id)
    task = next(t for t in tasks if t.id == payload["task_id"])
    assert task.payload["topic"] == "find the thing"
    assert task.payload["gap_id"] == gap.id
    assert task.payload["full_key"] == "Doc|p1|Thing|Test"
