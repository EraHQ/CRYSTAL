"""S6 — structural critics + the shadow dollar cap (2026-07-08).

The substrate channel reaches every part of the system that affects
outcomes (Anthony, 2026-07-08). This suite covers the first structural
writer — the blob-fact ingestion detector — and the spend_budgets gate
on the shadow critic (MCR §11 Q10's hard cap, closed by S4's substrate).
"""
from __future__ import annotations

from crystal_cache.infrastructure.schema import CrystalRow, FactRow
from crystal_cache.metacognition.structural import (
    run_structural_ingestion_scan,
)


async def _seed_crystal(store, customer_id, cid, claim, n_facts=1):
    async with store.session() as s:
        s.add(CrystalRow(
            id=cid, customer_id=customer_id,
            crystal_type="customer:legacy", summary_vector=[],
            quality_tier="neutral",
        ))
    for i in range(n_facts):
        async with store.session() as s:
            s.add(FactRow(
                id=f"{cid}_f{i}", crystal_id=cid,
                pair_type="entity_attribute",
                prompt_text=f"Company|Era HQ|Overview{i}|Test",
                claim_text=claim,
                source_kind="model_reasoning", vector=[],
            ))


async def test_blob_fact_files_a_structural_critique(store, customer):
    """The erahq incident, correctly channeled: a single-fact crystal
    with a jumbo claim becomes a substrate_observation blaming
    INGESTION — not a knowledge gap addressed to the human."""
    await _seed_crystal(store, customer.id, "crys_blob", "x" * 1200)
    await _seed_crystal(store, customer.id, "crys_ok", "Raleigh, NC")
    await _seed_crystal(  # multi-fact crystals are extraction-shaped: fine
        store, customer.id, "crys_multi", "y" * 1200, n_facts=3)

    out = await run_structural_ingestion_scan(store=store)
    assert out["found"] == 1 and out["filed"] == 1

    items = await store.list_substrate_action_items(customer_id=customer.id)
    assert len(items) == 1
    content = items[0].content
    assert content["subsystem"] == "ingestion"
    assert content["crystal_id"] == "crys_blob"
    assert content["claim_chars"] == 1200
    critique = await store.get_critique(items[0].critique_id)
    assert critique.critic_role == "structural"
    assert critique.critic_model == "store-signal"

    # No gap rows were created — the judgment lives in the critique
    # channel now (S1 stands).
    assert await store.count_knowledge_gaps(customer.id, status="open") == 0


async def test_structural_scan_is_idempotent(store, customer):
    await _seed_crystal(store, customer.id, "crys_blob2", "z" * 1000)

    first = await run_structural_ingestion_scan(store=store)
    second = await run_structural_ingestion_scan(store=store)

    assert first["filed"] == 1
    assert second["filed"] == 0 and second["skipped_existing"] == 1
    items = await store.list_substrate_action_items(customer_id=customer.id)
    assert len(items) == 1


async def test_shadow_budget_row_gates_by_dollars(store, customer):
    """No row = allowed (live behavior preserved). A zero-cap row = the
    tenant turned the shadow off. A funded row meters the
    origin='shadow_critic' ledger."""
    from crystal_cache.control.admission import function_budget_allows

    # No row: the worker's gate treats budget=None as allowed.
    assert await store.get_spend_budget(
        customer.id, function="shadow_critic") is None

    # Zero-cap row: off.
    await store.upsert_spend_budget(
        customer.id, function="shadow_critic", cap_micro_usd=0)
    assert await function_budget_allows(
        store, customer, "shadow_critic", origin="shadow_critic") is False

    # Funded row: on until the shadow-stamped ledger reaches the cap.
    await store.upsert_spend_budget(
        customer.id, function="shadow_critic", cap_micro_usd=50)
    assert await function_budget_allows(
        store, customer, "shadow_critic", origin="shadow_critic") is True
    rec = await store.record_llm_call(
        customer.id, model="claude-opus-4-8",
        input_tokens=0, output_tokens=0, origin="shadow_critic",
        price_table={},
    )
    from crystal_cache.infrastructure.schema import LlmCallRow
    async with store.session() as session:
        row = await session.get(LlmCallRow, rec["id"])
        row.computed_cost_micro_usd = 50
    assert await function_budget_allows(
        store, customer, "shadow_critic", origin="shadow_critic") is False


# --- C1 + C2 (2026-07-08): critiques visibility/dismiss + agent query logs ------

def test_critiques_are_platform_admin_only():
    """C1: the substrate endpoints left the tenant allowlist — System
    Critiques are a super-admin surface."""
    from crystal_cache.ingress.auth import _tenant_readable

    assert not _tenant_readable(
        "GET", "/admin/api/metacognition/substrate-observations")
    assert not _tenant_readable(
        "GET", "/admin/api/metacognition/substrate-observations/grouped")
    assert not _tenant_readable(
        "POST", "/admin/api/metacognition/substrate-observations/x/dismiss")


async def test_dismiss_hides_but_keeps_the_row(store, customer):
    """C1: dismiss = status 'dropped' — vanishes from the review surface,
    row survives in the table."""
    from crystal_cache.metacognition.structural import (
        run_structural_ingestion_scan,
    )
    await _seed_crystal(store, customer.id, "crys_blob3", "w" * 1000)
    await run_structural_ingestion_scan(store=store)
    items = await store.list_substrate_action_items(customer_id=customer.id)
    assert len(items) == 1

    updated = await store.update_action_item_status(items[0].id, "dropped")
    assert updated is not None and updated.status == "dropped"
    # Gone from the surface…
    assert await store.list_substrate_action_items(
        customer_id=customer.id) == []
    # …but the row survives.
    refetched = await store.list_action_items_for_critique(items[0].critique_id)
    assert len(refetched) == 1 and refetched[0].status == "dropped"


async def test_agent_turn_writes_a_query_log(store, customer):
    """C2: the agent surface logs its turns — the Logs tab was
    proxy-only. Grounded stats map to match_type; the agent's method is
    'agent_tools'."""
    from crystal_cache.agent.turn_finalize import finalize_agent_turn

    result = {
        "final_text": "Era HQ is an applied AI lab in Raleigh.",
        "tool_calls": [],
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "iterations": 1,
        "model": "m",
    }
    await finalize_agent_turn(
        store=store, encoder=object(), customer=customer,
        result=result, user_query="what is era hq",
        sequence_id="seq_test", skip_self_critique=True,
    )
    total, logs = await store.list_query_logs_for_customer(
        customer.id, limit=5)
    assert total == 1 and len(logs) == 1
    log = logs[0]
    assert log.query_text == "what is era hq"
    assert log.injection_method == "agent_tools"
    assert log.match_type == "none"  # no retrieval tools in this run
    assert log.response_text.startswith("Era HQ")
    assert log.prompt_tokens == 100 and log.completion_tokens == 20


# --- S7 (2026-07-08): chat history & resume ------------------------------------

async def test_chat_sessions_group_and_transcribe(store, customer):
    """Sessions group by sequence_id over agent_tools logs; the title is
    the first query; the transcript is ordered; foreign customers see
    nothing."""
    from crystal_cache.agent.turn_finalize import finalize_agent_turn

    async def _turn(seq, q, a):
        await finalize_agent_turn(
            store=store, encoder=object(), customer=customer,
            result={"final_text": a, "tool_calls": [],
                    "prompt_tokens": 1, "completion_tokens": 1,
                    "iterations": 1, "model": "m"},
            user_query=q, sequence_id=seq, skip_self_critique=True,
        )

    await _turn("seq_a", "first question", "first answer")
    await _turn("seq_a", "second question", "second answer")
    await _turn("seq_b", "other chat", "other answer")

    sessions = await store.list_chat_sessions(customer.id)
    assert len(sessions) == 2
    by_id = {s["sequence_id"]: s for s in sessions}
    assert by_id["seq_a"]["title"] == "first question"
    assert by_id["seq_a"]["turn_count"] == 2
    assert by_id["seq_b"]["turn_count"] == 1

    transcript = await store.get_session_transcript(customer.id, "seq_a")
    assert [t["user"] for t in transcript] == [
        "first question", "second question"]
    assert transcript[1]["assistant"] == "second answer"

    # Customer scoping: a different customer sees nothing.
    other = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="enc:v1:x")
    assert await store.list_chat_sessions(other.id) == []
    assert await store.get_session_transcript(other.id, "seq_a") == []


def test_chat_history_is_tenant_readable():
    """S7: the history endpoints are pinned tenant reads (GET only)."""
    from crystal_cache.ingress.auth import _tenant_readable

    assert _tenant_readable("GET", "/admin/api/chat/sessions")
    assert _tenant_readable("GET", "/admin/api/chat/sessions/seq_123")
    assert not _tenant_readable("POST", "/admin/api/chat/sessions")
