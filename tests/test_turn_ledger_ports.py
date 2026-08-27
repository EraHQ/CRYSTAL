"""Audit (e) Stage 1, Gate 1 — the finalize-cluster ports (S2-205, S2-208,
S2-209, must-port #6 / N1), ratified 2026-08-26: Q2=Option 2 (layered
routed-crystal), Q3=A (cache-aware finalize), Q4=A (operator from context).

Pins, in layer order:
  1. cost/emit.record_model_call now RETURNS the persisted row (S2-205
     consolidation prerequisite — finalize's contract promises the row).
  2. The agent cost row attributes the request-context operator (Q4=A) and
     None on the system lane.
  3. finalize mints ONE query-log id shared by the citations rail, the
     query-log row, and the MCR trace (S2-209 — the co_cited join lives).
  4. A cache-hit finalize (Q3=A) writes an honest row — cache_hit /
     upstream_call_made=False / match_type high / the routing columns —
     records NO cost row, and forces the critique off (P0.58).
  5. Q2 Option 2 layering: explicit caller routing wins; a tool-driven
     turn falls back to the top surfaced crystal (grounded-first) + the
     tools' fact-lane scores; nothing surfaced → honest NULLs.
  6. The preflight CARRIES its routing telemetry instead of discarding it
     (cache-hit, warm-start, and empty branches all populate the fields).

asyncio_mode=auto.
"""
from __future__ import annotations

from types import SimpleNamespace

from crystal_cache.agent.principal import (
    reset_current_operator,
    set_current_operator,
)
from crystal_cache.agent.turn_finalize import (
    finalize_agent_turn,
    record_agent_llm_cost,
)
from crystal_cache.config import settings
from crystal_cache.cost.emit import record_model_call

_MODEL = "claude-sonnet-4-5-20250929"
_GROUNDING_PATH = (
    "crystal_cache.retrieval.citation_grounding.ground_sources_against_answer"
)
_MCR_PATH = "crystal_cache.agent.turn_finalize.emit_mcr_artifacts"

_LONG = "x" * 120  # comfortably over the uncited-gap floor; content unused


def _run_result(*, tool_calls: list | None = None, **over) -> dict:
    base = {
        "id": "run_ledger_1",
        "model": _MODEL,
        "final_text": _LONG,
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "tool_calls": tool_calls or [],
    }
    base.update(over)
    return base


def _patch_mcr(monkeypatch) -> dict:
    captured: dict = {}

    async def _fake_mcr(**kwargs):
        captured.update(kwargs)
        return {"trace_id": "tr_1", "critique_id": None, "action_item_ids": []}

    monkeypatch.setattr(_MCR_PATH, _fake_mcr)
    return captured


def _patch_grounding_map(monkeypatch, grounded_scores: dict[str, float]):
    """Grounding fake: crystal_id -> grounding score; grounded when >= 0.5."""
    async def _fake(encoder, answer_text, sources, *, threshold=0.25):
        return [
            {
                "source": s,
                "claim_span": "",
                "grounding_score": grounded_scores.get(s.crystal_id, 0.0),
                "grounded": grounded_scores.get(s.crystal_id, 0.0) >= 0.5,
            }
            for s, _ in sources
        ]
    monkeypatch.setattr(_GROUNDING_PATH, _fake)


async def _latest_ql(store, customer_id):
    total, logs = await store.list_query_logs_for_customer(
        customer_id=customer_id, limit=10, offset=0,
    )
    assert total >= 1
    return logs[0]


# ---------------------------------------------------------------------------
# 1 + 2 — the cost emitter
# ---------------------------------------------------------------------------

async def test_record_model_call_returns_the_row(store, customer, monkeypatch):
    monkeypatch.setattr(settings, "enable_cost_accounting", True)
    row = await record_model_call(
        customer_id=customer.id, model=_MODEL, origin="agent",
        input_tokens=100, output_tokens=10, store=store,
    )
    assert row is not None
    assert row["computed_cost_micro_usd"] > 0


async def test_cost_row_carries_context_operator(store, customer, monkeypatch):
    monkeypatch.setattr(settings, "enable_cost_accounting", True)
    op, _ = await store.create_operator(
        team_id=customer.id, display_name="Spender",
    )
    token = set_current_operator(op)
    try:
        row = await record_agent_llm_cost(
            store=store, customer_id=customer.id,
            result=_run_result(), sequence_id="seq_op",
        )
    finally:
        reset_current_operator(token)
    assert row is not None
    assert row["operator_id"] == op.id
    assert row["session_id"] == "seq_op"

    # System lane (no context): None, exactly the pre-port attribution.
    row2 = await record_agent_llm_cost(
        store=store, customer_id=customer.id,
        result=_run_result(), sequence_id=None,
    )
    assert row2["operator_id"] is None


# ---------------------------------------------------------------------------
# 3 — one query-log id for the whole turn (S2-209)
# ---------------------------------------------------------------------------

async def test_finalize_mints_one_query_log_id(store, customer, monkeypatch):
    monkeypatch.setattr(settings, "enable_cost_accounting", False)
    monkeypatch.setattr(settings, "enable_citations", True)
    monkeypatch.setattr(settings, "enable_marketplace_metering", False)
    _patch_grounding_map(monkeypatch, {"cry_A": 0.9})
    mcr_seen = _patch_mcr(monkeypatch)

    cite_seen: dict = {}
    _orig_record = store.record_citations

    async def _cite_spy(customer_id, *, query_log_id=None, citations=None):
        cite_seen["query_log_id"] = query_log_id
        return await _orig_record(
            customer_id, query_log_id=query_log_id, citations=citations,
        )
    monkeypatch.setattr(store, "record_citations", _cite_spy)

    await finalize_agent_turn(
        store=store, encoder=object(), customer=customer,
        result=_run_result(tool_calls=[
            {"name": "knowledge_search", "output": {
                "matched_crystal_ids": ["cry_A"],
                "matched_fact_ids": ["f1"],
                "top_score": 0.8,
            }},
        ]),
        user_query="what is the ledger", sequence_id="seq_led",
    )

    ql = await _latest_ql(store, customer.id)
    assert ql.id.startswith("ql_")
    # The citations rail, the query-log row, and the MCR trace share it.
    assert cite_seen["query_log_id"] == ql.id
    assert mcr_seen["query_log_id"] == ql.id
    # FIX 4 preserved: turn_index derived from the sequence.
    assert ql.turn_index == 0


# ---------------------------------------------------------------------------
# 4 — cache-hit finalize (Q3=A)
# ---------------------------------------------------------------------------

async def test_cache_hit_finalize_writes_honest_row(store, customer, monkeypatch):
    monkeypatch.setattr(settings, "enable_cost_accounting", True)  # prove skip
    monkeypatch.setattr(settings, "enable_citations", True)
    mcr_seen = _patch_mcr(monkeypatch)

    async def _no_cost_row(*a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("cost row written for a zero-model-call turn")
    monkeypatch.setattr(store, "record_llm_call", _no_cost_row)

    hit_result = _run_result(
        tool_calls=[], prompt_tokens=0, completion_tokens=0,
        final_text="the cached answer", stop_reason="cache_hit",
    )
    await finalize_agent_turn(
        store=store, encoder=object(), customer=customer,
        result=hit_result,
        user_query="the cached question", sequence_id="seq_hit",
        cache_hit=True,
        routed_crystal_id="cry_hit",
        top1_score=0.97, top2_score=0.41,
    )

    ql = await _latest_ql(store, customer.id)
    assert ql.injection_method == "cache_hit"
    assert ql.upstream_call_made is False
    assert ql.match_type == "high"
    assert ql.routed_crystal_id == "cry_hit"
    assert abs(ql.top1_score - 0.97) < 1e-6
    assert abs(ql.top2_score - 0.41) < 1e-6
    assert ql.response_text == "the cached answer"
    # P0.58: trace yes, critique no — forced regardless of the caller.
    assert mcr_seen["skip_self_critique"] is True
    assert mcr_seen["query_log_id"] == ql.id


# ---------------------------------------------------------------------------
# 5 — Q2 Option 2 layering
# ---------------------------------------------------------------------------

async def test_tool_turn_falls_back_to_top_surfaced(store, customer, monkeypatch):
    monkeypatch.setattr(settings, "enable_cost_accounting", False)
    monkeypatch.setattr(settings, "enable_citations", True)
    monkeypatch.setattr(settings, "enable_marketplace_metering", False)
    # A surfaced first but ungrounded; B grounded — grounded wins the
    # routed column even though A's tool score is higher.
    _patch_grounding_map(monkeypatch, {"cry_A": 0.1, "cry_B": 0.9})
    _patch_mcr(monkeypatch)

    await finalize_agent_turn(
        store=store, encoder=object(), customer=customer,
        result=_run_result(tool_calls=[
            {"name": "knowledge_search", "output": {
                "matched_crystal_ids": ["cry_A"],
                "matched_fact_ids": ["fa"],
                "top_score": 0.8,
            }},
            {"name": "content_search", "output": {
                "matched_crystal_ids": ["cry_B"],
                "matched_fact_ids": ["fb"],
                "top_score": 0.6,
            }},
        ]),
        user_query="layered", sequence_id="seq_q2",
    )
    ql = await _latest_ql(store, customer.id)
    assert ql.routed_crystal_id == "cry_B"       # grounded-first
    assert abs(ql.top1_score - 0.8) < 1e-6       # fact-lane scores, desc
    assert abs(ql.top2_score - 0.6) < 1e-6
    assert ql.injection_method == "agent_tools"  # the lane disclosure
    assert sorted(ql.matched_facts) == ["fa", "fb"]  # full set, untruncated


async def test_explicit_routing_beats_fallback(store, customer, monkeypatch):
    monkeypatch.setattr(settings, "enable_cost_accounting", False)
    monkeypatch.setattr(settings, "enable_citations", True)
    monkeypatch.setattr(settings, "enable_marketplace_metering", False)
    _patch_grounding_map(monkeypatch, {"cry_tool": 0.9})
    _patch_mcr(monkeypatch)

    await finalize_agent_turn(
        store=store, encoder=object(), customer=customer,
        result=_run_result(tool_calls=[
            {"name": "knowledge_search", "output": {
                "matched_crystal_ids": ["cry_tool"],
                "matched_fact_ids": ["ft"],
                "top_score": 0.5,
            }},
        ]),
        user_query="explicit wins", sequence_id="seq_ex",
        routed_crystal_id="cry_preflight",
        top1_score=0.91, top2_score=0.33,
    )
    ql = await _latest_ql(store, customer.id)
    assert ql.routed_crystal_id == "cry_preflight"
    assert abs(ql.top1_score - 0.91) < 1e-6
    assert abs(ql.top2_score - 0.33) < 1e-6


async def test_nothing_surfaced_writes_honest_nulls(store, customer, monkeypatch):
    monkeypatch.setattr(settings, "enable_cost_accounting", False)
    monkeypatch.setattr(settings, "enable_citations", True)
    _patch_mcr(monkeypatch)

    await finalize_agent_turn(
        store=store, encoder=object(), customer=customer,
        result=_run_result(tool_calls=[]),
        user_query="no retrieval", sequence_id="seq_null",
    )
    ql = await _latest_ql(store, customer.id)
    assert ql.routed_crystal_id is None
    assert ql.top1_score is None
    assert ql.top2_score is None
    assert ql.match_type == "none"


# ---------------------------------------------------------------------------
# 6 — the preflight carries its routing telemetry
# ---------------------------------------------------------------------------

_RAI = "crystal_cache.retrieval.pipeline.retrieve_and_inject"

_USER_TURN = [{"role": "user", "content": "opening question"}]


def _outcome(**over) -> SimpleNamespace:
    base = dict(
        cache_hit_response=None,
        cache_hit_crystal_id=None,
        injected_text=None,
        match_type="none",
        matched_crystal_ids=[],
        routing_top1=None,
        routing_top2=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


async def test_preflight_cache_hit_carries_routing(store, customer, monkeypatch):
    from crystal_cache.endpoints.agent import agent_retrieval_preflight

    monkeypatch.setattr(settings, "agent_retrieval_preflight", True)

    async def _fake(**kwargs):
        return _outcome(
            cache_hit_response="cached!", cache_hit_crystal_id="cry_c",
            routing_top1=0.95, routing_top2=0.30,
        )
    monkeypatch.setattr(_RAI, _fake)

    pf = await agent_retrieval_preflight(
        messages=_USER_TURN, customer=customer, store=store,
        vector_index=None, encoder=None,
    )
    assert pf.cache_hit_text == "cached!"
    assert pf.routed_crystal_id == "cry_c"
    assert abs(pf.top1_score - 0.95) < 1e-6
    assert abs(pf.top2_score - 0.30) < 1e-6


async def test_preflight_warm_start_carries_routing(store, customer, monkeypatch):
    from crystal_cache.endpoints.agent import agent_retrieval_preflight

    monkeypatch.setattr(settings, "agent_retrieval_preflight", True)

    async def _fake(**kwargs):
        return _outcome(
            injected_text="retrieved context", match_type="high",
            matched_crystal_ids=["cry_w", "cry_x"],
            routing_top1=0.71, routing_top2=0.55,
        )
    monkeypatch.setattr(_RAI, _fake)

    pf = await agent_retrieval_preflight(
        messages=_USER_TURN, customer=customer, store=store,
        vector_index=None, encoder=None,
    )
    assert pf.warm_start_context and "retrieved context" in pf.warm_start_context
    assert pf.routed_crystal_id == "cry_w"   # routing top-1
    assert abs(pf.top1_score - 0.71) < 1e-6
    assert abs(pf.top2_score - 0.55) < 1e-6


async def test_preflight_no_match_still_carries_scores(store, customer, monkeypatch):
    """Independent nullability, same as the QueryLog columns: routing can
    score candidates without matching (below threshold)."""
    from crystal_cache.endpoints.agent import agent_retrieval_preflight

    monkeypatch.setattr(settings, "agent_retrieval_preflight", True)

    async def _fake(**kwargs):
        return _outcome(routing_top1=0.15, routing_top2=0.12)
    monkeypatch.setattr(_RAI, _fake)

    pf = await agent_retrieval_preflight(
        messages=_USER_TURN, customer=customer, store=store,
        vector_index=None, encoder=None,
    )
    assert pf.cache_hit_text is None
    assert pf.warm_start_context is None
    assert pf.routed_crystal_id is None
    assert abs(pf.top1_score - 0.15) < 1e-6


# ---------------------------------------------------------------------------
# 7 — stage 1.10: unconditional Mem0 turn write (Q1=A / Q2=A)
# ---------------------------------------------------------------------------
# Patch site is finalize's own import of add_conversation_turn; the call
# rides asyncio.to_thread, so the spy is sync, like the real client fn.

_MEM0_PATH = "crystal_cache.agent.turn_finalize.add_conversation_turn"


async def test_finalize_writes_mem0_turn(store, customer, monkeypatch):
    """Every finalize feeds the turn to Mem0 with the turn's own data —
    not only when the model calls mem0_write (retirement item 11)."""
    monkeypatch.setattr(settings, "enable_cost_accounting", False)
    monkeypatch.setattr(settings, "enable_citations", False)
    _patch_mcr(monkeypatch)

    seen: dict = {}
    def _spy(**kw):
        seen.update(kw)
    monkeypatch.setattr(_MEM0_PATH, _spy)

    await finalize_agent_turn(
        store=store, encoder=object(), customer=customer,
        result=_run_result(),
        user_query="what is the ledger", sequence_id="seq_mem0",
    )
    assert seen["query_text"] == "what is the ledger"
    assert seen["response_text"] == _LONG
    assert seen["customer_id"] == customer.id
    assert seen["sequence_id"] == "seq_mem0"


async def test_cache_hit_finalize_still_writes_mem0(store, customer, monkeypatch):
    """Q1=A rider: cache hits included — session memory records what was
    said, however it was produced."""
    monkeypatch.setattr(settings, "enable_cost_accounting", False)
    monkeypatch.setattr(settings, "enable_citations", False)
    _patch_mcr(monkeypatch)

    seen: dict = {}
    def _spy(**kw):
        seen.update(kw)
    monkeypatch.setattr(_MEM0_PATH, _spy)

    await finalize_agent_turn(
        store=store, encoder=object(), customer=customer,
        result=_run_result(
            tool_calls=[], prompt_tokens=0, completion_tokens=0,
            final_text="the cached answer", stop_reason="cache_hit",
        ),
        user_query="the cached question", sequence_id="seq_mem0_hit",
        cache_hit=True, routed_crystal_id="cry_hit",
        top1_score=0.97, top2_score=0.41,
    )
    assert seen["query_text"] == "the cached question"
    assert seen["response_text"] == "the cached answer"


async def test_mem0_explosion_does_not_break_finalize(store, customer, monkeypatch):
    """The write is fail-safe: a Mem0 explosion warns and finalize still
    returns its full result (add_conversation_turn never raises in prod;
    this pins the belt over the suspenders)."""
    monkeypatch.setattr(settings, "enable_cost_accounting", False)
    monkeypatch.setattr(settings, "enable_citations", False)
    _patch_mcr(monkeypatch)

    def _boom(**kw):
        raise RuntimeError("mem0 exploded")
    monkeypatch.setattr(_MEM0_PATH, _boom)

    out = await finalize_agent_turn(
        store=store, encoder=object(), customer=customer,
        result=_run_result(),
        user_query="q", sequence_id="seq_mem0_boom",
    )
    assert "mcr" in out and "cost" in out
