"""C2 — agent retrieval pre-flight (endpoints/agent.py).

Opening-turn cache-hit short-circuit + warm-start, gated on
settings.agent_retrieval_preflight (default off) and the no-assistant-turn
(fresh) gate. retrieve_and_inject is monkeypatched — its internals are covered
in the proxy/pipeline tests; here we verify the pre-flight's gating, outcome
mapping, fail-safe, and the cache-hit result shape. asyncio_mode=auto.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from crystal_cache.agent.principal import (
    reset_current_operator,
    set_current_operator,
)
from crystal_cache.config import settings
from crystal_cache.endpoints.agent import (
    _build_cache_hit_result,
    _validate_crystal_type,
    agent_retrieval_preflight,
)

# Local import inside the helper resolves to this module attribute at call
# time, so patching it here intercepts the pre-flight's retrieval call.
_RAI = "crystal_cache.retrieval.pipeline.retrieve_and_inject"

_OPENING = [{"role": "user", "content": "what is the capital of France?"}]


def _outcome(**over: object) -> SimpleNamespace:
    base = {
        "cache_hit_response": None,
        "cache_hit_crystal_id": None,
        "injected_text": None,
        "match_type": "none",
        # Audit (e) stage 1.3 (Q2 Option 2, 2026-08-26): the preflight now
        # reads the routing decision's telemetry off the outcome — the fake
        # carries the real RetrievalOutcome's defaults for those fields.
        "matched_crystal_ids": [],
        "routing_top1": None,
        "routing_top2": None,
    }
    base.update(over)
    return SimpleNamespace(**base)


def _patch_rai(monkeypatch, outcome, counter=None) -> None:
    async def _fake(*args, **kwargs):
        if counter is not None:
            counter["n"] += 1
        return outcome
    monkeypatch.setattr(_RAI, _fake)


def _patch_rai_capture(monkeypatch, outcome, captured) -> None:
    """Patch that RECORDS the call shape — the assertion this file lacked.
    S1-42 hid precisely because the original patch swallowed *args/**kwargs
    without ever looking at them."""
    async def _fake(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return outcome
    monkeypatch.setattr(_RAI, _fake)


async def test_disabled_returns_none_and_does_not_retrieve(store, customer, monkeypatch):
    monkeypatch.setattr(settings, "agent_retrieval_preflight", False)
    counter = {"n": 0}
    _patch_rai(monkeypatch, _outcome(), counter)
    result = await agent_retrieval_preflight(
        messages=_OPENING, customer=customer, store=store,
        vector_index=None, encoder=None,
    )
    assert result is None
    assert counter["n"] == 0


async def test_skips_when_assistant_turn_present(store, customer, monkeypatch):
    monkeypatch.setattr(settings, "agent_retrieval_preflight", True)
    counter = {"n": 0}
    _patch_rai(monkeypatch, _outcome(cache_hit_response="x"), counter)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "what is X?"},
    ]
    result = await agent_retrieval_preflight(
        messages=messages, customer=customer, store=store,
        vector_index=None, encoder=None,
    )
    assert result is None
    assert counter["n"] == 0  # the gate skipped retrieval entirely


async def test_cache_hit_maps_to_result(store, customer, monkeypatch):
    monkeypatch.setattr(settings, "agent_retrieval_preflight", True)
    _patch_rai(monkeypatch, _outcome(
        cache_hit_response="Paris.", cache_hit_crystal_id="cry_42",
    ))
    result = await agent_retrieval_preflight(
        messages=_OPENING, customer=customer, store=store,
        vector_index=None, encoder=None,
    )
    assert result is not None
    assert result.cache_hit_text == "Paris."
    assert result.cache_hit_crystal_id == "cry_42"
    assert result.warm_start_context is None


async def test_warm_start_wraps_injected_text(store, customer, monkeypatch):
    monkeypatch.setattr(settings, "agent_retrieval_preflight", True)
    _patch_rai(monkeypatch, _outcome(
        injected_text="Capital of France: Paris.", match_type="high",
    ))
    result = await agent_retrieval_preflight(
        messages=_OPENING, customer=customer, store=store,
        vector_index=None, encoder=None,
    )
    assert result is not None
    assert result.cache_hit_text is None
    assert "Retrieved context" in result.warm_start_context
    assert "Capital of France: Paris." in result.warm_start_context


async def test_no_match_returns_empty_result(store, customer, monkeypatch):
    monkeypatch.setattr(settings, "agent_retrieval_preflight", True)
    # Ran, but nothing matched: no cache hit, no injection.
    _patch_rai(monkeypatch, _outcome(injected_text=None, match_type="none"))
    result = await agent_retrieval_preflight(
        messages=_OPENING, customer=customer, store=store,
        vector_index=None, encoder=None,
    )
    assert result is not None
    assert result.cache_hit_text is None
    assert result.warm_start_context is None


async def test_failsafe_on_retrieval_error(store, customer, monkeypatch):
    monkeypatch.setattr(settings, "agent_retrieval_preflight", True)

    async def _boom(*args, **kwargs):
        raise RuntimeError("router exploded")
    monkeypatch.setattr(_RAI, _boom)

    result = await agent_retrieval_preflight(
        messages=_OPENING, customer=customer, store=store,
        vector_index=None, encoder=None,
    )
    assert result is None  # swallowed; caller proceeds with the normal loop


# --- the S1-42 kwargs pin (Phase 1.4 gate 3) -------------------------------

async def test_preflight_calls_pipeline_with_kwargs_and_operator(
    store, customer, monkeypatch,
):
    """The pipeline call is keyword-only and carries the acting operator
    from the request context (Q1=A). Five positional args once landed
    `operator=None` silently with the flag ON by default (S1-42); an empty
    positional tuple here is the pin that keeps that shape dead."""
    monkeypatch.setattr(settings, "agent_retrieval_preflight", True)
    captured: dict = {}
    _patch_rai_capture(monkeypatch, _outcome(), captured)

    op = SimpleNamespace(id="op_pf", role="operator", team_id=customer.id)
    token = set_current_operator(op)
    try:
        await agent_retrieval_preflight(
            messages=_OPENING, customer=customer, store=store,
            vector_index="VIDX", encoder="ENC",
        )
    finally:
        reset_current_operator(token)

    assert captured["args"] == ()  # no positional args — the S1-42 shape is dead
    kw = captured["kwargs"]
    assert kw["customer"] is customer
    assert kw["messages"] == _OPENING
    assert kw["store"] is store
    assert kw["vector_index"] == "VIDX"
    assert kw["encoder"] == "ENC"
    assert kw["operator"] is op
    # Stage 1.8 pins: the ratified defaults reach the pipeline explicitly.
    assert kw["crystal_type"] == "customer:legacy"  # Q1=A default
    assert kw["cite"] is False  # Q2=A — no agent-side renderer; native surface


async def test_preflight_threads_explicit_crystal_type(
    store, customer, monkeypatch,
):
    """Stage 1.8 (Q1=A): an explicit crystal_type reaches the pipeline
    verbatim — the c10 port's whole point (type-scoped turns)."""
    monkeypatch.setattr(settings, "agent_retrieval_preflight", True)
    captured: dict = {}
    _patch_rai_capture(monkeypatch, _outcome(), captured)
    await agent_retrieval_preflight(
        messages=_OPENING, customer=customer, store=store,
        vector_index=None, encoder=None,
        crystal_type="customer:medical",
    )
    assert captured["kwargs"]["crystal_type"] == "customer:medical"
    assert captured["kwargs"]["cite"] is False


async def test_preflight_operator_none_on_system_lane(
    store, customer, monkeypatch,
):
    """No request context (the keyless admin wrapper, tests, workers) →
    operator=None flows to the pipeline: the deliberately-unfiltered lane
    (Q2=A), not an error."""
    monkeypatch.setattr(settings, "agent_retrieval_preflight", True)
    captured: dict = {}
    _patch_rai_capture(monkeypatch, _outcome(), captured)
    await agent_retrieval_preflight(
        messages=_OPENING, customer=customer, store=store,
        vector_index=None, encoder=None,
    )
    assert captured["args"] == ()
    assert captured["kwargs"]["operator"] is None


# --- stage 1.8: crystal_type validation (Q1=A / Q3=A) ----------------------

async def test_validate_crystal_type_none_defaults_without_store_read(
    store, monkeypatch,
):
    """None/empty → the pipeline default, and get_crystal_type is NOT
    called — the fast path costs nothing."""
    async def _boom(*a, **k):  # noqa: ANN002, ANN003
        raise AssertionError("store must not be read for the default")
    monkeypatch.setattr(store, "get_crystal_type", _boom)
    assert await _validate_crystal_type(store, None) == "customer:legacy"
    assert await _validate_crystal_type(store, "") == "customer:legacy"


async def test_validate_crystal_type_unknown_is_honest_400(store):
    """Unknown type → 400 BEFORE the expensive loop. The message names
    the offending type and no admin route — the S4-108 non-reproduction
    pin (the proxy's 400 pointed at PUT /admin/api/crystal_types/<id>,
    a route that never existed)."""
    with pytest.raises(HTTPException) as exc:
        await _validate_crystal_type(store, "customer:typo")
    assert exc.value.status_code == 400
    detail = str(exc.value.detail)
    assert "customer:typo" in detail
    assert "/admin" not in detail
    assert "PUT" not in detail


async def test_validate_crystal_type_known_resolves(store, customer):
    """A type that exists resolves to itself."""
    from crystal_cache.models.crystal_type import CrystalType

    await store.upsert_crystal_type(CrystalType(
        id="customer:medical", display_name="Medical records",
        scope="customer",
    ))
    assert await _validate_crystal_type(store, "customer:medical") \
        == "customer:medical"


# --- the short-circuit result shape ---------------------------------------

def test_build_cache_hit_result_shape():
    msgs = [{"role": "user", "content": "what is the capital of France?"}]
    out = _build_cache_hit_result(
        messages=msgs,
        model="claude-sonnet-4-5-20250929",
        cache_hit_text="Paris.",
    )
    assert out["final_text"] == "Paris."
    assert out["stop_reason"] == "cache_hit"
    assert out["iterations"] == 0
    assert out["prompt_tokens"] == 0
    assert out["completion_tokens"] == 0
    assert out["cache_creation_tokens"] == 0
    assert out["cache_read_tokens"] == 0
    assert out["tool_calls"] == []
    assert out["model"] == "claude-sonnet-4-5-20250929"
    # Trajectory carries the synthetic assistant turn with the cached answer.
    assert out["messages"][-1] == {"role": "assistant", "content": "Paris."}
    assert out["messages"][0] == msgs[0]
    # Original messages list not mutated.
    assert len(msgs) == 1
