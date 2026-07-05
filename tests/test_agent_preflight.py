"""C2 — agent retrieval pre-flight (endpoints/agent.py).

Opening-turn cache-hit short-circuit + warm-start, gated on
settings.agent_retrieval_preflight (default off) and the no-assistant-turn
(fresh) gate. retrieve_and_inject is monkeypatched — its internals are covered
in the proxy/pipeline tests; here we verify the pre-flight's gating, outcome
mapping, fail-safe, and the cache-hit result shape. asyncio_mode=auto.
"""
from __future__ import annotations

from types import SimpleNamespace

from crystal_cache.config import settings
from crystal_cache.endpoints.agent import (
    _build_cache_hit_result,
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
    }
    base.update(over)
    return SimpleNamespace(**base)


def _patch_rai(monkeypatch, outcome, counter=None) -> None:
    async def _fake(*args, **kwargs):
        if counter is not None:
            counter["n"] += 1
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
