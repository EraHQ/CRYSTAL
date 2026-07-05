"""WS D / D.3 — the shared cost-emit helper (cost/emit.py::record_model_call).

The proxy / cognition / depth funnel through this; the agent surface keeps its
own result-dict emitter (record_agent_llm_cost, tested in test_agent_cost.py).
These cover the flag gate, usage-object token extraction (incl. Anthropic cache
fields), the explicit-token fallback, lazy store-singleton resolution, and the
fail-safe swallow. Pure — a fake store + monkeypatched settings, no DB.
"""
from __future__ import annotations

import pytest

from crystal_cache.cost import emit as emit_mod


class _Usage:
    """Stand-in for an Anthropic usage object."""
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _FakeStore:
    def __init__(self, *, raise_on_record: bool = False):
        self.calls: list[dict] = []
        self._raise = raise_on_record

    async def record_llm_call(self, customer_id, **kw):
        if self._raise:
            raise RuntimeError("boom")
        self.calls.append({"customer_id": customer_id, **kw})
        return {"id": "llm_test", **kw}


class _Settings:
    def __init__(self, *, enabled: bool, overrides=None):
        self.enable_cost_accounting = enabled
        self.llm_price_table_overrides = overrides


def _use_settings(monkeypatch, *, enabled: bool, overrides=None):
    monkeypatch.setattr(
        emit_mod, "get_settings",
        lambda: _Settings(enabled=enabled, overrides=overrides),
    )


async def test_noop_when_flag_off(monkeypatch):
    _use_settings(monkeypatch, enabled=False)
    store = _FakeStore()
    await emit_mod.record_model_call(
        store=store, customer_id="c1", model="claude-sonnet-4-6",
        usage=_Usage(input_tokens=100, output_tokens=50), origin="agent",
    )
    assert store.calls == []


async def test_records_from_usage_with_cache(monkeypatch):
    _use_settings(monkeypatch, enabled=True)
    store = _FakeStore()
    await emit_mod.record_model_call(
        store=store, customer_id="c1", model="claude-sonnet-4-6",
        usage=_Usage(
            input_tokens=1000, output_tokens=200,
            cache_creation_input_tokens=500, cache_read_input_tokens=4000,
        ),
        origin="cognition", session_id="env_1",
    )
    assert len(store.calls) == 1
    call = store.calls[0]
    assert call["customer_id"] == "c1"
    assert call["model"] == "claude-sonnet-4-6"
    assert call["input_tokens"] == 1000
    assert call["output_tokens"] == 200
    assert call["cache_creation_tokens"] == 500
    assert call["cache_read_tokens"] == 4000
    assert call["origin"] == "cognition"
    assert call["session_id"] == "env_1"
    assert "claude-sonnet-4-6" in call["price_table"]  # threaded from settings


async def test_records_from_explicit_tokens(monkeypatch):
    _use_settings(monkeypatch, enabled=True)
    store = _FakeStore()
    await emit_mod.record_model_call(
        store=store, customer_id="c1", model="m", origin="depth",
        input_tokens=10, output_tokens=20,
    )
    call = store.calls[0]
    assert call["input_tokens"] == 10
    assert call["output_tokens"] == 20
    assert call["cache_creation_tokens"] == 0
    assert call["cache_read_tokens"] == 0
    assert call["origin"] == "depth"


async def test_usage_missing_fields_default_zero(monkeypatch):
    _use_settings(monkeypatch, enabled=True)
    store = _FakeStore()
    await emit_mod.record_model_call(
        store=store, customer_id="c1", model="m", origin="agent",
        usage=_Usage(input_tokens=5),  # no output / cache fields
    )
    call = store.calls[0]
    assert call["input_tokens"] == 5
    assert call["output_tokens"] == 0
    assert call["cache_creation_tokens"] == 0
    assert call["cache_read_tokens"] == 0


async def test_fail_safe_swallows_store_error(monkeypatch):
    _use_settings(monkeypatch, enabled=True)
    store = _FakeStore(raise_on_record=True)
    # Must NOT raise — cost telemetry never breaks the call path.
    await emit_mod.record_model_call(
        store=store, customer_id="c1", model="m", origin="agent",
        usage=_Usage(input_tokens=5),
    )
    assert store.calls == []


async def test_lazy_store_singleton_when_not_passed(monkeypatch):
    _use_settings(monkeypatch, enabled=True)
    store = _FakeStore()
    monkeypatch.setattr(emit_mod, "get_metadata_store", lambda: store)
    await emit_mod.record_model_call(
        customer_id="c1", model="m", origin="agent",
        usage=_Usage(input_tokens=7),
    )
    assert len(store.calls) == 1
    assert store.calls[0]["input_tokens"] == 7


async def test_singleton_not_touched_when_flag_off(monkeypatch):
    _use_settings(monkeypatch, enabled=False)

    def _boom():
        raise AssertionError("get_metadata_store must not be called when flag off")

    monkeypatch.setattr(emit_mod, "get_metadata_store", _boom)
    # Early return on the flag means the singleton is never resolved.
    await emit_mod.record_model_call(
        customer_id="c1", model="m", origin="agent", usage=_Usage(input_tokens=1),
    )
