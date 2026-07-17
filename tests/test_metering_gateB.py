"""Gate B (2026-07-16): metering the previously unmetered LLM call sites.

Every ingestion/curation/agent spender now stamps the llm_calls ledger
with an origin-tagged record_model_call: document_extraction,
sparse_keys, code_descriptions, reflection, consolidation,
meta_reflection, inline_research, self_critique, agent_llm_invoke,
shadow_eval. Pattern everywhere: prefer complete_detailed (usage-bearing),
getattr-fallback to complete() for injected fakes — unmetered but
behaviorally identical; metering never alters behavior.

Fast tests: fake clients + a fake store via the seams' store= passthrough;
cost accounting enabled by monkeypatching cost.emit.get_settings.
"""
from __future__ import annotations

import pytest

from crystal_cache.cost import emit as emit_mod


class _Result:
    """LLMResult stand-in."""

    def __init__(self, text, model="fake-model-1", it=100, ot=25):
        self.text = text
        self.model = model
        self.input_tokens = it
        self.output_tokens = ot
        self.cache_read_tokens = 0
        self.cache_creation_tokens = 0


class _DetailedClient:
    """Fake exposing complete_detailed (metered lane)."""

    def __init__(self, text):
        self._text = text
        self.calls = 0

    def complete_detailed(self, **kwargs):
        self.calls += 1
        return _Result(self._text)

    def complete(self, **kwargs):  # pragma: no cover — detailed preferred
        self.calls += 1
        return self._text


class _LegacyClient:
    """Fake exposing only complete() (unmetered fallback lane)."""

    def __init__(self, text):
        self._text = text

    def complete(self, **kwargs):
        return self._text


class _FakeStore:
    def __init__(self):
        self.rows: list[dict] = []

    async def record_llm_call(self, customer_id, **kw):
        self.rows.append({"customer_id": customer_id, **kw})
        return {"id": "llm_test", **kw}


class _Settings:
    enable_cost_accounting = True
    llm_price_table_overrides = None


@pytest.fixture
def ledger(monkeypatch):
    monkeypatch.setattr(emit_mod, "get_settings", lambda: _Settings())
    return _FakeStore()


def _origins(ledger):
    return [r["origin"] for r in ledger.rows]


# ---------------------------------------------------------------------------
# document_extraction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extraction_meters_per_window(ledger):
    from crystal_cache.ingestion.document_pipeline import DocumentPipeline

    client = _DetailedClient(
        '[{"key": "k", "segments": ["A", "B"], "value": "v", '
        '"citation": "", "type": "fact"}]'
    )
    pipeline = DocumentPipeline(store=None, encoder=None,
                                vector_store=None, client=client)
    chunks = [
        {"label": "D", "locator": "S1", "text": "one"},
        {"label": "D", "locator": "S2", "text": "two " * 2000},
    ]
    items = await pipeline.extract_items(
        text="", content_chunks=chunks, detected_type="general",
        customer_id="cust-m1", store=ledger,
    )
    assert items
    assert _origins(ledger).count("document_extraction") == client.calls
    assert all(r["customer_id"] == "cust-m1" for r in ledger.rows)


@pytest.mark.asyncio
async def test_extraction_legacy_client_unmetered_but_identical(ledger):
    from crystal_cache.ingestion.document_pipeline import DocumentPipeline

    client = _LegacyClient(
        '[{"key": "k", "segments": ["A"], "value": "v", "type": "fact"}]'
    )
    pipeline = DocumentPipeline(store=None, encoder=None,
                                vector_store=None, client=client)
    items = await pipeline.extract_items(
        text="hello world", customer_id="cust-m2", store=ledger,
    )
    assert items and items[0].value == "v"
    assert ledger.rows == []


# ---------------------------------------------------------------------------
# sparse_keys — metering at the cache boundary
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sparse_key_meters_misses_not_hits(ledger, monkeypatch):
    from crystal_cache.encoding import sparse_keys as sk

    sk.clear_cache()
    client = _DetailedClient('["Infra", "DB", "Postgres"]')
    monkeypatch.setattr("crystal_cache.llm.get_llm_client", lambda: client)

    key1 = await sk.generate_sparse_key_metered(
        "we use postgres", customer_id="cust-sk", store=ledger,
    )
    assert key1 == "Infra|DB|Postgres"
    assert _origins(ledger) == ["sparse_keys"]

    # Same text again: cache hit — no new model call, no new row.
    key2 = await sk.generate_sparse_key_metered(
        "we use postgres", customer_id="cust-sk", store=ledger,
    )
    assert key2 == key1
    assert len(ledger.rows) == 1
    assert client.calls == 1


@pytest.mark.asyncio
async def test_sparse_key_sync_and_metered_share_one_cache(ledger, monkeypatch):
    from crystal_cache.encoding import sparse_keys as sk

    sk.clear_cache()
    client = _DetailedClient('["A", "B"]')
    monkeypatch.setattr("crystal_cache.llm.get_llm_client", lambda: client)

    # Sync path first (unmetered by design)...
    key_sync = sk.generate_sparse_key("shared text")
    assert key_sync == "A|B"
    assert ledger.rows == []
    # ...then the metered path HITS the same cache: no call, no row.
    key_async = await sk.generate_sparse_key_metered(
        "shared text", customer_id="cust-sk2", store=ledger,
    )
    assert key_async == key_sync
    assert client.calls == 1
    assert ledger.rows == []


# ---------------------------------------------------------------------------
# code_descriptions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_code_descriptions_meter(ledger):
    from crystal_cache.ingestion.code_describer import describe_code_file

    client = _DetailedClient('{"file_summary": "does x", "by_index": {"0": "adds"}}')
    chunks = [{"index": 0, "text": "def f(): pass", "locator": "m.py::f"}]
    out = await describe_code_file(
        file_text="def f(): pass", chunks=chunks, client=client,
        file_label="m.py", customer_id="cust-cd", store=ledger,
    )
    assert out["file_summary"] == "does x"
    assert "code_descriptions" in _origins(ledger)


# ---------------------------------------------------------------------------
# reflection (crystallizer) — tuple contract + fallback
# ---------------------------------------------------------------------------

def test_reflect_returns_usage_tuple():
    from crystal_cache.learning.crystallizer import _reflect_on_failure

    rule, usage = _reflect_on_failure(
        question_text="q", wrong_answer="w",
        failed_reasoning="trace of failure",
        client=_DetailedClient("Always check the sign."),
    )
    assert rule == "Always check the sign."
    assert usage is not None and usage.model == "fake-model-1"

    rule2, usage2 = _reflect_on_failure(
        question_text="q", wrong_answer="w",
        failed_reasoning="trace of failure",
        client=_LegacyClient("Rule text."),
    )
    assert rule2 == "Rule text." and usage2 is None


# ---------------------------------------------------------------------------
# consolidation + meta_reflection — tuple contracts
# ---------------------------------------------------------------------------

def test_consolidation_llm_returns_usage(monkeypatch):
    from crystal_cache.maintenance.consolidation_service import (
        ConsolidationService,
    )

    client = _DetailedClient('{"rules": []}')
    monkeypatch.setattr(
        "crystal_cache.maintenance.consolidation_service.get_llm_client",
        lambda: client,
    )
    svc = ConsolidationService(store=None)
    parsed, usage = svc._consolidate_llm(["rule a"], [])
    assert parsed == {"rules": []}
    assert usage is not None

    parsed2, usages = svc._run_meta_reflection_llm(
        [{"reflection": "r1"}],
    )
    assert parsed2 == {"rules": []}
    assert len(usages) == 1


# ---------------------------------------------------------------------------
# self_critique
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_self_critique_meters(ledger):
    from crystal_cache.agent.mcr_emitter import run_self_critique

    obs, items, summary = await run_self_critique(
        anthropic_client=_DetailedClient(
            '{"observations": [], "action_items": []}'
        ),
        user_query="q", agent_final_text="a",
        tool_calls_log=[], crystals_used=[],
        customer_id="cust-sc", store=ledger,
    )
    assert _origins(ledger) == ["self_critique"]

    # Legacy str-only client: no row, still functional.
    await run_self_critique(
        anthropic_client=_LegacyClient(
            '{"observations": [], "action_items": []}'
        ),
        user_query="q", agent_final_text="a",
        tool_calls_log=[], crystals_used=[],
        customer_id="cust-sc", store=ledger,
    )
    assert len(ledger.rows) == 1


# ---------------------------------------------------------------------------
# shadow_eval — upstream lane with billing semantics
# ---------------------------------------------------------------------------

class _UpstreamResp:
    prompt_tokens = 500
    completion_tokens = 80


class _UpstreamClient:
    async def complete(self, **kwargs):
        return _UpstreamResp()


class _Cust:
    def __init__(self, cid, mode):
        self.id = cid
        self.inference_mode = mode


@pytest.mark.asyncio
async def test_shadow_eval_meters_with_billing(ledger):
    from crystal_cache.execution.shadow_evaluator import ShadowEvaluator

    ev = ShadowEvaluator()
    resp = await ev.run_shadow(
        client=_UpstreamClient(), original_messages=[], model="m-up",
        customer=_Cust("cust-sh", "managed"), store=ledger,
    )
    assert resp is not None
    row = ledger.rows[0]
    assert row["origin"] == "shadow_eval"
    assert row["billing"] == "managed"
    assert row["input_tokens"] == 500

    await ev.run_shadow(
        client=_UpstreamClient(), original_messages=[], model="m-up",
        customer=_Cust("cust-sh2", "byok"), store=ledger,
    )
    assert ledger.rows[1]["billing"] is None


# ---------------------------------------------------------------------------
# inline_research + agent_llm_invoke — source pins (heavy runtime deps)
# ---------------------------------------------------------------------------

def test_remaining_origins_source_pins():
    import inspect
    from crystal_cache.retrieval import v3_signal_handler as sig
    from crystal_cache.agent.tools import llm as agent_llm

    s1 = inspect.getsource(sig)
    assert '"inline_research"' in s1 and "complete_detailed" in s1
    s2 = inspect.getsource(agent_llm)
    assert '"agent_llm_invoke"' in s2
    assert 'getattr(customer, "inference_mode", "byok")' in s2


# ---------------------------------------------------------------------------
# Import-order hygiene (v34 deploy incident, 2026-07-16)
# ---------------------------------------------------------------------------

def test_alembic_order_import_is_cycle_free():
    """Fresh interpreter, alembic env.py's exact first import. Pytest's own
    import order masks early-chain cycles (cost.emit was fully loaded before
    sparse_keys in-suite), so only a clean subprocess reproduces the boot
    path that crashed v34: schema -> infrastructure/__init__ ->
    metadata_store -> encoding -> sparse_keys -> cost.emit -> metadata_store
    (partial)."""
    import os
    import subprocess
    import sys

    r = subprocess.run(
        [sys.executable, "-c",
         "from crystal_cache.infrastructure.schema import Base"],
        capture_output=True, text=True, env=dict(os.environ),
    )
    assert r.returncode == 0, f"circular import at boot:\n{r.stderr}"
