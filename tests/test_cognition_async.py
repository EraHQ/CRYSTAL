"""Async cognition (2026-07-13, ratified Q3A).

The agent's cognition_run tool ENQUEUES a cognition task and returns
the task id immediately — the synchronous shape died at Cloud Run's
request timeout (Inspector chat `504: null`) while the run survived
server-side. This proves:
  - cognition_run creates an agent_research task (priority urgent,
    payload carries the former kwargs) and returns task_id/started;
  - cognition_status enforces tenancy at the agent boundary, and maps
    pending/running -> in_progress, failed -> error, complete -> the
    INTACT deliverable text;
  - claim_pending_cognition_task serves urgent tasks before older
    background ones;
  - the worker branch honors agent_research: output_type/report,
    trigger_type='agent', max_attempts from payload, and stores the
    deliverable uncut past the 2000-char research cap.

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from crystal_cache.agent.tools import cognition as cog_tools


# --- cognition_run enqueues -------------------------------------------------

class _ReadyLLM:
    def is_ready(self):
        return True


class _EnqueueStore:
    def __init__(self):
        self.calls = []

    async def create_cognition_task(self, customer_id, *, task_type,
                                    payload, priority="background",
                                    source_query_id=None):
        self.calls.append({
            "customer_id": customer_id, "task_type": task_type,
            "payload": payload, "priority": priority,
        })
        return SimpleNamespace(id="cog_test123")


async def test_cognition_run_enqueues_and_returns_task_id(monkeypatch):
    from crystal_cache.llm import reset_llm_client, set_llm_client
    store = _EnqueueStore()
    monkeypatch.setattr(
        cog_tools, "_get_state",
        lambda: {"store": store, "fact_vector_store": None, "encoder": None},
    )
    set_llm_client(_ReadyLLM())
    try:
        out = await cog_tools.cognition_run(
            customer_id="cust1",
            goal="research the video editing landscape",
            conversation_context="ctx",
            output_type="report",
            max_attempts=2,
        )
    finally:
        reset_llm_client()
    assert out == {"success": True, "task_id": "cog_test123",
                   "status": "started", "reason": None}
    call = store.calls[0]
    assert call["task_type"] == "agent_research"
    assert call["priority"] == "urgent"
    assert call["payload"]["topic"] == "research the video editing landscape"
    assert call["payload"]["output_type"] == "report"
    assert call["payload"]["max_attempts"] == 2


async def test_cognition_run_fails_fast_without_llm(monkeypatch):
    from crystal_cache.llm import reset_llm_client, set_llm_client

    class _NotReady:
        def is_ready(self):
            return False

    store = _EnqueueStore()
    monkeypatch.setattr(
        cog_tools, "_get_state",
        lambda: {"store": store, "fact_vector_store": None, "encoder": None},
    )
    set_llm_client(_NotReady())
    try:
        out = await cog_tools.cognition_run(customer_id="c", goal="g")
    finally:
        reset_llm_client()
    assert out["success"] is False
    assert out["task_id"] is None
    assert store.calls == []


# --- cognition_status -------------------------------------------------------

class _StatusStore:
    def __init__(self, task):
        self._task = task

    async def get_cognition_task(self, task_id):
        return self._task


def _task(**kw):
    base = dict(customer_id="cust1", status="pending", result=None,
                result_crystal_id=None, error_message=None)
    base.update(kw)
    return SimpleNamespace(**base)


async def test_status_tenancy_foreign_task_is_not_found(monkeypatch):
    monkeypatch.setattr(
        cog_tools, "_get_state",
        lambda: {"store": _StatusStore(_task(customer_id="OTHER"))},
    )
    out = await cog_tools.cognition_status(customer_id="cust1",
                                           task_id="cog_x")
    assert out["status"] == "not_found"


async def test_status_maps_lifecycle(monkeypatch):
    monkeypatch.setattr(
        cog_tools, "_get_state",
        lambda: {"store": _StatusStore(_task(status="running"))},
    )
    out = await cog_tools.cognition_status(customer_id="cust1",
                                           task_id="t")
    assert out["status"] == "in_progress"

    monkeypatch.setattr(
        cog_tools, "_get_state",
        lambda: {"store": _StatusStore(_task(
            status="failed", error_message="boom"))},
    )
    out = await cog_tools.cognition_status(customer_id="cust1",
                                           task_id="t")
    assert out["status"] == "failed"
    assert out["error"] == "boom"

    monkeypatch.setattr(
        cog_tools, "_get_state",
        lambda: {"store": _StatusStore(_task(
            status="complete",
            result={"findings": "THE REPORT", "confidence": 0.91,
                    "crystal_id": "cry_1", "action": "inferred_fact_created"},
        ))},
    )
    out = await cog_tools.cognition_status(customer_id="cust1",
                                           task_id="t")
    assert out["status"] == "complete"
    assert out["text"] == "THE REPORT"
    assert out["confidence"] == 0.91
    assert out["crystal_id"] == "cry_1"
    assert out["error"] is None


# --- urgent-first claim (real store) ----------------------------------------

async def test_claim_serves_urgent_before_older_background(store):
    cust = (await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="ref")).id
    bg = await store.create_cognition_task(
        cust, task_type="research", payload={"topic": "old background"},
    )
    urgent = await store.create_cognition_task(
        cust, task_type="agent_research", payload={"topic": "agent ask"},
        priority="urgent",
    )
    first = await store.claim_pending_cognition_task()
    assert first.id == urgent.id
    second = await store.claim_pending_cognition_task()
    assert second.id == bg.id


# --- worker branch ----------------------------------------------------------

async def test_worker_agent_research_branch(monkeypatch, store):
    from crystal_cache.llm import reset_llm_client, set_llm_client
    from crystal_cache.workers import cognition as worker_mod

    cust = (await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="ref")).id
    task = await store.create_cognition_task(
        cust, task_type="agent_research",
        payload={"topic": "goal text", "conversation_context": "cc",
                 "output_type": "report", "max_attempts": 2},
        priority="urgent",
    )

    captured = {}

    async def fake_workflow(**kw):
        captured.update(kw)
        return SimpleNamespace(
            success=True, text="X" * 10_000, crystal_id=None,
            confidence=0.9, reason=None, tokens_used=10, cost_usd=0.01,
        )

    # The worker imports run_cognition_workflow lazily inside the
    # function body; patch it at the source module.
    import crystal_cache.cognition.engine as engine_mod
    monkeypatch.setattr(engine_mod, "run_cognition_workflow", fake_workflow)

    set_llm_client(_ReadyLLM())
    try:
        n = await worker_mod._process_pending_tasks(
            store=store, fact_vector_store=None, encoder=None, max_tasks=1,
        )
    finally:
        reset_llm_client()

    assert n == 1
    assert captured["output_type"] == "report"
    assert captured["trigger_type"] == "agent"
    assert captured["max_attempts"] == 2
    assert captured["trigger_id"] == task.id

    done = await store.get_cognition_task(task.id)
    assert done.status == "complete"
    # Deliverable stored INTACT past the 2000-char research cap.
    assert len(done.result["findings"]) == 10_000
