"""Cognition cycles (ratified 2026-07-16: Q1B/Q2B/Q3A).

needs_human_review was a backstop functioning as the terminus. Now a
rejected-exhausted run requeues its OWN trigger for a fresh cycle: the
next run's orchestrator sees the prior runs' verdicts (the only thing
that crosses the cycle boundary as trusted) plus the previous cycle's
findings as UNVERIFIED HINTS (Q1B). give_up is honored on a later
cycle's first attempt (Q2B); the cap bounds total runs per trigger and
gates AUTO-recycle only. Both lanes participate (Q3A): tasks via
worker requeue, gaps via the sweep's natural retry + a durable
cycles_exhausted park.

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from crystal_cache.cognition import engine as engine_mod
from crystal_cache.cognition.engine import run_cognition_workflow
from crystal_cache.cognition.models import (
    CognitionEnvironment,
    GoalDocument,
    Plan,
    PlanStep,
    StepAction,
    StepOutput,
    StepStatus,
    ValidationResult,
)


class _ReadyLLM:
    def is_ready(self):
        return True


# ---------------------------------------------------------------------------
# Store — the cross-run verdict read and the requeue flip
# ---------------------------------------------------------------------------

def _detail(reasoning: str, findings=None, cycle: int = 1) -> dict:
    return {
        "trigger_id": "trig-1",
        "cycle": cycle,
        "attempts": 3,
        "validation": {
            "score": 0.4,
            "reasoning": reasoning,
            "issues": ["missing changelog data"],
            "suggestions": ["narrower steps"],
        },
        "carried_findings": findings or [],
    }


async def test_run_verdicts_for_trigger_reads_terminal_runs(store):
    cust = (await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="ref")).id

    await store.upsert_cognition_run(
        "run-old", cust, status="needs_review", trigger_type="research",
        trigger_id="trig-1", detail=_detail("older verdict"),
        terminal=True,
    )
    await store.upsert_cognition_run(
        "run-new", cust, status="needs_review", trigger_type="research",
        trigger_id="trig-1",
        detail=_detail(
            "newest verdict",
            findings=[{"title": "Hint", "url": "https://x", "snippet": "s"}],
            cycle=2,
        ),
        terminal=True,
    )
    # Active run on the same trigger — excluded (it IS the new cycle).
    await store.upsert_cognition_run(
        "run-live", cust, status="working", trigger_type="research",
        trigger_id="trig-1", detail={"trigger_id": "trig-1"},
    )
    # Terminal run on a DIFFERENT trigger — excluded.
    await store.upsert_cognition_run(
        "run-other", cust, status="complete", trigger_type="research",
        trigger_id="trig-2", detail=_detail("unrelated"), terminal=True,
    )

    out = await store.list_run_verdicts_for_trigger(
        cust, trigger_id="trig-1", exclude_run_id="run-live",
    )
    assert out["run_count"] == 2
    assert [v["run_id"] for v in out["verdicts"]] == ["run-new", "run-old"]
    assert out["verdicts"][0]["reasoning"] == "newest verdict"
    assert out["verdicts"][0]["cycle"] == 2
    assert out["verdicts"][0]["issues"] == ["missing changelog data"]
    # Hints come from the NEWEST prior run only.
    assert out["hint_findings"] == [
        {"title": "Hint", "url": "https://x", "snippet": "s"},
    ]


async def test_requeue_cognition_task_flips_terminal_to_pending(store):
    cust = (await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="ref")).id
    task = await store.create_cognition_task(
        cust, task_type="agent_research", payload={"topic": "t"},
    )
    await store.mark_cognition_task_complete(
        task.id, result={"action": "no_actionable_findings"},
        completed_at=datetime.now(timezone.utc),
    )
    assert await store.requeue_cognition_task(task.id) is True
    row = await store.get_cognition_task(task.id)
    assert row.status == "pending"
    assert row.completed_at is None
    # Already pending — the flip refuses (no double-requeue races).
    assert await store.requeue_cognition_task(task.id) is False
    assert await store.requeue_cognition_task("no-such-task") is False


# ---------------------------------------------------------------------------
# Engine — cycle context, give_up at the cycle boundary, outcomes
# ---------------------------------------------------------------------------

def _goal() -> GoalDocument:
    return GoalDocument(title="t", description="d",
                        acceptance_criteria=["c"])


def _compose_plan() -> Plan:
    return Plan(steps=[
        PlanStep(id=1, action=StepAction.FORMAT, description="f"),
    ], reasoning="scripted")


def _scripted(monkeypatch, orchestrator_outputs, validator):
    observed: list[dict[str, Any]] = []

    async def fake_orchestrator(*, env, store, fact_store, encoder=None):
        observed.append({
            "cycle": env.cycle,
            "cycle_cap": env.cycle_cap,
            "verdicts": [dict(v) for v in env.prior_run_verdicts],
            "hints": [dict(h) for h in env.prior_cycle_findings],
            "event_kinds": [e.get("kind") for e in env.events],
        })
        return orchestrator_outputs.pop(0)

    async def fake_worker(env, step, _store, _fact_store, _encoder):
        out = StepOutput(step_id=step.id, action=step.action.value,
                         status=StepStatus.COMPLETE)
        out.output = {
            "content": ("deliverable text comfortably longer than "
                        "fifty characters for the salvage gate"),
            "is_deliverable": True,
        }
        return out

    monkeypatch.setattr(engine_mod, "run_orchestrator", fake_orchestrator)
    monkeypatch.setattr(engine_mod, "run_worker", fake_worker)
    monkeypatch.setattr(engine_mod, "run_validator", validator)
    return observed


async def _seed_prior_run(store, cust, trigger="trig-c"):
    d = _detail("prior verdict text",
                findings=[{"title": "Old find", "url": "https://o"}])
    d["trigger_id"] = trigger
    await store.upsert_cognition_run(
        "run-prior", cust, status="needs_review",
        trigger_type="research", trigger_id=trigger, detail=d,
        terminal=True,
    )


async def test_engine_cycle_context_reaches_orchestrator(
        monkeypatch, store):
    cust = (await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="ref")).id
    await _seed_prior_run(store, cust)

    async def approve(*, env):
        return ValidationResult(approved=True, score=0.9, reasoning="ok")

    observed = _scripted(
        monkeypatch, [(_goal(), _compose_plan())], approve)
    result = await run_cognition_workflow(
        "goal", cust, store, None, None,
        output_type="report", trigger_type="research",
        trigger_id="trig-c", max_attempts=1,
    )
    assert result.success is True
    assert result.outcome == "approved"
    assert result.cycle == 2
    assert observed[0]["cycle"] == 2
    assert observed[0]["verdicts"][0]["reasoning"] == "prior verdict text"
    # Q1B: findings cross the boundary as hints, never as trust.
    assert observed[0]["hints"] == [
        {"title": "Old find", "url": "https://o"},
    ]
    assert "cycle_context" in observed[0]["event_kinds"]


async def test_give_up_honored_on_first_attempt_of_later_cycle(
        monkeypatch, store):
    """Q2B: the prior verdicts are the evidence attempt 1 otherwise
    lacks — give_up on a later cycle's first attempt parks the run
    without spending a single worker or validator call."""
    cust = (await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="ref")).id
    await _seed_prior_run(store, cust, trigger="trig-g")

    async def booby_trapped_validator(*, env):
        raise AssertionError("validator must not run after give_up")

    give_up_plan = Plan(steps=[], retry_route="give_up",
                        reasoning="verdicts show the goal is unachievable")
    observed = _scripted(
        monkeypatch, [(_goal(), give_up_plan)], booby_trapped_validator)
    result = await run_cognition_workflow(
        "goal", cust, store, None, None,
        output_type="report", trigger_type="research",
        trigger_id="trig-g", max_attempts=3,
    )
    assert result.success is False
    assert result.outcome == "gave_up"
    assert "orchestrator_gave_up" in (result.reason or "")
    assert observed[0]["cycle"] == 2


async def test_give_up_ignored_on_cycle_one_first_attempt(
        monkeypatch, store):
    """Unchanged pre-cycles behavior: attempt 1 of cycle 1 has no
    verdict to classify, so the route is not honored there."""
    cust = (await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="ref")).id

    async def approve(*, env):
        return ValidationResult(approved=True, score=0.9, reasoning="ok")

    plan = _compose_plan()
    plan.retry_route = "give_up"
    observed = _scripted(monkeypatch, [(_goal(), plan)], approve)
    result = await run_cognition_workflow(
        "goal", cust, store, None, None,
        output_type="report", trigger_type="research",
        trigger_id="trig-fresh", max_attempts=1,
    )
    assert result.success is True
    assert result.outcome == "approved"
    assert observed[0]["cycle"] == 1


async def test_rejected_exhausted_outcome(monkeypatch, store):
    cust = (await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="ref")).id

    async def reject(*, env):
        return ValidationResult(approved=False, score=0.2,
                                reasoning="not good enough",
                                issues=["not good enough"])

    _scripted(monkeypatch, [(_goal(), _compose_plan())], reject)
    result = await run_cognition_workflow(
        "goal", cust, store, None, None,
        output_type="report", trigger_type="research",
        trigger_id="trig-r", max_attempts=1,
    )
    assert result.success is False
    assert result.outcome == "rejected_exhausted"
    assert result.cycle == 1


# ---------------------------------------------------------------------------
# Worker — the recycle branch (Q3A task lane)
# ---------------------------------------------------------------------------

def _cog_result(outcome: str, cycle: int) -> SimpleNamespace:
    return SimpleNamespace(
        success=False, text=None, crystal_id=None, confidence=0.0,
        reason="Failed after 3 attempts", tokens_used=1, cost_usd=0.0,
        outcome=outcome, cycle=cycle,
    )


async def _run_worker_once(monkeypatch, store, cog_result):
    from crystal_cache.llm import reset_llm_client, set_llm_client
    from crystal_cache.workers import cognition as worker_mod
    import crystal_cache.cognition.engine as eng

    cust = (await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="ref")).id
    task = await store.create_cognition_task(
        cust, task_type="agent_research",
        payload={"topic": "goal", "output_type": "report",
                 "max_attempts": 3},
        priority="urgent",
    )

    async def fake_workflow(**kw):
        return cog_result

    monkeypatch.setattr(eng, "run_cognition_workflow", fake_workflow)
    set_llm_client(_ReadyLLM())
    try:
        n = await worker_mod._process_pending_tasks(
            store=store, fact_vector_store=None, encoder=None,
            max_tasks=1,
        )
    finally:
        reset_llm_client()
    assert n == 1
    return await store.get_cognition_task(task.id)


async def test_worker_recycles_rejected_exhausted_under_cap(
        monkeypatch, store):
    row = await _run_worker_once(
        monkeypatch, store, _cog_result("rejected_exhausted", cycle=1))
    # Same task row back to pending — same trigger, next cycle.
    assert row.status == "pending"
    assert row.completed_at is None


async def test_worker_does_not_recycle_gave_up(monkeypatch, store):
    row = await _run_worker_once(
        monkeypatch, store, _cog_result("gave_up", cycle=1))
    assert row.status == "complete"


async def test_worker_does_not_recycle_at_cap(monkeypatch, store):
    from crystal_cache.workers.cognition import settings as worker_settings
    cap = max(1, int(worker_settings.cognition_cycle_cap))
    row = await _run_worker_once(
        monkeypatch, store, _cog_result("rejected_exhausted", cycle=cap))
    assert row.status == "complete"


# ---------------------------------------------------------------------------
# Barriers and seams (source-level pins, critiques-test precedent)
# ---------------------------------------------------------------------------

async def test_cycle_context_feeds_orchestrator_only():
    import inspect
    import crystal_cache.cognition.roles as roles_mod
    src = inspect.getsource(roles_mod)
    assert "PRIOR RUN VERDICTS" in src
    assert "UNVERIFIED HINTS" in src
    # Barrier discipline: workers never see cross-run context.
    worker_src = inspect.getsource(roles_mod._assemble_prior_context)
    assert "prior_run_verdicts" not in worker_src
    assert "prior_cycle_findings" not in worker_src


async def test_engine_cycle_seam_is_wired():
    import inspect
    src = inspect.getsource(engine_mod.run_cognition_workflow)
    assert "list_run_verdicts_for_trigger" in src
    assert "cycle_context" in src


def test_requeue_path_is_tenant_writable():
    from crystal_cache.ingress.auth import _tenant_writable
    assert _tenant_writable(
        "POST", "/admin/api/cognition/tasks/task-1/requeue")
    assert not _tenant_writable(
        "DELETE", "/admin/api/cognition/tasks/task-1/requeue")
    assert not _tenant_writable(
        "POST", "/admin/api/cognition/tasks/task-1")


# ---------------------------------------------------------------------------
# Gate 2 — bench surface plumbing
# ---------------------------------------------------------------------------

async def test_count_cognition_runs_by_triggers(store):
    cust = (await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="ref")).id
    await store.upsert_cognition_run(
        "g2-r1", cust, status="needs_review", trigger_type="fill_gap",
        trigger_id="gap-a", detail={}, terminal=True,
    )
    await store.upsert_cognition_run(
        "g2-r2", cust, status="working", trigger_type="fill_gap",
        trigger_id="gap-a", detail={},
    )
    await store.upsert_cognition_run(
        "g2-r3", cust, status="complete", trigger_type="fill_gap",
        trigger_id="gap-b", detail={}, terminal=True,
    )
    out = await store.count_cognition_runs_by_triggers(
        cust, ["gap-a", "gap-b", "gap-none"],
    )
    # Terminal count excludes the live run; last_run is the NEWEST
    # run overall (the live one) for click-through.
    assert out["gap-a"]["run_count"] == 1
    assert out["gap-a"]["last_run_id"] == "g2-r2"
    assert out["gap-a"]["last_run_status"] == "working"
    assert out["gap-b"]["run_count"] == 1
    assert "gap-none" not in out
    assert await store.count_cognition_runs_by_triggers(cust, []) == {}


async def test_env_summary_carries_cycle_context():
    from crystal_cache.cognition.engine import env_summary
    env = CognitionEnvironment(customer_id="c", trigger_id="t-1")
    env.cycle = 2
    env.cycle_cap = 3
    s = env_summary(env)
    assert s["trigger_id"] == "t-1"
    assert s["cycle"] == 2
    assert s["cycle_cap"] == 3
