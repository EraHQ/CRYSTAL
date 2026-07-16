"""Q1A fail-fast (ratified 2026-07-15).

A FAILED step whose un-executed transitive dependents reach the
deliverable-producing sink dooms the attempt: composition steps would
honestly report the holes and the validator would rightly reject — a
verdict already known before paying for it. The engine fails the
attempt fast instead. Pins:

  - `_fail_fast_step` detects the broken deliverable chain and stays
    quiet for the two deliberate exclusions (a failed SINK, and a
    failed step nothing depends on);
  - on trigger, composition and the VALIDATOR are skipped entirely —
    the fake validator raises if called;
  - the synthetic rejection rides the SAME rails as a real verdict:
    rejection_log carries the FAIL-FAST reasoning, the findings
    harvest still carries completed retrieval work into the next
    attempt, and the retry's orchestrator sees both;
  - the final NEEDS_REVIEW reason names the failed step and its real
    error (the empty-str(TimeoutError()) disease is the belt fix's
    job, but the rail must relay whatever error text exists).

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

from typing import Any, Optional

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


# ---------------------------------------------------------------------------
# Unit — _fail_fast_step truth table
# ---------------------------------------------------------------------------

def _out(step_id: int, status: StepStatus, error: str = "") -> StepOutput:
    o = StepOutput(step_id=step_id, action="x", status=status)
    o.error = error
    return o


def test_fail_fast_detects_broken_deliverable_chain():
    plan = Plan(steps=[
        PlanStep(id=1, action=StepAction.RESEARCH, description="r"),
        PlanStep(id=2, action=StepAction.SYNTHESIZE, description="s",
                 depends_on=[1]),
        PlanStep(id=3, action=StepAction.FORMAT, description="f",
                 depends_on=[2]),
    ])
    env = CognitionEnvironment(customer_id="c")
    env.step_outputs[1] = _out(1, StepStatus.FAILED, "belt timeout")
    step, affected = engine_mod._fail_fast_step(
        plan, {1}, [plan.steps[1], plan.steps[2]], env)
    assert step is plan.steps[0]
    assert affected == {2, 3}


def test_failed_sink_is_not_fail_fast():
    """Format itself failing has no pending dependents — the existing
    deliverable-salvage + validator path owns that case."""
    plan = Plan(steps=[
        PlanStep(id=1, action=StepAction.WEB_SEARCH, description="w"),
        PlanStep(id=2, action=StepAction.FORMAT, description="f",
                 depends_on=[1]),
    ])
    env = CognitionEnvironment(customer_id="c")
    env.step_outputs[1] = _out(1, StepStatus.COMPLETE)
    env.step_outputs[2] = _out(2, StepStatus.FAILED, "empty output")
    step, affected = engine_mod._fail_fast_step(plan, {1, 2}, [], env)
    assert step is None
    assert affected == set()


def test_failed_optional_branch_is_not_fail_fast():
    """A failed step nothing depends on: its findings were optional."""
    plan = Plan(steps=[
        PlanStep(id=1, action=StepAction.WEB_SEARCH, description="w"),
        PlanStep(id=2, action=StepAction.WEB_SEARCH, description="extra"),
        PlanStep(id=3, action=StepAction.FORMAT, description="f",
                 depends_on=[1]),
    ])
    env = CognitionEnvironment(customer_id="c")
    env.step_outputs[1] = _out(1, StepStatus.COMPLETE)
    env.step_outputs[2] = _out(2, StepStatus.FAILED, "404")
    step, affected = engine_mod._fail_fast_step(
        plan, {1, 2}, [plan.steps[2]], env)
    assert step is None
    assert affected == set()


def test_no_failures_is_not_fail_fast():
    plan = Plan(steps=[
        PlanStep(id=1, action=StepAction.WEB_SEARCH, description="w"),
        PlanStep(id=2, action=StepAction.FORMAT, description="f",
                 depends_on=[1]),
    ])
    env = CognitionEnvironment(customer_id="c")
    env.step_outputs[1] = _out(1, StepStatus.COMPLETE)
    step, affected = engine_mod._fail_fast_step(
        plan, {1}, [plan.steps[1]], env)
    assert step is None


# ---------------------------------------------------------------------------
# Engine integration — the synthetic rejection rides the real rails
# ---------------------------------------------------------------------------

_BELT_ERROR = ("agentic session cancelled at the 540s belt "
               "(soft deadline 360s did not compose)")


def _goal() -> GoalDocument:
    return GoalDocument(title="t", description="d",
                        acceptance_criteria=["c"])


def _plan_with_research() -> Plan:
    """web_search(1) completes, research(2) dies, format(3) depends on
    both — the deliverable chain is broken at step 2."""
    return Plan(steps=[
        PlanStep(id=1, action=StepAction.WEB_SEARCH, description="w"),
        PlanStep(id=2, action=StepAction.RESEARCH, description="r"),
        PlanStep(id=3, action=StepAction.FORMAT, description="f",
                 depends_on=[1, 2]),
    ], reasoning="scripted")


def _plan_compose_only() -> Plan:
    return Plan(steps=[
        PlanStep(id=1, action=StepAction.WEB_SEARCH, description="w"),
        PlanStep(id=2, action=StepAction.FORMAT, description="f",
                 depends_on=[1]),
    ], retry_route="compose_only", reasoning="scripted")


def _scripted_engine(monkeypatch, orchestrator_outputs, validator):
    """Mirror of test_cognition_revision's harness, with a worker that
    fails RESEARCH steps the way a dead agentic session does and a
    validator the caller scripts (or booby-traps)."""
    observed: list[dict[str, Any]] = []
    format_calls: list[int] = []

    async def fake_orchestrator(*, env, store, fact_store, encoder=None):
        observed.append({
            "attempt": env.attempts,
            "carried": [dict(f) for f in env.carried_findings],
            "rejections": [dict(r) for r in env.rejection_log],
            "prior_deliverable": env.prior_deliverable,
            "event_kinds": [e.get("kind") for e in env.events],
        })
        return orchestrator_outputs.pop(0)

    async def fake_worker(env, step, _store, _fact_store, _encoder):
        out = StepOutput(step_id=step.id, action=step.action.value,
                         status=StepStatus.COMPLETE)
        if step.action == StepAction.RESEARCH:
            out.status = StepStatus.FAILED
            out.error = _BELT_ERROR
            out.output = {}
        elif step.action == StepAction.WEB_SEARCH:
            out.output = {
                "content_text": "evidence",
                "findings": [{"title": "t", "url": "u",
                              "snippet": "evidence"}],
            }
        elif step.action == StepAction.FORMAT:
            format_calls.append(len(format_calls) + 1)
            out.output = {
                "content": ("final deliverable text comfortably longer "
                            "than fifty characters for the salvage gate"),
                "is_deliverable": True,
            }
        return out

    async def fake_snapshot(store, env, terminal=False):
        return None

    monkeypatch.setattr(engine_mod, "run_orchestrator", fake_orchestrator)
    monkeypatch.setattr(engine_mod, "run_worker", fake_worker)
    monkeypatch.setattr(engine_mod, "run_validator", validator)
    monkeypatch.setattr(engine_mod, "_persist_snapshot", fake_snapshot)
    return observed, format_calls


async def test_fail_fast_skips_validator_and_names_the_step(
        monkeypatch, store):
    async def booby_trapped_validator(*, env):
        raise AssertionError("validator must be skipped under fail-fast")

    observed, format_calls = _scripted_engine(
        monkeypatch,
        orchestrator_outputs=[(_goal(), _plan_with_research())],
        validator=booby_trapped_validator,
    )
    result = await run_cognition_workflow(
        "goal", "cus_t", store, None, None,
        output_type="report", max_attempts=1,
    )
    assert result.success is False
    # The final reason relays the real failure, not an empty string.
    assert "step 2 (research) failed" in result.reason
    assert "540s belt" in result.reason
    # Composition never ran; the validator never ran (it would raise).
    assert format_calls == []


async def test_fail_fast_rejection_rides_the_revision_rails(
        monkeypatch, store):
    """Attempt 1 fail-fasts; attempt 2's orchestrator sees the
    FAIL-FAST verdict in rejection_log AND the harvested findings from
    the step that DID complete, then a compose-only plan succeeds."""
    verdicts = [ValidationResult(approved=True, score=0.9,
                                 reasoning="ok")]

    async def scripted_validator(*, env):
        return verdicts.pop(0)

    observed, format_calls = _scripted_engine(
        monkeypatch,
        orchestrator_outputs=[(_goal(), _plan_with_research()),
                              (_goal(), _plan_compose_only())],
        validator=scripted_validator,
    )
    result = await run_cognition_workflow(
        "goal", "cus_t", store, None, None,
        output_type="report", max_attempts=2,
    )
    assert result.success is True
    # Attempt 2's orchestrator observed the synthetic verdict...
    rejections = observed[1]["rejections"]
    assert len(rejections) == 1
    assert rejections[0]["reasoning"].startswith("FAIL-FAST")
    # ...the fail_fast lifecycle event...
    assert "fail_fast" in observed[1]["event_kinds"]
    # ...the carryover from the COMPLETED web_search of attempt 1...
    assert observed[1]["carried"], "harvest must survive fail-fast"
    # ...and no phantom deliverable (composition was skipped).
    assert observed[1]["prior_deliverable"] == ""
    # Format ran exactly once — on attempt 2. Validator consumed its
    # single scripted verdict (also attempt 2 only).
    assert format_calls == [1]
    assert verdicts == []


async def test_failed_sink_still_goes_to_the_validator(
        monkeypatch, store):
    """A failed FORMAT (the sink) is NOT fail-fast: the classic path —
    deliverable salvage + validator — owns it unchanged."""
    validator_called = []

    async def scripted_validator(*, env):
        validator_called.append(True)
        return ValidationResult(approved=False, score=0.1,
                                reasoning="format died",
                                issues=["format died"])

    observed, format_calls = _scripted_engine(
        monkeypatch,
        orchestrator_outputs=[(_goal(), _plan_compose_only())],
        validator=scripted_validator,
    )

    # Override the worker for this test: FORMAT fails as the sink.
    async def sink_failing_worker(env, step, _store, _fact_store,
                                  _encoder):
        out = StepOutput(step_id=step.id, action=step.action.value,
                         status=StepStatus.COMPLETE)
        if step.action == StepAction.WEB_SEARCH:
            out.output = {
                "content_text": "evidence",
                "findings": [{"title": "t", "url": "u",
                              "snippet": "evidence"}],
            }
        else:
            out.status = StepStatus.FAILED
            out.error = "empty output"
            out.output = {}
        return out

    monkeypatch.setattr(engine_mod, "run_worker", sink_failing_worker)

    result = await run_cognition_workflow(
        "goal", "cus_t", store, None, None,
        output_type="report", max_attempts=1,
    )
    assert result.success is False
    assert validator_called == [True]
    assert "FAIL-FAST" not in result.reason
    assert "format died" in result.reason
