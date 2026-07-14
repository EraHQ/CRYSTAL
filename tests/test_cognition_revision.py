"""Revision-aware cognition retry (2026-07-10, ratified Q1A–Q5A).

Attempts are REVISIONS, not independent samples. Pins:
  - the rejection handler harvests findings + the deliverable BEFORE the
    retry hygiene clears them (Q1A);
  - the verdict + trimmed rejected deliverable + carried-findings
    inventory reach the retry's ORCHESTRATOR prompt, and the route
    field parses into the Plan (Q2A);
  - composition steps on a revision see the carried findings AND the
    revision block (verdict as work order), bounded (Q1A/Q3A);
  - the "replan" route drops the carryover (the anchoring hedge) and
    "give_up" short-circuits to NEEDS_REVIEW with the explanation (Q5A).

(2026-07-13: the orchestrator budget proposal + platform clamp were
DELETED — one flat _COMPOSITION_MAX_TOKENS cap for every composition
call; the tests that pinned the clamp went with the mechanism.)

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

import json
from typing import Any, Optional

import pytest

from crystal_cache.cognition.engine import (
    _harvest_findings,
    run_cognition_workflow,
)
from crystal_cache.cognition.models import (
    CognitionEnvironment,
    GoalDocument,
    OutputType,
    Plan,
    PlanStep,
    StepAction,
    StepOutput,
    StepStatus,
    ValidationResult,
    WorkflowStatus,
)
from crystal_cache.cognition import roles as roles_mod
from crystal_cache.cognition import engine as engine_mod
from crystal_cache.cognition.roles import (
    _COMPOSITION_MAX_TOKENS,
    _REVISION_DELIVERABLE_CHARS,
    _trim_head_tail,
    _worker_llm_step,
    run_orchestrator,
)
from crystal_cache.llm import reset_llm_client, set_llm_client
from crystal_cache.llm.client import LLMResult


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeLLM:
    """Captures prompts; returns scripted texts in order (last repeats)."""

    def __init__(self, texts: list[str]):
        self._texts = list(texts)
        self.prompts: list[str] = []
        self.max_tokens_seen: list[int] = []

    def complete_detailed(self, *, system, messages, max_tokens,
                          temperature=1.0, tier="small", model=None,
                          json_schema=None) -> LLMResult:
        self.prompts.append(messages[-1]["content"])
        self.max_tokens_seen.append(max_tokens)
        text = self._texts.pop(0) if len(self._texts) > 1 else self._texts[0]
        return LLMResult(text=text, model="fake", input_tokens=10,
                         output_tokens=10)

    def is_ready(self) -> bool:
        return True


def _orchestrator_json(*, route: str = "",
                       reasoning: str = "plan reasoning",
                       steps: Optional[list[dict]] = None) -> str:
    return json.dumps({
        "goal": {
            "title": "T",
            "description": "D",
            "acceptance_criteria": ["addresses the request"],
        },
        "plan": {
            "reasoning": reasoning,
            "steps": steps if steps is not None else [
                {"id": 1, "action": "analyze", "description": "revise",
                 "input": {"instruction": "revise"}, "depends_on": [],
                 "parallel_group": None},
            ],
            "expected_output": "text",
            "suggested_key": "k",
            "parent_crystal_id": "",
            "retry_route": route,
        },
    })


def _rejected_env(*, deliverable: str = "old report body",
                  findings_text: str = "web finding about FFmpeg 7.1",
                  ) -> CognitionEnvironment:
    """An env in the state the retry's orchestrator sees it: verdict in
    rejection_log, carryover populated (as the engine's rejection handler
    leaves it)."""
    env = CognitionEnvironment(customer_id="cus_t", task_goal="research X")
    env.attempts = 1
    env.rejection_log.append({
        "attempt": 1,
        "reasoning": "report is placeholder-riddled",
        "issues": ["section 2 contains [TODO] placeholders"],
        "suggestions": ["fill section 2 from the gathered findings"],
        "score": 0.3,
    })
    env.prior_deliverable = deliverable
    env.carried_findings = [{
        "attempt": 1, "step_id": 1, "action": "web_search",
        "description": "search FFmpeg releases", "text": findings_text,
    }]
    return env


# ---------------------------------------------------------------------------
# Q1A — the harvest: findings + deliverable survive the hygiene clear
# ---------------------------------------------------------------------------

def test_harvest_collects_completed_retrieval_steps_only():
    env = CognitionEnvironment(customer_id="c")
    env.attempts = 1
    env.plan = Plan(steps=[
        PlanStep(id=1, action=StepAction.WEB_SEARCH, description="web"),
        PlanStep(id=2, action=StepAction.CRYSTAL_SEARCH, description="bank"),
        PlanStep(id=3, action=StepAction.ANALYZE, description="analyze"),
    ])
    env.step_outputs[1] = StepOutput(
        step_id=1, action="web_search", status=StepStatus.COMPLETE,
        output={"content_text": "FFmpeg 7.1 released 2026-05"},
    )
    env.step_outputs[2] = StepOutput(
        step_id=2, action="crystal_search", status=StepStatus.FAILED,
        output={}, error="boom",
    )
    env.step_outputs[3] = StepOutput(
        step_id=3, action="analyze", status=StepStatus.COMPLETE,
        output={"content": "analysis text — composition, not evidence"},
    )
    harvested = _harvest_findings(env)
    assert len(harvested) == 1
    assert harvested[0]["action"] == "web_search"
    assert harvested[0]["description"] == "web"
    assert "FFmpeg 7.1" in harvested[0]["text"]


def test_harvest_accumulates_across_attempts_and_dedups():
    env = CognitionEnvironment(customer_id="c")
    env.attempts = 2
    env.carried_findings = [{
        "attempt": 1, "step_id": 1, "action": "web_search",
        "description": "old", "text": "attempt-1 finding",
    }]
    env.plan = Plan(steps=[
        PlanStep(id=1, action=StepAction.WEB_SEARCH, description="gapfill"),
    ])
    env.step_outputs[1] = StepOutput(
        step_id=1, action="web_search", status=StepStatus.COMPLETE,
        output={"content_text": "attempt-2 gap-fill finding"},
    )
    harvested = _harvest_findings(env)
    texts = [f["text"] for f in harvested]
    assert "attempt-1 finding" in texts
    assert "attempt-2 gap-fill finding" in texts
    # Re-harvesting the same attempt does not duplicate.
    env.carried_findings = harvested
    assert len(_harvest_findings(env)) == 2


def test_harvest_renders_findings_list_when_no_content_text():
    env = CognitionEnvironment(customer_id="c")
    env.attempts = 1
    env.plan = Plan(steps=[
        PlanStep(id=1, action=StepAction.WEB_SEARCH, description="w")])
    env.step_outputs[1] = StepOutput(
        step_id=1, action="web_search", status=StepStatus.COMPLETE,
        output={"findings": [
            {"title": "Release notes", "url": "https://x.test/r",
             "snippet": "OTIO 0.17 shipped"},
        ]},
    )
    harvested = _harvest_findings(env)
    assert len(harvested) == 1
    assert "OTIO 0.17" in harvested[0]["text"]
    assert "https://x.test/r" in harvested[0]["text"]


# ---------------------------------------------------------------------------
# Q2A — the retry's orchestrator prompt + route parse
# ---------------------------------------------------------------------------

async def test_verdict_deliverable_and_findings_reach_retry_orchestrator():
    env = _rejected_env()
    fake = _FakeLLM([_orchestrator_json(route="compose_only")])
    set_llm_client(fake)
    try:
        goal, plan = await run_orchestrator(env=env, store=None,
                                            fact_store=None)
    finally:
        reset_llm_client()
    prompt = fake.prompts[0]
    # Verdict text.
    assert "placeholder-riddled" in prompt
    assert "[TODO] placeholders" in prompt
    # Trimmed rejected deliverable.
    assert "old report body" in prompt
    assert "REJECTED DELIVERABLE" in prompt
    # Carried-findings inventory.
    assert "FINDINGS ALREADY GATHERED" in prompt
    assert "FFmpeg" in prompt
    # Route instructions offered.
    for route in ("compose_only", "gap_fill", "replan", "give_up"):
        assert route in prompt
    # Route parsed into the plan.
    assert plan.retry_route == "compose_only"


async def test_first_attempt_prompt_has_no_revision_scaffolding():
    env = CognitionEnvironment(customer_id="c", task_goal="research X")
    fake = _FakeLLM([_orchestrator_json()])
    set_llm_client(fake)
    try:
        _, plan = await run_orchestrator(env=env, store=None,
                                         fact_store=None)
    finally:
        reset_llm_client()
    prompt = fake.prompts[0]
    assert "REJECTED DELIVERABLE" not in prompt
    assert "THIS IS A REVISION" not in prompt
    assert plan.retry_route == ""


async def test_unknown_route_normalizes():
    env = _rejected_env()
    fake = _FakeLLM([_orchestrator_json(route="try_harder")])
    set_llm_client(fake)
    try:
        _, plan = await run_orchestrator(env=env, store=None,
                                         fact_store=None)
    finally:
        reset_llm_client()
    assert plan.retry_route == ""


# ---------------------------------------------------------------------------
# Q1A/Q3A — composition sees carried findings + the revision block, bounded
# ---------------------------------------------------------------------------

async def test_composition_revision_prompt_carries_findings_and_verdict():
    env = _rejected_env()
    env.plan = Plan(steps=[
        PlanStep(id=1, action=StepAction.ANALYZE, description="revise")])
    step = env.plan.steps[0]
    fake = _FakeLLM(["revised text"])
    set_llm_client(fake)
    try:
        result = await _worker_llm_step(
            env, step,
            StepOutput(step_id=1, action="analyze",
                       status=StepStatus.RUNNING),
        )
    finally:
        reset_llm_client()
    prompt = fake.prompts[0]
    assert "Carried finding" in prompt
    assert "FFmpeg" in prompt
    assert "THIS IS A REVISION" in prompt
    assert "placeholder-riddled" in prompt
    assert "old report body" in prompt
    assert "without regressing what was adequate" in prompt
    assert result.status == StepStatus.COMPLETE


async def test_composition_first_attempt_has_no_revision_block():
    env = CognitionEnvironment(customer_id="c")
    env.plan = Plan(steps=[
        PlanStep(id=1, action=StepAction.ANALYZE, description="a")])
    fake = _FakeLLM(["text"])
    set_llm_client(fake)
    try:
        await _worker_llm_step(
            env, env.plan.steps[0],
            StepOutput(step_id=1, action="analyze",
                       status=StepStatus.RUNNING),
        )
    finally:
        reset_llm_client()
    assert "THIS IS A REVISION" not in fake.prompts[0]


def test_trim_head_tail_keeps_both_ends():
    text = "HEAD" + ("x" * 20000) + "TAIL"
    out = _trim_head_tail(text, _REVISION_DELIVERABLE_CHARS)
    assert len(out) <= _REVISION_DELIVERABLE_CHARS + 60
    assert out.startswith("HEAD")
    assert out.endswith("TAIL")
    assert "middle elided" in out
    short = "short text"
    assert _trim_head_tail(short, 100) == short


# ---------------------------------------------------------------------------
# Engine integration — carryover across the loop, replan drop, give_up
# ---------------------------------------------------------------------------

def _scripted_engine(monkeypatch, orchestrator_outputs, validator_verdicts):
    """Monkeypatch the three roles in the ENGINE's namespace with scripted
    fakes; capture the env state each retry's orchestrator observes."""
    observed: list[dict[str, Any]] = []

    async def fake_orchestrator(*, env, store, fact_store, encoder=None):
        observed.append({
            "attempt": env.attempts,
            "carried": [dict(f) for f in env.carried_findings],
            "prior_deliverable": env.prior_deliverable,
        })
        return orchestrator_outputs.pop(0)

    async def fake_worker(env, step, store, fact_store, encoder):
        out = StepOutput(step_id=step.id, action=step.action.value,
                         status=StepStatus.COMPLETE)
        if step.action == StepAction.WEB_SEARCH:
            # content_text feeds the harvest; findings feeds the C2
            # answerability gate's grounding count.
            out.output = {
                "content_text": f"evidence-{env.attempts}",
                "findings": [{"title": "t", "url": "u",
                              "snippet": f"evidence-{env.attempts}"}],
            }
        else:
            out.output = {"content": f"deliverable-attempt-{env.attempts}",
                          "is_deliverable": True}
        return out

    async def fake_validator(*, env):
        return validator_verdicts.pop(0)

    async def fake_snapshot(store, env, terminal=False):
        return None

    monkeypatch.setattr(engine_mod, "run_orchestrator", fake_orchestrator)
    monkeypatch.setattr(engine_mod, "run_worker", fake_worker)
    monkeypatch.setattr(engine_mod, "run_validator", fake_validator)
    monkeypatch.setattr(engine_mod, "_persist_snapshot", fake_snapshot)
    return observed


def _plan(route: str = "", steps: Optional[list[PlanStep]] = None) -> Plan:
    return Plan(
        steps=steps if steps is not None else [
            PlanStep(id=1, action=StepAction.WEB_SEARCH, description="w"),
            PlanStep(id=2, action=StepAction.FORMAT, description="f",
                     depends_on=[1]),
        ],
        retry_route=route,
        reasoning="scripted",
    )


def _goal() -> GoalDocument:
    return GoalDocument(title="t", description="d",
                        acceptance_criteria=["c"])


def _reject(reason: str = "bad") -> ValidationResult:
    return ValidationResult(approved=False, score=0.2, reasoning=reason,
                            issues=[reason])


def _approve() -> ValidationResult:
    return ValidationResult(approved=True, score=0.9, reasoning="ok")


async def test_engine_carries_findings_and_deliverable_into_retry(
        monkeypatch, store):
    observed = _scripted_engine(
        monkeypatch,
        orchestrator_outputs=[(_goal(), _plan()),
                              (_goal(), _plan(route="compose_only"))],
        validator_verdicts=[_reject("placeholders"), _approve()],
    )
    result = await run_cognition_workflow(
        "goal", "cus_t", store, None, None,
        output_type="report", max_attempts=2,
    )
    assert result.success is True
    # Attempt 1: cold.
    assert observed[0]["carried"] == []
    assert observed[0]["prior_deliverable"] == ""
    # Attempt 2: the revision sees attempt 1's evidence + deliverable.
    assert len(observed[1]["carried"]) == 1
    assert observed[1]["carried"][0]["text"] == "evidence-1"
    assert observed[1]["prior_deliverable"] == "deliverable-attempt-1"


async def test_engine_replan_route_drops_carryover(monkeypatch, store):
    observed = _scripted_engine(
        monkeypatch,
        orchestrator_outputs=[(_goal(), _plan()),
                              (_goal(), _plan(route="replan"))],
        validator_verdicts=[_reject(), _approve()],
    )
    result = await run_cognition_workflow(
        "goal", "cus_t", store, None, None,
        output_type="report", max_attempts=2,
    )
    assert result.success is True
    env_dump = result.workflow_summary
    # The orchestrator SAW the carryover (it needs it to classify)...
    assert observed[1]["prior_deliverable"] == "deliverable-attempt-1"
    # ...but after choosing replan the env ran cold.
    assert env_dump["carried_findings"] == []
    assert env_dump["prior_deliverable"] == ""


async def test_engine_give_up_short_circuits_with_explanation(
        monkeypatch, store):
    give_up_plan = Plan(steps=[], retry_route="give_up",
                        reasoning="web search is unconfigured; the goal "
                                  "needs external data")
    observed = _scripted_engine(
        monkeypatch,
        orchestrator_outputs=[(_goal(), _plan()),
                              (_goal(), give_up_plan)],
        validator_verdicts=[_reject()],  # attempt 2 never validates
    )
    result = await run_cognition_workflow(
        "goal", "cus_t", store, None, None,
        output_type="report", max_attempts=3,
    )
    assert result.success is False
    assert "orchestrator_gave_up" in (result.reason or "")
    assert "unconfigured" in (result.reason or "")
    assert len(observed) == 2  # no third attempt burned


async def test_engine_give_up_ignored_on_first_attempt(monkeypatch, store):
    """A first-attempt give_up is not honored — attempt 1 has no verdict
    to classify; the C2 park covers genuinely unanswerable tasks."""
    _scripted_engine(
        monkeypatch,
        orchestrator_outputs=[(_goal(), _plan(route="give_up"))],
        validator_verdicts=[_approve()],
    )
    result = await run_cognition_workflow(
        "goal", "cus_t", store, None, None,
        output_type="report", max_attempts=1,
    )
    assert result.success is True  # the plan simply ran


async def test_attempt_history_still_archives_every_attempt(
        monkeypatch, store):
    _scripted_engine(
        monkeypatch,
        orchestrator_outputs=[(_goal(), _plan()),
                              (_goal(), _plan(route="compose_only"))],
        validator_verdicts=[_reject("first"), _reject("second")],
    )
    result = await run_cognition_workflow(
        "goal", "cus_t", store, None, None,
        output_type="report", max_attempts=2,
    )
    assert result.success is False
    history = result.workflow_summary["attempt_history"]
    assert [h["attempt"] for h in history] == [1, 2]
    assert history[1]["plan"]["retry_route"] == "compose_only"
