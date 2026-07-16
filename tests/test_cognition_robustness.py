"""Cognition robustness (2026-07-16 evening): the live-run triage.

Three findings from the cycle-2/3 forensics: (1) a targets-less
research step burned an agent session asking a question nobody
answers, then composed it as a "finding"; (2) the composing agent
hunted the bank for this run's own step outputs and pulled in an
unrelated Python tutorial; (3) validator JSON truncated AGAIN after
residual_gaps grew the verdict against the fixed 4000 cap (third
recurrence — 1500 -> 4000 -> 16000; caps are blast radius, never
sized to expected output). Q1-C: one terse re-judge before
fail-closed.

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

import json as _json

import pytest

from crystal_cache.cognition import agentic as agentic_mod
from crystal_cache.cognition import roles as roles_mod
from crystal_cache.cognition.models import (
    CognitionEnvironment,
    GoalDocument,
)
from crystal_cache.llm import reset_llm_client, set_llm_client
from crystal_cache.llm.client import LLMResult


class _SequenceLLM:
    """Returns scripted texts in order; counts calls."""

    def __init__(self, texts: list[str]):
        self._texts = list(texts)
        self.calls = 0
        self.prompts: list[str] = []

    def complete_detailed(self, *, system, messages, max_tokens,
                          temperature=1.0, tier="small", model=None,
                          json_schema=None) -> LLMResult:
        self.calls += 1
        self.prompts.append(messages[-1]["content"])
        text = self._texts.pop(0) if self._texts else ""
        return LLMResult(text=text, model="fake",
                         input_tokens=10, output_tokens=10)

    def is_ready(self) -> bool:
        return True


class _StubStep:
    def __init__(self, step_input=None, description=""):
        self.id = 1
        self.input = step_input or {}
        self.description = description


# ---------------------------------------------------------------------------
# Targets guard
# ---------------------------------------------------------------------------

async def test_research_step_fails_fast_without_targets():
    """No targets -> error output, ZERO model calls (an LLM client is
    deliberately not installed; reaching one would explode)."""
    env = CognitionEnvironment(customer_id="c")
    out = await agentic_mod.run_research_step(
        env=env, step=_StubStep({}), store=None,
        fact_store=None, encoder=None,
    )
    assert out["findings"] == []
    assert out["results_count"] == 0
    assert "without targets" in out["error"]
    assert "cannot see the orchestrator's context" in out["error"]


async def test_run_worker_marks_research_error_failed():
    """The RESEARCH branch converts an error output into a FAILED
    step (fail-fast rails), never COMPLETE with a question-finding."""
    import inspect
    src = inspect.getsource(roles_mod.run_worker)
    assert 'result.output.get("error")' in src or \
        "(result.output or {}).get(\"error\")" in src
    assert "StepStatus.FAILED" in src


# ---------------------------------------------------------------------------
# Validator re-judge (Q1-C)
# ---------------------------------------------------------------------------

def _env_for_validator() -> CognitionEnvironment:
    env = CognitionEnvironment(customer_id="c")
    env.goal = GoalDocument(title="t", description="d",
                            acceptance_criteria=["c1"])
    env.deliverables = {"final": "deliverable body " * 10}
    return env


def _verdict_json() -> str:
    return _json.dumps({
        "approved": True, "score": 0.9, "reasoning": "ok",
        "criteria_evaluation": [
            {"criterion": "c1", "status": "MET", "evidence": "e"},
        ],
        "issues": [], "suggestions": [], "residual_gaps": [],
    })


async def test_validator_rejudges_once_after_parse_failure():
    fake = _SequenceLLM(["THIS IS NOT JSON {truncated", _verdict_json()])
    set_llm_client(fake)
    try:
        result = await roles_mod.run_validator(env=_env_for_validator())
    finally:
        reset_llm_client()
    assert fake.calls == 2
    assert result.approved is True
    assert result.score == 0.9
    # The retry prompt carries the terse escalation.
    assert "COULD NOT BE PARSED" in fake.prompts[1]


async def test_validator_fails_closed_after_two_parse_failures():
    fake = _SequenceLLM(["garbage one {", "garbage two {"])
    set_llm_client(fake)
    try:
        result = await roles_mod.run_validator(env=_env_for_validator())
    finally:
        reset_llm_client()
    assert fake.calls == 2  # exactly ONE re-judge, then fail closed
    assert result.approved is False
    assert "could not be parsed" in result.reasoning.lower()


def test_validator_cap_is_blast_radius():
    assert roles_mod._VALIDATOR_MAX_TOKENS == 16000


# ---------------------------------------------------------------------------
# Charter / rule pins
# ---------------------------------------------------------------------------

def test_orchestrator_discipline_pins():
    import inspect
    src = inspect.getsource(roles_mod)
    assert "SELF-CONTAINED STEP INPUTS" in src
    assert "CITATION STANDARD" in src
    assert "Citation sufficiency" in src          # validator-side rule
    assert "INHERITED CONTRACT" in src            # Q2-A prompt block


def test_worker_charter_bank_rule_pin():
    charter = agentic_mod._worker_charter()
    assert "PRIOR STEP OUTPUTS are already included" in charter
    assert "NEVER use them to look for this run's own steps" in charter


def test_engine_inheritance_seam():
    import inspect
    from crystal_cache.cognition import engine as engine_mod
    src = inspect.getsource(engine_mod.run_cognition_workflow)
    assert "goal_inheritance_enforced" in src
    assert "prior_cycle_goal" in src


# ---------------------------------------------------------------------------
# Fail-closed on validator transport errors (the 529 incident)
# ---------------------------------------------------------------------------

class _RaiseThenValidLLM:
    """First call raises (like an upstream 529); second returns valid
    JSON — a transient overload must cost one retry, not the attempt."""

    def __init__(self, valid_text: str):
        self._valid = valid_text
        self.calls = 0

    def complete_detailed(self, *, system, messages, max_tokens,
                          temperature=1.0, tier="small", model=None,
                          json_schema=None) -> LLMResult:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("Error code: 529 - overloaded_error")
        return LLMResult(text=self._valid, model="fake",
                         input_tokens=10, output_tokens=10)

    def is_ready(self) -> bool:
        return True


class _AlwaysRaiseLLM:
    def __init__(self):
        self.calls = 0

    def complete_detailed(self, **kwargs) -> LLMResult:
        self.calls += 1
        raise RuntimeError("Error code: 529 - overloaded_error")

    def is_ready(self) -> bool:
        return True


async def test_validator_survives_transient_transport_error():
    fake = _RaiseThenValidLLM(_verdict_json())
    set_llm_client(fake)
    try:
        result = await roles_mod.run_validator(env=_env_for_validator())
    finally:
        reset_llm_client()
    assert fake.calls == 2
    assert result.approved is True
    assert result.score == 0.9


async def test_validator_fails_closed_when_both_calls_die():
    fake = _AlwaysRaiseLLM()
    set_llm_client(fake)
    try:
        result = await roles_mod.run_validator(env=_env_for_validator())
    finally:
        reset_llm_client()
    assert fake.calls == 2  # one retry, never more
    assert result.approved is False
    assert "could not be parsed" in result.reasoning.lower()


async def test_engine_never_commits_on_validator_crash(monkeypatch):
    """The fail-open hole: a validator exception with a deliverable
    present used to mark the run COMPLETE and commit it unvalidated.
    Now: FAILED, nothing committed."""
    from crystal_cache.cognition import engine as engine_mod
    from crystal_cache.cognition.engine import run_cognition_workflow
    from crystal_cache.cognition.models import (
        Plan, PlanStep, StepAction, StepOutput, StepStatus,
    )

    async def fake_orchestrator(*, env, store, fact_store, encoder=None):
        goal = GoalDocument(title="t", description="d",
                            acceptance_criteria=["c"])
        plan = Plan(reasoning="r", steps=[PlanStep(
            id=1, action=StepAction.SYNTHESIZE, description="s",
            input={"instruction": "compose"},
        )])
        return (goal, plan)

    async def fake_worker(env, step, _store, _fact_store, _encoder):
        out = StepOutput(step_id=step.id, action=step.action.value,
                         status=StepStatus.COMPLETE)
        out.output = {
            "content": ("a deliverable comfortably longer than the "
                        "fifty-char salvage gate so env.deliverables "
                        "is populated when the validator dies"),
            "is_deliverable": True,
        }
        return out

    async def crashing_validator(*, env):
        raise RuntimeError("validator bug")

    committed = {"called": False}

    async def spy_commit(env, store, encoder, fact_store):
        committed["called"] = True
        raise AssertionError("commit must never run on validator crash")

    monkeypatch.setattr(engine_mod, "run_orchestrator", fake_orchestrator)
    monkeypatch.setattr(engine_mod, "run_worker", fake_worker)
    monkeypatch.setattr(engine_mod, "run_validator", crashing_validator)
    monkeypatch.setattr(engine_mod, "_commit_and_finalize", spy_commit)

    result = await run_cognition_workflow(
        "goal", "cust-x", None, None, None,
        output_type="report", trigger_type="research",
        trigger_id=None, max_attempts=1,
    )
    assert result.success is False
    assert "Validator failed" in (result.reason or "")
    assert committed["called"] is False
