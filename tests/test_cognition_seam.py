"""Slice 4: the cognition roles run through the provider-neutral seam.

roles.py maps the persisted wire-keys ("haiku"/"sonnet") onto the seam
tiers ("small"/"large") and calls complete_detailed; slm_client is gone
from the role signatures and the engine. Tests inject a seam-shaped fake
via set_llm_client. FakeAnthropic records the tier as the call's "model"
when no explicit model is passed, which is what the tier assertions
below rely on.

R14 note: these assertions are verified by `pytest`; they describe
expected behavior and have not yet been run at authoring time.
"""
from __future__ import annotations

import json

from crystal_cache.cognition.models import (
    CognitionEnvironment,
    GoalDocument,
    PlanStep,
    StepAction,
    StepOutput,
    StepStatus,
)
from crystal_cache.cognition.roles import _worker_llm_step, run_validator
from crystal_cache.llm import reset_llm_client, set_llm_client

from fakes import FakeAnthropic


def _env_with_goal() -> CognitionEnvironment:
    env = CognitionEnvironment(customer_id="")
    env.goal = GoalDocument(
        title="Answer the question",
        description="Produce a short factual answer.",
        acceptance_criteria=["The deliverable answers the question."],
    )
    env.deliverables["main"] = "The answer is 42."
    return env


async def test_validator_runs_through_seam_at_large_tier():
    fake = FakeAnthropic()
    fake.script_text(json.dumps({
        "approved": True,
        "score": 0.9,
        "reasoning": "criterion met",
        "criteria_evaluation": [],
        "issues": [],
        "suggestions": [],
    }))
    env = _env_with_goal()

    set_llm_client(fake)
    try:
        validation = await run_validator(env=env)
    finally:
        reset_llm_client()

    assert validation.approved is True
    assert validation.score == 0.9
    # The sonnet wire-key maps to the large tier (FakeAnthropic records
    # the tier as the model when none is passed explicitly).
    call = fake.assert_called_once()
    assert call["model"] == "large"
    # Wire-key token labels are preserved on the env.
    assert env.tokens_used == 0  # fake reports no usage


async def test_validator_fail_closed_on_unparseable_response():
    """An unparseable validator response rejects (never approves).

    Q1-C (2026-07-16): the validator re-judges ONCE before failing
    closed, so both scripted responses are garbage.
    """
    fake = FakeAnthropic()
    fake.script_text("this is not json at all")
    fake.script_text("still not json (the re-judge)")
    env = _env_with_goal()

    set_llm_client(fake)
    try:
        validation = await run_validator(env=env)
    finally:
        reset_llm_client()

    assert validation.approved is False


async def test_worker_synthesize_forces_sonnet_key_and_large_tier():
    fake = FakeAnthropic()
    fake.script_text("synthesized deliverable text")
    env = CognitionEnvironment(customer_id="")
    step = PlanStep(
        id=1,
        action=StepAction.SYNTHESIZE,
        description="combine prior work",
        input={"instruction": "combine"},
        model="haiku",  # persisted plans may say haiku; SYNTHESIZE overrides
    )
    result = StepOutput(
        step_id=step.id,
        action=step.action.value,
        status=StepStatus.RUNNING,
    )

    set_llm_client(fake)
    try:
        out = await _worker_llm_step(env, step, result)
    finally:
        reset_llm_client()

    assert out.status == StepStatus.COMPLETE
    assert out.model_used == "sonnet"  # wire-key preserved in persisted output
    assert out.output["content"] == "synthesized deliverable text"
    call = fake.assert_called_once()
    assert call["model"] == "large"


async def test_worker_analyze_defaults_to_haiku_key_and_small_tier():
    fake = FakeAnthropic()
    fake.script_text("analysis text")
    env = CognitionEnvironment(customer_id="")
    step = PlanStep(
        id=1,
        action=StepAction.ANALYZE,
        description="analyze prior work",
        input={"instruction": "analyze"},
        model="not-a-known-key",  # unknown persisted key falls back to haiku
    )
    result = StepOutput(
        step_id=step.id,
        action=step.action.value,
        status=StepStatus.RUNNING,
    )

    set_llm_client(fake)
    try:
        out = await _worker_llm_step(env, step, result)
    finally:
        reset_llm_client()

    assert out.model_used == "haiku"
    call = fake.assert_called_once()
    assert call["model"] == "small"
