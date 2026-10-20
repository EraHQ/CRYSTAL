"""The deliverable is a document for a person, whatever model plans it
(M3 follow-up, 2026-10-19).

Found live on the first Nemotron-planned scheduled run: the
orchestrator prompt said `OUTPUT TYPE: crystal` with no explanation,
Nemotron read it as "the document must be a crystal-format JSON
object", and the brief that reached the Scheduled runs page was a
```json block echoing SOURCE CRYSTAL / OUTPUT TYPE. Claude had
happened to ignore the line. Pins:
  - the orchestrator prompt explains OUTPUT TYPE as storage, not
    document format, and says the deliverable is Markdown, never JSON;
  - the FORMAT worker carries a document-form rule that overrides the
    step instruction; analyze/synthesize steps do not carry it.

Second finding, same run, next attempt: the planner (the SMALL tier
under a Nemotron pick) wrote acceptance criteria demanding "an inline
date and a source identifier tag on every claim", and the validator
enforced them: attempts 2 and 3 scored 40% on citation syntax alone.
Pins: the planner is told criteria say WHAT the reader gets and never
dictate citation syntax; the validator is told grounding is judged by
whether a claim traces to the material, never by how a citation is
written.

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

import json

from crystal_cache.cognition.models import (
    CognitionEnvironment,
    Plan,
    PlanStep,
    StepAction,
    StepOutput,
    StepStatus,
)
from crystal_cache.cognition.roles import _worker_llm_step, run_orchestrator
from crystal_cache.llm import reset_llm_client, set_llm_client
from crystal_cache.llm.client import LLMResult


class _FakeLLM:
    def __init__(self, texts: list[str]):
        self._texts = list(texts)
        self.prompts: list[str] = []
        self.systems: list[str] = []

    def complete_detailed(self, *, system, messages, max_tokens,
                          temperature=1.0, tier="small", model=None,
                          json_schema=None) -> LLMResult:
        self.prompts.append(messages[-1]["content"])
        self.systems.append(system or "")
        text = self._texts.pop(0) if len(self._texts) > 1 else self._texts[0]
        return LLMResult(text=text, model="fake", input_tokens=1,
                         output_tokens=1)

    def is_ready(self) -> bool:
        return True


def _plan_json() -> str:
    return json.dumps({
        "goal": {"title": "T", "description": "D",
                 "acceptance_criteria": ["addresses the request"]},
        "plan": {
            "reasoning": "r",
            "steps": [{"id": 1, "action": "format", "description": "f",
                       "input": {"format": "markdown"}, "depends_on": [],
                       "parallel_group": None}],
            "expected_output": "text", "suggested_key": "k",
            "parent_crystal_id": "", "retry_route": "",
        },
    })


async def test_orchestrator_prompt_explains_output_type_as_storage():
    env = CognitionEnvironment(customer_id="c", task_goal="summarize news")
    fake = _FakeLLM([_plan_json()])
    set_llm_client(fake)
    try:
        await run_orchestrator(env=env, store=None, fact_store=None)
    finally:
        reset_llm_client()
    prompt = fake.prompts[0]
    assert "OUTPUT TYPE: crystal (how the SYSTEM stores" in prompt
    assert "NOT a document format" in prompt
    assert "readable Markdown document written for a person" in prompt
    assert "Never JSON, YAML or a code block" in prompt
    # Contracts are about content, never citation syntax.
    assert "Never prescribe citation" in prompt
    assert "no inline dates, identifier tags, bracket formats" in prompt


async def test_validator_never_rejects_for_citation_syntax():
    from crystal_cache.cognition.models import GoalDocument
    from crystal_cache.cognition.roles import run_validator

    env = CognitionEnvironment(customer_id="c", task_goal="brief")
    env.goal = GoalDocument(title="T", description="D",
                            acceptance_criteria=["covers the changes"])
    env.deliverables["brief.md"] = "# Brief\n\nA change [fact_cr1]."
    fake = _FakeLLM([json.dumps({
        "approved": True, "score": 0.9, "reasoning": "ok",
        "criteria_evaluation": [], "issues": [], "suggestions": [],
        "residual_gaps": [],
    })])
    set_llm_client(fake)
    try:
        verdict = await run_validator(env)
    finally:
        reset_llm_client()
    assert verdict.approved is True
    system = fake.systems[0]
    assert "never by citation syntax" in system
    assert "never rejected for how its citations are written" in system
    assert "the source title or the reference tag the material carries is the citation" in system


async def _worker_prompt(action: StepAction) -> str:
    env = CognitionEnvironment(customer_id="c")
    env.plan = Plan(steps=[PlanStep(
        id=1, action=action, description="produce the brief",
        input={"instruction": "Output a crystal-format JSON object"},
    )])
    fake = _FakeLLM(["# Brief\n\ntext"])
    set_llm_client(fake)
    try:
        out = await _worker_llm_step(
            env, env.plan.steps[0],
            StepOutput(step_id=1, action=action.value,
                       status=StepStatus.RUNNING),
        )
    finally:
        reset_llm_client()
    assert out.status == StepStatus.COMPLETE
    return fake.prompts[0]


async def test_format_step_carries_the_document_form_rule():
    prompt = await _worker_prompt(StepAction.FORMAT)
    assert "THIS IS THE FINAL DOCUMENT STEP" in prompt
    assert "DOCUMENT FORM: readable Markdown for a person" in prompt
    assert "whatever the step instruction says about format" in prompt


async def test_intermediate_steps_do_not_carry_the_final_document_rules():
    for action in (StepAction.ANALYZE, StepAction.SYNTHESIZE):
        prompt = await _worker_prompt(action)
        assert "THIS IS THE FINAL DOCUMENT STEP" not in prompt
        assert "DOCUMENT FORM:" not in prompt
