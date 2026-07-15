"""Contract amendment — the appeal, properly located (ratified
2026-07-14, all A, after rematch #11).

The failure class: the orchestrator invented a fixed count ("at least
3 emerging projects") the world isn't obligated to contain; the
validator rightly enforced it; the run rejected forever. The appeal
lives at the ORCHESTRATOR seat between attempts, adjudicated on
documented evidence — never negotiated with the validator (the clean
room is what makes the bank-entry gate safe). This proves:
  - Q1A: the goal-writing rule (criteria satisfiable regardless of
    what the world contains) is in the orchestrator prompt;
  - the validator schema carries possibly_infeasible with the
    evidence-only rule, parsed into CriterionEval;
  - the amend_contract route is offered ONLY when the last verdict
    flagged criteria; the payload parses into Plan;
  - the ENGINE applies amendments only to flagged indices, writes the
    permanent audit trail onto the goal, and ignores+logs unflagged
    proposals;
  - the validator prompt shows the amendment audit and judges against
    the CURRENT criteria;
  - Q3A: the research charter carries the FFmpeg-class rule (version
    truth off-GitHub; never pair a version with another version's
    changelog).

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from crystal_cache.cognition import roles as roles_mod
from crystal_cache.cognition.agentic import _research_charter
from crystal_cache.cognition.models import (
    CognitionEnvironment,
    CriterionEval,
    GoalDocument,
    Plan,
    ValidationResult,
)
from crystal_cache.cognition.roles import run_orchestrator, run_validator
from crystal_cache.llm.client import LLMResult


class _PromptLLM:
    def __init__(self, text):
        self.prompts = []
        self._text = text

    def complete_detailed(self, *, system, messages, max_tokens,
                          temperature=1.0, tier="small", model=None,
                          json_schema=None):
        self.prompts.append(messages[0]["content"])
        return LLMResult(text=self._text, model="fake", input_tokens=1,
                         output_tokens=1, stop_reason="end_turn")

    def is_ready(self):
        return True


def _orch_json(route="", amendments=None):
    return json.dumps({
        "goal": {"title": "t", "description": "d",
                 "acceptance_criteria": ["c1", "c2"],
                 "output_type": "report"},
        "plan": {"reasoning": "r", "steps": [
            {"id": 1, "action": "analyze", "description": "a",
             "input": {}, "depends_on": [], "parallel_group": None}],
            "expected_output": "o", "suggested_key": "k",
            "parent_crystal_id": "", "retry_route": route,
            "contract_amendments": amendments or [],
            "bank_finding_ids": []},
    })


_VERDICT = (
    '{"approved": true, "score": 0.9, "reasoning": "ok",'
    ' "criteria_evaluation": [], "issues": [], "suggestions": []}'
)


def _flagged_history_env():
    env = CognitionEnvironment(customer_id="c", task_goal="g")
    env.rejection_log = [{"attempt": 1, "reasoning": "rej", "issues": [],
                          "suggestions": []}]
    env.attempt_history = [{
        "attempt": 1, "plan": {}, "steps": [], "deliverable": "",
        "validation": {
            "approved": False, "score": 0.4, "reasoning": "rej",
            "criteria_evaluation": [
                {"criterion": "c1", "status": "MET", "evidence": "",
                 "possibly_infeasible": False},
                {"criterion": "at least 3 emerging projects",
                 "status": "NOT_MET",
                 "evidence": "9 searches, 6 repos examined, 2 found",
                 "possibly_infeasible": True},
            ],
            "issues": [], "suggestions": [],
        },
    }]
    return env


async def _run_orch(env, text, monkeypatch):
    async def fake_source(env, store, fact_store, encoder):
        return []

    monkeypatch.setattr(roles_mod, "_source_bank_findings", fake_source)
    from crystal_cache.llm import reset_llm_client, set_llm_client
    fake = _PromptLLM(text)
    set_llm_client(fake)
    try:
        goal, plan = await run_orchestrator(env=env, store=None,
                                            fact_store=None)
    finally:
        reset_llm_client()
    return goal, plan, fake.prompts[0]


# --- Q1A + route offering ----------------------------------------------------

async def test_goal_writing_rule_always_present(monkeypatch):
    env = CognitionEnvironment(customer_id="c", task_goal="g")
    _, _, prompt = await _run_orch(env, _orch_json(), monkeypatch)
    assert "CRITERIA MUST BE SATISFIABLE" in prompt
    assert "fixed count the world isn't obligated to contain" in prompt
    # No flagged history -> the amend route is NOT offered.
    assert '"amend_contract"' not in prompt


async def test_amend_route_offered_only_when_flagged(monkeypatch):
    env = _flagged_history_env()
    _, plan, prompt = await _run_orch(
        env,
        _orch_json(route="amend_contract", amendments=[
            {"criterion_index": 1,
             "amended": "all emerging projects that verifiably exist",
             "evidence": "9 searches documented"},
        ]),
        monkeypatch,
    )
    assert '"amend_contract"' in prompt
    assert "POSSIBLY" in prompt and "INFEASIBLE" in prompt
    assert "at least 3 emerging projects" in prompt
    assert "never simply delete a criterion" in prompt
    # Payload parsed into the plan.
    assert plan.retry_route == "amend_contract"
    assert plan.contract_amendments[0]["criterion_index"] == 1
    assert plan.contract_amendments[0]["amended"].startswith("all emerging")


# --- validator: flag parse + audit visibility ---------------------------------

async def test_validator_parses_flag_and_shows_amendment_audit():
    from crystal_cache.llm import reset_llm_client, set_llm_client
    env = CognitionEnvironment(customer_id="c")
    env.goal = GoalDocument(
        title="T", description="D",
        acceptance_criteria=["all emerging projects that verifiably exist"],
        amendments=[{
            "attempt": 2, "index": 0,
            "original": "at least 3 emerging projects",
            "amended": "all emerging projects that verifiably exist",
            "evidence": "9 searches documented",
        }],
    )
    env.deliverables["main"] = "SECTION A"
    verdict = (
        '{"approved": false, "score": 0.5, "reasoning": "r",'
        ' "criteria_evaluation": [{"criterion": "c", "status": "NOT_MET",'
        ' "evidence": "e", "possibly_infeasible": true}],'
        ' "issues": [], "suggestions": []}'
    )
    fake = _PromptLLM(verdict)
    set_llm_client(fake)
    try:
        result = await run_validator(env=env)
    finally:
        reset_llm_client()
    assert result.criteria_evaluation[0].possibly_infeasible is True
    prompt = fake.prompts[0]
    assert "CONTRACT AMENDMENTS" in prompt
    assert "at least 3 emerging projects" in prompt
    assert "judge against the CURRENT criteria" in prompt
    assert "possibly_infeasible" in prompt
    assert "Absence of effort is NEVER infeasibility" in prompt
    # Round-trip: the flag survives to_dict (attempt_history persistence).
    assert result.to_dict()["criteria_evaluation"][0][
        "possibly_infeasible"] is True


# --- engine application ------------------------------------------------------

def _engine_apply(env, plan, attempt=1):
    """Run the engine's amendment stanza in isolation (mirrors the
    attempt-loop code path)."""
    import structlog
    logger = structlog.get_logger("test")
    _last_val = {}
    if env.attempt_history:
        _last_val = (env.attempt_history[-1] or {}).get("validation") or {}
    _flagged_idx = {
        i for i, c in enumerate(_last_val.get("criteria_evaluation") or [])
        if isinstance(c, dict) and c.get("possibly_infeasible")
    }
    applied, rejected = [], []
    for a in plan.contract_amendments:
        idx = a.get("criterion_index")
        if (idx not in _flagged_idx or not env.goal
                or idx >= len(env.goal.acceptance_criteria)):
            rejected.append(idx)
            continue
        original = env.goal.acceptance_criteria[idx]
        env.goal.acceptance_criteria[idx] = a["amended"]
        env.goal.amendments.append({
            "attempt": attempt + 1, "index": idx, "original": original,
            "amended": a["amended"], "evidence": a.get("evidence", ""),
        })
        applied.append(idx)
    return applied, rejected


def test_engine_applies_only_flagged_amendments():
    env = _flagged_history_env()
    env.goal = GoalDocument(
        title="T", description="D",
        acceptance_criteria=["c1", "at least 3 emerging projects"],
    )
    plan = Plan(retry_route="amend_contract", contract_amendments=[
        {"criterion_index": 1,
         "amended": "all emerging projects that verifiably exist",
         "evidence": "9 searches"},
        {"criterion_index": 0,  # NOT flagged — must be rejected
         "amended": "weakened c1", "evidence": ""},
    ])
    applied, rejected = _engine_apply(env, plan)
    assert applied == [1]
    assert rejected == [0]
    # The flagged criterion is amended; the unflagged one untouched.
    assert env.goal.acceptance_criteria[0] == "c1"
    assert env.goal.acceptance_criteria[1] == (
        "all emerging projects that verifiably exist")
    # Permanent audit trail.
    audit = env.goal.amendments[0]
    assert audit["original"] == "at least 3 emerging projects"
    assert audit["index"] == 1
    assert audit["evidence"] == "9 searches"


def test_engine_stanza_matches_this_test():
    """Guard: the engine source contains the same flagged-only logic
    this test mirrors — anchor strings that break if the stanza is
    edited without updating the mirror above."""
    import inspect

    import crystal_cache.cognition.engine as engine_mod
    src = inspect.getsource(engine_mod)
    assert 'plan.retry_route == "amend_contract"' in src
    assert "cognition.amendment_rejected" in src
    assert "cognition.contract_amended" in src
    assert "possibly_infeasible" in src


# --- Q3A ---------------------------------------------------------------------

def test_research_charter_carries_ffmpeg_class_rule():
    charter = _research_charter()
    assert "do NOT publish GitHub releases" in charter
    assert "ffmpeg.org" in charter
    assert "different version's changelog" in charter
