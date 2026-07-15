"""Workers-as-CRYS (ratified 2026-07-13, Q1–Q5 all A).

Composition steps behind CC_COGNITION_AGENTIC_WORKERS run as bounded
agent sessions with a fixed read-only toolset. This proves:
  - the scoped registry holds EXACTLY the five read verbs (Q2A) —
    write tools and cognition_run do not exist in the worker's
    universe, so recursion and writes are structurally impossible;
  - each tool impl routes through dispatch_cognition_retrieval — the
    same adapter the deterministic retrieval steps use;
  - flag OFF: the classic single-call path is untouched;
  - flag ON: the agent session's final text becomes the step output,
    with the tool trace attached; the classic path is never called;
  - ANY agentic failure (exception, timeout, empty output) falls back
    to the classic path — the new machinery cannot lose an attempt;
  - Q3A caps: Agent constructed with max_iterations = 6 and
    max_tokens = the flat composition cap; wall clock enforced via
    asyncio.wait_for;
  - metering: one aggregated llm_calls row per session + env totals.

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from crystal_cache.cognition import agentic as agentic_mod
from crystal_cache.cognition import roles as roles_mod
from crystal_cache.cognition.agentic import (
    _AGENTIC_MAX_TOOL_CALLS,
    build_worker_registry,
    run_agentic_composition,
)
from crystal_cache.cognition.models import (
    CognitionEnvironment,
    Plan,
    PlanStep,
    StepAction,
    StepOutput,
    StepStatus,
)
from crystal_cache.cognition.roles import (
    _COMPOSITION_MAX_TOKENS,
    _worker_llm_step,
)


# --- Q2A: the scoped registry ------------------------------------------------

_READ_VERBS = {"web_search", "web_fetch", "crystal_search",
               "crystal_key_scan", "source_lookup"}


def test_worker_registry_is_exactly_the_five_read_verbs():
    registry = build_worker_registry(store=None, fact_store=None,
                                     encoder=None)
    names = set(registry._tools.keys())
    assert names == _READ_VERBS
    # Structural enforcement: the dangerous names simply don't exist.
    for forbidden in ("cognition_run", "crystal_write", "document_upload",
                      "llm_invoke", "crystal_push_store"):
        assert forbidden not in names


async def test_worker_tools_route_through_the_dispatch_adapter(monkeypatch):
    from crystal_cache.cognition import retrieval_adapter as ra
    calls = []

    async def fake_dispatch(*, action_value, step_input, customer_id,
                            store, fact_store, encoder):
        calls.append((action_value, step_input, customer_id))
        return {"findings": [], "results_count": 0}

    monkeypatch.setattr(ra, "dispatch_cognition_retrieval", fake_dispatch)
    registry = build_worker_registry(store="S", fact_store="F", encoder="E")

    out = await registry._tools["web_fetch"].impl(
        customer_id="cust1",
        urls=["https://github.com/Breakthrough/PySceneDetect"],
    )
    assert out["results_count"] == 0
    action, step_input, cust = calls[0]
    assert action == "web_fetch"
    assert step_input == {
        "urls": ["https://github.com/Breakthrough/PySceneDetect"]}
    assert cust == "cust1"

    await registry._tools["web_search"].impl(
        customer_id="cust1", queries=["pyscenedetect github"])
    assert calls[1][0] == "web_search"
    assert calls[1][1] == {"queries": ["pyscenedetect github"]}


# --- the roles branch --------------------------------------------------------

class _ScriptedLLM:
    """Classic-path fake."""

    def __init__(self, text="classic text"):
        self.calls = 0
        self._text = text

    def complete_detailed(self, *, system, messages, max_tokens,
                          temperature=1.0, tier="small", model=None,
                          json_schema=None):
        self.calls += 1
        from crystal_cache.llm.client import LLMResult
        return LLMResult(text=self._text, model="fake", input_tokens=5,
                         output_tokens=5, stop_reason="end_turn")

    def is_ready(self):
        return True


class _ExplodingLLM(_ScriptedLLM):
    def complete_detailed(self, **kw):
        raise AssertionError("classic path must not be called")


def _analyze_env():
    env = CognitionEnvironment(customer_id="c")
    env.plan = Plan(steps=[
        PlanStep(id=1, action=StepAction.ANALYZE, description="a")])
    return env


def _flag(monkeypatch, on: bool):
    import crystal_cache.config as config_mod
    monkeypatch.setattr(
        config_mod, "get_settings",
        lambda: SimpleNamespace(cognition_agentic_workers=on),
    )


async def test_flag_off_classic_path_unchanged(monkeypatch):
    from crystal_cache.llm import reset_llm_client, set_llm_client
    _flag(monkeypatch, False)
    env = _analyze_env()
    fake = _ScriptedLLM()
    set_llm_client(fake)
    try:
        result = await _worker_llm_step(
            env, env.plan.steps[0],
            StepOutput(step_id=1, action="analyze",
                       status=StepStatus.RUNNING),
        )
    finally:
        reset_llm_client()
    assert result.status == StepStatus.COMPLETE
    assert result.output["content"] == "classic text"
    assert "agentic" not in result.output
    assert fake.calls == 1


async def test_flag_on_agentic_output_wins(monkeypatch):
    from crystal_cache.llm import reset_llm_client, set_llm_client
    _flag(monkeypatch, True)

    async def fake_agentic(*, env, step, prompt, store, fact_store,
                           encoder):
        assert "executing step 1" in prompt or prompt  # prompt threaded
        return {"content": "AGENTIC OUTPUT", "tool_calls": [
            {"tool": "web_fetch", "iteration": 1}], "iterations": 2,
            "model": "claude-x", "stop_reason": "end_turn"}

    monkeypatch.setattr(agentic_mod, "run_agentic_composition",
                        fake_agentic)
    env = _analyze_env()
    set_llm_client(_ExplodingLLM())
    try:
        result = await _worker_llm_step(
            env, env.plan.steps[0],
            StepOutput(step_id=1, action="analyze",
                       status=StepStatus.RUNNING),
        )
    finally:
        reset_llm_client()
    assert result.status == StepStatus.COMPLETE
    assert result.output["content"] == "AGENTIC OUTPUT"
    assert result.output["agentic"] is True
    assert result.output["tool_calls"][0]["tool"] == "web_fetch"
    assert result.model_used == "claude-x"


async def test_agentic_failure_falls_back_to_classic(monkeypatch):
    from crystal_cache.llm import reset_llm_client, set_llm_client
    _flag(monkeypatch, True)

    async def broken(*, env, step, prompt, store, fact_store, encoder):
        raise RuntimeError("agent loop exploded")

    monkeypatch.setattr(agentic_mod, "run_agentic_composition", broken)
    env = _analyze_env()
    fake = _ScriptedLLM()
    set_llm_client(fake)
    try:
        result = await _worker_llm_step(
            env, env.plan.steps[0],
            StepOutput(step_id=1, action="analyze",
                       status=StepStatus.RUNNING),
        )
    finally:
        reset_llm_client()
    assert result.status == StepStatus.COMPLETE
    assert result.output["content"] == "classic text"
    assert "agentic" not in result.output
    assert fake.calls == 1


async def test_agentic_empty_output_falls_back(monkeypatch):
    from crystal_cache.llm import reset_llm_client, set_llm_client
    _flag(monkeypatch, True)

    async def empty(*, env, step, prompt, store, fact_store, encoder):
        return {"content": "   ", "tool_calls": [], "iterations": 1,
                "model": "m", "stop_reason": "end_turn"}

    monkeypatch.setattr(agentic_mod, "run_agentic_composition", empty)
    env = _analyze_env()
    fake = _ScriptedLLM()
    set_llm_client(fake)
    try:
        result = await _worker_llm_step(
            env, env.plan.steps[0],
            StepOutput(step_id=1, action="analyze",
                       status=StepStatus.RUNNING),
        )
    finally:
        reset_llm_client()
    assert result.output["content"] == "classic text"
    assert fake.calls == 1


# --- run_agentic_composition itself ------------------------------------------

class _FakeAgent:
    ctor_kwargs: dict = {}
    run_kwargs: dict = {}
    run_result = {
        "final_text": "final composed text",
        "tool_calls": [{"tool_name": "web_search",
                        "input": {"queries": ["q"]},
                        "output": {"results_count": 3, "findings": ["x" * 900]},
                        "iteration": 1}],
        "iterations": 3,
        "model": "claude-agent",
        "stop_reason": "end_turn",
        "prompt_tokens": 111,
        "completion_tokens": 42,
        "cache_read_tokens": 7,
        "cache_creation_tokens": 3,
    }

    def __init__(self, **kwargs):
        type(self).ctor_kwargs = kwargs

    async def run(self, *, messages, system=None,
                  extra_system_context=None):
        type(self).run_kwargs = {"messages": messages, "system": system}
        return dict(type(self).run_result)


async def test_run_agentic_composition_caps_metering_and_trace(monkeypatch):
    import crystal_cache.agent.agent as agent_pkg
    monkeypatch.setattr(agent_pkg, "Agent", _FakeAgent)

    metered = []

    async def fake_meter(**kw):
        metered.append(kw)

    monkeypatch.setattr(agentic_mod, "record_model_call", fake_meter)

    from crystal_cache.llm import reset_llm_client, set_llm_client
    set_llm_client(_ScriptedLLM())
    env = _analyze_env()
    try:
        out = await run_agentic_composition(
            env=env, step=env.plan.steps[0], prompt="THE STEP PROMPT",
            store=None, fact_store=None, encoder=None,
        )
    finally:
        reset_llm_client()

    # Q3A caps + scoped registry + flat output budget.
    ctor = _FakeAgent.ctor_kwargs
    assert ctor["max_iterations"] == _AGENTIC_MAX_TOOL_CALLS
    assert ctor["max_tokens"] == _COMPOSITION_MAX_TOKENS
    assert set(ctor["registry"]._tools.keys()) == _READ_VERBS
    assert ctor["customer"].id == "c"

    # The charter is the system prompt; the step prompt is the message.
    system = _FakeAgent.run_kwargs["system"]
    assert "READ-ONLY" in system
    assert "TODAY'S DATE IS" in system
    assert "404" in system
    assert _FakeAgent.run_kwargs["messages"][0]["content"] == "THE STEP PROMPT"

    # Output shape + trimmed trace.
    assert out["content"] == "final composed text"
    assert out["iterations"] == 3
    assert len(out["tool_calls"]) == 1
    assert out["tool_calls"][0]["tool"] == "web_search"
    # 4K cap (was 500) — the Evidence Bench's zero-truncation
    # reading pane needs real outputs in the snapshot (2026-07-15).
    assert len(out["tool_calls"][0]["output_head"]) <= 4000

    # One aggregated meter row + env totals.
    assert len(metered) == 1
    assert metered[0]["input_tokens"] == 111
    assert metered[0]["output_tokens"] == 42
    assert metered[0]["origin"] == "cognition"
    assert env.tokens_used == 111 + 42


async def test_wall_clock_cap_raises_timeout(monkeypatch):
    import crystal_cache.agent.agent as agent_pkg

    class _SlowAgent(_FakeAgent):
        async def run(self, *, messages, system=None,
                      extra_system_context=None):
            await asyncio.sleep(0.2)
            return dict(type(self).run_result)

    monkeypatch.setattr(agent_pkg, "Agent", _SlowAgent)
    monkeypatch.setattr(agentic_mod, "_AGENTIC_WALL_SECONDS", 0.01)

    from crystal_cache.llm import reset_llm_client, set_llm_client
    set_llm_client(_ScriptedLLM())
    env = _analyze_env()
    try:
        with pytest.raises(asyncio.TimeoutError):
            await run_agentic_composition(
                env=env, step=env.plan.steps[0], prompt="p",
                store=None, fact_store=None, encoder=None,
            )
    finally:
        reset_llm_client()


# ---------------------------------------------------------------------------
# Research action + identity discipline (ratified 2026-07-14 after the
# quiet failures of rematch #10: wrong-but-existing repos return rich
# 200s; verification capacity must scale with target count; the
# orchestrator directs, it does not fabricate URLs)
# ---------------------------------------------------------------------------

from crystal_cache.cognition.agentic import run_research_step
from crystal_cache.cognition.retrieval_adapter import _render_github_json


def test_github_repo_data_carries_identity_stamp():
    text = _render_github_json("repo", {
        "full_name": "openai/whisper",
        "html_url": "https://github.com/openai/whisper",
        "description": "Robust Speech Recognition",
        "stargazers_count": 104931, "forks_count": 12755,
        "open_issues_count": 134, "pushed_at": "2026-04-15",
        "created_at": "2022-09-16", "default_branch": "main",
    })
    first_line = text.split("\n")[0]
    assert first_line.startswith("FETCHED REPOSITORY: openai/whisper")
    assert "verify" in first_line


async def test_research_step_agentic_path(monkeypatch):
    _flag(monkeypatch, True)
    captured = {}

    async def fake_session(*, env, step, prompt, store, fact_store,
                           encoder, system=""):
        captured["prompt"] = prompt
        captured["system"] = system
        return {"content": "VERIFIED: WhisperX is m-bain/whisperX ...",
                "tool_calls": [{"tool": "web_fetch", "iteration": 1}],
                "iterations": 2, "model": "m", "stop_reason": "end_turn"}

    monkeypatch.setattr(agentic_mod, "run_agentic_composition",
                        fake_session)
    env = _analyze_env()
    step = PlanStep(id=2, action=StepAction.RESEARCH, description="verify",
                    input={"targets": [
                        "WhisperX — latest version and release date",
                        "PySceneDetect — canonical repo and last release",
                    ]})
    out = await run_research_step(env=env, step=step, store=None,
                                  fact_store=None, encoder=None)
    # Findings-shaped: grounding + carryover treat it as evidence.
    assert out["results_count"] == 2
    assert out["findings"][0]["content"].startswith("VERIFIED")
    assert out["content_text"].startswith("VERIFIED")
    assert out["agentic"] is True
    # Targets reached the session prompt; the research charter was used.
    assert "WhisperX — latest version" in captured["prompt"]
    assert "CONFIRM IDENTITY" in captured["system"]
    assert "READ-ONLY" in captured["system"]


async def test_research_step_degrades_without_flag(monkeypatch):
    _flag(monkeypatch, False)
    from crystal_cache.cognition import retrieval_adapter as ra
    calls = []

    async def fake_dispatch(*, action_value, step_input, customer_id,
                            store, fact_store, encoder):
        calls.append((action_value, step_input))
        return {"findings": [{"title": "t", "url": "u", "content": "c"}],
                "results_count": 1}

    monkeypatch.setattr(ra, "dispatch_cognition_retrieval", fake_dispatch)
    env = _analyze_env()
    step = PlanStep(id=2, action=StepAction.RESEARCH, description="verify",
                    input={"targets": ["WhisperX — latest version"]})
    out = await run_research_step(env=env, step=step, store=None,
                                  fact_store=None, encoder=None)
    assert calls[0][0] == "web_search"
    assert calls[0][1]["queries"] == ["WhisperX"]
    assert out["degraded"] == "research_without_agentic_flag"


async def test_run_worker_routes_research(monkeypatch):
    _flag(monkeypatch, True)

    async def fake_research(*, env, step, store, fact_store, encoder):
        return {"content_text": "verified dossier", "findings": [],
                "results_count": 1, "agentic": True, "tool_calls": []}

    monkeypatch.setattr(agentic_mod, "run_research_step", fake_research)
    from crystal_cache.cognition.roles import run_worker
    env = _analyze_env()
    step = PlanStep(id=3, action=StepAction.RESEARCH, description="d",
                    input={"targets": ["X — version"]})
    result = await run_worker(env, step, store=None, fact_store=None,
                              encoder=None)
    assert result.status == StepStatus.COMPLETE
    assert result.output["content_text"] == "verified dossier"


async def test_orchestrator_prompt_discipline(monkeypatch):
    async def fake_source(env, store, fact_store, encoder):
        return []

    monkeypatch.setattr(roles_mod, "_source_bank_findings", fake_source)
    from crystal_cache.llm import reset_llm_client, set_llm_client

    class _Capture:
        def __init__(self):
            self.prompts = []

        def complete_detailed(self, *, system, messages, max_tokens,
                              temperature=1.0, tier="small", model=None,
                              json_schema=None):
            self.prompts.append(messages[0]["content"])
            from crystal_cache.llm.client import LLMResult
            import json as _json
            return LLMResult(text=_json.dumps({
                "goal": {"title": "t", "description": "d",
                         "acceptance_criteria": ["c"],
                         "output_type": "report"},
                "plan": {"reasoning": "r", "steps": [
                    {"id": 1, "action": "analyze", "description": "a",
                     "input": {}, "depends_on": [],
                     "parallel_group": None}],
                    "expected_output": "o", "suggested_key": "k",
                    "parent_crystal_id": "", "retry_route": "",
                    "bank_finding_ids": []},
            }), model="fake", input_tokens=1, output_tokens=1)

        def is_ready(self):
            return True

    # Flag ON: research advertised + discipline rules present.
    _flag(monkeypatch, True)
    env = CognitionEnvironment(customer_id="c", task_goal="g")
    fake = _Capture()
    set_llm_client(fake)
    try:
        await roles_mod.run_orchestrator(env=env, store=None,
                                         fact_store=None)
    finally:
        reset_llm_client()
    prompt = fake.prompts[0]
    assert "- research:" in prompt
    assert "|research|" in prompt
    assert "URL DISCIPLINE" in prompt
    assert "copied VERBATIM" in prompt
    assert "VALIDATOR ALIGNMENT" in prompt
    assert "CONFIRMS IDENTITY" in prompt

    # Flag OFF: research absent; discipline rules remain.
    _flag(monkeypatch, False)
    env2 = CognitionEnvironment(customer_id="c", task_goal="g")
    fake2 = _Capture()
    set_llm_client(fake2)
    try:
        await roles_mod.run_orchestrator(env=env2, store=None,
                                         fact_store=None)
    finally:
        reset_llm_client()
    prompt2 = fake2.prompts[0]
    assert "- research:" not in prompt2
    assert "|research|" not in prompt2
    assert "URL DISCIPLINE" in prompt2


async def test_composition_prompt_carries_identity_rule():
    from crystal_cache.llm import reset_llm_client, set_llm_client

    class _One(_ScriptedLLM):
        def __init__(self):
            super().__init__("out")
            self.prompts = []

        def complete_detailed(self, **kw):
            self.prompts.append(kw["messages"][0]["content"])
            return super().complete_detailed(**kw)

    env = _analyze_env()
    fake = _One()
    set_llm_client(fake)
    try:
        await _worker_llm_step(
            env, env.plan.steps[0],
            StepOutput(step_id=1, action="analyze",
                       status=StepStatus.RUNNING),
        )
    finally:
        reset_llm_client()
    assert "FETCHED REPOSITORY" in fake.prompts[0]
    assert "WRONG thing" in fake.prompts[0]
