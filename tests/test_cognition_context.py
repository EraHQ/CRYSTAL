"""Bank relevance gate + fair-share composition context (2026-07-11).

Rematch #4 evidence: an off-topic bank prefired 20 Python-tutorial facts
into a video-infrastructure research task — ~10K chars of noise that (a)
ate the head of the dependency-ordered 16K context window, truncating the
REAL web findings (the emerging-projects data the verdict said was never
found) to zero, and (b) counted as C2 grounding so the answerability park
could not fire. Ratified fixes pinned here:

  - COGNITION_BANK_RELEVANCE_FLOOR: all-or-nothing gate on the tools'
    top_score in the registry adapter — sub-floor bank results contribute
    NOTHING (zero findings, zero grounding); missing top_score (legacy
    fakes) is not gated. Per-fact floor on the v1 fallback helper, which
    still has per-fact scores.
  - _fair_share_allocations + _assemble_prior_context: max-min fair
    split of _PRIOR_CONTEXT_MAX_CHARS across carried findings and
    dependency blocks — no part starves another.

R14: verified by pytest.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from crystal_cache.cognition.engine import _retrieval_grounding_count
from crystal_cache.cognition.models import (
    CognitionEnvironment,
    Plan,
    PlanStep,
    StepAction,
    StepOutput,
    StepStatus,
)
from crystal_cache.cognition.retrieval_adapter import (
    COGNITION_BANK_RELEVANCE_FLOOR,
    _do_crystal_search,
)
from crystal_cache.cognition.roles import (
    _PRIOR_CONTEXT_MAX_CHARS,
    _assemble_prior_context,
    _fair_share_allocations,
    _worker_crystal_search,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeRegistry:
    """Registry whose tools return scripted matched ids + top_score."""

    def __init__(self, tool_output: dict):
        self._out = tool_output

    def get(self, name):
        out = self._out

        async def impl(**kwargs):
            return out

        return SimpleNamespace(impl=impl)


class _FakeStore:
    """Hydration source: one crystal with on-topic facts."""

    async def list_facts_for_crystal(self, cid):
        return [
            SimpleNamespace(
                id="fact_1", prompt_text="Video|FFmpeg",
                claim_text="FFmpeg 8.1.2 released", answer_value=None,
                pair_type="content_chunk",
            ),
        ]


# ---------------------------------------------------------------------------
# Registry-path gate (all-or-nothing on top_score)
# ---------------------------------------------------------------------------

async def test_adapter_gates_subfloor_bank_results_to_nothing():
    registry = _FakeRegistry({
        "matched_fact_ids": ["fact_1"],
        "matched_crystal_ids": ["crys_1"],
        "top_score": COGNITION_BANK_RELEVANCE_FLOOR - 0.15,
    })
    out = await _do_crystal_search(
        registry, _FakeStore(), "cus_t", {"query": "video editing 2026"},
    )
    assert out["findings"] == []
    assert out["results_count"] == 0
    assert out["content_text"] == ""
    assert "content" not in out
    assert "floor" in out["note"]
    assert out["gated_top_score"] == pytest.approx(
        COGNITION_BANK_RELEVANCE_FLOOR - 0.15, abs=1e-4)
    # The gated shape is ZERO grounding for the C2 answerability park —
    # off-topic noise no longer fakes "the bank has material".
    assert _retrieval_grounding_count(out) == 0


async def test_adapter_passes_results_at_or_above_floor():
    registry = _FakeRegistry({
        "matched_fact_ids": ["fact_1"],
        "matched_crystal_ids": ["crys_1"],
        "top_score": COGNITION_BANK_RELEVANCE_FLOOR + 0.2,
    })
    out = await _do_crystal_search(
        registry, _FakeStore(), "cus_t", {"query": "FFmpeg release"},
    )
    assert out["results_count"] == 1
    assert out["findings"][0]["fact_id"] == "fact_1"
    assert "FFmpeg" in out["content_text"]
    assert _retrieval_grounding_count(out) == 1


async def test_adapter_does_not_gate_when_top_score_missing():
    """Legacy fakes / older tool outputs without top_score pass through."""
    registry = _FakeRegistry({
        "matched_fact_ids": ["fact_1"],
        "matched_crystal_ids": ["crys_1"],
    })
    out = await _do_crystal_search(
        registry, _FakeStore(), "cus_t", {"query": "anything"},
    )
    assert out["results_count"] == 1


# ---------------------------------------------------------------------------
# Fallback-helper gate (per-fact scores available on this path)
# ---------------------------------------------------------------------------

class _ScoredFactStore:
    def __init__(self, scored):
        self._scored = scored

    async def search(self, *, customer_id, query_vector, pair_types, k):
        return self._scored


class _FactRowStore:
    async def list_facts_for_crystal(self, cid):
        return [
            SimpleNamespace(id=f"fact_{i}", prompt_text=f"Key|{i}",
                            claim_text=f"claim {i}", answer_value=None)
            for i in range(4)
        ]


async def test_fallback_helper_drops_subfloor_facts(monkeypatch):
    import crystal_cache.cognition.roles as roles_mod

    async def fake_encode(encoder, text):
        return [0.0]

    monkeypatch.setattr(roles_mod, "encode_native_async", fake_encode)
    lo = COGNITION_BANK_RELEVANCE_FLOOR - 0.2
    hi = COGNITION_BANK_RELEVANCE_FLOOR + 0.2
    store = _FactRowStore()
    fact_store = _ScoredFactStore([
        ("fact_0", "crys_a", "content_chunk", hi),
        ("fact_1", "crys_a", "content_chunk", lo),
        ("fact_2", "crys_a", "content_chunk", lo),
    ])
    env = CognitionEnvironment(customer_id="cus_t")
    step = PlanStep(id=1, action=StepAction.CRYSTAL_SEARCH, description="s",
                    input={"query": "video"})
    result = await _worker_crystal_search(
        env, step,
        StepOutput(step_id=1, action="crystal_search",
                   status=StepStatus.RUNNING),
        store, fact_store, encoder=None,
    )
    ids = [f["fact_id"] for f in result.output["findings"]]
    assert ids == ["fact_0"]


async def test_fallback_helper_all_subfloor_yields_empty(monkeypatch):
    import crystal_cache.cognition.roles as roles_mod

    async def fake_encode(encoder, text):
        return [0.0]

    monkeypatch.setattr(roles_mod, "encode_native_async", fake_encode)
    lo = COGNITION_BANK_RELEVANCE_FLOOR - 0.25
    fact_store = _ScoredFactStore([
        ("fact_0", "crys_a", "content_chunk", lo),
        ("fact_1", "crys_a", "content_chunk", lo),
    ])
    env = CognitionEnvironment(customer_id="cus_t")
    step = PlanStep(id=1, action=StepAction.CRYSTAL_SEARCH, description="s",
                    input={"query": "video"})
    result = await _worker_crystal_search(
        env, step,
        StepOutput(step_id=1, action="crystal_search",
                   status=StepStatus.RUNNING),
        _FactRowStore(), fact_store, encoder=None,
    )
    assert result.output["findings"] == []
    assert _retrieval_grounding_count(result.output) == 0


# ---------------------------------------------------------------------------
# Fair-share allocator
# ---------------------------------------------------------------------------

def test_fair_share_equal_split_of_oversized_parts():
    # The rematch shape: three big parts, none may starve another.
    allocs = _fair_share_allocations([30000, 30000, 30000], 48000)
    assert sum(allocs) <= 48000
    assert all(a == 16000 for a in allocs)


def test_fair_share_small_parts_keep_everything():
    allocs = _fair_share_allocations([1000, 1000, 60000], 48000)
    assert allocs[0] == 1000
    assert allocs[1] == 1000
    assert allocs[2] == 46000
    assert sum(allocs) <= 48000


def test_fair_share_under_budget_is_untouched():
    sizes = [500, 700, 900]
    assert _fair_share_allocations(sizes, 48000) == sizes


def test_fair_share_empty_and_zero_budget():
    assert _fair_share_allocations([], 100) == []
    assert _fair_share_allocations([10, 10], 0) == [0, 0]


# ---------------------------------------------------------------------------
# _assemble_prior_context under pressure (the exact rematch failure)
# ---------------------------------------------------------------------------

def _env_with_three_deps() -> tuple[CognitionEnvironment, PlanStep]:
    env = CognitionEnvironment(customer_id="cus_t")
    env.plan = Plan(steps=[
        PlanStep(id=1, action=StepAction.CRYSTAL_SEARCH, description="bank"),
        PlanStep(id=2, action=StepAction.WEB_SEARCH, description="ffmpeg"),
        PlanStep(id=3, action=StepAction.WEB_SEARCH, description="projects"),
        PlanStep(id=4, action=StepAction.ANALYZE, description="analyze",
                 depends_on=[1, 2, 3]),
    ])
    env.step_outputs[1] = StepOutput(
        step_id=1, action="crystal_search", status=StepStatus.COMPLETE,
        output={"content_text": "NOISE " * 4000},   # ~24K chars
    )
    env.step_outputs[2] = StepOutput(
        step_id=2, action="web_search", status=StepStatus.COMPLETE,
        output={"content_text": "FFMPEG-8.1.2 " + ("f" * 24000)},
    )
    env.step_outputs[3] = StepOutput(
        step_id=3, action="web_search", status=StepStatus.COMPLETE,
        output={"content_text": "VIMAX-EMERGING " + ("p" * 24000)},
    )
    return env, env.plan.steps[3]


def test_context_no_dependency_starves_another():
    env, step = _env_with_three_deps()
    ctx = _assemble_prior_context(env, step)
    # The rematch bug: step 3 truncated to zero. Every dependency's HEAD
    # must survive — that's where titles/URLs/versions live.
    assert "NOISE" in ctx
    assert "FFMPEG-8.1.2" in ctx
    assert "VIMAX-EMERGING" in ctx
    assert len(ctx) <= _PRIOR_CONTEXT_MAX_CHARS + 200  # header slack


def test_context_carried_findings_share_fairly_with_deps():
    env, step = _env_with_three_deps()
    env.carried_findings = [{
        "attempt": 1, "step_id": 9, "action": "web_search",
        "description": "carried", "text": "CARRIED-EVIDENCE " + ("c" * 30000),
    }]
    ctx = _assemble_prior_context(env, step)
    assert "CARRIED-EVIDENCE" in ctx
    assert "VIMAX-EMERGING" in ctx
    assert len(ctx) <= _PRIOR_CONTEXT_MAX_CHARS + 200


def test_context_failed_dep_marker_survives_pressure():
    env, step = _env_with_three_deps()
    env.step_outputs[2] = StepOutput(
        step_id=2, action="web_search", status=StepStatus.FAILED,
        output={}, error="timeout",
    )
    ctx = _assemble_prior_context(env, step)
    assert "Step 2 FAILED: timeout" in ctx
    assert "VIMAX-EMERGING" in ctx


# ---------------------------------------------------------------------------
# Orchestrator bank sourcing + curation (2026-07-11, Q1A/Q2A/Q3A)
# ---------------------------------------------------------------------------

import json as _json

from crystal_cache.cognition import roles as _roles_mod
from crystal_cache.cognition.roles import run_orchestrator
from crystal_cache.llm import reset_llm_client, set_llm_client
from crystal_cache.llm.client import LLMResult


class _PromptCaptureLLM:
    def __init__(self, text: str):
        self._text = text
        self.prompts: list[str] = []

    def complete_detailed(self, *, system, messages, max_tokens,
                          temperature=1.0, tier="small", model=None,
                          json_schema=None) -> LLMResult:
        self.prompts.append(messages[-1]["content"])
        return LLMResult(text=self._text, model="fake",
                         input_tokens=1, output_tokens=1)

    def is_ready(self) -> bool:
        return True


def _orch_json(bank_ids: list[str]) -> str:
    return _json.dumps({
        "goal": {"title": "T", "description": "D",
                 "acceptance_criteria": ["c"]},
        "plan": {
            "reasoning": "r",
            "steps": [{"id": 1, "action": "analyze", "description": "a",
                       "input": {}, "depends_on": [],
                       "parallel_group": None}],
            "expected_output": "o", "suggested_key": "k",
            "parent_crystal_id": "", "retry_route": "",
            "max_output_tokens": 0,
            "bank_finding_ids": bank_ids,
        },
    })


_SOURCED = [
    {"fact_id": "fact_vid", "crystal_id": "c1", "key": "Video|FFmpeg",
     "content": "FFmpeg 8.1.2 shipped", "pair_type": "content_chunk"},
    {"fact_id": "fact_py", "crystal_id": "c2", "key": "Python|PEP8",
     "content": "spaces not tabs", "pair_type": "content_chunk"},
]


async def test_orchestrator_curates_sourced_findings_onto_plan(monkeypatch):
    async def fake_source(env, store, fact_store, encoder):
        return list(_SOURCED)

    monkeypatch.setattr(_roles_mod, "_source_bank_findings", fake_source)
    env = CognitionEnvironment(customer_id="c", task_goal="video research")
    fake = _PromptCaptureLLM(_orch_json(["fact_vid", "fact_unknown"]))
    set_llm_client(fake)
    try:
        _, plan = await run_orchestrator(env=env, store=None,
                                         fact_store=None)
    finally:
        reset_llm_client()
    prompt = fake.prompts[0]
    # The orchestrator SAW the sourced material and the curation ask.
    assert "BANK MATERIAL" in prompt
    assert "FFmpeg 8.1.2" in prompt
    assert "bank_finding_ids" in prompt
    # The blind mandatory-first-step rule is gone.
    assert "always comes first" not in prompt
    # Curation: only the sourced id it named rides; unknown ids ignored.
    assert [f["fact_id"] for f in plan.bank_findings] == ["fact_vid"]


async def test_orchestrator_empty_sourcing_states_no_material(monkeypatch):
    async def fake_source(env, store, fact_store, encoder):
        return []

    monkeypatch.setattr(_roles_mod, "_source_bank_findings", fake_source)
    env = CognitionEnvironment(customer_id="c", task_goal="video research")
    fake = _PromptCaptureLLM(_orch_json([]))
    set_llm_client(fake)
    try:
        _, plan = await run_orchestrator(env=env, store=None,
                                         fact_store=None)
    finally:
        reset_llm_client()
    assert "found no material" in fake.prompts[0]
    assert plan.bank_findings == []


async def test_sourcing_failure_never_blocks_planning():
    """store=None (isolated tests / degraded deploys) → empty findings,
    planning proceeds. _source_bank_findings never raises."""
    from crystal_cache.cognition.roles import _source_bank_findings
    env = CognitionEnvironment(customer_id="c", task_goal="anything")
    assert await _source_bank_findings(env, None, None, None) == []


def test_context_includes_plan_bank_findings_under_fair_share():
    env, step = _env_with_three_deps()
    env.plan.bank_findings = [{
        "fact_id": "f1", "crystal_id": "c1", "key": "Video|Prior",
        "content": "BANK-PRIOR-RESEARCH " + ("b" * 30000),
        "pair_type": "content_chunk",
    }]
    ctx = _assemble_prior_context(env, step)
    assert "Bank finding (Video|Prior)" in ctx
    assert "BANK-PRIOR-RESEARCH" in ctx
    assert "VIMAX-EMERGING" in ctx  # deps still not starved
    assert len(ctx) <= _PRIOR_CONTEXT_MAX_CHARS + 250


def test_c2_park_disarmed_by_plan_bank_findings():
    from crystal_cache.cognition.engine import _should_park_unanswerable
    env = CognitionEnvironment(customer_id="c")
    plan = Plan(steps=[
        PlanStep(id=1, action=StepAction.WEB_SEARCH, description="w"),
        PlanStep(id=2, action=StepAction.FORMAT, description="f",
                 depends_on=[1]),
    ], bank_findings=[{"fact_id": "f1", "content": "bank material"}])
    env.plan = plan
    env.step_outputs[1] = StepOutput(
        step_id=1, action="web_search", status=StepStatus.COMPLETE,
        output={"findings": [], "content_text": ""},
    )
    # Web found nothing, composition remains — but curated bank material
    # rides the plan, so composition is grounded: no park.
    assert _should_park_unanswerable(plan, {1}, env) is False


def test_plan_to_dict_truncates_bank_findings():
    plan = Plan(bank_findings=[{
        "fact_id": "f1", "crystal_id": "c1", "key": "K",
        "content": "x" * 2000, "pair_type": "content_chunk",
    }])
    d = plan.to_dict()
    assert len(d["bank_findings"][0]["content"]) == 300


# ---------------------------------------------------------------------------
# Web query hygiene: keywordize + zero-result retry (2026-07-11, Q1C)
# ---------------------------------------------------------------------------

from crystal_cache.cognition.retrieval_adapter import _keywordize


def test_keywordize_strips_instruction_prose():
    q = ("Extract WhisperX release data: latest stable version, recent "
         "releases (last 6 months), changelog, and commit activity from "
         "GitHub API endpoints.")
    out = _keywordize(q)
    assert "extract" not in out
    assert "whisperx" in out
    assert "changelog" in out
    assert len(out.split()) <= 8


def test_keywordize_leaves_good_queries_semantically_intact():
    assert _keywordize("WhisperX latest release changelog") == \
        "whisperx latest release changelog"


class _ScriptedWebRegistry:
    """Web tool that returns nothing for prose, results for keywords."""

    def __init__(self):
        self.queries: list[str] = []

    def get_by_cognition_action(self, action):
        outer = self

        async def impl(*, customer_id, query):
            outer.queries.append(query)
            if len(query.split()) > 8:
                return {"query": query, "results": []}
            return {"query": query,
                    "results": [{"title": "t", "url": "u", "snippet": "s"}]}

        return SimpleNamespace(impl=impl, contexts={"cognition"})


async def test_web_search_zero_results_retries_keywordized(monkeypatch):
    from crystal_cache.cognition import retrieval_adapter as ra
    registry = _ScriptedWebRegistry()
    monkeypatch.setattr(ra, "_load_registry",
                        lambda store, fact_store, encoder: registry)
    out = await ra.dispatch_cognition_retrieval(
        action_value="web_search",
        step_input={"query": (
            "Extract WhisperX release data: latest stable version, recent "
            "releases, changelog, and commit activity from GitHub API "
            "endpoints please and thank you")},
        customer_id="c", store=object(), fact_store=None, encoder=None,
    )
    assert len(registry.queries) == 2          # original + one retry
    assert len(registry.queries[1].split()) <= 8
    assert out["findings"]                     # retry found results
    assert out["retried_query"] == registry.queries[1]
    assert "original_query" in out


async def test_web_search_good_query_not_retried(monkeypatch):
    from crystal_cache.cognition import retrieval_adapter as ra
    registry = _ScriptedWebRegistry()
    monkeypatch.setattr(ra, "_load_registry",
                        lambda store, fact_store, encoder: registry)
    out = await ra.dispatch_cognition_retrieval(
        action_value="web_search",
        step_input={"query": "WhisperX latest release changelog"},
        customer_id="c", store=object(), fact_store=None, encoder=None,
    )
    assert len(registry.queries) == 1
    assert out["findings"]


async def test_orchestrator_prompt_carries_query_rule(monkeypatch):
    async def fake_source(env, store, fact_store, encoder):
        return []

    monkeypatch.setattr(_roles_mod, "_source_bank_findings", fake_source)
    env = CognitionEnvironment(customer_id="c", task_goal="research")
    fake = _PromptCaptureLLM(_orch_json([]))
    set_llm_client(fake)
    try:
        await run_orchestrator(env=env, store=None, fact_store=None)
    finally:
        reset_llm_client()
    prompt = fake.prompts[0]
    assert "SEARCH-ENGINE KEYWORD QUERY" in prompt
    assert "GOOD:" in prompt and "BAD:" in prompt
