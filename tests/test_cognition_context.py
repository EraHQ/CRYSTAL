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
    assert "SEARCH-ENGINE KEYWORD QUERIES" in prompt
    assert '"queries"' in prompt
    assert "GOOD:" in prompt and "BAD:" in prompt


# ---------------------------------------------------------------------------
# Batch tool language + continuation + composition retry (2026-07-11)
# ---------------------------------------------------------------------------

from crystal_cache.cognition.retrieval_adapter import _WEB_BATCH_MAX_QUERIES
from crystal_cache.cognition.roles import (
    _COMPOSITION_MAX_CONTINUATIONS,
)
from crystal_cache.llm.client import LLMResult as _LLMResult


class _BatchWebRegistry:
    """Web tool returning one distinct finding per query."""

    def __init__(self):
        self.queries: list[str] = []

    def get_by_cognition_action(self, action):
        outer = self

        async def impl(*, customer_id, query):
            outer.queries.append(query)
            return {"query": query,
                    "results": [{"title": f"t:{query}", "url": f"u:{query}",
                                 "snippet": f"s:{query}",
                                 "content": f"content for {query}"}]}

        return SimpleNamespace(impl=impl, contexts={"cognition"})


async def test_batch_queries_fan_out_and_merge(monkeypatch):
    from crystal_cache.cognition import retrieval_adapter as ra
    registry = _BatchWebRegistry()
    monkeypatch.setattr(ra, "_load_registry",
                        lambda store, fact_store, encoder: registry)
    out = await ra.dispatch_cognition_retrieval(
        action_value="web_search",
        step_input={"queries": ["whisperx release", "mlt release",
                                "otio release notes"]},
        customer_id="c", store=object(), fact_store=None, encoder=None,
    )
    assert sorted(registry.queries) == sorted(
        ["whisperx release", "mlt release", "otio release notes"])
    assert out["results_count"] == 3
    # Findings carry per-query provenance.
    assert {f["query"] for f in out["findings"]} == set(registry.queries)
    # Per-finding content survives the merge (the composer renders
    # findings via _finding_to_text; content_text is empty by design).
    assert all(f["content"].startswith("content for ") for f in out["findings"])
    assert out["per_query_counts"]["mlt release"] == 1


async def test_batch_caps_query_count(monkeypatch):
    from crystal_cache.cognition import retrieval_adapter as ra
    registry = _BatchWebRegistry()
    monkeypatch.setattr(ra, "_load_registry",
                        lambda store, fact_store, encoder: registry)
    out = await ra.dispatch_cognition_retrieval(
        action_value="web_search",
        step_input={"queries": [f"query number {i}" for i in range(9)]},
        customer_id="c", store=object(), fact_store=None, encoder=None,
    )
    assert len(registry.queries) == _WEB_BATCH_MAX_QUERIES
    assert len(out["queries"]) == _WEB_BATCH_MAX_QUERIES


async def test_single_query_form_still_works(monkeypatch):
    from crystal_cache.cognition import retrieval_adapter as ra
    registry = _BatchWebRegistry()
    monkeypatch.setattr(ra, "_load_registry",
                        lambda store, fact_store, encoder: registry)
    out = await ra.dispatch_cognition_retrieval(
        action_value="web_search",
        step_input={"query": "ffmpeg changelog"},
        customer_id="c", store=object(), fact_store=None, encoder=None,
    )
    assert registry.queries == ["ffmpeg changelog"]
    assert out["results_count"] == 1


class _StopReasonLLM:
    """Scripted (text, stop_reason) pairs; captures messages per call."""

    def __init__(self, script):
        self._script = list(script)
        self.calls: list[list[dict]] = []

    def complete_detailed(self, *, system, messages, max_tokens,
                          temperature=1.0, tier="small", model=None,
                          json_schema=None):
        self.calls.append(messages)
        text, stop = (self._script.pop(0) if self._script
                      else ("", "end_turn"))
        return _LLMResult(text=text, model="fake", input_tokens=5,
                          output_tokens=5, stop_reason=stop)

    def is_ready(self):
        return True


def _one_analyze_env():
    env = CognitionEnvironment(customer_id="c")
    env.plan = Plan(steps=[
        PlanStep(id=1, action=StepAction.ANALYZE, description="a")])
    return env


async def test_continuation_concatenates_on_max_tokens():
    from crystal_cache.llm import reset_llm_client, set_llm_client
    env = _one_analyze_env()
    fake = _StopReasonLLM([
        ("PART-ONE ", "max_tokens"),
        ("PART-TWO ", "max_tokens"),
        ("PART-THREE.", "end_turn"),
    ])
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
    assert result.output["content"] == "PART-ONE PART-TWO PART-THREE."
    assert len(fake.calls) == 3
    # Continuation calls carry the partial as an assistant turn.
    assert fake.calls[1][1]["role"] == "assistant"
    assert fake.calls[1][1]["content"] == "PART-ONE "
    # Token totals accumulate across every call.
    assert result.tokens_out == 15


async def test_continuation_capped():
    from crystal_cache.llm import reset_llm_client, set_llm_client
    env = _one_analyze_env()
    fake = _StopReasonLLM([("X", "max_tokens")] * 10)
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
    assert len(fake.calls) == 1 + _COMPOSITION_MAX_CONTINUATIONS
    assert result.output["content"] == "X" * (
        1 + _COMPOSITION_MAX_CONTINUATIONS)


async def test_empty_then_good_retries_to_complete():
    from crystal_cache.llm import reset_llm_client, set_llm_client
    env = _one_analyze_env()
    fake = _StopReasonLLM([("   ", "end_turn"), ("recovered", "end_turn")])
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
    assert result.output["content"] == "recovered"
    assert len(fake.calls) == 2


async def test_empty_twice_fails_the_step():
    from crystal_cache.llm import reset_llm_client, set_llm_client
    env = _one_analyze_env()
    fake = _StopReasonLLM([("", "end_turn"), ("  ", "end_turn")])
    set_llm_client(fake)
    try:
        result = await _worker_llm_step(
            env, env.plan.steps[0],
            StepOutput(step_id=1, action="analyze",
                       status=StepStatus.RUNNING),
        )
    finally:
        reset_llm_client()
    assert result.status == StepStatus.FAILED
    assert "empty" in result.error
    assert len(fake.calls) == 2


# ---------------------------------------------------------------------------
# web_fetch action + validator envelope loop (2026-07-13, rematch #7)
# ---------------------------------------------------------------------------

from crystal_cache.cognition.roles import (
    _VALIDATOR_DELIVERABLE_CHARS,
    run_validator,
)


async def test_web_fetch_batch_urls_through_fetch_pipeline(monkeypatch):
    from crystal_cache.cognition import retrieval_adapter as ra
    from crystal_cache.search import fetch as fetch_mod

    fetched: list[str] = []

    def fake_fill(payload, *, max_pages, content_cap, deadline_seconds=0.0,
                  render_enabled=False, render_timeout_seconds=20.0,
                  http_client=None, resolver=None):
        for r in payload["results"]:
            fetched.append(r["url"])
            r["content"] = f"page text for {r['url']}"
        return payload

    monkeypatch.setattr(fetch_mod, "fill_missing_content", fake_fill)
    out = await ra.dispatch_cognition_retrieval(
        action_value="web_fetch",
        step_input={"urls": [
            "https://ffmpeg.org/download.html",
            "https://www.scenedetect.com/changelog",
        ]},
        customer_id="c", store=None, fact_store=None, encoder=None,
    )
    assert len(fetched) == 2
    assert out["results_count"] == 2
    assert all(f["content"].startswith("page text for ") for f in out["findings"])
    # Counts as C2 grounding like any retrieval step.
    assert _retrieval_grounding_count(out) == 2


async def test_web_fetch_no_urls_is_explicit_empty():
    from crystal_cache.cognition import retrieval_adapter as ra
    out = await ra.dispatch_cognition_retrieval(
        action_value="web_fetch", step_input={},
        customer_id="c", store=None, fact_store=None, encoder=None,
    )
    assert out["results_count"] == 0
    assert "no urls" in out["note"]


async def test_orchestrator_prompt_offers_web_fetch(monkeypatch):
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
    assert "web_fetch" in prompt
    assert '"urls"' in prompt
    assert "Search is for discovery; fetch is for known sources." in prompt


def _validator_env(deliverable: str) -> CognitionEnvironment:
    env = CognitionEnvironment(customer_id="c")
    env.goal = GoalDocument(
        title="T", description="D",
        acceptance_criteria=["has section A", "has section B"],
    )
    env.deliverables["main"] = deliverable
    return env


_VERDICT_JSON = (
    '{"approved": true, "score": 0.9, "reasoning": "ok",'
    ' "criteria_evaluation": [], "issues": [], "suggestions": []}'
)


async def test_validator_small_deliverable_single_call():
    from crystal_cache.llm import reset_llm_client, set_llm_client
    env = _validator_env("SECTION A ... SECTION B ... complete")
    fake = _StopReasonLLM([(_VERDICT_JSON, "end_turn")])
    set_llm_client(fake)
    try:
        result = await run_validator(env=env)
    finally:
        reset_llm_client()
    assert result.approved is True
    assert len(fake.calls) == 1
    # The deliverable itself is in the prompt, no digest framing.
    assert "DIGEST OF PART" not in fake.calls[0][0]["content"]


async def test_validator_oversized_deliverable_uses_envelopes():
    from crystal_cache.llm import reset_llm_client, set_llm_client
    big = ("SECTION A " + "a" * _VALIDATOR_DELIVERABLE_CHARS
           + " SECTION B " + "b" * (_VALIDATOR_DELIVERABLE_CHARS // 2)
           + " SOURCES: everything cited. THE END")
    env = _validator_env(big)
    fake = _StopReasonLLM([
        ("Part 1: SECTION A present, evidence...", "end_turn"),
        ("Part 2: SECTION B present, SOURCES present, ends properly.",
         "end_turn"),
        (_VERDICT_JSON, "end_turn"),
    ])
    set_llm_client(fake)
    try:
        result = await run_validator(env=env)
    finally:
        reset_llm_client()
    assert result.approved is True
    # Two envelope digests + one verdict.
    assert len(fake.calls) == 3
    assert "PART 1 of 2" in fake.calls[0][0]["content"]
    assert "PART 2 of 2" in fake.calls[1][0]["content"]
    final = fake.calls[2][0]["content"]
    assert "DIGEST OF PART 1/2" in final
    assert "DIGEST OF PART 2/2" in final
    assert "Judge completeness" in final
    # The tail of the deliverable REACHED a digest call (the old bug:
    # everything past 24K was invisible to the validator).
    assert "THE END" in fake.calls[1][0]["content"]


# ---------------------------------------------------------------------------
# GitHub API routing + date grounding + render salvage (2026-07-13, run #8)
# ---------------------------------------------------------------------------

from crystal_cache.cognition.retrieval_adapter import (
    _github_api_targets,
    _render_github_json,
)


def test_github_url_maps_to_api_targets():
    targets = _github_api_targets("https://github.com/mltframework/mlt/releases")
    assert targets is not None
    labels = [t[0] for t in targets]
    assert labels == ["repo", "releases", "contributors"]
    assert targets[0][1] == "https://api.github.com/repos/mltframework/mlt"
    assert "per_page=5" in targets[1][1]
    # Non-github URLs pass through to the HTML pipeline.
    assert _github_api_targets("https://fossies.org/linux/ffmpeg/") is None
    assert _github_api_targets("https://ffmpeg.org/download.html") is None


def test_github_json_renders_citable_facts():
    releases = [{
        "tag_name": "v7.40.0", "name": "MLT 7.40.0",
        "published_at": "2026-06-14T10:00:00Z",
        "html_url": "https://github.com/mltframework/mlt/releases/tag/v7.40.0",
        "body": "Fixes and improvements",
    }]
    text = _render_github_json("releases", releases)
    assert "v7.40.0" in text
    assert "2026-06-14" in text
    assert "releases/tag/v7.40.0" in text
    repo = {"html_url": "https://github.com/x/y", "description": "d",
            "stargazers_count": 6000, "forks_count": 120,
            "open_issues_count": 5, "pushed_at": "2026-07-01T00:00:00Z",
            "created_at": "2025-08-01T00:00:00Z", "default_branch": "main"}
    text = _render_github_json("repo", repo)
    assert "Stars: 6000" in text
    assert "Last push: 2026-07-01" in text


async def test_web_fetch_routes_github_to_api(monkeypatch):
    from crystal_cache.cognition import retrieval_adapter as ra

    async def fake_api(url):
        if "github.com" in url:
            return {"title": f"GitHub API data for {url}", "url": url,
                    "content": "Stars: 6000", "source": "github_api"}
        return None

    monkeypatch.setattr(ra, "_fetch_github_via_api", fake_api)

    from crystal_cache.search import fetch as fetch_mod
    html_fetched: list[str] = []

    def fake_fill(payload, **kw):
        for r in payload["results"]:
            html_fetched.append(r["url"])
            r["content"] = f"html for {r['url']}"
        return payload

    monkeypatch.setattr(fetch_mod, "fill_missing_content", fake_fill)
    out = await ra.dispatch_cognition_retrieval(
        action_value="web_fetch",
        step_input={"urls": [
            "https://github.com/mltframework/mlt/releases",
            "https://ffmpeg.org/download.html",
        ]},
        customer_id="c", store=None, fact_store=None, encoder=None,
    )
    # github went to the API; the non-github URL went through HTML.
    assert html_fetched == ["https://ffmpeg.org/download.html"]
    assert out["results_count"] == 2
    sources = {f.get("source", "html") for f in out["findings"]}
    assert "github_api" in sources


async def test_web_fetch_all_github_skips_html_pipeline(monkeypatch):
    from crystal_cache.cognition import retrieval_adapter as ra

    async def fake_api(url):
        return {"title": "t", "url": url, "content": "Stars: 1",
                "source": "github_api"}

    monkeypatch.setattr(ra, "_fetch_github_via_api", fake_api)
    from crystal_cache.search import fetch as fetch_mod

    def boom(payload, **kw):
        raise AssertionError("HTML pipeline must not run")

    monkeypatch.setattr(fetch_mod, "fill_missing_content", boom)
    out = await ra.dispatch_cognition_retrieval(
        action_value="web_fetch",
        step_input={"urls": ["https://github.com/a/b"]},
        customer_id="c", store=None, fact_store=None, encoder=None,
    )
    assert out["results_count"] == 1


async def test_prompts_carry_todays_date(monkeypatch):
    async def fake_source(env, store, fact_store, encoder):
        return []

    monkeypatch.setattr(_roles_mod, "_source_bank_findings", fake_source)
    from crystal_cache.llm import reset_llm_client, set_llm_client
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    env = CognitionEnvironment(customer_id="c", task_goal="research")
    fake = _PromptCaptureLLM(_orch_json([]))
    set_llm_client(fake)
    try:
        await run_orchestrator(env=env, store=None, fact_store=None)
    finally:
        reset_llm_client()
    assert f"TODAY'S DATE IS {today}" in fake.prompts[0]

    venv = _validator_env("SECTION A SECTION B")
    vfake = _StopReasonLLM([(_VERDICT_JSON, "end_turn")])
    set_llm_client(vfake)
    try:
        await run_validator(env=venv)
    finally:
        reset_llm_client()
    assert f"TODAY'S DATE IS {today}" in vfake.calls[0][0]["content"]
    assert "not against your training data" in vfake.calls[0][0]["content"]


async def test_composer_prompt_carries_date_and_citation_rule():
    from crystal_cache.llm import reset_llm_client, set_llm_client
    env = _one_analyze_env()
    fake = _StopReasonLLM([("text", "end_turn")])
    set_llm_client(fake)
    try:
        await _worker_llm_step(
            env, env.plan.steps[0],
            StepOutput(step_id=1, action="analyze",
                       status=StepStatus.RUNNING),
        )
    finally:
        reset_llm_client()
    prompt = fake.calls[0][0]["content"]
    assert "TODAY'S DATE IS" in prompt
    assert "never internal step numbers" in prompt


# ---------------------------------------------------------------------------
# Q4A report crystallization + empty-output failure + budget floor
# ---------------------------------------------------------------------------

from crystal_cache.cognition.engine import _commit_deliverable_to_scratchpad
from crystal_cache.cognition.models import GoalDocument, OutputType
from crystal_cache.cognition.roles import (
    _worker_llm_step,
)


class _UploadCaptureStore:
    def __init__(self, fail=False):
        self.uploads: list[dict] = []
        self._fail = fail

    async def create_document_upload(self, *, customer_id, label, text,
                                     detected_type):
        if self._fail:
            raise RuntimeError("db down")
        self.uploads.append({"customer_id": customer_id, "label": label,
                             "text": text, "detected_type": detected_type})
        return SimpleNamespace(id=f"doc_{len(self.uploads)}")


def _approved_report_env(deliverable: str) -> CognitionEnvironment:
    env = CognitionEnvironment(customer_id="cus_t",
                               output_type=OutputType.REPORT)
    env.goal = GoalDocument(title="Video ecosystem report")
    env.plan = Plan(suggested_key="Video|Report")
    env.deliverables["main"] = deliverable
    return env


async def test_approved_report_commits_to_scratchpad():
    env = _approved_report_env("A substantial validated research report " * 5)
    store = _UploadCaptureStore()
    upload_id = await _commit_deliverable_to_scratchpad(
        env, store, env.deliverables["main"])
    assert upload_id == "doc_1"
    up = store.uploads[0]
    assert up["detected_type"] == "inferred_knowledge"
    assert "Video|Report" in up["label"]
    # Review-gated lane: the upload text is the full deliverable.
    assert up["text"] == env.deliverables["main"]


async def test_report_commit_failure_never_raises():
    env = _approved_report_env("A substantial validated research report " * 5)
    store = _UploadCaptureStore(fail=True)
    upload_id = await _commit_deliverable_to_scratchpad(
        env, store, env.deliverables["main"])
    assert upload_id is None  # logged, not raised — report still returns


async def test_trivial_deliverable_not_committed():
    env = _approved_report_env("too short")
    store = _UploadCaptureStore()
    assert await _commit_deliverable_to_scratchpad(env, store, "short") is None
    assert store.uploads == []


async def test_empty_model_output_fails_the_step():
    env = CognitionEnvironment(customer_id="c")
    env.plan = Plan(steps=[
        PlanStep(id=1, action=StepAction.SYNTHESIZE, description="s")])
    from crystal_cache.llm import reset_llm_client, set_llm_client
    fake = _PromptCaptureLLM("   ")  # whitespace-only "completion"
    set_llm_client(fake)
    try:
        result = await _worker_llm_step(
            env, env.plan.steps[0],
            StepOutput(step_id=1, action="synthesize",
                       status=StepStatus.RUNNING),
        )
    finally:
        reset_llm_client()
    assert result.status == StepStatus.FAILED
    assert "empty" in result.error



