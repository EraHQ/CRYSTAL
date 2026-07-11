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
