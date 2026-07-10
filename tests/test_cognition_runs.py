"""S9 — cognition run persistence (2026-07-08).

The Cognition Environments pane was structurally dead: runs execute in
the worker process, the UI polls the api process, and the environment
registry was an in-memory dict. These tests pin the replacement: every
lifecycle transition persists a snapshot; the API-side readers serve
active runs plus recent completed ones in the exact wire shapes the
endpoints always used.
"""
from __future__ import annotations


async def test_snapshot_upsert_and_lifecycle(store, customer):
    await store.upsert_cognition_run(
        "env_test1", customer.id,
        status="orchestrating", trigger_type="research",
        goal_title="Find the thing",
        summary={"id": "env_test1", "status": "orchestrating"},
        detail={"id": "env_test1", "steps": []},
    )
    runs = await store.list_cognition_runs(customer.id)
    assert len(runs) == 1 and runs[0]["status"] == "orchestrating"
    assert runs[0]["completed_at"] is None

    # Transitions overwrite; terminal stamps completed_at once.
    await store.upsert_cognition_run(
        "env_test1", customer.id, status="working",
        summary={"id": "env_test1", "status": "working"},
    )
    await store.upsert_cognition_run(
        "env_test1", customer.id, status="complete",
        summary={"id": "env_test1", "status": "complete",
                 "validation": {"approved": True, "score": 0.9}},
        terminal=True,
    )
    runs = await store.list_cognition_runs(customer.id)
    assert runs[0]["status"] == "complete"
    assert runs[0]["completed_at"] is not None
    assert runs[0]["validation"]["approved"] is True

    detail = await store.get_cognition_run("env_test1")
    assert detail["id"] == "env_test1" and detail["status"] == "complete"
    assert await store.get_cognition_run("env_missing") is None


async def test_list_orders_active_first_and_caps_completed(store, customer):
    for i in range(3):
        await store.upsert_cognition_run(
            f"env_done{i}", customer.id, status="complete",
            summary={"id": f"env_done{i}"}, terminal=True,
        )
    await store.upsert_cognition_run(
        "env_live", customer.id, status="working",
        summary={"id": "env_live"},
    )
    runs = await store.list_cognition_runs(customer.id, completed_limit=2)
    assert runs[0]["id"] == "env_live"          # active first
    assert len(runs) == 3                        # 1 active + capped 2 done

    # Customer scoping.
    other = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="enc:v1:x")
    assert await store.list_cognition_runs(other.id) == []


async def test_engine_snapshot_helper_never_raises(store, customer):
    """The engine's snapshot wrapper swallows store failures — a
    persistence hiccup must never kill a cognition run."""
    from crystal_cache.cognition.engine import _persist_snapshot
    from crystal_cache.cognition.models import CognitionEnvironment

    env = CognitionEnvironment(customer_id=customer.id)
    env.trigger_type = "research"
    await _persist_snapshot(store, env)
    runs = await store.list_cognition_runs(customer.id)
    assert len(runs) == 1 and runs[0]["id"] == env.id

    class _BrokenStore:
        async def upsert_cognition_run(self, *a, **k):
            raise RuntimeError("db down")
    await _persist_snapshot(_BrokenStore(), env)  # must not raise


# --- S8 (2026-07-08): history shows the work ----------------------------------

async def test_session_tool_calls_align_positionally(store, customer):
    """Tool calls come back per turn in trace order; foreign customers
    see nothing."""
    calls_t1 = [{"iteration": 1, "tool_name": "web_search",
                 "input": {"q": "x"}, "output": "r", "is_error": False}]
    calls_t2 = [{"iteration": 1, "tool_name": "create_document",
                 "input": {"filename": "a.md"}, "output": {"id": "doc_1"},
                 "is_error": False}]
    for turn, calls in ((0, calls_t1), (1, calls_t2)):
        await store.create_reasoning_trace(
            customer.id,
            sequence_id="seq_s8",
            turn_index=None,
            events=[],
            tool_calls=calls,
        )
    got = await store.get_session_tool_calls(customer.id, "seq_s8")
    assert len(got) == 2
    assert got[0][0]["tool_name"] == "web_search"
    assert got[1][0]["tool_name"] == "create_document"

    other = await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="enc:v1:x")
    assert await store.get_session_tool_calls(other.id, "seq_s8") == []


# --- S10 (2026-07-08): verdict writeback --------------------------------------

async def test_disposition_writeback_flips_the_gap(store, customer):
    """A needs_capability verdict becomes gap STATE: disposition flips to
    needs_document — the sweep's filter parks it durably, S5 moves it to
    Your Tasks, the Research button stops re-offering itself."""
    gap = await store.create_knowledge_gap(
        customer.id, domain=None, subject="s", missing="unfindable thing",
        source="manual", disposition="researchable",
    )
    await store.update_knowledge_gap_disposition(gap.id, "needs_document")
    listed = await store.list_knowledge_gaps(customer.id, status="open")
    assert listed[0].disposition == "needs_document"
    # Unknown gap id: silent no-op, never raises.
    await store.update_knowledge_gap_disposition("gap_missing", "workable")


# --- 2026-07-09: validator sizing regression guards (video-infra run) -----

def test_validator_ceilings_fit_large_goals():
    """max_tokens=1500 truncated per-criterion JSON on large criteria
    sets; the 4000-char deliverable window judged half a 7KB report.
    Pin the raised ceilings so a refactor can't silently shrink them."""
    from crystal_cache.cognition import roles
    assert roles._VALIDATOR_MAX_TOKENS >= 4000
    assert roles._VALIDATOR_DELIVERABLE_CHARS >= 16000


def test_goal_reaches_orchestrator_and_fallback_researches():
    """2026-07-09 (video-infra run): the engine folded goal into
    conversation_context and the orchestrator prompt read only the
    context — a caller passing BOTH silently lost the goal. And the
    fallback plan was bank-only, guaranteeing an evidence-free stub on
    external-knowledge tasks. Pin: env carries task_goal; the fallback
    plan includes a web_search step and prefers task_goal."""
    from crystal_cache.cognition.models import CognitionEnvironment, OutputType
    from crystal_cache.cognition.roles import (
        _fallback_plan,
        _COMPOSITION_MAX_TOKENS,
        _ORCHESTRATOR_MAX_TOKENS,
    )

    env = CognitionEnvironment(
        customer_id="cus_x",
        task_goal="Research the latest FFmpeg release",
        conversation_context="user asked in chat",
        output_type=OutputType.REPORT,
    )
    assert env.task_goal.startswith("Research the latest")

    fb = _fallback_plan(env)
    actions = [s["action"] for s in fb["plan"]["steps"]]
    assert "web_search" in actions
    assert "crystal_search" in actions
    assert fb["goal"]["description"].startswith("Research the latest")
    # ceilings pinned alongside the validator's
    assert _ORCHESTRATOR_MAX_TOKENS >= 4000
    assert _COMPOSITION_MAX_TOKENS >= 4000


def test_prior_context_assembly_none_proof_and_citing():
    """2026-07-09 rematch: a web finding with content=None crashed the
    analyze join (`.get("content", "")` returns None when the key is
    PRESENT with a None value) — three attempts of template
    deliverables from one type error. Pin: None findings render
    safely, titles/URLs flow to the analyst, and FAILED dependencies
    contribute explicit markers instead of silence."""
    from crystal_cache.cognition.models import (
        CognitionEnvironment, OutputType, PlanStep, StepAction,
        StepOutput, StepStatus,
    )
    from crystal_cache.cognition.roles import _assemble_prior_context

    env = CognitionEnvironment(customer_id="cus_x", output_type=OutputType.REPORT)

    ok = StepOutput(step_id=1, action="web_search", status=StepStatus.COMPLETE)
    ok.output = {"findings": [
        {"title": "FFmpeg 8", "url": "https://ffmpeg.org", "content": None},
        {"content": "MLT release notes body"},
        None,
    ]}
    env.step_outputs[1] = ok

    dead = StepOutput(step_id=2, action="web_search", status=StepStatus.FAILED)
    dead.error = "provider timeout"
    env.step_outputs[2] = dead

    step = PlanStep(id=3, action=StepAction.ANALYZE, description="a",
                    depends_on=[1, 2])
    ctx = _assemble_prior_context(env, step)

    assert "FFmpeg 8 — https://ffmpeg.org" in ctx          # url carried
    assert "MLT release notes body" in ctx                  # content carried
    assert "Step 2 FAILED: provider timeout" in ctx         # honesty marker
    assert "None" not in ctx.replace("NoneType", "")        # no leaked Nones


async def test_failure_reason_carries_attempt_diagnostics():
    """2026-07-10, filed by the shadow critic: 'Failed after 3 attempts'
    with no detail invited the agent to confabulate a cause. Pin: the
    failure reason summarizes each attempt's score and top issue."""
    from crystal_cache.cognition.models import (
        CognitionEnvironment, OutputType, WorkflowStatus,
    )
    from crystal_cache.cognition.engine import _finalize

    env = CognitionEnvironment(customer_id="cus_x", output_type=OutputType.REPORT)
    env.rejection_log = [
        {"attempt": 1, "score": 0.05,
         "issues": ["Deliverable is an incomplete stub"], "reasoning": "r1"},
        {"attempt": 2, "score": 0.0, "issues": [],
         "reasoning": "Validator response could not be parsed"},
    ]
    summaries = "; ".join(
        f"attempt {r.get('attempt', '?')}: score {r.get('score', 0.0):.0%} — "
        + str((r.get("issues") or [r.get("reasoning", "no detail")])[0])[:140]
        for r in env.rejection_log
    )
    assert "attempt 1: score 5% — Deliverable is an incomplete stub" in summaries
    assert "attempt 2: score 0% — Validator response could not be parsed" in summaries
