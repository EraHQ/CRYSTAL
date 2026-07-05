"""Phase 8 smoke tests for the cognition §6.5.5 refactor.

Per the locked Phase 8 decisions (P0.29, P0.32) + AN-13:

  Test C1: registry-dispatch path runs and is observably distinct
           from the v1-fallback path
  Test C2: fallback to v1 helper when the registry surface raises
           (the actual AN-13 risk path — graceful degradation when
           the agent package's registry call fails at runtime)
  Test C3: COMPOSITION_ACTIONS still route to _worker_llm_step (the
           cognition-only path per D-A10)
  Test C4: cognition_action_alias name resolution — passing
           StepAction.CRYSTAL_KEY_SCAN dispatches to the
           navigation_search tool (not crystal_key_scan, which doesn't
           exist as an agent name)

The AN-13 concern: cognition's §6.5.5 refactor uses a lazy import of
the agent registry inside `_dispatch_tool_via_registry`. If anything
in that path raises (broken module, missing tool registration,
broken registry call), the dispatcher falls back to v1 helpers and
logs at INFO level — graceful degradation but silently. Test C1
distinguishes "registry path actually ran" from "fell back to v1
helper" via the StepOutput.model_used field; Test C2 verifies the
fallback IS reachable when the registry surface raises.

Phase 8 findings (2026-05-26 / 2026-05-27):
  - Original C2 "reset_registry + assert empty + run_worker" didn't
    work because the dispatcher's `import_all_tools()` call inside
    its try-block repopulates the registry before the lookup.
  - Monkey-patching `builtins.__import__` to raise on the dotted
    registry name didn't fire reliably either — `from X import Y`
    against a cached module short-circuits the loader and the
    monkeypatch's name comparison didn't match the name string
    Python actually passes in that path.
  - The reliable approach is to monkey-patch a function that the
    dispatcher demonstrably CALLS (not just imports) inside its
    try-block. `get_registry` from `tool_registry` is such a
    function — replacing it with one that raises lands in the
    except cleanly regardless of import caching.
"""
from __future__ import annotations

from typing import Any

import pytest

from crystal_cache.agent import (
    get_registry,
    import_all_tools,
    reset_registry,
)
from crystal_cache.agent import tool_registry as _tool_registry_module
from crystal_cache.cognition.models import (
    CognitionEnvironment,
    OutputType,
    PlanStep,
    StepAction,
    StepStatus,
)
from crystal_cache.cognition.roles import run_worker
from crystal_cache.llm import reset_llm_client, set_llm_client


# ===========================================================================
# Test C1 — Registry path runs and is observably distinct
# ===========================================================================

@pytest.mark.asyncio
async def test_cognition_dispatches_via_registry_not_v1_fallback(
    customer: Any,
    store: Any,
    fact_vector_store: Any,
    semantic_encoder_stub: Any,
):
    """The §6.5.5 refactor must dispatch CRYSTAL_SEARCH steps through
    the agent tool registry, NOT through `_worker_crystal_search`.

    Distinguishing signal: when the registry/adapter path runs, the
    StepOutput.model_used field is `"registry_adapter:<action>"`. When
    the v1 fallback runs, it is `"none (tool call)"` (or similar
    per the v1 helper's hardcoded string).

    This is the AN-13 distinguishing test — without it, a registry
    breakage at runtime would silently fall back to v1 helpers, and
    we'd never know cognition stopped going through the shared
    surface.
    """
    # Trigger registration explicitly. (The dispatcher calls
    # import_all_tools internally too, but we want this test to be
    # robust against changes that might remove that internal call.)
    import_all_tools()

    env = CognitionEnvironment(
        customer_id=customer.id,
        trigger_type="test",
        trigger_id="t1",
        conversation_context="test",
        source_crystal_id="",
        output_type=OutputType.REPORT,
        max_attempts=1,
    )

    step = PlanStep(
        id=1,
        action=StepAction.CRYSTAL_SEARCH,
        description="search for something",
        input={"query": "nonexistent query against empty bank"},
        depends_on=[],
        parallel_group=None,
    )

    # Pure tool call — no model involved.
    result = await run_worker(
        env=env,
        step=step,
        store=store,
        fact_store=fact_vector_store,
        encoder=semantic_encoder_stub,
    )

    assert result.status == StepStatus.COMPLETE, (
        f"unexpected status: {result.status}, error: {result.error}"
    )
    # The decisive assertion: the registry/adapter path was taken, not
    # the v1 fallback. Post-B (§6.5.5 unification) the cognition
    # retrieval adapter stamps model_used = "registry_adapter:<action>";
    # the v1 fallback would stamp "none (tool call)".
    assert result.model_used == "registry_adapter:crystal_search", (
        f"expected registry-adapter dispatch, got model_used={result.model_used!r}. "
        f"This means cognition fell back to the v1 helper — "
        f"AN-13's silent-degradation concern is real."
    )
    # Post-B the adapter returns cognition's consumed shape (findings +
    # content_text + results_count), not the raw agent-tool
    # injection_text — that shape unification is the whole point of B.
    assert "findings" in result.output
    assert "content_text" in result.output
    assert "matched_fact_ids" in result.output
    # Empty bank — no matches.
    assert result.output["matched_fact_ids"] == []


# ===========================================================================
# Test C2 — Fallback fires when the registry surface raises
# ===========================================================================

@pytest.mark.asyncio
async def test_cognition_falls_back_to_v1_helper_when_registry_call_raises(
    customer: Any,
    store: Any,
    fact_vector_store: Any,
    semantic_encoder_stub: Any,
    monkeypatch: pytest.MonkeyPatch,
):
    """When `get_registry()` raises inside
    `_dispatch_tool_via_registry`, the dispatcher must fall back to
    the v1 worker helper. This is the AN-13 risk path: any runtime
    failure in the agent surface should degrade to v1 behavior (with
    only an INFO-level log) rather than crashing the worker.

    We simulate the failure by monkey-patching `get_registry` in the
    `crystal_cache.agent.tool_registry` module with a function that
    raises. The dispatcher's `from ..agent.tool_registry import
    get_registry` statement resolves at call-time to the patched
    function (module-level attribute lookup), and calling it raises,
    which lands in the dispatcher's except block.

    Distinguishing signal: model_used is the v1 helper's string
    `"none (tool call)"` for crystal_search; output dict carries the
    v1 shape (findings list + content_text).

    Phase 8 alternative-fix note: an earlier attempt to monkey-patch
    `builtins.__import__` to block the dotted-name import was
    unreliable — for cached modules Python's `from X import Y` does
    not call `__import__` with the dotted name in a form our filter
    matched. Patching the resolved function itself is the simpler
    and more direct path to exercising the catch-block.
    """

    def _raise_registry_unavailable() -> Any:
        raise RuntimeError(
            "simulated registry-surface failure (Phase 8 C2 test). "
            "The dispatcher should catch this and fall back to v1."
        )

    monkeypatch.setattr(
        _tool_registry_module, "get_registry", _raise_registry_unavailable,
    )

    env = CognitionEnvironment(
        customer_id=customer.id,
        trigger_type="test",
        trigger_id="t2",
        conversation_context="test",
        source_crystal_id="",
        output_type=OutputType.REPORT,
        max_attempts=1,
    )

    step = PlanStep(
        id=1,
        action=StepAction.CRYSTAL_SEARCH,
        description="search via v1 fallback",
        input={"query": "query"},
        depends_on=[],
        parallel_group=None,
    )

    result = await run_worker(
        env=env,
        step=step,
        store=store,
        fact_store=fact_vector_store,
        encoder=semantic_encoder_stub,
    )

    assert result.status == StepStatus.COMPLETE, (
        f"unexpected status: {result.status}, error: {result.error}"
    )
    # v1 fallback signature.
    assert result.model_used == "none (tool call)", (
        f"expected v1 fallback path, got model_used={result.model_used!r}. "
        f"The patched get_registry should have raised, the dispatcher's "
        f"except block should have caught it, and `_dispatch_via_fallback` "
        f"should have routed into `_worker_crystal_search`."
    )
    # v1 output shape.
    assert "findings" in result.output
    assert "content_text" in result.output
    assert "query" in result.output


# ===========================================================================
# Test C3 — Composition actions stay in _worker_llm_step
# ===========================================================================

@pytest.mark.asyncio
async def test_cognition_composition_actions_use_worker_llm_step(
    customer: Any,
    store: Any,
    fact_vector_store: Any,
    semantic_encoder_stub: Any,
    fake_anthropic: Any,
):
    """Per D-A10 + §6.5.2, ANALYZE / SYNTHESIZE / FORMAT are
    cognition-only composition primitives. They must NOT route
    through the agent registry — they need the prior-step-outputs
    shape that the agent's llm_invoke doesn't have.

    The §6.5.5 refactor guards this via COMPOSITION_ACTIONS frozenset.
    This test confirms the guard: an ANALYZE step routes to
    _worker_llm_step (which makes a real SLM call via the supplied
    client) regardless of whether the registry is populated.
    """
    # Populate the registry — proving that COMPOSITION_ACTIONS path
    # is independent of registry contents.
    import_all_tools()

    fake_anthropic.script_text("Analysis output text.")

    env = CognitionEnvironment(
        customer_id=customer.id,
        trigger_type="test",
        trigger_id="t3",
        conversation_context="test",
        source_crystal_id="",
        output_type=OutputType.REPORT,
        max_attempts=1,
    )

    step = PlanStep(
        id=1,
        action=StepAction.ANALYZE,
        description="analyze prior outputs",
        input={"instruction": "summarize the findings"},
        depends_on=[],  # no deps; prior_context will be empty
        parallel_group=None,
    )

    set_llm_client(fake_anthropic)
    try:
        result = await run_worker(
            env=env,
            step=step,
            store=store,
            fact_store=fact_vector_store,
            encoder=semantic_encoder_stub,
        )
    finally:
        reset_llm_client()

    assert result.status == StepStatus.COMPLETE
    # Composition actions stamp model_used = the model_key
    # ("haiku" or "sonnet"). They do NOT stamp "registry:..."
    assert result.model_used in ("haiku", "sonnet"), (
        f"composition action used unexpected model: {result.model_used!r}"
    )
    assert "content" in result.output
    assert "Analysis output" in result.output["content"]

    # The fake was called exactly once (the analyze step makes one
    # model call).
    fake_anthropic.assert_call_count(1)


# ===========================================================================
# Test C4 — cognition_action_alias name resolution
# ===========================================================================

@pytest.mark.asyncio
async def test_cognition_action_alias_resolves_correctly(
    customer: Any,
    store: Any,
    fact_vector_store: Any,
    semantic_encoder_stub: Any,
):
    """StepAction.CRYSTAL_KEY_SCAN must dispatch via the
    cognition_action_alias index. Post-B (§6.5.5 unification) the
    alias resolves to the `key_scan` enumeration tool (raw findings),
    not `navigation_search` (the what-do-I-know overview) — see the B
    redesign. Verifies the alias mapping survives runtime dispatch.
    """
    import_all_tools()

    env = CognitionEnvironment(
        customer_id=customer.id,
        trigger_type="test",
        trigger_id="t4",
        conversation_context="test",
        source_crystal_id="",
        output_type=OutputType.REPORT,
        max_attempts=1,
    )

    step = PlanStep(
        id=1,
        action=StepAction.CRYSTAL_KEY_SCAN,
        description="enumerate keys",
        input={"subject_contains": "test_subject"},
        depends_on=[],
        parallel_group=None,
    )

    result = await run_worker(
        env=env,
        step=step,
        store=store,
        fact_store=fact_vector_store,
        encoder=semantic_encoder_stub,
    )

    # Registry dispatch should have resolved crystal_key_scan to the
    # key_scan tool via the cognition_action_alias index.
    assert result.status == StepStatus.COMPLETE, (
        f"key scan failed: {result.error}"
    )
    assert result.model_used == "registry_adapter:crystal_key_scan", (
        f"alias resolution failed; got model_used={result.model_used!r}"
    )
    # key_scan returns a findings-shaped dict.
    assert "results_count" in result.output
    assert "findings" in result.output
    # Empty bank — no matching keys.
    assert result.output["results_count"] == 0
