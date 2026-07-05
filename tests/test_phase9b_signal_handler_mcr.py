"""Phase 9B smoke tests for MCR integration in v3_signal_handler.

Per P0.54: five tests covering the BD-3 and BD-11 integration
landings. The tests call `handle_signals(...)` directly with
`mcr_enabled=True` (the new default-False kwarg) and verify both
the existing artifact rows (KnowledgeGapRow, CognitionTask) AND
the new MCR rows (Critique + ActionItem pairs) land correctly.

Test scope (P0.54):
  S1: push_gap with mcr_enabled=True → BOTH KnowledgeGapRow AND
      Critique + ActionItem(gap_declaration); content carries
      conversation_context per P0.42 (BD-11 resolution).
  S2: push_correct with mcr_enabled=True → Critique(source_contradiction)
      + ActionItem(edit_proposal); content carries key/old/new/rationale
      per P0.43 (BD-3 resolution).
  S3: push_gap with mcr_enabled=False (default) → ONLY KnowledgeGapRow,
      no MCR rows. Proves the feature flag works and Phase 9B is
      inert until Phase 9C flips it.
  S4: MCR write failure during Pass 2 does NOT abort the rest — one
      bad Critique write doesn't prevent the next gap's KnowledgeGapRow.
      Failure-mode discipline matches Phase 9A's emit_mcr_artifacts.
  S5: Mixed batch — one push_gap + one push_correct + one push_research
      in one handle_signals call → 1 KnowledgeGapRow + 1 CognitionTask
      + 2 Critique rows + 2 ActionItem rows.

Phase 8 / 8.5 / 9A test files are NOT touched.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from crystal_cache.retrieval.v3_push_pull import ParsedSignals
from crystal_cache.retrieval.v3_signal_handler import handle_signals


# ---------------------------------------------------------------------------
# Helpers — build ParsedSignals from raw tool-call dicts
# ---------------------------------------------------------------------------

def _tool_call(
    name: str,
    args: dict[str, Any],
    tool_call_id: str = "tc_test",
) -> dict[str, Any]:
    """Build one raw tool_call dict shaped like the upstream LLM emits.

    The signal handler's Pass 1 re-parses arguments from JSON, so
    we serialize them at construction time to match the production
    wire format.
    """
    return {
        "id": tool_call_id,
        "function": {
            "name": name,
            "arguments": json.dumps(args),
        },
    }


def _signals(*tool_calls: dict[str, Any]) -> ParsedSignals:
    """Wrap a list of tool_call dicts as a ParsedSignals object.

    handle_signals reads `signals.raw_tool_calls` in Pass 1 and
    iterates dispatch by tool name; the per-type lists
    (push_gaps / push_corrections / etc) are not used in Pass 1
    so we don't populate them here. has_signals checks the
    per-type lists, so we populate them from raw_tool_calls to
    make the property return True.
    """
    sig = ParsedSignals()
    sig.raw_tool_calls = list(tool_calls)
    # Populate per-type lists so has_signals returns True. Pass 1
    # re-parses from raw_tool_calls so the values here don't matter
    # for behavior; only the True/False of has_signals does.
    for tc in tool_calls:
        name = tc.get("function", {}).get("name", "")
        if name == "crystal_push_store":
            sig.push_stores.append({})
        elif name == "crystal_push_gap":
            sig.push_gaps.append({})
        elif name == "crystal_push_correct":
            sig.push_corrections.append({})
        elif name == "crystal_pull_research":
            sig.pull_research.append({})
        elif name == "crystal_pull_expand":
            sig.pull_expand.append({})
    return sig


# ===========================================================================
# Test S1 — push_gap with mcr_enabled=True
# ===========================================================================

@pytest.mark.asyncio
async def test_push_gap_writes_both_knowledge_gap_and_mcr_pair(
    customer: Any,
    store: Any,
):
    """push_gap with mcr_enabled=True must write BOTH a
    KnowledgeGapRow (existing path) AND a Critique +
    ActionItem(gap_declaration) (new MCR path per P0.42).

    The action_item.content carries the conversation_context that
    v1 silently dropped per BD-11.
    """
    signals = _signals(_tool_call(
        "crystal_push_gap",
        {
            "domain": "engineering",
            "subject": "build pipeline",
            "missing": "which CI runner is used",
        },
        tool_call_id="tc_s1",
    ))

    stats = await handle_signals(
        signals,
        customer_id=customer.id,
        store=store,
        conversation_context="user: tell me about CI\nassistant: I don't know which runner",
        sequence_id="seq_s1",
        turn_index=0,
        agent_model="claude-sonnet-4-5-20250929",
        mcr_enabled=True,
    )

    # Pass-1 stats: gap was dispatched, tool_result produced.
    assert stats["gaps_recorded"] == 1
    assert stats["processed"] == 1
    assert len(stats["tool_results"]) == 1
    assert stats["tool_results"][0]["tool_call_id"] == "tc_s1"
    # Pass-2 stats: BOTH a critique AND an action item landed.
    assert stats["mcr_critiques_written"] == 1
    assert stats["mcr_action_items_written"] == 1

    # KnowledgeGapRow side — the existing path still works.
    # Look it up via the store. The mixin has list_knowledge_gaps
    # but for this test we use list_critiques_for_sequence + verify
    # the gap row exists by querying directly through the open
    # method.
    # The simplest check is that the critique + action item have
    # the right content, since they're what's new in Phase 9B.

    # Critique side via the soft-join key.
    critiques = await store.list_critiques_for_sequence(
        customer_id=customer.id,
        sequence_id="seq_s1",
    )
    assert len(critiques) == 1
    critique = critiques[0]
    assert critique.critic_role == "agent_self"
    assert critique.critic_model == "claude-sonnet-4-5-20250929"
    assert critique.trace_id is None  # Phase 9B doesn't have a trace yet
    assert critique.turn_index == 0
    assert critique.total_action_items == 1
    assert len(critique.observations) == 1
    obs = critique.observations[0]
    assert obs["type"] == "gap_papered_over"
    assert obs["confidence"] == 0.8
    assert "which CI runner" in obs["text"]

    # ActionItem(gap_declaration) side via FK.
    items = await store.list_action_items_for_critique(critique.id)
    assert len(items) == 1
    item = items[0]
    assert item.action_type == "gap_declaration"
    assert item.status == "pending"
    assert item.customer_id == customer.id
    assert item.critique_id == critique.id
    assert item.critic_confidence == 0.8
    # P0.42 / P0.53 content schema: {want, why_needed, domain,
    # conversation_context}.
    assert item.content["want"] == "which CI runner is used"
    assert item.content["why_needed"] == "build pipeline"
    assert item.content["domain"] == "engineering"
    # The conversation_context that v1 silently dropped per BD-11
    # is now persisted.
    assert "tell me about CI" in item.content["conversation_context"]
    assert "don't know which runner" in item.content["conversation_context"]


# ===========================================================================
# Test S2 — push_correct with mcr_enabled=True (BD-3 resolution)
# ===========================================================================

@pytest.mark.asyncio
async def test_push_correct_writes_critique_and_edit_proposal(
    customer: Any,
    store: Any,
):
    """push_correct with mcr_enabled=True must write a
    Critique(source_contradiction) + ActionItem(edit_proposal)
    pair per P0.43 (BD-3 resolution). Default confidence is 0.8;
    content carries key/old/new/rationale.
    """
    signals = _signals(_tool_call(
        "crystal_push_correct",
        {
            "key": "project.deadline",
            "old_value": "March 15, 2027",
            "new_value": "April 1, 2027",
        },
        tool_call_id="tc_s2",
    ))

    stats = await handle_signals(
        signals,
        customer_id=customer.id,
        store=store,
        sequence_id="seq_s2",
        turn_index=2,
        agent_model="claude-sonnet-4-5-20250929",
        mcr_enabled=True,
    )

    assert stats["corrections_flagged"] == 1
    assert stats["processed"] == 1
    assert stats["mcr_critiques_written"] == 1
    assert stats["mcr_action_items_written"] == 1

    # Critique side.
    critiques = await store.list_critiques_for_sequence(
        customer_id=customer.id,
        sequence_id="seq_s2",
        turn_index=2,
    )
    assert len(critiques) == 1
    critique = critiques[0]
    assert critique.critic_role == "agent_self"
    assert critique.critic_model == "claude-sonnet-4-5-20250929"
    assert critique.turn_index == 2
    assert critique.total_action_items == 1
    assert len(critique.observations) == 1
    obs = critique.observations[0]
    assert obs["type"] == "source_contradiction"
    assert obs["confidence"] == 0.8
    assert "project.deadline" in obs["text"]
    # Anchors carry the (key, old_value) pair per P0.43.
    assert len(obs["anchors"]) == 1
    anchor = obs["anchors"][0]
    assert anchor["key"] == "project.deadline"
    assert "March 15" in anchor["old_value"]

    # ActionItem(edit_proposal) side.
    items = await store.list_action_items_for_critique(critique.id)
    assert len(items) == 1
    item = items[0]
    assert item.action_type == "edit_proposal"
    assert item.status == "pending"
    assert item.critic_confidence == 0.8
    # P0.43 / P0.53 content schema: {key, old_value, new_value, rationale}.
    assert item.content["key"] == "project.deadline"
    assert item.content["old_value"] == "March 15, 2027"
    assert item.content["new_value"] == "April 1, 2027"
    assert item.content["rationale"] == (
        "agent self-correction via crystal_push_correct"
    )


# ===========================================================================
# Test S3 — mcr_enabled=False (default) leaves the new paths inert
# ===========================================================================

@pytest.mark.asyncio
async def test_push_gap_with_mcr_disabled_writes_only_knowledge_gap(
    customer: Any,
    store: Any,
):
    """With mcr_enabled=False (the default), Phase 9B's new code
    paths are inert. push_gap continues to write a KnowledgeGapRow
    as it has since Phase 6 Wave D; no Critique or ActionItem rows
    are produced.

    This is the critical regression-protection test: Phase 9C will
    flip mcr_enabled=True at the chat_proxy boundary, but until
    that lands, the proxy must behave exactly as before.
    """
    signals = _signals(
        _tool_call(
            "crystal_push_gap",
            {"domain": "x", "subject": "y", "missing": "z"},
            tool_call_id="tc_s3_gap",
        ),
        _tool_call(
            "crystal_push_correct",
            {"key": "k", "old_value": "old", "new_value": "new"},
            tool_call_id="tc_s3_corr",
        ),
    )

    stats = await handle_signals(
        signals,
        customer_id=customer.id,
        store=store,
        sequence_id="seq_s3",
        turn_index=0,
        # mcr_enabled defaults to False — DO NOT pass it.
    )

    # Existing paths fire normally.
    assert stats["gaps_recorded"] == 1
    assert stats["corrections_flagged"] == 1
    # New paths are inert: no critiques or action items written.
    assert stats["mcr_critiques_written"] == 0
    assert stats["mcr_action_items_written"] == 0

    # Verify the absence of MCR rows directly via the store.
    critiques = await store.list_critiques_for_sequence(
        customer_id=customer.id,
        sequence_id="seq_s3",
    )
    assert critiques == []


# ===========================================================================
# Test S4 — MCR write failure does not abort the rest
# ===========================================================================

@pytest.mark.asyncio
async def test_mcr_write_failure_does_not_break_other_writes(
    customer: Any,
    store: Any,
    monkeypatch: Any,
):
    """If one MCR write raises mid-Pass-2, the loop continues with
    the next entry. Mirrors Phase 9A's NEVER-raises failure-mode
    discipline (P0.44, P0.54-S4).

    We monkeypatch store.create_critique to fail on the FIRST call
    and succeed on the second. With two push_gap entries in the
    batch, the second gap's MCR pair should land even though the
    first's didn't.
    """
    original_create_critique = store.create_critique
    call_count = {"n": 0}

    async def flaky_create_critique(*args: Any, **kwargs: Any) -> Any:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated DB hiccup on first critique")
        return await original_create_critique(*args, **kwargs)

    monkeypatch.setattr(store, "create_critique", flaky_create_critique)

    signals = _signals(
        _tool_call(
            "crystal_push_gap",
            {"domain": "a", "subject": "b", "missing": "first gap"},
            tool_call_id="tc_s4_1",
        ),
        _tool_call(
            "crystal_push_gap",
            {"domain": "c", "subject": "d", "missing": "second gap"},
            tool_call_id="tc_s4_2",
        ),
    )

    # Must NOT raise.
    stats = await handle_signals(
        signals,
        customer_id=customer.id,
        store=store,
        sequence_id="seq_s4",
        turn_index=0,
        agent_model="claude-sonnet-4-5-20250929",
        mcr_enabled=True,
    )

    # Both gaps were dispatched in Pass 1.
    assert stats["gaps_recorded"] == 2
    # The flaky path was exercised: store.create_critique was called
    # twice (once per gap entry in the Pass-2 loop).
    assert call_count["n"] == 2
    # The first attempt failed; the second succeeded.
    assert stats["mcr_critiques_written"] == 1
    # The action item for the failed critique never got written
    # (the helper returns early on critique failure); the action
    # item for the successful critique did get written.
    assert stats["mcr_action_items_written"] == 1

    # And the existing KnowledgeGapRow writes were NOT affected by
    # the MCR failure — they happen first in each iteration of the
    # gap loop. Verify by counting critiques actually persisted.
    critiques = await store.list_critiques_for_sequence(
        customer_id=customer.id,
        sequence_id="seq_s4",
    )
    assert len(critiques) == 1
    # The surviving critique describes the SECOND gap (call_count
    # 2's input was "second gap").
    assert "second gap" in critiques[0].observations[0]["text"]


# ===========================================================================
# Test S5 — Mixed batch: gap + correct + research in one call
# ===========================================================================

@pytest.mark.asyncio
async def test_mixed_batch_produces_correct_artifact_counts(
    customer: Any,
    store: Any,
):
    """A single handle_signals call with one push_gap + one
    push_correct + one push_research must produce:
      - 1 KnowledgeGapRow (existing path)
      - 1 CognitionTask (existing path)
      - 2 Critique rows (one per MCR-emitting signal: gap + correct)
      - 2 ActionItem rows (one per MCR-emitting signal: gap +
        correct, FK-linked to their respective critiques)

    push_research does NOT emit MCR rows per P0.50 — the research
    task IS the artifact.

    This guards against a Phase 11.5 bug where someone refactors
    the Pass-2 loops and forgets that gap-MCR + correct-MCR +
    research-task all need to coexist.
    """
    signals = _signals(
        _tool_call(
            "crystal_push_gap",
            {"domain": "d", "subject": "s", "missing": "m"},
            tool_call_id="tc_s5_gap",
        ),
        _tool_call(
            "crystal_push_correct",
            {"key": "k", "old_value": "o", "new_value": "n"},
            tool_call_id="tc_s5_corr",
        ),
        _tool_call(
            "crystal_pull_research",
            {"topic": "t", "scope": "narrow", "priority": "background"},
            tool_call_id="tc_s5_res",
        ),
    )

    stats = await handle_signals(
        signals,
        customer_id=customer.id,
        store=store,
        conversation_context="user: question\nassistant: partial answer",
        sequence_id="seq_s5",
        turn_index=0,
        agent_model="claude-sonnet-4-5-20250929",
        mcr_enabled=True,
    )

    # Pass-1 dispatch: all three landed.
    assert stats["processed"] == 3
    assert stats["gaps_recorded"] == 1
    assert stats["corrections_flagged"] == 1
    assert stats["research_queued"] == 1
    # Pass-2 MCR writes: 2 critiques + 2 action items (gap + correct).
    # Research does NOT emit per P0.50.
    assert stats["mcr_critiques_written"] == 2
    assert stats["mcr_action_items_written"] == 2

    # Verify the two critiques exist with the right observation types.
    critiques = await store.list_critiques_for_sequence(
        customer_id=customer.id,
        sequence_id="seq_s5",
    )
    assert len(critiques) == 2
    obs_types = sorted(c.observations[0]["type"] for c in critiques)
    assert obs_types == ["gap_papered_over", "source_contradiction"]

    # Verify the action items exist and are correctly FK-linked.
    for critique in critiques:
        items = await store.list_action_items_for_critique(critique.id)
        assert len(items) == 1
        # Each critique has one item; the action_type matches the
        # observation type:
        #   gap_papered_over → gap_declaration
        #   source_contradiction → edit_proposal
        item = items[0]
        obs_type = critique.observations[0]["type"]
        if obs_type == "gap_papered_over":
            assert item.action_type == "gap_declaration"
            assert item.content["want"] == "m"
            assert item.content["domain"] == "d"
        else:
            assert item.action_type == "edit_proposal"
            assert item.content["key"] == "k"
            assert item.content["old_value"] == "o"
            assert item.content["new_value"] == "n"

    # And the research task path is unaffected (no MCR row produced
    # for it, but the CognitionTask itself landed via Pass 2's
    # second loop). We don't have a list_cognition_tasks helper on
    # the test surface, so we rely on the counter for evidence.
    assert stats["research_queued"] == 1
