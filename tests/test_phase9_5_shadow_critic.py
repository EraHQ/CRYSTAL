"""Phase 9.5 tests for the MCR shadow critic.

Per P0.62 + P0.68: six tests covering the shadow critic — the second
MCR critic (MCR_FRAMEWORK.md §5.2, D-MCR-10).

Test scope (P0.68):
  SH1: should_shadow_trace returns True when the self-critique flagged
       >=1 observation, regardless of sample_rate (always-sample-on-flag).
  SH2: should_shadow_trace Bernoulli behavior for CLEAN self-critiques
       (rate=1.0 -> always True; rate=0.0 -> always False).
  SH3: shadow_review_trace loads a persisted trace + its agent_self
       critique, runs the shadow LLM (scripted FakeAnthropic), persists
       Critique(critic_role="shadow") with the right critic_model and
       parsed observations.
  SH4: the shadow critique's trace_id is the HARD FK pointer (not None)
       — confirms P0.63 resolves CU-19 for the shadow path.
  SH5: shadow LLM call failure -> emit does NOT raise; persists a shadow
       critique with empty observations + a failure summary.
  SH6: the shadow reviews the SELF-CRITIQUE — the review message sent to
       the LLM includes the agent_self critique's observation text
       (§5.2: "the shadow critiques the self-critique too").

No Phase 8/8.5/9A/9B/9C files touched. Reuses the `store`, `customer`,
and `fake_anthropic` fixtures from conftest.py.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from crystal_cache.agent.shadow_critic import (
    ShadowSamplingPolicy,
    shadow_review_trace,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _seed_trace_and_self_critique(
    store: Any,
    customer_id: str,
    *,
    sequence_id: str = "seq_sh",
    turn_index: int = 0,
    self_observations: list[dict[str, Any]] | None = None,
    final_text: str = "The deadline is April 1.",
    crystals_used: list[str] | None = None,
) -> tuple[str, str]:
    """Create a persisted ReasoningTrace + an agent_self Critique for it.

    Returns (trace_id, self_critique_id).

    The trace carries a trailing final_text event (mirroring what
    build_trace_from_agent_result emits) so the shadow critic can
    reconstruct the agent's response.
    """
    events: list[dict[str, Any]] = [
        {
            "iteration": 1,
            "tool_name": "content_search",
            "tool_use_id": "tu_seed",
            "input": {"query": "deadline"},
            "output": {"matched_fact_ids": crystals_used or ["cry_seed"]},
            "is_error": False,
        },
        {
            "type": "final_text",
            "text": final_text,
            "stop_reason": "end_turn",
        },
    ]
    trace = await store.create_reasoning_trace(
        customer_id=customer_id,
        events=events,
        sequence_id=sequence_id,
        turn_index=turn_index,
        crystals_used=crystals_used or ["cry_seed"],
        tool_calls=[events[0]],
        gaps_felt=[],
    )

    critique = await store.create_critique(
        customer_id=customer_id,
        critic_role="agent_self",
        critic_model="claude-haiku-4-5-20251001",
        trace_id=trace.id,
        sequence_id=sequence_id,
        turn_index=turn_index,
        observations=self_observations or [],
        summary_text="Self-critique: reasoning looked clean.",
        total_action_items=0,
    )
    return trace.id, critique.id


def _script_shadow_response(
    fake_anthropic: Any,
    *,
    observations: list[dict[str, Any]] | None = None,
    action_items: list[dict[str, Any]] | None = None,
    summary: str = "Shadow agrees with the self-critique.",
) -> None:
    """Queue a scripted shadow-critique JSON response on the fake."""
    payload = {
        "observations": observations or [],
        "action_items": action_items or [],
        "summary_text": summary,
    }
    fake_anthropic.script_text(json.dumps(payload))


# ---------------------------------------------------------------------------
# SH1 — always-sample when the self-critique flagged something
# ---------------------------------------------------------------------------

def test_sh1_always_shadow_when_self_critique_flagged():
    """ShadowSamplingPolicy.should_shadow_trace returns True whenever the
    self-critique has >=1 observation, EVEN at sample_rate=0.0.
    """
    policy = ShadowSamplingPolicy(sample_rate=0.0, random_seed=42)

    flagged = [
        {"type": "gap_papered_over", "text": "x", "confidence": 0.7, "anchors": []}
    ]
    # Self-critique flagged → always shadow, despite rate=0.
    assert policy.should_shadow_trace(flagged) is True


# ---------------------------------------------------------------------------
# SH2 — Bernoulli behavior for clean self-critiques
# ---------------------------------------------------------------------------

def test_sh2_bernoulli_for_clean_self_critique():
    """For a CLEAN self-critique (no observations), should_shadow_trace
    follows the Bernoulli rate: rate>=1.0 always True, rate<=0.0 always
    False.
    """
    always = ShadowSamplingPolicy(sample_rate=1.0, random_seed=1)
    never = ShadowSamplingPolicy(sample_rate=0.0, random_seed=1)

    clean: list[dict[str, Any]] = []

    # rate=1.0 → always shadow even with a clean self-critique.
    assert always.should_shadow_trace(clean) is True
    # rate=0.0 → never shadow a clean self-critique.
    assert never.should_shadow_trace(clean) is False


# ---------------------------------------------------------------------------
# SH3 — happy path: load trace + self-critique, run shadow, persist
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sh3_shadow_review_persists_shadow_critique(
    store: Any,
    customer: Any,
    fake_anthropic: Any,
):
    """shadow_review_trace loads a persisted trace + its agent_self
    critique, runs the shadow LLM, and persists a
    Critique(critic_role="shadow") with the parsed observation and the
    configured critic_model.
    """
    trace_id, _ = await _seed_trace_and_self_critique(
        store,
        customer.id,
        sequence_id="seq_sh3",
        self_observations=[
            {
                "type": "gap_papered_over",
                "text": "agent guessed the deadline",
                "confidence": 0.6,
                "anchors": [],
            }
        ],
    )

    # Shadow emits one observation disagreeing with the self-critique.
    _script_shadow_response(
        fake_anthropic,
        observations=[
            {
                "type": "border_crossing_unflagged",
                "text": "self-critique missed an inference leap",
                "confidence": 0.8,
                "anchors": [],
            }
        ],
        summary="Shadow found an additional border crossing.",
    )

    result = await shadow_review_trace(
        store=store,
        trace_id=trace_id,
        anthropic_client=fake_anthropic,
        shadow_model="claude-opus-4-7",
        force=True,  # bypass sampling; SH1/SH2 cover the policy
    )

    assert result["sampled"] is True
    assert result["reason"] == "shadowed"
    assert result["shadow_critique_id"] is not None

    # A shadow critique landed, addressable via the soft-join key.
    critiques = await store.list_critiques_for_sequence(
        customer_id=customer.id,
        sequence_id="seq_sh3",
    )
    # Two critiques now exist for this sequence: agent_self + shadow.
    roles = sorted(c.critic_role for c in critiques)
    assert roles == ["agent_self", "shadow"]

    shadow = next(c for c in critiques if c.critic_role == "shadow")
    assert shadow.critic_model == "claude-opus-4-7"
    assert len(shadow.observations) == 1
    assert shadow.observations[0]["type"] == "border_crossing_unflagged"
    assert shadow.summary_text == "Shadow found an additional border crossing."

    # The fake recorded exactly one model call (the shadow LLM call).
    assert len(fake_anthropic.calls) == 1


# ---------------------------------------------------------------------------
# SH4 — shadow critique trace_id is the hard FK (CU-19 resolution)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sh4_shadow_critique_has_hard_trace_id(
    store: Any,
    customer: Any,
    fake_anthropic: Any,
):
    """Per P0.63: the shadow runs AFTER the trace is persisted, so the
    shadow critique's trace_id is a HARD FK pointer (not None). This is
    the CU-19 resolution for the shadow path — no update_critique_trace_id
    upgrade method needed.
    """
    trace_id, _ = await _seed_trace_and_self_critique(
        store,
        customer.id,
        sequence_id="seq_sh4",
    )
    _script_shadow_response(fake_anthropic)

    result = await shadow_review_trace(
        store=store,
        trace_id=trace_id,
        anthropic_client=fake_anthropic,
        force=True,
    )

    shadow_id = result["shadow_critique_id"]
    assert shadow_id is not None

    # Look it up via the HARD pointer: list_critiques_for_trace filters
    # on trace_id, so a shadow critique appearing here proves trace_id
    # was set (not NULL).
    by_trace = await store.list_critiques_for_trace(trace_id)
    shadow_critiques = [c for c in by_trace if c.critic_role == "shadow"]
    assert len(shadow_critiques) == 1
    assert shadow_critiques[0].id == shadow_id
    assert shadow_critiques[0].trace_id == trace_id


# ---------------------------------------------------------------------------
# SH5 — shadow LLM failure does not raise; persists failure summary
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sh5_shadow_llm_failure_persists_failure_summary(
    store: Any,
    customer: Any,
    fake_anthropic: Any,
):
    """When the shadow LLM call fails, shadow_review_trace does NOT raise;
    it persists a shadow critique with empty observations and a summary
    noting the failure (inherits mcr_emitter's NEVER-raises discipline).

    We force failure by NOT scripting any response on the fake — its
    .create raises AssertionError when the script is empty, which
    run_shadow_critique catches and converts to a failure summary.
    """
    trace_id, _ = await _seed_trace_and_self_critique(
        store,
        customer.id,
        sequence_id="seq_sh5",
    )
    # Deliberately do NOT script a response → the fake raises on call.

    result = await shadow_review_trace(
        store=store,
        trace_id=trace_id,
        anthropic_client=fake_anthropic,
        force=True,
    )

    # Did not raise; a shadow critique still landed.
    assert result["shadow_critique_id"] is not None
    assert result["sampled"] is True

    by_trace = await store.list_critiques_for_trace(trace_id)
    shadow = next(c for c in by_trace if c.critic_role == "shadow")
    # Empty observations on failure.
    assert shadow.observations == []
    # Summary carries the failure detail for forensic review.
    assert "shadow critique call failed" in (shadow.summary_text or "")


# ---------------------------------------------------------------------------
# SH6 — the shadow reviews the self-critique (§5.2)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sh6_shadow_review_message_includes_self_critique(
    store: Any,
    customer: Any,
    fake_anthropic: Any,
):
    """Per P0.64 + §5.2: the shadow critiques the self-critique too. The
    review message sent to the shadow LLM must include the agent_self
    critique's observation text so the shadow can assess it.
    """
    distinctive_self_obs_text = "agent assumed the fiscal year without checking"
    trace_id, _ = await _seed_trace_and_self_critique(
        store,
        customer.id,
        sequence_id="seq_sh6",
        self_observations=[
            {
                "type": "assumption_identified",
                "text": distinctive_self_obs_text,
                "confidence": 0.5,
                "anchors": [],
            }
        ],
    )
    _script_shadow_response(fake_anthropic)

    await shadow_review_trace(
        store=store,
        trace_id=trace_id,
        anthropic_client=fake_anthropic,
        force=True,
    )

    # Inspect the captured model call. The user message must carry the
    # self-critique observation text.
    assert len(fake_anthropic.calls) == 1
    call = fake_anthropic.calls[0]
    user_message = call["messages"][0]["content"]
    assert distinctive_self_obs_text in user_message
    # And the system prompt must be the SHADOW prompt (reviews both
    # reasoning and self-critique), not the self-critique prompt.
    assert "SHADOW critic" in call["system"]
    assert "self-critique" in call["system"]
