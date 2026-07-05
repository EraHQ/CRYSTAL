"""Phase 12 production-hardening tests.

Per P0.114: tests for the Phase 12 cleanup items that ship code.

  CU-27 (per-customer shadow cost cap, P0.111):
    H1 — get/set_customer_shadow_cap round-trip (override, clear,
         missing customer, negative rejection).
    H2 — worker _shadow_pass honors a per-customer override that is
         tighter than the global default.
    H3 — worker _shadow_pass falls through to the global/injected
         default when the customer has no override.

  CU-28 (drop-on-low-trust-critic, P0.112):
    H4 — synthesize_for_trace pure function: low-trust critic's
         would-be-deferred items are dropped; substrate observations
         are exempt; promotions are unaffected; the guards (no
         calibrations, insufficient samples, healthy promotion rate)
         all decline to drop.
    H5 — compute_alignment_and_synthesis_for_trace end-to-end: a
         seeded low-trust calibration causes the engine to drop a
         shadow critic's solo item (status='dropped', recorded in the
         synthesis row with the frozen rationale).

CU-26 (worker default) and CU-31 (schema prose) are ledger-only /
WONT_FIX and carry no test.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from crystal_cache.llm import reset_llm_client, set_llm_client
from crystal_cache.metacognition import (
    compute_alignment_and_synthesis_for_trace,
    synthesize_for_trace,
)
from crystal_cache.metacognition.synthesis import (
    RATIONALE_AGENT_SELF_SOLO,
    RATIONALE_DROPPED_LOW_TRUST_CRITIC,
    RATIONALE_SHADOW_SOLO,
    RATIONALE_SUBSTRATE_DEFERRED,
)
from crystal_cache.models.action_item import ActionItem
from crystal_cache.models.critic_calibration import CriticCalibration
from crystal_cache.models.critique import Critique
from crystal_cache.models.item_alignment import ItemAlignment
from crystal_cache.workers.metacognition import _shadow_pass

from fakes import FakeAnthropic


# ---------------------------------------------------------------------------
# Pure-function helpers (mirror the Phase 10A test patterns; local copy so
# the test files stay independent).
# ---------------------------------------------------------------------------

def _make_item(
    item_id: str,
    critique_id: str,
    action_type: str,
    content: dict[str, Any],
    *,
    customer_id: str = "cus_test",
    status: str = "pending",
) -> ActionItem:
    return ActionItem(
        id=item_id,
        critique_id=critique_id,
        customer_id=customer_id,
        action_type=action_type,
        content=content,
        status=status,
    )


def _make_critique(
    critique_id: str,
    critic_role: str,
    *,
    critic_model: str = "test-model",
    customer_id: str = "cus_test",
) -> Critique:
    return Critique(
        id=critique_id,
        customer_id=customer_id,
        critic_role=critic_role,  # type: ignore[arg-type]
        critic_model=critic_model,
        observations=[],
        summary_text=None,
        total_action_items=0,
    )


def _make_alignment(
    item_id: str,
    alignment_class: str,
    paired_ids: list[str],
    *,
    customer_id: str = "cus_test",
) -> ItemAlignment:
    return ItemAlignment(
        id=f"al_{item_id}",
        customer_id=customer_id,
        trace_id="trace_test",
        focus_item_id=item_id,
        alignment_class=alignment_class,  # type: ignore[arg-type]
        paired_item_ids=paired_ids,
        confidence=1.0,
        computed_at=datetime.now(timezone.utc),
    )


def _make_calibration(
    critic_role: str,
    critic_model: str,
    *,
    total: int,
    promoted: int,
    customer_id: str = "cus_test",
) -> CriticCalibration:
    """Build a calibration row with `total` proposals, `promoted` of
    which were promoted (the rest counted as deferred).
    """
    return CriticCalibration(
        id=f"cal_{critic_role}_{critic_model}",
        customer_id=customer_id,
        critic_role=critic_role,
        critic_model=critic_model,
        total_proposals=total,
        promoted_count=promoted,
        deferred_count=total - promoted,
        dropped_count=0,
    )


# ===========================================================================
# CU-27 — per-customer shadow cost cap
# ===========================================================================

@pytest.mark.asyncio
async def test_h1_customer_shadow_cap_round_trip(store: Any, customer: Any):
    """CU-27 / P0.111 — get/set_customer_shadow_cap round-trip.

    Covers: default is None (no override); setting an integer override
    and reading it back; clearing it (set None); a missing customer
    returns False on set and None on get; a negative cap raises.
    """
    # Fresh customer has no override.
    assert await store.get_customer_shadow_cap_override(customer.id) is None

    # Set an explicit override.
    updated = await store.set_customer_shadow_cap(customer.id, 250)
    assert updated is True
    assert await store.get_customer_shadow_cap_override(customer.id) == 250

    # Setting 0 is valid (disables shadowing for the customer).
    await store.set_customer_shadow_cap(customer.id, 0)
    assert await store.get_customer_shadow_cap_override(customer.id) == 0

    # Clear the override (revert to global default).
    await store.set_customer_shadow_cap(customer.id, None)
    assert await store.get_customer_shadow_cap_override(customer.id) is None

    # Missing customer: set returns False, get returns None.
    assert await store.set_customer_shadow_cap("cus_nonexistent", 50) is False
    assert await store.get_customer_shadow_cap_override("cus_nonexistent") is None

    # Negative cap is rejected.
    with pytest.raises(ValueError):
        await store.set_customer_shadow_cap(customer.id, -1)


@pytest.mark.asyncio
async def test_h2_worker_honors_per_customer_override(
    store: Any, customer: Any
):
    """CU-27 / P0.111 — the shadow pass uses the customer's override
    when it's tighter than the global default.

    Customer override = 0 (no budget). Global default passed = 100.
    The eligible trace must be skipped for cost cap (override wins),
    and no shadow critique runs. The cost-cap check happens BEFORE
    the anthropic client is touched, so a dummy non-None client is
    sufficient — the LLM is never called.
    """
    await store.set_customer_shadow_cap(customer.id, 0)

    trace = await store.create_reasoning_trace(
        customer_id=customer.id, sequence_id="seq_h2", events=[],
    )
    await store.create_critique(
        customer_id=customer.id,
        critic_role="agent_self",
        critic_model="test-model",
        trace_id=trace.id,
    )

    # Ready but unscripted seam fake — raises if ever invoked (the cap
    # is hit first, so the LLM path must never run).
    set_llm_client(FakeAnthropic())
    try:
        out = await _shadow_pass(
            store=store,
            shadow_max_per_day=100,  # global default, overridden by customer's 0
        )
    finally:
        reset_llm_client()

    assert out["skipped_cost_cap"] == 1
    assert out["shadowed"] == 0


@pytest.mark.asyncio
async def test_h3_worker_falls_through_to_global_default(
    store: Any, customer: Any
):
    """CU-27 / P0.111 — with no per-customer override, the shadow pass
    uses the injected global default.

    No override set (stays None). Global default passed = 0 → the
    eligible trace is skipped for cost cap via the fallthrough path.
    """
    trace = await store.create_reasoning_trace(
        customer_id=customer.id, sequence_id="seq_h3", events=[],
    )
    await store.create_critique(
        customer_id=customer.id,
        critic_role="agent_self",
        critic_model="test-model",
        trace_id=trace.id,
    )

    set_llm_client(FakeAnthropic())
    try:
        out = await _shadow_pass(
            store=store,
            shadow_max_per_day=0,  # global default; no customer override exists
        )
    finally:
        reset_llm_client()

    assert out["skipped_cost_cap"] == 1
    assert out["shadowed"] == 0


# ===========================================================================
# CU-28 — drop-on-low-trust-critic
# ===========================================================================

def test_h4_low_trust_drop_rule_pure_function():
    """CU-28 / P0.112 — synthesize_for_trace drop behavior + guards.

    A low-trust shadow critic (2/30 promoted = 6.7% < 10%, 30 >= 20
    samples) has its would-be-deferred divergent item DROPPED. Its
    substrate observation is EXEMPT (still deferred). A high-trust
    agent_self critic's divergent item is PROMOTED (unaffected).

    Guards verified: passing calibrations_by_critic=None reproduces
    pre-Phase-12 behavior (no drops); a critic below the sample-size
    floor is not low-trust; a critic with a healthy promotion rate is
    not low-trust.
    """
    critiques = {
        "crit_shadow": _make_critique("crit_shadow", "shadow"),
        "crit_agent": _make_critique("crit_agent", "agent_self"),
    }

    item_shadow_div = _make_item(
        "i_shadow_div", "crit_shadow", "research_task", {"topic": "x"},
    )
    item_shadow_sub = _make_item(
        "i_shadow_sub", "crit_shadow", "substrate_observation",
        {"subsystem": "retrieval", "complaint": "noisy"},
    )
    item_agent_div = _make_item(
        "i_agent_div", "crit_agent", "research_task", {"topic": "y"},
    )

    alignments = {
        "i_shadow_div": _make_alignment("i_shadow_div", "divergent_action", []),
        "i_agent_div": _make_alignment("i_agent_div", "divergent_action", []),
        # substrate item handled before alignment lookup.
    }

    pending = [item_shadow_div, item_shadow_sub, item_agent_div]

    # --- Low-trust shadow critic present. -----------------------------
    low_trust = {
        ("shadow", "test-model"): _make_calibration(
            "shadow", "test-model", total=30, promoted=2,
        ),
    }
    promoted, deferred, dropped, rationales = synthesize_for_trace(
        pending_items=pending,
        critiques_by_id=critiques,
        alignments_by_focus_id=alignments,
        calibrations_by_critic=low_trust,
    )
    # Low-trust shadow's divergent item → DROPPED.
    assert "i_shadow_div" in dropped
    assert rationales["i_shadow_div"] == RATIONALE_DROPPED_LOW_TRUST_CRITIC
    # Substrate exempt → still deferred despite low-trust critic.
    assert "i_shadow_sub" in deferred
    assert rationales["i_shadow_sub"] == RATIONALE_SUBSTRATE_DEFERRED
    # Agent_self divergent → promoted (drop rule never touches promotes).
    assert "i_agent_div" in promoted
    assert rationales["i_agent_div"] == RATIONALE_AGENT_SELF_SOLO

    # --- Guard 1: no calibrations → no drops (v1 behavior). -----------
    p2, d2, dr2, r2 = synthesize_for_trace(
        pending_items=pending,
        critiques_by_id=critiques,
        alignments_by_focus_id=alignments,
        calibrations_by_critic=None,
    )
    assert dr2 == []
    assert "i_shadow_div" in d2
    assert r2["i_shadow_div"] == RATIONALE_SHADOW_SOLO

    # --- Guard 2: below sample-size floor → not low-trust. ------------
    insufficient = {
        ("shadow", "test-model"): _make_calibration(
            "shadow", "test-model", total=10, promoted=0,  # 0% but <20 samples
        ),
    }
    _, d3, dr3, r3 = synthesize_for_trace(
        pending_items=pending,
        critiques_by_id=critiques,
        alignments_by_focus_id=alignments,
        calibrations_by_critic=insufficient,
    )
    assert dr3 == []
    assert "i_shadow_div" in d3

    # --- Guard 3: healthy promotion rate → not low-trust. -------------
    healthy = {
        ("shadow", "test-model"): _make_calibration(
            "shadow", "test-model", total=40, promoted=20,  # 50% >> 10%
        ),
    }
    _, d4, dr4, _ = synthesize_for_trace(
        pending_items=pending,
        critiques_by_id=critiques,
        alignments_by_focus_id=alignments,
        calibrations_by_critic=healthy,
    )
    assert dr4 == []
    assert "i_shadow_div" in d4


@pytest.mark.asyncio
async def test_h5_engine_drops_low_trust_critic_item(
    store: Any, customer: Any
):
    """CU-28 / P0.112 — end-to-end: a seeded low-trust calibration
    drives the engine to drop a shadow critic's solo item.

    Seeds (shadow, claude-opus-4-7) at 2/30 promoted (low-trust), then
    runs a trace with a single shadow divergent-solo item. Expected:
    the item is dropped (not deferred), recorded in the synthesis row's
    dropped_item_ids with the frozen rationale, and transitioned to
    status='dropped'.
    """
    await store.upsert_critic_calibration(
        customer_id=customer.id,
        critic_role="shadow",
        critic_model="claude-opus-4-7",
        promoted_delta=2,
        deferred_delta=28,
    )

    trace = await store.create_reasoning_trace(
        customer_id=customer.id, sequence_id="seq_h5", events=[],
    )
    crit_shadow = await store.create_critique(
        customer_id=customer.id,
        critic_role="shadow",
        critic_model="claude-opus-4-7",
        trace_id=trace.id,
    )
    item = await store.create_action_item(
        critique_id=crit_shadow.id,
        customer_id=customer.id,
        action_type="research_task",
        content={"topic": "shadow solo, low-trust critic"},
    )

    result = await compute_alignment_and_synthesis_for_trace(
        store=store, trace_id=trace.id,
    )

    assert result["reason"] == "synthesized"
    assert result["dropped_count"] == 1
    assert result["deferred_count"] == 0
    assert result["promoted_count"] == 0

    # Synthesis row records the drop with the frozen rationale.
    syntheses = await store.list_syntheses_for_trace(trace.id)
    assert len(syntheses) == 1
    synth = syntheses[0]
    assert item.id in synth.dropped_item_ids
    assert item.id not in synth.deferred_item_ids
    assert (
        synth.promotion_rationales[item.id]
        == RATIONALE_DROPPED_LOW_TRUST_CRITIC
    )

    # Item transitioned to 'dropped'.
    items = await store.list_action_items_for_critique(crit_shadow.id)
    assert len(items) == 1
    assert items[0].status == "dropped"
