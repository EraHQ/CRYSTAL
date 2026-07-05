"""Phase 10.5 tests — substrate review surface (D-MCR-13 V1).

Per P0.96: 3 tests covering filtering correctness, cross-tenant
scope, and defensive composition. Tests the library function
`list_substrate_observations` end-to-end through the store; the
CLI and HTTP endpoint are thin pass-throughs and don't require
their own tests.

SR1 — filtering: only deferred substrate_observation items appear
  with full critique + trace context; non-substrate items don't
  appear; promoted/pending substrate items don't appear.

SR2 — cross-tenant scope: customer_id=None returns items across
  multiple customers; customer_id="cust_a" scopes correctly.

SR3 — defensive composition: when a substrate item's critique is
  missing from DB (orphaned ID), the view's critique field is
  None and the function continues processing other items.
"""
from __future__ import annotations

import uuid

import pytest

from crystal_cache.metacognition.substrate_review import (
    SubstrateObservationView,
    TraceSummary,
    list_substrate_observations,
)


# ---------------------------------------------------------------------------
# SR1 — filtering: only deferred substrate_observation items appear
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sr1_filters_to_deferred_substrate_only(store, customer):
    """Three action items: substrate-deferred, substrate-pending,
    research-deferred. Only the first appears in the list, with
    full critique + trace context attached.
    """
    # Set up a trace + critique to anchor the action items.
    trace = await store.create_reasoning_trace(
        customer_id=customer.id,
        sequence_id="seq_sr1",
        turn_index=0,
        events=[],
    )
    critique = await store.create_critique(
        customer_id=customer.id,
        critic_role="agent_self",
        critic_model="haiku",
        trace_id=trace.id,
        summary_text="agent's substrate complaint",
        observations=[
            {"type": "substrate_complaint", "text": "retrieval framing pushed",
             "confidence": 0.7, "anchors": []}
        ],
    )

    # Item A: substrate-deferred → SHOULD appear.
    item_substrate_deferred = await store.create_action_item(
        critique_id=critique.id,
        customer_id=customer.id,
        action_type="substrate_observation",
        content={
            "subsystem": "retrieval",
            "complaint": "framing biases toward agreement on contentious topics",
        },
    )
    await store.update_action_item_status(
        item_substrate_deferred.id,
        status="deferred",
    )

    # Item B: substrate-pending → SHOULD NOT appear (wrong status).
    item_substrate_pending = await store.create_action_item(
        critique_id=critique.id,
        customer_id=customer.id,
        action_type="substrate_observation",
        content={"complaint": "different complaint"},
    )
    # No status update — stays pending.

    # Item C: research_task-deferred → SHOULD NOT appear (wrong type).
    item_research_deferred = await store.create_action_item(
        critique_id=critique.id,
        customer_id=customer.id,
        action_type="research_task",
        content={"topic": "test"},
    )
    await store.update_action_item_status(
        item_research_deferred.id,
        status="deferred",
    )

    views = await list_substrate_observations(
        store=store,
        customer_id=customer.id,
    )

    # Only item A appears.
    assert len(views) == 1
    view = views[0]
    assert isinstance(view, SubstrateObservationView)
    assert view.action_item.id == item_substrate_deferred.id
    assert view.action_item.action_type == "substrate_observation"
    assert view.action_item.status == "deferred"

    # Critique was composed in.
    assert view.critique is not None
    assert view.critique.id == critique.id
    assert view.critique.critic_role == "agent_self"
    assert view.critique.summary_text == "agent's substrate complaint"

    # Trace summary was composed in (slim shape — no events).
    assert view.trace_summary is not None
    assert isinstance(view.trace_summary, TraceSummary)
    assert view.trace_summary.trace_id == trace.id
    assert view.trace_summary.sequence_id == "seq_sr1"
    assert view.trace_summary.turn_index == 0


# ---------------------------------------------------------------------------
# SR2 — cross-tenant scope: customer_id=None vs scoped
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sr2_cross_tenant_scope(store):
    """Two customers each have one substrate-deferred observation.
    customer_id=None returns both; customer_id=<cust_a.id> returns
    only cust_a's.
    """
    # Create two customers via the production create_customer path.
    # Signature matches the conftest's `customer` fixture.
    cust_a = await store.create_customer(
        provider="anthropic",
        model_id="claude-sonnet-4-5-20250929",
        api_key_ref="sk-test-cust-a",
    )
    cust_b = await store.create_customer(
        provider="anthropic",
        model_id="claude-sonnet-4-5-20250929",
        api_key_ref="sk-test-cust-b",
    )
    assert cust_a.id != cust_b.id  # sanity

    # For each, create a trace + critique + substrate-deferred item.
    for cust, label in [(cust_a, "a"), (cust_b, "b")]:
        trace = await store.create_reasoning_trace(
            customer_id=cust.id, events=[],
        )
        critique = await store.create_critique(
            customer_id=cust.id,
            critic_role="agent_self",
            critic_model="haiku",
            trace_id=trace.id,
            summary_text=f"customer_{label}'s observation",
        )
        item = await store.create_action_item(
            critique_id=critique.id,
            customer_id=cust.id,
            action_type="substrate_observation",
            content={"label": label},
        )
        await store.update_action_item_status(item.id, status="deferred")

    # Cross-tenant scan: should see both customers' observations.
    cross_views = await list_substrate_observations(
        store=store,
        customer_id=None,
    )
    cross_customer_ids = {v.action_item.customer_id for v in cross_views}
    assert cust_a.id in cross_customer_ids
    assert cust_b.id in cross_customer_ids
    assert len(cross_views) >= 2

    # Customer-scoped scan: should see only cust_a's.
    scoped_views = await list_substrate_observations(
        store=store,
        customer_id=cust_a.id,
    )
    assert len(scoped_views) == 1
    assert scoped_views[0].action_item.customer_id == cust_a.id


# ---------------------------------------------------------------------------
# SR3 — defensive composition: orphaned critique_id → critique=None
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sr3_orphaned_critique_yields_none_field(store, customer):
    """An action item whose critique_id doesn't resolve (e.g.
    critique deleted, race, test setup) should appear in the list
    with critique=None and trace_summary=None, NOT cause the
    function to raise or hide other items.
    """
    # First: a properly-formed substrate-deferred item.
    trace_good = await store.create_reasoning_trace(
        customer_id=customer.id, events=[],
    )
    critique_good = await store.create_critique(
        customer_id=customer.id,
        critic_role="agent_self",
        critic_model="haiku",
        trace_id=trace_good.id,
    )
    item_good = await store.create_action_item(
        critique_id=critique_good.id,
        customer_id=customer.id,
        action_type="substrate_observation",
        content={"label": "well-formed"},
    )
    await store.update_action_item_status(item_good.id, status="deferred")

    # Second: an item with a bogus critique_id (never created).
    bogus_critique_id = uuid.uuid4().hex[:16]
    item_orphan = await store.create_action_item(
        critique_id=bogus_critique_id,
        customer_id=customer.id,
        action_type="substrate_observation",
        content={"label": "orphan"},
    )
    await store.update_action_item_status(item_orphan.id, status="deferred")

    # The library function should return BOTH items without raising.
    views = await list_substrate_observations(
        store=store,
        customer_id=customer.id,
    )
    assert len(views) == 2

    # Find each view by content label.
    by_label = {v.action_item.content["label"]: v for v in views}
    assert "well-formed" in by_label
    assert "orphan" in by_label

    # Well-formed: critique + trace_summary both resolved.
    good_view = by_label["well-formed"]
    assert good_view.critique is not None
    assert good_view.critique.id == critique_good.id
    assert good_view.trace_summary is not None
    assert good_view.trace_summary.trace_id == trace_good.id

    # Orphan: critique is None, trace_summary is None (can't resolve
    # without critique). Action item itself is intact.
    orphan_view = by_label["orphan"]
    assert orphan_view.critique is None
    assert orphan_view.trace_summary is None
    assert orphan_view.action_item.id == item_orphan.id
