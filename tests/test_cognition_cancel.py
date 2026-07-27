"""Cooperative cancellation (2026-07-27) — flag, endpoint, engine boundary.

Born from two orphaned runs in one night: api+worker deploys replaced the
executor mid-run twice, and the only stop the operator ever had was
watching a run spend. Three mechanisms, one flag:

  pending          — endpoint finalizes directly (nothing to cooperate)
  running + live   — flag set; the ENGINE stops at its next step/attempt
                     boundary, never mid-LLM-call
  running + stale  — orphan: endpoint finalizes the task AND its frozen
                     run rows (the gravestone cleanup), same 10-minute
                     heartbeat test the stale reclaim uses

The engine test drives run_cognition_workflow for real: the attempt-top
cancel check fires before run_orchestrator, so no model and no mocks.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from crystal_cache.cognition.api import (
    _STALE_RUN_MINUTES, cancel_task, requeue_task,
)


class _Req:
    def __init__(self, pin=None):
        self.state = type("S", (), {"tenant_pin": pin})()


@pytest.fixture(autouse=True)
def _register_store(store):
    """Same pattern as test_cognition_stale_reclaim: the endpoints
    resolve the store through the module global; restore the previous
    value so module interleaving stays safe."""
    from crystal_cache.infrastructure import metadata_store as ms

    previous = ms._store
    ms.set_metadata_store(store)
    yield
    ms.set_metadata_store(previous)


async def _body(resp):
    import json
    return json.loads(resp.body)


async def _claimed_task(store, customer):
    task = await store.create_cognition_task(
        customer.id, task_type="research",
        payload={"topic": "landed cost"}, priority="background",
    )
    claimed = await store.claim_pending_cognition_task()
    assert claimed is not None and claimed.id == task.id
    return task


async def _run_for(store, customer, task_id, *, age_minutes=None,
                   status="working"):
    run_id = f"env_{task_id}"
    await store.upsert_cognition_run(
        run_id, customer.id, status=status, trigger_type="research",
        trigger_id=task_id, goal_title="landed cost",
        summary={"id": run_id, "status": status},
        detail={"id": run_id, "status": status},
    )
    if age_minutes is not None:
        from crystal_cache.infrastructure.schema import CognitionRunRow
        async with store.session() as session:
            row = await session.get(CognitionRunRow, run_id)
            row.updated_at = (
                datetime.now(timezone.utc) - timedelta(minutes=age_minutes)
            )
    return run_id


# ---------------------------------------------------------------------------
# The flag
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_request_sets_flag_without_touching_status(store, customer):
    task = await _claimed_task(store, customer)
    assert await store.request_cognition_cancel(task.id) is True
    after = await store.get_cognition_task(task.id)
    assert after.cancel_requested is True
    assert after.status == "running"          # a REQUEST, not a state


@pytest.mark.asyncio
async def test_request_noops_on_terminal(store, customer):
    task = await _claimed_task(store, customer)
    await store.mark_cognition_task_failed(
        task.id, error_message="x",
        completed_at=datetime.now(timezone.utc),
    )
    assert await store.request_cognition_cancel(task.id) is False


@pytest.mark.asyncio
async def test_missing_task_reads_as_cancel(store):
    """A task row that vanished mid-run has nothing to run FOR."""
    assert await store.is_cognition_cancel_requested("cog_gone") is True


# ---------------------------------------------------------------------------
# The endpoint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pending_task_cancels_directly(store, customer):
    task = await store.create_cognition_task(
        customer.id, task_type="research", payload={},
        priority="background",
    )
    resp = await cancel_task(_Req(), task.id)
    out = await _body(resp)
    assert resp.status_code == 200 and out["cancelled"] is True
    assert (await store.get_cognition_task(task.id)).status == "cancelled"


@pytest.mark.asyncio
async def test_live_run_gets_the_flag_not_the_axe(store, customer):
    """Finalizing a LIVE run directly would race its executor — a later
    mark_complete would overwrite 'cancelled'. So liveness selects the
    cooperative mechanism."""
    task = await _claimed_task(store, customer)
    await _run_for(store, customer, task.id, age_minutes=0)

    resp = await cancel_task(_Req(), task.id)
    out = await _body(resp)
    assert out["cancelled"] == "requested"
    after = await store.get_cognition_task(task.id)
    assert after.status == "running"          # engine will finalize
    assert after.cancel_requested is True


@pytest.mark.asyncio
async def test_orphan_is_finalized_with_its_gravestones(store, customer):
    task = await _claimed_task(store, customer)
    run_id = await _run_for(
        store, customer, task.id, age_minutes=_STALE_RUN_MINUTES + 5,
    )

    resp = await cancel_task(_Req(), task.id)
    out = await _body(resp)
    assert out["cancelled"] is True
    assert out["runs_finalized"] == 1
    assert (await store.get_cognition_task(task.id)).status == "cancelled"

    # The gravestone left the active list: terminal status, completed_at
    # stamped, and the stored wire blobs patched so the tracker renders
    # it without special-casing.
    run = await store.get_cognition_run(run_id)
    assert run["status"] == "cancelled"
    assert run["completed_at"] is not None
    active = await store.list_cognition_runs(customer.id)
    assert all(r["status"] not in
               ("orchestrating", "working", "validating", "rejected")
               for r in active)


@pytest.mark.asyncio
async def test_terminal_task_noops(store, customer):
    task = await _claimed_task(store, customer)
    await store.mark_cognition_task_complete(
        task.id, result={}, completed_at=datetime.now(timezone.utc),
    )
    resp = await cancel_task(_Req(), task.id)
    out = await _body(resp)
    assert out["cancelled"] is False and "no-op" in out["note"]
    assert (await store.get_cognition_task(task.id)).status == "complete"


@pytest.mark.asyncio
async def test_foreign_task_is_not_an_existence_oracle(store, customer):
    task = await _claimed_task(store, customer)
    resp = await cancel_task(_Req(pin="cus_someone_else"), task.id)
    assert resp.status_code == 404
    assert (await store.get_cognition_task(task.id)).status == "running"


# ---------------------------------------------------------------------------
# The engine boundary
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_engine_honors_the_flag_before_spending(store, customer):
    """The attempt-top check fires BEFORE run_orchestrator, so a flagged
    task exits CANCELLED with zero model calls — this test drives the
    real engine with no mocks and would hang on any LLM attempt."""
    from crystal_cache.cognition.engine import run_cognition_workflow

    task = await _claimed_task(store, customer)
    await store.request_cognition_cancel(task.id)

    result = await run_cognition_workflow(
        goal="landed cost brief",
        customer_id=customer.id,
        store=store,
        fact_store=None,
        encoder=None,
        trigger_type="research",
        trigger_id=task.id,
        max_attempts=3,
    )
    assert result.success is False
    assert result.outcome == "cancelled"
    assert "cancelled by operator" in (result.reason or "")


@pytest.mark.asyncio
async def test_engine_ignores_the_flag_for_gap_triggers(store, customer):
    """fill_gap runs carry a GAP id as trigger_id. The boundary read
    must not consult the task table for one — missing→True would
    self-cancel every sweep run instantly."""
    from crystal_cache.cognition.engine import _cancel_requested

    class _Env:
        trigger_id = "gap_0123456789abcdef"

    assert await _cancel_requested(store, _Env()) is False


# ---------------------------------------------------------------------------
# Requeue's gravestone cleanup
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stale_reclaim_now_buries_its_gravestone(store, customer):
    task = await _claimed_task(store, customer)
    run_id = await _run_for(
        store, customer, task.id, age_minutes=_STALE_RUN_MINUTES + 5,
    )
    resp = await requeue_task(_Req(), task.id)
    assert resp.status_code == 200
    assert (await store.get_cognition_task(task.id)).status == "pending"
    # Abandoned run: 'failed' (its executor died), never 'cancelled'
    # (nobody stopped it) — the two must stay distinguishable.
    run = await store.get_cognition_run(run_id)
    assert run["status"] == "failed"
    assert run["completed_at"] is not None


@pytest.mark.asyncio
async def test_finalize_leaves_terminal_rows_alone(store, customer):
    task = await _claimed_task(store, customer)
    done_id = f"env_done_{task.id}"
    await store.upsert_cognition_run(
        done_id, customer.id, status="complete", trigger_type="research",
        trigger_id=task.id, goal_title="done",
        summary={"id": done_id, "status": "complete"},
        detail={"id": done_id, "status": "complete"},
        terminal=True,
    )
    n = await store.finalize_stale_runs_for_trigger(
        task.id, customer_id=customer.id, status="cancelled",
    )
    assert n == 0
    assert (await store.get_cognition_run(done_id))["status"] == "complete"
