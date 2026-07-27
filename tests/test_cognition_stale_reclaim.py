"""Stale-run reclaim (2026-07-26) — requeue may take an abandoned `running`.

The incident: an api+worker deploy replaced the executor mid-run. The
cognition_tasks row stayed 'running' with no process behind it, forever,
because claim_pending_cognition_task only claims 'pending' and the manual
Re-run endpoint 409'd on 'running'. The one state that needed reclaiming
was the one state nothing could reclaim, and the run sat at 40 minutes.

The evidence to tell abandoned from slow was already on disk:
cognition_runs.updated_at has carried onupdate=utcnow since S9
(2026-07-08), so every lifecycle transition stamps it. Nothing read it.
These tests pin the read and the threshold, including the case that
must STILL be refused — a live run, where requeueing would put two
executors on one task_id.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from crystal_cache.cognition.api import _STALE_RUN_MINUTES, requeue_task


class _Req:
    """Minimal Request stand-in: the endpoint only reads state.tenant_pin."""

    def __init__(self, pin=None):
        self.state = type("S", (), {"tenant_pin": pin})()


@pytest.fixture(autouse=True)
def _register_store(store):
    """requeue_task resolves its store through the module-level
    get_metadata_store() (inline, not Depends), which the app lifespan
    normally sets. Calling the route function directly skips the lifespan.
    set_metadata_store(store) is the codebase's established pattern for
    this (test_endpoint_smoke.py does the same); the delta here is
    restoring the PREVIOUS value rather than leaving ours behind, so
    interleaving with other endpoint-test modules stays safe."""
    from crystal_cache.infrastructure import metadata_store as ms

    previous = ms._store
    ms.set_metadata_store(store)
    yield
    ms.set_metadata_store(previous)


async def _body(resp):
    import json
    return json.loads(resp.body)


async def _running_task(store, customer, *, heartbeat_age_minutes=None,
                        started_age_minutes=None, with_run=True):
    """A claimed task, optionally with a run whose heartbeat is aged."""
    task = await store.create_cognition_task(
        customer.id,
        task_type="research",
        payload={"topic": "landed cost"},
        priority="background",
    )
    claimed = await store.claim_pending_cognition_task()
    assert claimed is not None and claimed.id == task.id

    if started_age_minutes is not None:
        await _age_task_started(store, task.id, started_age_minutes)

    if with_run:
        await store.upsert_cognition_run(
            f"env_{task.id}",
            customer.id,
            status="working",
            trigger_type="research",
            trigger_id=task.id,
            goal_title="landed cost",
            summary={"id": f"env_{task.id}"},
            detail={"id": f"env_{task.id}"},
        )
        if heartbeat_age_minutes is not None:
            await _age_run_heartbeat(
                store, f"env_{task.id}", heartbeat_age_minutes,
            )
    return task


async def _age_run_heartbeat(store, run_id, minutes):
    """Backdate the heartbeat. Written through the store's own session so
    the test never reaches around the R9 boundary."""
    from crystal_cache.infrastructure.schema import CognitionRunRow

    async with store.session() as session:
        row = await session.get(CognitionRunRow, run_id)
        row.updated_at = datetime.now(timezone.utc) - timedelta(minutes=minutes)


async def _age_task_started(store, task_id, minutes):
    from crystal_cache.infrastructure.schema import CognitionTaskRow

    async with store.session() as session:
        row = await session.get(CognitionTaskRow, task_id)
        row.started_at = datetime.now(timezone.utc) - timedelta(minutes=minutes)


# ---------------------------------------------------------------------------
# The heartbeat read
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_heartbeat_read_returns_the_newest_runs_timestamp(
    store, customer,
):
    task = await _running_task(store, customer)
    beat = await store.latest_run_heartbeat_for_trigger(
        task.id, customer_id=customer.id,
    )
    assert beat is not None


@pytest.mark.asyncio
async def test_heartbeat_read_is_unknown_not_fresh_when_no_run_exists(
    store, customer,
):
    """None means UNKNOWN. A caller that read it as 'fresh' would refuse
    to reclaim exactly the runs that died before writing a snapshot."""
    assert await store.latest_run_heartbeat_for_trigger(
        "trg_never_ran", customer_id=customer.id,
    ) is None
    assert await store.latest_run_heartbeat_for_trigger("") is None


@pytest.mark.asyncio
async def test_heartbeat_read_is_tenant_scoped(store, customer):
    task = await _running_task(store, customer)
    assert await store.latest_run_heartbeat_for_trigger(
        task.id, customer_id="cus_someone_else",
    ) is None


# ---------------------------------------------------------------------------
# The reclaim decision
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_live_running_task_is_still_refused(store, customer):
    """The load-bearing refusal: two executors on one task_id is worse
    than a hang."""
    task = await _running_task(store, customer, heartbeat_age_minutes=0)
    resp = await requeue_task(_Req(), task.id)
    assert resp.status_code == 409
    assert "still alive" in (await _body(resp))["error"]
    assert (await store.get_cognition_task(task.id)).status == "running"


@pytest.mark.asyncio
async def test_an_abandoned_running_task_is_reclaimed(store, customer):
    task = await _running_task(
        store, customer, heartbeat_age_minutes=_STALE_RUN_MINUTES + 5,
    )
    resp = await requeue_task(_Req(), task.id)
    assert resp.status_code == 200
    assert (await _body(resp))["requeued"] is True
    # Back to pending means the worker's claim query can see it again.
    assert (await store.get_cognition_task(task.id)).status == "pending"
    assert await store.claim_pending_cognition_task() is not None


@pytest.mark.asyncio
async def test_falls_back_to_claim_time_when_the_engine_never_snapshotted(
    store, customer,
):
    """A task claimed by a process that died before writing any run row.
    started_at is stamped by the claim itself, so staleness is still
    judgeable."""
    task = await _running_task(
        store, customer, with_run=False,
        started_age_minutes=_STALE_RUN_MINUTES + 5,
    )
    resp = await requeue_task(_Req(), task.id)
    assert resp.status_code == 200
    assert (await store.get_cognition_task(task.id)).status == "pending"


@pytest.mark.asyncio
async def test_pending_is_still_a_conflict(store, customer):
    task = await store.create_cognition_task(
        customer.id, task_type="research", payload={}, priority="background",
    )
    resp = await requeue_task(_Req(), task.id)
    assert resp.status_code == 409
    assert "already pending" in (await _body(resp))["error"]


@pytest.mark.asyncio
async def test_terminal_tasks_requeue_unchanged(store, customer):
    """The 2026-07-16 behaviour must survive: a failed task requeues with
    no staleness question asked."""
    task = await store.create_cognition_task(
        customer.id, task_type="research", payload={}, priority="background",
    )
    await store.claim_pending_cognition_task()
    await store.mark_cognition_task_failed(
        task.id, error_message="validator rejected",
        completed_at=datetime.now(timezone.utc),
    )
    resp = await requeue_task(_Req(), task.id)
    assert resp.status_code == 200
    assert (await store.get_cognition_task(task.id)).status == "pending"


@pytest.mark.asyncio
async def test_foreign_task_is_not_an_existence_oracle(store, customer):
    task = await _running_task(
        store, customer, heartbeat_age_minutes=_STALE_RUN_MINUTES + 5,
    )
    resp = await requeue_task(_Req(pin="cus_someone_else"), task.id)
    assert resp.status_code == 404
    assert (await store.get_cognition_task(task.id)).status == "running"
