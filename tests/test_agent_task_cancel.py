"""--cancel: cancel a queued task / stop a recurring series (backlog §8).

Exercises store.cancel_agent_task across the task lifecycle: a queued task
becomes terminal 'cancelled' (and the daemon's claim skips it), a recurring
occurrence's series is stopped (recur_seconds nulled), a running occurrence is
left to finish but won't recur, and terminal / missing tasks are honest no-ops.
asyncio_mode=auto — plain async tests need no marker.
"""
from __future__ import annotations


async def _queue(store, customer, **kw):
    return await store.create_agent_task(
        customer.id, project_dir="/tmp/proj", task="do a thing", **kw
    )


async def test_cancel_queued_task_marks_cancelled(store, customer):
    t = await _queue(store, customer)
    res = await store.cancel_agent_task(t["id"])
    assert res["found"] is True
    assert res["outcome"] == "cancelled"
    assert res["status"] == "queued"          # the PRIOR status
    row = await store.get_agent_task(t["id"])
    assert row["status"] == "cancelled"


async def test_cancelled_task_is_not_claimed(store, customer):
    t = await _queue(store, customer)
    await store.cancel_agent_task(t["id"])
    # The daemon's claim filters status='queued' — a cancelled task is skipped.
    assert await store.claim_next_agent_task() is None


async def test_cancel_queued_recurring_stops_the_series(store, customer):
    t = await _queue(store, customer, recur_seconds=1800)
    res = await store.cancel_agent_task(t["id"])
    assert res["outcome"] == "cancelled"
    assert res["was_recurring"] is True
    row = await store.get_agent_task(t["id"])
    assert row["status"] == "cancelled"
    assert row["recur_seconds"] is None       # successor can never be scheduled


async def test_cancel_running_recurring_stops_recurrence_keeps_status(store, customer):
    t = await _queue(store, customer, recur_seconds=1800)
    claimed = await store.claim_next_agent_task()   # -> running
    assert claimed["id"] == t["id"] and claimed["status"] == "running"
    res = await store.cancel_agent_task(t["id"])
    assert res["outcome"] == "recurrence_stopped"
    assert res["status"] == "running"
    row = await store.get_agent_task(t["id"])
    assert row["status"] == "running"           # the in-flight run is left alone
    assert row["recur_seconds"] is None          # but the series won't recur


async def test_cancel_running_nonrecurring_is_uncancelable(store, customer):
    t = await _queue(store, customer)
    await store.claim_next_agent_task()          # -> running
    res = await store.cancel_agent_task(t["id"])
    assert res["outcome"] == "running_uncancelable"
    row = await store.get_agent_task(t["id"])
    assert row["status"] == "running"


async def test_cancel_terminal_task_is_noop(store, customer):
    t = await _queue(store, customer)
    await store.claim_next_agent_task()
    await store.finish_agent_task(t["id"], status="done", report="ok")
    res = await store.cancel_agent_task(t["id"])
    assert res["outcome"] == "already_terminal"
    assert res["status"] == "done"
    row = await store.get_agent_task(t["id"])
    assert row["status"] == "done"               # unchanged


async def test_cancel_missing_task_not_found(store, customer):
    res = await store.cancel_agent_task("atask_does_not_exist")
    assert res["found"] is False
    assert res["outcome"] == "not_found"
