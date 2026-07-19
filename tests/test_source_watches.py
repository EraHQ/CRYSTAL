"""Gate M slice 1 (2026-07-18): source_watches — the general watch
registration. One table for every scheme (M-Q1=A); these pin the CRUD
contract and the due-cycle semantics the sync worker builds on."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


@pytest.mark.asyncio
async def test_watch_lifecycle(store, customer):
    w = await store.create_source_watch(
        customer.id, scheme="git",
        source_name="crystal-cache-v2",
        config={"repo": "https://github.com/EraHQ/crystal-cache-v2",
                "branch": "master"},
        cadence_minutes=15,
    )
    assert w.id.startswith("watch_")
    assert w.review_mode == "auto"          # M-Q3 default

    got = await store.get_source_watch(w.id, customer.id)
    assert got is not None
    assert got.scheme == "git"
    assert got.config["branch"] == "master"

    listed = await store.list_source_watches(customer.id)
    assert [x.id for x in listed] == [w.id]

    assert await store.set_source_watch_status(w.id, customer.id, "paused")
    assert (await store.get_source_watch(w.id, customer.id)).status == "paused"

    assert await store.delete_source_watch(w.id, customer.id)
    assert await store.get_source_watch(w.id, customer.id) is None


@pytest.mark.asyncio
async def test_tenancy_guard(store, customer):
    w = await store.create_source_watch(
        customer.id, scheme="git", source_name="x", config={},
    )
    assert await store.get_source_watch(w.id, "cus_other") is None
    assert not await store.delete_source_watch(w.id, "cus_other")


@pytest.mark.asyncio
async def test_due_cycle_semantics(store, customer):
    now = datetime.now(timezone.utc)
    w = await store.create_source_watch(
        customer.id, scheme="git", source_name="due-test",
        config={}, cadence_minutes=15,
    )
    # Never checked -> due immediately (first sync, M-Q4).
    due = await store.list_source_watches_due(now)
    assert w.id in {x.id for x in due}

    # Freshly checked -> not due.
    await store.update_source_watch_state(
        w.id, customer.id, last_state={"head": "abc123"}, checked_at=now,
    )
    due = await store.list_source_watches_due(now + timedelta(minutes=5))
    assert w.id not in {x.id for x in due}

    # Cadence elapsed -> due again, state intact.
    due = await store.list_source_watches_due(now + timedelta(minutes=16))
    assert w.id in {x.id for x in due}
    got = await store.get_source_watch(w.id, customer.id)
    assert got.last_state == {"head": "abc123"}
    assert got.last_error is None

    # Paused watches never come due.
    await store.set_source_watch_status(w.id, customer.id, "paused")
    due = await store.list_source_watches_due(now + timedelta(hours=1))
    assert w.id not in {x.id for x in due}
