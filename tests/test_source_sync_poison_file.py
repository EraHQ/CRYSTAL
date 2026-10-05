"""RC-12 (2026-10-05): a poisoned file cannot burn money forever.

A file that failed extraction or encoding was retried on every sync
cycle with no attempt counter, at a cadence as low as one minute, each
try a paid call. Now attempts are counted per file per repository head
(beside the handler's "head" in last_state), a file is skipped after
MAX_FILE_ATTEMPTS until the head moves, the cycle advances without it,
the watch loop passes the tenant door before spending, and the cadence
floor is 15 minutes.

The pin registers a test handler whose poll always reports two changed
files, makes ingestion fail for one of them, runs ten cycles, and counts
how many times the poisoned file was actually attempted.
"""
import pytest

from crystal_cache.ingestion.source_handlers import (
    ChangeSet, SourceEnvelope, register_handler,
)
from crystal_cache.workers import source_sync


class _Handler:
    scheme = "testrepo"

    def __init__(self, head="h1"):
        self.head = head

    async def check(self, watch, token):
        # Faithful to GitSourceHandler.check (git_handler.py:115-131): an
        # unchanged head is one request and None; otherwise both files
        # are reported for the current head. Mirrors a repo whose head
        # never moves while one file keeps failing.
        if (watch.last_state or {}).get("head") == self.head:
            return None
        return ChangeSet(new_state={"head": self.head}, changed=["bad.md", "good.md"])

    async def fetch(self, watch, path, token):
        return SourceEnvelope(payload_bytes=b"x", mime_type="text/plain",
                              source_uri=f"repo://r/{path}", label=path)


@pytest.fixture
def harness(store, customer, monkeypatch):
    handler = _Handler()
    register_handler(handler)
    calls = {"bad": 0, "good": 0}

    async def _ingest(store_, encoder, vs, fvs, llm, watch, envelope):
        if envelope.label == "bad.md":
            calls["bad"] += 1
            raise RuntimeError("encoding exploded")
        calls["good"] += 1

    monkeypatch.setattr(source_sync, "_ingest_envelope", _ingest)
    return handler, calls


async def _watch(store, customer):
    return await store.create_source_watch(
        customer.id, scheme="testrepo", source_name="r", config={},
        cadence_minutes=15, review_mode="auto", encrypted_token=None,
    )


async def _cycle(store, watch_id, customer_id):
    w = await store.get_source_watch(watch_id, customer_id)
    return await source_sync.sync_one_watch(store, None, None, None, None, w)


@pytest.mark.asyncio
async def test_poisoned_file_is_attempted_three_times_then_skipped(store, customer, harness):
    _handler, calls = harness
    w = await _watch(store, customer)
    for _ in range(10):
        await _cycle(store, w.id, customer.id)
    assert calls["bad"] == source_sync.MAX_FILE_ATTEMPTS
    # The good file is re-ingested only while the head is held back by
    # the live failure (dedup replaces it); once the poisoned file is
    # skipped the head advances and the next seven polls are no-ops.
    assert calls["good"] == source_sync.MAX_FILE_ATTEMPTS
    w = await store.get_source_watch(w.id, customer.id)
    assert (w.last_state or {}).get("head") == "h1"
    assert (w.last_state or {})[source_sync.FILE_FAILURES_KEY]["bad.md"]["attempts"] == 3
    assert "skipped" in (w.last_error or "")


@pytest.mark.asyncio
async def test_a_new_head_retries_the_file(store, customer, harness):
    handler, calls = harness
    w = await _watch(store, customer)
    for _ in range(5):
        await _cycle(store, w.id, customer.id)
    assert calls["bad"] == 3
    handler.head = "h2"  # the repo moved; maybe the file was fixed
    for _ in range(5):
        await _cycle(store, w.id, customer.id)
    assert calls["bad"] == 6  # three fresh attempts for the new head


@pytest.mark.asyncio
async def test_walled_tenant_syncs_nothing(store, customer, harness, monkeypatch):
    _handler, calls = harness
    w = await _watch(store, customer)
    await store.set_customer_subscription(customer.id, "free", None)
    await store.set_customer_inference_mode(customer.id, "managed")

    async def _facts(cid):
        return 10**6  # over the fact cap

    monkeypatch.setattr(store, "count_billable_facts", _facts)
    # The loop body that the worker runs per watch: door first.
    from crystal_cache.ingress.auth import require_write_capacity
    from fastapi import HTTPException

    c = await store.get_customer_by_id(customer.id)
    with pytest.raises(HTTPException):
        await require_write_capacity(c, store, spend=True)
    # And the loop itself skips the watch and records why.
    await source_sync._sync_due_watches(store, None, None, None, None)
    w = await store.get_source_watch(w.id, customer.id)
    assert "waiting on plan capacity" in (w.last_error or "")
    assert calls["bad"] == 0 and calls["good"] == 0


def test_cadence_floor_is_fifteen_minutes():
    import inspect

    from crystal_cache.endpoints import admin

    assert 'max(15, int(body.get("cadence_minutes") or 15))' in inspect.getsource(admin)
