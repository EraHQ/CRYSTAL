"""RC-08 (2026-10-05): workers do not fail silently.

- A worker loop task that dies is logged at ERROR with the worker's name
  and restarted with backoff (workers.supervise).
- A document a dead worker left 'crystallizing' is returned to the
  status it was claimed from once it is older than the window
  (reclaim_stale_document_claims), so another worker picks it up.
- Production log lines are JSON with `severity` and `message`
  (CC_LOG_FORMAT=json), the fields Cloud Logging alerts read.
"""
import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from crystal_cache.infrastructure.schema import DocumentUploadRow
from crystal_cache.workers import supervise


@pytest.mark.asyncio
async def test_supervise_restarts_a_crashed_loop_and_logs_error(monkeypatch):
    shutdown = asyncio.Event()
    runs = {"n": 0}
    records = []

    async def flaky(*, store, shutdown_event):
        # Same shape as every real worker: keyword-only, takes
        # shutdown_event=. The first deploy of supervise collided on that
        # name because this fake did not have it.
        runs["n"] += 1
        if runs["n"] == 1:
            raise RuntimeError("loop exploded")
        shutdown_event.set()  # second run is the healthy one; end the test

    import structlog
    # supervise fetches its logger at call time, so this is the seam.
    monkeypatch.setattr(structlog, "get_logger", lambda *a, **k: _Capture(records))
    await asyncio.wait_for(
        _with_fast_backoff(supervise, "test_worker", shutdown, flaky,
                           store=object(), shutdown_event=shutdown), timeout=5,
    )
    assert runs["n"] == 2
    crashed = [r for r in records if r["event"] == "worker.crashed"]
    assert len(crashed) == 1
    assert crashed[0]["level"] == "error"
    assert crashed[0]["kw"]["worker"] == "test_worker"


class _Capture:
    def __init__(self, sink):
        self._sink = sink

    def _log(self, level, event, **kw):
        self._sink.append({"level": level, "event": event, "kw": kw})

    def error(self, event, **kw):
        self._log("error", event, **kw)

    def info(self, event, **kw):
        self._log("info", event, **kw)

    def warning(self, event, **kw):
        self._log("warning", event, **kw)


async def _with_fast_backoff(supervise_fn, name, shutdown, fn, **kwargs):
    """supervise's first backoff is five seconds; the pin cannot wait on
    that, so the shutdown event is used as the clock: the healthy run
    sets it, which also ends the backoff wait early."""
    original_wait_for = asyncio.wait_for

    async def quick_wait_for(awaitable, timeout):
        return await original_wait_for(awaitable, timeout=min(timeout, 0.05))

    asyncio.wait_for = quick_wait_for  # restored below
    try:
        await supervise_fn(name, shutdown, fn, **kwargs)
    finally:
        asyncio.wait_for = original_wait_for


def test_every_supervised_start_passes_shutdown_event_through(monkeypatch):
    """Both processes start every worker as
    supervise(<name>, shutdown_event, run_x, ..., shutdown_event=shutdown_event).
    The supervisor's own parameter must not be called shutdown_event or
    the kwargs collide (the v125 startup failure)."""
    import inspect
    import re

    from crystal_cache import app as app_mod
    from crystal_cache.workers import __main__ as main_mod, supervise as sup

    assert "shutdown_event" not in inspect.signature(sup).parameters
    for mod in (app_mod, main_mod):
        src = inspect.getsource(mod)
        starts = re.findall(r"supervise\(\"(\w+)\", shutdown_event, (run_\w+),", src)
        assert len(starts) >= 6, (mod.__name__, starts)
        for name, fn in starts:
            # The worker's own kwargs must include shutdown_event=.
            block = src.split(f'supervise("{name}", shutdown_event, {fn},', 1)[1].split("))", 1)[0]
            assert "shutdown_event=shutdown_event" in block, (mod.__name__, name)


@pytest.mark.asyncio
async def test_stale_crystallizing_claims_are_reclaimed_to_their_prior_status(store, customer):
    now = datetime.now(timezone.utc)
    async with store.session() as s:
        s.add(DocumentUploadRow(id="doc_stale_approved", customer_id=customer.id, label="a", text="t",
                                status="crystallizing", claimed_from="approved",
                                claimed_at=now - timedelta(minutes=45)))
        s.add(DocumentUploadRow(id="doc_stale_pending", customer_id=customer.id, label="p", text="t",
                                status="crystallizing", claimed_from="pending",
                                claimed_at=now - timedelta(minutes=45)))
        s.add(DocumentUploadRow(id="doc_fresh", customer_id=customer.id, label="f", text="t",
                                status="crystallizing", claimed_from="approved",
                                claimed_at=now - timedelta(minutes=2)))
    reclaimed = await store.reclaim_stale_document_claims(older_than_minutes=30, now=now)
    assert reclaimed == 2
    async with store.session() as s:
        assert (await s.get(DocumentUploadRow, "doc_stale_approved")).status == "approved"
        assert (await s.get(DocumentUploadRow, "doc_stale_pending")).status == "pending"
        fresh = await s.get(DocumentUploadRow, "doc_fresh")
        assert fresh.status == "crystallizing" and fresh.claimed_at is not None


@pytest.mark.asyncio
async def test_claim_stamps_when_and_from_what(store, customer):
    async with store.session() as s:
        s.add(DocumentUploadRow(id="doc_claim", customer_id=customer.id, label="c", text="t",
                                status="approved"))
    rows = await store.claim_approved_documents_batch(limit=5)
    assert [r.id for r in rows] == ["doc_claim"]
    async with store.session() as s:
        row = await s.get(DocumentUploadRow, "doc_claim")
        assert row.status == "crystallizing"
        assert row.claimed_from == "approved" and row.claimed_at is not None


def test_json_log_lines_carry_severity_and_message(monkeypatch):
    """The production chain (CC_LOG_FORMAT=json) renders ERROR as a JSON
    object with severity=ERROR and message=<event>."""
    import io

    import structlog

    from crystal_cache import _configure_logging

    monkeypatch.setenv("CC_LOG_FORMAT", "json")
    structlog.reset_defaults()
    _configure_logging()
    out = io.StringIO()
    structlog.configure(
        processors=structlog.get_config()["processors"],
        logger_factory=structlog.PrintLoggerFactory(file=out),
    )
    structlog.get_logger("rc08").error("worker.crashed", worker="purge")
    line = json.loads(out.getvalue().strip().splitlines()[-1])
    assert line["severity"] == "ERROR"
    assert line["message"] == "worker.crashed"
    assert line["worker"] == "purge"
    structlog.reset_defaults()
    monkeypatch.delenv("CC_LOG_FORMAT", raising=False)
    _configure_logging()
