"""Scheduled runs (launch feature, 2026-10-18; Q78 A, Q79 A, Q80 A, Q81 B).

Pins from docs/HACKATHON_NEBIUS.md 3.2:
- next occurrence is correct across the DST change on 2026-11-01
  (daily 8:00 AM America/New_York before and after); weekdays skip the
  weekend; weekly lands on the day; hourly aligns to the interval; a
  one-shot in the past has nothing left.
- the due claim picks only due active rows, and the claim is a
  compare-and-set (a stale expected value loses).
- a tick enqueues ONE agent_research cognition task carrying the prompt
  and schedule id, then advances next_run_at.
- a capacity-blocked tenant's occurrence is skipped and recorded
  (Q80 A); the next tick runs normally once capacity is back.
- three consecutive failed runs park the schedule.
- creating a schedule is never capacity-gated; the per-tier active cap
  holds (free 1); the builder refuses bad shapes with a sentence.
- the console routes and the registry tools share the builder.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from crystal_cache.scheduling import BRIEF_PROMPT, ScheduleSpec, build_schedule, next_occurrence, validate_spec
from crystal_cache.workers import scheduler as sched_mod

NY = ZoneInfo("America/New_York")
UTC = timezone.utc


def test_recurrence_presets_and_dst():
    daily = ScheduleSpec(cadence="daily", tz="America/New_York", at_minute=8 * 60)
    oct31 = datetime(2026, 10, 31, 13, 0, tzinfo=UTC)          # 09:00 EDT, after that day's 08:00
    nov1 = next_occurrence(daily, oct31)
    assert nov1 == datetime(2026, 11, 1, 13, 0, tzinfo=UTC)     # 08:00 EST (DST ended at 02:00)
    assert nov1.astimezone(NY).strftime("%H:%M") == "08:00"
    nov2 = next_occurrence(daily, nov1)
    assert nov2.astimezone(NY).strftime("%Y-%m-%d %H:%M") == "2026-11-02 08:00"
    # Weekdays: Friday 09:00 local -> Monday.
    weekdays = ScheduleSpec(cadence="weekdays", tz="America/New_York", at_minute=8 * 60)
    fri = datetime(2026, 10, 30, 13, 0, tzinfo=UTC)
    assert next_occurrence(weekdays, fri).astimezone(NY).strftime("%a %H:%M") == "Mon 08:00"
    weekly = ScheduleSpec(cadence="weekly", tz="America/New_York", at_minute=9 * 60, weekday=0)
    assert next_occurrence(weekly, fri).astimezone(NY).strftime("%a %Y-%m-%d %H:%M") == "Mon 2026-11-02 09:00"
    hourly = ScheduleSpec(cadence="hourly", every_hours=6)
    assert next_occurrence(hourly, datetime(2026, 10, 18, 17, 5, tzinfo=UTC)) == datetime(2026, 10, 18, 18, 0, tzinfo=UTC)
    assert next_occurrence(ScheduleSpec(cadence="hourly", every_hours=5), datetime(2026, 10, 18, 17, 5, tzinfo=UTC)) == datetime(2026, 10, 18, 22, 0, tzinfo=UTC)
    once = ScheduleSpec(cadence="once", run_once_at=datetime(2026, 10, 19, 1, 0, tzinfo=UTC))
    assert next_occurrence(once, datetime(2026, 10, 18, 0, 0, tzinfo=UTC)) == once.run_once_at
    assert next_occurrence(once, datetime(2026, 10, 19, 2, 0, tzinfo=UTC)) is None
    assert "time zone" in validate_spec(ScheduleSpec(cadence="daily", tz="Mars/Olympus", at_minute=0))
    assert validate_spec(ScheduleSpec(cadence="weekly", tz="UTC", at_minute=0)) is not None


def test_builder_shapes():
    kw, err = build_schedule(kind="brief", cadence="weekdays", tz="America/New_York", time="08:00",
                             now=datetime(2026, 10, 17, 12, 0, tzinfo=UTC))
    assert err is None and kw["prompt"] == BRIEF_PROMPT and kw["title"].startswith("Brief (Weekdays at 08:00")
    assert kw["next_run_at"].astimezone(NY).strftime("%a %H:%M") == "Mon 08:00"
    _, err = build_schedule(kind="research", cadence="daily", time="07:30")
    assert "what the run should do" in err
    # Open-ended (2026-10-18): no kind + a task = research.
    kw, err = build_schedule(kind="", cadence="daily", time="07:30", question="Gather news relevant to my knowledge base")
    assert err is None and kw["kind"] == "research" and kw["prompt"].startswith("Gather news")
    kw, err = build_schedule(kind="research", cadence="weekly", weekday="monday", time="9:00", question="What changed in MCP auth?")
    assert err is None and kw["weekday"] == 0 and kw["at_minute"] == 540 and kw["title"].startswith("What changed in MCP auth? (Every Monday at 09:00 UTC)")
    _, err = build_schedule(kind="brief", cadence="daily")
    assert "HH:MM" in err
    _, err = build_schedule(kind="brief", cadence="once", run_at="2020-01-01T00:00:00+00:00")
    assert "past" in err
    _, err = build_schedule(kind="brief", cadence="hourly", every_hours="lots")
    assert "whole number" in err
    _, err = build_schedule(kind="walk", cadence="daily", time="08:00")
    assert "kind" in err


@pytest.mark.asyncio
async def test_tick_claims_due_rows_enqueues_once_and_advances(store, customer, monkeypatch):
    from crystal_cache.control import admission

    async def _open(store_, cust):
        return None
    monkeypatch.setattr(admission, "daily_capacity_block", _open)
    await store.set_customer_inference_mode(customer.id, "managed")
    now = datetime(2026, 10, 18, 13, 0, 30, tzinfo=UTC)
    due = await store.create_scheduled_task(customer.id, kind="brief", title="Brief", prompt=BRIEF_PROMPT,
                                            cadence="daily", tz="America/New_York", at_minute=8 * 60,
                                            next_run_at=now - timedelta(seconds=30))
    future = await store.create_scheduled_task(customer.id, kind="research", title="Later", prompt="q?",
                                               cadence="daily", tz="UTC", at_minute=20 * 60,
                                               next_run_at=now + timedelta(hours=7))
    paused = await store.create_scheduled_task(customer.id, kind="brief", title="Paused", prompt=BRIEF_PROMPT,
                                               cadence="daily", tz="UTC", at_minute=0, next_run_at=now - timedelta(hours=1))
    await store.set_scheduled_task_status(paused.id, customer.id, "paused")
    assert [s.id for s in await store.list_due_scheduled_tasks(now)] == [due.id]

    out = await sched_mod.tick(store, now=now)
    assert out["due"] == 1 and out["queued"] == 1 and out["skipped_capacity"] == 0
    after = await store.get_scheduled_task(due.id, customer.id)
    assert after.last_outcome == "queued" and after.last_task_id
    assert after.next_run_at.replace(tzinfo=UTC).astimezone(NY).strftime("%Y-%m-%d %H:%M") == "2026-10-19 08:00"
    task = await store.get_cognition_task(after.last_task_id)
    assert task.task_type == "agent_research" and task.priority == "background"
    assert task.payload["topic"] == BRIEF_PROMPT and task.payload["schedule_id"] == due.id
    # The same tick again enqueues nothing (the row advanced).
    assert (await sched_mod.tick(store, now=now))["due"] == 0
    # A stale compare-and-set loses.
    assert await store.advance_scheduled_task(future.id, expected_next_run_at=now, next_run_at=now) is False
    assert (await store.get_scheduled_task(future.id, customer.id)).next_run_at.replace(tzinfo=UTC) == now + timedelta(hours=7)


@pytest.mark.asyncio
async def test_capacity_skip_then_failures_park(store, customer, monkeypatch):
    from crystal_cache.control import admission

    blocked = {"on": True}

    async def _door(store_, cust):
        return {"code": "daily_capacity", "error": "x"} if blocked["on"] else None
    monkeypatch.setattr(admission, "daily_capacity_block", _door)
    await store.set_customer_inference_mode(customer.id, "managed")
    t0 = datetime(2026, 10, 18, 0, 0, 30, tzinfo=UTC)
    s = await store.create_scheduled_task(customer.id, kind="research", title="R", prompt="why?",
                                          cadence="hourly", tz="UTC", every_hours=1, next_run_at=t0 - timedelta(seconds=30))
    out = await sched_mod.tick(store, now=t0)
    assert out["skipped_capacity"] == 1 and out["queued"] == 0
    row = await store.get_scheduled_task(s.id, customer.id)
    assert row.last_outcome == "skipped_capacity" and row.last_task_id is None and row.status == "active"
    # Capacity back: the next occurrence runs; then three failed runs park it.
    blocked["on"] = False
    now = t0
    for i in range(3):
        now = now + timedelta(hours=1)
        out = await sched_mod.tick(store, now=now)
        assert out["queued"] == 1, (i, out)
        row = await store.get_scheduled_task(s.id, customer.id)
        await store.mark_cognition_task_failed(row.last_task_id, error_message="boom", completed_at=now)
    now = now + timedelta(hours=1)
    out = await sched_mod.tick(store, now=now)
    # The third failure is settled on this tick and parks the row before any enqueue.
    assert out["parked"] == 1 and out["queued"] == 0
    row = await store.get_scheduled_task(s.id, customer.id)
    assert row.status == "parked" and row.series_failures == 3 and row.last_outcome == "failed"
    # Resume clears the streak and recomputes the next run.
    back = await store.set_scheduled_task_status(s.id, customer.id, "active", next_run_at=now + timedelta(hours=1))
    assert back.status == "active" and back.series_failures == 0


@pytest.mark.asyncio
async def test_routes_cap_and_lifecycle(store, customer, semantic_encoder_stub, vector_store, fact_vector_store):
    try:
        from tests.test_endpoint_smoke import _build_app
    except ModuleNotFoundError:
        from test_endpoint_smoke import _build_app
    from httpx import ASGITransport, AsyncClient

    app = _build_app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    base = f"/admin/api/customers/{customer.id}/schedules"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as cl:
        r = await cl.post(base, json={"kind": "brief", "cadence": "weekdays", "time": "08:00", "tz": "America/New_York"})
        assert r.status_code == 201, r.text[:300]
        sched = r.json()["schedule"]
        assert sched["when"] == "Weekdays at 08:00 America/New_York" and sched["status"] == "active"
        # Free tier (tier None): one schedule. The cap is the plan, not capacity.
        r = await cl.post(base, json={"kind": "research", "cadence": "daily", "time": "07:00", "question": "q?"})
        assert r.status_code == 409 and "allows 1 schedule" in r.json()["detail"]
        r = await cl.get(base)
        assert r.status_code == 200 and r.json()["active"] == 1 and r.json()["max_active"] == 1
        r = await cl.patch(f"{base}/{sched['id']}", json={"status": "paused"})
        assert r.status_code == 200 and r.json()["schedule"]["status"] == "paused"
        r = await cl.patch(f"{base}/{sched['id']}", json={"status": "active"})
        assert r.status_code == 200 and r.json()["schedule"]["next_run_at"]
        r = await cl.post(base, json={"kind": "brief", "cadence": "daily"})
        assert r.status_code == 400 and "HH:MM" in r.json()["detail"]
        r = await cl.delete(f"{base}/{sched['id']}")
        assert r.status_code == 200
        r = await cl.get(base)
        assert r.json()["count"] == 0
        r = await cl.delete(f"{base}/{sched['id']}")
        assert r.status_code == 404


@pytest.mark.asyncio
async def test_registry_tools_share_the_builder(store, customer):
    from crystal_cache.agent.tool_registry import get_registry, import_all_tools
    from crystal_cache.agent.tools.retrievers import set_tool_state

    import_all_tools()
    set_tool_state({"store": store})
    reg = get_registry()
    create, listing, remove = reg.get("schedule_run"), reg.get("list_schedules"), reg.get("unschedule_run")
    assert create.contexts == frozenset({"agent"}) and remove.contexts == frozenset({"agent"})
    out = await create.impl(customer.id, kind="research", cadence="weekly", weekday="friday", time="17:00",
                            tz="America/New_York", question="What changed in MCP auth this week?")
    assert out["scheduled"] is True and out["schedule"]["when"] == "Every Friday at 17:00 America/New_York"
    out2 = await create.impl(customer.id, kind="brief", cadence="daily", time="08:00")
    assert out2["scheduled"] is False and out2["code"] == "schedule_cap"
    bad = await create.impl(customer.id, kind="brief", cadence="daily")
    assert bad["code"] == "bad_arguments"  # shape before cap
    shown = await listing.impl(customer.id)
    assert shown["count"] == 1 and shown["schedules"][0]["question"].startswith("What changed")
    gone = await remove.impl(customer.id, schedule_id=out["schedule"]["schedule_id"])
    assert gone["deleted"] is True
    assert (await remove.impl(customer.id, schedule_id="sched_nope"))["code"] == "not_found"


# ---------------------------------------------------------------------------
# Delivery (Q90=A, 2026-10-18): one https webhook per schedule, shaped per
# host, signed for generic hooks, guarded against private addresses,
# delivered at most once with the outcome on the row; the latest result
# surfaces in MCP status.
# ---------------------------------------------------------------------------
from crystal_cache.scheduling import build_event, check_deliver_url, deliver, run_context, shape, sign  # noqa: E402


def _resolver(mapping):
    def _r(host, port):
        if host not in mapping:
            import socket
            raise socket.gaierror("no such host")
        return mapping[host]
    return _r


def test_deliver_url_guard_blocks_private_and_non_https():
    r = _resolver({"hooks.slack.com": ["3.3.3.3"], "intranet": ["10.0.0.5"], "lo": ["127.0.0.1"],
                   "mapped": ["::ffff:192.168.1.9"], "both": ["3.3.3.3", "10.1.1.1"]})
    assert check_deliver_url("https://hooks.slack.com/services/T/B/x", resolve=r) is None
    assert check_deliver_url("https://8.8.8.8/hook", resolve=r) is None
    assert "https://" in check_deliver_url("http://hooks.slack.com/services/T/B/x", resolve=r)
    assert "public host" in check_deliver_url("https://intranet/hook", resolve=r)
    assert "public host" in check_deliver_url("https://lo/hook", resolve=r)
    assert "public host" in check_deliver_url("https://mapped/hook", resolve=r)
    assert "public host" in check_deliver_url("https://both/hook", resolve=r)
    assert "public host" in check_deliver_url("https://10.0.0.1/hook", resolve=r)
    assert "public host" in check_deliver_url("https://169.254.169.254/latest/meta-data", resolve=r)
    assert "public host" in check_deliver_url("https://[::1]/hook", resolve=r)
    assert "username" in check_deliver_url("https://user:pw@hooks.slack.com/x", resolve=r)
    assert "resolved" in check_deliver_url("https://nope.invalid/x", resolve=r)
    assert "required" in check_deliver_url("", resolve=r)
    assert "too long" in check_deliver_url("https://hooks.slack.com/" + "a" * 2100, resolve=r)


def test_shape_per_host_and_signature():
    class S:
        id, title, kind, prompt = "sched_1", "Morning brief", "brief", "..."
    event = build_event(S(), summary="Two things changed.", crystal_id="crys_9", ran_at="2026-10-19T12:00:00+00:00",
                        console_url="https://inspector.erahq.ai/")
    assert event["console_url"] == "https://inspector.erahq.ai/bank/crys_9" and event["event"] == "scheduled_run.completed"
    slack = shape("https://hooks.slack.com/services/T/B/x", event)
    assert set(slack) == {"text"} and "Morning brief" in slack["text"] and "Two things changed." in slack["text"]
    discord = shape("https://discord.com/api/webhooks/1/abc", event)
    assert set(discord) == {"content"} and "Morning brief" in discord["content"]
    assert shape("https://hooks.zapier.com/hooks/catch/1/x", event) is event
    body = b'{"a":1}'
    assert sign("whsec_k", body) == "sha256=" + __import__("hmac").new(b"whsec_k", body, "sha256").hexdigest()


@pytest.mark.asyncio
async def test_deliver_posts_signed_json_or_slack_text():
    import json
    from httpx import MockTransport, Response

    seen = []

    def handler(request):
        seen.append((str(request.url), dict(request.headers), request.content))
        return Response(500 if "bad" in str(request.url) else 204)
    t = MockTransport(handler)
    r = _resolver({"hooks.zapier.com": ["3.3.3.3"], "hooks.slack.com": ["3.3.3.3"], "bad.example": ["3.3.3.3"],
                   "intranet": ["10.0.0.5"]})
    event = {"event": "scheduled_run.completed", "schedule_id": "sched_1", "title": "T", "kind": "brief",
             "summary": "s", "crystal_id": None, "console_url": "https://inspector.erahq.ai/cognition",
             "question": None, "ran_at": "x"}
    ok, outcome = await deliver("https://hooks.zapier.com/hooks/catch/1/x", "whsec_k", event, transport=t, resolve=r)
    assert ok and outcome == "sent"
    url, headers, content = seen[-1]
    assert json.loads(content) == event
    assert headers["x-crystal-signature"] == sign("whsec_k", content) and headers["x-crystal-event"] == "scheduled_run.completed"
    ok, outcome = await deliver("https://hooks.slack.com/services/T/B/x", "whsec_k", event, transport=t, resolve=r)
    assert ok and json.loads(seen[-1][2]) == {"text": shape("https://hooks.slack.com/x", event)["text"]}
    assert "x-crystal-signature" not in seen[-1][1]  # Slack's shape is not ours to sign
    ok, outcome = await deliver("https://bad.example/x", None, event, transport=t, resolve=r)
    assert not ok and outcome == "failed: 500"
    n = len(seen)
    ok, outcome = await deliver("https://intranet/x", None, event, transport=t, resolve=r)
    assert not ok and outcome == "failed: blocked address" and len(seen) == n  # never sent


@pytest.mark.asyncio
async def test_settle_records_result_and_delivers_once(store, customer, monkeypatch):
    from crystal_cache.control import admission

    async def _open(store_, cust):
        return None
    monkeypatch.setattr(admission, "daily_capacity_block", _open)
    calls = []

    async def fake_deliver(url, secret, event, *, transport=None):
        calls.append((url, secret, event))
        return (True, "sent") if "good" in url else (False, "failed: 404")

    t0 = datetime(2026, 10, 19, 12, 0, 30, tzinfo=UTC)
    s = await store.create_scheduled_task(customer.id, kind="research", title="News", prompt="news relevant to my knowledge base",
                                          cadence="daily", tz="UTC", at_minute=12 * 60, next_run_at=t0 - timedelta(seconds=30),
                                          deliver_url="https://relay.good.example/x", deliver_secret="whsec_k")
    out = await sched_mod.tick(store, now=t0, deliver_fn=fake_deliver)
    assert out["queued"] == 1
    row = await store.get_scheduled_task(s.id, customer.id)
    task = await store.get_cognition_task(row.last_task_id)
    ctx = task.payload["conversation_context"]
    assert "knowledge base" in ctx and "first run" in ctx and "News" in ctx
    # Nothing finished yet: the next tick settles nothing and delivers nothing.
    out = await sched_mod.tick(store, now=t0 + timedelta(minutes=1), deliver_fn=fake_deliver)
    assert out["settled"] == 0 and calls == []
    # The person pauses it while the run is in flight; the run still settles.
    await store.set_scheduled_task_status(s.id, customer.id, "paused")
    await store.mark_cognition_task_complete(
        row.last_task_id, result={"findings": "Three stories matter today.", "crystal_id": "crys_news"},
        result_crystal_id="crys_news", completed_at=t0 + timedelta(minutes=5),
    )
    t1 = t0 + timedelta(minutes=6)
    out = await sched_mod.tick(store, now=t1, deliver_fn=fake_deliver)
    assert out["settled"] == 1 and len(calls) == 1
    url, secret, event = calls[0]
    assert secret == "whsec_k" and event["summary"] == "Three stories matter today." and event["crystal_id"] == "crys_news"
    assert event["console_url"].endswith("/bank/crys_news") and event["question"] == "news relevant to my knowledge base"
    row = await store.get_scheduled_task(s.id, customer.id)
    assert row.last_outcome == "completed" and row.status == "paused"
    assert row.last_result_summary == "Three stories matter today." and row.last_crystal_id == "crys_news"
    assert row.last_delivery == "sent" and row.last_delivery_at is not None
    # At-most-once: the next tick does not deliver again.
    out = await sched_mod.tick(store, now=t1 + timedelta(minutes=1), deliver_fn=fake_deliver)
    assert len(calls) == 1
    # The result is what MCP status shows for the last 24 hours.
    recent = await store.list_recent_scheduled_results(customer.id, since=t1 - timedelta(hours=24))
    assert [r.id for r in recent] == [s.id] and recent[0].last_result_summary.startswith("Three")
    assert await store.list_recent_scheduled_results(customer.id, since=t1 + timedelta(hours=1)) == []
    # The second run knows when the first ran, and a failed delivery is recorded, not retried.
    await store.set_scheduled_task_delivery(s.id, customer.id, deliver_url="https://relay.down.example/x", deliver_secret=None)
    await store.set_scheduled_task_status(s.id, customer.id, "active", next_run_at=t1 + timedelta(hours=1))
    t2 = t1 + timedelta(hours=1, seconds=5)
    out = await sched_mod.tick(store, now=t2, deliver_fn=fake_deliver)
    assert out["queued"] == 1
    row = await store.get_scheduled_task(s.id, customer.id)
    ctx2 = (await store.get_cognition_task(row.last_task_id)).payload["conversation_context"]
    assert "first run" not in ctx2 and "previous run of this schedule: 2026-10-19T12:00:30+00:00" in ctx2
    await store.mark_cognition_task_complete(row.last_task_id, result={"findings": "Nothing new."}, result_crystal_id=None,
                                             completed_at=t2 + timedelta(minutes=2))
    await sched_mod.tick(store, now=t2 + timedelta(minutes=3), deliver_fn=fake_deliver)
    row = await store.get_scheduled_task(s.id, customer.id)
    assert len(calls) == 2 and calls[1][1] == "whsec_k"  # the secret survives a URL change
    assert row.last_delivery == "failed: 404" and row.last_result_summary == "Nothing new." and row.last_crystal_id is None
    assert calls[1][2]["console_url"].endswith("/cognition")
    # Clearing the URL drops the secret and the delivery record.
    cleared = await store.set_scheduled_task_delivery(s.id, customer.id, deliver_url=None, deliver_secret=None)
    assert cleared.deliver_url is None and cleared.deliver_secret is None and cleared.last_delivery is None


@pytest.mark.asyncio
async def test_routes_and_tools_take_a_webhook_and_show_the_secret_once(store, customer, semantic_encoder_stub, vector_store, fact_vector_store, monkeypatch):
    try:
        from tests.test_endpoint_smoke import _build_app
    except ModuleNotFoundError:
        from test_endpoint_smoke import _build_app
    from httpx import ASGITransport, AsyncClient
    from crystal_cache.scheduling import delivery as dmod
    from crystal_cache.endpoints import schedules as routes
    from crystal_cache.agent.tools import schedules as tools

    r = _resolver({"hooks.slack.com": ["3.3.3.3"], "intranet": ["10.0.0.5"]})
    monkeypatch.setattr(routes, "check_deliver_url", lambda url: dmod.check_deliver_url(url, resolve=r))
    monkeypatch.setattr(tools, "check_deliver_url", lambda url: dmod.check_deliver_url(url, resolve=r))
    app = _build_app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    base = f"/admin/api/customers/{customer.id}/schedules"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as cl:
        r1 = await cl.post(base, json={"kind": "brief", "cadence": "daily", "time": "08:00", "deliver_url": "http://hooks.slack.com/x"})
        assert r1.status_code == 400 and "https://" in r1.json()["detail"]
        r1 = await cl.post(base, json={"kind": "brief", "cadence": "daily", "time": "08:00", "deliver_url": "https://intranet/x"})
        assert r1.status_code == 400 and "public host" in r1.json()["detail"]
        r1 = await cl.post(base, json={"kind": "brief", "cadence": "daily", "time": "08:00", "deliver_url": "https://hooks.slack.com/services/T/B/x"})
        assert r1.status_code == 201, r1.text[:300]
        body = r1.json()
        assert body["deliver_secret"].startswith("whsec_") and "deliver_secret" not in body["schedule"]
        assert body["schedule"]["delivers"] is True and body["schedule"]["deliver_url"].startswith("https://hooks.slack.com")
        sid = body["schedule"]["id"]
        listed = (await cl.get(base)).json()["schedules"][0]
        assert "deliver_secret" not in listed and listed["delivers"] is True
        # Same URL again: no new secret. Cleared: delivers false. Set again: a fresh secret, once.
        r2 = await cl.patch(f"{base}/{sid}", json={"deliver_url": "https://hooks.slack.com/services/T/B/y"})
        assert r2.status_code == 200 and "deliver_secret" not in r2.json() and r2.json()["schedule"]["delivers"] is True
        r2 = await cl.patch(f"{base}/{sid}", json={"deliver_url": ""})
        assert r2.status_code == 200 and r2.json()["schedule"]["delivers"] is False
        r2 = await cl.patch(f"{base}/{sid}", json={"deliver_url": "https://hooks.slack.com/services/T/B/z", "status": "paused"})
        assert r2.status_code == 200 and r2.json()["deliver_secret"].startswith("whsec_") and r2.json()["schedule"]["status"] == "paused"
        r2 = await cl.patch(f"{base}/{sid}", json={"deliver_url": "https://intranet/x"})
        assert r2.status_code == 400
        await cl.delete(f"{base}/{sid}")
    from crystal_cache.agent.tool_registry import get_registry, import_all_tools
    from crystal_cache.agent.tools.retrievers import set_tool_state
    import_all_tools()
    set_tool_state({"store": store})
    create = get_registry().get("schedule_run")
    bad = await create.impl(customer.id, kind="brief", cadence="daily", time="08:00", deliver_url="https://intranet/x")
    assert bad["scheduled"] is False and bad["code"] == "bad_deliver_url"
    out = await create.impl(customer.id, kind="brief", cadence="daily", time="08:00", deliver_url="https://hooks.slack.com/services/T/B/x")
    assert out["scheduled"] is True and out["deliver_secret"].startswith("whsec_") and out["schedule"]["delivers"] is True
    assert "deliver_secret" not in out["schedule"]
