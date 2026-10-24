"""Goals (Q109 A, 2026-10-24).

Anthony: "'goals' its just a long standing directive that the model
addresses periodically. And consistently works at the next step of each
goal's plans. We will use a very strong model for digesting a goal and
making a plan." And (Q107): "we can enable send and post too."

The frontier tier writes the plan (normalized: unknown apps dropped,
cadence floored); a large-tier check-in works it with a fresh toolset:
reads what the apps showed, files action items, writes drafts, and sends
only when the goal may and the app's Act is on. A person's approval is
the permission for anything else. Every send is ledgered with its text.
"""
from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi import HTTPException

from crystal_cache import beta
from crystal_cache.config import get_settings
from crystal_cache.connectors import nango
from crystal_cache.connectors import service as conn_service
from crystal_cache.goals import planner, runner, service
from crystal_cache.goals.session import run_goal_session
from fakes import FakeAnthropic


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    beta.reset_for_tests()
    s = get_settings()
    monkeypatch.setattr(s, "nango_secret_key", "nango-secret-test")
    monkeypatch.setattr(s, "nango_base_url", "https://nango.test")
    yield
    nango.set_transport(None)


class _Nango:
    def __init__(self, routes):
        self.calls = []
        self.routes = routes

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.calls.append({"method": request.method, "path": request.url.path, "json": body,
                           "headers": dict(request.headers)})
        for (method, prefix), (status, payload) in self.routes.items():
            if request.method == method and request.url.path.startswith(prefix):
                return httpx.Response(status, json=payload)
        return httpx.Response(404, json={"error": "no route"})

    def install(self):
        nango.set_transport(httpx.MockTransport(self))
        return self


def _plan_json(**over) -> str:
    plan = {
        "summary": "Every email gets a reply within 3 hours.",
        "success_measure": "No email older than 3 hours without a reply or a draft.",
        "check_every_minutes": 30,
        "steps": [{"title": "Each check-in: triage new mail", "detail": "Find mail that needs a reply."},
                  {"title": "Draft replies", "detail": "Draft a reply for each one."}],
        "needs_apps": ["google-mail", "dropbox"],
        "needs_acts": ["send_email", "teleport"],
        "standing_instructions": "Short, warm replies. Never commit to dates.",
    }
    plan.update(over)
    return json.dumps(plan)


async def _gmail(store, customer, *, act: bool = True):
    await store.create_beta_invite("gmail", customer_id=customer.id)
    conn = await store.upsert_app_connection(customer.id, integration="google-mail", nango_connection_id="nc_mail",
                                             account_label="dana@kerning.test")
    if act:
        await conn_service.set_uses(store, customer.id, conn["id"], uses={"act": True})
    return await store.get_app_connection(customer.id, conn["id"])


async def _planned_goal(store, customer, **kw):
    g = await service.create(store, customer.id, title="Inbox zero",
                             directive="Reply to every email within 3 hours; draft replies for me.", **kw)
    plan = planner.normalize_plan(json.loads(_plan_json()), min_minutes=15)
    return await store.update_goal(customer.id, g["id"], plan=plan, planned_at=datetime.now(timezone.utc),
                                   next_run_at=None)


# --- planning -------------------------------------------------------------------

def test_a_plan_is_normalized_never_trusted():
    raw = json.loads(_plan_json(check_every_minutes=2, steps=[{"title": f"Step {i}", "detail": "d"} for i in range(20)]))
    p = planner.normalize_plan(raw, min_minutes=15, previous={"steps": [{"title": "Step 3", "done": True}]})
    assert p["check_every_minutes"] == 15
    assert len(p["steps"]) == planner.MAX_STEPS and [s["n"] for s in p["steps"]][:3] == [1, 2, 3]
    assert [s["done"] for s in p["steps"]][:5] == [False, False, False, True, False]  # kept from the old plan
    assert p["needs"] == {"apps": ["google-mail"], "acts": ["send_email"]}
    assert planner.normalize_plan(json.loads(_plan_json(check_every_minutes=10**9)), min_minutes=15)["check_every_minutes"] \
        == planner.MAX_CHECK_MINUTES
    with pytest.raises(ValueError):
        planner.normalize_plan({"steps": []}, min_minutes=15)


@pytest.mark.asyncio
async def test_a_new_goal_is_planned_by_the_frontier_tier_on_the_next_pass(store, customer):
    await _gmail(store, customer)
    g = await service.create(store, customer.id, title="Inbox zero",
                             directive="Reply to every email within 3 hours; draft replies for me.")
    assert g["status"] == "active" and g["planned_at"] is None
    anth = FakeAnthropic()
    anth.script_text(_plan_json())
    out = await runner.tick(store, llm=anth)
    assert out == {"planned": 1, "ran": 0, "walled": 0}
    call = anth.calls[0]
    assert call["model"] == "frontier" and call["json_schema"] == planner.PLAN_SCHEMA
    assert "Reply to every email within 3 hours" in call["messages"][0]["content"]
    assert "google-mail: watch on, act on, status active" in call["messages"][0]["content"]
    after = await store.get_goal(customer.id, g["id"])
    assert after["plan"]["summary"].startswith("Every email") and after["plan"]["check_every_minutes"] == 30
    assert after["planned_at"] and after["next_run_at"]  # due now: the first check-in comes next pass
    tasks = [t for t in await store.list_cognition_tasks(customer.id) if t.task_type == "goal"]
    assert len(tasks) == 1 and tasks[0].status == "complete" and tasks[0].payload["phase"] == "plan"
    events = [e["event_type"] for e in await store.list_curation_events(customer.id)]
    assert "goal_planned" in events


@pytest.mark.asyncio
async def test_a_failed_plan_leaves_the_goal_unplanned_and_retries_within_the_hour(store, customer):
    g = await service.create(store, customer.id, title="Weekly review", directive="Every Friday, summarize my week.")
    anth = FakeAnthropic()
    anth.script_text("not json")
    out = await runner.tick(store, llm=anth)
    assert out["planned"] == 0
    after = await store.get_goal(customer.id, g["id"])
    assert after["planned_at"] is None and "trying again" in after["last_summary"]
    assert datetime.fromisoformat(after["next_run_at"]) > datetime.now(timezone.utc) + timedelta(minutes=50)
    assert await store.list_due_goals() == []


# --- a check-in -------------------------------------------------------------------

async def _mail(store, customer, conn):
    a = await store.upsert_watched_item(customer.id, conn["id"], kind="email", external_id="m1",
                                        title="Contract question", author="theo@acme.test",
                                        occurred_at=datetime.now(timezone.utc) - timedelta(minutes=40),
                                        snippet="Can you confirm the renewal?",
                                        body="Hi Dana, can you confirm the renewal terms by Friday? Theo",
                                        meta={"threadId": "t1", "messageId": "<m1@acme.test>"})
    b = await store.upsert_watched_item(customer.id, conn["id"], kind="email", external_id="m2",
                                        title="Newsletter", author="news@x.test",
                                        occurred_at=datetime.now(timezone.utc) - timedelta(minutes=20),
                                        snippet="This week in widgets")
    return a, b


def _script_check_in(anth, a):
    anth.script_tool_use("goal_plan", {}, "tu_01")
    anth.script_tool_use("goal_items", {}, "tu_02")
    anth.script_tool_use("watched_items", {}, "tu_03")
    anth.script_tool_use("read_item", {"item_id": a["id"]}, "tu_04")
    anth.script_tool_use("write_draft", {"app": "google-mail", "title": "Re: Contract question",
                                         "body": "Hi Theo, yes: the renewal terms stand as written. Dana",
                                         "watched_item_id": a["id"], "to": "theo@acme.test",
                                         "subject": "Contract question", "thread_id": "t1",
                                         "in_reply_to": "<m1@acme.test>"}, "tu_05")
    anth.script_tool_use("add_action_item", {"title": "Answer Theo", "watched_item_id": a["id"]}, "tu_06")
    anth.script_tool_use("report_stuck", {"need": "Let this goal send simple confirmations on its own."}, "tu_07")
    anth.script_tool_use("mark_step", {"n": 2, "done": True, "note": "Drafted the one reply needed."}, "tu_08")
    anth.script_text("Did: drafted a reply to Theo.\nWaiting: nothing.\nNext: triage new mail.")


def _result(anth, i):
    msg = anth.calls[i + 1]["messages"][-1]["content"][0]
    return json.loads(msg["content"]) if isinstance(msg["content"], str) else msg["content"]


@pytest.mark.asyncio
async def test_a_check_in_files_and_drafts_but_never_sends_without_permission(store, customer):
    conn = await _gmail(store, customer)
    a, b = await _mail(store, customer, conn)
    goal = await _planned_goal(store, customer)
    anth = FakeAnthropic()
    _script_check_in(anth, a)
    out = await run_goal_session(store=store, customer=customer, goal=goal, llm=anth)
    assert out["ok"] and out["actions"] == 1 and out["calls"] == 8
    names = sorted(t["name"] for t in anth.calls[0]["tools"])
    assert names == sorted(["goal_plan", "watched_items", "read_item", "crystal_search", "crystal_key_scan", "goal_items",
                            "add_action_item", "write_draft", "send_draft", "close_item", "mark_step", "report_stuck"])
    assert "Never commit to dates." in anth.calls[0]["system"][0]["text"]
    plan_view = _result(anth, 0)
    assert plan_view["may_send_email"] is False and plan_view["connected_apps"][0]["act"] is True
    seen = _result(anth, 2)["items"]
    assert [i["title"] for i in seen] == ["Newsletter", "Contract question"] and seen[1]["thread"] == "t1"
    assert "renewal terms by Friday" in _result(anth, 3)["body"]
    draft = _result(anth, 4)
    assert draft["ok"] and draft["may_send_on_its_own"] is False and "waits for the person" in draft["note"]
    assert _result(anth, 5) == {"ok": False, "code": "already_filed",
                                "error": "This goal already filed something for that item."}
    assert _result(anth, 6)["ok"]
    items = await store.list_goal_items(customer.id, goal["id"], limit=10)
    assert sorted((i["kind"], i["status"]) for i in items) == [("draft", "open"), ("note", "open")]
    d = next(i for i in items if i["kind"] == "draft")
    assert d["target"]["action"] == "send_email" and d["target"]["connection_id"] == conn["id"]
    after = await store.get_goal(customer.id, goal["id"])
    assert after["plan"]["steps"][1]["done"] is True and after["last_summary"].startswith("Did: drafted")
    nxt = datetime.fromisoformat(after["next_run_at"]) - datetime.fromisoformat(after["last_run_at"])
    assert timedelta(minutes=29) < nxt < timedelta(minutes=31)
    # A second pass shows the same mail as already filed.
    since = await store.goal_items_for_watched(customer.id, goal["id"], [a["id"], b["id"]])
    assert since == {a["id"]}


@pytest.mark.asyncio
async def test_a_permitted_goal_sends_through_the_app_and_the_ledger_keeps_the_text(store, customer):
    conn = await _gmail(store, customer)
    a, _ = await _mail(store, customer, conn)
    goal = await _planned_goal(store, customer, may_send=True)
    fake = _Nango({("POST", "/proxy/gmail/v1/users/me/messages/send"): (200, {"id": "gm_1", "threadId": "t1"})}).install()
    draft = await store.add_goal_item(customer.id, goal["id"], kind="draft", title="Re: Contract question",
                                      body="Hi Theo, confirmed. Dana", watched_item_id=a["id"],
                                      target={"app": "google-mail", "connection_id": conn["id"], "action": "send_email",
                                              "to": "theo@acme.test", "subject": "Contract question", "thread_id": "t1",
                                              "in_reply_to": "<m1@acme.test>"})
    anth = FakeAnthropic()
    anth.script_tool_use("send_draft", {"item_id": draft["id"], "reason": "A plain confirmation the instructions allow."}, "tu_01")
    anth.script_text("Did: sent one reply.\nWaiting: nothing.\nNext: triage.")
    out = await run_goal_session(store=store, customer=customer, goal=goal, llm=anth)
    assert out["ok"] and out["actions"] == 1
    assert _result(anth, 0)["ok"] and _result(anth, 0)["ref"] == "gm_1"
    call = fake.calls[0]
    assert call["headers"]["connection-id"] == "nc_mail" and call["headers"]["provider-config-key"] == "google-mail"
    assert call["json"]["threadId"] == "t1"
    raw = base64.urlsafe_b64decode(call["json"]["raw"] + "==").decode()
    assert "To: theo@acme.test" in raw and "Subject: Re: Contract question" in raw
    assert "In-Reply-To: <m1@acme.test>" in raw and "Hi Theo, confirmed. Dana" in raw
    sent = await store.get_goal_item(customer.id, draft["id"])
    assert sent["status"] == "sent" and sent["target"]["ref"] == "gm_1" and sent["target"]["sent_by"] == f"goal:{goal['id']}"
    rows = await store.list_steward_actions(customer.id, limit=10)
    row = next(r for r in rows if r["action"] == "send_email")
    assert row["actor"] == f"goal:{goal['id']}" and row["undoable"] is False
    assert row["subject_id"] == draft["id"] and "theo@acme.test" in row["label"]
    assert "goal_action" in [e["event_type"] for e in await store.list_curation_events(customer.id)]


@pytest.mark.asyncio
async def test_the_person_approves_a_draft_and_act_off_still_blocks(store, customer):
    conn = await _gmail(store, customer, act=False)
    goal = await _planned_goal(store, customer)
    target = {"app": "google-mail", "connection_id": conn["id"], "action": "send_email", "to": "theo@acme.test",
              "subject": "Hello"}
    d = await store.add_goal_item(customer.id, goal["id"], kind="draft", title="Hello", body="Hi.", target=target)
    with pytest.raises(service.GoalRefused) as e:
        await service.item_action(store, customer.id, d["id"], "send", actor="console")
    assert e.value.code == "act_off"
    await conn_service.set_uses(store, customer.id, conn["id"], uses={"act": True})
    fake = _Nango({("POST", "/proxy/gmail/"): (200, {"id": "gm_2"})}).install()
    out = await service.item_action(store, customer.id, d["id"], "send", actor="console", body="Hi Theo, edited by me.")
    assert out["ok"] and out["item"]["status"] == "sent" and out["item"]["body"] == "Hi Theo, edited by me."
    assert "edited by me" in base64.urlsafe_b64decode(fake.calls[0]["json"]["raw"] + "==").decode()
    row = (await store.list_steward_actions(customer.id, limit=5))[0]
    assert row["actor"] == "console" and row["label"].startswith("You send email")
    with pytest.raises(service.GoalRefused) as e:
        await service.item_action(store, customer.id, d["id"], "send", actor="console")
    assert e.value.code == "already_handled"


@pytest.mark.asyncio
async def test_a_refused_post_marks_the_draft_failed(store, customer):
    conn = await store.upsert_app_connection(customer.id, integration="slack", nango_connection_id="nc_s")
    await conn_service.set_uses(store, customer.id, conn["id"], uses={"act": True})
    goal = await _planned_goal(store, customer, may_post=True)
    fake = _Nango({("POST", "/proxy/chat.postMessage"): (200, {"ok": False, "error": "channel_not_found"})}).install()
    d = await store.add_goal_item(customer.id, goal["id"], kind="draft", title="Standup", body="Shipped the fix.",
                                  target={"app": "slack", "connection_id": conn["id"], "action": "post_message",
                                          "channel": "C123"})
    with pytest.raises(service.GoalRefused) as e:
        await service.item_action(store, customer.id, d["id"], "send", actor="chat")
    assert e.value.code == "send_failed" and e.value.status == 502
    assert fake.calls[0]["json"] == {"channel": "C123", "text": "Shipped the fix."}
    assert (await store.get_goal_item(customer.id, d["id"]))["status"] == "failed"


# --- the console's and MCP's door ---------------------------------------------------------

@pytest.mark.asyncio
async def test_goal_changes_have_limits_and_a_new_directive_means_a_new_plan(store, customer, monkeypatch):
    with pytest.raises(service.GoalRefused) as e:
        await service.create(store, customer.id, title="x", directive="Reply within 3 hours.")
    assert e.value.code == "title_required"
    with pytest.raises(service.GoalRefused) as e:
        await service.create(store, customer.id, title="Inbox", directive="short")
    assert e.value.code == "directive_required"
    with pytest.raises(service.GoalRefused) as e:
        await service.create(store, customer.id, title="Inbox", directive="Reply within 3 hours.", may_send="yes")
    assert e.value.code == "invalid_request"
    goal = await _planned_goal(store, customer)
    same = await service.update(store, customer.id, goal["id"], directive=goal["directive"])
    assert same["planned_at"] is not None
    changed = await service.update(store, customer.id, goal["id"], directive="Reply within 1 hour, and flag VIPs.")
    assert changed["planned_at"] is None and changed["next_run_at"] is None
    with pytest.raises(service.GoalRefused) as e:
        await service.update(store, customer.id, goal["id"], status="exploded")
    assert e.value.code == "invalid_status"
    paused = await service.update(store, customer.id, goal["id"], status="paused")
    with pytest.raises(service.GoalRefused) as e:
        await service.run_now(store, customer.id, paused["id"])
    assert e.value.code == "goal_not_active"
    monkeypatch.setattr(get_settings(), "goals_max_per_tenant", 1)
    with pytest.raises(service.GoalRefused) as e:
        await service.create(store, customer.id, title="Second", directive="Another standing goal here.")
    assert e.value.code == "goal_limit" and e.value.status == 409
    await store.set_beta_feature_status("goals", "invite")
    with pytest.raises(service.GoalRefused) as e:
        await service.update(store, customer.id, goal["id"], status="active")
    assert e.value.code == "beta_required" and e.value.status == 403
    item = await store.add_goal_item(customer.id, goal["id"], kind="action", title="Call Theo")
    assert (await service.item_action(store, customer.id, item["id"], "done", actor="console"))["item"]["status"] == "done"
    assert (await service.item_action(store, customer.id, item["id"], "reopen", actor="console"))["item"]["status"] == "open"
    with pytest.raises(service.GoalRefused) as e:
        await service.item_action(store, customer.id, item["id"], "send", actor="console")
    assert e.value.code == "not_a_draft"


@pytest.mark.asyncio
async def test_the_runner_waits_at_the_wall_and_never_runs_a_goal_twice(store, customer, monkeypatch):
    goal = await _planned_goal(store, customer)

    async def walled(*a, **k):
        raise HTTPException(status_code=402, detail="cap")
    monkeypatch.setattr("crystal_cache.ingress.auth.require_write_capacity", walled)
    out = await runner.tick(store, llm=FakeAnthropic())
    assert out == {"planned": 0, "ran": 0, "walled": 1}
    after = await store.get_goal(customer.id, goal["id"])
    assert after["last_summary"].startswith("Paused: today's capacity")
    assert datetime.fromisoformat(after["next_run_at"]) > datetime.now(timezone.utc) + timedelta(minutes=50)
    # The claim is a compare-and-set: the second taker loses.
    await store.update_goal(customer.id, goal["id"], next_run_at=None)
    now = datetime.now(timezone.utc)
    assert await store.claim_goal(goal["id"], now=now, lease_until=now + timedelta(minutes=10)) is True
    assert await store.claim_goal(goal["id"], now=now, lease_until=now + timedelta(minutes=10)) is False
    monkeypatch.setattr(get_settings(), "enable_goals", False)
    assert await runner.tick(store, llm=FakeAnthropic()) == {"planned": 0, "ran": 0, "walled": 0}


@pytest.mark.asyncio
async def test_erase_keeps_goals_and_drops_what_they_produced(store, customer):
    goal = await _planned_goal(store, customer)
    await store.add_goal_item(customer.id, goal["id"], kind="action", title="Call Theo")
    await store.erase_tenant_bank(customer.id)
    assert [g["id"] for g in await store.list_goals(customer.id)] == [goal["id"]]
    assert await store.list_goal_items(customer.id) == []
    assert await store.delete_goal(customer.id, goal["id"]) is True
    assert await store.list_goals(customer.id) == []
