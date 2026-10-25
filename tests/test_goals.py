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
from crystal_cache.connectors import catalog, nango
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
    import time
    conn_service.reset_for_tests()
    conn_service._configured.update(keys={a.integration for a in catalog.APPS}, at=time.monotonic())
    yield
    conn_service.reset_for_tests()
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


ALL_CONNECTED = {a.integration: "connected" for a in catalog.APPS}


def _plan_json(**over) -> str:
    plan = {
        "summary": "Every email gets a reply within 3 hours.",
        "success_measure": "No email older than 3 hours without a reply or a draft.",
        "check_every_minutes": 30,
        "steps": [{"title": "Each check-in: triage new mail", "detail": "Find mail that needs a reply."},
                  {"title": "Draft replies", "detail": "Draft a reply for each one."}],
        "needs_apps": ["google-mail", "myspace"],
        "needs_acts": ["send_email", "teleport"],
        "standing_instructions": "Short, warm replies. Never commit to dates.",
        "missing": [],
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
    plan = planner.normalize_plan(json.loads(_plan_json()), min_minutes=15, statuses=ALL_CONNECTED)
    return await store.update_goal(customer.id, g["id"], plan=plan, planned_at=datetime.now(timezone.utc),
                                   next_run_at=None)


# --- planning -------------------------------------------------------------------

def test_a_plan_is_normalized_never_trusted():
    raw = json.loads(_plan_json(check_every_minutes=2, steps=[{"title": f"Step {i}", "detail": "d"} for i in range(20)]))
    p = planner.normalize_plan(raw, min_minutes=15, previous={"steps": [{"title": "Step 3", "done": True}]},
                               statuses=ALL_CONNECTED)
    assert p["check_every_minutes"] == 15
    assert len(p["steps"]) == planner.MAX_STEPS and [s["n"] for s in p["steps"]][:3] == [1, 2, 3]
    assert [s["done"] for s in p["steps"]][:5] == [False, False, False, True, False]  # kept from the old plan
    assert p["needs"] == {"apps": ["google-mail"], "acts": ["send_email"]}
    assert planner.normalize_plan(json.loads(_plan_json(check_every_minutes=10**9)), min_minutes=15,
                                  statuses=ALL_CONNECTED)["check_every_minutes"] \
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
    assert out == {"planned": 1, "ran": 0, "walled": 0, "unheld": 0}
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
                            "web_search", "web_fetch", "save_fact",
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
    assert out == {"planned": 0, "ran": 0, "walled": 1, "unheld": 0}
    after = await store.get_goal(customer.id, goal["id"])
    assert after["last_summary"].startswith("Paused: today's capacity")
    assert datetime.fromisoformat(after["next_run_at"]) > datetime.now(timezone.utc) + timedelta(minutes=50)
    # The claim is a compare-and-set: the second taker loses.
    await store.update_goal(customer.id, goal["id"], next_run_at=None)
    now = datetime.now(timezone.utc)
    assert await store.claim_goal(goal["id"], now=now, lease_until=now + timedelta(minutes=10)) is True
    assert await store.claim_goal(goal["id"], now=now, lease_until=now + timedelta(minutes=10)) is False
    monkeypatch.setattr(get_settings(), "enable_goals", False)
    assert await runner.tick(store, llm=FakeAnthropic()) == {"planned": 0, "ran": 0, "walled": 0, "unheld": 0}


@pytest.mark.asyncio
async def test_erase_keeps_goals_and_drops_what_they_produced(store, customer):
    goal = await _planned_goal(store, customer)
    await store.add_goal_item(customer.id, goal["id"], kind="action", title="Call Theo")
    await store.erase_tenant_bank(customer.id)
    assert [g["id"] for g in await store.list_goals(customer.id)] == [goal["id"]]
    assert await store.list_goal_items(customer.id) == []
    assert await store.delete_goal(customer.id, goal["id"]) is True
    assert await store.list_goals(customer.id) == []


# --- Q112 A: the plan says what's missing, and a blocked goal waits --------------------------

@pytest.mark.asyncio
async def test_a_goal_that_needs_an_unconnected_app_is_held_and_released_when_it_connects(store, customer):
    g = await service.create(store, customer.id, title="Standups",
                             directive="Post a standup summary to #team in Slack every weekday at 9.")
    anth = FakeAnthropic()
    anth.script_text(_plan_json(needs_apps=["slack"], needs_acts=["post_message"], missing=[
        {"kind": "data_source", "need": "Jira sprint data", "why": "Step 1 summarizes tickets.",
         "suggestion": "Use web search on the public board for now.", "blocking": False},
        {"kind": "app", "need": "Connect Slack", "why": "Every step posts.", "suggestion": "Add Slack.", "blocking": True},
        {"kind": "app", "need": "Slack (again)", "why": "dup", "suggestion": "", "blocking": False},
    ]))
    out = await runner.tick(store, llm=anth)
    assert out["planned"] == 1
    held = await store.get_goal(customer.id, g["id"])
    assert held["status"] == "needs_setup" and held["next_run_at"] is None
    missing = held["plan"]["missing"]
    assert [(m["kind"], m.get("integration"), m["blocking"]) for m in missing] == [
        ("data_source", None, False), ("app", "slack", True)]
    assert held["plan"]["blocked"] is True
    assert "goal_needs_setup" in [e["event_type"] for e in await store.list_curation_events(customer.id)]
    assert "cannot" in anth.calls[0]["system"].lower() or "CANNOT" in anth.calls[0]["system"]
    # Held: no check-in, no spend.
    assert await store.list_due_goals() == []
    assert await runner.tick(store, llm=FakeAnthropic()) == {"planned": 0, "ran": 0, "walled": 0, "unheld": 0}
    # Slack connects: the goal goes back to the planner by itself.
    await store.upsert_app_connection(customer.id, integration="slack", nango_connection_id="nc_s")
    anth2 = FakeAnthropic()
    anth2.script_text(_plan_json(needs_apps=["slack"], missing=[]))
    out = await runner.tick(store, llm=anth2)
    assert out["unheld"] == 1 and out["planned"] == 1
    again = await store.get_goal(customer.id, g["id"])
    assert again["status"] == "active" and again["plan"]["blocked"] is False and again["next_run_at"]


@pytest.mark.asyncio
async def test_run_anyway_releases_a_held_goal_and_survives_a_replan(store, customer):
    g = await service.create(store, customer.id, title="Sports catalog",
                             directive="Catalog every NBA and NFL game with full player stats.")
    anth = FakeAnthropic()
    anth.script_text(_plan_json(needs_apps=[], missing=[
        {"kind": "data_source", "need": "A sports statistics API", "why": "Box scores for every game.",
         "suggestion": "Pick a provider.", "blocking": True}]))
    await runner.tick(store, llm=anth)
    assert (await store.get_goal(customer.id, g["id"]))["status"] == "needs_setup"
    with pytest.raises(service.GoalRefused) as e:
        await service.run_now(store, customer.id, g["id"])
    assert e.value.code == "goal_not_active"
    ran = await service.run_anyway(store, customer.id, g["id"])
    assert ran["status"] == "active" and ran["plan"]["run_anyway"] is True
    with pytest.raises(service.GoalRefused) as e:
        await service.run_anyway(store, customer.id, g["id"])
    assert e.value.code == "not_held"
    # A replan finds the same gap but keeps running: the person chose.
    await service.replan(store, customer.id, g["id"])
    anth2 = FakeAnthropic()
    anth2.script_text(_plan_json(needs_apps=[], missing=[
        {"kind": "data_source", "need": "A sports statistics API", "why": "x", "suggestion": "y", "blocking": True}]))
    await runner.tick(store, llm=anth2)
    after = await store.get_goal(customer.id, g["id"])
    assert after["status"] == "active" and after["plan"]["run_anyway"] is True and after["next_run_at"]


# --- Q113 A: a check-in can search the web and save facts, undoably -------------------------

@pytest.mark.asyncio
async def test_a_check_in_saves_a_fact_on_the_ledger_and_undo_forgets_it(store, customer, semantic_encoder_stub,
                                                                          vector_store, fact_vector_store):
    encoder = semantic_encoder_stub
    from crystal_cache.cognition.controls import undo_action
    goal = await _planned_goal(store, customer)
    anth = FakeAnthropic()
    anth.script_tool_use("save_fact", {"key": "Celtics|Jayson Tatum|3PT% 2026-27",
                                       "value": "As of 2026-10-24 per nba.com, Jayson Tatum shoots 41% from three.",
                                       "reason": "The goal tracks player shooting."}, "tu_01")
    anth.script_tool_use("save_fact", {"key": "k", "value": "x" * 900, "reason": "Too long on purpose here."}, "tu_02")
    anth.script_tool_use("save_fact", {"key": "k", "value": "v", "reason": "short"}, "tu_03")
    anth.script_text("Did: saved one fact.\nWaiting: nothing.\nNext: more stats.")
    out = await run_goal_session(store=store, customer=customer, goal=goal, encoder=encoder, fact_store=fact_vector_store,
                                 vector_store=vector_store, llm=anth)
    assert out["ok"] and out["actions"] == 1
    saved = _result(anth, 0)
    assert saved["ok"] and saved["fact_id"]
    assert _result(anth, 1)["code"] == "content_too_long"
    assert _result(anth, 2)["code"] == "reason_required"
    fact = await store.get_fact(saved["fact_id"])
    assert "41% from three" in fact.claim_text
    row = next(r for r in await store.list_steward_actions(customer.id, limit=10) if r["action"] == "save_fact")
    assert row["actor"] == f"goal:{goal['id']}" and row["undoable"] is True
    state = {"encoder": encoder, "vector_store": vector_store, "fact_vector_store": fact_vector_store}
    undone = await undo_action(store, customer.id, row["id"], state=state, actor="console")
    assert undone["undone"] is True
    assert await store.get_fact(saved["fact_id"]) is None


@pytest.mark.asyncio
async def test_save_fact_stops_at_the_memory_wall(store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
                                                 monkeypatch):
    encoder = semantic_encoder_stub
    goal = await _planned_goal(store, customer)

    async def full(*a, **k):
        raise HTTPException(status_code=402, detail="memory_full")
    monkeypatch.setattr("crystal_cache.ingress.auth.require_write_capacity", full)
    anth = FakeAnthropic()
    anth.script_tool_use("save_fact", {"key": "k", "value": "v", "reason": "A fact the goal needs kept."}, "tu_01")
    anth.script_text("Did: nothing.\nWaiting: memory is full.\nNext: -")
    await run_goal_session(store=store, customer=customer, goal=goal, encoder=encoder, fact_store=fact_vector_store,
                           vector_store=vector_store, llm=anth)
    assert _result(anth, 0)["code"] == "memory_full"


# --- planning step 2: decisions before a plan ----------------------------------------------

@pytest.mark.asyncio
async def test_a_goal_that_turns_on_choices_asks_first_then_plans_on_the_answers(store, customer):
    g = await service.create(store, customer.id, title="Sports catalog",
                             directive="Catalog every NBA and NFL game; let me browse by player or team.")
    anth = FakeAnthropic()
    anth.script_text(_plan_json(steps=[], needs_apps=["notion"], missing=[
        {"kind": "app", "need": "Connect Notion", "why": "x", "suggestion": "y", "blocking": True},
        {"kind": "data_source", "need": "A sports stats API", "why": "x", "suggestion": "y", "blocking": True}],
        decisions=[
            {"question": "Where should the catalog live?", "why": "Every step writes there.", "recommended": 0,
             "options": [{"label": "Crystal memory", "detail": "Facts in your bank"},
                         {"label": "Notion", "detail": "Connect Notion; databases there"}]},
            {"question": "How far back?", "why": "Backfill size.", "recommended": 1,
             "options": [{"label": "From today", "detail": ""}, {"label": "This season", "detail": ""}]},
            {"question": "Bad", "why": "one option", "recommended": 0, "options": [{"label": "Only", "detail": ""}]},
        ]))
    await runner.tick(store, llm=anth)
    asking = await store.get_goal(customer.id, g["id"])
    assert asking["status"] == "deciding" and asking["next_run_at"] is None
    plan = asking["plan"]
    assert [d["question"] for d in plan["decisions"]] == ["Where should the catalog live?", "How far back?"]
    assert plan["steps"] == [] and plan["deciding"] is True
    # While deciding, nothing blocks and no app is demanded: the answer picks the app.
    assert [m["kind"] for m in plan["missing"]] == ["data_source"] and plan["blocked"] is False
    assert "goal_deciding" in [e["event_type"] for e in await store.list_curation_events(customer.id)]
    assert await store.list_due_goals() == []
    with pytest.raises(service.GoalRefused) as e:
        await service.decide(store, customer.id, g["id"], [{"index": 0, "choice": 0}])
    assert e.value.code == "answers_missing"
    with pytest.raises(service.GoalRefused) as e:
        await service.decide(store, customer.id, g["id"], [{"index": 0, "choice": 7}, {"index": 1, "choice": 0}])
    assert e.value.code == "bad_answer"
    decided = await service.decide(store, customer.id, g["id"],
                                   [{"index": 0, "choice": 0}, {"index": 1, "other": "Only the playoffs"}])
    assert decided["status"] == "active" and decided["planned_at"] is None
    assert [d["answer"] for d in decided["plan"]["decisions"]] == ["Crystal memory (Facts in your bank)", "Only the playoffs"]
    # The planner gets the answers as final and writes the plan; a repeated question is dropped.
    anth2 = FakeAnthropic()
    anth2.script_text(_plan_json(needs_apps=[], decisions=[
        {"question": "Where should the catalog live?", "why": "again", "recommended": 0,
         "options": [{"label": "a", "detail": ""}, {"label": "b", "detail": ""}]}]))
    await runner.tick(store, llm=anth2)
    assert "Where should the catalog live? -> Crystal memory (Facts in your bank)" in anth2.calls[0]["messages"][0]["content"]
    assert "Only the playoffs" in anth2.calls[0]["messages"][0]["content"]
    planned = await store.get_goal(customer.id, g["id"])
    assert planned["status"] == "active" and len(planned["plan"]["steps"]) == 2 and planned["plan"]["deciding"] is False
    assert len(planned["plan"]["decisions"]) == 2 and planned["next_run_at"]
    with pytest.raises(service.GoalRefused) as e:
        await service.decide(store, customer.id, g["id"], [])
    assert e.value.code == "not_deciding"


# --- the planner never builds on an app nobody chose (2026-10-24) ---------------------------

@pytest.mark.asyncio
async def test_a_plan_built_on_an_unchosen_app_becomes_a_question_with_sheets_as_an_option(store, customer):
    g = await service.create(store, customer.id, title="Sports catalog",
                             directive="Catalog every NBA and NFL game; let me browse by player or team.")
    # The model ignores the instructions and plans on Notion with no decisions.
    anth = FakeAnthropic()
    anth.script_text(_plan_json(needs_apps=["notion"], decisions=[], missing=[
        {"kind": "app", "need": "Notion integration connected with write access", "why": "x", "suggestion": "y", "blocking": True}]))
    await runner.tick(store, llm=anth)
    system = anth.calls[0]["system"]
    assert "- google-sheet: Google Sheets (records; acts: create_spreadsheet, append_rows;" in system
    asking = await store.get_goal(customer.id, g["id"])
    assert asking["status"] == "deciding" and asking["plan"]["steps"] == []
    d = asking["plan"]["decisions"][0]
    assert d["question"] == "This goal would use Notion, which isn't connected. What should it use?"
    labels = [o["label"] for o in d["options"]]
    assert labels[:3] == ["Crystal memory", "Notion", "Google Sheets"] and len(labels) == 4
    assert d["recommended"] == 0 and "can write: create spreadsheets, add rows" in d["options"][2]["detail"]
    # The person picks Google Sheets; the planner may still ask for Notion: it is never asked twice.
    await service.decide(store, customer.id, g["id"], [{"index": 0, "choice": 2}])
    anth2 = FakeAnthropic()
    anth2.script_text(_plan_json(needs_apps=["notion", "google-sheet"], needs_acts=["append_rows"], decisions=[]))
    await runner.tick(store, llm=anth2)
    assert "Never build a plan around an app" in anth2.calls[0]["system"]
    assert "The current plan" not in anth2.calls[0]["messages"][0]["content"]  # nothing done: no anchoring
    planned = await store.get_goal(customer.id, g["id"])
    assert planned["plan"]["needs"]["apps"] == ["google-sheet"]
    assert len(planned["plan"]["decisions"]) == 1
    # Sheets isn't connected yet: held for setup, not asked again.
    assert planned["status"] == "needs_setup"
    assert [m.get("integration") for m in planned["plan"]["missing"] if m["kind"] == "app"] == ["google-sheet"]


@pytest.mark.asyncio
async def test_a_goal_creates_a_spreadsheet_and_appends_rows_through_sheets(store, customer):
    conn = await store.upsert_app_connection(customer.id, integration="google-sheet", nango_connection_id="nc_gs")
    await conn_service.set_uses(store, customer.id, conn["id"], uses={"act": True})
    goal = await _planned_goal(store, customer, may_post=True)
    fake = _Nango({("POST", "/proxy/v4/spreadsheets/sheet_1/values/"): (200, {"updates": {"updatedRange": "'Games'!A2:C3"}}),
                   ("POST", "/proxy/v4/spreadsheets"): (200, {"spreadsheetId": "sheet_1"})}).install()
    made = await store.add_goal_item(customer.id, goal["id"], kind="draft", title="Sports catalog", body="Games\nPlayers",
                                     target={"app": "google-sheet", "connection_id": conn["id"], "action": "create_spreadsheet"})
    out = await service.item_action(store, customer.id, made["id"], "send", actor="console")
    assert out["ref"] == "sheet_1"
    assert fake.calls[0]["json"] == {"properties": {"title": "Sports catalog"},
                                     "sheets": [{"properties": {"title": "Games"}}, {"properties": {"title": "Players"}}]}
    rows = await store.add_goal_item(customer.id, goal["id"], kind="draft", title="2 games",
                                     body='[["2026-10-22", "BOS", 112], ["2026-10-22", "NYK", 104]]',
                                     target={"app": "google-sheet", "connection_id": conn["id"], "action": "append_rows",
                                             "spreadsheet_id": "sheet_1", "sheet": "Games"})
    out = await service.item_action(store, customer.id, rows["id"], "send", actor="console")
    call = fake.calls[1]
    assert call["path"] == "/proxy/v4/spreadsheets/sheet_1/values/'Games'!A1:append"
    assert call["json"] == {"values": [["2026-10-22", "BOS", 112], ["2026-10-22", "NYK", 104]]}
    assert out["ref"] == "'Games'!A2:C3"
    bad = await store.add_goal_item(customer.id, goal["id"], kind="draft", title="bad", body="not rows",
                                    target={"app": "google-sheet", "connection_id": conn["id"], "action": "append_rows",
                                            "spreadsheet_id": "sheet_1"})
    with pytest.raises(service.GoalRefused) as e:
        await service.item_action(store, customer.id, bad["id"], "send", actor="console")
    assert e.value.code == "rows_invalid"
