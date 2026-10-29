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
    # The survey (Q138 A) has its own tests (test_goal_criteria.py); here each
    # planner script starts with the plan.
    monkeypatch.setattr(s, "goal_survey_max_tool_calls", 0)
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
    anth.script_tool_use("submit_step", {"n": 2, "note": "Drafted the one reply needed."}, "tu_08")
    # The step has no criteria: an independent reviewer checks its own words (Q135 A).
    anth.script_text(json.dumps({"passed": True, "why": "A reply to the one email that needs one is drafted.", "missing": []}))
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
                            "web_search", "web_fetch", "save_fact", "lattices", "lattice_rows", "lattice_stale", "lattice_aggregate", "lattice_sql", "web_tables", "lattice_import",
                            "lattice_create", "lattice_write",
                            "add_action_item", "write_draft", "send_draft", "close_item", "submit_step", "report_stuck", "note_report"])
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
    # The model's "Crystal memory" is Crystal storage, with what that means.
    assert [d["answer"] for d in decided["plan"]["decisions"]] == [f"Crystal storage ({planner.CRYSTAL_STORAGE_DETAIL})",
                                                                  "Only the playoffs"]
    # The planner gets the answers as final and writes the plan; a repeated question is dropped.
    anth2 = FakeAnthropic()
    anth2.script_text(_plan_json(needs_apps=[], decisions=[
        {"question": "Where should the catalog live?", "why": "again", "recommended": 0,
         "options": [{"label": "a", "detail": ""}, {"label": "b", "detail": ""}]}]))
    await runner.tick(store, llm=anth2)
    assert "Where should the catalog live? -> Crystal storage (No app needed." in anth2.calls[0]["messages"][0]["content"]
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
    assert labels[:3] == ["Crystal storage", "Notion", "Google Sheets"] and len(labels) == 4
    assert d["recommended"] == 0 and "can write: create spreadsheets, add rows" in d["options"][2]["detail"]
    # The person picks Google Sheets; the planner may still ask for Notion: it is never asked twice.
    await service.decide(store, customer.id, g["id"], [{"index": 0, "choice": 2}])
    anth2 = FakeAnthropic()
    anth2.script_text(_plan_json(needs_apps=["notion", "google-sheet"], needs_acts=["append_rows"], decisions=[]))
    await runner.tick(store, llm=anth2)
    assert "Never build a plan around an app" in anth2.calls[0]["system"]
    assert "Already done" not in anth2.calls[0]["messages"][0]["content"]  # nothing done: no anchoring
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


@pytest.mark.asyncio
async def test_the_models_own_app_question_gains_the_writable_alternatives_it_left_out(store, customer):
    g = await service.create(store, customer.id, title="Sports catalog",
                             directive="Catalog every NBA and NFL game; let me browse by player or team.")
    anth = FakeAnthropic()
    anth.script_text(_plan_json(steps=[], needs_apps=["notion"], decisions=[
        {"question": "Where should the game catalog live?", "why": "All writes go there.", "recommended": 0,
         "options": [{"label": "Notion (recommended)", "detail": "Databases and views"},
                     {"label": "Crystal memory only", "detail": "Short facts"}]}]))
    await runner.tick(store, llm=anth)
    goal = await store.get_goal(customer.id, g["id"])
    assert goal["status"] == "deciding" and len(goal["plan"]["decisions"]) == 1  # no duplicate question
    labels = [o["label"] for o in goal["plan"]["decisions"][0]["options"]]
    assert labels == ["Notion", "Crystal storage", "Google Sheets"]
    assert goal["plan"]["decisions"][0]["recommended"] == 1  # memory, not the unconnected app
    assert "can write: create spreadsheets, add rows" in goal["plan"]["decisions"][0]["options"][2]["detail"]


# --- lattices: Crystal storage's tables (2026-10-25) ------------------------------------

GAME_LATTICE = {"name": "Games 2026-27", "description": "Every NBA and NFL game.",
                "columns": [{"key": "date", "label": "Date", "type": "date", "kind": "snapshot", "entity": False, "refresh_minutes": 0},
                            {"key": "league", "label": "League", "type": "text", "kind": "snapshot", "entity": True, "refresh_minutes": 0},
                            {"key": "home", "label": "Home", "type": "text", "kind": "snapshot", "entity": True, "refresh_minutes": 0},
                            {"key": "away", "label": "Away", "type": "text", "kind": "snapshot", "entity": True, "refresh_minutes": 0},
                            {"key": "home_score", "label": "Home score", "type": "number", "kind": "snapshot", "entity": False, "refresh_minutes": 0}],
                "key_columns": ["date", "home", "away"]}
SHOOTING_LATTICE = {"name": "Player shooting", "description": "Season shooting by player.",
                    "columns": [{"key": "player", "label": "Player", "type": "text", "kind": "snapshot", "entity": True, "refresh_minutes": 0},
                                {"key": "fg_pct", "label": "FG %", "type": "number", "kind": "live", "entity": False, "refresh_minutes": 1440}],
                    "key_columns": ["player"]}


@pytest.mark.asyncio
async def test_crystal_storage_plans_lattices_and_check_ins_write_rows(store, customer):
    from crystal_cache.cognition.controls import undo_action
    from crystal_cache.lattices import service as lat
    g = await service.create(store, customer.id, title="Sports catalog",
                             directive="Catalog every NBA and NFL game; let me browse by player or team.")
    await store.update_goal(customer.id, g["id"], plan={"decisions": [
        {"question": "Where should the game catalog live?", "options": [{"label": "Crystal storage", "detail": "x"}],
         "recommended": 0, "answer": "Crystal storage"}]})
    bad = {"name": "Broken", "description": "", "columns": [{"key": "Bad Key", "label": "", "type": "text", "kind": "snapshot",
                                                              "entity": False, "refresh_minutes": 0}], "key_columns": ["bad key"]}
    anth = FakeAnthropic()
    anth.script_text(_plan_json(needs_apps=[], needs_acts=[], decisions=[], lattices=[GAME_LATTICE, SHOOTING_LATTICE, bad],
                                steps=[{"title": "Backfill the season", "detail": "Write games to Games 2026-27."},
                                       {"title": "Each check-in: add results, refresh shooting", "detail": "lattice_stale first."}]))
    await runner.tick(store, llm=anth)
    system = anth.calls[0]["system"]
    assert "a LATTICE for many records with the same fields" in system and '"Crystal storage" is always one option' in system
    planned = await store.get_goal(customer.id, g["id"])
    assert planned["status"] == "active"
    assert [x["name"] for x in planned["plan"]["lattices"]] == ["Games 2026-27", "Player shooting"]  # the broken one dropped
    made = await lat.list_view(store, customer.id)
    assert sorted(x["name"] for x in made) == ["Games 2026-27", "Player shooting"]
    shooting = next(x for x in made if x["name"] == "Player shooting")
    assert shooting["columns"][1]["kind"] == "live" and shooting["columns"][1]["refresh_minutes"] == 1440
    assert "refresh_minutes" not in shooting["columns"][0]
    # A replan reuses the tables; it never makes a second one.
    from crystal_cache.goals.planner import ensure_plan_lattices
    await ensure_plan_lattices(store, customer.id, planned, dict(planned["plan"]))
    assert len(await lat.list_view(store, customer.id)) == 2

    # A check-in writes rows through the tools, ledgered and undoable.
    anth2 = FakeAnthropic()
    anth2.script_tool_use("goal_plan", {}, "tu_00")
    anth2.script_tool_use("lattice_write", {"lattice": "Games 2026-27", "source": "https://www.nba.com/schedule",
                                            "reason": "Backfill the season's first games.",
                                            "rows": [{"date": "2026-10-21", "league": "NBA", "home": "Celtics", "away": "Knicks", "home_score": 109},
                                                     {"date": "2026-10-21", "league": "NBA", "home": "Lakers", "away": "Warriors"}]}, "tu_01")
    anth2.script_tool_use("lattice_write", {"lattice": "Games 2026-27", "source": "https://espn.com", "reason": "Results came in tonight.",
                                            "rows": [{"date": "2026-10-21", "home": "Celtics", "away": "Knicks", "home_score": 111},
                                                     {"date": "2026-10-21", "home": "Lakers", "away": "Warriors", "home_score": 118}]}, "tu_02")
    anth2.script_tool_use("lattice_rows", {"lattice": "Games 2026-27", "filters": [{"column": "home", "op": "eq", "value": "celtics"}]}, "tu_03")
    anth2.script_tool_use("lattice_stale", {"lattice": "Player shooting"}, "tu_04")
    anth2.script_tool_use("lattice_write", {"lattice": "Games 2026-27", "source": "x", "reason": "short", "rows": [{"date": "2026-10-22"}]}, "tu_05")
    anth2.script_text("Did: wrote 2 games.\nWaiting: nothing.\nNext: more games.")
    out = await run_goal_session(store=store, customer=customer, goal=planned, llm=anth2)
    assert out["ok"] and out["actions"] == 2
    assert _result(anth2, 0)["lattices"][0]["name"] == "Games 2026-27" and _result(anth2, 0)["lattices"][0]["lattice_id"]
    first = _result(anth2, 1)
    assert (first["inserted"], first["updated"]) == (2, 0) and "inverse" not in first and "row_ids" not in first
    second = _result(anth2, 2)
    assert (second["inserted"], second["updated"]) == (0, 1)
    assert second["locked"] == [{"row_key": "2026-10-21 | celtics | knicks", "column": "home_score", "kept": 109, "offered": 111}]
    assert [r["values"]["home_score"] for r in _result(anth2, 3)["rows"]] == [109]
    assert _result(anth2, 4)["stale"] == []
    assert _result(anth2, 5)["code"] == "reason_required"
    rows = [r for r in await store.list_steward_actions(customer.id, limit=10) if r["action"] == "lattice_write"]
    assert len(rows) == 2 and all(r["actor"] == f"goal:{g['id']}" and r["undoable"] for r in rows)
    assert rows[0]["label"].startswith("Goal wrote 1 changed row to Games 2026-27") or rows[1]["label"].startswith("Goal wrote 1 changed row")
    first_action = next(r for r in rows if "2 new" in r["label"])
    assert (await undo_action(store, customer.id, first_action["id"], actor="console"))["undone"] is True
    assert await store.count_lattice_rows(customer.id) == 0


# --- Q120 A: lattice designs are checked against the data-model rules ----------------------

def _cov(lattice, *columns):
    return {"kind": "coverage", "label": "", "lattice": lattice, "column": "", "columns": list(columns), "filters": [],
            "min": 0, "min_fraction": 0.98, "of_lattice": "", "of_column": "", "min_per": 0, "runs": 0, "text": "",
            "basis": "estimated", "basis_detail": ""}


def _col(key, type_="text", kind="snapshot", entity=False, refresh=0):
    return {"key": key, "label": key, "type": type_, "kind": kind, "entity": entity, "refresh_minutes": refresh}


def test_lint_names_what_breaks_the_rules():
    bad = planner.normalize_lattices([{"name": "games", "description": "", "key_columns": ["home", "away"],
                                       "columns": [_col("home", entity=True), _col("away", entity=True),
                                                   _col("home_score"), _col("fg"), _col("fg_pct", "number"),
                                                   _col("winner")]}])
    issues = planner.lint_lattices(bad)
    assert "games: key column 'home' is a display value; key rows on the source's id (rule 2)." in issues
    assert any("'home_score' looks numeric but is text" in i for i in issues)
    assert any("'fg_pct' looks computed" in i for i in issues)
    assert any("'winner' looks computed" in i for i in issues)
    assert "games: no date or season column to filter by (rule 5)." in issues
    good = planner.normalize_lattices([{"name": "nba_games", "description": "", "key_columns": ["game_id"],
                                        "columns": [_col("game_id"), _col("date", "date"), _col("season"),
                                                    _col("home_team", entity=True), _col("away_team", entity=True),
                                                    _col("home_points", "number"), _col("away_points", "number")]}])
    assert planner.lint_lattices(good) == []
    wide = planner.normalize_lattices([{"name": "wide", "description": "", "key_columns": ["game_id"],
                                        "columns": [_col("game_id"), _col("date", "date")] + [_col(f"c{i}", "number") for i in range(30)]}])
    assert planner.lint_lattices(wide) == ["wide: 32 columns; split it by level of detail (rule 6)."]


@pytest.mark.asyncio
async def test_a_design_that_breaks_the_rules_is_reviewed_before_it_is_created(store, customer):
    from crystal_cache.lattices import service as lat
    g = await service.create(store, customer.id, title="Sports catalog",
                             directive="Catalog every NBA game; let me browse by player or team.")
    await store.update_goal(customer.id, g["id"], plan={"decisions": [
        {"question": "Where?", "options": [{"label": "Crystal storage", "detail": "x"}], "recommended": 0, "answer": "Crystal storage"}]})
    first = {"name": "games", "description": "Games.", "key_columns": ["date", "home", "away"],
             "columns": [_col("date", "date"), _col("home", entity=True), _col("away", entity=True), _col("score")]}
    fixed = [{"name": "nba_games", "description": "One row per NBA game.", "key_columns": ["game_id"],
              "columns": [_col("game_id"), _col("date", "date"), _col("season"), _col("home_team", entity=True),
                          _col("away_team", entity=True), _col("home_points", "number"), _col("away_points", "number")]},
             {"name": "nba_player_games", "description": "One row per player per game.", "key_columns": ["game_id", "player_id"],
              "columns": [{**_col("game_id", "ref"), "ref": "nba_games"}, _col("player_id"), _col("player", entity=True),
                          _col("team", entity=True),
                          _col("date", "date"), _col("points", "number")]}]
    anth = FakeAnthropic()
    anth.script_text(_plan_json(needs_apps=[], needs_acts=[], decisions=[], lattices=[first],
                                steps=[{"title": "Backfill games", "detail": "Write each scoreboard to games."},
                                       {"title": "Each check-in: add new games", "detail": "games"}]))
    anth.script_text(json.dumps({"lattices": fixed, "steps": [
        {"title": "Backfill games from daily scoreboards", "detail": "Write each scoreboard to nba_games.",
         "criteria": [_cov("nba_games", "home_points", "away_points")]},
        {"title": "Each check-in: add new games", "detail": "nba_games, then box scores into nba_player_games."}]}))
    await runner.tick(store, llm=anth)
    assert "Key every row on stable ids from the source" in anth.calls[0]["system"]
    assert "plan it as catch-up passes" in anth.calls[0]["system"]
    review = anth.calls[1]
    assert "You review the table design" in review["system"]
    assert "key column 'home' is a display value" in review["messages"][0]["content"]
    planned = await store.get_goal(customer.id, g["id"])
    plan = planned["plan"]
    assert [x["name"] for x in plan["lattices"]] == ["nba_games", "nba_player_games"]
    assert plan["lattice_review"]["revised"] is True and plan["lattice_review"]["remaining"] == []
    assert plan["steps"][0]["title"] == "Backfill games from daily scoreboards"
    assert sorted(x["name"] for x in await lat.list_view(store, customer.id)) == ["nba_games", "nba_player_games"]


@pytest.mark.asyncio
async def test_a_failed_review_keeps_the_first_design(store, customer):
    from crystal_cache.lattices import service as lat
    g = await service.create(store, customer.id, title="Sports catalog", directive="Catalog every NBA game for me please.")
    await store.update_goal(customer.id, g["id"], plan={"decisions": [
        {"question": "Where?", "options": [{"label": "Crystal storage", "detail": "x"}], "recommended": 0, "answer": "Crystal storage"}]})
    first = {"name": "games", "description": "", "key_columns": ["home"], "columns": [_col("home"), _col("date", "date")]}
    anth = FakeAnthropic()
    anth.script_text(_plan_json(needs_apps=[], needs_acts=[], decisions=[], lattices=[first]))
    anth.script_text("not json")
    await runner.tick(store, llm=anth)
    plan = (await store.get_goal(customer.id, g["id"]))["plan"]
    assert plan["lattice_review"]["revised"] is False and plan["lattice_review"]["remaining"]
    assert [x["name"] for x in await lat.list_view(store, customer.id)] == ["games"]


def test_lint_catches_invented_keys_and_names_without_ids():
    specs = planner.normalize_lattices([{"name": "nba_player_games", "description": "", "key_columns": ["stat_id"],
                                         "columns": [_col("stat_id"), _col("game_id"), _col("date", "date"),
                                                     _col("player_name", entity=True), _col("points", "number")]}])
    issues = planner.lint_lattices(specs)
    assert any("key 'stat_id' is invented" in i for i in issues)
    assert any("'player_name' is browsed by name but there is no 'player_id'" in i for i in issues)


@pytest.mark.asyncio
async def test_every_design_is_reviewed_even_when_the_checker_finds_nothing(store, customer):
    g = await service.create(store, customer.id, title="Orders", directive="Keep every order we receive in a table.")
    await store.update_goal(customer.id, g["id"], plan={"decisions": [
        {"question": "Where?", "options": [{"label": "Crystal storage", "detail": "x"}], "recommended": 0, "answer": "Crystal storage"}]})
    clean = {"name": "orders", "description": "", "key_columns": ["order_id"],
             "columns": [_col("order_id"), _col("date", "date"), _col("customer_id"), _col("total_cents", "number")]}
    anth = FakeAnthropic()
    steps = [{"title": "Import past orders", "detail": "x", "catch_up": True, "criteria": [_cov("orders", "total_cents")]}]
    anth.script_text(_plan_json(needs_apps=[], needs_acts=[], decisions=[], lattices=[clean], steps=steps))
    anth.script_text(json.dumps({"lattices": [clean], "steps": steps}))
    await runner.tick(store, llm=anth)
    assert len(anth.calls) == 2 and "none (it only reads names and types; review anyway)" in anth.calls[1]["messages"][0]["content"]
    assert "each line item of an order" in anth.calls[1]["system"]
    plan = (await store.get_goal(customer.id, g["id"]))["plan"]
    assert plan["lattice_review"]["issues"] == [] and plan["steps"][0]["catch_up"] is True


def test_catch_up_steps_check_in_faster_until_they_are_done():
    raw = json.loads(_plan_json(check_every_minutes=1440, catch_up_every_minutes=5, steps=[
        {"title": "Import history", "detail": "x", "catch_up": True}, {"title": "Each check-in: new records", "detail": "y"}]))
    p = planner.normalize_plan(raw, min_minutes=15, statuses=ALL_CONNECTED)
    assert p["catch_up_every_minutes"] == 15 and p["steps"][0]["catch_up"] is True and p["steps"][1]["catch_up"] is False
    assert planner.next_check_minutes(p, 15) == 15
    p["steps"][0]["done"] = True
    assert planner.next_check_minutes(p, 15) == 1440
    assert planner.normalize_plan(json.loads(_plan_json()), min_minutes=15, statuses=ALL_CONNECTED)["catch_up_every_minutes"] is None
    assert planner.next_check_minutes({"check_every_minutes": 30}, 15) == 30


def test_lint_flags_a_kind_column_on_a_wide_lattice():
    games = planner.normalize_lattices([{"name": "games", "description": "", "key_columns": ["game_id"],
                                         "columns": [_col("game_id"), _col("league", entity=True), _col("date", "date")]
                                         + [_col(f"stat{i}", "number") for i in range(8)]}])
    assert any("'league' says which kind of record a row is" in i for i in planner.lint_lattices(games))


@pytest.mark.asyncio
async def test_a_replan_shows_what_is_done_as_facts_not_the_old_steps(store, customer):
    goal = await _planned_goal(store, customer)
    plan = dict(goal["plan"])
    plan["steps"] = [dict(s) for s in plan["steps"]]
    plan["steps"][1]["done"] = True
    goal = await store.update_goal(customer.id, goal["id"], plan=plan)
    text = planner.planner_user(goal, [])
    assert "Already done (facts, not a plan to copy" in text and "Draft replies" in text
    assert "Each check-in: triage new mail" not in text  # the unfinished old step is not shown


def test_a_missing_tool_never_holds_a_goal_and_known_tools_are_not_missing():
    items = planner.normalize_missing([
        {"kind": "tool", "need": "Automated lattice_import from HTML tables", "why": "x", "suggestion": "confirm it", "blocking": True},
        {"kind": "tool", "need": "A headless browser for pages drawn with JavaScript", "why": "x", "suggestion": "y", "blocking": True},
        {"kind": "permission", "need": "May post on its own", "why": "x", "suggestion": "y", "blocking": True},
    ], [], [])
    assert [(i["kind"], i["blocking"]) for i in items] == [("tool", False), ("permission", True)]
    assert "never ask the person to confirm or test it" in planner.planner_system(15)
    assert "Never drop data the goal asks for" in planner.review_system()


@pytest.mark.asyncio
async def test_the_review_records_a_lattice_it_removed(store, customer):
    g = await service.create(store, customer.id, title="Orders", directive="Keep every order and each line item.")
    await store.update_goal(customer.id, g["id"], plan={"decisions": [
        {"question": "Where?", "options": [{"label": "Crystal storage", "detail": "x"}], "recommended": 0, "answer": "Crystal storage"}]})
    orders = {"name": "orders", "description": "", "key_columns": ["order_id"],
              "columns": [_col("order_id"), _col("date", "date"), _col("total_cents", "number")]}
    lines = {"name": "order_lines", "description": "", "key_columns": ["order_id", "line_id"],
             "columns": [_col("order_id"), _col("line_id"), _col("date", "date"), _col("qty", "number")]}
    anth = FakeAnthropic()
    anth.script_text(_plan_json(needs_apps=[], needs_acts=[], decisions=[], lattices=[orders, lines]))
    anth.script_text(json.dumps({"lattices": [orders], "steps": [{"title": "Import orders", "detail": "x", "catch_up": True}]}))
    await runner.tick(store, llm=anth)
    review = (await store.get_goal(customer.id, g["id"]))["plan"]["lattice_review"]
    assert review["removed"] == ["order_lines"]


@pytest.mark.asyncio
async def test_a_check_in_cannot_create_a_badly_shaped_lattice_and_catch_up_gets_room(store, customer):
    from crystal_cache.lattices import service as lat
    goal = await _planned_goal(store, customer)
    plan = dict(goal["plan"])
    plan["steps"] = [{**dict(s), "catch_up": i == 0} for i, s in enumerate(plan["steps"])]
    goal = await store.update_goal(customer.id, goal["id"], plan=plan)
    anth = FakeAnthropic()
    anth.script_tool_use("lattice_create", {"name": "nba_schedule_import", "description": "copy of the page",
                                            "columns": [{"key": "date"}, {"key": "away_team"}, {"key": "home_team"}],
                                            "key_columns": ["date", "away_team", "home_team"]}, "tu_01")
    anth.script_tool_use("lattice_create", {"name": "nba_games", "description": "one row per game",
                                            "columns": [{"key": "game_id"}, {"key": "date", "type": "date"},
                                                        {"key": "home_team", "entity": True}],
                                            "key_columns": ["game_id"]}, "tu_02")
    anth.script_text("Did: made nba_games.\nWaiting: nothing.\nNext: import.")
    await run_goal_session(store=store, customer=customer, goal=goal, llm=anth)
    assert "40 tool calls" in anth.calls[0]["system"][0]["text"]
    bad = _result(anth, 0)
    assert bad["code"] == "design_rejected" and "key column 'date' is a display value" in bad["error"]
    assert _result(anth, 1)["ok"] is True
    assert [x["name"] for x in await lat.list_view(store, customer.id)] == ["nba_games"]
    assert "never into a copy of the page" in anth.calls[0]["system"][0]["text"]


@pytest.mark.asyncio
async def test_a_check_in_cannot_copy_a_lattice_after_a_failed_import(store, customer):
    # Series 372 (2026-10-26): an import into nba_games failed on one column,
    # and the check-in made nba_games2 and nba_games3 with the same shape.
    from crystal_cache.lattices import service as lat
    goal = await _planned_goal(store, customer)
    cols = [{"key": "game_id"}, {"key": "date", "type": "date"}, {"key": "home_team", "entity": True},
            {"key": "away_team", "entity": True}, {"key": "source_url", "type": "url"}]
    await lat.create(store, customer.id, name="nba_games", columns=cols, key_columns=["game_id"])
    anth = FakeAnthropic()
    anth.script_tool_use("lattice_create", {"name": "nba_games2", "description": "same again",
                                            "columns": cols[:4] + [{"key": "season"}], "key_columns": ["game_id"]}, "tu_01")
    anth.script_tool_use("lattice_create", {"name": "nba_player_games", "description": "one row per player per game",
                                            # Series 380: the existing nba_games is a target, so game_id points at it.
                                            "columns": [{"key": "game_id", "type": "ref", "ref": "nba_games"},
                                                        {"key": "player_id"}, {"key": "date", "type": "date"},
                                                        {"key": "pts", "type": "number"}],
                                            "key_columns": ["game_id", "player_id"]}, "tu_02")
    anth.script_text("Did: nothing new.\nWaiting: nothing.\nNext: import.")
    await run_goal_session(store=store, customer=customer, goal=goal, llm=anth)
    copy = _result(anth, 0)
    assert copy["code"] == "lattice_exists" and "'nba_games'" in copy["error"]
    assert _result(anth, 1)["ok"] is True
    assert sorted(x["name"] for x in await lat.list_view(store, customer.id)) == ["nba_games", "nba_player_games"]


@pytest.mark.asyncio
async def test_a_plan_knows_the_limits_covers_every_subject_and_is_never_held_on_a_choice(store, customer):
    # Series 375 (2026-10-26): a fresh two-league goal kept 4 lattices (all
    # NBA; NFL silently cut), planned all history since 1946, stored season
    # totals on per-game rows, keyed browsing on bare ids and was held on
    # "which id system", a choice with nowhere for the person to answer.
    g = await service.create(store, customer.id, title="Sports", directive="Catalog all NBA and NFL games; browse by player or team.")
    await store.update_goal(customer.id, g["id"], plan={"decisions": [
        {"question": "Where?", "options": [{"label": "Crystal storage", "detail": "x"}], "recommended": 0, "answer": "Crystal storage"}]})
    many = [{"name": f"t{i}", "description": "", "key_columns": ["game_id"],
             "columns": [_col("game_id"), _col("date", "date"), _col("points", "number")]} for i in range(10)]
    player_games = {"name": "nba_player_games", "description": "", "key_columns": ["game_id", "player_id"],
                    "columns": [_col("game_id"), _col("player_id", entity=True), _col("date", "date"),
                                _col("points", "number"), _col("season_points", "number", "live", refresh=1440)]}
    anth = FakeAnthropic()
    anth.script_text(_plan_json(needs_apps=[], needs_acts=[], decisions=[], lattices=[player_games] + many, missing=[
        {"kind": "info", "need": "Which id system to key on", "why": "x", "suggestion": "decide", "blocking": True}]))
    fixed = {"name": "nba_player_games", "description": "", "key_columns": ["game_id", "player_id"],
             "columns": [_col("game_id"), _col("player_id", entity=True), _col("player_name"), _col("date", "date"),
                         _col("points", "number")]}
    anth.script_text(json.dumps({"lattices": [fixed], "steps": [{"title": "Import games", "detail": "x", "catch_up": True,
                                                                  "criteria": [_cov("nba_player_games", "points")]}]}))
    await runner.tick(store, llm=anth)
    system = anth.calls[0]["system"]
    assert "at most 10 lattices per plan, 250,000 rows per lattice" in system
    assert "the scope is a decision" in system and "Technical choices are yours" in system
    assert "each subject (each league" in system
    review = anth.calls[1]
    assert "Does every subject the person named have its lattices" in review["system"] and "LIMITS:" in review["system"]
    planned = await store.get_goal(customer.id, g["id"])
    plan = planned["plan"]
    assert planned["status"] != "needs_setup" and plan["blocked"] is False
    assert [m["blocking"] for m in plan["missing"]] == [False]
    assert [x["name"] for x in plan["lattices"]] == ["nba_player_games"]
    # t9 was over the cap of 10 and never reached the review; it is still on record as removed.
    assert sorted(plan["lattice_review"]["removed"]) == sorted(f"t{i}" for i in range(10))
    issues = plan["lattice_review"]["issues"]
    assert any("'season_points' is a total across rows" in i for i in issues)
    assert any("'player_id' is browsed by id with no name beside it" in i for i in issues)
    assert plan["lattice_review"]["remaining"] == []


@pytest.mark.asyncio
async def test_plan_limits_come_from_the_tier(store, customer):
    from types import SimpleNamespace
    lim = await planner.plan_limits(store, SimpleNamespace(id=customer.id, subscription_tier="starter_29"))
    assert lim == {"row_cap": 250_000, "rows_used": 0}
    assert "250,000 rows across all lattices" in planner.limits_text(lim)
    assert "no total row cap" in planner.limits_text({"row_cap": None})


def test_lint_keeps_one_kind_of_record_in_one_lattice_and_knows_a_winner_reference():
    # Q133 A (2026-10-26): a 50,000-row cap made the planner split player
    # games into nba_player_games_2023/2024/2025; the cap is 250,000 and a
    # split by period is a rule-1 issue. tipoff_winner_team_id is who won,
    # recorded, not a computed value.
    specs = planner.normalize_lattices([
        {"name": "nba_player_games_2024", "description": "", "key_columns": ["game_id", "player_id"],
         "columns": [_col("game_id"), _col("player_id"), _col("date", "date"), _col("pts", "number")]},
        {"name": "nba_games", "description": "", "key_columns": ["game_id"],
         "columns": [_col("game_id"), _col("date", "date"), _col("tipoff_winner_team_id"),
                     _col("tipoff_winner_team_name"), _col("winner")]}])
    issues = planner.lint_lattices(specs)
    assert any("nba_player_games_2024: one kind of record split by period" in i for i in issues)
    assert not any("tipoff_winner" in i for i in issues)
    assert any("'winner' looks computed" in i for i in issues)
    assert "never an option you know does not fit" in planner.planner_system(15)
    assert "never split across lattices by season" in planner.planner_system(15)


@pytest.mark.asyncio
async def test_a_catch_up_check_in_has_room_for_imports_and_an_empty_import_is_not_an_action(store, customer, monkeypatch):
    # 2026-10-26: the first check-in of the fresh sports goal hit its 8-action
    # cap while importing; failed imports counted as actions.
    from crystal_cache.lattices import service as lat
    from crystal_cache.search import tables as wt
    page = ("<table id='s'><tr><th>Game</th><th>Date</th></tr>"
            "<tr><td><a href='/g/1.html'>Box</a></td><td>Oct 22, 2024</td></tr></table>")

    async def loader(url):
        return "https://example.test/s", wt.absolute_links(wt.parse_tables(page), "https://example.test/s")
    monkeypatch.setattr(wt, "load_tables", loader)
    goal = await _planned_goal(store, customer)
    plan = dict(goal["plan"])
    plan["steps"] = [{**dict(s), "catch_up": i == 0} for i, s in enumerate(plan["steps"])]
    goal = await store.update_goal(customer.id, goal["id"], plan=plan)
    await lat.create(store, customer.id, name="games", key_columns=["game_id"],
                     columns=[{"key": "game_id"}, {"key": "date", "type": "date"}])
    anth = FakeAnthropic()
    bad = {"lattice": "games", "url": "example.test/s", "table": "s", "reason": "Backfill the schedule.",
           "columns": {"game_id": {"from": "Game", "link": True, "pattern": "/nope/(\\d+)"}, "date": "Date"}}
    good = {**bad, "columns": {"game_id": {"from": "Game", "link": True, "pattern": "/g/(\\d+)\\.html"}, "date": "Date"}}
    anth.script_tool_use("lattice_import", bad, "tu_01")
    anth.script_tool_use("lattice_import", good, "tu_02")
    anth.script_text("Did: imported games.\nWaiting: nothing.\nNext: more months.")
    from crystal_cache.config import get_settings
    assert get_settings().goal_catch_up_max_actions == 30
    monkeypatch.setattr(get_settings(), "goal_catch_up_max_actions", 1)
    monkeypatch.setattr(get_settings(), "goal_max_actions_per_session", 1)
    await run_goal_session(store=store, customer=customer, goal=goal, llm=anth)
    assert "at most 1 items" in anth.calls[0]["system"][0]["text"]
    assert _result(anth, 0)["inserted"] == 0 and _result(anth, 1)["inserted"] == 1
    q = await lat.query(store, customer.id, "games")
    assert q["rows"][0]["values"] == {"game_id": "1", "date": "2024-10-22"}
