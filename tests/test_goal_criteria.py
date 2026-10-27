"""Completion criteria, the survey, comments and rollback (2026-10-27;
Q135 A, Q136 A, Q137 A, Q138 A).

2026-10-26 a check-in called "import every game of three seasons" done
after the October pages (82 games of about 3,900); the plan had sized the
seasons at 22,000 games and 10 players a game from the planner's head.
Anthony: "define a completion criteria and if that step submission falls
short it is rejected"; "A quick research about the questions it has
would answer all those questions"; comments after a check-in steer the
next; "a rollback to any step should be possible".
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import update

from crystal_cache import beta
from crystal_cache.config import get_settings
from crystal_cache.connectors import catalog
from crystal_cache.connectors import service as conn_service
from crystal_cache.goals import criteria as crit
from crystal_cache.goals import planner, runner, service
from crystal_cache.goals.session import run_goal_session
from crystal_cache.infrastructure.schema import GoalSnapshotRow, StewardActionRow
from crystal_cache.lattices import service as lat
from fakes import FakeAnthropic

ALL_CONNECTED = {a.integration: "connected" for a in catalog.APPS}


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    import time
    beta.reset_for_tests()
    conn_service.reset_for_tests()
    conn_service._configured.update(keys={a.integration for a in catalog.APPS}, at=time.monotonic())
    monkeypatch.setattr(get_settings(), "goal_survey_max_tool_calls", 0)
    yield
    conn_service.reset_for_tests()


def _c(kind, **kw):
    base = {"kind": kind, "label": "", "lattice": "", "column": "", "columns": [], "filters": [], "min": 0,
            "min_fraction": 0, "of_lattice": "", "of_column": "", "min_per": 0, "runs": 0, "text": "",
            "basis": "estimated", "basis_detail": ""}
    base.update(kw)
    return base


GAMES = {"name": "games", "description": "One row per game.", "key_columns": ["game_id"],
         "columns": [{"key": "game_id"}, {"key": "date", "type": "date"}, {"key": "season"},
                     {"key": "home_points", "type": "number"}]}
PLAYER_GAMES = {"name": "player_games", "description": "One row per player per game.",
                "key_columns": ["game_id", "player_id"],
                "columns": [{"key": "game_id"}, {"key": "player_id"}, {"key": "player_name"},
                            {"key": "date", "type": "date"}, {"key": "points", "type": "number"}]}

SEASON_DONE = [
    _c("coverage", lattice="games", columns=["home_points"], min_fraction=0.98, basis="measured",
       basis_detail="nba.com lists 3 games"),
    _c("reference", lattice="player_games", column="game_id", of_lattice="games", of_column="game_id", min_per=2,
       min_fraction=1.0),
    _c("distinct", lattice="games", column="date", min=2),
    _c("rows", lattice="games", min=3),
]


async def _goal(store, customer, step_criteria, *, second=None):
    g = await service.create(store, customer.id, title="Season", directive="Keep every game of the season with its players.")
    raw = {"summary": "s", "success_measure": "m", "check_every_minutes": 60, "catch_up_every_minutes": 60,
           "steps": [{"title": "Import the season", "detail": "games and player_games", "catch_up": True,
                      "repeats": False, "criteria": step_criteria},
                     second or {"title": "Check the players", "detail": "x", "catch_up": False, "repeats": False,
                                "criteria": [_c("rows", lattice="player_games", min=1)]},
                     {"title": "Each check-in: add new games", "detail": "y", "catch_up": False, "repeats": True,
                      "criteria": []}],
           "needs_apps": [], "needs_acts": [], "standing_instructions": "", "missing": [], "decisions": [],
           "lattices": [GAMES, PLAYER_GAMES]}
    plan = planner.normalize_plan(raw, min_minutes=15, statuses=ALL_CONNECTED)
    await planner.ensure_plan_lattices(store, customer.id, g, plan)
    g = await store.update_goal(customer.id, g["id"], plan=plan, planned_at=datetime.now(timezone.utc), next_run_at=None)
    await store.add_goal_snapshot(customer.id, g["id"], step_n=0, plan=plan)
    return g


async def _write(store, cid, goal, name, rows, source="https://example.test/s"):
    actor = f"goal:{goal['id']}"
    out = await lat.write_rows(store, cid, name, rows, source=source, actor=actor)
    await lat.ledger(store, cid, out, actor=actor, reason="Backfill the season from the schedule.")
    return out


def _result(anth, i):
    msg = anth.calls[i + 1]["messages"][-1]["content"][0]
    return json.loads(msg["content"]) if isinstance(msg["content"], str) else msg["content"]


# --- criteria are cleaned ----------------------------------------------------------------

def test_criteria_that_cannot_fail_or_name_nothing_are_dropped():
    specs = planner.normalize_lattices([GAMES, PLAYER_GAMES])
    clean, dropped = crit.normalize_criteria([
        _c("rows", lattice="games", min=0),
        _c("coverage", lattice="games", columns=["nope"]),
        _c("coverage", lattice="teams", columns=["x"]),
        _c("rubric", text="ok"),
        _c("coverage", lattice="games", columns=["home_points"]),
        _c("reference", lattice="player_games", column="game_id", of_lattice="games", min_per=20),
        _c("mystery", lattice="games"),
    ], specs)
    assert [c["kind"] for c in clean] == ["coverage", "reference"]
    assert clean[0]["min_fraction"] == crit.DEFAULT_FRACTION and clean[0]["label"] == "games: 98% of rows have home_points"
    assert clean[1]["of_column"] == "game_id" and clean[1]["min_per"] == 20
    assert len(dropped) == 4 and "cannot fail" in dropped[0] and "no column nope" in dropped[1]
    assert "no such lattice" in dropped[2] and "checkable text" in dropped[3]


def test_a_step_that_moves_rows_without_a_measure_is_a_review_issue():
    plan = {"lattices": [GAMES], "steps": [
        {"n": 1, "title": "Import games", "detail": "into games", "criteria": [_c("rubric", text="Looks complete enough.")]},
        {"n": 2, "title": "Write a weekly note", "detail": "x", "criteria": []},
        {"n": 3, "title": "Each check-in", "detail": "games", "repeats": True, "criteria": []}]}
    issues = crit.lint_criteria(plan)
    assert len(issues) == 2 and "works games but no criterion measures it" in issues[0] and "no criteria" in issues[1]


# --- submit, measure, reject, accept -------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_short_step_is_rejected_with_what_is_missing_and_accepted_when_it_is_whole(store, customer):
    cid = customer.id
    goal = await _goal(store, customer, SEASON_DONE)
    await _write(store, cid, goal, "games", [
        {"game_id": "g1", "date": "2024-10-22", "season": "2024-25", "home_points": 110},
        {"game_id": "g2", "date": "2024-10-22", "season": "2024-25", "home_points": 99},
        {"game_id": "g3", "date": "2024-10-23", "season": "2024-25"}])
    await _write(store, cid, goal, "player_games", [{"game_id": "g1", "player_id": "p1"}, {"game_id": "g1", "player_id": "p2"}])

    anth = FakeAnthropic()
    anth.script_tool_use("goal_plan", {}, "tu_01")
    anth.script_tool_use("submit_step", {"n": 1, "note": "Imported the season."}, "tu_02")
    anth.script_tool_use("submit_step", {"n": 3}, "tu_03")
    anth.script_text("Did: submitted step 1 (rejected).\nWaiting: nothing.\nNext: box scores for g2 and g3.")
    out = await run_goal_session(store=store, customer=customer, goal=goal, llm=anth)
    assert out["ok"]
    view = _result(anth, 0)
    assert view["steps"][0]["criteria"][0] == "games: 98% of rows have home_points" and view["steps"][2]["repeats"] is True
    rej = _result(anth, 1)
    assert rej["ok"] is False and rej["code"] == "step_rejected"
    short = {s["criterion"]: s for s in rej["short"]}
    cov = short["games: 98% of rows have home_points"]
    assert cov["measured"] == "2 of 3 filled" and cov["examples"] == ["g3"]
    ref = short["each game_id in games has 2+ rows in player_games"]
    assert ref["measured"] == "1 of 3 covered" and ref["examples"] == ["g2", "g3"] and "fewest 0, most 2" in ref["gap"]
    assert "games: at least 3 rows" not in short and "games: at least 2 distinct date" not in short
    assert _result(anth, 2)["code"] == "step_repeats"
    assert len(anth.calls) == 4  # measured criteria fell short: no reviewer was asked
    after = await store.get_goal(cid, goal["id"])
    s1 = after["plan"]["steps"][0]
    assert s1["done"] is False and s1["state"] == "rejected" and s1["attempts"] == 1
    assert s1["last_check"]["verdict"] == "rejected"
    runs = await service.runs_view(store, cid, goal["id"])
    assert runs[0]["submissions"] == [{"n": 1, "verdict": "rejected"}] and runs[0]["tools"]["submit_step"] == 2

    # The live view measures the same thing without a check-in.
    prog = await crit.progress(store, cid, after)
    assert [r["passed"] for r in prog[0]["results"]] == [False, False, True, True]

    await _write(store, cid, goal, "games", [{"game_id": "g3", "home_points": 101}])
    await _write(store, cid, goal, "player_games", [{"game_id": g, "player_id": p} for g in ("g2", "g3") for p in ("p1", "p2")])
    anth2 = FakeAnthropic()
    anth2.script_tool_use("submit_step", {"n": 1, "note": "Box scores filled."}, "tu_01")
    anth2.script_text("Did: step 1 accepted.\nWaiting: nothing.\nNext: step 2.")
    await run_goal_session(store=store, customer=customer, goal=after, llm=anth2)
    ok = _result(anth2, 0)
    assert ok["verdict"] == "accepted" and len(ok["checked"]) == 4
    done = await store.get_goal(cid, goal["id"])
    assert done["plan"]["steps"][0]["done"] is True and done["plan"]["steps"][0]["accepted_at"]
    snaps = await store.list_goal_snapshots(cid, goal["id"])
    assert [s["step_n"] for s in snaps] == [0, 1] and snaps[1]["rows"] == {"games": 3, "player_games": 6}


@pytest.mark.asyncio
async def test_the_reviewer_is_asked_only_when_the_counts_pass_and_its_no_is_a_rejection(store, customer):
    cid = customer.id
    goal = await _goal(store, customer, [_c("rows", lattice="games", min=1),
                                         _c("rubric", text="Every game has a plausible final score.")])
    anth = FakeAnthropic()
    anth.script_tool_use("submit_step", {"n": 1}, "tu_01")
    anth.script_text("Did: nothing.\nWaiting: nothing.\nNext: import.")
    await run_goal_session(store=store, customer=customer, goal=goal, llm=anth)
    first = _result(anth, 0)
    assert first["code"] == "step_rejected" and len(anth.calls) == 2
    assert any(s["gap"] == "Checked once the measured criteria pass." for s in first["short"])

    await _write(store, cid, goal, "games", [{"game_id": "g1", "date": "2024-10-22", "season": "2024-25", "home_points": 900}])
    goal = await store.get_goal(cid, goal["id"])
    anth2 = FakeAnthropic()
    anth2.script_tool_use("submit_step", {"n": 1, "note": "All games in."}, "tu_01")
    anth2.script_text(json.dumps({"passed": False, "why": "900 points is not a basketball score.",
                                  "missing": ["g1's home_points looks wrong"]}))
    anth2.script_text("Did: step 1 rejected.\nWaiting: nothing.\nNext: fix g1.")
    await run_goal_session(store=store, customer=customer, goal=goal, llm=anth2)
    judge = anth2.calls[1]
    assert judge["model"] == "large" and judge["json_schema"] == crit.VERDICT_SCHEMA
    assert "Every game has a plausible final score." in judge["messages"][0]["content"]
    assert '"home_points": 900' in judge["messages"][0]["content"]
    second = _result(anth2, 1)  # calls[1] was the reviewer
    assert second["code"] == "step_rejected"
    assert [s["gap"] for s in second["short"]] == ["g1's home_points looks wrong"]


@pytest.mark.asyncio
async def test_a_person_criterion_waits_for_the_person(store, customer):
    cid = customer.id
    goal = await _goal(store, customer, [_c("rows", lattice="games", min=1),
                                         _c("person", text="The game list matches the official schedule.")])
    await _write(store, cid, goal, "games", [{"game_id": "g1", "date": "2024-10-22", "season": "2024-25", "home_points": 9}])
    anth = FakeAnthropic()
    anth.script_tool_use("submit_step", {"n": 1}, "tu_01")
    anth.script_tool_use("submit_step", {"n": 1}, "tu_02")
    anth.script_text("Did: step 1 waits for you.\nWaiting: your review.\nNext: step 2.")
    await run_goal_session(store=store, customer=customer, goal=goal, llm=anth)
    assert _result(anth, 0)["verdict"] == "awaiting_review" and _result(anth, 1)["code"] == "awaiting_review"
    with pytest.raises(service.GoalRefused) as e:
        await service.review_step(store, cid, goal["id"], 1, approve=False, note="")
    assert e.value.code == "note_required"
    with pytest.raises(service.GoalRefused) as e:
        await service.review_step(store, cid, goal["id"], 2, approve=True)
    assert e.value.code == "not_awaiting_review"
    g = await service.review_step(store, cid, goal["id"], 1, approve=True, note="Matches nba.com.")
    assert g["plan"]["steps"][0]["done"] is True and g["plan"]["steps"][0]["state"] == "accepted"
    assert [s["step_n"] for s in await store.list_goal_snapshots(cid, goal["id"])] == [0, 1]
    assert (await store.list_goal_comments(cid, goal["id"]))[0]["body"] == "Step 1 approved: Matches nba.com."


@pytest.mark.asyncio
async def test_the_person_edits_criteria_and_bad_ones_are_refused(store, customer):
    cid = customer.id
    goal = await _goal(store, customer, SEASON_DONE)
    with pytest.raises(service.GoalRefused) as e:
        await service.set_criteria(store, cid, goal["id"], 1, [_c("coverage", lattice="games", columns=["nope"])])
    assert e.value.code == "bad_criteria" and "no column nope" in e.value.message
    with pytest.raises(service.GoalRefused) as e:
        await service.set_criteria(store, cid, goal["id"], 3, [])
    assert e.value.code == "step_repeats"
    g = await service.set_criteria(store, cid, goal["id"], 1, [_c("distinct", lattice="games", column="season", min=3,
                                                                    basis="measured", basis_detail="3 seasons asked")])
    assert g["plan"]["steps"][0]["criteria"] == [{
        "kind": "distinct", "label": "games: at least 3 distinct season", "basis": "measured",
        "basis_detail": "3 seasons asked", "lattice": "games", "min": 3, "filters": [], "column": "season"}]


@pytest.mark.asyncio
async def test_saturation_passes_only_after_passes_that_found_nothing_new(store, customer):
    cid = customer.id
    goal = await _goal(store, customer, [_c("saturation", lattice="games", runs=2)])
    sat = goal["plan"]["steps"][0]["criteria"][0]
    first = await crit.measure(store, cid, goal, sat, current_actions=[{"action": "lattice_write", "lattice": "games",
                                                                        "inserted": 0}])
    assert first["passed"] is False and "Only 1 check-in" in first["gap"]
    run = await store.open_goal_run(cid, payload={"goal_id": goal["id"], "phase": "run"})
    await store.mark_cognition_task_complete(run, result={"actions": [{"action": "lattice_import", "lattice": "games",
                                                                        "inserted": 0}]},
                                             completed_at=datetime.now(timezone.utc))
    second = await crit.measure(store, cid, goal, sat, current_actions=[{"action": "lattice_write", "lattice": "games",
                                                                         "inserted": 0}])
    assert second["passed"] is True
    third = await crit.measure(store, cid, goal, sat, current_actions=[{"action": "lattice_write", "lattice": "games",
                                                                        "inserted": 4}])
    assert third["passed"] is False and "still found 4 new rows" in third["gap"]


@pytest.mark.asyncio
async def test_sources_count_the_pages_behind_a_column(store, customer):
    cid = customer.id
    goal = await _goal(store, customer, [_c("sources", lattice="games", column="home_points", min=2)])
    await _write(store, cid, goal, "games", [{"game_id": "g1", "date": "2024-10-22", "home_points": 1}], source="https://a.test")
    c = goal["plan"]["steps"][0]["criteria"][0]
    assert (await crit.measure(store, cid, goal, c))["passed"] is False
    await _write(store, cid, goal, "games", [{"game_id": "g2", "date": "2024-10-22", "home_points": 2}], source="https://b.test")
    assert (await crit.measure(store, cid, goal, c))["value"] == 2


# --- comments --------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_comment_steers_the_next_check_in_once(store, customer):
    cid = customer.id
    goal = await _goal(store, customer, SEASON_DONE)
    with pytest.raises(service.GoalRefused):
        await service.comment(store, cid, goal["id"], " ")
    c = await service.comment(store, cid, goal["id"], "Skip the preseason games; start with November.")
    anth = FakeAnthropic()
    anth.script_text("Comments: skipping preseason, starting with November.\nDid: nothing yet.\nWaiting: nothing.\nNext: November.")
    out = await run_goal_session(store=store, customer=customer, goal=goal, llm=anth)
    system = anth.calls[0]["system"][0]["text"]
    assert "THE PERSON'S COMMENTS SINCE THE LAST CHECK-IN" in system and "Skip the preseason games" in system
    assert "Comments: what you did about each" in system
    seen = (await store.list_goal_comments(cid, goal["id"]))[0]
    assert seen["id"] == c["id"] and seen["seen_by_run_id"] == out["run_id"]
    runs = await service.runs_view(store, cid, goal["id"])
    assert runs[0]["comments"] == [{"id": c["id"], "body": "Skip the preseason games; start with November."}]
    anth2 = FakeAnthropic()
    anth2.script_text("Did: nothing.\nWaiting: nothing.\nNext: more.")
    await run_goal_session(store=store, customer=customer, goal=await store.get_goal(cid, goal["id"]), llm=anth2)
    assert "Skip the preseason" not in anth2.calls[0]["system"][0]["text"]


# --- rollback --------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rollback_returns_to_a_step_undoing_later_acts_at_any_age(store, customer):
    cid = customer.id
    goal = await _goal(store, customer, [_c("rows", lattice="games", min=1)])
    gid = goal["id"]
    await _write(store, cid, goal, "games", [{"game_id": "g1", "date": "2024-10-22", "home_points": 1}])
    anth = FakeAnthropic()
    anth.script_tool_use("submit_step", {"n": 1}, "tu_01")
    anth.script_text("Did: step 1.\nWaiting: nothing.\nNext: step 2.")
    await run_goal_session(store=store, customer=customer, goal=goal, llm=anth)
    assert _result(anth, 0)["verdict"] == "accepted"
    goal = await store.get_goal(cid, gid)
    await _write(store, cid, goal, "games", [{"game_id": "g2", "date": "2024-10-23", "home_points": 2}])
    await _write(store, cid, goal, "player_games", [{"game_id": "g2", "player_id": "p1"}])
    draft = await store.add_goal_item(cid, gid, kind="draft", title="Weekly recap", body="x")
    await store.add_goal_item(cid, gid, kind="draft", title="Sent recap", body="y", status="sent")
    # The goal's undo records last its whole life: age everything but the g2
    # write by days, past the 24-hour window, keeping the order.
    now = datetime.now(timezone.utc)
    snaps = await store.list_goal_snapshots(cid, gid)
    acts = {a["label"]: a["id"] for a in await store.list_steward_actions(cid, limit=10)}
    g1 = next(v for k, v in acts.items() if "games" in k and "player" not in k and "1 new" in k)
    pg = next(v for k, v in acts.items() if "player_games" in k)
    async with store.session() as s:
        await s.execute(update(GoalSnapshotRow).where(GoalSnapshotRow.id == snaps[0]["id"]).values(created_at=now - timedelta(days=6)))
        await s.execute(update(StewardActionRow).where(StewardActionRow.id == g1).values(created_at=now - timedelta(days=5)))
        await s.execute(update(GoalSnapshotRow).where(GoalSnapshotRow.id == snaps[1]["id"]).values(created_at=now - timedelta(days=4)))
        await s.execute(update(StewardActionRow).where(StewardActionRow.id == pg).values(created_at=now - timedelta(days=3)))
        await s.commit()
    old = await store.get_steward_action(pg, cid)
    assert old["undoable"] is False and old["has_inverse"] is True

    with pytest.raises(service.GoalRefused) as e:
        await service.rollback(store, cid, gid, 3)
    assert e.value.code == "no_snapshot"
    out = await service.rollback(store, cid, gid, 2, reason="Redo the player rows.")
    assert out["undone"] == 2 and out["dismissed_items"] == 1 and out["sent_items"] == ["Sent recap"]
    assert await store.count_lattice_rows(cid) == 1  # g1 stays: it was in before step 1 was accepted
    assert (await store.get_goal_item(cid, draft["id"]))["status"] == "dismissed"
    g = out["goal"]
    assert g["plan"]["steps"][0]["done"] is True and g["plan"]["steps"][1]["done"] is False
    assert [s["step_n"] for s in await store.list_goal_snapshots(cid, gid)] == [0, 1]

    out = await service.rollback(store, cid, gid, 1)
    assert out["undone"] == 1 and await store.count_lattice_rows(cid) == 0
    assert out["goal"]["plan"]["steps"][0]["done"] is False and "last_check" not in out["goal"]["plan"]["steps"][0]
    assert [s["step_n"] for s in await store.list_goal_snapshots(cid, gid)] == [0]
    events = [e["event_type"] for e in await store.list_curation_events(cid)]
    assert events.count("goal_rolled_back") == 2


# --- the survey ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_plan_is_sized_on_a_survey_and_a_replan_reuses_it(store, customer, monkeypatch):
    monkeypatch.setattr(get_settings(), "goal_survey_max_tool_calls", 6)
    g = await service.create(store, customer.id, title="Season",
                             directive="Keep every NBA game of the last three seasons with each player's line.")
    findings = {"findings": [
        {"question": "How many games in a season?", "answer": "1,230 regular-season games.", "number": 1230,
         "source": "https://www.basketball-reference.com/leagues/NBA_2025_games.html", "measured": True},
        {"question": "Players per box score?", "answer": "About 22 to 26 across both teams.", "number": 24,
         "source": "", "measured": True},
    ]}
    anth = FakeAnthropic()
    anth.script_text("Here is what I found:\n" + json.dumps(findings))
    plan = {"summary": "s", "success_measure": "m", "check_every_minutes": 60, "catch_up_every_minutes": 60,
            "steps": [{"title": "Import", "detail": "x", "catch_up": True, "repeats": False, "criteria": []},
                      {"title": "Each check-in: new games", "detail": "y", "catch_up": False, "repeats": True, "criteria": []}],
            "needs_apps": [], "needs_acts": [], "standing_instructions": "", "missing": [], "decisions": [], "lattices": []}
    anth.script_text(json.dumps(plan))
    out = await runner.tick(store, llm=anth)
    assert out["planned"] == 1
    assert "You research the facts a plan is sized and shaped on" in anth.calls[0]["system"][0]["text"]
    user = anth.calls[1]["messages"][0]["content"]
    assert "The survey" in user and "1,230 regular-season games. (measured, https://www.basketball-reference.com" in user
    assert "About 22 to 26 across both teams. (estimated)" in user  # no page behind it: not measured
    after = await store.get_goal(customer.id, g["id"])
    assert [f["number"] for f in after["plan"]["survey"]["findings"]] == [1230, 24]
    assert [s["step_n"] for s in await store.list_goal_snapshots(customer.id, g["id"])] == [0]
    assert "criteria" in planner.PLAN_SCHEMA["properties"]["steps"]["items"]["required"]
    assert "Targets come from the survey's measured numbers" in anth.calls[1]["system"]

    await service.replan(store, customer.id, g["id"])
    anth.script_text(json.dumps(plan))
    await runner.tick(store, llm=anth)
    assert len(anth.calls) == 3 and "The survey" in anth.calls[2]["messages"][0]["content"]


def test_a_listing_page_shows_its_sibling_pages():
    from crystal_cache.search.tables import sibling_pages
    base = "https://www.basketball-reference.com/leagues/NBA_2025_games.html"
    html = ('<a href="/leagues/NBA_2025_games-october.html">October</a>'
            '<a href="/leagues/NBA_2025_games-november.html">November</a>'
            '<a href="/leagues/NBA_2025_games.html">this page</a>'
            '<a href="/players/j/jamesle01.html">LeBron</a>'
            '<a href="https://other.test/leagues/NBA_2025_games-december.html">elsewhere</a>'
            '<a href="/search?season=2025&page=2">p2</a>')
    assert sibling_pages(html, base) == [
        "https://www.basketball-reference.com/leagues/NBA_2025_games-october.html",
        "https://www.basketball-reference.com/leagues/NBA_2025_games-november.html"]
    assert sibling_pages('<a href="?season=2025&page=3">3</a><a href="?season=2024&page=4">x</a>',
                         "https://s.test/list?season=2025&page=2") == ["https://s.test/list?season=2025&page=3"]


@pytest.mark.asyncio
async def test_a_check_in_starts_from_the_gaps_and_is_sent_back_while_the_step_is_short(store, customer):
    # 2026-10-27: a check-in imported one month page with 4 of its 40 calls
    # and stopped ("batches of ~50 games per check-in").
    cid = customer.id
    goal = await _goal(store, customer, [_c("rows", lattice="games", min=3)])
    await _write(store, cid, goal, "games", [{"game_id": "g1", "date": "2024-10-22"}])
    anth = FakeAnthropic()
    anth.script_tool_use("lattice_write", {"lattice": "games", "source": "https://x.test/oct", "reason": "October games.",
                                           "rows": [{"game_id": "g2", "date": "2024-10-23"}]}, "tu_01")
    anth.script_text("**Did:** wrote one game.\n## Next\nNovember.")
    anth.script_tool_use("lattice_write", {"lattice": "games", "source": "https://x.test/nov", "reason": "November games.",
                                           "rows": [{"game_id": "g3", "date": "2024-11-01"}]}, "tu_02")
    anth.script_text("Did: wrote October and November.\nWaiting: nothing.\nNext: submit step 1.")
    out = await run_goal_session(store=store, customer=customer, goal=goal, llm=anth)
    system = anth.calls[0]["system"][0]["text"]
    assert "WHERE STEP 1 (Import the season) STANDS" in system and "games: at least 3 rows: 1 rows" in system
    assert "every month of a season in one check-in" in system
    nudge = anth.calls[2]["messages"][-1]["content"]
    nudge = nudge if isinstance(nudge, str) else " ".join(b.get("text", "") for b in nudge)
    assert "Step 1 is still short" in nudge and "games: at least 3 rows: 2 rows" in nudge
    assert out["ok"] and out["calls"] == 2 and out["actions"] == 2
    assert await store.count_lattice_rows(cid, goal["plan"]["lattices"][0]["lattice_id"]) == 3
    runs = await service.runs_view(store, cid, goal["id"])
    assert runs[0]["calls"] == 2 and runs[0]["rows"]["inserted"] == 2
    # The step passes now: no third pass. And the summary is stored plain.
    assert len(anth.calls) == 4
    assert (await store.get_goal(cid, goal["id"]))["last_summary"].startswith("Did: wrote October and November.")


def test_summaries_are_stored_plain():
    from crystal_cache.goals.session import plain
    assert plain("**Did:**  \n- Ran `goal_plan`.\n\n\n## Next\nMore.") == "Did:\n- Ran goal_plan.\n\nNext\nMore."
