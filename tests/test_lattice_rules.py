"""Lattice rules and two-sided criteria (Series 384, Q152, 2026-10-28).

2026-10-27 an import stamped season "2022-23" on 93 games from October
2021; nothing objected, and a row count accepted the step with them in it
while every season's playoffs were missing. Anthony: the fix "has to be
general and not absolutely mandatory so we dont block a use case that
doesnt need the rules". A lattice may carry rules (SQL conditions every
row must meet); completion criteria can set a ceiling and check each
group against the values that must be there.
"""
from __future__ import annotations

import pytest

from crystal_cache.goals import criteria as crit
from crystal_cache.goals.session import ran_out_summary
from crystal_cache.infrastructure import lattice_tables as lt
from crystal_cache.lattices import rules as rl
from crystal_cache.lattices import service as lat

COLS = [{"key": "game_id"}, {"key": "date", "type": "date"}, {"key": "season"}, {"key": "home"}, {"key": "away"},
        {"key": "pts", "type": "number"}]


def _season_rule(store):
    # The rule that would have refused October 2021 as "2022-23", per dialect.
    if lt.is_pg(store._lattice_engine):
        check = "date >= make_date(cast(left(season, 4) as int), 7, 1) AND date < make_date(cast(left(season, 4) as int) + 1, 7, 1)"
    else:
        check = "date >= substr(season, 1, 4) || '-07-01' AND date < (cast(substr(season, 1, 4) as integer) + 1) || '-07-01'"
    return {"label": "The date falls inside its season", "check": check, "mode": "refuse"}


@pytest.mark.asyncio
async def test_a_lattice_without_rules_takes_any_row(store, customer):
    g = await lat.create(store, customer.id, name="games", key_columns=["game_id"], columns=COLS)
    out = await lat.write_rows(store, customer.id, g["id"], [{"game_id": "a", "date": "2021-10-19", "season": "2022-23",
                                                              "home": "BOS", "away": "BOS"}])
    assert out["inserted"] == 1 and "rules" not in out


@pytest.mark.asyncio
async def test_refuse_rules_stop_rows_warn_rules_count_and_null_passes(store, customer):
    cid = customer.id
    g = await lat.create(store, cid, name="games", key_columns=["game_id"], columns=COLS, rules=[
        _season_rule(store),
        {"label": "Two different teams", "check": "home <> away", "mode": "refuse"},
        {"label": "Points look like basketball", "check": "pts BETWEEN 40 AND 200", "mode": "warn"}])
    assert [r["mode"] for r in lat.describe(g)["rules"]] == ["refuse", "refuse", "warn"]
    out = await lat.write_rows(store, cid, g["id"], [
        {"game_id": "ok", "date": "2022-10-18", "season": "2022-23", "home": "BOS", "away": "PHI", "pts": 126},
        {"game_id": "old", "date": "2021-10-19", "season": "2022-23", "home": "MIL", "away": "BRK", "pts": 127},
        {"game_id": "self", "date": "2022-11-01", "season": "2022-23", "home": "BOS", "away": "BOS"},
        {"game_id": "odd", "date": "2022-12-01", "season": "2022-23", "home": "LAL", "away": "DEN", "pts": 9},
        # A missing value can't break a rule (NULL passes, as a CHECK does).
        {"game_id": "blank", "season": "2022-23", "home": "NYK", "away": "MIA"}])
    assert out["inserted"] == 3 and out["error_count"] == 2
    assert out["rules"]["refused"] == {"The date falls inside its season": 1, "Two different teams": 1}
    assert out["rules"]["warned"] == {"Points look like basketball": 1}
    assert {e["row_key"]: e["rule"] for e in out["errors"]} == {"old": "The date falls inside its season",
                                                                "self": "Two different teams"}
    # An update is checked as the row would stand after it.
    again = await lat.write_rows(store, cid, g["id"], [{"game_id": "ok", "away": "BOS"}])
    assert again["updated"] == 0 and again["errors"][0]["rule"] == "Two different teams"


@pytest.mark.asyncio
async def test_rules_read_other_lattices_and_a_rule_that_cannot_run_never_blocks(store, customer, monkeypatch):
    cid = customer.id
    teams = await lat.create(store, cid, name="teams", key_columns=["team_id"], columns=[{"key": "team_id"}])
    await lat.write_rows(store, cid, teams["id"], [{"team_id": "BOS"}, {"team_id": "PHI"}])
    g = await lat.create(store, cid, name="games", key_columns=["game_id"], columns=COLS, rules=[
        {"label": "Home team is a known team", "check": "home IN (SELECT team_id FROM teams)", "mode": "refuse"},
        {"label": "Nonsense", "check": "no_such_column > 1", "mode": "refuse"}])
    # The rule that doesn't run was left out when the lattice was made, and said.
    assert [r["label"] for r in g["rules"]] == ["Home team is a known team"] and "Nonsense" in g["rules_dropped"][0]
    out = await lat.write_rows(store, cid, g["id"], [{"game_id": "1", "home": "BOS", "away": "PHI"},
                                                     {"game_id": "2", "home": "XXX", "away": "PHI"}])
    assert out["inserted"] == 1 and out["errors"][0]["rule"] == "Home team is a known team"
    # A rule that fails at write time (the database is down, a lattice it reads is gone) lets the rows through.
    real = store.lattice_rule_query

    async def broken(customer_id, query):
        raise lt.SqlRefused("sql_error", "relation does not exist")
    monkeypatch.setattr(store, "lattice_rule_query", broken)
    out = await lat.write_rows(store, cid, g["id"], [{"game_id": "3", "home": "YYY", "away": "PHI"}])
    assert out["inserted"] == 1 and out["rules"]["not_run"] == {"Home team is a known team": "relation does not exist"}
    monkeypatch.setattr(store, "lattice_rule_query", real)
    # Setting a rule that doesn't run is refused; so is one that tries to write.
    for check, code in (("no_such_column > 1", "bad_rule"), ("1 = 1; DELETE FROM teams", "bad_rule"),
                        ("x", "rule_check_required")):
        with pytest.raises(lat.LatticeRefused) as e:
            await lat.update(store, cid, g["id"], add_rules=[{"label": "x", "check": check}])
        assert e.value.code == code, check


@pytest.mark.asyncio
async def test_the_console_sees_rows_already_breaking_a_rule_and_only_the_person_turns_one_off(store, customer):
    cid = customer.id
    g = await lat.create(store, cid, name="games", key_columns=["game_id"], columns=COLS)
    await lat.write_rows(store, cid, g["id"], [
        {"game_id": "a", "date": "2021-10-19", "season": "2022-23", "home": "MIL", "away": "BRK"},
        {"game_id": "b", "date": "2022-10-18", "season": "2022-23", "home": "BOS", "away": "PHI"}])
    # An agent may add a rule; existing rows stay, and the console shows which break it.
    g = await lat.update(store, cid, g["id"], add_rules=[_season_rule(store)])
    view = await lat.rules_view(store, cid, g["id"])
    assert view["rules"][0]["breaking"] == {"rows": 1, "examples": ["a"]}
    # The person turns it off (or edits it) by replacing the list.
    rid = view["rules"][0]["id"]
    g = await lat.update(store, cid, g["id"], rules=[{**view["rules"][0], "id": rid, "on": False}])
    out = await lat.write_rows(store, cid, g["id"], [{"game_id": "c", "date": "2020-01-01", "season": "2022-23",
                                                      "home": "X", "away": "Y"}])
    assert out["inserted"] == 1 and "rules" not in out


@pytest.mark.asyncio
async def test_an_import_that_mostly_breaks_a_rule_writes_nothing_and_shows_what_it_read(store, customer):
    from crystal_cache.search import tables as wt
    cid = customer.id
    g = await lat.create(store, cid, name="games", key_columns=["game_id"], columns=COLS, rules=[_season_rule(store)])
    rows = "".join(f"<tr><td>g{i}</td><td>Oct {19 + i}, 2021</td><td>BOS</td><td>PHI</td></tr>" for i in range(8))
    page = f"<table id='s'><tr><th>Game</th><th>Date</th><th>Home</th><th>Away</th></tr>{rows}</table>"

    async def loader(url):
        return url, wt.absolute_links(wt.parse_tables(page), url)
    cols = {"game_id": "Game", "date": "Date", "home": "Home", "away": "Away"}
    with pytest.raises(lat.LatticeRefused) as e:
        await lat.import_table(store, cid, g["id"], url="https://x.test/2022-october", table="s", columns=cols,
                               constants={"season": "2022-23"}, loader=loader)
    assert e.value.code == "import_breaks_rule" and e.value.detail["breaking"] == 8
    assert e.value.detail["spans"]["date"] == ["2021-10-19", "2021-10-26"] and e.value.detail["constants"] == {"season": "2022-23"}
    assert await store.count_lattice_rows(cid, g["id"]) == 0
    # The right constant goes through, and the result shows the span beside it.
    out = await lat.import_table(store, cid, g["id"], url="https://x.test/2022-october", table="s", columns=cols,
                                 constants={"season": "2021-22"}, loader=loader)
    assert out["inserted"] == 8 and out["spans"]["date"] == ["2021-10-19", "2021-10-26"]


@pytest.mark.asyncio
async def test_groups_and_ceilings_fail_on_extra_rows_and_name_what_is_missing(store, customer):
    cid = customer.id
    g = await lat.create(store, cid, name="games", key_columns=["game_id"], columns=COLS)
    rows = ([{"game_id": f"a{i}", "season": "2022-23"} for i in range(5)] + [{"game_id": f"b{i}", "season": "2023-24"} for i in range(3)]
            + [{"game_id": f"c{i}", "season": "2021-22"} for i in range(2)])
    await lat.write_rows(store, cid, g["id"], rows)
    goal = {"id": "goal_x", "plan": {}}
    clean, dropped = crit.normalize_criteria([
        {"kind": "groups", "label": "", "lattice": "games", "column": "season", "min": 3, "max": 4,
         "values": ["2022-23", "2023-24", "2024-25"], "exclusive": True},
        {"kind": "rows", "label": "", "lattice": "games", "min": 3, "max": 8},
        {"kind": "groups", "label": "", "lattice": "games", "column": "season"},
        {"kind": "rows", "label": "", "lattice": "games", "min": 9, "max": 3}])
    assert len(clean) == 2 and len(dropped) == 2
    assert clean[0]["label"] == "games: each season (2022-23, 2023-24, 2024-25 only) has 3 to 4 rows"
    g1 = await crit.measure(store, cid, goal, clean[0])
    assert g1["passed"] is False and g1["measured"] == "1 of 4 season in range"
    assert "missing 2024-25" in g1["gap"].lower() and "don't belong: 2021-22" in g1["gap"] and "over 4: 2022-23 (5)" in g1["gap"]
    assert set(g1["missing_sample"]) >= {"2024-25", "2021-22", "2022-23"}
    r = await crit.measure(store, cid, goal, clean[1])
    assert r["passed"] is False and "2 rows over the most expected (8)" in r["gap"]


def test_a_summary_that_ran_out_says_what_was_submitted():
    text = ran_out_summary([{"action": "lattice_import", "lattice": "games", "inserted": 371, "updated": 0}],
                           {"n": 2, "title": "NFL", "short": [{"label": "NFL games", "measured": "0 rows"}]}, 40,
                           "max_iterations", [{"n": 1, "verdict": "accepted"}])
    assert "Did: games: 371 rows added.\nSubmitted: step 1 accepted.\nStep 2 is still short" in text


def test_rule_text_is_one_condition():
    assert rl.check_text(" home <> away; ") == "home <> away"
    for bad in ("1=1; DROP TABLE x", "pg_sleep(10) IS NULL", "a" * 1001):
        with pytest.raises(rl.RuleError):
            rl.check_text(bad)
