"""Lattices (2026-10-25; Q115 A, Q116 A, Q117 A).

Anthony: "Tables carry structured data with child crystals that are
editable. An example of a crystal fact needing edits is if the table
changes with tatums shot %. That isnt a static fact." And on typing: "not
everyone would need a recompute." A lattice is a named table in Crystal
storage: snapshot cells are true once and locked (only the person
corrects them), live cells take new values and keep their history, and
live columns with a refresh window report when they are stale. Rows have
their own cap (Q117 A), apart from the fact cap.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from crystal_cache.control import admission
from crystal_cache.lattices import service as lat

GAMES = [
    {"key": "date", "type": "date"},
    {"key": "league", "type": "text", "entity": True},
    {"key": "home", "type": "text", "entity": True},
    {"key": "away", "type": "text", "entity": True},
    {"key": "home_score", "type": "number"},
    {"key": "away_score", "type": "number"},
]
PLAYERS = [
    {"key": "player", "type": "text", "entity": True},
    {"key": "team", "type": "text", "entity": True},
    {"key": "fg_pct", "label": "FG %", "type": "number", "kind": "live", "refresh_minutes": 1440},
]


async def _games(store, customer):
    return await lat.create(store, customer.id, name="NBA and NFL games 2026-27", columns=GAMES,
                            key_columns=["date", "home", "away"], description="Every game, both leagues.")


@pytest.mark.asyncio
async def test_create_validates_the_shape_and_names_are_unique(store, customer):
    g = await _games(store, customer)
    assert g["id"].startswith("lat_") and g["key_columns"] == ["date", "home", "away"]
    assert [c["kind"] for c in g["columns"]] == ["snapshot"] * 6
    assert g["columns"][4]["label"] == "Home score"
    with pytest.raises(lat.LatticeRefused) as e:
        await _games(store, customer)
    assert e.value.code == "name_taken" and e.value.status == 409
    for cols, keys, code in (
        ([], ["a"], "columns_required"),
        ([{"key": "Bad Key"}], ["bad key"], "bad_column_key"),
        ([{"key": "a", "type": "money"}], ["a"], "bad_column_type"),
        ([{"key": "a", "kind": "sometimes"}], ["a"], "bad_cell_kind"),
        ([{"key": "a", "type": "number", "entity": True}], ["a"], "bad_entity_column"),
        ([{"key": "a", "refresh_minutes": 60}], ["a"], "refresh_on_snapshot"),
        ([{"key": "a"}], [], "key_columns_required"),
        ([{"key": "a"}], ["b"], "unknown_key_column"),
        ([{"key": "a", "kind": "live"}], ["a"], "live_key_column"),
    ):
        with pytest.raises(lat.LatticeRefused) as e:
            await lat.create(store, customer.id, name=f"x {code}", columns=cols, key_columns=keys)
        assert e.value.code == code, code


@pytest.mark.asyncio
async def test_rows_upsert_by_key_and_snapshot_cells_lock(store, customer):
    g = await _games(store, customer)
    out = await lat.write_rows(store, customer.id, g["name"], [
        {"date": "2026-10-21", "league": "NBA", "home": "Celtics", "away": "Knicks"},
        {"date": "2026-10-22", "league": "NBA", "home": "Lakers", "away": "Warriors", "home_score": "118", "away_score": 112},
        {"date": "nope", "league": "NBA", "home": "Heat", "away": "Magic"},
        {"league": "NFL", "home": "Bills"},
    ], source="nba.com", actor="goal:g1")
    assert (out["inserted"], out["updated"], out["error_count"]) == (2, 0, 2)
    assert [e["code"] for e in out["errors"]] == ["bad_value", "key_missing"]
    # The score arrives later: an empty snapshot cell fills; a filled one is locked.
    out = await lat.write_rows(store, customer.id, g["id"], [
        {"date": "2026-10-21", "home": "CELTICS ", "away": "knicks", "home_score": 109, "away_score": 101},
        {"date": "2026-10-22", "home": "Lakers", "away": "Warriors", "home_score": 120},
    ], source="espn.com", actor="goal:g1")
    assert (out["inserted"], out["updated"], out["unchanged"]) == (0, 1, 1)
    assert out["locked"] == [{"row_key": "2026-10-22 | lakers | warriors", "column": "home_score", "kept": 118, "offered": 120}]
    # A person's correction changes it and keeps the old value in history.
    out = await lat.write_rows(store, customer.id, g["id"], [
        {"date": "2026-10-22", "home": "Lakers", "away": "Warriors", "home_score": 120}], actor="console", correct=True)
    assert out["updated"] == 1
    q = await lat.query(store, customer.id, g["id"], filters={"home": "lakers"}, history=True)
    row = q["rows"][0]
    assert row["values"]["home_score"] == 120
    assert row["cells"]["home_score"]["history"][0]["value"] == 118
    assert row["cells"]["home_score"]["corrected_by"] == "console"


@pytest.mark.asyncio
async def test_live_cells_take_new_values_keep_history_and_go_stale(store, customer):
    p = await lat.create(store, customer.id, name="NBA players", columns=PLAYERS, key_columns=["player"])
    await lat.write_rows(store, customer.id, p["id"], [{"player": "Jayson Tatum", "team": "Celtics", "fg_pct": "45.1%"}],
                         source="nba.com")
    out = await lat.write_rows(store, customer.id, p["id"], [{"player": "Jayson Tatum", "fg_pct": 46.3}], source="nba.com")
    assert out["updated"] == 1 and out["locked"] == []
    row = (await lat.query(store, customer.id, p["id"], history=True))["rows"][0]
    assert row["values"]["fg_pct"] == 46.3
    assert [h["value"] for h in row["cells"]["fg_pct"]["history"]] == [45.1]
    assert "stale" not in row
    assert (await lat.stale(store, customer.id, p["id"]))["stale"] == []
    # A day and a half later the live cell is past its refresh window.
    rid = row["row_id"]
    data = (await store.get_lattice_rows(customer.id, p["id"], [rid]))[0]
    old = (datetime.now(timezone.utc) - timedelta(hours=36)).isoformat()
    cells = {**data["cells"], "fg_pct": {**data["cells"]["fg_pct"], "as_of": old}}
    await store.write_lattice_rows(customer.id, p["id"], inserts=[], updates=[{"id": rid, "data": data["data"], "cells": cells}])
    st = await lat.stale(store, customer.id, p["id"])
    assert st["stale"] == [{"row_id": rid, "key": {"player": "Jayson Tatum"}, "stale": {"fg_pct": {"value": 46.3, "as_of": old}}}]
    # History keeps five.
    for v in (40, 41, 42, 43, 44, 45):
        await lat.write_rows(store, customer.id, p["id"], [{"player": "Jayson Tatum", "fg_pct": v}])
    row = (await lat.query(store, customer.id, p["id"], history=True))["rows"][0]
    assert [h["value"] for h in row["cells"]["fg_pct"]["history"]] == [44, 43, 42, 41, 40]


@pytest.mark.asyncio
async def test_query_filters_search_sort_and_browse_by_entity(store, customer):
    g = await _games(store, customer)
    rows = []
    for i, (home, away, hs) in enumerate((("Celtics", "Knicks", 109), ("Lakers", "Celtics", 99),
                                          ("Celtics", "Heat", 120), ("Bills", "Jets", 27))):
        rows.append({"date": f"2026-10-2{i}", "league": "NFL" if home == "Bills" else "NBA", "home": home,
                     "away": away, "home_score": hs})
    await lat.write_rows(store, customer.id, g["id"], rows)
    q = await lat.query(store, customer.id, g["id"], filters=[{"column": "league", "op": "eq", "value": "nba"}],
                        sort="home_score", descending=True)
    assert q["total"] == 3 and [r["values"]["home_score"] for r in q["rows"]] == [120, 109, 99]
    q = await lat.query(store, customer.id, g["id"], search="celtics")
    assert q["total"] == 3
    q = await lat.query(store, customer.id, g["id"], filters=[{"column": "home_score", "op": "gte", "value": 100},
                                                              {"column": "date", "op": "lte", "value": "2026-10-21"}])
    assert [r["values"]["home"] for r in q["rows"]] == ["Celtics"]
    q = await lat.query(store, customer.id, g["id"], limit=2, offset=2)
    assert q["total"] == 4 and len(q["rows"]) == 2
    ent = await lat.entities(store, customer.id, g["id"])
    by = {e["column"]: e["values"] for e in ent["entities"]}
    assert set(by) == {"league", "home", "away"}
    assert by["home"][0] == {"value": "Celtics", "rows": 2}
    assert by["league"] == [{"value": "NBA", "rows": 3}, {"value": "NFL", "rows": 1}]
    with pytest.raises(lat.LatticeRefused) as e:
        await lat.query(store, customer.id, g["id"], filters={"nope": 1})
    assert e.value.code == "unknown_column"
    with pytest.raises(lat.LatticeRefused) as e:
        await lat.query(store, customer.id, g["id"], filters=[{"column": "home", "op": "gte", "value": "a"}])
    assert e.value.code == "bad_filter"


@pytest.mark.asyncio
async def test_unknown_columns_refuse_the_batch_and_columns_can_be_added(store, customer):
    g = await _games(store, customer)
    with pytest.raises(lat.LatticeRefused) as e:
        await lat.write_rows(store, customer.id, g["id"], [{"date": "2026-10-21", "home": "A", "away": "B", "venue": "TD Garden"}])
    assert e.value.code == "unknown_columns" and e.value.detail["unknown"] == ["venue"]
    g = await lat.update(store, customer.id, g["id"], add_columns=[{"key": "venue", "entity": True}],
                         column_changes={"home_score": {"label": "Home pts"}})
    assert g["columns"][-1]["key"] == "venue" and g["columns"][4]["label"] == "Home pts"
    out = await lat.write_rows(store, customer.id, g["id"], [{"date": "2026-10-21", "home": "A", "away": "B", "venue": "TD Garden"}])
    assert out["inserted"] == 1
    with pytest.raises(lat.LatticeRefused) as e:
        await lat.update(store, customer.id, g["id"], column_changes={"home_score": {"type": "text"}})
    assert e.value.code == "bad_column"
    with pytest.raises(lat.LatticeRefused) as e:
        await lat.update(store, customer.id, g["id"], column_changes={"date": {"kind": "live"}})
    assert e.value.code == "live_key_column"


@pytest.mark.asyncio
async def test_the_row_cap_is_its_own_and_counts_only_new_rows(store, customer, monkeypatch):
    monkeypatch.setitem(admission.TIER_TABLE, "free",
                        dataclasses.replace(admission.TIER_TABLE["free"], lattice_row_cap=3))
    free = SimpleNamespace(id=customer.id, subscription_tier="free")
    g = await _games(store, customer)
    rows = [{"date": f"2026-10-2{i}", "home": "A", "away": "B"} for i in range(3)]
    await lat.write_rows(store, customer.id, g["id"], rows, customer=free)
    # cap 3 x grace 1.1 = 3 rows: a fourth new row is refused...
    with pytest.raises(lat.LatticeRefused) as e:
        await lat.write_rows(store, customer.id, g["id"], [{"date": "2026-10-29", "home": "A", "away": "B"}], customer=free)
    assert e.value.code == "lattice_full" and e.value.status == 402 and e.value.detail == {"rows": 3, "cap": 3}
    # ...updates to existing rows still go through, and the fact cap never moved.
    out = await lat.write_rows(store, customer.id, g["id"], [{**rows[0], "home_score": 100}], customer=free)
    assert out["updated"] == 1
    assert await store.count_billable_facts(customer.id) == 0
    # Uncapped tiers and self-host never cap.
    scale = SimpleNamespace(id=customer.id, subscription_tier="scale_49_seat")
    out = await lat.write_rows(store, customer.id, g["id"], [{"date": "2026-10-29", "home": "A", "away": "B"}], customer=scale)
    assert out["inserted"] == 1
    assert admission.TIER_TABLE["starter_29"].lattice_row_cap == 250_000


@pytest.mark.asyncio
async def test_undo_deletes_new_rows_and_restores_changed_ones(store, customer):
    p = await lat.create(store, customer.id, name="NBA players", columns=PLAYERS, key_columns=["player"])
    await lat.write_rows(store, customer.id, p["id"], [{"player": "Jayson Tatum", "fg_pct": 45.1}])
    out = await lat.write_rows(store, customer.id, p["id"], [{"player": "Jayson Tatum", "fg_pct": 47.0},
                                                             {"player": "Jaylen Brown", "fg_pct": 49.0}])
    assert (out["inserted"], out["updated"]) == (1, 1)
    await lat.undo_write(store, customer.id, out["inverse"])
    q = await lat.query(store, customer.id, p["id"])
    assert [(r["values"]["player"], r["values"]["fg_pct"]) for r in q["rows"]] == [("Jayson Tatum", 45.1)]


@pytest.mark.asyncio
async def test_set_cell_is_a_correction_and_key_cells_stay(store, customer):
    g = await _games(store, customer)
    await lat.write_rows(store, customer.id, g["id"], [{"date": "2026-10-21", "home": "A", "away": "B", "home_score": 1}])
    rid = (await lat.query(store, customer.id, g["id"]))["rows"][0]["row_id"]
    out = await lat.set_cell(store, customer.id, g["id"], rid, "home_score", "101")
    assert out["updated"] == 1
    assert (await lat.query(store, customer.id, g["id"]))["rows"][0]["values"]["home_score"] == 101
    with pytest.raises(lat.LatticeRefused) as e:
        await lat.set_cell(store, customer.id, g["id"], rid, "home", "C")
    assert e.value.code == "key_column_locked"
    await lat.set_cell(store, customer.id, g["id"], rid, "home_score", "")
    assert (await lat.query(store, customer.id, g["id"]))["rows"][0]["values"]["home_score"] is None


@pytest.mark.asyncio
async def test_tenants_never_see_each_others_lattices_and_erase_takes_rows(store, customer):
    other = await store.create_customer(provider="anthropic", model_id="claude-sonnet-4-5-20250929", api_key_ref="sk-2")
    g = await _games(store, customer)
    await lat.write_rows(store, customer.id, g["id"], [{"date": "2026-10-21", "home": "A", "away": "B"}])
    with pytest.raises(lat.LatticeRefused) as e:
        await lat.query(store, other.id, g["id"])
    assert e.value.code == "not_found"
    assert await lat.list_view(store, other.id) == []
    assert (await lat.list_view(store, customer.id))[0]["rows"] == 1
    await store.erase_rows(customer.id, keeps=__import__("crystal_cache.infrastructure.metadata_store_erase_ext",
                                                         fromlist=["ERASE_KEEPS"]).ERASE_KEEPS)
    assert await lat.list_view(store, customer.id) == []
    assert await store.count_lattice_rows(customer.id) == 0


@pytest.mark.asyncio
async def test_delete_takes_the_rows(store, customer):
    g = await _games(store, customer)
    await lat.write_rows(store, customer.id, g["id"], [{"date": "2026-10-21", "home": "A", "away": "B"},
                                                       {"date": "2026-10-22", "home": "A", "away": "B"}])
    rid = (await lat.query(store, customer.id, g["id"]))["rows"][0]["row_id"]
    assert (await lat.delete_rows(store, customer.id, g["id"], [rid]))["deleted"] == 1
    assert (await lat.delete(store, customer.id, g["name"]))["rows"] == 1
    assert await store.count_lattice_rows(customer.id) == 0


def _console(store):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from crystal_cache.endpoints import lattices as ep
    from crystal_cache.infrastructure.metadata_store import get_metadata_store

    app = FastAPI()
    app.include_router(ep.router)
    app.dependency_overrides[get_metadata_store] = lambda: store
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t")


@pytest.mark.asyncio
async def test_console_routes_create_browse_edit_and_undo(store, customer):
    from crystal_cache.cognition import controls

    base = f"/admin/api/customers/{customer.id}/lattices"
    async with _console(store) as c:
        r = await c.post(base, json={"name": "NBA players", "columns": PLAYERS, "key_columns": ["player"]})
        assert r.status_code == 201, r.text
        lid = r.json()["lattice"]["lattice_id"]
        r = await c.post(base, json={"name": "NBA players", "columns": PLAYERS, "key_columns": ["player"]})
        assert r.status_code == 409 and r.json()["detail"]["error"]["code"] == "name_taken"
        r = await c.post(base, json={"name": "x", "columns": PLAYERS, "key_columns": ["player"], "extra": 1})
        assert r.status_code == 400
        r = await c.post(f"{base}/{lid}/rows", json={"rows": [
            {"player": "Jayson Tatum", "team": "Celtics", "fg_pct": 45.1},
            {"player": "Jaylen Brown", "team": "Celtics", "fg_pct": 49.0},
            {"player": "LeBron James", "team": "Lakers", "fg_pct": 52.0}]})
        assert r.status_code == 200 and r.json()["inserted"] == 3 and r.json()["action_id"]
        assert "inverse" not in r.json()
        r = await c.get(base)
        assert r.json()["lattices"][0]["rows"] == 3 and r.json()["rows"]["used"] == 3
        r = await c.get(f"{base}/{lid}")
        ents = {e["column"]: e["values"] for e in r.json()["entities"]}
        assert ents["team"][0] == {"value": "Celtics", "rows": 2}
        r = await c.get(f"{base}/{lid}/rows", params={"column": "team", "value": "Celtics", "sort": "fg_pct", "desc": "true"})
        assert [x["values"]["player"] for x in r.json()["rows"]] == ["Jaylen Brown", "Jayson Tatum"]
        rid = r.json()["rows"][1]["row_id"]
        r = await c.patch(f"{base}/{lid}/rows/{rid}", json={"column": "fg_pct", "value": "46.0"})
        assert r.status_code == 200 and r.json()["row"]["values"]["fg_pct"] == 46.0
        assert r.json()["row"]["cells"]["fg_pct"]["history"][0]["value"] == 45.1
        undo = await controls.undo_action(store, customer.id, r.json()["action_id"], actor="console")
        assert undo["undone"] is True
        r = await c.get(f"{base}/{lid}/rows/{rid}")
        assert r.json()["row"]["values"]["fg_pct"] == 45.1
        r = await c.get(f"{base}/{lid}/rows", params={"filters": "not json"})
        assert r.status_code == 400 and r.json()["detail"]["error"]["code"] == "bad_filter"
        r = await c.post(f"{base}/{lid}/rows/delete", json={"row_ids": [rid]})
        assert r.json()["deleted"] == 1
        r = await c.get(f"{base}/{lid}/entities", params={"column": "team"})
        assert r.json()["entities"][0]["values"] == [{"value": "Celtics", "rows": 1}, {"value": "Lakers", "rows": 1}]
        r = await c.delete(f"{base}/{lid}")
        assert r.json() == {"deleted": True, "lattice_id": lid, "rows": 2}
        r = await c.get(f"{base}/{lid}")
        assert r.status_code == 404


@pytest.mark.asyncio
async def test_recall_finds_rows_by_the_entity_a_question_names(store, customer):
    assert await lat.recall_hits(store, customer.id, "What is Jayson Tatum shooting?") == []
    p = await lat.create(store, customer.id, name="NBA players", columns=PLAYERS, key_columns=["player"])
    await lat.write_rows(store, customer.id, p["id"], [{"player": "Jayson Tatum", "team": "Celtics", "fg_pct": 46.3},
                                                       {"player": "Jaylen Brown", "team": "Celtics", "fg_pct": 49.0},
                                                       {"player": "LeBron James", "team": "Lakers", "fg_pct": 52.0}])
    hits = await lat.recall_hits(store, customer.id, "What is Jayson Tatum shooting this season?")
    assert len(hits) == 1 and hits[0]["matched"] == {"player": "Jayson Tatum"}
    assert [r["values"]["fg_pct"] for r in hits[0]["rows"]] == [46.3]
    hits = await lat.recall_hits(store, customer.id, "how are the celtics doing")
    assert hits[0]["matched"] == {"team": "Celtics"} and hits[0]["total"] == 2
    assert await lat.recall_hits(store, customer.id, "what is the weather") == []


# --- Q121 A: analytics (aggregates, CSV export) -------------------------------------------

PG = [
    {"key": "game_id", "type": "text"},
    {"key": "player_id", "type": "text"},
    {"key": "player", "type": "text", "entity": True},
    {"key": "team", "type": "text", "entity": True},
    {"key": "date", "type": "date"},
    {"key": "points", "type": "number"},
    {"key": "minutes", "type": "number"},
]


async def _player_games(store, customer):
    g = await lat.create(store, customer.id, name="NBA player games", columns=PG, key_columns=["game_id", "player_id"])
    rows = [
        {"game_id": "g1", "player_id": "p1", "player": "Jayson Tatum", "team": "BOS", "date": "2026-10-21", "points": 30, "minutes": 36},
        {"game_id": "g1", "player_id": "p2", "player": "Jaylen Brown", "team": "BOS", "date": "2026-10-21", "points": 22, "minutes": 34},
        {"game_id": "g2", "player_id": "p1", "player": "Jayson Tatum", "team": "BOS", "date": "2026-10-23", "points": 24, "minutes": 35},
        {"game_id": "g2", "player_id": "p3", "player": "LeBron James", "team": "LAL", "date": "2026-10-23", "points": 28, "minutes": 33},
        {"game_id": "g3", "player_id": "p3", "player": "LeBron James", "team": "LAL", "date": "2026-10-25", "points": None, "minutes": None},
    ]
    await lat.write_rows(store, customer.id, g["id"], rows)
    return g


@pytest.mark.asyncio
async def test_aggregates_group_count_and_average(store, customer):
    g = await _player_games(store, customer)
    out = await lat.aggregate(store, customer.id, g["id"], group_by=["player"],
                              metrics=[{"fn": "avg", "column": "points"}, "max:minutes", "count"], sort="avg_points")
    assert out["metrics"] == ["count", "avg_points", "max_minutes"]
    assert out["groups"] == [
        {"player": "LeBron James", "count": 2, "avg_points": 28.0, "max_minutes": 33.0},
        {"player": "Jayson Tatum", "count": 2, "avg_points": 27.0, "max_minutes": 36.0},
        {"player": "Jaylen Brown", "count": 1, "avg_points": 22.0, "max_minutes": 34.0},
    ]
    assert out["total_groups"] == 3
    out = await lat.aggregate(store, customer.id, g["id"], group_by="team,date", metrics=["sum:points"],
                              filters={"team": "bos"}, sort="sum_points", descending=False)
    assert [(x["date"], x["sum_points"]) for x in out["groups"]] == [("2026-10-23", 24.0), ("2026-10-21", 52.0)]
    totals = await lat.aggregate(store, customer.id, g["id"], metrics=["sum:points", "avg:minutes"])
    assert totals["groups"] == [{"count": 5, "sum_points": 104.0, "avg_minutes": 34.5}]
    for kw, code in (({"metrics": ["avg:player"]}, "bad_metrics"), ({"metrics": ["median:points"]}, "bad_metrics"),
                     ({"group_by": ["a", "b", "c", "d"]}, "bad_group_by"), ({"group_by": ["nope"]}, "unknown_column"),
                     ({"sort": "avg_nothing"}, "bad_sort")):
        with pytest.raises(lat.LatticeRefused) as e:
            await lat.aggregate(store, customer.id, g["id"], **kw)
        assert e.value.code == code, kw


@pytest.mark.asyncio
async def test_console_aggregate_and_csv_export(store, customer):
    g = await _player_games(store, customer)
    base = f"/admin/api/customers/{customer.id}/lattices/{g['id']}"
    async with _console(store) as c:
        r = await c.get(f"{base}/aggregate", params={"group_by": "team", "metrics": "avg:points,count", "sort": "count"})
        assert r.status_code == 200, r.text
        assert r.json()["groups"][0] == {"team": "BOS", "count": 3, "avg_points": 25.3333}
        r = await c.get(f"{base}/aggregate", params={"metrics": "avg:player"})
        assert r.status_code == 400 and r.json()["detail"]["error"]["code"] == "bad_metrics"
        r = await c.get(f"{base}/export.csv", params={"filters": '[{"column": "team", "op": "eq", "value": "LAL"}]'})
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
        assert r.headers["content-disposition"] == 'attachment; filename="nba-player-games.csv"'
        assert r.text.splitlines() == [
            "game_id,player_id,player,team,date,points,minutes",
            "g2,p3,LeBron James,LAL,2026-10-23,28,33",
            "g3,p3,LeBron James,LAL,2026-10-25,,",
        ]


@pytest.mark.asyncio
async def test_big_lattices_browse_from_indexes_and_export_by_keyset(store, customer, monkeypatch):
    # Q133 A (2026-10-26): a lattice holds 250,000 rows. What browsing filters
    # and searches on is kept beside each row (facets, search_text: GIN
    # indexes on Postgres); the browse lists are cached until the next
    # write; the CSV export walks rows by keyset, each read once.
    assert lat.MAX_ROWS_PER_LATTICE == 250_000
    g = await lat.create(store, customer.id, name="player_games", key_columns=["game_id", "player_id"], columns=[
        {"key": "game_id"}, {"key": "player_id", "entity": True}, {"key": "player_name", "entity": True},
        {"key": "date", "type": "date"}, {"key": "pts", "type": "number"}])
    rows = [{"game_id": f"g{i // 3:04d}", "player_id": f"p{i % 3}", "player_name": f"Player {i % 3}",
             "date": "2024-10-22", "pts": i % 40} for i in range(1203)]
    for i in range(0, len(rows), 500):
        await lat.write_rows(store, customer.id, g["id"], rows[i:i + 500], source="test")
    stored = await store.query_lattice_rows(customer.id, g["id"], limit=1)
    from crystal_cache.infrastructure.schema import LatticeRecordRow
    from sqlalchemy import select
    async with store.session() as s:
        r = (await s.execute(select(LatticeRecordRow).where(LatticeRecordRow.id == stored[0][0]["id"]))).scalar_one()
    assert r.facets["player_name"] == "player 0" and r.facets["pts"] == 0 and "player 0" in r.search_text
    q = await lat.query(store, customer.id, g["id"], filters={"player_name": "PLAYER 1"})
    assert q["total"] == 401
    first = await lat.entities(store, customer.id, g["id"], "player_name")
    assert first["entities"][0]["values"][0]["rows"] == 401
    await lat.write_rows(store, customer.id, g["id"], [{"game_id": "g9999", "player_id": "p1", "player_name": "Player 1",
                                                        "date": "2024-10-23", "pts": 1}], source="test")
    after = await lat.entities(store, customer.id, g["id"], "player_name")
    assert after["entities"][0]["values"][0] == {"value": "Player 1", "rows": 402}  # the cache moved with the write
    seen = []
    async for batch in store.iter_lattice_rows(customer.id, g["id"], batch=250):
        seen.extend(x["row_key"] for x in batch)
    assert len(seen) == 1204 and len(set(seen)) == 1204
    _l, it = await lat.export_rows(store, customer.id, g["id"], filters={"player_name": "player 2"})
    text = "".join([c async for c in it])
    assert text.count("\n") == 1 + 401 and text.startswith("game_id,player_id,player_name,date,pts")
    monkeypatch.setattr(lat, "MAX_ROWS_PER_LATTICE", 1204)
    with pytest.raises(lat.LatticeRefused) as e:
        await lat.write_rows(store, customer.id, g["id"], [{"game_id": "g9998", "player_id": "p1", "date": "2024-10-24"}])
    assert e.value.code == "lattice_too_big" and "rather than splitting" in str(e.value)
