"""Lattices as real tables (Series 378, Q144 A, 2026-10-27).

Anthony: "lets not remake sql what can we use" and, on ids: "ideally we
would be making a table for player and team ids in general. And reference
them like you traditionally do in sql." Each lattice is its own typed
table in its customer's schema; a ref column points at another lattice's
key (a foreign key on Postgres), so a name lives once; a read-only SQL
tool answers questions over a customer's lattices and nobody else's.
"""
from __future__ import annotations

import pytest
from sqlalchemy import select

from crystal_cache.goals import planner
from crystal_cache.infrastructure import lattice_tables as lt
from crystal_cache.infrastructure.schema import LatticeRecordRow, LatticeRow
from crystal_cache.lattices import service as lat


async def _players_and_games(store, cid):
    players = await lat.create(store, cid, name="NBA players", key_columns=["player_id"], columns=[
        {"key": "player_id"}, {"key": "player_name", "entity": True}, {"key": "position"}])
    games = await lat.create(store, cid, name="nba player games", key_columns=["game_id", "player_id"], columns=[
        {"key": "game_id"}, {"key": "player_id", "type": "ref", "ref": "NBA players", "entity": True},
        {"key": "date", "type": "date"}, {"key": "pts", "type": "number"}])
    return players, games


@pytest.mark.asyncio
async def test_a_ref_column_points_at_one_lattice_and_names_live_there(store, customer):
    cid = customer.id
    players, games = await _players_and_games(store, cid)
    assert players["sql_name"] == "nba_players" and games["sql_name"] == "nba_player_games"
    assert games["columns"][1]["ref"] == players["id"]
    await lat.write_rows(store, cid, players["id"], [{"player_id": "jamesle01", "player_name": "LeBron James"}])
    out = await lat.write_rows(store, cid, games["id"], [
        {"game_id": "g1", "player_id": "JAMESLE01", "date": "2024-10-22", "pts": 16},
        {"game_id": "g1", "player_id": "davisan02", "date": "2024-10-22", "pts": 36}], source="https://x.test/g1")
    # A player seen before the players list gets a row there holding only the id.
    assert out["inserted"] == 2 and out["stubs"] == {"NBA players": 1}
    q = await lat.query(store, cid, games["id"], sort="pts", descending=True)
    assert [r["values"]["player_id"] for r in q["rows"]] == ["davisan02", "jamesle01"]  # the players list's spelling
    assert q["rows"][1]["labels"] == {"player_id": "LeBron James"} and "labels" not in q["rows"][0]
    assert q["rows"][0]["values"] == {"game_id": "g1", "player_id": "davisan02", "date": "2024-10-22", "pts": 36}
    with pytest.raises(lat.LatticeRefused) as e:
        await lat.delete(store, cid, players["id"])
    assert e.value.code == "lattice_referenced" and e.value.detail["referenced_by"] == ["nba player games"]
    # Undo takes the stub back with the rows that made it.
    from crystal_cache.cognition.controls import undo_action
    aid = await lat.ledger(store, cid, out, actor="console", reason="Box score for game one.")
    assert (await undo_action(store, cid, aid, actor="console"))["undone"] is True
    assert await store.count_lattice_rows(cid, games["id"]) == 0
    assert [r["values"]["player_id"] for r in (await lat.query(store, cid, players["id"]))["rows"]] == ["jamesle01"]
    with pytest.raises(lat.LatticeRefused) as e:
        await lat.create(store, cid, name="team games", key_columns=["game_id"],
                         columns=[{"key": "game_id"}, {"key": "team_id", "type": "ref", "ref": "NBA teams"}])
    assert e.value.code == "ref_target_missing"


@pytest.mark.asyncio
async def test_sql_answers_over_the_customers_lattices_and_nothing_else(store, customer):
    cid = customer.id
    players, games = await _players_and_games(store, cid)
    await lat.write_rows(store, cid, players["id"], [{"player_id": "p1", "player_name": "Ada"},
                                                      {"player_id": "p2", "player_name": "Bo"}])
    await lat.write_rows(store, cid, games["id"], [
        {"game_id": f"g{i}", "player_id": "p1" if i % 2 else "p2", "date": f"2024-10-{20 + i}", "pts": 10 + i}
        for i in range(6)])
    out = await lat.sql(store, cid, "SELECT p.player_name, count(*) AS games, avg(g.pts) AS avg_pts "
                                    "FROM nba_player_games g JOIN nba_players p USING (player_id) "
                                    "WHERE g.date >= '2024-10-21' GROUP BY 1 ORDER BY 3 DESC, 1;")
    assert out["columns"] == ["player_name", "games", "avg_pts"]
    assert out["rows"] == [["Ada", 3, 13.0], ["Bo", 2, 13.0]] and out["truncated"] is False
    assert out["lattices"] == {"nba_players": "NBA players", "nba_player_games": "nba player games"}
    many = await lat.sql(store, cid, "SELECT game_id FROM nba_player_games", limit=4)
    assert many["row_count"] == 4 and many["truncated"] is True
    # Another customer's lattice of the same name is out of reach, as is any table that isn't a lattice.
    other = await store.create_customer(provider="anthropic", model_id="m", api_key_ref="sk-2")
    o = await lat.create(store, other.id, name="secrets", key_columns=["k"], columns=[{"key": "k"}])
    schema, phys = lt.located(store._lattice_engine, other.id, o["id"])
    theirs = f'"{schema}"."{phys}"' if schema else f'"{phys}"'
    # On Postgres an unqualified name outside the customer's schema doesn't resolve at all.
    other_table = "sql_error" if store._lattice_engine.dialect.name == "postgresql" else "sql_denied"
    for q, code in [(f'SELECT * FROM {theirs}', "sql_denied"), ("SELECT * FROM customers", other_table),
                    ("DELETE FROM nba_players", "sql_read_only"), ("SELECT 1; DROP TABLE x", "sql_one_statement"),
                    ("SELECT * FROM pg_catalog.pg_class", "sql_denied"), ("SELECT * FROM secrets", "sql_error"),
                    ("SELECT nope FROM nba_players", "sql_error")]:
        with pytest.raises(lat.LatticeRefused) as e:
            await lat.sql(store, cid, q)
        assert e.value.code == code, (q, e.value.code, e.value.message)
    if store._lattice_engine.dialect.name == "postgresql":
        # The role reads views only: not even this customer's own tables (which hold provenance) directly.
        own_schema, own = lt.located(store._lattice_engine, cid, players["id"])
        with pytest.raises(lat.LatticeRefused) as e:
            await lat.sql(store, cid, f'SELECT * FROM "{own_schema}"."{own}"')
        assert e.value.code == "sql_denied"
        with pytest.raises(lat.LatticeRefused) as e:
            await lat.sql(store, cid, "SELECT count(*) FROM public.customers")
        # Denied in one database; with lattices in their own database (Q147 A) it isn't there at all.
        assert e.value.code in ("sql_denied", "sql_error")
    # A semicolon inside a string is data, not a second statement.
    assert (await lat.sql(store, cid, "SELECT player_name FROM nba_players WHERE player_name <> 'a;b'"))["row_count"] == 2


@pytest.mark.asyncio
async def test_a_lattice_from_before_378_moves_into_its_table_on_first_touch(store, customer):
    cid = customer.id
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    old = "lat_old" + cid[-8:]  # tenant-suffixed: the same test runs on a shared Postgres
    async with store.session() as s:
        s.add(LatticeRow(id=old, customer_id=cid, name="Old games", description="", key_columns=["game_id"],
                         columns=[{"key": "game_id", "label": "Game", "type": "text", "kind": "snapshot", "entity": False},
                                  {"key": "pts", "label": "Pts", "type": "number", "kind": "snapshot", "entity": False},
                                  {"key": "day", "label": "Day", "type": "date", "kind": "snapshot", "entity": False}],
                         created_at=now, updated_at=now))
        s.add(LatticeRecordRow(id="lr_a", lattice_id=old, customer_id=cid, row_key="g1",
                               data={"game_id": "g1", "pts": 101, "day": "2024-10-22"},
                               cells={"pts": {"as_of": now.isoformat(), "source": "https://x"}}, created_at=now, updated_at=now))
        s.add(LatticeRecordRow(id="lr_b", lattice_id=old, customer_id=cid, row_key="g2",
                               data={"game_id": "g2", "pts": "n/a", "day": None}, cells={}, created_at=now, updated_at=now))
        await s.commit()
    q = await lat.query(store, cid, "Old games")
    assert [r["values"] for r in q["rows"]] == [{"game_id": "g1", "pts": 101, "day": "2024-10-22"},
                                                {"game_id": "g2", "pts": None, "day": None}]
    async with store.session() as s:
        assert (await s.execute(select(LatticeRecordRow).where(LatticeRecordRow.customer_id == cid))).scalars().all() == []
        row = await s.get(LatticeRow, old)
    assert row.table_name == lt.physical_name(old) and row.sql_name == "old_games"
    assert (await lat.sql(store, cid, "SELECT sum(pts) AS total FROM old_games"))["rows"] == [[101.0]]


@pytest.mark.asyncio
async def test_erase_drops_the_tables_and_columns_can_be_added(store, customer):
    cid = customer.id
    players, games = await _players_and_games(store, cid)
    await lat.update(store, cid, players["id"], name="Players", add_columns=[{"key": "height_cm", "type": "number"}])
    await lat.write_rows(store, cid, players["id"], [{"player_id": "p1", "player_name": "Ada", "height_cm": 201}])
    assert (await lat.sql(store, cid, "SELECT height_cm FROM players"))["rows"] == [[201.0]]
    await store.erase_tenant_bank(cid)
    assert await store.list_lattices(cid) == []
    _s, phys = lt.located(store._lattice_engine, cid, games["id"])
    async with store._lattice_engine.connect() as c:
        if store._lattice_engine.dialect.name == "postgresql":
            left = (await c.exec_driver_sql(f"SELECT count(*) FROM pg_namespace WHERE nspname = '{lt.schema_name(cid)}'")).scalar()
            roles = (await c.exec_driver_sql(f"SELECT count(*) FROM pg_roles WHERE rolname = '{lt.role_name(cid)}'")).scalar()
            assert (left, roles) == (0, 0)
        else:
            names = [r[0] for r in (await c.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table'")).all()]
            assert not [n for n in names if n.startswith(lt.schema_name(cid))] and phys not in names


def test_lint_asks_for_reference_lattices_instead_of_copied_names():
    def col(key, typ="text", entity=False, ref=""):
        return {"key": key, "label": key, "type": typ, "kind": "snapshot", "entity": entity, "ref": ref}
    specs = [
        {"name": "nba_players", "key_columns": ["player_id"], "columns": [col("player_id"), col("player_name", entity=True)]},
        {"name": "nba_teams", "key_columns": ["team_id"], "columns": [col("team_id"), col("team_name", entity=True)]},
        {"name": "nba_player_games", "key_columns": ["game_id", "player_id"],
         "columns": [col("game_id"), col("player_id", entity=True), col("player_name"), col("date", "date")]},
        {"name": "nba_team_games", "key_columns": ["game_id", "team_id"],
         "columns": [col("game_id"), col("team_id", "ref", True, "nba_teams"), col("team_name"), col("date", "date"),
                     col("tip_off_winner_id", "ref", False, "nba_teams")]},
    ]
    issues = planner.lint_lattices(specs)
    assert any("nba_player_games: 'player_id' is the key of 'nba_players'; make it a ref" in i for i in issues)
    assert any("nba_team_games: 'team_name' copies a name the reference lattice holds" in i for i in issues)
    # Reference lattices need no date, and a ref that names a winner is a recorded fact, not a computed one.
    assert not any(i.startswith(("nba_players:", "nba_teams:")) for i in issues)
    assert not any("tip_off_winner" in i for i in issues)


@pytest.mark.asyncio
async def test_lattice_tables_live_in_their_own_database(tmp_path):
    # Q147 A (2026-10-27): SQL an agent writes connects only to a database
    # that holds lattices; the main database keeps the row describing each.
    from crystal_cache.config import Settings
    from crystal_cache.infrastructure import MetadataStore
    data_url = f"sqlite+aiosqlite:///{(tmp_path / 'lattice_data.db').as_posix()}"
    s = MetadataStore(settings_override=Settings(database_url="sqlite+aiosqlite:///:memory:", lattice_database_url=data_url))
    await s.init()
    try:
        assert s._lattice_engine is not s.engine
        c = await s.create_customer(provider="anthropic", model_id="m", api_key_ref="sk")
        players, games = await _players_and_games(s, c.id)
        await lat.write_rows(s, c.id, games["id"], [{"game_id": "g1", "player_id": "p1", "date": "2024-10-22", "pts": 9}])
        assert (await lat.sql(s, c.id, "SELECT count(*) FROM nba_players"))["rows"] == [[1]]  # the stub
        with pytest.raises(lat.LatticeRefused) as e:
            await lat.sql(s, c.id, "SELECT * FROM customers")
        assert e.value.code in ("sql_denied", "sql_error")
        _sch, phys = lt.located(s._lattice_engine, c.id, games["id"])
        async with s.engine.connect() as main:
            main_tables = [r[0] for r in (await main.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table'")).all()]
        async with s._lattice_engine.connect() as data:
            data_tables = [r[0] for r in (await data.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table'")).all()]
        assert phys in data_tables and phys not in main_tables
        assert "customers" in main_tables and "customers" not in data_tables and "lattices" not in data_tables
        assert (await lat.query(s, c.id, games["id"]))["rows"][0]["values"]["pts"] == 9
        await lat.update(s, c.id, players["id"], name="Players")
        assert (await lat.sql(s, c.id, "SELECT player_id FROM players"))["rows"] == [["p1"]]
        await s.erase_tenant_bank(c.id)
        async with s._lattice_engine.connect() as data:
            left = [r[0] for r in (await data.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table'")).all()]
        assert phys not in left
    finally:
        await s.dispose()
