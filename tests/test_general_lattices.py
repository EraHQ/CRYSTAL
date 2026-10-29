"""General lattices (Series 380, Q145 and Q146 B, 2026-10-27).

Anthony: "General tables and crystals have owners. The person who made
them. They are the only one that can write" and "I want to be able to
test the private to general promotion". A lattice starts private; its
owner may publish it to the shared general schema, other accounts
subscribe to read it (rows, SQL as general.<name>, ref columns pointing
at it), and only the owner writes.
"""
from __future__ import annotations

import uuid

import pytest

from crystal_cache.goals import planner
from crystal_cache.infrastructure import lattice_tables as lt
from crystal_cache.lattices import service as lat

# General names are one namespace for everyone, and these tests also run on
# a shared Postgres, so each test names its tables uniquely.
TEAMS = TEAMS_SQL = GAMES = GAMES_SQL = ""


@pytest.fixture(autouse=True)
def _fresh_names():
    global TEAMS, TEAMS_SQL, GAMES, GAMES_SQL
    tag = uuid.uuid4().hex[:6]
    TEAMS, TEAMS_SQL = f"NBA teams {tag}", f"nba_teams_{tag}"
    GAMES, GAMES_SQL = f"team games {tag}", f"team_games_{tag}"


async def _teams(store, cid):
    teams = await lat.create(store, cid, name=TEAMS, key_columns=["team_id"], columns=[
        {"key": "team_id"}, {"key": "team_name", "entity": True}])
    await lat.write_rows(store, cid, teams["id"], [{"team_id": "BOS", "team_name": "Celtics"},
                                                    {"team_id": "LAL", "team_name": "Lakers"}])
    return teams


async def _other(store, n=2):
    return await store.create_customer(provider="anthropic", model_id="m", api_key_ref=f"sk-{n}")


@pytest.mark.asyncio
async def test_promote_subscribe_read_and_only_the_owner_writes(store, customer):
    owner = customer.id
    teams = await _teams(store, owner)
    reader = (await _other(store)).id
    # Private: another account can't see it, even by name.
    assert [x for x in await lat.general_list(store, reader) if x["lattice_id"] == teams["id"]] == []
    with pytest.raises(lat.LatticeRefused) as e:
        await lat.subscribe(store, reader, teams["id"], on=True)
    assert e.value.code == "not_found"

    out = await lat.publish(store, owner, teams["id"], general=True)
    assert out["scope"] == "general"
    d = lat.describe(out)
    assert d["scope"] == "general" and d["sql_name"] == f"general.{TEAMS_SQL}"
    # The owner keeps reading and writing it, rows intact after the move.
    assert (await lat.sql(store, owner, f"SELECT team_name FROM general.{TEAMS_SQL} ORDER BY 1"))["rows"] == [["Celtics"], ["Lakers"]]
    await lat.write_rows(store, owner, teams["id"], [{"team_id": "NYK", "team_name": "Knicks"}])

    listed = {x["lattice_id"]: x for x in await lat.general_list(store, reader)}
    assert listed[teams["id"]]["yours"] is False and listed[teams["id"]]["subscribed"] is False
    with pytest.raises(lat.LatticeRefused) as e:
        await lat.subscribe(store, owner, teams["id"], on=True)
    assert e.value.code == "own_lattice"
    # Not subscribed yet: out of reach.
    assert await store.get_lattice_by_name(reader, TEAMS) is None
    await lat.subscribe(store, reader, f"general.{TEAMS_SQL}", on=True)
    assert next(x for x in await lat.general_list(store, owner) if x["lattice_id"] == teams["id"])["readers"] == 1

    q = await lat.query(store, reader, TEAMS)
    assert sorted(r["values"]["team_id"] for r in q["rows"]) == ["BOS", "LAL", "NYK"]
    got = await lat.sql(store, reader, f"SELECT count(*) AS n FROM general.{TEAMS_SQL}")
    assert got["rows"] == [[3]] and got["lattices"] == {f"general.{TEAMS_SQL}": TEAMS}
    views = await lat.list_view(store, reader)
    assert [(v["name"], v["yours"]) for v in views] == [(TEAMS, False)]

    # The reader can't write it, in any way.
    for call in (lambda: lat.write_rows(store, reader, teams["id"], [{"team_id": "MIA", "team_name": "Heat"}]),
                 lambda: lat.update(store, reader, teams["id"], name="Mine now"),
                 lambda: lat.delete(store, reader, teams["id"]),
                 lambda: lat.publish(store, reader, teams["id"], general=False)):
        with pytest.raises(lat.LatticeRefused) as e:
            await call()
        assert e.value.code == "general_read_only"

    # The reader's own lattice points at the general one.
    games = await lat.create(store, reader, name=GAMES, key_columns=["game_id", "team_id"], columns=[
        {"key": "game_id"}, {"key": "team_id", "type": "ref", "ref": TEAMS, "entity": True},
        {"key": "pts", "type": "number"}])
    await lat.write_rows(store, reader, games["id"], [{"game_id": "g1", "team_id": "bos", "pts": 110},
                                                       {"game_id": "g1", "team_id": "LAL", "pts": 104}])
    rows = (await lat.query(store, reader, games["id"], sort="pts", descending=True))["rows"]
    assert [r["labels"] for r in rows] == [{"team_id": "Celtics"}, {"team_id": "Lakers"}]
    joined = await lat.sql(store, reader, f"SELECT t.team_name, g.pts FROM {GAMES_SQL} g "
                                          f"JOIN general.{TEAMS_SQL} t USING (team_id) ORDER BY 2 DESC")
    assert joined["rows"] == [["Celtics", 110.0], ["Lakers", 104.0]]
    # A team it hasn't got can't be stubbed into someone else's table.
    with pytest.raises(lat.LatticeRefused) as e:
        await lat.write_rows(store, reader, games["id"], [{"game_id": "g2", "team_id": "SEA", "pts": 90}])
    assert e.value.code == "unknown_reference"
    assert await store.count_lattice_rows(owner, teams["id"]) == 3

    # Unsubscribing while a lattice points at it is refused; so is making it private.
    with pytest.raises(lat.LatticeRefused) as e:
        await lat.subscribe(store, reader, teams["id"], on=False)
    assert e.value.code == "lattice_referenced"
    with pytest.raises(lat.LatticeRefused) as e:
        await lat.publish(store, owner, teams["id"], general=False, even_with_readers=True)
    assert e.value.code == "general_referenced"
    await lat.delete(store, reader, games["id"])
    with pytest.raises(lat.LatticeRefused) as e:
        await lat.publish(store, owner, teams["id"], general=False)
    assert e.value.code == "general_has_readers" and e.value.detail["readers"] == 1
    back = await lat.publish(store, owner, teams["id"], general=False, even_with_readers=True)
    assert back["scope"] == "private"
    assert await store.get_lattice_by_name(reader, TEAMS) is None
    assert (await lat.sql(store, owner, f"SELECT count(*) FROM {TEAMS_SQL}"))["rows"] == [[3]]
    with pytest.raises(lat.LatticeRefused):
        await lat.sql(store, reader, f"SELECT count(*) FROM general.{TEAMS_SQL}")


@pytest.mark.asyncio
async def test_publishing_needs_its_references_public_and_a_free_name(store, customer):
    owner = customer.id
    teams = await _teams(store, owner)
    games = await lat.create(store, owner, name=GAMES, key_columns=["game_id", "team_id"], columns=[
        {"key": "game_id"}, {"key": "team_id", "type": "ref", "ref": TEAMS, "entity": True}])
    with pytest.raises(lat.LatticeRefused) as e:
        await lat.publish(store, owner, games["id"], general=True)
    assert e.value.code == "refs_private" and e.value.detail["private"] == [TEAMS]
    await lat.publish(store, owner, teams["id"], general=True)
    await lat.write_rows(store, owner, games["id"], [{"game_id": "g1", "team_id": "BOS"}])
    assert (await lat.publish(store, owner, games["id"], general=True))["scope"] == "general"
    # Both general; the foreign key still holds and the join still runs.
    assert (await lat.sql(store, owner, f"SELECT t.team_name FROM general.{GAMES_SQL} g "
                                        f"JOIN general.{TEAMS_SQL} t USING (team_id)"))["rows"] == [["Celtics"]]
    other = (await _other(store)).id
    mine = await _teams(store, other)
    with pytest.raises(lat.LatticeRefused) as e:
        await lat.publish(store, other, mine["id"], general=True)
    assert e.value.code == "general_name_taken"


@pytest.mark.asyncio
async def test_erase_drops_the_general_lattices_an_account_owns(store, customer):
    owner = customer.id
    teams = await _teams(store, owner)
    await lat.publish(store, owner, teams["id"], general=True)
    reader = (await _other(store)).id
    await lat.subscribe(store, reader, teams["id"], on=True)
    await store.erase_tenant_bank(owner)
    assert [x for x in await lat.general_list(store, reader) if x["lattice_id"] == teams["id"]] == []
    assert await lat.list_view(store, reader) == []
    _s, phys = lt.located(store._lattice_engine, owner, teams["id"], general=True)
    async with store._lattice_engine.connect() as c:
        if store._lattice_engine.dialect.name == "postgresql":
            n = (await c.exec_driver_sql(f"SELECT count(*) FROM pg_tables WHERE schemaname = 'general' "
                                         f"AND tablename = '{phys}'")).scalar()
        else:
            n = (await c.exec_driver_sql(f"SELECT count(*) FROM sqlite_master WHERE name = '{phys}'")).scalar()
    assert n == 0


def test_lint_lets_a_plan_point_at_a_lattice_it_reads():
    def col(key, typ="text", entity=False, ref=""):
        return {"key": key, "label": key, "type": typ, "kind": "snapshot", "entity": entity, "ref": ref}
    games = [{"name": "team_games", "key_columns": ["game_id", "team_id"],
              "columns": [col("game_id"), col("team_id", "ref", True, TEAMS), col("date", "date")]}]
    assert any("not one of the plan's lattices" in i for i in planner.lint_lattices(games))
    known = [{"name": TEAMS, "key_columns": ["team_id"], "columns": [{"key": "team_id"}]}]
    assert not any("team_id" in i for i in planner.lint_lattices(games, known))
    copied = [{"name": "team_games", "key_columns": ["game_id", "team_id"],
               "columns": [col("game_id"), col("team_id", entity=True), col("date", "date")]}]
    assert any(f"'team_id' is the key of '{TEAMS}'; make it a ref" in i for i in planner.lint_lattices(copied, known))
    text = planner.planner_user({"title": "t", "directive": "d"}, [], None,
                                [{"name": TEAMS, "sql_name": TEAMS_SQL, "key_columns": ["team_id"],
                                  "columns": [{"key": "team_id"}, {"key": "team_name"}]}])
    assert f"general.{TEAMS_SQL}" in text and "not write" in text


@pytest.mark.asyncio
async def test_console_routes_publish_list_and_subscribe(store, customer):
    from test_lattices import _console
    owner = customer.id
    teams = await _teams(store, owner)
    reader = (await _other(store)).id
    async with _console(store) as c:
        r = await c.post(f"/admin/api/customers/{owner}/lattices/{teams['id']}/publish", json={"general": "yes"})
        assert r.status_code == 400
        r = await c.post(f"/admin/api/customers/{owner}/lattices/{teams['id']}/publish", json={"general": True})
        assert r.status_code == 200 and r.json()["lattice"]["scope"] == "general"
        r = await c.get(f"/admin/api/customers/{reader}/lattices/general")
        mine = [x for x in r.json()["lattices"] if x["lattice_id"] == teams["id"]]
        assert r.status_code == 200 and mine[0]["subscribed"] is False and "customer_id" not in mine[0]
        r = await c.post(f"/admin/api/customers/{reader}/lattices/{teams['id']}/subscribe", json={"subscribe": True})
        assert r.status_code == 200 and r.json()["subscribed"] is True
        r = await c.post(f"/admin/api/customers/{reader}/lattices/{teams['id']}/rows",
                         json={"rows": [{"team_id": "MIA", "team_name": "Heat"}]})
        assert r.status_code == 403 and r.json()["detail"]["error"]["code"] == "general_read_only"
        r = await c.post(f"/admin/api/customers/{owner}/lattices/{teams['id']}/publish", json={"general": False})
        assert r.status_code == 409 and r.json()["detail"]["error"]["code"] == "general_has_readers"
