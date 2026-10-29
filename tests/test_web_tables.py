"""Reading web tables and importing them into lattices (Q124 A, 2026-10-25).

The first sports check-ins wrote one, then five games: web_fetch hands back
a page's prose 12,000 characters at a time, so a schedule full of tables
came through a few rows per call. web_tables reads every table on a page
(including those a site ships inside HTML comments) with each cell's link;
lattice_import moves a table into a lattice on the server from a column
map, so no model copies rows. Generic: any page, any table.
"""
from __future__ import annotations

import pytest

from crystal_cache.lattices import service as lat
from crystal_cache.search import tables as wt

PAGE = """
<html><body>
<table id="schedule"><caption>2024-25 Schedule</caption>
<thead><tr><th>Date</th><th>Visitor</th><th>PTS</th><th>Home</th><th>PTS</th><th colspan="2">Box</th></tr></thead>
<tbody>
<tr><th data-stat="date_game">Tue, Oct 22, 2024</th><td><a href="/teams/NYK/2025.html">New York Knicks</a></td><td>109</td>
    <td><a href="/teams/BOS/2025.html">Boston Celtics</a></td><td>132</td><td><a href="/boxscores/202410220BOS.html">Box Score</a></td><td></td></tr>
<tr><th>Date</th><th>Visitor</th><th>PTS</th><th>Home</th><th>PTS</th><th>Box</th><th></th></tr>
<tr><th>Tue, Oct 22, 2024</th><td>Minnesota Timberwolves</td><td>103</td><td>Los Angeles Lakers</td><td>110</td>
    <td><a href="/boxscores/202410220LAL.html">Box Score</a></td><td></td></tr>
<tr><th>Wed, Oct 23, 2024</th><td>Brooklyn Nets</td><td></td><td>Atlanta Hawks</td><td></td><td></td><td></td></tr>
</tbody></table>
<div><!--
<table id="standings"><tr><th>Team</th><th>W</th></tr><tr><td>Boston &amp; co</td><td>61</td></tr></table>
--></div>
<table><tr><td></td></tr></table>
</body></html>"""


def test_tables_are_read_with_links_repeated_headers_dropped_and_comment_tables_found():
    tables = wt.parse_tables(PAGE)
    assert [t["id"] for t in tables] == ["schedule", "standings"]
    sched = tables[0]
    assert sched["caption"] == "2024-25 Schedule"
    assert sched["headers"] == ["Date", "Visitor", "PTS", "Home", "PTS (2)", "Box", "col7"]
    assert sched["row_count"] == 3  # the repeated header row is gone
    first = sched["rows"][0]
    assert [c["text"] for c in first][:5] == ["Tue, Oct 22, 2024", "New York Knicks", "109", "Boston Celtics", "132"]
    assert first[5]["href"] == "/boxscores/202410220BOS.html"
    assert tables[1]["rows"][0][0]["text"] == "Boston & co"
    preview = wt.table_preview(sched, rows=1)
    assert preview["links"] == ["Visitor", "Home", "Box"] and preview["sample"][0][2] == "109"
    assert wt.to_iso_date("Tue, Oct 22, 2024") == "2024-10-22"
    assert wt.to_iso_date("10/22/2024") == "2024-10-22" and wt.to_iso_date("Oct. 22, 2024") == "2024-10-22"
    assert wt.to_iso_date("someday") is None
    assert wt.pick(first[5], link=True, pattern=r"boxscores/(\w+)\.html") == "202410220BOS"


async def _loader(url):
    return "https://example.test/schedule", wt.parse_tables(PAGE)


GAMES = [{"key": "game_id"}, {"key": "date", "type": "date"}, {"key": "season"},
         {"key": "away_team", "entity": True}, {"key": "home_team", "entity": True},
         {"key": "away_points", "type": "number"}, {"key": "home_points", "type": "number"}]


@pytest.mark.asyncio
async def test_import_maps_a_table_into_a_lattice_on_the_server(store, customer):
    g = await lat.create(store, customer.id, name="nba_games", columns=GAMES, key_columns=["game_id"])
    cols = {"game_id": {"from": "Box", "link": True, "pattern": r"boxscores/(\w+)\.html"}, "date": "Date",
            "away_team": "Visitor", "home_team": "Home", "away_points": "PTS", "home_points": "PTS (2)"}
    out = await lat.import_table(store, customer.id, "nba_games", url="https://example.test/schedule", table="schedule",
                                 columns=cols, constants={"season": "2024-25"}, actor="goal:g1",
                                 reason="Backfill the season schedule.", loader=_loader)
    assert (out["rows_read"], out["inserted"], out["error_count"]) == (3, 2, 1)
    assert out["errors"][0]["code"] == "key_missing" and out["errors"][0]["row"] == 2  # a game not played yet: no box score
    assert out["next_offset"] is None and len(out["action_ids"]) == 1
    q = await lat.query(store, customer.id, g["id"], sort="home_points", descending=True)
    assert q["rows"][0]["values"] == {"game_id": "202410220BOS", "date": "2024-10-22", "season": "2024-25",
                                      "away_team": "New York Knicks", "home_team": "Boston Celtics",
                                      "away_points": 109, "home_points": 132}
    # Series 383 (Q150 A): a page already in the lattice is refused, here
    # after the redirect ("x" lands on the schedule page) ...
    with pytest.raises(lat.LatticeRefused) as e:
        await lat.import_table(store, customer.id, "nba_games", url="x", table=0, columns=cols,
                               constants={"season": "2024-25"}, loader=_loader)
    assert e.value.code == "page_already_imported" and e.value.detail["rows"] == 2
    # ... and before any fetch when the URL is the page itself.
    async def never(url):
        raise AssertionError("fetched a page already imported")
    with pytest.raises(lat.LatticeRefused) as e:
        await lat.import_table(store, customer.id, "nba_games", url="https://EXAMPLE.test/schedule/#x", table=0,
                               columns=cols, loader=never)
    assert e.value.code == "page_already_imported"
    assert [s["rows"] for s in await store.lattice_sources(customer.id, out["lattice_id"])] == [2]
    # Read again on purpose, it updates by key: nothing new, nothing changed.
    again = await lat.import_table(store, customer.id, "nba_games", url="x", table=0, columns=cols,
                                   constants={"season": "2024-25"}, loader=_loader, refresh=True)
    assert (again["inserted"], again["unchanged"]) == (0, 2)
    for kw, code in (({"columns": {"nope": "Date"}}, "unknown_columns"),
                     ({"columns": {"date": "Date"}}, "key_columns_unmapped"),
                     ({"columns": {**cols, "date": "When"}}, "no_such_header"),
                     ({"columns": cols, "table": 9}, "no_such_table"),
                     ({"columns": {**cols, "game_id": {"from": "Box", "pattern": "("}}}, "bad_pattern")):
        args = {"url": "x", "table": "schedule", **kw}
        with pytest.raises(lat.LatticeRefused) as e:
            await lat.import_table(store, customer.id, "nba_games", loader=_loader, refresh=True, **args)
        assert e.value.code == code, kw


@pytest.mark.asyncio
async def test_web_tables_tool_previews_then_pages_rows(monkeypatch):
    from crystal_cache.agent.tool_registry import get_registry, import_all_tools
    monkeypatch.setattr(wt, "load_tables", _loader)
    import_all_tools()
    tool = get_registry().get("web_tables")
    out = await tool.impl(customer_id="c1", url="example.test/schedule")
    assert [t["id"] for t in out["tables"]] == ["schedule", "standings"]
    out = await tool.impl(customer_id="c1", url="example.test/schedule", table="schedule", offset=1, limit=1)
    assert out["rows"] == [["Tue, Oct 22, 2024", "Minnesota Timberwolves", "103", "Los Angeles Lakers", "110", "Box Score", ""]]
    assert out["row_links"] == [{"Box": "/boxscores/202410220LAL.html"}] and out["next_offset"] == 2
    out = await tool.impl(customer_id="c1", url="example.test/schedule", table="7")
    assert out["code"] == "no_such_table"


@pytest.mark.asyncio
async def test_relative_links_become_absolute_and_a_failing_column_is_named(store, customer, monkeypatch):
    # Series 372 (2026-10-26): the first live import mapped source_url to the
    # box score link; the link was relative, so every row was refused as
    # "not a url" and the check-in made copies of the lattice instead.
    wt.reset_for_tests()
    monkeypatch.setattr("crystal_cache.search.fetch.fetch_and_extract",
                        lambda u, raw_html=False: {"kind": "html", "html": PAGE, "url": "https://example.test/leagues/s.html"})
    final, tables = await wt.load_tables("example.test/leagues/s.html")
    first = tables[0]["rows"][0]
    assert first[5]["href"] == "https://example.test/boxscores/202410220BOS.html"
    assert first[1]["href"] == "https://example.test/teams/NYK/2025.html"
    assert wt.table_preview(tables[0])["link_samples"]["Box"] == "https://example.test/boxscores/202410220BOS.html"
    await lat.create(store, customer.id, name="nba_games", columns=GAMES + [{"key": "source_url", "type": "url"}],
                     key_columns=["game_id"])
    cols = {"game_id": {"from": "Box", "link": True, "pattern": r"boxscores/(\w+)\.html"}, "date": "Date",
            "home_team": "Home", "source_url": {"from": "Box", "link": True}}
    out = await lat.import_table(store, customer.id, "nba_games", url="example.test/leagues/s.html", table=0, columns=cols)
    assert (out["inserted"], out["error_count"]) == (2, 1) and "hint" not in out
    assert out["error_columns"] == {"key_missing": 1}

    async def relative(url):
        return "https://example.test/s", wt.parse_tables(PAGE)
    out = await lat.import_table(store, customer.id, "nba_games", url="x", table=0,
                                 columns={**cols, "source_url": "Home"}, loader=relative)
    assert out["inserted"] == 0 and out["error_columns"] == {"source_url": 3}
    assert "source_url" in out["hint"] and "'nba_games'" in out["hint"]
    wt.reset_for_tests()
