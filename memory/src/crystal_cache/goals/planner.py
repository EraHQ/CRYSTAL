"""A goal's plan, written by the frontier tier (Q109 A).

One structured call: the person's words, the apps they connected (and
what each may do), and the goal's permissions go in; a plan comes out:

  summary            one sentence: what this goal is for
  success_measure    how anyone can tell it is working
  check_every_minutes how often a session should check in (floored at
                     goal_min_check_minutes; capped at a week)
  steps              ordered; each {title, detail, done}
  needs              {apps: [integration...], acts: [act...]} the plan
                     relies on; missing ones are reported, never assumed
  missing            Q112 A (2026-10-24): what the goal needs that this
                     account doesn't have yet (an app, a data source, a
                     tool, a permission, an answer), each with why, a next
                     step and whether it blocks. A needed app that isn't
                     connected is added here by the server, not left to the
                     model. Any blocking item holds the goal in
                     "needs_setup": no check-ins (no spend) until the person
                     fixes it or chooses Run anyway.
  decisions          (2026-10-24) planning is two steps when the goal
                     turns on choices only the person can make (which app,
                     where the data lives, scope). Anthony: "I don't think
                     the plan should plan for Notion specifically if I never
                     connected it. It should instead give decisions/options
                     as a step 2 to planning." The first call returns the
                     decisions (2-4 options, one recommended) and no steps;
                     the goal waits in "deciding"; the person's answers go
                     back to the planner, which then writes the plan on
                     them. Answers are final: the planner never re-asks.

  lattices           (2026-10-25, Q115 A) when results live in Crystal
                     storage and they are rows (games, stats, prices), the
                     tables to keep them in: name, columns (snapshot or
                     live cells, entity columns to browse by) and key
                     columns. Validated by lattices.service; the server
                     creates them when the plan is written, so every
                     check-in writes to the same table.

normalize_plan() never trusts the model's shape: unknown apps and acts
are dropped, counts and lengths are capped, the cadence is clamped.
"""
from __future__ import annotations

import re

import json
from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from ..connectors import catalog
from .criteria import CRITERIA_RULES, CRITERION_SCHEMA, lint_criteria, normalize_criteria

logger = structlog.get_logger(__name__)

PLAN_ORIGIN = "goal_plan"
PLAN_TIER = "frontier"
MAX_STEPS = 12
MAX_CHECK_MINUTES = 7 * 24 * 60
MAX_MISSING = 8
MISSING_KINDS = ("app", "data_source", "tool", "permission", "info")
MAX_DECISIONS = 4
# What a check-in can call (goals/session.py). The planner's prompt lists
# these as abilities; a "missing tool" naming one is the planner doubting
# its own instructions and is dropped.
SESSION_TOOL_NAMES = ("lattice_import", "web_tables", "web_fetch", "web_search", "lattice_write", "lattice_create",
                      "lattice_aggregate", "lattice_rows", "lattice_stale", "lattice_sql", "save_fact", "crystal_search",
                      "write_draft")
# Series 375 (2026-10-26): was 4. A plan for two leagues designed eight
# lattices and kept the first four, all NBA; NFL vanished without a word.
MAX_PLAN_LATTICES = 10
CRYSTAL_STORAGE = "Crystal storage"
# Q120 A (2026-10-25): how a plan's lattices must be shaped so the data can
# be analyzed, not only browsed. In the planner's prompt, in the review
# pass, and checked by lint_lattices().
DATA_MODEL_RULES = """1. One lattice per kind of record and level of detail (its grain): a game, a team in a game and a player in a game are three lattices. Records that share few fields (NBA and NFL games) get separate lattices, never one table with mostly empty columns. One kind of record is never split across lattices by season, year or date range: one lattice holds every season, filtered by its season or date column.
2. Key every row on stable ids from the source (game_id, player_id, team_id), never on display names. Each thing people browse by (players, teams) gets its own reference lattice keyed on its id, holding its name and other fixed facts (nba_players: player_id, player_name, position). Every other lattice holds that id in a ref column pointing at it (type ref, ref: the reference lattice's name), never a copy of the name: a name lives once, and browsing looks it up. Lattices that relate share the id column under the same key so they can be joined (game_id in the games lattice and in the player-games lattice). List reference lattices before the lattices that point at them.
3. One value per cell, typed: numbers as number, dates as date. Split compound values: "5-12" is two columns (fg_made, fg_attempted); "W 112-104" is a score per side.
4. Store raw counts, never values computed from other columns: no percentages, rates, totals or winners that the other columns already give. Totals across rows (a season's points on each game's row, a running record) are never stored: lattice_aggregate computes them from the rows. A live column belongs only on a lattice whose rows are themselves the thing that changes (a team's current standing, a product's price).
5. Every row carries what people filter by: a date and season (and league when one lattice covers several). Mark what people browse by (team, player) as entity.
6. At most 30 columns per lattice; split by level of detail rather than going wide."""
MAX_LATTICE_COLUMNS_ADVISED = 30
CRYSTAL_STORAGE_DETAIL = ("No app needed. Crystal keeps it in the form it has: a lattice (a table you can browse "
                          "and edit) for rows, facts for single statements")
_CRYSTAL_OPTION = re.compile(r"\bcrystal\b", re.I)

_LATTICE_COLUMN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"key": {"type": "string"}, "label": {"type": "string"},
                   "type": {"type": "string", "enum": ["text", "number", "date", "bool", "url", "ref"]},
                   "kind": {"type": "string", "enum": ["snapshot", "live"]}, "entity": {"type": "boolean"},
                   "refresh_minutes": {"type": "integer"}, "ref": {"type": "string"}},
    "required": ["key", "label", "type", "kind", "entity", "refresh_minutes", "ref"],
    "additionalProperties": False,
}

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "success_measure": {"type": "string"},
        "check_every_minutes": {"type": "integer"},
        "catch_up_every_minutes": {"type": "integer"},
        "steps": {"type": "array", "items": {
            "type": "object",
            "properties": {"title": {"type": "string"}, "detail": {"type": "string"}, "catch_up": {"type": "boolean"},
                           "repeats": {"type": "boolean"}, "criteria": {"type": "array", "items": CRITERION_SCHEMA}},
            "required": ["title", "detail", "catch_up", "repeats", "criteria"],
            "additionalProperties": False,
        }},
        "needs_apps": {"type": "array", "items": {"type": "string"}},
        "needs_acts": {"type": "array", "items": {"type": "string"}},
        "standing_instructions": {"type": "string"},
        "missing": {"type": "array", "items": {
            "type": "object",
            "properties": {"kind": {"type": "string", "enum": list(MISSING_KINDS)}, "need": {"type": "string"},
                           "why": {"type": "string"}, "suggestion": {"type": "string"}, "blocking": {"type": "boolean"}},
            "required": ["kind", "need", "why", "suggestion", "blocking"],
            "additionalProperties": False,
        }},
        "decisions": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "question": {"type": "string"}, "why": {"type": "string"},
                "options": {"type": "array", "items": {
                    "type": "object", "properties": {"label": {"type": "string"}, "detail": {"type": "string"}},
                    "required": ["label", "detail"], "additionalProperties": False}},
                "recommended": {"type": "integer"},
            },
            "required": ["question", "why", "options", "recommended"],
            "additionalProperties": False,
        }},
        "lattices": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "description": {"type": "string"},
                           "columns": {"type": "array", "items": _LATTICE_COLUMN_SCHEMA},
                           "key_columns": {"type": "array", "items": {"type": "string"}}},
            "required": ["name", "description", "columns", "key_columns"],
            "additionalProperties": False,
        }},
    },
    "required": ["summary", "success_measure", "check_every_minutes", "catch_up_every_minutes", "steps", "needs_apps", "needs_acts",
                 "standing_instructions", "missing", "decisions", "lattices"],
    "additionalProperties": False,
}


APP_STATUS = {"connected": "connected", "ready": "ready to connect",
              "setup": "not set up on this server yet (choosing it means setup first)"}


def app_statuses(connections: list[dict[str, Any]], offered_keys: Optional[set[str]] = None) -> dict[str, str]:
    """integration -> connected | ready | setup, for every catalog app."""
    live = {c["integration"] for c in connections if c["status"] == "active"}
    keys = offered_keys if offered_keys is not None else {a.integration for a in catalog.APPS}
    return {a.integration: "connected" if a.integration in live else "ready" if a.integration in keys else "setup"
            for a in catalog.APPS}


def limits_text(limits: Optional[dict[str, Any]] = None) -> str:
    """What fits, in rows, for the planner and the review."""
    from ..lattices import service as lat
    limits = limits or {}
    cap, used = limits.get("row_cap"), int(limits.get("rows_used") or 0)
    budget = (f"this person's plan holds {cap:,} rows across all lattices ({used:,} used)" if cap
              else "this person's plan has no total row cap")
    return (f"LIMITS: at most {MAX_PLAN_LATTICES} lattices per plan, {lat.MAX_ROWS_PER_LATTICE:,} rows per lattice, "
            f"{lat.MAX_COLUMNS} columns per lattice; {budget}. Estimate each lattice's rows (records times participants: "
            "a season of games times the players in each game) before you plan.")


def planner_system(min_minutes: int, statuses: Optional[dict[str, str]] = None,
                   limits: Optional[dict[str, Any]] = None) -> str:
    """statuses: every catalog app's status for this person (connected,
    ready to connect, not set up yet). The planner may offer any of them
    as an option in a decision, but builds steps only on connected ones."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    statuses = statuses or {a.integration: "ready" for a in catalog.APPS}
    apps = "\n".join(f"- {a.integration}: {a.name} ({a.kind}s; acts: {', '.join(a.acts) or 'none'}; "
                     f"{APP_STATUS[statuses.get(a.integration, 'setup')]})" for a in catalog.APPS)
    return f"""You plan a standing goal for one person. TODAY IS {today} (UTC). A smaller model will run this plan in short sessions, again and again, every check_every_minutes. Each session can: read what the person's connected apps showed in the last 7 days (emails, messages, events, files, pages), search the web and read web pages, search the person's memory bank, read the tables on a web page (web_tables) and move a whole table into a lattice on the server in one call (lattice_import, thousands of rows), save short facts to it (one fact per save, at most a few per session), keep rows in lattices (Crystal storage's tables: create one, write up to 500 rows per call, read and filter them, ask them anything in read-only SQL, list the live cells that are due for a refresh), file action items for the person, write drafts (email replies, Slack posts, calendar events, additions to Notion pages), send or post a draft only when the goal is permitted to, submit a step for the server to check against its criteria, and report when it is stuck.

Everything above exists and works today: never list any of it as missing, and never ask the person to confirm or test it.

What a session CANNOT do: call any API or data feed other than the connected apps below and the open web, run code, or download files in bulk. Plans that need those must say so in missing.

Crystal storage is where results live when no app is chosen. It keeps data in the form it has: a LATTICE for many records with the same fields (games, standings, prices, stats; browsable by entity, editable by the person), FACTS for single statements. A lattice's cells are typed: snapshot for what is true once and then fixed (a game's date and final score), live for what keeps changing (a season shot %, a standing), with refresh_minutes saying how fresh it must stay so check-ins recompute it. When the plan keeps rows in Crystal storage, define each lattice in lattices (name, one-line description, columns with key lowercase_with_underscores, label, type, kind, entity true for the columns to browse by such as team or player, refresh_minutes for live columns and 0 otherwise, ref: for a column of type ref the name of the lattice it points at and "" otherwise, and key_columns: the snapshot columns that identify a row). They are created when the plan is written; steps refer to them by name. lattices is empty when nothing is kept as rows. Design lattices for analysis; the server checks every design against these rules and sends it back for revision when it breaks one:
{DATA_MODEL_RULES}
{limits_text(limits)} When what the person asked for does not fit (all of the history of something large), the scope is a decision: offer 2 to 4 scopes that each fit (never an option you know does not fit), recommend the widest that fits, and give each option's row estimate in its detail. Never plan beyond the limits.
Cover everything the person named: each subject (each league, each account, each product line) gets its lattices, and each value they named gets a column.
Catching up on existing data (a backfill, history, a backlog): plan it as catch-up passes. First the pages that list many records at once (a schedule, a results or standings page, a catalog or directory listing), moved with lattice_import when they are tables (thousands of rows per call) or written in batches of up to 500; then detail pages (one record's page) in later passes. Steps name the kind of page to use. A check-in moves hundreds or thousands of rows, not one.

Never build a plan around an app the person has not connected, or around a place for the data they did not choose. When the plan turns on a choice only the person can make (which app or connector to use, where results should live, how far back or how wide the scope goes, whether the goal may act on its own), FIRST return those choices as decisions and leave steps empty: each decision is one plain question, why it matters, 2 to 4 options (label plus one line of what it means), and the index of the option you recommend. "{CRYSTAL_STORAGE}" is always one option for where results live, and the one to recommend unless the person named an app for it or one is already connected that fits: it needs no setup. Ask only what changes the plan; at most {MAX_DECISIONS}. While any decision is open, return only the decisions, summary and success_measure: steps, missing and lattices stay empty until the person answers. When the person has already answered decisions (given below), those answers are final: never ask them again, and write the full plan on them with decisions empty.

Every app Crystal supports, with its status for this person. Steps may only use apps marked connected. Any app may appear as an option in a decision, labelled with its status. Acts are the only writes possible through an app (for example Google Sheets: create_spreadsheet, append_rows; Notion: append_page, which adds to an existing page and cannot create databases):
{apps}

Write the plan so that smaller model can follow it without guessing:
- summary: one sentence, what this goal is for.
- success_measure: how anyone can tell it is working, concrete and checkable (a number, a deadline, a state).
- check_every_minutes: how often a session should check in once the goal is caught up. Match the goal's clock: "reply within 3 hours" needs about 30 to 60; a weekly review needs 10080. Never below {min_minutes}.
- catch_up: true on steps that work through what already exists (importing history, a backfill, a backlog, a first pass over old records); catch_up_every_minutes: how often to check in while any catch-up step is not done, usually 60 so the catching up finishes in days, not months (0 when no step is catch-up). The goal slows to check_every_minutes by itself once every catch-up step is done.
- steps: the ordered work, 2 to {MAX_STEPS} steps. Ongoing goals end with a repeating step (for example "Each check-in: ...") with repeats true; every other step has repeats false and is done only when its criteria pass. Each step's detail says what to look at and what to produce, with the numbers from the survey (how many records, which pages, how they are split).
- criteria: {CRITERIA_RULES}
- needs_apps / needs_acts: only the integration ids and acts above that the plan truly relies on.
- standing_instructions: the rules every session follows (what counts as needing a reply, tone of drafts, what never to do). Short and specific.
- Technical choices are yours, never the person's: which site or page to read, which id system to key on, formats and column names. Make them and write them into the steps.
- A choice the person must make is a decision, never a missing item. missing is for what the person must DO or PROVIDE (connect something, hand over a key), not for what they must choose.
- missing: everything the goal needs that this account does not have right now, one item each: kind app (an app above that is not connected, or connected with Act off when the plan must send or post), data_source (an API, feed or dataset none of the apps provide), tool (an ability sessions lack, such as running code), permission (the goal must be allowed to send or post on its own), info (something only the person can tell you). need says what, why says which step needs it, suggestion is the person's next step. blocking is true only when no step of the plan can make real progress without it; when web search can carry the goal for now, even imperfectly, it is not blocking, and the step says so. An empty list when nothing is missing.
Do not invent facts about the person. If the goal would need an app that is not connected and the person has not chosen it in a decision, do not plan around it: ask which app to use (that one, its alternatives, or {CRYSTAL_STORAGE}) as a decision."""


def answered_decisions(plan: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    return [d for d in (plan or {}).get("decisions", []) if d.get("answer")]


def planner_user(goal: dict[str, Any], connections: list[dict[str, Any]], survey: Optional[dict[str, Any]] = None) -> str:
    from .survey import survey_text
    conns = [f"- {c['integration']}: watch {'on' if c['uses'].get('watch') else 'off'}, "
             f"act {'on' if c['uses'].get('act') else 'off'}, status {c['status']}" for c in connections]
    previous = ""
    p = goal.get("plan") or {}
    done = [s["title"] for s in p.get("steps", []) if s.get("done")]
    if done:
        # Only what was finished is shown, as facts, never the old steps
        # themselves (2026-10-25: a replan copied the old plan word for word
        # because its steps were in front of the planner). The plan is
        # written fresh; a step keeps its done mark only when its title is
        # kept.
        previous = (f"\n\nAlready done (facts, not a plan to copy; write the plan fresh and keep a step's exact title "
                    f"only if it is truly still done): {'; '.join(done)[:1500]}"
                    f"\nLast check-in: {(goal.get('last_summary') or 'none')[:1200]}")
    answered = answered_decisions(goal.get("plan"))
    if answered:
        previous = "\n\nThe person's decisions (final; plan on them):\n" + "\n".join(
            f"- {d['question']} -> {d['answer']}" for d in answered) + previous
    return (f"Goal title: {goal['title']}\n\nIn the person's words:\n{goal['directive'][:4000]}\n\n"
            f"Connected apps:\n{chr(10).join(conns) or '- none yet'}\n\n"
            f"Permissions: may send email on its own: {'yes' if goal.get('may_send') else 'no (drafts wait for the person)'}; "
            f"may post/create on its own: {'yes' if goal.get('may_post') else 'no (drafts wait for the person)'}."
            f"{previous}{survey_text(survey)}")


def normalize_steps(raw_steps: Any, previous: Optional[dict[str, Any]],
                    lattices: Optional[list[dict[str, Any]]] = None) -> tuple[list[dict[str, Any]], list[str]]:
    """Steps, numbered, with their criteria (normalized against the plan's
    lattices). A step whose exact title was done in the previous plan
    stays done, with its record."""
    was = {str(s.get("title", "")).strip().lower(): s for s in (previous or {}).get("steps", []) if s.get("done")}
    steps: list[dict[str, Any]] = []
    dropped: list[str] = []
    for s in (raw_steps or [])[:MAX_STEPS]:
        if not isinstance(s, dict):
            continue
        title = str(s.get("title") or "").strip()[:200]
        if not title:
            continue
        repeats = bool(s.get("repeats")) or title.lower().startswith(("each check-in", "every check-in"))
        crit, why = ([], []) if repeats else normalize_criteria(s.get("criteria"), lattices)
        dropped += [f"step {len(steps) + 1}: {w}" for w in why]
        old = was.get(title.lower())
        step = {"n": len(steps) + 1, "title": title, "detail": str(s.get("detail") or "").strip()[:1200],
                "catch_up": bool(s.get("catch_up")), "repeats": repeats, "criteria": crit,
                "done": old is not None and not repeats, "note": ""}
        if old is not None and not repeats:
            for k in ("note", "accepted_at", "last_check", "state"):
                if old.get(k) is not None:
                    step[k] = old[k]
        steps.append(step)
    return steps, dropped


def normalize_missing(raw_items: Any, needs_apps: list[str], connections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The model's list, cleaned, plus what the server knows for certain:
    a needed app that isn't connected (or needs signing in again) blocks."""
    out: list[dict[str, Any]] = []
    seen_apps: set[str] = set()
    live = {c["integration"]: c for c in connections}
    for it in raw_items or []:
        if not isinstance(it, dict):
            continue
        need = str(it.get("need") or "").strip()[:200]
        if not need:
            continue
        kind = str(it.get("kind") or "info")
        kind = kind if kind in MISSING_KINDS else "info"
        if kind == "tool" and any(t in need.lower() for t in SESSION_TOOL_NAMES):
            continue  # a check-in already has it (2026-10-25: a goal was held to "confirm lattice_import works")
        item = {"kind": kind, "need": need, "why": str(it.get("why") or "").strip()[:400],
                "suggestion": str(it.get("suggestion") or "").strip()[:400],
                # A missing tool never holds a goal: the person has nothing to connect or hand over for it.
                # Nor does info (Series 375): the person has nowhere to type the answer, and a held goal
                # waited on "which id system", a choice the planner makes itself.
                "blocking": bool(it.get("blocking")) and kind not in ("tool", "info")}
        if kind == "app":
            app = next((a for a in catalog.APPS if a.integration in need.lower() or a.name.lower() in need.lower()), None)
            if app is not None:
                if app.integration in seen_apps:
                    continue
                c = live.get(app.integration)
                if c is not None and c["status"] == "active":
                    continue  # connected after all; the model was wrong
                item["integration"] = app.integration
                seen_apps.add(app.integration)
        out.append(item)
    for integ in needs_apps:
        c = live.get(integ)
        if integ in seen_apps or (c is not None and c["status"] == "active"):
            continue
        app = catalog.get(integ)
        name = app.name if app else integ
        out.append({"kind": "app", "integration": integ, "blocking": True,
                    "need": f"Connect {name}" if c is None else f"Sign in to {name} again",
                    "why": "The plan's steps read or act through it.",
                    "suggestion": "Add it from the Command center (Add connection)." if c is None
                    else "Settings > Connections > Sign in again."})
        seen_apps.add(integ)
    return out[:MAX_MISSING]


def normalize_decisions(raw_items: Any, previous: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    """The answered decisions carry over (final); new ones are cleaned and
    capped. A new decision that repeats an answered question is dropped."""
    kept = answered_decisions(previous)
    asked = {d["question"].strip().lower() for d in kept}
    out = list(kept)
    for it in raw_items or []:
        if len(out) >= MAX_DECISIONS + len(kept) or not isinstance(it, dict):
            continue
        q = str(it.get("question") or "").strip()[:300]
        if not q or q.lower() in asked:
            continue
        opts = [{"label": str(o.get("label") or "").strip()[:120], "detail": str(o.get("detail") or "").strip()[:300]}
                for o in (it.get("options") or []) if isinstance(o, dict) and str(o.get("label") or "").strip()][:4]
        if len(opts) < 2:
            continue
        try:
            rec = int(it.get("recommended") or 0)
        except (TypeError, ValueError):
            rec = 0
        out.append({"question": q, "why": str(it.get("why") or "").strip()[:400], "options": opts,
                    "recommended": rec if 0 <= rec < len(opts) else 0, "answer": None})
        asked.add(q.lower())
    return out


def normalize_lattices(raw_items: Any) -> list[dict[str, Any]]:
    """The plan's lattices, through the same validation every door uses
    (lattices.service); one that doesn't validate is dropped, never
    half-made. refresh_minutes 0 (the schema's "none") means no refresh."""
    from ..lattices import service as lat
    out: list[dict[str, Any]] = []
    names: set[str] = set()
    for it in raw_items or []:
        if len(out) >= MAX_PLAN_LATTICES or not isinstance(it, dict):
            continue
        name = str(it.get("name") or "").strip()[:120]
        if len(name) < 2 or name.lower() in names:
            continue
        cols = []
        for c in it.get("columns") or []:
            if isinstance(c, dict):
                c = dict(c)
                if not c.get("refresh_minutes") or c.get("kind") != "live":
                    c.pop("refresh_minutes", None)
                cols.append(c)
        try:
            columns = lat.normalize_columns(cols)
            keys = lat.normalize_keys(it.get("key_columns"), columns)
        except lat.LatticeRefused as e:
            logger.info("goals.plan_lattice_dropped", name=name, code=e.code)
            continue
        out.append({"name": name, "description": str(it.get("description") or "").strip()[:500],
                    "columns": columns, "key_columns": keys})
        names.add(name.lower())
    return out


_NUMERIC_NAME = re.compile(r"(^|_)(score|scores|pts|points|made|att|attempts|attempted|yards|yds|rebounds|reb|assists|ast|"
                           r"steals|blocks|turnovers|minutes|min|fouls|goals|count|total|q[1-4]|ot|first_downs|"
                           r"sacks|interceptions|tds|touchdowns|fgm|fga|ftm|fta|tpm|tpa)(_|$)")
_DERIVED_NAME = re.compile(r"(^|_)(pct|percent|percentage|rate|winner|win_pct|efficiency|avg|average|ratio)(_|$)")


_PERIOD_SUFFIX = re.compile(r"_(19|20)\d{2}(_\d{2,4})?$|_(season|year|q)\d+$")
_ROLLUP_NAME = re.compile(r"^(season|career|ytd|cumulative|running|to_date)_|_(to_date|ytd|cumulative)$")
_KIND_COLUMNS = frozenset(("league", "sport", "type", "kind", "category", "record_type", "product_line", "segment"))
_INVENTED_KEYS = frozenset(("id", "row_id", "stat_id", "record_id", "entry_id", "line_id", "uuid", "pk", "key", "row_key"))


def lint_lattices(specs: list[dict[str, Any]]) -> list[str]:
    """What breaks DATA_MODEL_RULES, as plain sentences naming the lattice
    and column. Heuristics on names and types: the review pass decides."""
    issues: list[str] = []
    names = {s["name"].lower(): s for s in specs}
    pointed = {str(c.get("ref") or "").lower() for s in specs for c in s["columns"] if c["type"] == "ref"}
    # A lattice keyed on one id is a reference lattice (players, teams).
    refs_by_key = {s["key_columns"][0]: s["name"] for s in specs
                   if len(s["key_columns"]) == 1 and s["key_columns"][0].endswith("_id")}
    for spec in specs:
        name = spec["name"]
        cols = {c["key"]: c for c in spec["columns"]}
        is_reference = name.lower() in pointed or (len(spec["key_columns"]) == 1
                                                   and refs_by_key.get(spec["key_columns"][0]) == name)
        for k, c in cols.items():
            target = refs_by_key.get(k)
            if c["type"] == "text" and target and target != name:
                issues.append(f"{name}: '{k}' is the key of '{target}'; make it a ref column pointing at '{target}' "
                              "(type ref) so the id is checked and its name is looked up (rule 2).")
            if c["type"] == "ref" and str(c.get("ref") or "").lower() not in names:
                issues.append(f"{name}: '{k}' points at '{c.get('ref')}', which is not one of the plan's lattices (rule 2).")
            stem = k[:-3] if k.endswith("_id") else None
            if c["type"] == "ref" and stem and f"{stem}_name" in cols:
                issues.append(f"{name}: '{stem}_name' copies a name the reference lattice holds; drop it, browsing looks "
                              "it up (rule 2).")
        if len(spec["key_columns"]) == 1 and spec["key_columns"][0] in _INVENTED_KEYS:
            issues.append(f"{name}: key '{spec['key_columns'][0]}' is invented; key rows on the ids that identify the "
                          "record in the source (for example the parent's id plus the participant's id) (rule 2).")
        for k, c in cols.items():
            if c["entity"] and (k == "name" or k.endswith("_name")):
                stem = k[:-5] if k.endswith("_name") else "record"
                if f"{stem}_id" not in cols:
                    issues.append(f"{name}: '{k}' is browsed by name but there is no '{stem}_id'; names collide and change "
                                  "(rule 2).")
        for k in spec["key_columns"]:
            c = cols[k]
            if c["type"] == "text" and not (k.endswith("_id") or k == "id"):
                issues.append(f"{name}: key column '{k}' is a display value; key rows on the source's id (rule 2).")
        for k, c in cols.items():
            if c["type"] == "text" and _NUMERIC_NAME.search(k):
                issues.append(f"{name}: '{k}' looks numeric but is text; store numbers as number and split compound values (rule 3).")
            # "tipoff_winner_team_id" names who won (a recorded fact), not a computed value.
            if _DERIVED_NAME.search(k) and c["kind"] != "live" and c["type"] != "ref" and not k.endswith(("_id", "_name")):
                issues.append(f"{name}: '{k}' looks computed from other columns; store the raw counts instead (rule 4).")
        kinds = [k for k in cols if k in _KIND_COLUMNS and k not in spec["key_columns"]]
        if kinds and len(cols) > 8:
            issues.append(f"{name}: '{kinds[0]}' says which kind of record a row is; if some columns only apply to some "
                          "kinds, give each kind its own lattice (rule 1).")
        per_record = any(c["type"] == "date" and c["kind"] == "snapshot" for c in cols.values())
        for k, c in cols.items():
            if k in spec["key_columns"]:
                continue
            if _ROLLUP_NAME.search(k) or (per_record and c["kind"] == "live"):
                issues.append(f"{name}: '{k}' is a total across rows on a per-record lattice; lattice_aggregate computes it "
                              "from the rows, so drop it (rule 4).")
        for k, c in cols.items():
            if c["entity"] and c["type"] != "ref" and not is_reference and k.endswith("_id") \
                    and not any(n in cols for n in (k[:-3] + "_name", k[:-3])):
                issues.append(f"{name}: '{k}' is browsed by id with no name beside it; add '{k[:-3]}_name' so browsing "
                              "shows names (rule 2).")
        if not is_reference and not any(c["type"] == "date" for c in cols.values()) \
                and not any(k.startswith("season") for k in cols):
            issues.append(f"{name}: no date or season column to filter by (rule 5).")
        if _PERIOD_SUFFIX.search(name):
            issues.append(f"{name}: one kind of record split by period; keep every season in one lattice and filter by "
                          "its season or date column (rule 1).")
        if len(cols) > MAX_LATTICE_COLUMNS_ADVISED:
            issues.append(f"{name}: {len(cols)} columns; split it by level of detail (rule 6).")
    return issues


REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"lattices": PLAN_SCHEMA["properties"]["lattices"], "steps": PLAN_SCHEMA["properties"]["steps"]},
    "required": ["lattices", "steps"],
    "additionalProperties": False,
}


def review_system(limits: Optional[dict[str, Any]] = None) -> str:
    return f"""You review the table design of a plan that keeps data in Crystal storage lattices, so the data can be analyzed later (counts, averages, rankings, joins on shared ids), not only browsed. The rules:
{DATA_MODEL_RULES}
{limits_text(limits)}
Check each of these yourself; the automatic checker only reads names and types:
- Does each lattice hold one kind of record at one level of detail? A value that belongs to one participant of a record (each team's third downs in a game, each line item of an order, each attendee of a meeting) needs its own lattice at that level, keyed on the record's id plus the participant's id.
- Do records of different kinds share a lattice (two leagues, two product lines with different fields)? Give each its own. A column that names the kind of record (league, type, category) on a lattice whose other columns only apply to some kinds is the sign.
These rules override any lattice names the steps, the success measure or the person's answers use: rename and split freely, and rewrite the steps to match.
The steps carry criteria: how the server decides each step is complete. Check them too: every subject and value the person named is measured by some criterion; targets come from the survey's measured numbers when it has one (never a guess beside a measured number); a criterion that cannot fail (a minimum of 0, coverage of a column nobody fills) is replaced with one that can; prefer coverage and reference over raw row counts; criteria name the corrected lattices and columns. {CRITERIA_RULES}
Never drop data the goal asks for. Fix a lattice that breaks a rule instead of removing it: a computed or text column becomes the raw values it comes from (a quarter's winner becomes each team's points in that quarter), usually in a lattice at the right level of detail. Every value the person named in their words must have a column in some lattice when you are done.
- Does every subject the person named have its lattices (each league, each account, each product line)? Add the ones that are missing, shaped like their siblings.
- Totals across rows (season_points on each game's row) are dropped: lattice_aggregate computes them.
- Is every key made of real ids from the source? Invented keys (stat_id, row_id) and names are not.
- Do players, teams and the like have their own reference lattice, with every other lattice pointing at it through a ref column instead of copying names?
- Can related lattices be joined on a shared id column with the same key in both?
You get the goal, the survey, the plan's steps with their criteria, its lattices, the limits and what the checker found; the design must fit the limits. Return the corrected lattices (same JSON shape: name, description, columns with key, label, type, kind, entity, refresh_minutes (0 when not live), key_columns) and every step (title, detail, catch_up, repeats, criteria), rewritten where they name lattices or columns that changed or where their criteria fall short. Fix every problem the checker found unless it is wrong for this data, and anything else that breaks a rule. Keep what the goal needs; do not add columns nobody asked for."""


async def review_lattices(client: Any, store: Any, customer_id: str, goal: dict[str, Any],
                          plan: dict[str, Any], *, max_tokens: int, limits: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """One design-review pass when lint finds problems. Never fails the
    plan: an unusable review keeps the original design. Records what it
    found in plan["lattice_review"]."""
    from ..scan._seam import metered_call
    from .survey import survey_text
    # Q123 A (2026-10-25): every design is reviewed. The checker only sees
    # names and types; mixed kinds of record, a participant's values on the
    # parent's row and invented keys with a good-looking name need a reader.
    issues = lint_lattices(plan.get("lattices") or []) + lint_criteria(plan)
    before_names = {x["name"] for x in plan.get("lattices") or []} | set(plan.pop("lattices_over_cap", None) or [])
    user = (f"Goal: {goal['title']}\nIn the person's words: {goal['directive'][:2000]}\n\n"
            f"{survey_text(plan.get('survey'))}\n\n"
            f"Steps: {json.dumps([{'title': s['title'], 'detail': s['detail'], 'catch_up': s.get('catch_up', False), 'repeats': s.get('repeats', False), 'criteria': s.get('criteria', [])} for s in plan['steps']])[:12000]}\n\n"
            f"Lattices: {json.dumps(plan['lattices'])[:20000]}\n\nProblems the automatic checker found:\n"
            + ("\n".join(f"- {i}" for i in issues) if issues else "- none (it only reads names and types; review anyway)"))
    revised = False
    try:
        raw = await metered_call(client, customer_id=customer_id, origin=PLAN_ORIGIN, system=review_system(limits), user=user,
                                 max_tokens=max_tokens, tier=PLAN_TIER, store=store, json_schema=REVIEW_SCHEMA)
        out = json.loads((raw or "").strip())
        lattices = normalize_lattices(out.get("lattices"))
        if lattices:
            plan["lattices"] = lattices
            revised = True
            steps, dropped = normalize_steps(out.get("steps"), {"steps": plan["steps"]}, lattices)
            if steps:
                plan["steps"] = steps
                plan["criteria_dropped"] = dropped[:20]
    except Exception as e:  # noqa: BLE001 — the first design stands
        logger.warning("goals.lattice_review_failed", goal_id=goal["id"], error_type=type(e).__name__, error=str(e)[:200])
    after = {x["name"] for x in plan.get("lattices") or []}
    plan["lattice_review"] = {"issues": issues[:20], "revised": revised,
                              "removed": sorted(before_names - after),
                              "remaining": (lint_lattices(plan.get("lattices") or []) + lint_criteria(plan))[:20]}
    return plan


async def ensure_plan_lattices(store: Any, customer_id: str, goal: dict[str, Any], plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Create the plan's lattices that don't exist yet (a lattice of the
    same name is reused as it is: the person's table is never reshaped
    behind their back). Each entry gains its lattice_id."""
    from ..lattices import service as lat
    pending = list(plan.get("lattices") or [])
    # Reference lattices first: a ref column needs its target to exist.
    for _round in range(len(pending) + 1):
        waiting = []
        for spec in pending:
            found = await store.get_lattice_by_name(customer_id, spec["name"])
            if found is None:
                try:
                    found = await lat.create(store, customer_id, name=spec["name"], columns=spec["columns"],
                                             key_columns=spec["key_columns"], description=spec["description"],
                                             goal_id=goal["id"], actor=f"goal:{goal['id']}")
                except lat.LatticeRefused as e:
                    if e.code == "ref_target_missing":
                        waiting.append(spec)
                    else:
                        logger.info("goals.plan_lattice_not_created", name=spec["name"], code=e.code)
                    continue
            spec["lattice_id"] = found["id"]
        if not waiting or len(waiting) == len(pending):
            for spec in waiting:
                logger.info("goals.plan_lattice_not_created", name=spec["name"], code="ref_target_missing")
            break
        pending = waiting
    return plan.get("lattices") or []


STORAGE_GROUPS = ("Docs and knowledge", "Files")
_STATUS_RANK = {"connected": 0, "ready": 1, "setup": 2}


def _option(app: catalog.App, status: str) -> dict[str, str]:
    can = (", can write: " + ", ".join(catalog.ACT_LABEL.get(a, a).lower() for a in app.acts)) if app.acts else ", read only"
    return {"label": app.name, "detail": f"{APP_STATUS[status].split(' (')[0].capitalize()}{can}"}


def app_decision(app: catalog.App, statuses: dict[str, str]) -> dict[str, Any]:
    """The question the server asks when a plan leans on an app the person
    hasn't connected or chosen: that app, up to two alternatives from the
    same group (writable and connected first), and Crystal storage where
    results could live there instead (recommended unless an option is
    connected)."""
    alts = sorted((a for a in catalog.APPS if a.group == app.group and a.integration != app.integration),
                  key=lambda a: (not a.acts, _STATUS_RANK[statuses.get(a.integration, "setup")]))[:2]
    options = []
    if app.group in STORAGE_GROUPS:
        options.append({"label": CRYSTAL_STORAGE, "detail": CRYSTAL_STORAGE_DETAIL})
    options += [_option(a, statuses.get(a.integration, "setup")) for a in [app, *alts]]
    rec = next((i for i, a in enumerate([app, *alts]) if statuses.get(a.integration) == "connected"), None)
    recommended = (rec + (1 if app.group in STORAGE_GROUPS else 0)) if rec is not None else 0
    return {"question": f"This goal would use {app.name}, which isn't connected. What should it use?",
            "why": "The steps depend on it; Crystal won't plan around an app you haven't chosen.",
            "options": options[:4], "recommended": recommended, "answer": None, "asked_by": "crystal"}


def _catch_up_every(raw: dict[str, Any], min_minutes: int, every: int) -> Optional[int]:
    """How often to check in while a catch-up step is open: between the
    floor and the regular cadence; None when no step is catch-up."""
    if not any(isinstance(s, dict) and s.get("catch_up") for s in raw.get("steps") or []):
        return None
    try:
        v = int(raw.get("catch_up_every_minutes") or 60)
    except (TypeError, ValueError):
        v = 60
    return max(min_minutes, min(v, every))


def next_check_minutes(plan: Optional[dict[str, Any]], floor: int) -> int:
    """The cadence right now: catch-up pace while any catch-up step is not
    done, the regular pace after (Q125 A)."""
    plan = plan or {}
    every = max(floor, int(plan.get("check_every_minutes") or 60))
    fast = plan.get("catch_up_every_minutes")
    if fast and any(s.get("catch_up") and not s.get("done") for s in plan.get("steps", [])):
        return max(floor, min(int(fast), every))
    return every


def normalize_plan(raw: dict[str, Any], *, min_minutes: int, previous: Optional[dict[str, Any]] = None,
                   connections: Optional[list[dict[str, Any]]] = None,
                   statuses: Optional[dict[str, str]] = None, directive: str = "") -> dict[str, Any]:
    acts_known = {a for app in catalog.APPS for a in app.acts}
    try:
        every = int(raw.get("check_every_minutes") or 60)
    except (TypeError, ValueError):
        every = 60
    every = max(min_minutes, min(every, MAX_CHECK_MINUTES))
    lattices = normalize_lattices(raw.get("lattices"))
    steps, criteria_dropped = normalize_steps(raw.get("steps"), previous, lattices)
    decisions = normalize_decisions(raw.get("decisions"), previous)
    deciding = any(not d.get("answer") for d in decisions)
    if not steps and not deciding:
        raise ValueError("the plan has no steps")
    apps = [a for a in dict.fromkeys(str(x) for x in raw.get("needs_apps") or []) if catalog.get(a)]
    # The server's guarantee (2026-10-24, Anthony: "Instead of defaulting to
    # Notion I should at least get an option for Google Sheets"): a plan
    # that leans on an app the person hasn't connected, and hasn't chosen in
    # an answered decision, becomes a question instead of a plan.
    statuses = statuses or app_statuses(connections or [])
    model_apps = [m.get("integration") for m in normalize_missing(raw.get("missing"), [], connections or [])
                  if m.get("integration")]
    # Named in the person's own words counts as chosen ("post to #team in Slack").
    answered_text = (directive + " " + " ".join(str(d.get("answer") or "") for d in decisions)).lower()
    open_text = " ".join(d["question"] + " " + " ".join(o["label"] for o in d["options"])
                         for d in decisions if not d.get("answer")).lower()
    answered_q = {d["question"] for d in decisions if d.get("answer")}
    for integ in dict.fromkeys([*apps, *model_apps]):
        app = catalog.get(integ)
        if app is None or statuses.get(integ) == "connected" or app.name.lower() in answered_text:
            continue
        if app_decision(app, statuses)["question"] in answered_q:
            # Asked and answered with something else: the person said no to
            # this app. Never ask twice; drop it from what the plan needs.
            apps = [a for a in apps if a != integ]
            continue
        if app.name.lower() in open_text or sum(1 for d in decisions if not d.get("answer")) >= MAX_DECISIONS:
            continue
        decisions.append(app_decision(app, statuses))
        open_text += " " + app.name.lower()
    # The model's own app questions get the writable alternatives it left
    # out (it asked "Notion or Crystal memory?" and never mentioned Google
    # Sheets): every open question that offers an app gains, up to four
    # options, the same-group apps that can write, connected first.
    for d in decisions:
        if d.get("answer") or d.get("asked_by") == "crystal":
            continue
        # One name for Crystal's own place (2026-10-25, Anthony: "based on data
        # type crystal storage can mean lattice, facts, or something else"):
        # the model's "Crystal memory only" becomes Crystal storage, and a
        # question offering a place to keep things always offers it, first.
        for o in d["options"]:
            if _CRYSTAL_OPTION.search(o["label"]):
                o["label"], o["detail"] = CRYSTAL_STORAGE, CRYSTAL_STORAGE_DETAIL
        labels = " ".join(o["label"] for o in d["options"]).lower()
        named = [a for a in catalog.APPS if a.name.lower() in labels]
        if any(a.group in STORAGE_GROUPS for a in named) and CRYSTAL_STORAGE.lower() not in labels:
            rec_label = d["options"][d.get("recommended") or 0]["label"] if d["options"] else None
            d["options"] = [{"label": CRYSTAL_STORAGE, "detail": CRYSTAL_STORAGE_DETAIL}, *d["options"]][:4]
            d["recommended"] = next((i for i, o in enumerate(d["options"]) if o["label"] == rec_label), 0)
            labels += " " + CRYSTAL_STORAGE.lower()
        for app in named:
            for alt in sorted((a for a in catalog.APPS if a.group == app.group and a.acts and a.name.lower() not in labels),
                              key=lambda a: _STATUS_RANK[statuses.get(a.integration, "setup")]):
                if len(d["options"]) >= 4:
                    break
                d["options"].append(_option(alt, statuses.get(alt.integration, "setup")))
                labels += " " + alt.name.lower()
        # The console marks the recommendation itself; and an app that isn't
        # connected is never recommended over one that is or over memory.
        for o in d["options"]:
            o["label"] = re.sub(r"\s*\(recommended\)\s*$", "", o["label"], flags=re.I)

        def _cost(o: dict) -> int:
            hit = [x for x in catalog.APPS if x.name.lower() in o["label"].lower()]
            return _STATUS_RANK[statuses.get(hit[0].integration, "setup")] if hit else 0
        rec = d.get("recommended") or 0
        if 0 <= rec < len(d["options"]) and _cost(d["options"][rec]) > 0:
            d["recommended"] = min(range(len(d["options"])), key=lambda i: (_cost(d["options"][i]), i))
    if any(not d.get("answer") for d in decisions) and not deciding:
        deciding = True
    if deciding:
        steps = []
    # While the person is still choosing, nothing is "missing" yet: the app
    # they pick decides what to connect. Only the model's own notes stay.
    missing = normalize_missing(raw.get("missing"), [] if deciding else apps, connections or [])
    if deciding:
        missing = [{**m, "blocking": False} for m in missing if m["kind"] != "app"]
    return {
        "summary": str(raw.get("summary") or "").strip()[:400],
        "success_measure": str(raw.get("success_measure") or "").strip()[:400],
        "check_every_minutes": every,
        "catch_up_every_minutes": _catch_up_every(raw, min_minutes, every),
        "steps": steps,
        "needs": {
            "apps": apps,
            "acts": [a for a in dict.fromkeys(str(x) for x in raw.get("needs_acts") or []) if a in acts_known],
        },
        "standing_instructions": str(raw.get("standing_instructions") or "").strip()[:2000],
        "missing": missing,
        "blocked": any(m["blocking"] for m in missing),
        "decisions": decisions,
        "deciding": deciding,
        "lattices": [] if deciding else lattices,
        "criteria_dropped": [] if deciding else criteria_dropped[:20],
        "lattices_over_cap": [] if deciding else [str(x.get("name") or "")[:120] for x in (raw.get("lattices") or [])
                                                   [MAX_PLAN_LATTICES:] if isinstance(x, dict)],
        # Run anyway survives a replan: the person already chose.
        "run_anyway": bool((previous or {}).get("run_anyway")),
    }


async def plan_limits(store: Any, customer: Any) -> dict[str, Any]:
    """The row budget the planner sizes the plan to."""
    cap = None
    tier = getattr(customer, "subscription_tier", None)
    if tier:
        from ..control.admission import resolve_tier
        cap = resolve_tier(tier).lattice_row_cap
    try:
        used = await store.count_lattice_rows(customer.id)
    except Exception:  # noqa: BLE001
        used = 0
    return {"row_cap": cap, "rows_used": used}


async def plan_goal(store: Any, customer: Any, goal: dict[str, Any], *, llm: Any = None, encoder: Any = None,
                    fact_store: Any = None) -> dict[str, Any]:
    """Write (or revise) one goal's plan. Opens a goal run so the spend
    shows in Activity; never raises for the worker."""
    from ..config import get_settings
    from ..llm import get_llm_client
    from ..llm.lanes import tenant_background
    from ..scan._seam import metered_call

    settings = get_settings()
    min_minutes = int(settings.goal_min_check_minutes or 15)
    run_id = await store.open_goal_run(customer.id, payload={
        "topic": f"Goal plan: {goal['title'][:80]}", "goal_id": goal["id"], "phase": "plan"})
    now = datetime.now(timezone.utc)
    try:
        connections = await store.list_app_connections(customer.id)
        from ..connectors.service import available_apps
        offered = await available_apps(store, customer.id)
        statuses = app_statuses(connections, {a.integration for a in offered})
        limits = await plan_limits(store, customer)
        from .survey import reusable, run_survey
        survey = reusable(goal.get("plan"), goal.get("directive") or "")
        budget = int(settings.goal_survey_max_tool_calls or 0)
        async with tenant_background(customer, store):
            client = llm or get_llm_client()
            if survey is None and budget > 0:
                # Q138 A (2026-10-27): research the sizing questions first.
                survey = await run_survey(store, customer, goal, llm=client, budget=budget,
                                          wall=float(settings.goal_survey_wall_seconds or 300.0), encoder=encoder,
                                          fact_store=fact_store, run_id=run_id)
            raw = await metered_call(client, customer_id=customer.id, origin=PLAN_ORIGIN,
                                     system=planner_system(min_minutes, statuses, limits),
                                     user=planner_user(goal, connections, survey),
                                     max_tokens=int(settings.goal_plan_max_tokens or 16000), tier=PLAN_TIER,
                                     store=store, json_schema=PLAN_SCHEMA)
            try:
                model = client._resolve_model(PLAN_TIER, None) if hasattr(client, "_resolve_model") else None
            except Exception:  # noqa: BLE001
                model = None
        plan = normalize_plan(json.loads((raw or "").strip()), min_minutes=min_minutes, previous=goal.get("plan"),
                              connections=connections, statuses=statuses, directive=goal.get("directive") or "")
        if survey:
            plan["survey"] = survey
        if not plan["deciding"] and plan.get("lattices"):
            # Q120 A: a design that breaks the data-model rules goes back for one review.
            async with tenant_background(customer, store):
                plan = await review_lattices(client, store, customer.id, goal, plan,
                                             max_tokens=int(settings.goal_plan_max_tokens or 16000), limits=limits)
    except Exception as e:  # noqa: BLE001 — a failed plan leaves the goal as it was and retries in an hour
        logger.warning("goals.plan_failed", goal_id=goal["id"], error_type=type(e).__name__, error=str(e)[:200])
        await store.mark_cognition_task_failed(run_id, error_message=f"{type(e).__name__}: {str(e)[:300]}",
                                               completed_at=datetime.now(timezone.utc))
        from datetime import timedelta
        await store.update_goal(customer.id, goal["id"], next_run_at=now + timedelta(hours=1),
                                last_summary="Couldn't write a plan just now; trying again within the hour.")
        return {"ok": False, "code": "plan_failed", "run_id": run_id}
    deciding = plan["deciding"]
    if not deciding and plan.get("lattices"):
        try:
            await ensure_plan_lattices(store, customer.id, goal, plan)
        except Exception as e:  # noqa: BLE001 — the session can still create them
            logger.warning("goals.plan_lattices_failed", goal_id=goal["id"], error=str(e)[:200])
    hold = deciding or (plan["blocked"] and not plan["run_anyway"])
    status = "deciding" if deciding else "needs_setup" if hold else None
    await store.update_goal(customer.id, goal["id"], plan=plan, plan_model=model, planned_at=now,
                            next_run_at=None if hold else now, **({"status": status} if status else {}))
    if not deciding:
        # Q137 A: the point a rollback to step 1 returns to.
        try:
            await store.add_goal_snapshot(customer.id, goal["id"], step_n=0, plan=plan, run_id=run_id)
        except Exception as e:  # noqa: BLE001 — a missing snapshot only limits rollback
            logger.warning("goals.snapshot_failed", goal_id=goal["id"], error=str(e)[:200])
    await store.mark_cognition_task_complete(run_id, result={
        "output_type": "report", "deliverable": plan["summary"], "plan": plan}, completed_at=datetime.now(timezone.utc))
    try:
        await store.record_curation_event(customer.id, event_type=("goal_deciding" if deciding else "goal_needs_setup" if hold
                                                                   else "goal_planned"),
                                          subject_id=goal["id"],
                                          label=(f"Goal needs your answers: {goal['title'][:120]}" if deciding
                                                 else f"Goal needs setup: {goal['title'][:120]}" if hold
                                                 else f"Goal planned: {goal['title'][:120]}"),
                                          payload={"goal_id": goal["id"], "run_id": run_id, "steps": len(plan["steps"]),
                                                   "check_every_minutes": plan["check_every_minutes"]})
    except Exception:  # noqa: BLE001
        logger.debug("goals.event_failed", exc_info=True)
    logger.info("goals.planned", goal_id=goal["id"], steps=len(plan["steps"]), every=plan["check_every_minutes"],
                missing=len(plan["missing"]), held=hold)
    return {"ok": True, "plan": plan, "run_id": run_id, "held": hold}
