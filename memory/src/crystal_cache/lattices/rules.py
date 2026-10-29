"""Lattice rules (Series 384, Q152, 2026-10-28).

2026-10-27 an import stamped season "2022-23" on 93 games from October
2021 and nothing objected. Anthony: the fix "has to be general and not
absolutely mandatory so we dont block a use case that doesnt need the
rules". A lattice may carry rules: SQL conditions every row must meet,
written over the row's own columns, e.g.

    home_team <> away_team
    date >= make_date(cast(left(season, 4) as int), 7, 1)
    pts IS NULL OR pts BETWEEN 0 AND 200
    team_id IN (SELECT team_id FROM nba_teams)

A rule is checked on every write and import into its lattice. A row that
breaks a "refuse" rule is not written (the write names the rule); a row
that breaks a "warn" rule is written and counted. A NULL result passes,
as a Postgres CHECK does. A rule that can't run (bad SQL, a function the
database doesn't have) never blocks a write: the write says so. A lattice
with no rules, or with its rules off, takes any row.

Checks run like the SQL tool (as the customer's read-only role on
Postgres, behind the authorizer on SQLite), so a rule can read the
customer's lattices and nothing else, and can never write.
"""
from __future__ import annotations

import re
import uuid
from datetime import date
from typing import Any, Optional

from ..infrastructure import lattice_tables as lt

MAX_RULES = 12
MAX_CHECK_CHARS = 1000
MODES = ("refuse", "warn")
# An import whose rows mostly break a refuse rule is a wrong mapping or a
# wrong constant, not a few bad rows: nothing is written.
IMPORT_REFUSE_SHARE = 0.5
_PG_TYPES = {"text": "text", "url": "text", "ref": "text", "number": "double precision", "date": "date", "bool": "boolean"}


class RuleError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def check_text(check: Any) -> str:
    """One SQL condition, or a refusal (the database is the real wall)."""
    c = str(check or "").strip().rstrip(";").strip()
    if len(c) < 3:
        raise RuleError("rule_check_required", "A rule needs a condition, e.g. home_team <> away_team.")
    if len(c) > MAX_CHECK_CHARS:
        raise RuleError("rule_check_too_long", f"Keep a rule's condition under {MAX_CHECK_CHARS} characters.")
    try:
        lt.check_sql(f"SELECT 1 WHERE ({c})")
    except lt.SqlRefused as e:
        raise RuleError("bad_rule", e.message)
    return c


def normalize(raw: Any, existing: Optional[list[dict[str, Any]]] = None) -> list[dict[str, Any]]:
    """The rules as given (label, check, mode, on), cleaned; a rule keeps
    its id when it has one."""
    if raw is None:
        return list(existing or [])
    if not isinstance(raw, list):
        raise RuleError("bad_rules", "rules is a list of {label, check, mode, on}.")
    if len(raw) > MAX_RULES:
        raise RuleError("too_many_rules", f"A lattice holds at most {MAX_RULES} rules.")
    out: list[dict[str, Any]] = []
    for r in raw:
        if not isinstance(r, dict):
            raise RuleError("bad_rules", "Each rule is {label, check, mode, on}.")
        check = check_text(r.get("check"))
        mode = str(r.get("mode") or "refuse").strip().lower()
        if mode not in MODES:
            raise RuleError("bad_rule_mode", "A rule's mode is refuse (the row isn't written) or warn (written and counted).")
        label = re.sub(r"\s+", " ", str(r.get("label") or "").strip())[:200] or check[:200]
        rid = str(r.get("id") or "").strip()[:24] or f"r_{uuid.uuid4().hex[:8]}"
        out.append({"id": rid, "label": label, "check": check, "mode": mode, "on": r.get("on") is not False})
    if len({r["id"] for r in out}) != len(out):
        raise RuleError("bad_rules", "Two rules have the same id.")
    return out


def active(lat: dict[str, Any]) -> list[dict[str, Any]]:
    return [r for r in lat.get("rules") or [] if r.get("on", True)]


def _literal(pg: bool, col_type: str, v: Any) -> str:
    if v is None:
        return f"CAST(NULL AS {_PG_TYPES[col_type]})" if pg else "NULL"
    if col_type == "bool":
        return ("TRUE" if v else "FALSE") if pg else ("1" if v else "0")
    if col_type == "number":
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")):
            return "NULL"
        return repr(f)
    s = v.isoformat() if isinstance(v, date) else str(v)
    q = "'" + s.replace("'", "''") + "'"
    return f"CAST({q} AS {_PG_TYPES[col_type]})" if pg else q


def rows_cte(pg: bool, lat: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    """WITH _rows(_i, <columns>) AS (VALUES ...): the rows as the check sees them."""
    cols = [c for c in lat["columns"]]
    names = ", ".join(['"_i"'] + ['"' + c["key"] + '"' for c in cols])
    values = ", ".join("(" + ", ".join([str(i)] + [_literal(pg, c["type"], r.get(c["key"])) for c in cols]) + ")"
                       for i, r in enumerate(rows))
    return f"WITH _rows({names}) AS (VALUES {values})"


async def failing(store: Any, customer_id: str, lat: dict[str, Any], rows: list[dict[str, Any]],
                  rules: Optional[list[dict[str, Any]]] = None) -> tuple[dict[str, list[int]], dict[str, str]]:
    """({rule id: indexes of rows that break it}, {rule id: why it couldn't run})."""
    rules = active(lat) if rules is None else rules
    if not rules or not rows:
        return {}, {}
    pg = lt.is_pg(store._lattice_engine)
    broken: dict[str, list[int]] = {}
    errors: dict[str, str] = {}
    for start in range(0, len(rows), 250):
        chunk = rows[start:start + 250]
        cte = rows_cte(pg, lat, chunk)
        for r in rules:
            if r["id"] in errors:
                continue
            q = f"{cte} SELECT _i FROM _rows WHERE NOT COALESCE(({r['check']}), TRUE)"
            try:
                hits = await store.lattice_rule_query(customer_id, q)
            except lt.SqlRefused as e:
                errors[r["id"]] = e.message
                continue
            if hits:
                broken.setdefault(r["id"], []).extend(start + int(h[0]) for h in hits)
    return broken, errors


async def try_rule(store: Any, customer_id: str, lat: dict[str, Any], rule: dict[str, Any]) -> None:
    """A new rule must run: checked against one empty row."""
    pg = lt.is_pg(store._lattice_engine)
    q = f"{rows_cte(pg, lat, [{}])} SELECT _i FROM _rows WHERE NOT COALESCE(({rule['check']}), TRUE)"
    try:
        await store.lattice_rule_query(customer_id, q)
    except lt.SqlRefused as e:
        raise RuleError("bad_rule", f"'{rule['label']}' doesn't run: {e.message}")


async def violations(store: Any, customer_id: str, lat: dict[str, Any], rule: dict[str, Any], *,
                     sample: int = 5) -> dict[str, Any]:
    """How many rows already in the lattice break a rule, with examples of
    their keys. Nothing is changed."""
    ref = (f"{lt.GENERAL_SCHEMA}.{lat['sql_name']}" if lat.get("scope") == "general" else lat["sql_name"])
    keys = ", ".join(lat["key_columns"])
    try:
        n = await store.lattice_sql(customer_id, f"SELECT count(*) FROM {ref} WHERE NOT COALESCE(({rule['check']}), TRUE)")
        ex = await store.lattice_sql(customer_id, f"SELECT {keys} FROM {ref} WHERE NOT COALESCE(({rule['check']}), TRUE) "
                                                   f"LIMIT {int(sample)}")
    except lt.SqlRefused as e:
        return {"rows": None, "error": e.message, "examples": []}
    return {"rows": int(n["rows"][0][0] or 0),
            "examples": [" · ".join(str(v) for v in r) for r in ex["rows"]]}
