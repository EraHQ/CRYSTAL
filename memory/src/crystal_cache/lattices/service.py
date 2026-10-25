"""Lattices, as every door changes them (2026-10-25; Q115 A, Q116 A, Q117 A).

Crystal storage holds what a person or a goal keeps in the form the data
has: a FACT for one statement ("Anthony's sister runs GLDN Girl"), a
DOCUMENT for long text, and a LATTICE for many records with the same
fields (every NBA game this season, each player's shooting). A lattice is
a named table: typed columns, the columns that identify a row, and cells
of two kinds (Q116 A, "not everyone would need a recompute"):

  snapshot   true once and then fixed: a game's date, its final score.
             A goal or a client may fill an empty snapshot cell; a value
             already there is LOCKED and only the person can correct it.
  live       expected to change: a season shot %, a standing. A new value
             replaces the old one and the old one goes to the cell's
             history (five kept). refresh_minutes says how fresh it must
             stay; stale() lists the live cells past it so a goal knows
             what to recompute.

Entity columns (a team, a player) are what the console browses by. Rows
count against the plan's lattice row cap (Q117 A), never the fact cap.
One implementation, every door: endpoints/lattices.py (console),
agent/mcp_server.py (any AI client), goals/session.py (a goal's check-in).
Every refusal carries a code. R9: no SQL here.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

COLUMN_TYPES = ("text", "number", "date", "bool", "url")
CELL_KINDS = ("snapshot", "live")
FILTER_OPS = ("eq", "contains", "gte", "lte")
MAX_COLUMNS = 40
MAX_LATTICES = 100
MAX_ROWS_PER_LATTICE = 50_000
MAX_BATCH = 500
MAX_TEXT = 2000
HISTORY_KEPT = 5
_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ][0-9:.+\-Z]{0,24})?$")
_NUM_STRIP = re.compile(r"[,\s%$]")


class LatticeRefused(Exception):
    def __init__(self, code: str, message: str, status: int = 400, detail: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.detail = detail or {}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ------------------------------------------------------------------ shape

def normalize_columns(columns: Any) -> list[dict[str, Any]]:
    if not isinstance(columns, list) or not columns:
        raise LatticeRefused("columns_required", "A lattice needs at least one column.")
    if len(columns) > MAX_COLUMNS:
        raise LatticeRefused("too_many_columns", f"A lattice holds at most {MAX_COLUMNS} columns.")
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for c in columns:
        if isinstance(c, str):
            c = {"key": c}
        if not isinstance(c, dict):
            raise LatticeRefused("bad_column", "Each column is {key, label, type, kind, entity, refresh_minutes}.")
        key = str(c.get("key") or "").strip().lower()
        if not _KEY_RE.match(key):
            raise LatticeRefused("bad_column_key",
                                 f"Column key '{key}' must be lowercase letters, digits and _ (start with a letter, 40 at most).")
        if key in seen:
            raise LatticeRefused("duplicate_column", f"Column '{key}' appears twice.")
        seen.add(key)
        typ = str(c.get("type") or "text").lower()
        if typ not in COLUMN_TYPES:
            raise LatticeRefused("bad_column_type", f"Column '{key}': type is one of {', '.join(COLUMN_TYPES)}.")
        kind = str(c.get("kind") or "snapshot").lower()
        if kind not in CELL_KINDS:
            raise LatticeRefused("bad_cell_kind", f"Column '{key}': kind is snapshot (fixed once true) or live (changes).")
        entity = bool(c.get("entity", False))
        if entity and typ != "text":
            raise LatticeRefused("bad_entity_column", f"Column '{key}': only text columns can be browsed as entities.")
        col: dict[str, Any] = {"key": key, "label": str(c.get("label") or key.replace("_", " ").capitalize()).strip()[:80],
                               "type": typ, "kind": kind, "entity": entity}
        refresh = c.get("refresh_minutes")
        if refresh not in (None, ""):
            if kind != "live":
                raise LatticeRefused("refresh_on_snapshot", f"Column '{key}': only live columns refresh.")
            try:
                refresh = int(refresh)
            except (TypeError, ValueError):
                raise LatticeRefused("bad_refresh", f"Column '{key}': refresh_minutes is a whole number.")
            if not 15 <= refresh <= 43_200:
                raise LatticeRefused("bad_refresh", f"Column '{key}': refresh_minutes is 15 to 43200 (30 days).")
            col["refresh_minutes"] = refresh
        out.append(col)
    return out


def normalize_keys(key_columns: Any, columns: list[dict[str, Any]]) -> list[str]:
    if isinstance(key_columns, str):
        key_columns = [key_columns]
    if not isinstance(key_columns, list) or not key_columns:
        raise LatticeRefused("key_columns_required",
                             "Say which columns identify a row (key_columns), e.g. ['date', 'home', 'away'].")
    by = {c["key"]: c for c in columns}
    keys = []
    for k in key_columns:
        k = str(k).strip().lower()
        if k not in by:
            raise LatticeRefused("unknown_key_column", f"key column '{k}' is not one of the columns.")
        if by[k]["kind"] != "snapshot":
            raise LatticeRefused("live_key_column", f"key column '{k}' must be a snapshot column; a row's identity never changes.")
        if k not in keys:
            keys.append(k)
    if len(keys) > 5:
        raise LatticeRefused("too_many_key_columns", "At most five key columns.")
    return keys


def coerce(col: dict[str, Any], value: Any) -> tuple[bool, Any]:
    """(ok, value) for one cell. Empty is None (a cell with nothing yet)."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return True, None
    typ = col["type"]
    if typ == "number":
        if isinstance(value, bool):
            return False, None
        if isinstance(value, (int, float)):
            n = float(value)
        else:
            try:
                n = float(_NUM_STRIP.sub("", str(value)))
            except ValueError:
                return False, None
        if n != n or n in (float("inf"), float("-inf")):
            return False, None
        return True, int(n) if n.is_integer() and abs(n) < 2**53 else n
    if typ == "bool":
        if isinstance(value, bool):
            return True, value
        s = str(value).strip().lower()
        if s in ("true", "yes", "y", "1"):
            return True, True
        if s in ("false", "no", "n", "0"):
            return True, False
        return False, None
    s = str(value).strip()
    if typ == "date":
        return (True, s[:32]) if _DATE_RE.match(s) else (False, None)
    if typ == "url":
        return (True, s[:1000]) if s.startswith(("https://", "http://")) else (False, None)
    return (True, s[:MAX_TEXT]) if len(s) <= MAX_TEXT else (False, None)


def row_key(lattice: dict[str, Any], data: dict[str, Any]) -> Optional[str]:
    parts = []
    for k in lattice["key_columns"]:
        v = data.get(k)
        if v is None:
            return None
        parts.append(str(v).strip().lower())
    return " | ".join(parts)[:512]


def _cols(lattice: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["key"]: c for c in lattice["columns"]}


# ----------------------------------------------------------- the lattices

async def _lattice(store: Any, customer_id: str, lattice_ref: str) -> dict[str, Any]:
    ref = str(lattice_ref or "").strip()
    lat = await store.get_lattice(customer_id, ref) if ref.startswith("lat_") else None
    if lat is None and ref:
        lat = await store.get_lattice_by_name(customer_id, ref)
    if lat is None:
        raise LatticeRefused("not_found", "No such lattice (give its id or exact name).", 404)
    return lat


async def get(store: Any, customer_id: str, lattice_ref: str) -> dict[str, Any]:
    return await _lattice(store, customer_id, lattice_ref)


async def create(store: Any, customer_id: str, *, name: Any, columns: Any, key_columns: Any,
                 description: Any = "", goal_id: Optional[str] = None, actor: str = "console") -> dict[str, Any]:
    name = str(name or "").strip()
    if len(name) < 2:
        raise LatticeRefused("name_required", "Name the lattice (at least 2 characters).")
    if len(name) > 120:
        raise LatticeRefused("name_too_long", "Keep the name under 120 characters.")
    description = str(description or "").strip()
    if len(description) > 2000:
        raise LatticeRefused("description_too_long", "Keep the description under 2000 characters.")
    cols = normalize_columns(columns)
    keys = normalize_keys(key_columns, cols)
    if await store.get_lattice_by_name(customer_id, name) is not None:
        raise LatticeRefused("name_taken", f"A lattice named '{name}' already exists; write to it instead.", 409)
    if await store.count_lattices(customer_id) >= MAX_LATTICES:
        raise LatticeRefused("too_many_lattices", f"An account holds at most {MAX_LATTICES} lattices.", 409)
    return await store.create_lattice(customer_id, name=name, description=description, columns=cols,
                                      key_columns=keys, goal_id=goal_id, created_by=actor)


async def update(store: Any, customer_id: str, lattice_ref: str, *, name: Any = None, description: Any = None,
                 add_columns: Any = None, column_changes: Any = None) -> dict[str, Any]:
    """Rename, describe, add columns, or change a column's label, kind,
    entity flag or refresh. A column's key and type never change (the rows
    hold values of that type); key columns stay snapshot."""
    lat = await _lattice(store, customer_id, lattice_ref)
    fields: dict[str, Any] = {}
    if name is not None:
        name = str(name).strip()
        if not 2 <= len(name) <= 120:
            raise LatticeRefused("name_required", "The name is 2 to 120 characters.")
        other = await store.get_lattice_by_name(customer_id, name)
        if other is not None and other["id"] != lat["id"]:
            raise LatticeRefused("name_taken", f"A lattice named '{name}' already exists.", 409)
        fields["name"] = name
    if description is not None:
        description = str(description).strip()
        if len(description) > 2000:
            raise LatticeRefused("description_too_long", "Keep the description under 2000 characters.")
        fields["description"] = description
    cols = [dict(c) for c in lat["columns"]]
    if column_changes:
        if not isinstance(column_changes, dict):
            raise LatticeRefused("bad_column", "column_changes is {column_key: {label, kind, entity, refresh_minutes}}.")
        by = {c["key"]: c for c in cols}
        for key, change in column_changes.items():
            if key not in by or not isinstance(change, dict):
                raise LatticeRefused("unknown_column", f"No column '{key}'.")
            extra = set(change) - {"label", "kind", "entity", "refresh_minutes"}
            if extra:
                raise LatticeRefused("bad_column", f"A column's key and type never change ({sorted(extra)}).")
            merged = {**by[key], **change}
            if change.get("kind") == "snapshot" and "refresh_minutes" not in change:
                merged.pop("refresh_minutes", None)
            by[key].clear()
            by[key].update(normalize_columns([merged])[0])
        if any(by[k]["kind"] != "snapshot" for k in lat["key_columns"]):
            raise LatticeRefused("live_key_column", "Key columns stay snapshot; a row's identity never changes.")
    if add_columns:
        new = normalize_columns(add_columns)
        clash = {c["key"] for c in new} & {c["key"] for c in cols}
        if clash:
            raise LatticeRefused("duplicate_column", f"Already a column: {sorted(clash)}.")
        cols += new
        if len(cols) > MAX_COLUMNS:
            raise LatticeRefused("too_many_columns", f"A lattice holds at most {MAX_COLUMNS} columns.")
    if cols != lat["columns"]:
        fields["columns"] = cols
    if not fields:
        return lat
    return await store.update_lattice(customer_id, lat["id"], **fields)


async def delete(store: Any, customer_id: str, lattice_ref: str) -> dict[str, Any]:
    lat = await _lattice(store, customer_id, lattice_ref)
    n = await store.delete_lattice(customer_id, lat["id"])
    return {"deleted": True, "lattice_id": lat["id"], "rows": n}


# ------------------------------------------------------------------ rows

async def _cap(store: Any, customer: Any, customer_id: str, incoming: int) -> None:
    if incoming <= 0:
        return
    tier_name = getattr(customer, "subscription_tier", None) if customer is not None else None
    if not tier_name:
        return
    from ..control.admission import GRACE_FACTOR, resolve_tier
    cap = resolve_tier(tier_name).lattice_row_cap
    if cap is None:
        return
    have = await store.count_lattice_rows(customer_id)
    if have + incoming > int(cap * GRACE_FACTOR):
        raise LatticeRefused("lattice_full",
                             f"Lattices are full ({have:,} rows; your plan holds {cap:,}). Everything stored stays "
                             "readable and exportable. Upgrade in the console, or delete rows you no longer need.",
                             402, {"rows": have, "cap": cap})


async def write_rows(store: Any, customer_id: str, lattice_ref: str, rows: Any, *, source: str = "",
                     actor: str = "console", customer: Any = None, correct: bool = False) -> dict[str, Any]:
    """Insert or update rows by their key. Snapshot cells already holding a
    value are kept unless `correct` (a person's correction); live cells take
    the new value and keep the old one in history. Returns what happened,
    plus `inverse`: what undoes it (the new rows to delete, the old state of
    the changed rows)."""
    lat = await _lattice(store, customer_id, lattice_ref)
    if not isinstance(rows, list) or not rows:
        raise LatticeRefused("rows_required", "Give rows: a list of {column: value} objects.")
    if len(rows) > MAX_BATCH:
        raise LatticeRefused("batch_too_large", f"At most {MAX_BATCH} rows per write; send the rest in another.")
    cols = _cols(lat)
    unknown = sorted({k for r in rows if isinstance(r, dict) for k in r} - set(cols))
    if unknown:
        raise LatticeRefused("unknown_columns",
                             f"Not columns of '{lat['name']}': {unknown[:10]}. Its columns: {sorted(cols)}. "
                             "Add columns first (update the lattice) or leave them out.", 400, {"unknown": unknown[:20]})
    source = str(source or "").strip()[:300]
    errors: list[dict[str, Any]] = []
    staged: dict[str, dict[str, Any]] = {}
    for i, raw in enumerate(rows):
        if not isinstance(raw, dict):
            errors.append({"row": i, "code": "bad_row", "error": "A row is an object of {column: value}."})
            continue
        data: dict[str, Any] = {}
        bad = None
        for k, v in raw.items():
            ok, val = coerce(cols[k], v)
            if not ok:
                bad = {"row": i, "column": k, "code": "bad_value",
                       "error": f"'{str(v)[:60]}' is not a {cols[k]['type']}" + (" (YYYY-MM-DD)" if cols[k]["type"] == "date" else "")}
                break
            data[k] = val
        if bad:
            errors.append(bad)
            continue
        key = row_key(lat, data)
        if key is None:
            errors.append({"row": i, "code": "key_missing", "error": f"Every row needs {lat['key_columns']}."})
            continue
        if key in staged:
            staged[key].update(data)  # a later row with the same key wins, cell by cell
        else:
            staged[key] = data
    existing = await store.get_lattice_rows_by_keys(customer_id, lat["id"], staged.keys()) if staged else {}
    inserts = [k for k in staged if k not in existing]
    if inserts:
        if lat.get("rows", 0) + len(inserts) > MAX_ROWS_PER_LATTICE:
            raise LatticeRefused("lattice_too_big",
                                 f"One lattice holds at most {MAX_ROWS_PER_LATTICE:,} rows; start another for the rest.", 409)
        await _cap(store, customer, customer_id, len(inserts))
    now = _now().isoformat()
    to_insert: list[dict[str, Any]] = []
    to_update: list[dict[str, Any]] = []
    before: list[dict[str, Any]] = []
    locked: list[dict[str, Any]] = []
    unchanged = 0
    for key, data in staged.items():
        if key not in existing:
            to_insert.append({"row_key": key, "data": data,
                              "cells": {k: {"as_of": now, "source": source} for k, v in data.items() if v is not None}})
            continue
        cur = existing[key]
        new_data, new_cells = dict(cur["data"]), {k: dict(v) for k, v in cur["cells"].items()}
        changed = False
        for k, v in data.items():
            old = new_data.get(k)
            if v is None or v == old or k in lat["key_columns"]:  # the key matched; its spelling stays
                continue
            if old is not None and cols[k]["kind"] == "snapshot" and not correct:
                if len(locked) < 20:
                    locked.append({"row_key": key, "column": k, "kept": old, "offered": v})
                continue
            meta = new_cells.get(k) or {}
            if old is not None:
                hist = [{"value": old, "as_of": meta.get("as_of"), "source": meta.get("source", "")}] + list(meta.get("history") or [])
                meta = {**meta, "history": hist[:HISTORY_KEPT]}
            meta.update({"as_of": now, "source": source})
            if correct:
                meta["corrected_by"] = actor
            new_data[k] = v
            new_cells[k] = meta
            changed = True
        if changed:
            before.append({"id": cur["id"], "data": cur["data"], "cells": cur["cells"]})
            to_update.append({"id": cur["id"], "data": new_data, "cells": new_cells})
        else:
            unchanged += 1
    new_ids: list[str] = []
    if to_insert or to_update:
        new_ids = await store.write_lattice_rows(customer_id, lat["id"], inserts=to_insert, updates=to_update)
    return {"ok": True, "lattice_id": lat["id"], "lattice": lat["name"], "inserted": len(to_insert),
            "updated": len(to_update), "unchanged": unchanged, "locked": locked, "errors": errors[:20],
            "error_count": len(errors), "row_ids": new_ids,
            "inverse": {"lattice_id": lat["id"], "delete_rows": new_ids, "restore": before}}


async def set_cell(store: Any, customer_id: str, lattice_ref: str, row_id: str, column: str, value: Any, *,
                   actor: str = "console", customer: Any = None) -> dict[str, Any]:
    """A person's edit of one cell: a correction, so snapshot cells change
    too (the old value goes to the cell's history)."""
    lat = await _lattice(store, customer_id, lattice_ref)
    cols = _cols(lat)
    if column not in cols:
        raise LatticeRefused("unknown_column", f"No column '{column}'.")
    if column in lat["key_columns"]:
        raise LatticeRefused("key_column_locked", "A key column identifies the row; delete the row and add it again instead.")
    found = await store.get_lattice_rows(customer_id, lat["id"], [row_id])
    if not found:
        raise LatticeRefused("row_not_found", "No such row in this lattice.", 404)
    ok, val = coerce(cols[column], value)
    if not ok:
        raise LatticeRefused("bad_value", f"That is not a {cols[column]['type']}.")
    row = found[0]
    if val is None:
        data = dict(row["data"])
        cells = {k: dict(v) for k, v in row["cells"].items()}
        data[column] = None
        cells.pop(column, None)
        await store.write_lattice_rows(customer_id, lat["id"], inserts=[], updates=[{"id": row_id, "data": data, "cells": cells}])
        return {"ok": True, "lattice_id": lat["id"], "lattice": lat["name"], "inserted": 0, "updated": 1,
                "unchanged": 0, "locked": [], "errors": [], "error_count": 0, "row_ids": [],
                "inverse": {"lattice_id": lat["id"], "delete_rows": [],
                            "restore": [{"id": row["id"], "data": row["data"], "cells": row["cells"]}]}}
    keyed = {k: row["data"].get(k) for k in lat["key_columns"]}
    out = await write_rows(store, customer_id, lat["id"], [{**keyed, column: val}], source="edited by you",
                           actor=actor, customer=customer, correct=True)
    return out


async def delete_rows(store: Any, customer_id: str, lattice_ref: str, row_ids: Any) -> dict[str, Any]:
    lat = await _lattice(store, customer_id, lattice_ref)
    if not isinstance(row_ids, list) or not row_ids or len(row_ids) > MAX_BATCH:
        raise LatticeRefused("row_ids_required", f"Give 1 to {MAX_BATCH} row ids.")
    n = await store.delete_lattice_rows(customer_id, lat["id"], [str(r) for r in row_ids])
    return {"ok": True, "lattice_id": lat["id"], "deleted": n}


def _filters(lat: dict[str, Any], filters: Any) -> list[tuple[str, str, str, Any]]:
    cols = _cols(lat)
    out: list[tuple[str, str, str, Any]] = []
    if not filters:
        return out
    if isinstance(filters, dict):
        filters = [{"column": k, "op": "eq", "value": v} for k, v in filters.items()]
    if not isinstance(filters, list):
        raise LatticeRefused("bad_filter", "filters is {column: value} or [{column, op, value}].")
    for f in filters[:10]:
        if not isinstance(f, dict):
            raise LatticeRefused("bad_filter", "Each filter is {column, op, value}.")
        col = str(f.get("column") or "").strip().lower()
        op = str(f.get("op") or "eq").lower()
        if col not in cols:
            raise LatticeRefused("unknown_column", f"No column '{col}'. Columns: {sorted(cols)}.")
        if op not in FILTER_OPS:
            raise LatticeRefused("bad_filter", f"op is one of {', '.join(FILTER_OPS)}.")
        if op == "contains":
            out.append((col, "text", op, str(f.get("value") or "")[:200]))
            continue
        ok, val = coerce(cols[col], f.get("value"))
        if not ok or val is None:
            raise LatticeRefused("bad_filter", f"'{f.get('value')}' is not a {cols[col]['type']}.")
        if op in ("gte", "lte") and cols[col]["type"] not in ("number", "date"):
            raise LatticeRefused("bad_filter", "gte / lte compare numbers and dates.")
        out.append((col, cols[col]["type"], op, val))
    return out


def stale_cells(lat: dict[str, Any], row: dict[str, Any], now: Optional[datetime] = None) -> list[str]:
    now = now or _now()
    out = []
    for c in lat["columns"]:
        if c["kind"] != "live" or not c.get("refresh_minutes"):
            continue
        as_of = _parse_iso((row["cells"].get(c["key"]) or {}).get("as_of"))
        if as_of is None or now - as_of > timedelta(minutes=int(c["refresh_minutes"])):
            out.append(c["key"])
    return out


def row_view(lat: dict[str, Any], row: dict[str, Any], *, history: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {"row_id": row["id"], "values": {c["key"]: row["data"].get(c["key"]) for c in lat["columns"]},
                           "updated_at": row["updated_at"]}
    stale = stale_cells(lat, row)
    if stale:
        out["stale"] = stale
    if history:
        out["cells"] = row["cells"]
    return out


async def query(store: Any, customer_id: str, lattice_ref: str, *, filters: Any = None, search: Any = None,
                sort: Any = None, descending: bool = False, limit: Any = 50, offset: Any = 0,
                history: bool = False) -> dict[str, Any]:
    lat = await _lattice(store, customer_id, lattice_ref)
    cols = _cols(lat)
    where = _filters(lat, filters)
    s = None
    if search:
        s = ([c for c, spec in cols.items() if spec["type"] in ("text", "url", "date")], str(search).strip()[:200])
    order = None
    if sort:
        sort = str(sort).strip().lower()
        if sort not in cols:
            raise LatticeRefused("unknown_column", f"Can't sort by '{sort}'; no such column.")
        order = (sort, cols[sort]["type"], bool(descending))
    try:
        lim = max(1, min(int(limit or 50), 200))
        off = max(0, int(offset or 0))
    except (TypeError, ValueError):
        raise LatticeRefused("bad_page", "limit and offset are whole numbers.")
    rows, total = await store.query_lattice_rows(customer_id, lat["id"], where=where, search=s, sort=order,
                                                 limit=lim, offset=off)
    return {"lattice": describe(lat), "total": total, "offset": off,
            "rows": [row_view(lat, r, history=history) for r in rows]}


async def entities(store: Any, customer_id: str, lattice_ref: str, column: Optional[str] = None, *,
                   search: Optional[str] = None, limit: int = 30) -> dict[str, Any]:
    """Browse by entity: each entity column's values with their row counts
    (one column when given)."""
    lat = await _lattice(store, customer_id, lattice_ref)
    cols = [c for c in lat["columns"] if c["entity"]]
    if column:
        cols = [c for c in lat["columns"] if c["key"] == column]
        if not cols:
            raise LatticeRefused("unknown_column", f"No column '{column}'.")
    lim = max(1, min(int(limit or 30), 200))
    out = []
    for c in cols:
        vals = await store.lattice_column_values(customer_id, lat["id"], c["key"], search=search, limit=lim)
        out.append({"column": c["key"], "label": c["label"], "values": [{"value": v, "rows": n} for v, n in vals]})
    return {"lattice_id": lat["id"], "name": lat["name"], "entities": out}


async def stale(store: Any, customer_id: str, lattice_ref: str, limit: int = 50) -> dict[str, Any]:
    """Live cells past their refresh window, oldest rows first: what a goal
    should recompute."""
    lat = await _lattice(store, customer_id, lattice_ref)
    live = [c for c in lat["columns"] if c["kind"] == "live" and c.get("refresh_minutes")]
    if not live:
        return {"lattice_id": lat["id"], "stale": [], "note": "No live column has a refresh window."}
    now = _now()
    out = []
    for r in await store.list_lattice_rows_oldest(customer_id, lat["id"], limit=500):
        cols = stale_cells(lat, r, now)
        if cols:
            out.append({"row_id": r["id"], "key": {k: r["data"].get(k) for k in lat["key_columns"]},
                        "stale": {k: {"value": r["data"].get(k), "as_of": (r["cells"].get(k) or {}).get("as_of")} for k in cols}})
        if len(out) >= max(1, min(int(limit or 50), 200)):
            break
    return {"lattice_id": lat["id"], "stale": out}


def describe(lat: dict[str, Any]) -> dict[str, Any]:
    return {"lattice_id": lat["id"], "name": lat["name"], "description": lat["description"],
            "rows": lat.get("rows", 0), "key_columns": lat["key_columns"],
            "columns": [{k: c[k] for k in ("key", "label", "type", "kind", "entity", "refresh_minutes") if k in c}
                        for c in lat["columns"]],
            "updated_at": lat.get("updated_at")}


async def list_view(store: Any, customer_id: str) -> list[dict[str, Any]]:
    return [describe(lat) for lat in await store.list_lattices(customer_id)]


# ------------------------------------------------------------------- undo

async def undo_write(store: Any, customer_id: str, inverse: dict[str, Any]) -> None:
    """The inverse of write_rows: delete the rows it added, put back the
    rows it changed."""
    lid = inverse.get("lattice_id")
    if not lid or await store.get_lattice(customer_id, lid) is None:
        return
    if inverse.get("delete_rows"):
        await store.delete_lattice_rows(customer_id, lid, inverse["delete_rows"])
    restore = [r for r in inverse.get("restore") or [] if r.get("id")]
    if restore:
        live = {r["id"] for r in await store.get_lattice_rows(customer_id, lid, [r["id"] for r in restore])}
        ups = [{"id": r["id"], "data": r.get("data") or {}, "cells": r.get("cells") or {}} for r in restore if r["id"] in live]
        if ups:
            await store.write_lattice_rows(customer_id, lid, inserts=[], updates=ups)


async def ledger(store: Any, customer_id: str, out: dict[str, Any], *, actor: str, reason: str,
                 run_id: Optional[str] = None) -> Optional[str]:
    """Put a write that changed something on the curation ledger, undoable
    for 24 hours (the inverse rides along unless it is too large to keep,
    in which case the act is recorded without an undo)."""
    if not (out.get("inserted") or out.get("updated")):
        return None
    import json

    from ..cognition import controls
    inverse = out.get("inverse")
    undoable = len(json.dumps(inverse, default=str)) <= 1_000_000
    bits = [f"{out['inserted']} new" if out.get("inserted") else "", f"{out['updated']} changed" if out.get("updated") else ""]
    label = f"{controls.who(actor)} wrote {' and '.join(b for b in bits if b)} row{'s' if (out.get('inserted', 0) + out.get('updated', 0)) != 1 else ''} to {out.get('lattice', 'a lattice')}"
    row = await controls.record(store, customer_id, actor=actor, run_id=run_id, action="lattice_write",
                                subject_kind="lattice", subject_id=out["lattice_id"], label=label[:256],
                                reason=reason, inverse=inverse if undoable else None, undoable=undoable)
    return row["id"]


def public(out: dict[str, Any]) -> dict[str, Any]:
    """A write's result for a caller: everything but the inverse."""
    return {k: v for k, v in out.items() if k != "inverse"}


async def recall_hits(store: Any, customer_id: str, query: str, *, max_lattices: int = 3,
                      rows_each: int = 5) -> list[dict[str, Any]]:
    """Rows that answer a recall question, found the way entities are
    (deterministically, never by vector similarity): an entity value of a
    lattice ("Jayson Tatum", "Celtics") named in the question pulls that
    lattice's rows for it. Nothing to do for a tenant without lattices."""
    q = " " + re.sub(r"[^a-z0-9%.\-' ]+", " ", (query or "").lower()) + " "
    if len(q.strip()) < 3 or await store.count_lattices(customer_id) == 0:
        return []
    out: list[dict[str, Any]] = []
    for lat in await store.list_lattices(customer_id):
        if len(out) >= max_lattices:
            break
        best: Optional[tuple[str, str]] = None
        for c in lat["columns"]:
            if not c["entity"]:
                continue
            for value, _n in await store.lattice_column_values(customer_id, lat["id"], c["key"], limit=200):
                v = value.strip().lower()
                if len(v) >= 3 and f" {v} " in q and (best is None or len(v) > len(best[1])):
                    best = (c["key"], value)
        if best is None:
            continue
        rows, total = await store.query_lattice_rows(customer_id, lat["id"], where=[(best[0], "text", "eq", best[1])],
                                                     limit=rows_each)
        out.append({"lattice": lat["name"], "lattice_id": lat["id"], "matched": {best[0]: best[1]}, "total": total,
                    "rows": [row_view(lat, r) for r in rows]})
    return out
