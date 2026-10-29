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

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

# ref (Series 378, Q144 A): the key value of a row in another lattice (a
# player_id pointing at the players lattice); a foreign key in the
# database, so a name lives once, in the lattice that owns it.
COLUMN_TYPES = ("text", "number", "date", "bool", "url", "ref")
CELL_KINDS = ("snapshot", "live")
FILTER_OPS = ("eq", "contains", "gte", "lte")
MAX_COLUMNS = 40
MAX_LATTICES = 100
# Q133 A (2026-10-26): was 50,000, which made a planner split one kind of
# record by season. Measured on Postgres at 250,000 rows with the facets
# indexes (migration c5e7a9b1d3f5). The plan's total row cap still applies.
MAX_ROWS_PER_LATTICE = 250_000
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
        if entity and typ not in ("text", "ref"):
            raise LatticeRefused("bad_entity_column", f"Column '{key}': only text and ref columns can be browsed as entities.")
        col: dict[str, Any] = {"key": key, "label": str(c.get("label") or key.replace("_", " ").capitalize()).strip()[:80],
                               "type": typ, "kind": kind, "entity": entity}
        if typ == "ref":
            target = str(c.get("ref") or "").strip()
            if not target:
                raise LatticeRefused("ref_target_required",
                                     f"Column '{key}': a ref column names the lattice it points at (ref: players).")
            col["ref"] = target[:120]
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


async def _owned(store: Any, customer_id: str, lattice_ref: str) -> dict[str, Any]:
    """A lattice this customer may change: its own (Q145: a general
    lattice's owner is its only writer)."""
    lat = await _lattice(store, customer_id, lattice_ref)
    if lat["customer_id"] != customer_id:
        raise LatticeRefused("general_read_only", f"'{lat['name']}' is a general lattice you read; only its owner "
                             "changes it. Keep your own rows in a lattice of yours that points at it (a ref column).", 403)
    return lat


async def get(store: Any, customer_id: str, lattice_ref: str) -> dict[str, Any]:
    return await _lattice(store, customer_id, lattice_ref)


async def create(store: Any, customer_id: str, *, name: Any, columns: Any, key_columns: Any,
                 description: Any = "", goal_id: Optional[str] = None, actor: str = "console",
                 rules: Any = None) -> dict[str, Any]:
    name = str(name or "").strip()
    if len(name) < 2:
        raise LatticeRefused("name_required", "Name the lattice (at least 2 characters).")
    if len(name) > 120:
        raise LatticeRefused("name_too_long", "Keep the name under 120 characters.")
    description = str(description or "").strip()
    if len(description) > 2000:
        raise LatticeRefused("description_too_long", "Keep the description under 2000 characters.")
    cols = await _resolve_refs(store, customer_id, normalize_columns(columns))
    keys = normalize_keys(key_columns, cols)
    if await store.get_lattice_by_name(customer_id, name) is not None:
        raise LatticeRefused("name_taken", f"A lattice named '{name}' already exists; write to it instead.", 409)
    if await store.count_lattices(customer_id) >= MAX_LATTICES:
        raise LatticeRefused("too_many_lattices", f"An account holds at most {MAX_LATTICES} lattices.", 409)
    lat = await store.create_lattice(customer_id, name=name, description=description, columns=cols,
                                     key_columns=keys, goal_id=goal_id, created_by=actor)
    if rules:
        # Series 384: rules are optional; one that doesn't run is left out
        # and said, never a reason the lattice isn't made.
        kept, dropped = await usable_rules(store, customer_id, lat, rules)
        if kept:
            lat = await store.update_lattice(customer_id, lat["id"], rules=kept)
        if dropped:
            lat = {**lat, "rules_dropped": dropped}
    return lat


async def usable_rules(store: Any, customer_id: str, lat: dict[str, Any], raw: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Each rule that is well formed and runs, and why each other one was left out."""
    from . import rules as rl
    kept: list[dict[str, Any]] = []
    dropped: list[str] = []
    for r in (raw if isinstance(raw, list) else [])[:rl.MAX_RULES]:
        try:
            one = rl.normalize([r])[0]
            await rl.try_rule(store, customer_id, lat, one)
        except rl.RuleError as e:
            dropped.append(f"{(r.get('label') if isinstance(r, dict) else '') or 'a rule'}: {e.message}"[:300])
            continue
        kept.append(one)
    return kept, dropped


async def _resolve_refs(store: Any, customer_id: str, cols: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A ref column's target by name or id becomes the target's id; the
    target is keyed on one column (the key a foreign key points at)."""
    for c in cols:
        if c["type"] != "ref":
            continue
        try:
            target = await _lattice(store, customer_id, c["ref"])
        except LatticeRefused:
            raise LatticeRefused("ref_target_missing", f"Column '{c['key']}' points at '{c['ref']}', which isn't a lattice "
                                 "yet. Create the lattice it points at first.", 409)
        if len(target["key_columns"]) != 1:
            raise LatticeRefused("ref_target_compound", f"'{target['name']}' is keyed on {target['key_columns']}; a ref "
                                 "points at a lattice keyed on one id column.")
        c["ref"] = target["id"]
    return cols


async def update(store: Any, customer_id: str, lattice_ref: str, *, name: Any = None, description: Any = None,
                 add_columns: Any = None, column_changes: Any = None, rules: Any = None,
                 add_rules: Any = None) -> dict[str, Any]:
    """Rename, describe, add columns, or change a column's label, kind,
    entity flag or refresh. A column's key and type never change (the rows
    hold values of that type); key columns stay snapshot."""
    lat = await _owned(store, customer_id, lattice_ref)
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
        new = await _resolve_refs(store, customer_id, normalize_columns(add_columns))
        clash = {c["key"] for c in new} & {c["key"] for c in cols}
        if clash:
            raise LatticeRefused("duplicate_column", f"Already a column: {sorted(clash)}.")
        cols += new
        if len(cols) > MAX_COLUMNS:
            raise LatticeRefused("too_many_columns", f"A lattice holds at most {MAX_COLUMNS} columns.")
    if cols != lat["columns"]:
        fields["columns"] = cols
    if rules is not None or add_rules:
        # Series 384: `rules` replaces the list (the person, in the console);
        # `add_rules` only adds (an agent may add a rule, never turn one off).
        from . import rules as rl
        try:
            new = rl.normalize(rules, lat.get("rules")) if rules is not None else list(lat.get("rules") or [])
            if add_rules:
                new = new + rl.normalize(add_rules)
                if len(new) > rl.MAX_RULES:
                    raise rl.RuleError("too_many_rules", f"A lattice holds at most {rl.MAX_RULES} rules.")
            old = {(r["id"], r["check"]) for r in lat.get("rules") or []}
            probe = {**lat, "columns": cols}
            for r in new:
                if (r["id"], r["check"]) not in old:
                    await rl.try_rule(store, customer_id, probe, r)
        except rl.RuleError as e:
            raise LatticeRefused(e.code, e.message)
        if new != lat.get("rules"):
            fields["rules"] = new
    if not fields:
        return lat
    return await store.update_lattice(customer_id, lat["id"], **fields)


async def rules_view(store: Any, customer_id: str, lattice_ref: str) -> dict[str, Any]:
    """Each rule with how many rows already in the lattice break it."""
    from . import rules as rl
    lat = await _lattice(store, customer_id, lattice_ref)
    out = []
    for r in lat.get("rules") or []:
        out.append({**r, "breaking": await rl.violations(store, customer_id, lat, r)})
    return {"lattice_id": lat["id"], "name": lat["name"], "rules": out, "yours": lat["customer_id"] == customer_id}


async def delete(store: Any, customer_id: str, lattice_ref: str) -> dict[str, Any]:
    lat = await _owned(store, customer_id, lattice_ref)
    users = await store.lattices_referencing(customer_id, lat["id"])
    if users:
        raise LatticeRefused("lattice_referenced", f"'{lat['name']}' is pointed at by {', '.join(users)}; delete or change "
                             "those first.", 409, {"referenced_by": users})
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
    lat = await _owned(store, customer_id, lattice_ref)
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
    checked = await _apply_rules(store, customer_id, lat, staged, existing, errors)
    stubs = await _stub_refs(store, customer_id, lat, staged, customer)
    inserts = [k for k in staged if k not in existing]
    if inserts:
        if lat.get("rows", 0) + len(inserts) > MAX_ROWS_PER_LATTICE:
            raise LatticeRefused("lattice_too_big",
                                 f"One lattice holds at most {MAX_ROWS_PER_LATTICE:,} rows. Keep less in it (fewer seasons or "
                                 "years) rather than splitting one kind of record across lattices.", 409)
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
            **({"stubs": {s["lattice"]: len(s["ids"]) for s in stubs}} if stubs else {}),
            **({"rules": checked} if checked else {}),
            "inverse": {"lattice_id": lat["id"], "delete_rows": new_ids, "restore": before,
                        "stubs": [{"lattice_id": s["lattice_id"], "ids": s["ids"]} for s in stubs]}}


async def _apply_rules(store: Any, customer_id: str, lat: dict[str, Any], staged: dict[str, dict[str, Any]],
                       existing: dict[str, Any], errors: list[dict[str, Any]]) -> dict[str, Any]:
    """The lattice's rules over the rows as they would be after this write
    (Series 384). Rows breaking a refuse rule leave `staged` with an error
    naming the rule; warn rules only count. Empty when the lattice has no
    rules on."""
    from . import rules as rl
    on = rl.active(lat)
    if not on or not staged:
        return {}
    keys = list(staged)
    merged = []
    for k in keys:
        cur = dict((existing.get(k) or {}).get("data") or {})
        cur.update({c: v for c, v in staged[k].items() if v is not None})
        merged.append(cur)
    broken, not_run = await rl.failing(store, customer_id, lat, merged, on)
    by_id = {r["id"]: r for r in on}
    out: dict[str, Any] = {"refused": {}, "warned": {}, "not_run": {}}
    gone: set[str] = set()
    for rid, idx in broken.items():
        r = by_id[rid]
        if r["mode"] == "refuse":
            out["refused"][r["label"]] = len(idx)
            for i in idx:
                if keys[i] not in gone:
                    gone.add(keys[i])
                    errors.append({"row_key": keys[i], "code": "rule_failed", "rule": r["label"],
                                   "error": f"Breaks the lattice's rule '{r['label']}' ({r['check'][:200]})."})
        else:
            out["warned"][r["label"]] = len(idx)
    for rid, why in not_run.items():
        out["not_run"][by_id[rid]["label"]] = why[:200]
    for k in gone:
        staged.pop(k, None)
    return {k: v for k, v in out.items() if v}


async def _stub_refs(store: Any, customer_id: str, lat: dict[str, Any], staged: dict[str, dict[str, Any]],
                     customer: Any) -> list[dict[str, Any]]:
    """Ref values the target lattice doesn't hold yet get a row there with
    only its key (a player seen in a box score before the players list):
    the reference holds, and the target's coverage criteria show what is
    still unnamed. Each value takes the target's own spelling."""
    out = []
    for c in lat["columns"]:
        if c["type"] != "ref":
            continue
        target = await store.get_lattice(customer_id, c["ref"])
        if target is None:
            continue
        key_col = target["key_columns"][0]
        vals = {str(d[c["key"]]).strip(): None for d in staged.values() if d.get(c["key"]) is not None}
        if not vals:
            continue
        have = await store.get_lattice_rows_by_keys(customer_id, target["id"], [v.lower() for v in vals])
        missing = []
        for v in vals:
            hit = have.get(v.lower())
            if hit is not None:
                vals[v] = hit["data"].get(key_col)
            else:
                missing.append(v)
        for d in staged.values():  # the target's spelling, so the foreign key matches
            if d.get(c["key"]) is not None:
                d[c["key"]] = vals.get(str(d[c["key"]]).strip()) or str(d[c["key"]]).strip()
        if missing and target["customer_id"] != customer_id:
            raise LatticeRefused("unknown_reference", f"{len(missing)} value(s) of '{c['key']}' are not in the general "
                                 f"lattice '{target['name']}' it points at (only its owner adds to it): "
                                 f"{', '.join(missing[:5])}.", 400, {"missing": missing[:20]})
        if missing:
            await _cap(store, customer, customer_id, len(missing))
            now = _now().isoformat()
            ids = await store.write_lattice_rows(customer_id, target["id"], inserts=[
                {"row_key": v.lower()[:512], "data": {key_col: v},
                 "cells": {key_col: {"as_of": now, "source": f"referenced by {lat['name']}"}}} for v in missing], updates=[])
            out.append({"lattice_id": target["id"], "lattice": target["name"], "ids": ids})
    return out


async def set_cell(store: Any, customer_id: str, lattice_ref: str, row_id: str, column: str, value: Any, *,
                   actor: str = "console", customer: Any = None) -> dict[str, Any]:
    """A person's edit of one cell: a correction, so snapshot cells change
    too (the old value goes to the cell's history)."""
    lat = await _owned(store, customer_id, lattice_ref)
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
    lat = await _owned(store, customer_id, lattice_ref)
    if not isinstance(row_ids, list) or not row_ids or len(row_ids) > MAX_BATCH:
        raise LatticeRefused("row_ids_required", f"Give 1 to {MAX_BATCH} row ids.")
    try:
        n = await store.delete_lattice_rows(customer_id, lat["id"], [str(r) for r in row_ids])
    except Exception as e:  # noqa: BLE001 — a foreign key refused it (Postgres)
        if "foreign key" in str(e).lower() or "violates" in str(e).lower():
            raise LatticeRefused("rows_referenced", "Other lattices point at these rows; delete those rows first.", 409)
        raise
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
    views = [row_view(lat, r, history=history) for r in rows]
    await _label(store, customer_id, lat, views)
    return {"lattice": describe(lat), "total": total, "offset": off, "rows": views}


def _name_column(lat: dict[str, Any]) -> Optional[str]:
    keys = [c["key"] for c in lat["columns"] if c["type"] == "text" and c["key"] not in lat["key_columns"]]
    return next((k for k in keys if k == "name" or k.endswith("_name")), None)


async def _label(store: Any, customer_id: str, lat: dict[str, Any], views: list[dict[str, Any]]) -> None:
    """Each ref value's name from the lattice it points at (browsing shows
    names; they are stored once, there)."""
    for c in lat["columns"]:
        if c["type"] != "ref" or not views:
            continue
        target = await store.get_lattice(customer_id, c["ref"])
        name_col = _name_column(target) if target else None
        if not name_col:
            continue
        wanted = {str(v["values"].get(c["key"])).strip().lower() for v in views if v["values"].get(c["key"]) is not None}
        found = await store.get_lattice_rows_by_keys(customer_id, target["id"], wanted)
        for v in views:
            hit = found.get(str(v["values"].get(c["key"])).strip().lower())
            if hit and hit["data"].get(name_col):
                v.setdefault("labels", {})[c["key"]] = hit["data"][name_col]


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
        # The browse lists group every row (1.7 s at 250,000 rows); they only
        # change when the lattice is written, which moves updated_at.
        ck = (lat["id"], lat.get("updated_at"), lat.get("rows"), c["key"], (search or "").lower(), lim)
        vals = _ENTITY_CACHE.get(ck)
        if vals is None:
            vals = await store.lattice_column_values(customer_id, lat["id"], c["key"], search=search, limit=lim)
            if len(_ENTITY_CACHE) >= 256:
                _ENTITY_CACHE.pop(next(iter(_ENTITY_CACHE)))
            _ENTITY_CACHE[ck] = vals
        out.append({"column": c["key"], "label": c["label"], "values": [{"value": v, "rows": n} for v, n in vals]})
    return {"lattice_id": lat["id"], "name": lat["name"], "entities": out}


_ENTITY_CACHE: dict[tuple, list[tuple[str, int]]] = {}


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
    general = lat.get("scope") == "general"
    return {"lattice_id": lat["id"], "name": lat["name"],
            "sql_name": (f"general.{lat['sql_name']}" if general and lat.get("sql_name") else lat.get("sql_name")),
            "scope": lat.get("scope", "private"),
            "description": lat["description"], "rows": lat.get("rows", 0), "key_columns": lat["key_columns"],
            "columns": [{k: c[k] for k in ("key", "label", "type", "kind", "entity", "refresh_minutes", "ref") if k in c}
                        for c in lat["columns"]],
            "rules": [{k: r.get(k) for k in ("id", "label", "check", "mode", "on")} for r in lat.get("rules") or []],
            "updated_at": lat.get("updated_at")}


async def list_view(store: Any, customer_id: str) -> list[dict[str, Any]]:
    return [{**describe(lat), "yours": lat["customer_id"] == customer_id} for lat in await store.list_lattices(customer_id)]


# ------------------------------------------------------------------- undo

async def undo_write(store: Any, customer_id: str, inverse: dict[str, Any]) -> None:
    """The inverse of write_rows: delete the rows it added, put back the
    rows it changed."""
    lid = inverse.get("lattice_id")
    if not lid or await store.get_lattice(customer_id, lid) is None:
        return
    if inverse.get("delete_rows"):
        await store.delete_lattice_rows(customer_id, lid, inverse["delete_rows"])
    for st in inverse.get("stubs") or []:  # after the rows that pointed at them
        if st.get("ids"):
            try:
                await store.delete_lattice_rows(customer_id, st["lattice_id"], st["ids"])
            except Exception:  # noqa: BLE001 — another row points at the stub now; it stays
                pass
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


# -------------------------------------------------------------- analytics
# Q121 A (2026-10-25): group-by aggregates on every door, and CSV export.

AGG_FNS = ("count", "sum", "avg", "min", "max")
MAX_GROUP_BY = 3
MAX_METRICS = 8
EXPORT_MAX_ROWS = MAX_ROWS_PER_LATTICE


def _metrics(lat: dict[str, Any], metrics: Any) -> list[tuple[str, str]]:
    cols = _cols(lat)
    if not metrics:
        return []
    if isinstance(metrics, (str, dict)):
        metrics = [metrics]
    if not isinstance(metrics, list) or len(metrics) > MAX_METRICS:
        raise LatticeRefused("bad_metrics", f"metrics is a list of up to {MAX_METRICS} {{fn, column}}.")
    out: list[tuple[str, str]] = []
    for m in metrics:
        if isinstance(m, str):  # "avg:fg_pct" or "count"
            fn, _, col = m.partition(":")
        elif isinstance(m, dict):
            fn, col = str(m.get("fn") or ""), str(m.get("column") or "")
        else:
            raise LatticeRefused("bad_metrics", "Each metric is {fn, column}.")
        fn, col = fn.strip().lower(), col.strip().lower()
        if fn not in AGG_FNS:
            raise LatticeRefused("bad_metrics", f"fn is one of {', '.join(AGG_FNS)}.")
        if fn == "count":
            continue  # every group carries its row count
        if col not in cols:
            raise LatticeRefused("unknown_column", f"No column '{col}'.")
        if cols[col]["type"] != "number":
            raise LatticeRefused("bad_metrics", f"'{col}' is {cols[col]['type']}; {fn} needs a number column.")
        if (fn, col) not in out:
            out.append((fn, col))
    return out


async def aggregate(store: Any, customer_id: str, lattice_ref: str, *, group_by: Any = None, metrics: Any = None,
                    filters: Any = None, search: Any = None, sort: Any = None, descending: bool = True,
                    limit: Any = 50) -> dict[str, Any]:
    """Group rows by up to three columns and compute count, sum, avg, min,
    max of number columns. sort: "count" or a metric label like
    "avg_fg_pct" (default: count, highest first). No group_by: one row of
    totals over every matching row."""
    lat = await _lattice(store, customer_id, lattice_ref)
    cols = _cols(lat)
    if isinstance(group_by, str):
        group_by = [g for g in group_by.split(",") if g.strip()]
    group_by = [str(g).strip().lower() for g in (group_by or [])]
    if len(group_by) > MAX_GROUP_BY:
        raise LatticeRefused("bad_group_by", f"Group by at most {MAX_GROUP_BY} columns.")
    for g in group_by:
        if g not in cols:
            raise LatticeRefused("unknown_column", f"No column '{g}'. Columns: {sorted(cols)}.")
    mets = _metrics(lat, metrics)
    labels = ["count"] + [f"{fn}_{c}" for fn, c in mets]
    order = (0, bool(descending))
    if sort:
        s = str(sort).strip().lower()
        if s not in labels:
            raise LatticeRefused("bad_sort", f"Sort by one of {labels}.")
        order = (labels.index(s), bool(descending))
    where = _filters(lat, filters)
    srch = None
    if search:
        srch = ([c for c, spec in cols.items() if spec["type"] in ("text", "url", "date")], str(search).strip()[:200])
    try:
        lim = max(1, min(int(limit or 50), 200))
    except (TypeError, ValueError):
        raise LatticeRefused("bad_page", "limit is a whole number.")
    groups, total = await store.aggregate_lattice_rows(customer_id, lat["id"], group_by=group_by, metrics=mets,
                                                       where=where, search=srch, order=order, limit=lim)
    rows = []
    for g in groups:
        row: dict[str, Any] = {k: v for k, v in zip(group_by, g["key"])}
        row["count"] = g["count"]
        for label, v in zip(labels[1:], g["values"]):
            row[label] = None if v is None else round(v, 4)
        rows.append(row)
    return {"lattice": lat["name"], "lattice_id": lat["id"], "group_by": group_by, "metrics": labels,
            "groups": rows, "total_groups": total}


async def export_rows(store: Any, customer_id: str, lattice_ref: str, *, filters: Any = None, search: Any = None):
    """(lattice, async iterator of CSV text chunks): every matching row,
    columns by key, in the order rows were added; up to the per-lattice
    row limit."""
    import csv
    import io

    lat = await _lattice(store, customer_id, lattice_ref)
    cols = _cols(lat)
    where = _filters(lat, filters)
    srch = None
    if search:
        srch = ([c for c, spec in cols.items() if spec["type"] in ("text", "url", "date")], str(search).strip()[:200])
    keys = [c["key"] for c in lat["columns"]]

    def line(values: list[Any]) -> str:
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerow(values)
        return buf.getvalue()

    async def chunks():
        yield line(keys)
        sent = 0
        async for rows in store.iter_lattice_rows(customer_id, lat["id"], where=where, search=srch):
            rows = rows[:EXPORT_MAX_ROWS - sent]
            yield "".join(line(["" if r["data"].get(k) is None else r["data"].get(k) for k in keys]) for r in rows)
            sent += len(rows)
            if sent >= EXPORT_MAX_ROWS:
                break
    return lat, chunks()


# ------------------------------------------------------------ table import
# Q124 A (2026-10-25): the server moves a web table into a lattice. The
# caller says which page, which table and how its columns map; no row is
# copied by a model.

MAX_IMPORT_ROWS = 2000


def _source_spec(spec: Any) -> dict[str, Any]:
    if isinstance(spec, str):
        return {"from": spec, "link": False, "pattern": None}
    if isinstance(spec, dict) and str(spec.get("from") or "").strip():
        pattern = spec.get("pattern")
        if pattern is not None:
            pattern = str(pattern)[:200]
            try:
                re.compile(pattern)
            except re.error:
                raise LatticeRefused("bad_pattern", f"'{pattern}' is not a valid pattern.")
        return {"from": str(spec["from"]).strip(), "link": bool(spec.get("link")), "pattern": pattern}
    raise LatticeRefused("bad_columns", "Each mapping is a table header, or {from: header, link: true, pattern}.")


def page_key(url: Any) -> str:
    """A page's identity for "already imported": scheme and host
    lowercased, no fragment, no trailing slash."""
    from urllib.parse import urlsplit, urlunsplit
    try:
        p = urlsplit(str(url or "").strip())
    except ValueError:
        return str(url or "").strip()
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path.rstrip("/") or "/", p.query, ""))


async def imported_from(store: Any, customer_id: str, lattice_id: str, url: str) -> Optional[dict[str, Any]]:
    want = page_key(url)
    for s in await store.lattice_sources(customer_id, lattice_id, limit=2000):
        if page_key(s["source"]) == want:
            return s
    return None


def _already(lat: dict[str, Any], hit: dict[str, Any]) -> LatticeRefused:
    return LatticeRefused("page_already_imported", (
        f"{hit['rows']:,} rows of '{lat['name']}' already came from this page (last written {hit['at'] or 'earlier'}). "
        "Import a page that isn't in it yet (the lattice's pages are listed where the step stands), or pass "
        "refresh: true to read this page again for rows added or changed since."), 409,
        {"rows": hit["rows"], "at": hit["at"], "source": hit["source"]})


async def import_table(store: Any, customer_id: str, lattice_ref: str, *, url: str, table: Any, columns: Any,
                       constants: Any = None, offset: int = 0, actor: str = "console", customer: Any = None,
                       reason: str = "", run_id: Optional[str] = None, loader: Any = None,
                       refresh: bool = False) -> dict[str, Any]:
    """Read one table on a page and write its rows into a lattice.
    columns: {lattice column: table header | {from: header, link: true
    (the cell's link instead of its text), pattern: regex (its first
    group)}}; constants: {lattice column: value} for every row (a season,
    a league). Dates in common formats become YYYY-MM-DD. Up to 2,000 rows
    per call from offset; the rest come with the next offset. Each batch
    of 500 is one write on the ledger."""
    from ..search.tables import TablesError, find_table, load_tables, pick, to_iso_date

    lat = await _owned(store, customer_id, lattice_ref)
    cols = _cols(lat)
    if not isinstance(columns, dict) or not columns:
        raise LatticeRefused("columns_required", "Map lattice columns to the table's headers: {column: header}.")
    specs = {str(k).strip().lower(): _source_spec(v) for k, v in columns.items()}
    consts = {str(k).strip().lower(): v for k, v in (constants or {}).items()} if isinstance(constants, dict) else {}
    unknown = sorted((set(specs) | set(consts)) - set(cols))
    if unknown:
        raise LatticeRefused("unknown_columns", f"Not columns of '{lat['name']}': {unknown[:10]}.", 400, {"unknown": unknown[:20]})
    missing_keys = [k for k in lat["key_columns"] if k not in specs and k not in consts]
    if missing_keys:
        raise LatticeRefused("key_columns_unmapped", f"Map the key columns too: {missing_keys}.")
    # Series 383 (Q150 A, 2026-10-27): two check-ins spent 80 calls
    # re-importing month pages already in nba_games (949 updates, 0 new
    # rows) while two April pages were never read. A page already in the
    # lattice is read again only on purpose.
    first = not offset and not refresh
    if first and (hit := await imported_from(store, customer_id, lat["id"], url)):
        raise _already(lat, hit)
    try:
        final, tables = await (loader or load_tables)(url)
        t = find_table(tables, table)
    except TablesError as e:
        raise LatticeRefused(e.code, e.message)
    if first and page_key(final) != page_key(url) and (hit := await imported_from(store, customer_id, lat["id"], final)):
        raise _already(lat, hit)
    lower = {h.lower(): i for i, h in enumerate(t["headers"])}
    index: dict[str, int] = {}
    for col, spec in specs.items():
        i = lower.get(spec["from"].lower())
        if i is None:
            raise LatticeRefused("no_such_header", f"The table has no header '{spec['from']}'. Its headers: {t['headers']}.")
        index[col] = i
    start = max(0, int(offset or 0))
    chunk = t["rows"][start:start + MAX_IMPORT_ROWS]
    rows: list[dict[str, Any]] = []
    for r in chunk:
        row: dict[str, Any] = dict(consts)
        for col, spec in specs.items():
            v = pick(r[index[col]], link=spec["link"], pattern=spec["pattern"])
            if v is not None and cols[col]["type"] == "date":
                v = to_iso_date(v) or v
            row[col] = v
        rows.append(row)
    total = {"ok": True, "lattice_id": lat["id"], "lattice": lat["name"], "source": final, "table": t["index"],
             "rows_read": len(chunk), "inserted": 0, "updated": 0, "unchanged": 0, "locked": [], "errors": [],
             "error_count": 0, "action_ids": []}
    # Series 384 (Q152): what the import is about to write, in one line per
    # column (dates and numbers as their range), beside any constants, so a
    # constant that doesn't fit the page shows in the result ...
    clean = []
    for r in rows:
        d = {}
        for k, v in r.items():
            ok, val = coerce(cols[k], v)
            if ok:
                d[k] = val
        clean.append(d)
    spans: dict[str, Any] = {}
    for k, c in cols.items():
        vals = [d[k] for d in clean if d.get(k) is not None]
        if vals and c["type"] in ("date", "number"):
            spans[k] = [str(min(vals)), str(max(vals))]
    if consts:
        total["constants"] = consts
    if spans:
        total["spans"] = spans
    # ... and an import whose rows mostly break one of the lattice's refuse
    # rules is a wrong mapping or a wrong constant: nothing is written.
    from . import rules as rl
    refuse = [r for r in rl.active(lat) if r["mode"] == "refuse"]
    if refuse and len(clean) >= 5:
        broken, _not_run = await rl.failing(store, customer_id, lat, clean, refuse)
        for r in refuse:
            n = len(broken.get(r["id"]) or [])
            if n >= max(5, rl.IMPORT_REFUSE_SHARE * len(clean)):
                ex = clean[(broken[r["id"]])[0]]
                raise LatticeRefused("import_breaks_rule", (
                    f"{n:,} of {len(clean):,} rows from this page break the rule '{r['label']}' ({r['check'][:200]}), "
                    f"so nothing was written. Check the mapping and the constants{f' {consts}' if consts else ''} "
                    f"against the page; one row as read: {json.dumps(ex, default=str)[:300]}."), 409,
                    {"rule": r["label"], "breaking": n, "rows": len(clean), "constants": consts, "spans": spans})
    by_column: dict[str, int] = {}
    for i in range(0, len(rows), MAX_BATCH):
        out = await write_rows(store, customer_id, lat["id"], rows[i:i + MAX_BATCH], source=final[:300], actor=actor,
                               customer=customer)
        for k in ("inserted", "updated", "unchanged", "error_count"):
            total[k] += out[k]
        total["locked"] = (total["locked"] + out["locked"])[:20]
        total["errors"] = (total["errors"] + [{**e, "row": e.get("row", 0) + start + i} for e in out["errors"]])[:20]
        for e in out["errors"]:
            c = str(e.get("column") or e.get("code") or "row")
            by_column[c] = by_column.get(c, 0) + 1
        if reason:
            aid = await ledger(store, customer_id, out, actor=actor, reason=reason, run_id=run_id)
            if aid:
                total["action_ids"].append(aid)
    if by_column:
        # Series 372: one bad mapping fails every row; say which column, so
        # the next call changes that mapping instead of making a new table.
        total["error_columns"] = by_column
        if total["error_count"] >= len(rows) > 0:
            worst = max(by_column, key=by_column.get)
            total["hint"] = (f"Every row failed, mostly on '{worst}'. Fix that column's mapping (or leave it out) "
                             f"and import again into '{lat['name']}'.")
    end = start + len(chunk)
    total["next_offset"] = end if end < t["row_count"] else None
    total["row_count"] = t["row_count"]
    return total


# ------------------------------------------------------------- the SQL tool
# Series 378 (Q144 A, 2026-10-27): questions about lattices in plain SQL,
# read-only, scoped to the customer by the database itself.

async def sql(store: Any, customer_id: str, query: Any, *, limit: Any = None) -> dict[str, Any]:
    from ..infrastructure.lattice_tables import SQL_ROW_LIMIT, SqlRefused
    try:
        lim = max(1, min(int(limit or SQL_ROW_LIMIT), SQL_ROW_LIMIT))
    except (TypeError, ValueError):
        raise LatticeRefused("bad_page", "limit is a whole number.")
    try:
        return await store.lattice_sql(customer_id, str(query or ""), limit=lim)
    except SqlRefused as e:
        raise LatticeRefused(e.code, e.message)


# --------------------------------------------------------- general lattices
# Series 380 (Q145, Q146 B, 2026-10-27). A general lattice is world facts
# (every NBA game) kept once for everyone: its table moves to the shared
# general schema, customers who subscribe read it, and its owner, the one
# who made it, is the only writer. Publishing and subscribing change who
# can read what, so they are the person's acts (console), never an
# agent's (the ratified rule: the agent never rewires access control).

async def general_list(store: Any, customer_id: str) -> list[dict[str, Any]]:
    out = []
    for lat in await store.list_general_lattices(customer_id):
        d = describe(lat)
        d.update(yours=lat["yours"], subscribed=lat["subscribed"], readers=lat["subscribers"])
        out.append(d)
    return out


async def publish(store: Any, customer_id: str, lattice_ref: str, *, general: bool,
                  even_with_readers: bool = False) -> dict[str, Any]:
    lat = await _owned(store, customer_id, lattice_ref)
    want = "general" if general else "private"
    if lat.get("scope", "private") == want:
        return lat
    if general:
        private_refs = []
        for c in lat["columns"]:
            if c["type"] == "ref":
                target = await store.get_lattice(customer_id, c["ref"])
                if target is not None and target.get("scope") != "general":
                    private_refs.append(target["name"])
        if private_refs:
            raise LatticeRefused("refs_private", f"'{lat['name']}' points at {', '.join(private_refs)}, which "
                                 "only you can read. Publish those first.", 409, {"private": private_refs})
        taken = {x["sql_name"] for x in await store.list_general_lattices(customer_id) if x["id"] != lat["id"]}
        if lat.get("sql_name") in taken:
            raise LatticeRefused("general_name_taken", f"A general lattice is already called {lat['sql_name']}; "
                                 "rename yours before publishing it.", 409)
    else:
        users = await store.lattice_readers_referencing(lat["id"], customer_id)
        if users:
            raise LatticeRefused("general_referenced", f"{len(users)} other lattice(s) point at '{lat['name']}'; it "
                                 "stays general while they do.", 409, {"referencing": len(users)})
        readers = next((x["subscribers"] for x in await store.list_general_lattices(customer_id) if x["id"] == lat["id"]), 0)
        if readers and not even_with_readers:
            raise LatticeRefused("general_has_readers", f"{readers} account(s) read '{lat['name']}'; making it private "
                                 "ends their access. Confirm to go ahead.", 409, {"readers": readers})
    out = await store.set_lattice_scope(customer_id, lat["id"], want)
    try:
        await store.record_curation_event(customer_id, event_type="lattice_published" if general else "lattice_unpublished",
                                          subject_id=lat["id"], label=f"{'Published' if general else 'Made private'}: "
                                          f"{lat['name']}"[:256], payload={"lattice_id": lat["id"], "scope": want})
    except Exception:  # noqa: BLE001 — the feed is a witness, never a gate
        pass
    return out


async def subscribe(store: Any, customer_id: str, lattice_ref: str, *, on: bool) -> dict[str, Any]:
    ref = str(lattice_ref or "").strip()
    target = None
    for x in await store.list_general_lattices(customer_id):
        if ref in (x["id"], x["name"], x["sql_name"], f"general.{x['sql_name']}"):
            target = x
            break
    if target is None:
        raise LatticeRefused("not_found", "No general lattice by that id or name.", 404)
    if target["yours"]:
        raise LatticeRefused("own_lattice", "You own this lattice; you read it already.", 409)
    if on:
        await store.subscribe_lattice(customer_id, target["id"])
    else:
        mine = [x["name"] for x in await store.list_lattices(customer_id) if x["customer_id"] == customer_id
                and any(c.get("type") == "ref" and c.get("ref") == target["id"] for c in x["columns"])]
        if mine:
            raise LatticeRefused("lattice_referenced", f"{', '.join(mine)} point(s) at it; change those first.", 409)
        await store.unsubscribe_lattice(customer_id, target["id"])
    return {"ok": True, "lattice_id": target["id"], "name": target["name"], "subscribed": bool(on)}
