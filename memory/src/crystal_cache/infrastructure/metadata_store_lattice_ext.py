"""Lattices (2026-10-25; Q115 A, Q116 A, Q117 A): the SQL for lattices.

A lattice is a named table in Crystal storage. Since Series 378 (Q144 A,
2026-10-27) its rows live in a real table of their own, in a schema that
belongs to one customer (infrastructure/lattice_tables.py has the
layout). The tables, and the read-only SQL tool, use the lattice
database (store._lattice_engine: CC_LATTICE_DATABASE_URL, Q147 A,
crystal_data in production; the main database when unset); the
`lattices` row that describes each one (columns, key columns, the
physical table's name, the name the SQL tool knows it by) stays in the
main database, where goals and the console point at it. Lattices
made before Series 378 kept their rows as JSON in `lattice_rows`; the
first time one is touched its rows move into its own table (claimed with
a compare-and-set, so two workers never both move it).

Typing, locking and caps are decided in lattices/service.py; this mixin
only reads and writes, and hands every row back in one shape: {id,
row_key, data: {column: value}, cells: {column: {as_of, source,
history}}, created_at, updated_at}. R9: lattice SQL lives here and in
lattice_tables.py and nowhere else.
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from typing import Any, Iterable, Optional

import structlog
from sqlalchemy import String, and_, case, cast, delete, func, literal_column, or_, select, text, tuple_, update

from . import lattice_tables as lt
from .schema import LatticeRecordRow, LatticeRow, LatticeSubscriptionRow

logger = structlog.get_logger(__name__)

_LATTICE_FIELDS = {"name", "description", "columns", "key_columns", "rules"}
_TABLES: dict[tuple, Any] = {}
_CLEAN: set[str] = set()  # lattices whose pre-378 JSON rows are known to be gone


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if isinstance(dt, str):
        return dt
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def lattice_dict(r: LatticeRow, rows: Optional[int] = None) -> dict[str, Any]:
    out = {"id": r.id, "customer_id": r.customer_id, "name": r.name, "description": r.description or "",
           "columns": [dict(c) for c in (r.columns or [])], "key_columns": list(r.key_columns or []),
           "goal_id": r.goal_id, "created_by": r.created_by or "", "sql_name": r.sql_name,
           "scope": getattr(r, "scope", None) or "private",
           "rules": [dict(x) for x in (getattr(r, "rules", None) or [])],
           "created_at": _iso(r.created_at), "updated_at": _iso(r.updated_at)}
    if rows is not None:
        out["rows"] = int(rows)
    return out


def lattice_row_dict(r: LatticeRecordRow) -> dict[str, Any]:
    """A pre-378 JSON row (only read when a lattice moves to its table)."""
    return {"id": r.id, "lattice_id": r.lattice_id, "row_key": r.row_key, "data": dict(r.data or {}),
            "cells": {k: dict(v) for k, v in (r.cells or {}).items()},
            "created_at": _iso(r.created_at), "updated_at": _iso(r.updated_at)}


def _is_pg(store: Any) -> bool:
    # Mixin methods are bound one by one (infrastructure/__init__), so this is a function, not a property.
    return store._lattice_engine.dialect.name == "postgresql"


@asynccontextmanager
async def _data(store: Any):
    """A session on the lattice database; rolled back if the block raises."""
    async with store._lattice_session_factory() as session:
        try:
            yield session
        except BaseException:
            await session.rollback()
            raise


# ------------------------------------------------------------- the tables

def _general(lattice: dict[str, Any]) -> bool:
    return lattice.get("scope") == "general"


async def _refs(store: Any, lattice: dict[str, Any]) -> dict[str, tuple[str, str, str, bool]]:
    """Each ref column's target: the owner's own lattice, or a general one."""
    out: dict[str, tuple[str, str, str, bool]] = {}
    for c in lattice["columns"]:
        if c["type"] != "ref" or not c.get("ref"):
            continue
        async with store.session() as session:
            target = await session.get(LatticeRow, c["ref"])
        if target is None or not target.key_columns:
            continue
        general = (target.scope or "private") == "general"
        if target.customer_id == lattice["customer_id"] or general:
            out[c["key"]] = (target.id, list(target.key_columns)[0], target.customer_id, general)
    return out


async def _table(store: Any, customer_id: str, lattice: dict[str, Any]):
    """The lattice's table, built once per shape and place. Its owner and
    scope decide where it lives, whoever asks."""
    owner = lattice["customer_id"]
    ck = (owner, lattice["id"], lattice.get("scope"), json.dumps(lattice["columns"], sort_keys=True),
          store._lattice_engine.dialect.name)
    t = _TABLES.get(ck)
    if t is None:
        refs = await _refs(store, lattice)
        t = lt.build(store._lattice_engine, owner, lattice, refs)
        for key, (tid, tcol, towner, tgen) in refs.items():  # FK targets must be known to the MetaData
            stub = lt.stub_target(store._lattice_engine, towner, tid, tcol, tgen)
            if stub.key not in t.metadata.tables:
                stub.to_metadata(t.metadata)
        if len(_TABLES) > 2000:
            _TABLES.clear()
        _TABLES[ck] = t
    return t


async def _ensure_schema(store: Any, conn: Any, customer_id: str) -> None:
    if not _is_pg(store):
        return
    # Runs only when a table is made (rare), so no cache: another worker
    # may have erased the schema since.
    schema = lt.schema_name(customer_id)
    role = lt.role_name(customer_id)
    await conn.exec_driver_sql(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    await conn.exec_driver_sql(
        f"DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{role}') THEN "
        f'CREATE ROLE "{role}" NOLOGIN; END IF; EXCEPTION WHEN duplicate_object THEN NULL; END $$')
    # The app's own user takes the role only to run the SQL tool (SET ROLE).
    await conn.exec_driver_sql(f'GRANT "{role}" TO CURRENT_USER')
    await conn.exec_driver_sql(f'GRANT USAGE ON SCHEMA "{schema}" TO "{role}"')


async def _create_physical(store: Any, conn: Any, customer_id: str, lattice: dict[str, Any]) -> Any:
    await _ensure_schema(store, conn, lattice["customer_id"])
    if _general(lattice) and _is_pg(store):
        await conn.exec_driver_sql(f'CREATE SCHEMA IF NOT EXISTS "{lt.GENERAL_SCHEMA}"')
    t = await _table(store, customer_id, lattice)
    await conn.run_sync(lambda sc: t.create(sc, checkfirst=True))
    for ddl in lt.index_ddl(store._lattice_engine, t, lattice):
        await conn.exec_driver_sql(ddl)
    await _view(store, conn, customer_id, lattice, t)
    return t


async def _view(store: Any, conn: Any, customer_id: str, lattice: dict[str, Any], t: Any,
                drop: Optional[str] = None) -> None:
    """Postgres: the view the SQL tool reads, readable by the customer's
    role and nothing else. (SQLite makes its views per query.)"""
    if not _is_pg(store) or not lattice.get("sql_name"):
        return
    owner = lattice["customer_id"]
    e, role = store._lattice_engine, lt.role_name(owner)
    schema = lt.GENERAL_SCHEMA if _general(lattice) else lt.schema_name(owner)
    if drop:
        await conn.exec_driver_sql(f"DROP VIEW IF EXISTS {lt.qualified(e, schema, drop)}")
    cols = ", ".join(lt.quote(e, c) for c in lt.view_columns(lattice))
    v = lt.qualified(e, schema, lattice["sql_name"])
    await conn.exec_driver_sql(f"DROP VIEW IF EXISTS {v}")
    await conn.exec_driver_sql(f"CREATE VIEW {v} AS SELECT _id AS row_id, {cols} FROM {lt.qualified(e, t.schema, t.name)}")
    await conn.exec_driver_sql(f'GRANT SELECT ON {v} TO "{role}"')
    if _general(lattice):
        for reader in await _subscribers(store, lattice["id"]):
            await _ensure_schema(store, conn, reader)
            await conn.exec_driver_sql(f'GRANT SELECT ON {v} TO "{lt.role_name(reader)}"')


async def _materialize(store: Any, customer_id: str, lattice_id: str) -> None:
    """Move a pre-378 lattice's JSON rows into its own table, once: claim
    it in the main database, build and fill the table in the lattice
    database, then drop the JSON rows. A move cut short is finished by
    the next touch (_copy_legacy skips rows already there)."""
    async with store.session() as session:
        row = await session.get(LatticeRow, lattice_id)
        if row is None or row.table_name:
            return
        taken = {x for x in (await session.execute(select(LatticeRow.sql_name).where(
            LatticeRow.customer_id == customer_id))).scalars().all() if x}
        claim = await session.execute(update(LatticeRow).where(LatticeRow.id == lattice_id)
                                      .where(LatticeRow.table_name.is_(None))
                                      .values(table_name=lt.physical_name(lattice_id),
                                              sql_name=row.sql_name or lt.sql_name(row.name, taken)))
        await session.commit()
        if not claim.rowcount:
            return
        await session.refresh(row)
        lat = lattice_dict(row)
    await _copy_legacy(store, customer_id, lat)


async def _copy_legacy(store: Any, customer_id: str, lat: dict[str, Any]) -> None:
    lattice_id = lat["id"]
    async with store.session() as session:
        legacy = (await session.execute(select(LatticeRecordRow).where(LatticeRecordRow.lattice_id == lattice_id)
                                        .where(LatticeRecordRow.customer_id == customer_id))).scalars().all()
        legacy = [lattice_row_dict(r) for r in legacy]
    cols = {c["key"]: c for c in lat["columns"]}
    moved = 0
    async with _data(store) as ds:
        conn = await ds.connection()
        t = await _create_physical(store, conn, customer_id, lat)
        have = set((await ds.execute(select(t.c._id))).scalars().all()) if legacy else set()
        batch = []
        for r in legacy:
            if r["id"] in have:
                continue
            batch.append(_values(cols, r["id"], r["row_key"], r["data"], r["cells"],
                                 r["created_at"] or _now(), r["updated_at"] or _now(), lenient=True))
            if len(batch) >= 1000:
                await ds.execute(t.insert(), batch)
                moved += len(batch)
                batch = []
        if batch:
            await ds.execute(t.insert(), batch)
            moved += len(batch)
        await ds.commit()
    if legacy:
        async with store.session() as session:
            await session.execute(delete(LatticeRecordRow).where(LatticeRecordRow.lattice_id == lattice_id)
                                  .where(LatticeRecordRow.customer_id == customer_id))
            await session.commit()
    _CLEAN.add(lattice_id)
    logger.info("lattices.materialized", lattice_id=lattice_id, rows=moved)


def _values(cols: dict[str, dict[str, Any]], rid: str, key: str, data: dict[str, Any], cells: dict[str, Any],
            created: Any, updated: Any, *, lenient: bool = False) -> dict[str, Any]:
    if isinstance(created, str):
        created = datetime.fromisoformat(created)
    if isinstance(updated, str):
        updated = datetime.fromisoformat(updated)
    v: dict[str, Any] = {"_id": rid, "_key": key, "_cells": cells or {}, "_search": lt.search_text(data),
                         "_created_at": created, "_updated_at": updated}
    for k, c in cols.items():
        try:
            v[k] = lt.to_db(c, data.get(k))
        except (TypeError, ValueError):
            if not lenient:
                raise
            v[k] = None
    return v


def _row(lattice: dict[str, Any], r: Any) -> dict[str, Any]:
    m = r._mapping
    return {"id": m["_id"], "lattice_id": lattice["id"], "row_key": m["_key"],
            "data": {c["key"]: lt.from_db(c["type"], m[c["key"]]) for c in lattice["columns"]},
            "cells": {k: dict(v) for k, v in (m["_cells"] or {}).items()},
            "created_at": _iso(m["_created_at"]), "updated_at": _iso(m["_updated_at"])}


async def _subscribers(store: Any, lattice_id: str) -> list[str]:
    async with store.session() as session:
        return list((await session.execute(select(LatticeSubscriptionRow.customer_id)
                                            .where(LatticeSubscriptionRow.lattice_id == lattice_id))).scalars().all())


async def _subscribed(store: Any, customer_id: str) -> set[str]:
    async with store.session() as session:
        return set((await session.execute(select(LatticeSubscriptionRow.lattice_id)
                                           .where(LatticeSubscriptionRow.customer_id == customer_id))).scalars().all())


async def _open(store: Any, customer_id: str, lattice_id: str, *, write: bool = False) -> tuple[Optional[dict[str, Any]], Any]:
    """(lattice, its table), moving a pre-378 lattice first. The owner may
    read and write; a customer subscribed to a general lattice may only
    read it (Q145: its owner is the only writer)."""
    async with store.session() as session:
        row = await session.get(LatticeRow, lattice_id)
        if row is None:
            return None, None
        if row.customer_id != customer_id:
            if write or (row.scope or "private") != "general" or lattice_id not in await _subscribed(store, customer_id):
                return None, None
        ready = bool(row.table_name)
        lat = lattice_dict(row)
    if not ready:
        await _materialize(store, lat["customer_id"], lattice_id)
        async with store.session() as session:
            row = await session.get(LatticeRow, lattice_id)
            lat = lattice_dict(row)
    elif lattice_id not in _CLEAN:
        async with store.session() as session:
            left = (await session.execute(select(func.count()).select_from(LatticeRecordRow)
                                          .where(LatticeRecordRow.lattice_id == lattice_id))).scalar_one()
        if left:
            await _copy_legacy(store, lat["customer_id"], lat)  # a move cut short
        _CLEAN.add(lattice_id)
    return lat, await _table(store, customer_id, lat)


def _col(t: Any, name: str):
    return t.c[name]


def _conditions(store: Any, lattice: dict[str, Any], t: Any, where: Optional[list[tuple[str, str, str, Any]]],
                search: Optional[tuple[list[str], str]]) -> list[Any]:
    conds: list[Any] = []
    for col, typ, op, value in where or []:
        c = _col(t, col)
        if typ == "date" and op != "contains" and isinstance(value, str):
            value = date.fromisoformat(value[:10])
        if op == "eq":
            conds.append(func.lower(c) == str(value).lower() if typ in ("text", "url", "ref") else c == value)
        elif op == "contains":
            conds.append(func.lower(cast(c, String)).like(f"%{str(value).lower()}%"))
        elif op == "gte":
            conds.append(c >= value)
        elif op == "lte":
            conds.append(c <= value)
    if search and search[1]:
        needle = search[1].lower()
        if _is_pg(store):
            words = [w for w in lt.re.split(r"[^\w]+", needle) if w][:8]
            if words:
                q = " & ".join(f"{w}:*" for w in words)
                conds.append(func.to_tsvector(literal_column("'simple'::regconfig"),
                                              func.coalesce(t.c._search, literal_column("''")))
                             .op("@@")(func.to_tsquery(literal_column("'simple'::regconfig"), q)))
        else:
            conds.append(t.c._search.like(f"%{needle}%"))
    return conds


async def _drop_physical(store: Any, customer_id: str, lat: dict[str, Any], t: Any) -> int:
    """The table and its view; returns how many rows it held."""
    e = store._lattice_engine
    async with _data(store) as ds:
        n = int((await ds.execute(select(func.count()).select_from(t))).scalar_one())
        conn = await ds.connection()
        if _is_pg(store) and lat.get("sql_name"):
            vs = lt.GENERAL_SCHEMA if _general(lat) else lt.schema_name(lat["customer_id"])
            await conn.exec_driver_sql(f"DROP VIEW IF EXISTS {lt.qualified(e, vs, lat['sql_name'])}")
        # CASCADE: a subscriber's ref to a general table loses its constraint, never its rows.
        await conn.exec_driver_sql(f"DROP TABLE IF EXISTS {lt.qualified(e, t.schema, t.name)}"
                                   + (" CASCADE" if _is_pg(store) else ""))
        await ds.commit()
    return n


class LatticeExtensionsMixin:
    """lattices and their tables, bound onto MetadataStore."""

    # ------------------------------------------------------------ lattices

    async def create_lattice(self, customer_id: str, *, name: str, description: str, columns: list[dict[str, Any]],
                             key_columns: list[str], goal_id: Optional[str] = None,
                             created_by: str = "") -> dict[str, Any]:
        now = _now()
        lid = f"lat_{uuid.uuid4().hex[:16]}"
        async with self.session() as session:  # type: ignore[attr-defined]
            taken = {x for x in (await session.execute(select(LatticeRow.sql_name).where(
                LatticeRow.customer_id == customer_id))).scalars().all() if x}
        row = LatticeRow(id=lid, customer_id=customer_id, name=name[:120], description=description or "",
                         columns=list(columns), key_columns=list(key_columns), goal_id=goal_id,
                         created_by=(created_by or "")[:96], table_name=lt.physical_name(lid),
                         sql_name=lt.sql_name(name, taken), created_at=now, updated_at=now)
        lat = lattice_dict(row, 0)
        # The table first, in the lattice database; then the row that
        # describes it. Two databases, so a failed second step removes the first.
        async with _data(self) as ds:
            t = await _create_physical(self, await ds.connection(), customer_id, lat)
            await ds.commit()
        try:
            async with self.session() as session:  # type: ignore[attr-defined]
                session.add(row)
                await session.commit()
        except Exception:
            await _drop_physical(self, customer_id, lat, t)
            raise
        _CLEAN.add(lid)
        return lat

    async def get_lattice(self, customer_id: str, lattice_id: str) -> Optional[dict[str, Any]]:
        lat, t = await _open(self, customer_id, lattice_id)
        if lat is None:
            return None
        async with _data(self) as session:
            lat["rows"] = int((await session.execute(select(func.count()).select_from(t))).scalar_one())
        return lat

    async def get_lattice_by_name(self, customer_id: str, name: str) -> Optional[dict[str, Any]]:
        needle = (name or "").strip().lower()
        if needle.startswith(lt.GENERAL_SCHEMA + "."):
            needle = needle[len(lt.GENERAL_SCHEMA) + 1:]
        subscribed = await _subscribed(self, customer_id)
        async with self.session() as session:  # type: ignore[attr-defined]
            # Its own lattice first; then a general one it reads.
            rows = (await session.execute(select(LatticeRow)
                                          .where(or_(LatticeRow.customer_id == customer_id, LatticeRow.id.in_(subscribed or [""])))
                                          .where(or_(func.lower(LatticeRow.name) == needle, LatticeRow.sql_name == needle))
                                          )).scalars().all()
        rows.sort(key=lambda r: (r.customer_id != customer_id, r.name.lower() != needle))
        return await self.get_lattice(customer_id, rows[0].id) if rows else None  # type: ignore[attr-defined]

    async def list_lattices(self, customer_id: str) -> list[dict[str, Any]]:
        """Its own lattices, then the general ones it subscribes to."""
        subscribed = await _subscribed(self, customer_id)
        async with self.session() as session:  # type: ignore[attr-defined]
            ids = list((await session.execute(select(LatticeRow.id).where(LatticeRow.customer_id == customer_id)
                                              .order_by(LatticeRow.updated_at.desc()))).scalars().all())
            if subscribed:
                ids += [x for x in (await session.execute(select(LatticeRow.id).where(LatticeRow.id.in_(subscribed))
                                                          .where(LatticeRow.customer_id != customer_id)
                                                          .order_by(LatticeRow.name))).scalars().all()]
        out = []
        for lid in ids:
            lat = await self.get_lattice(customer_id, lid)  # type: ignore[attr-defined]
            if lat is not None:
                out.append(lat)
        return out

    async def count_lattices(self, customer_id: str) -> int:
        async with self.session() as session:  # type: ignore[attr-defined]
            return int((await session.execute(select(func.count()).select_from(LatticeRow)
                                              .where(LatticeRow.customer_id == customer_id))).scalar_one())

    async def update_lattice(self, customer_id: str, lattice_id: str, **fields: Any) -> Optional[dict[str, Any]]:
        """Rename (the SQL name follows), describe, or change the columns;
        columns are only ever added to the table (a key and its type never
        change)."""
        bad = set(fields) - _LATTICE_FIELDS
        if bad:
            raise ValueError(f"unknown lattice fields {sorted(bad)}")
        before, t = await _open(self, customer_id, lattice_id, write=True)
        if before is None:
            return None
        values = dict(fields)
        old_sql = before.get("sql_name")
        if "name" in fields:
            async with self.session() as session:  # type: ignore[attr-defined]
                taken = {x for x in (await session.execute(select(LatticeRow.sql_name).where(
                    LatticeRow.customer_id == customer_id).where(LatticeRow.id != lattice_id))).scalars().all() if x}
            values["sql_name"] = lt.sql_name(fields["name"], taken)
        after = {**before, **values}
        e = self._lattice_engine  # type: ignore[attr-defined]
        # The table first (new columns are only ever added), then the row.
        async with _data(self) as ds:
            conn = await ds.connection()
            have = {c["key"] for c in before["columns"]}
            for c in fields.get("columns") or []:
                if c["key"] in have:
                    continue
                typ = lt._SQL_TYPES[c["type"]]().compile(dialect=e.dialect)
                await conn.exec_driver_sql(f"ALTER TABLE {lt.qualified(e, t.schema, t.name)} "
                                           f"ADD COLUMN {lt.quote(e, c['key'])} {typ}")
            if "columns" in fields or after.get("sql_name") != old_sql:
                nt = await _table(self, customer_id, after)
                for ddl in lt.index_ddl(e, nt, after):
                    await conn.exec_driver_sql(ddl)
                await _view(self, conn, customer_id, after, nt, drop=old_sql if old_sql != after.get("sql_name") else None)
            await ds.commit()
        async with self.session() as session:  # type: ignore[attr-defined]
            await session.execute(update(LatticeRow).where(LatticeRow.id == lattice_id)
                                  .where(LatticeRow.customer_id == customer_id).values(**values, updated_at=_now()))
            await session.commit()
        return await self.get_lattice(customer_id, lattice_id)  # type: ignore[attr-defined]

    async def touch_lattice(self, customer_id: str, lattice_id: str) -> None:
        async with self.session() as session:  # type: ignore[attr-defined]
            await session.execute(update(LatticeRow).where(LatticeRow.id == lattice_id)
                                  .where(LatticeRow.customer_id == customer_id).values(updated_at=_now()))
            await session.commit()

    async def lattices_referencing(self, customer_id: str, lattice_id: str) -> list[str]:
        """Names of the customer's lattices with a ref column pointing here."""
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(select(LatticeRow).where(LatticeRow.customer_id == customer_id)
                                          .where(LatticeRow.id != lattice_id))).scalars().all()
        return [r.name for r in rows if any(c.get("type") == "ref" and c.get("ref") == lattice_id for c in r.columns or [])]

    async def delete_lattice(self, customer_id: str, lattice_id: str) -> int:
        """The lattice and its table. Returns how many rows went."""
        lat, t = await _open(self, customer_id, lattice_id, write=True)
        if lat is None:
            return 0
        n = await _drop_physical(self, customer_id, lat, t)
        async with self.session() as session:  # type: ignore[attr-defined]
            await session.execute(delete(LatticeSubscriptionRow).where(LatticeSubscriptionRow.lattice_id == lattice_id))
            await session.execute(delete(LatticeRow).where(LatticeRow.id == lattice_id)
                                  .where(LatticeRow.customer_id == customer_id))
            await session.commit()
        return n

    async def drop_lattice_storage(self, customer_id: str) -> int:
        """Erase and purge: every lattice table of the customer, and on
        Postgres the schema and the role. Returns how many tables went."""
        e = self._lattice_engine  # type: ignore[attr-defined]
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(select(LatticeRow).where(LatticeRow.customer_id == customer_id))).scalars().all()
        # The general lattices it owns go with it (subscribers keep their own
        # rows; their refs lose the constraint), and its subscriptions end.
        for r in rows:
            if (r.scope or "private") == "general" and r.table_name:
                lat = lattice_dict(r)
                await _drop_physical(self, customer_id, lat, await _table(self, customer_id, lat))
        async with self.session() as session:  # type: ignore[attr-defined]
            await session.execute(delete(LatticeSubscriptionRow).where(or_(
                LatticeSubscriptionRow.customer_id == customer_id,
                LatticeSubscriptionRow.lattice_id.in_([r.id for r in rows] or [""]))))
            await session.commit()
        async with _data(self) as ds:
            conn = await ds.connection()
            if _is_pg(self):
                schema, role = lt.schema_name(customer_id), lt.role_name(customer_id)
                await conn.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
                await conn.exec_driver_sql(
                    # DROP OWNED first: it takes back what the role was granted
                    # here (USAGE on the general schema, subscribed views).
                    f"DO $$ BEGIN IF EXISTS (SELECT FROM pg_roles WHERE rolname = '{role}') THEN "
                    f'DROP OWNED BY "{role}"; DROP ROLE "{role}"; END IF; END $$')
            else:
                for r in rows:
                    if r.table_name and (r.scope or "private") != "general":
                        _, name = lt.located(e, customer_id, r.id)
                        await conn.exec_driver_sql(f"DROP TABLE IF EXISTS {lt.quote(e, name)}")
            await ds.commit()
        return len(rows)

    # ---------------------------------------------------------------- rows

    async def count_lattice_rows(self, customer_id: str, lattice_id: Optional[str] = None) -> int:
        """Rows the customer OWNS (what its plan's row cap counts); general
        lattices it only reads count against their owner."""
        if lattice_id:
            lat = await self.get_lattice(customer_id, lattice_id)  # type: ignore[attr-defined]
            return int(lat["rows"]) if lat else 0
        return sum(int(x["rows"]) for x in await self.list_lattices(customer_id)  # type: ignore[attr-defined]
                   if x["customer_id"] == customer_id)

    async def get_lattice_rows_by_keys(self, customer_id: str, lattice_id: str,
                                       keys: Iterable[str]) -> dict[str, dict[str, Any]]:
        lat, t = await _open(self, customer_id, lattice_id)
        keys = list(dict.fromkeys(keys))
        out: dict[str, dict[str, Any]] = {}
        if lat is None:
            return out
        async with _data(self) as session:
            for i in range(0, len(keys), 400):
                for r in (await session.execute(select(t).where(t.c._key.in_(keys[i:i + 400])))).all():
                    d = _row(lat, r)
                    out[d["row_key"]] = d
        return out

    async def get_lattice_rows(self, customer_id: str, lattice_id: str, row_ids: Iterable[str]) -> list[dict[str, Any]]:
        lat, t = await _open(self, customer_id, lattice_id)
        ids = list(dict.fromkeys(row_ids))
        if lat is None or not ids:
            return []
        async with _data(self) as session:
            return [_row(lat, r) for r in (await session.execute(select(t).where(t.c._id.in_(ids)))).all()]

    async def write_lattice_rows(self, customer_id: str, lattice_id: str, *, inserts: list[dict[str, Any]],
                                 updates: list[dict[str, Any]]) -> list[str]:
        """One transaction. inserts: [{row_key, data, cells}] -> new ids,
        in order. updates: [{id, data, cells}] replace both."""
        lat, t = await _open(self, customer_id, lattice_id, write=True)
        if lat is None:
            return []
        cols = {c["key"]: c for c in lat["columns"]}
        now = _now()
        ids: list[str] = []
        async with _data(self) as session:
            if inserts:
                batch = []
                for r in inserts:
                    rid = f"lr_{uuid.uuid4().hex[:16]}"
                    ids.append(rid)
                    batch.append(_values(cols, rid, r["row_key"], r["data"], r["cells"], now, now))
                await session.execute(t.insert(), batch)
            for r in updates:
                v = _values(cols, r["id"], "", r["data"], r["cells"], now, now)
                for k in ("_id", "_key", "_created_at"):
                    v.pop(k)
                await session.execute(t.update().where(t.c._id == r["id"]).values(**v))
            await session.commit()
        await self.touch_lattice(customer_id, lattice_id)  # type: ignore[attr-defined]
        return ids

    async def delete_lattice_rows(self, customer_id: str, lattice_id: str, row_ids: Iterable[str]) -> int:
        lat, t = await _open(self, customer_id, lattice_id, write=True)
        ids = list(dict.fromkeys(row_ids))
        if lat is None or not ids:
            return 0
        async with _data(self) as session:
            res = await session.execute(t.delete().where(t.c._id.in_(ids)))
            await session.commit()
            n = int(res.rowcount or 0)
        await self.touch_lattice(customer_id, lattice_id)  # type: ignore[attr-defined]
        return n

    async def query_lattice_rows(self, customer_id: str, lattice_id: str, *,
                                 where: Optional[list[tuple[str, str, str, Any]]] = None,
                                 search: Optional[tuple[list[str], str]] = None,
                                 sort: Optional[tuple[str, str, bool]] = None,
                                 limit: int = 50, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
        """where: [(column, type, op, value)] with op eq | contains | gte |
        lte; search: (_, needle) over every text value; sort: (column,
        type, descending). Returns (rows, total matching)."""
        lat, t = await _open(self, customer_id, lattice_id)
        if lat is None:
            return [], 0
        conds = _conditions(self, lat, t, where, search)
        stmt = select(t).where(and_(true_(), *conds))
        if sort:
            c = _col(t, sort[0])
            stmt = stmt.order_by(c.desc() if sort[2] else c.asc(), t.c._created_at.asc(), t.c._key.asc())
        else:
            stmt = stmt.order_by(t.c._created_at.asc(), t.c._key.asc())  # a batch shares one timestamp
        async with _data(self) as session:
            total = int((await session.execute(select(func.count()).select_from(t).where(and_(true_(), *conds)))).scalar_one())
            rows = (await session.execute(stmt.limit(limit).offset(offset))).all()
            return [_row(lat, r) for r in rows], total

    async def iter_lattice_rows(self, customer_id: str, lattice_id: str, *,
                                where: Optional[list[tuple[str, str, str, Any]]] = None,
                                search: Optional[tuple[list[str], str]] = None, batch: int = 1000):
        """Every matching row in the order rows were added, by keyset."""
        lat, t = await _open(self, customer_id, lattice_id)
        if lat is None:
            return
        conds = _conditions(self, lat, t, where, search)
        after: Optional[tuple[Any, str]] = None
        while True:
            stmt = select(t).where(and_(true_(), *conds))
            if after is not None:
                stmt = stmt.where(tuple_(t.c._created_at, t.c._key) > tuple_(*after))
            stmt = stmt.order_by(t.c._created_at.asc(), t.c._key.asc()).limit(batch)
            async with _data(self) as session:
                rows = (await session.execute(stmt)).all()
            if not rows:
                return
            yield [_row(lat, r) for r in rows]
            if len(rows) < batch:
                return
            after = (rows[-1]._mapping["_created_at"], rows[-1]._mapping["_key"])

    async def lattice_column_values(self, customer_id: str, lattice_id: str, column: str, *,
                                    search: Optional[str] = None, limit: int = 50) -> list[tuple[str, int]]:
        """The distinct values of one column with how many rows hold each,
        most rows first: the browse-by-entity list."""
        lat, t = await _open(self, customer_id, lattice_id)
        if lat is None:
            return []
        c = _col(t, column)
        n = func.count().label("n")
        stmt = select(c.label("v"), n).where(c.is_not(None))
        if search:
            stmt = stmt.where(func.lower(cast(c, String)).like(f"%{search.lower()}%"))
        stmt = stmt.group_by(c).order_by(n.desc(), c.asc()).limit(limit)
        typ = next((x["type"] for x in lat["columns"] if x["key"] == column), "text")
        async with _data(self) as session:
            return [(str(lt.from_db(typ, v)), int(k)) for v, k in (await session.execute(stmt)).all() if v not in (None, "")]

    async def list_lattice_rows_oldest(self, customer_id: str, lattice_id: str, limit: int = 500) -> list[dict[str, Any]]:
        lat, t = await _open(self, customer_id, lattice_id)
        if lat is None:
            return []
        async with _data(self) as session:
            rows = (await session.execute(select(t).order_by(t.c._updated_at.asc()).limit(limit))).all()
            return [_row(lat, r) for r in rows]

    async def aggregate_lattice_rows(self, customer_id: str, lattice_id: str, *, group_by: list[str],
                                     metrics: list[tuple[str, str]],
                                     where: Optional[list[tuple[str, str, str, Any]]] = None,
                                     search: Optional[tuple[list[str], str]] = None,
                                     order: Optional[tuple[int, bool]] = None,
                                     limit: int = 50) -> tuple[list[dict[str, Any]], int]:
        """Group rows by up to three columns and compute count plus metrics
        [(fn, column)] with fn sum | avg | min | max over number columns.
        order: (index into ["count", *metrics], descending). Returns
        (groups, how many groups there are)."""
        lat, t = await _open(self, customer_id, lattice_id)
        if lat is None:
            return [], 0
        types = {c["key"]: c["type"] for c in lat["columns"]}
        conds = and_(true_(), *_conditions(self, lat, t, where, search))
        g = [_col(t, x) for x in group_by]
        agg = {"sum": func.sum, "avg": func.avg, "min": func.min, "max": func.max}
        outs = [func.count().label("n")] + [agg[fn](_col(t, c)).label(f"a{j}") for j, (fn, c) in enumerate(metrics)]
        stmt = select(*[x.label(f"g{i}") for i, x in enumerate(g)], *outs).select_from(t).where(conds)
        if g:
            stmt = stmt.group_by(*g)
        idx, desc = order or (0, True)
        key = outs[idx] if 0 <= idx < len(outs) else outs[0]
        stmt = stmt.order_by(key.desc() if desc else key.asc(), *[x.asc() for x in g]).limit(limit)
        async with _data(self) as session:
            rows = (await session.execute(stmt)).all()
            total = 1
            if g:
                total = int((await session.execute(
                    select(func.count()).select_from(select(*g).where(conds).group_by(*g).subquery()))).scalar_one())
        out = []
        for r in rows:
            m = r._mapping
            out.append({"key": [None if m[f"g{i}"] is None else str(lt.from_db(types[group_by[i]], m[f"g{i}"]))
                                for i in range(len(group_by))],
                        "count": int(m["n"] or 0),
                        "values": [None if m[f"a{j}"] is None else float(m[f"a{j}"]) for j in range(len(metrics))]})
        return out, total

    # ------------------------------------------------- goal criteria (Q135 A)

    async def lattice_groups(self, customer_id: str, lattice_id: str, column: str, *,
                             where: Optional[list[tuple[str, str, str, Any]]] = None,
                             limit: int = 5000) -> dict[Any, int]:
        """{value of column: rows} over the matching rows (the groups criterion)."""
        lat, t = await _open(self, customer_id, lattice_id)
        if lat is None:
            return {}
        types = {c["key"]: c["type"] for c in lat["columns"]}
        col = _col(t, column)
        conds = and_(true_(), *_conditions(self, lat, t, where, None))
        async with _data(self) as session:
            rows = (await session.execute(select(col, func.count()).where(conds).group_by(col).limit(max(1, limit)))).all()
        return {lt.from_db(types.get(column, "text"), r[0]): int(r[1]) for r in rows}

    async def lattice_measure(self, customer_id: str, lattice_id: str, *,
                              where: Optional[list[tuple[str, str, str, Any]]] = None,
                              filled: Optional[list[str]] = None, distinct: Optional[str] = None,
                              source_of: Optional[str] = None, sample: int = 5) -> dict[str, Any]:
        """Counts over the matching rows in one pass: rows; filled (rows
        where every column in `filled` holds a value) with up to `sample`
        row keys that don't; distinct values of one column; distinct
        sources recorded on one column's cells."""
        lat, t = await _open(self, customer_id, lattice_id)
        if lat is None:
            return {"rows": 0}
        conds = and_(true_(), *_conditions(self, lat, t, where, None))
        outs: list[Any] = [func.count().label("rows")]
        full = None
        if filled:
            full = and_(*[_col(t, c).is_not(None) for c in filled])
            outs.append(func.coalesce(func.sum(case((full, 1), else_=0)), 0).label("filled"))
        if distinct:
            outs.append(func.count(func.distinct(_col(t, distinct))).label("distinct"))
        if source_of:
            src = t.c._cells[(source_of, "source")].as_string()
            outs.append(func.count(func.distinct(case((src != "", src), else_=None))).label("sources"))
        async with _data(self) as session:
            m = (await session.execute(select(*outs).select_from(t).where(conds))).one()._mapping
            out: dict[str, Any] = {"rows": int(m["rows"] or 0)}
            if filled:
                out["filled"] = int(m["filled"] or 0)
                miss = (await session.execute(select(t.c._key).where(conds).where(~full)
                                              .order_by(t.c._created_at.asc(), t.c._key.asc())
                                              .limit(max(0, sample)))).scalars().all()
                out["missing_sample"] = list(miss)
            if distinct:
                out["distinct"] = int(m["distinct"] or 0)
            if source_of:
                out["sources"] = int(m["sources"] or 0)
            return out

    async def lattice_sources(self, customer_id: str, lattice_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        """The pages a lattice's rows came from (the source on the first key
        column's cells), with how many rows each gave and when it was last
        written, oldest first (Series 383, Q150 A: check-ins re-imported
        pages they already had)."""
        lat, t = await _open(self, customer_id, lattice_id)
        if lat is None or not lat["key_columns"]:
            return []
        src = t.c._cells[(lat["key_columns"][0], "source")].as_string()
        stmt = (select(src.label("source"), func.count().label("rows"), func.max(t.c._updated_at).label("at"))
                .where(src.is_not(None)).where(src != "").group_by(src).order_by(func.min(t.c._created_at).asc())
                .limit(max(1, limit)))
        async with _data(self) as session:
            rows = (await session.execute(stmt)).all()
        return [{"source": r.source, "rows": int(r.rows or 0), "at": _iso(r.at) if r.at else None} for r in rows]

    async def lattice_reference_coverage(self, customer_id: str, parent_id: str, parent_col: str, child_id: str,
                                         child_col: str, *, min_per: int = 1,
                                         parent_where: Optional[list[tuple[str, str, str, Any]]] = None,
                                         sample: int = 5) -> dict[str, Any]:
        """How many distinct values of parent_col (each game) have at least
        min_per rows in the child lattice whose child_col holds that value
        (its players' rows), with a sample of the ones that don't and the
        fewest / most children any parent has."""
        plat, pt = await _open(self, customer_id, parent_id)
        clat, ct = await _open(self, customer_id, child_id)
        if plat is None or clat is None:
            return {"parents": 0, "covered": 0, "fewest": 0, "most": 0, "missing_sample": []}
        pexpr, cexpr = func.lower(cast(_col(pt, parent_col), String)), func.lower(cast(_col(ct, child_col), String))
        parents = (select(pexpr.label("v")).where(and_(true_(), *_conditions(self, plat, pt, parent_where, None)))
                   .where(_col(pt, parent_col).is_not(None)).distinct().subquery())
        children = (select(cexpr.label("v"), func.count().label("n")).where(_col(ct, child_col).is_not(None))
                    .group_by(cexpr).subquery())
        joined = parents.outerjoin(children, parents.c.v == children.c.v)
        n = func.coalesce(children.c.n, 0)
        async with _data(self) as session:
            m = (await session.execute(select(func.count().label("total"),
                                              func.coalesce(func.sum(case((n >= min_per, 1), else_=0)), 0).label("covered"),
                                              func.min(n).label("lo"), func.max(n).label("hi"))
                                       .select_from(joined))).one()._mapping
            miss = (await session.execute(select(parents.c.v).select_from(joined).where(n < min_per)
                                          .order_by(parents.c.v.asc()).limit(max(0, sample)))).scalars().all()
        return {"parents": int(m["total"] or 0), "covered": int(m["covered"] or 0),
                "fewest": int(m["lo"] or 0), "most": int(m["hi"] or 0), "missing_sample": [str(x) for x in miss]}

    # ------------------------------------------- general lattices (Series 380)

    async def list_general_lattices(self, customer_id: str) -> list[dict[str, Any]]:
        """Every general lattice, with whether this customer owns it or
        reads it. Owners are never named to other customers."""
        subscribed = await _subscribed(self, customer_id)
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(select(LatticeRow).where(LatticeRow.scope == "general")
                                          .order_by(LatticeRow.name))).scalars().all()
            subs = dict((await session.execute(select(LatticeSubscriptionRow.lattice_id, func.count())
                                               .group_by(LatticeSubscriptionRow.lattice_id))).all())
        out = []
        for r in rows:
            lat = lattice_dict(r)
            try:
                t = await _table(self, r.customer_id, lat)
                async with _data(self) as ds:
                    lat["rows"] = int((await ds.execute(select(func.count()).select_from(t))).scalar_one())
            except Exception:  # noqa: BLE001 — one missing table never hides the rest of the list
                lat["rows"] = 0
            lat.update(yours=r.customer_id == customer_id, subscribed=r.id in subscribed,
                       subscribers=int(subs.get(r.id, 0)))
            if not lat["yours"]:
                lat.pop("customer_id", None)
                lat.pop("created_by", None)
                lat.pop("goal_id", None)
            out.append(lat)
        return out

    async def subscribe_lattice(self, customer_id: str, lattice_id: str) -> bool:
        """Read a general lattice someone else owns. False when it isn't one."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(LatticeRow, lattice_id)
            if row is None or (row.scope or "private") != "general" or row.customer_id == customer_id:
                return False
            have = (await session.execute(select(LatticeSubscriptionRow).where(
                LatticeSubscriptionRow.customer_id == customer_id).where(
                LatticeSubscriptionRow.lattice_id == lattice_id))).scalar_one_or_none()
            if have is None:
                session.add(LatticeSubscriptionRow(id=f"lsub_{uuid.uuid4().hex[:16]}", customer_id=customer_id,
                                                   lattice_id=lattice_id, created_at=_now()))
                await session.commit()
            lat = lattice_dict(row)
        if _is_pg(self):
            async with _data(self) as ds:
                conn = await ds.connection()
                await _ensure_schema(self, conn, customer_id)
                await conn.exec_driver_sql(f'GRANT USAGE ON SCHEMA "{lt.GENERAL_SCHEMA}" TO "{lt.role_name(customer_id)}"')
                v = lt.qualified(self._lattice_engine, lt.GENERAL_SCHEMA, lat["sql_name"])  # type: ignore[attr-defined]
                await conn.exec_driver_sql(f'GRANT SELECT ON {v} TO "{lt.role_name(customer_id)}"')
                await ds.commit()
        return True

    async def unsubscribe_lattice(self, customer_id: str, lattice_id: str) -> bool:
        async with self.session() as session:  # type: ignore[attr-defined]
            res = await session.execute(delete(LatticeSubscriptionRow).where(
                LatticeSubscriptionRow.customer_id == customer_id).where(LatticeSubscriptionRow.lattice_id == lattice_id))
            await session.commit()
            row = await session.get(LatticeRow, lattice_id)
        if row is not None and _is_pg(self) and row.sql_name and (row.scope or "private") == "general":
            async with _data(self) as ds:
                v = lt.qualified(self._lattice_engine, lt.GENERAL_SCHEMA, row.sql_name)  # type: ignore[attr-defined]
                await ds.execute(text(f'REVOKE SELECT ON {v} FROM "{lt.role_name(customer_id)}"'))
                await ds.commit()
        return bool(res.rowcount)

    async def lattice_readers_referencing(self, lattice_id: str, owner_id: str) -> list[str]:
        """Other customers' lattices with a ref column pointing at this one
        (by lattice name, never by customer)."""
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(select(LatticeRow).where(LatticeRow.customer_id != owner_id))).scalars().all()
        return [r.name for r in rows if any(c.get("type") == "ref" and c.get("ref") == lattice_id for c in r.columns or [])]

    async def set_lattice_scope(self, customer_id: str, lattice_id: str, scope: str) -> Optional[dict[str, Any]]:
        """Move the owner's lattice between its private schema and the
        general one: the same table (rows, keys, history) under a new
        schema, its view rebuilt there. The service checks what may move."""
        lat, t = await _open(self, customer_id, lattice_id, write=True)
        if lat is None or lat["scope"] == scope:
            return lat
        e = self._lattice_engine  # type: ignore[attr-defined]
        moved = {**lat, "scope": scope}
        async with _data(self) as ds:
            conn = await ds.connection()
            old_view_schema = lt.GENERAL_SCHEMA if _general(lat) else lt.schema_name(customer_id)
            if _is_pg(self):
                await conn.exec_driver_sql(f'CREATE SCHEMA IF NOT EXISTS "{lt.GENERAL_SCHEMA}"')
                await _ensure_schema(self, conn, customer_id)
                if lat.get("sql_name"):
                    await conn.exec_driver_sql(f"DROP VIEW IF EXISTS {lt.qualified(e, old_view_schema, lat['sql_name'])}")
                new_schema, _ = lt.located(e, customer_id, lattice_id, scope == "general")
                await conn.exec_driver_sql(f'ALTER TABLE {lt.qualified(e, t.schema, t.name)} SET SCHEMA "{new_schema}"')
                if scope == "general":
                    await conn.exec_driver_sql(f'GRANT USAGE ON SCHEMA "{lt.GENERAL_SCHEMA}" TO "{lt.role_name(customer_id)}"')
            else:
                _s, new_name = lt.located(e, customer_id, lattice_id, scope == "general")
                await conn.exec_driver_sql(f"ALTER TABLE {lt.quote(e, t.name)} RENAME TO {lt.quote(e, new_name)}")
            await ds.commit()
        async with self.session() as session:  # type: ignore[attr-defined]
            await session.execute(update(LatticeRow).where(LatticeRow.id == lattice_id)
                                  .where(LatticeRow.customer_id == customer_id).values(scope=scope, updated_at=_now()))
            if scope != "general":
                await session.execute(delete(LatticeSubscriptionRow).where(LatticeSubscriptionRow.lattice_id == lattice_id))
            await session.commit()
        _TABLES.clear()  # tables pointing at it were built with its old place
        nt = await _table(self, customer_id, moved)
        async with _data(self) as ds:
            await _view(self, await ds.connection(), customer_id, moved, nt)
            await ds.commit()
        return await self.get_lattice(customer_id, lattice_id)  # type: ignore[attr-defined]

    # ---------------------------------------------------------- the SQL tool

    async def lattice_rule_query(self, customer_id: str, query: str) -> list[tuple]:
        """A lattice rule's check (Series 384), run exactly like the SQL tool:
        as the customer's read-only role (Postgres) or behind the authorizer
        (SQLite), so a rule can read the customer's lattices and nothing
        else, and can never write. The query is built by lattices/rules.py
        around a CTE named _rows holding the rows being written. Raises
        lattice_tables.SqlRefused."""
        if _is_pg(self):
            _cols, rows = await _sql_pg(self, customer_id, query)
        else:
            lats = await self.list_lattices(customer_id)  # type: ignore[attr-defined]
            _cols, rows = await _sql_sqlite(self, customer_id, lats, query, allow=frozenset({"_rows"}))
        return rows

    async def lattice_sql(self, customer_id: str, query: str, *, limit: int = lt.SQL_ROW_LIMIT) -> dict[str, Any]:
        """One read-only query over the customer's lattices, by their SQL
        names. Postgres runs it as the customer's role (it may read that
        customer's views and nothing else) in a read-only transaction with
        a time limit; SQLite (development) behind an authorizer that admits
        only this customer's tables. Raises lattice_tables.SqlRefused."""
        q = lt.check_sql(query)
        lim = max(1, min(int(limit or lt.SQL_ROW_LIMIT), lt.SQL_ROW_LIMIT))
        lats = [x for x in await self.list_lattices(customer_id)]  # type: ignore[attr-defined]  (moves pre-378 ones)
        if not lats:
            raise lt.SqlRefused("no_lattices", "There are no lattices to query yet.")
        wrapped = f"SELECT * FROM ({q}) AS _q LIMIT {lim + 1}"
        if _is_pg(self):
            cols, rows = await _sql_pg(self, customer_id, wrapped)
        else:
            cols, rows = await _sql_sqlite(self, customer_id, lats, wrapped)
        out_rows = [[lt.from_db("", v) if isinstance(v, (date, datetime)) else v for v in r] for r in rows[:lim]]
        return {"columns": cols, "rows": out_rows, "row_count": len(out_rows), "truncated": len(rows) > lim,
                "lattices": {_sql_ref(x): x["name"] for x in lats if x.get("sql_name")}}


def _sql_ref(lat: dict[str, Any]) -> str:
    """How a query names a lattice: general ones under general."""
    return f"{lt.GENERAL_SCHEMA}.{lat['sql_name']}" if _general(lat) else lat["sql_name"]


def true_():
    from sqlalchemy import true
    return true()


def _clean_error(e: Exception) -> str:
    msg = str(getattr(e, "orig", None) or e).splitlines()[0]
    for noise in ("(sqlalchemy.dialects.postgresql.asyncpg.Error) ", "<class 'asyncpg.exceptions.", "(sqlite3.OperationalError) "):
        msg = msg.replace(noise, "")
    return msg[:300]


async def _sql_pg(store: Any, customer_id: str, wrapped: str) -> tuple[list[str], list[tuple]]:
    schema, role = lt.schema_name(customer_id), lt.role_name(customer_id)
    async with store._lattice_engine.connect() as conn:
        trans = await conn.begin()
        try:
            await conn.exec_driver_sql("SET TRANSACTION READ ONLY")
            await conn.exec_driver_sql(f'SET LOCAL ROLE "{role}"')
            await conn.exec_driver_sql(f"SET LOCAL statement_timeout = {lt.SQL_TIMEOUT_MS}")
            await conn.exec_driver_sql(f'SET LOCAL search_path = "{schema}", "{lt.GENERAL_SCHEMA}"')
            res = await conn.exec_driver_sql(wrapped)
            return list(res.keys()), [tuple(r) for r in res.fetchall()]
        except Exception as e:  # noqa: BLE001 — the person's query failed, not the server
            text_ = _clean_error(e)
            code = ("sql_timeout" if "statement timeout" in text_ or "canceling statement" in text_
                    else "sql_denied" if "permission denied" in text_ else "sql_error")
            raise lt.SqlRefused(code, text_)
        finally:
            await trans.rollback()


async def _sql_sqlite(store: Any, customer_id: str, lats: list[dict[str, Any]], wrapped: str,
                      allow: frozenset = frozenset()) -> tuple[list[str], list[tuple]]:
    import sqlite3

    e = store._lattice_engine
    allowed: set[str] = set(allow)
    views: list[str] = []
    async with e.connect() as conn:
        raw = await conn.get_raw_connection()
        aio = raw.driver_connection
        try:
            for x in lats:
                if not x.get("sql_name"):
                    continue
                _, phys = lt.located(e, x["customer_id"], x["id"], _general(x))
                # SQLite has no schemas: general.x is the view general__x here.
                vname = f"{lt.GENERAL_SCHEMA}__{x['sql_name']}" if _general(x) else x["sql_name"]
                allowed |= {phys, vname}
                cols = ", ".join(lt.quote(e, c) for c in lt.view_columns(x))
                await aio.execute(f"DROP VIEW IF EXISTS temp.{lt.quote(e, vname)}")
                await aio.execute(f"CREATE TEMP VIEW {lt.quote(e, vname)} AS SELECT _id AS row_id, {cols} "
                                  f"FROM main.{lt.quote(e, phys)}")
                views.append(vname)
            wrapped = lt.re.sub(rf"\b{lt.GENERAL_SCHEMA}\.", f"{lt.GENERAL_SCHEMA}__", wrapped)

            def auth(action, a1, a2, db, src):
                if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE):
                    return sqlite3.SQLITE_OK
                if action == sqlite3.SQLITE_READ:
                    return sqlite3.SQLITE_OK if (a1 in allowed) else sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_DENY
            deadline = time.monotonic() + lt.SQL_TIMEOUT_MS / 1000
            await aio.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
            await aio.set_authorizer(auth)
            try:
                cur = await aio.execute(wrapped)
                rows = await cur.fetchall()
                cols = [d[0] for d in cur.description or []]
                await cur.close()
            except sqlite3.Error as err:
                msg = _clean_error(err)
                code = ("sql_timeout" if "interrupted" in msg else "sql_denied" if ("not authorized" in msg or "prohibited" in msg) else "sql_error")
                raise lt.SqlRefused(code, msg)
            finally:
                await aio.set_authorizer(None)
                await aio.set_progress_handler(None, 0)
            return cols, [tuple(r) for r in rows]
        finally:
            for v in views:
                await aio.execute(f"DROP VIEW IF EXISTS temp.{lt.quote(e, v)}")
