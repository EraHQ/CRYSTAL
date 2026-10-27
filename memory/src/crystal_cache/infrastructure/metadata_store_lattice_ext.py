"""Lattices (2026-10-25; Q115 A, Q116 A, Q117 A): the SQL for lattices and
lattice_rows.

A lattice is a named table in Crystal storage. Its rows live in
lattice_rows with their values in `data` ({column: value}) and each cell's
provenance in `cells` ({column: {as_of, source, history}}). Typing,
locking and caps are decided in lattices/service.py; this mixin only
reads and writes. Filters read JSON values with SQLAlchemy's JSON
accessors, which compile to json_extract on SQLite and ->> on Postgres.
R9: these tables' SQL lives here and nowhere else.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from sqlalchemy import Float, and_, case, cast, delete, func, literal, literal_column, or_, select, tuple_, update
from sqlalchemy.dialects.postgresql import JSONB

from .schema import LatticeRecordRow, LatticeRow

_LATTICE_FIELDS = {"name", "description", "columns", "key_columns"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def lattice_dict(r: LatticeRow, rows: Optional[int] = None) -> dict[str, Any]:
    out = {"id": r.id, "customer_id": r.customer_id, "name": r.name, "description": r.description or "",
           "columns": [dict(c) for c in (r.columns or [])], "key_columns": list(r.key_columns or []),
           "goal_id": r.goal_id, "created_by": r.created_by or "",
           "created_at": _iso(r.created_at), "updated_at": _iso(r.updated_at)}
    if rows is not None:
        out["rows"] = int(rows)
    return out


def lattice_row_dict(r: LatticeRecordRow) -> dict[str, Any]:
    return {"id": r.id, "lattice_id": r.lattice_id, "row_key": r.row_key, "data": dict(r.data or {}),
            "cells": {k: dict(v) for k, v in (r.cells or {}).items()},
            "created_at": _iso(r.created_at), "updated_at": _iso(r.updated_at)}


def _value_expr(column: str, col_type: str):
    v = LatticeRecordRow.data[column]
    if col_type == "number":
        return v.as_float()
    if col_type == "bool":
        return v.as_boolean()
    return v.as_string()


def derived(data: dict[str, Any]) -> tuple[dict[str, Any], Optional[str]]:
    """(facets, search_text) for one row's data: what browsing filters and
    searches on (Q133 A)."""
    facets = {k: (v.lower() if isinstance(v, str) else v) for k, v in (data or {}).items()}
    text = " ".join(v for v in (data or {}).values() if isinstance(v, str)).lower()
    return facets, (text or None)


_WORD = re.compile(r"[^\w]+", re.UNICODE)


def _tsquery(needle: str) -> Optional[str]:
    """Each word of the search as a prefix ("jay tat" finds Jayson Tatum)."""
    words = [w for w in _WORD.split(needle.lower()) if w][:8]
    return " & ".join(f"{w}:*" for w in words) or None


def _conditions(customer_id: str, lattice_id: str, where: Optional[list[tuple[str, str, str, Any]]],
                search: Optional[tuple[list[str], str]], pg: bool = False) -> list[Any]:
    conds = [LatticeRecordRow.lattice_id == lattice_id, LatticeRecordRow.customer_id == customer_id]
    for col, typ, op, value in where or []:
        expr = _value_expr(col, typ)
        if op == "eq" and pg and typ in ("text", "date", "url"):
            # GIN containment on facets (text already lowercased there).
            conds.append(LatticeRecordRow.facets.op("@>")(literal({col: str(value).lower()}, JSONB)))
        elif op == "eq":
            conds.append(func.lower(expr) == str(value).lower() if typ in ("text", "date", "url") else expr == value)
        elif op == "contains":
            conds.append(func.lower(_value_expr(col, "text")).like(f"%{str(value).lower()}%"))
        elif op == "gte":
            conds.append(expr >= value)
        elif op == "lte":
            conds.append(expr <= value)
    if search and search[0] and search[1]:
        q = _tsquery(search[1]) if pg else None
        if q:
            # Literals, not parameters: the expression must match the GIN index's exactly.
            conds.append(func.to_tsvector(literal_column("'simple'::regconfig"),
                                          func.coalesce(LatticeRecordRow.search_text, literal_column("''")))
                         .op("@@")(func.to_tsquery(literal_column("'simple'::regconfig"), q)))
        elif not pg:
            conds.append(LatticeRecordRow.search_text.like(f"%{search[1].lower()}%"))
    return conds


def _is_pg(store: Any) -> bool:
    # Mixin methods are bound one by one (infrastructure/__init__), so this is a function, not a property.
    return store._engine.dialect.name == "postgresql"


_AGG = {"sum": func.sum, "avg": func.avg, "min": func.min, "max": func.max}


class LatticeExtensionsMixin:
    """lattices / lattice_rows, bound onto MetadataStore."""

    # ------------------------------------------------------------ lattices

    async def create_lattice(self, customer_id: str, *, name: str, description: str, columns: list[dict[str, Any]],
                             key_columns: list[str], goal_id: Optional[str] = None,
                             created_by: str = "") -> dict[str, Any]:
        now = _now()
        row = LatticeRow(id=f"lat_{uuid.uuid4().hex[:16]}", customer_id=customer_id, name=name[:120],
                         description=description or "", columns=list(columns), key_columns=list(key_columns),
                         goal_id=goal_id, created_by=(created_by or "")[:96], created_at=now, updated_at=now)
        async with self.session() as session:  # type: ignore[attr-defined]
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return lattice_dict(row, 0)

    async def get_lattice(self, customer_id: str, lattice_id: str) -> Optional[dict[str, Any]]:
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(LatticeRow, lattice_id)
            if row is None or row.customer_id != customer_id:
                return None
            n = (await session.execute(select(func.count()).select_from(LatticeRecordRow)
                                       .where(LatticeRecordRow.lattice_id == lattice_id))).scalar_one()
            return lattice_dict(row, n)

    async def get_lattice_by_name(self, customer_id: str, name: str) -> Optional[dict[str, Any]]:
        async with self.session() as session:  # type: ignore[attr-defined]
            row = (await session.execute(select(LatticeRow).where(LatticeRow.customer_id == customer_id)
                                         .where(func.lower(LatticeRow.name) == (name or "").strip().lower())
                                         .limit(1))).scalar_one_or_none()
        return await self.get_lattice(customer_id, row.id) if row is not None else None  # type: ignore[attr-defined]

    async def list_lattices(self, customer_id: str) -> list[dict[str, Any]]:
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(select(LatticeRow).where(LatticeRow.customer_id == customer_id)
                                          .order_by(LatticeRow.updated_at.desc()))).scalars().all()
            counts = dict((await session.execute(
                select(LatticeRecordRow.lattice_id, func.count()).where(LatticeRecordRow.customer_id == customer_id)
                .group_by(LatticeRecordRow.lattice_id))).all())
            return [lattice_dict(r, counts.get(r.id, 0)) for r in rows]

    async def count_lattices(self, customer_id: str) -> int:
        async with self.session() as session:  # type: ignore[attr-defined]
            return int((await session.execute(select(func.count()).select_from(LatticeRow)
                                              .where(LatticeRow.customer_id == customer_id))).scalar_one())

    async def update_lattice(self, customer_id: str, lattice_id: str, **fields: Any) -> Optional[dict[str, Any]]:
        bad = set(fields) - _LATTICE_FIELDS
        if bad:
            raise ValueError(f"unknown lattice fields {sorted(bad)}")
        async with self.session() as session:  # type: ignore[attr-defined]
            await session.execute(update(LatticeRow).where(LatticeRow.id == lattice_id)
                                  .where(LatticeRow.customer_id == customer_id).values(**fields, updated_at=_now()))
            await session.commit()
        return await self.get_lattice(customer_id, lattice_id)  # type: ignore[attr-defined]

    async def touch_lattice(self, customer_id: str, lattice_id: str) -> None:
        async with self.session() as session:  # type: ignore[attr-defined]
            await session.execute(update(LatticeRow).where(LatticeRow.id == lattice_id)
                                  .where(LatticeRow.customer_id == customer_id).values(updated_at=_now()))
            await session.commit()

    async def delete_lattice(self, customer_id: str, lattice_id: str) -> int:
        """The lattice and every row. Returns how many rows went."""
        async with self.session() as session:  # type: ignore[attr-defined]
            res = await session.execute(delete(LatticeRecordRow).where(LatticeRecordRow.lattice_id == lattice_id)
                                        .where(LatticeRecordRow.customer_id == customer_id))
            await session.execute(delete(LatticeRow).where(LatticeRow.id == lattice_id)
                                  .where(LatticeRow.customer_id == customer_id))
            await session.commit()
            return int(res.rowcount or 0)

    # ---------------------------------------------------------------- rows

    async def count_lattice_rows(self, customer_id: str, lattice_id: Optional[str] = None) -> int:
        stmt = select(func.count()).select_from(LatticeRecordRow).where(LatticeRecordRow.customer_id == customer_id)
        if lattice_id:
            stmt = stmt.where(LatticeRecordRow.lattice_id == lattice_id)
        async with self.session() as session:  # type: ignore[attr-defined]
            return int((await session.execute(stmt)).scalar_one())

    async def get_lattice_rows_by_keys(self, customer_id: str, lattice_id: str,
                                       keys: Iterable[str]) -> dict[str, dict[str, Any]]:
        keys = list(dict.fromkeys(keys))
        out: dict[str, dict[str, Any]] = {}
        async with self.session() as session:  # type: ignore[attr-defined]
            for i in range(0, len(keys), 400):
                rows = (await session.execute(select(LatticeRecordRow)
                                              .where(LatticeRecordRow.lattice_id == lattice_id)
                                              .where(LatticeRecordRow.customer_id == customer_id)
                                              .where(LatticeRecordRow.row_key.in_(keys[i:i + 400])))).scalars().all()
                out.update({r.row_key: lattice_row_dict(r) for r in rows})
        return out

    async def get_lattice_rows(self, customer_id: str, lattice_id: str, row_ids: Iterable[str]) -> list[dict[str, Any]]:
        ids = list(dict.fromkeys(row_ids))
        if not ids:
            return []
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(select(LatticeRecordRow).where(LatticeRecordRow.lattice_id == lattice_id)
                                          .where(LatticeRecordRow.customer_id == customer_id)
                                          .where(LatticeRecordRow.id.in_(ids)))).scalars().all()
            return [lattice_row_dict(r) for r in rows]

    async def write_lattice_rows(self, customer_id: str, lattice_id: str, *, inserts: list[dict[str, Any]],
                                 updates: list[dict[str, Any]]) -> list[str]:
        """One transaction. inserts: [{row_key, data, cells}] -> new ids,
        in order. updates: [{id, data, cells}] replace both maps."""
        now = _now()
        ids: list[str] = []
        async with self.session() as session:  # type: ignore[attr-defined]
            for r in inserts:
                rid = f"lr_{uuid.uuid4().hex[:16]}"
                ids.append(rid)
                facets, text = derived(r["data"])
                session.add(LatticeRecordRow(id=rid, lattice_id=lattice_id, customer_id=customer_id,
                                             row_key=r["row_key"], data=dict(r["data"]), cells=dict(r["cells"]),
                                             facets=facets, search_text=text, created_at=now, updated_at=now))
            for r in updates:
                facets, text = derived(r["data"])
                await session.execute(update(LatticeRecordRow).where(LatticeRecordRow.id == r["id"])
                                      .where(LatticeRecordRow.lattice_id == lattice_id)
                                      .where(LatticeRecordRow.customer_id == customer_id)
                                      .values(data=dict(r["data"]), cells=dict(r["cells"]), facets=facets,
                                              search_text=text, updated_at=now))
            await session.execute(update(LatticeRow).where(LatticeRow.id == lattice_id)
                                  .where(LatticeRow.customer_id == customer_id).values(updated_at=now))
            await session.commit()
        return ids

    async def delete_lattice_rows(self, customer_id: str, lattice_id: str, row_ids: Iterable[str]) -> int:
        ids = list(dict.fromkeys(row_ids))
        if not ids:
            return 0
        async with self.session() as session:  # type: ignore[attr-defined]
            res = await session.execute(delete(LatticeRecordRow).where(LatticeRecordRow.lattice_id == lattice_id)
                                        .where(LatticeRecordRow.customer_id == customer_id)
                                        .where(LatticeRecordRow.id.in_(ids)))
            await session.commit()
            return int(res.rowcount or 0)

    async def query_lattice_rows(self, customer_id: str, lattice_id: str, *,
                                 where: Optional[list[tuple[str, str, str, Any]]] = None,
                                 search: Optional[tuple[list[str], str]] = None,
                                 sort: Optional[tuple[str, str, bool]] = None,
                                 limit: int = 50, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
        """where: [(column, type, op, value)] with op eq | contains | gte |
        lte; search: ([text columns], needle) matched case-insensitively in
        any of them; sort: (column, type, descending). Returns (rows, total
        matching)."""
        conds = _conditions(customer_id, lattice_id, where, search, _is_pg(self))
        stmt = select(LatticeRecordRow).where(and_(*conds))
        if sort:
            col, typ, desc = sort
            expr = cast(_value_expr(col, "text"), Float) if typ == "number" else _value_expr(col, typ)
            stmt = stmt.order_by(expr.desc() if desc else expr.asc(), LatticeRecordRow.created_at.asc())
        else:
            stmt = stmt.order_by(LatticeRecordRow.created_at.asc(), LatticeRecordRow.row_key.asc())  # a batch shares one timestamp
        async with self.session() as session:  # type: ignore[attr-defined]
            total = int((await session.execute(select(func.count()).select_from(LatticeRecordRow)
                                               .where(and_(*conds)))).scalar_one())
            rows = (await session.execute(stmt.limit(limit).offset(offset))).scalars().all()
            return [lattice_row_dict(r) for r in rows], total

    async def iter_lattice_rows(self, customer_id: str, lattice_id: str, *,
                                where: Optional[list[tuple[str, str, str, Any]]] = None,
                                search: Optional[tuple[list[str], str]] = None, batch: int = 1000):
        """Every matching row in the order rows were added, a batch at a
        time, by keyset (no count, no offset: an export of 250,000 rows
        reads each once)."""
        conds = _conditions(customer_id, lattice_id, where, search, _is_pg(self))
        after: Optional[tuple[Any, str]] = None
        while True:
            stmt = select(LatticeRecordRow).where(and_(*conds))
            if after is not None:
                # A row-value comparison walks ix_lattice_rows_lattice_order.
                stmt = stmt.where(tuple_(LatticeRecordRow.created_at, LatticeRecordRow.row_key) > tuple_(*after))
            stmt = stmt.order_by(LatticeRecordRow.created_at.asc(), LatticeRecordRow.row_key.asc()).limit(batch)
            async with self.session() as session:  # type: ignore[attr-defined]
                rows = (await session.execute(stmt)).scalars().all()
            if not rows:
                return
            yield [lattice_row_dict(r) for r in rows]
            if len(rows) < batch:
                return
            after = (rows[-1].created_at, rows[-1].row_key)

    async def lattice_column_values(self, customer_id: str, lattice_id: str, column: str, *,
                                    search: Optional[str] = None, limit: int = 50) -> list[tuple[str, int]]:
        """The distinct values of one column with how many rows hold each,
        most rows first: the browse-by-entity list."""
        # Grouped through a subquery: Postgres would see the JSON accessor in
        # SELECT and in GROUP BY as two expressions (each binds its own key
        # parameter) and refuse the grouping.
        expr = _value_expr(column, "text")
        inner = (select(expr.label("v")).where(LatticeRecordRow.lattice_id == lattice_id)
                 .where(LatticeRecordRow.customer_id == customer_id).where(expr.is_not(None)))
        if search:
            inner = inner.where(func.lower(expr).like(f"%{search.lower()}%"))
        sub = inner.subquery()
        n = func.count().label("n")
        stmt = select(sub.c.v, n).group_by(sub.c.v).order_by(n.desc(), sub.c.v.asc()).limit(limit)
        async with self.session() as session:  # type: ignore[attr-defined]
            return [(str(v), int(n)) for v, n in (await session.execute(stmt)).all() if v not in (None, "")]

    async def list_lattice_rows_oldest(self, customer_id: str, lattice_id: str, limit: int = 500) -> list[dict[str, Any]]:
        """Rows least recently written first: where stale live cells are."""
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(select(LatticeRecordRow).where(LatticeRecordRow.lattice_id == lattice_id)
                                          .where(LatticeRecordRow.customer_id == customer_id)
                                          .order_by(LatticeRecordRow.updated_at.asc()).limit(limit))).scalars().all()
            return [lattice_row_dict(r) for r in rows]

    async def aggregate_lattice_rows(self, customer_id: str, lattice_id: str, *, group_by: list[str],
                                     metrics: list[tuple[str, str]],
                                     where: Optional[list[tuple[str, str, str, Any]]] = None,
                                     search: Optional[tuple[list[str], str]] = None,
                                     order: Optional[tuple[int, bool]] = None,
                                     limit: int = 50) -> tuple[list[dict[str, Any]], int]:
        """Group rows by up to three columns (read as text) and compute
        count plus metrics [(fn, column)] with fn sum | avg | min | max over
        number columns. order: (index into ["count", *metrics], descending).
        Returns (groups, how many groups there are). Grouped through a
        subquery for the same Postgres reason as lattice_column_values."""
        cols = [_value_expr(g, "text").label(f"g{i}") for i, g in enumerate(group_by)]
        vals = [_value_expr(c, "number").label(f"m{j}") for j, (_fn, c) in enumerate(metrics)]
        inner = select(*cols, *vals, LatticeRecordRow.id.label("rid")).where(
            and_(*_conditions(customer_id, lattice_id, where, search, _is_pg(self)))).subquery()
        g = [inner.c[f"g{i}"] for i in range(len(group_by))]
        outs = [func.count(inner.c.rid).label("n")] + [
            _AGG[fn](inner.c[f"m{j}"]).label(f"a{j}") for j, (fn, _c) in enumerate(metrics)]
        stmt = select(*g, *outs)
        if g:
            stmt = stmt.group_by(*g)
        idx, desc = order or (0, True)
        key = outs[idx] if 0 <= idx < len(outs) else outs[0]
        stmt = stmt.order_by(key.desc() if desc else key.asc(), *[x.asc() for x in g]).limit(limit)
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(stmt)).all()
            total = 1
            if g:
                total = int((await session.execute(
                    select(func.count()).select_from(select(*g).group_by(*g).subquery()))).scalar_one())
        out = []
        for r in rows:
            m = r._mapping
            out.append({"key": [m[f"g{i}"] for i in range(len(group_by))], "count": int(m["n"] or 0),
                        "values": [None if m[f"a{j}"] is None else float(m[f"a{j}"]) for j in range(len(metrics))]})
        return out, total

    # ------------------------------------------------- goal criteria (Q135 A)

    async def lattice_measure(self, customer_id: str, lattice_id: str, *,
                              where: Optional[list[tuple[str, str, str, Any]]] = None,
                              filled: Optional[list[str]] = None, distinct: Optional[str] = None,
                              source_of: Optional[str] = None, sample: int = 5) -> dict[str, Any]:
        """Counts over the matching rows in one pass: rows; filled (rows
        where every column in `filled` holds a value) with up to `sample`
        row keys that don't; distinct values of one column; distinct
        sources recorded on one column's cells."""
        conds = _conditions(customer_id, lattice_id, where, None, _is_pg(self))
        outs: list[Any] = [func.count().label("rows")]
        full = None
        if filled:
            full = and_(*[_value_expr(c, "text").is_not(None) for c in filled])
            outs.append(func.coalesce(func.sum(case((full, 1), else_=0)), 0).label("filled"))
        if distinct:
            outs.append(func.count(func.distinct(func.lower(_value_expr(distinct, "text")))).label("distinct"))
        src = None
        if source_of:
            src = LatticeRecordRow.cells[(source_of, "source")].as_string()
            outs.append(func.count(func.distinct(case((src != "", src), else_=None))).label("sources"))
        async with self.session() as session:  # type: ignore[attr-defined]
            m = (await session.execute(select(*outs).where(and_(*conds)))).one()._mapping
            out: dict[str, Any] = {"rows": int(m["rows"] or 0)}
            if filled:
                out["filled"] = int(m["filled"] or 0)
                miss = (await session.execute(select(LatticeRecordRow.row_key).where(and_(*conds)).where(~full)
                                              .order_by(LatticeRecordRow.created_at.asc(), LatticeRecordRow.row_key.asc())
                                              .limit(max(0, sample)))).scalars().all()
                out["missing_sample"] = list(miss)
            if distinct:
                out["distinct"] = int(m["distinct"] or 0)
            if source_of:
                out["sources"] = int(m["sources"] or 0)
            return out

    async def lattice_reference_coverage(self, customer_id: str, parent_id: str, parent_col: str, child_id: str,
                                         child_col: str, *, min_per: int = 1,
                                         parent_where: Optional[list[tuple[str, str, str, Any]]] = None,
                                         sample: int = 5) -> dict[str, Any]:
        """How many distinct values of parent_col (each game in the games
        lattice) have at least min_per rows in the child lattice whose
        child_col holds that value (its players' rows), with a sample of
        the ones that don't and the fewest / most children any parent has."""
        pg = _is_pg(self)
        pexpr = _value_expr(parent_col, "text")
        parents = (select(func.lower(pexpr).label("v"))
                   .where(and_(*_conditions(customer_id, parent_id, parent_where, None, pg)))
                   .where(pexpr.is_not(None)).distinct().subquery())
        cexpr = _value_expr(child_col, "text")
        inner = (select(func.lower(cexpr).label("v")).where(and_(*_conditions(customer_id, child_id, None, None, pg)))
                 .where(cexpr.is_not(None)).subquery())
        children = select(inner.c.v, func.count().label("n")).group_by(inner.c.v).subquery()
        joined = parents.outerjoin(children, parents.c.v == children.c.v)
        n = func.coalesce(children.c.n, 0)
        async with self.session() as session:  # type: ignore[attr-defined]
            m = (await session.execute(select(func.count().label("total"),
                                              func.coalesce(func.sum(case((n >= min_per, 1), else_=0)), 0).label("covered"),
                                              func.min(n).label("lo"), func.max(n).label("hi"))
                                       .select_from(joined))).one()._mapping
            miss = (await session.execute(select(parents.c.v).select_from(joined).where(n < min_per)
                                          .order_by(parents.c.v.asc()).limit(max(0, sample)))).scalars().all()
        return {"parents": int(m["total"] or 0), "covered": int(m["covered"] or 0),
                "fewest": int(m["lo"] or 0), "most": int(m["hi"] or 0), "missing_sample": [str(x) for x in miss]}
