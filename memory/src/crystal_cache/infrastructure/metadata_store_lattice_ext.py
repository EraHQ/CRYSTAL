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

import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from sqlalchemy import Float, and_, cast, delete, func, or_, select, update

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


def _conditions(customer_id: str, lattice_id: str, where: Optional[list[tuple[str, str, str, Any]]],
                search: Optional[tuple[list[str], str]]) -> list[Any]:
    conds = [LatticeRecordRow.lattice_id == lattice_id, LatticeRecordRow.customer_id == customer_id]
    for col, typ, op, value in where or []:
        expr = _value_expr(col, typ)
        if op == "eq":
            conds.append(func.lower(expr) == str(value).lower() if typ in ("text", "date", "url") else expr == value)
        elif op == "contains":
            conds.append(func.lower(_value_expr(col, "text")).like(f"%{str(value).lower()}%"))
        elif op == "gte":
            conds.append(expr >= value)
        elif op == "lte":
            conds.append(expr <= value)
    if search and search[0] and search[1]:
        needle = f"%{search[1].lower()}%"
        conds.append(or_(*[func.lower(_value_expr(c, "text")).like(needle) for c in search[0]]))
    return conds


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
                session.add(LatticeRecordRow(id=rid, lattice_id=lattice_id, customer_id=customer_id,
                                             row_key=r["row_key"], data=dict(r["data"]), cells=dict(r["cells"]),
                                             created_at=now, updated_at=now))
            for r in updates:
                await session.execute(update(LatticeRecordRow).where(LatticeRecordRow.id == r["id"])
                                      .where(LatticeRecordRow.lattice_id == lattice_id)
                                      .where(LatticeRecordRow.customer_id == customer_id)
                                      .values(data=dict(r["data"]), cells=dict(r["cells"]), updated_at=now))
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
        conds = _conditions(customer_id, lattice_id, where, search)
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
            and_(*_conditions(customer_id, lattice_id, where, search))).subquery()
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
