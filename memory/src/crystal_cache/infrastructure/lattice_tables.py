"""Lattices as real tables (Q144 A, 2026-10-27).

Anthony: "lets not remake sql what can we use". Until Series 378 every
lattice row was a JSON object in one shared table and Crystal's code
imitated a database over it (filters, totals, joins, tenancy). Now each
lattice is its own table with typed columns, in a schema that belongs to
one customer:

  Postgres   schema t_<16 hex of the customer id>, table l_<lattice id>,
             and a view named for the lattice (nba_games) that the
             read-only SQL tool sees. A role r_<same hex> may read that
             customer's views and nothing else; the SQL tool runs as it.
  SQLite     (development and tests) one database, so the schema is a
             prefix: t_<hex>__l_<id>. The SQL tool makes temporary views
             per call and an authorizer refuses any other table.

Each table: _id (the row id), _key (the row's identity, from its key
columns, unique), one typed column per lattice column, _cells (each
cell's provenance and live history, JSON), _search (every text value,
lowercased) and _created_at / _updated_at. A column of type ref holds
another lattice's key value and is a foreign key to it: names live once,
in the lattice that owns them. R9: lattice SQL lives here and in
metadata_store_lattice_ext.py, nowhere else.
"""
from __future__ import annotations

import hashlib
import re
from datetime import date, datetime
from typing import Any, Optional

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

SQL_ROW_LIMIT = 200
SQL_TIMEOUT_MS = 5000
SQL_MAX_CHARS = 4000
_IDENT = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
# Catalogs name every customer's tables; the tool has no reason to read them.
_DENIED = re.compile(r"\b(pg_[a-z_]+|information_schema|set_config|current_setting|dblink|lo_import|lo_export|lo_get|lo_put)\b", re.I)
_SQL_TYPES = {"text": sa.Text, "url": sa.Text, "ref": sa.Text, "number": sa.Float, "date": sa.Date, "bool": sa.Boolean}


def tenant_hex(customer_id: str) -> str:
    return hashlib.sha1(customer_id.encode()).hexdigest()[:16]


def schema_name(customer_id: str) -> str:
    return f"t_{tenant_hex(customer_id)}"


def role_name(customer_id: str) -> str:
    return f"r_{tenant_hex(customer_id)}"


def physical_name(lattice_id: str) -> str:
    return "l_" + re.sub(r"[^a-z0-9]", "", lattice_id.lower().removeprefix("lat_"))[:40]


def sql_name(name: str, taken: set[str] = frozenset()) -> str:
    """The name the SQL tool knows a lattice by: lowercase words joined by _
    ("Games 2026-27" -> games_2026_27), unique within the customer."""
    base = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")[:48] or "lattice"
    if base[0].isdigit():
        base = f"t_{base}"[:48]
    out, n = base, 2
    while out in taken:
        out, n = f"{base}_{n}", n + 1
    return out


def is_pg(engine: Any) -> bool:
    return engine.dialect.name == "postgresql"


def located(engine: Any, customer_id: str, lattice_id: str) -> tuple[Optional[str], str]:
    """(schema, table name) as the engine addresses it."""
    if is_pg(engine):
        return schema_name(customer_id), physical_name(lattice_id)
    return None, f"{schema_name(customer_id)}__{physical_name(lattice_id)}"


def build(engine: Any, customer_id: str, lattice: dict[str, Any],
          refs: Optional[dict[str, tuple[str, str]]] = None) -> sa.Table:
    """The lattice's table as SQLAlchemy sees it. refs: {column key:
    (target lattice id, target key column)} for its ref columns."""
    schema, name = located(engine, customer_id, lattice["id"])
    md = sa.MetaData()
    cols: list[Any] = [sa.Column("_id", sa.Text, primary_key=True),
                       sa.Column("_key", sa.Text, nullable=False, unique=True)]
    for c in lattice["columns"]:
        cols.append(sa.Column(c["key"], _SQL_TYPES[c["type"]](), nullable=True))
    json_t = sa.JSON().with_variant(JSONB(), "postgresql")
    cols += [sa.Column("_cells", json_t, nullable=False, default=dict),
             sa.Column("_search", sa.Text, nullable=True),
             sa.Column("_created_at", sa.DateTime(timezone=True), nullable=False),
             sa.Column("_updated_at", sa.DateTime(timezone=True), nullable=False)]
    args: list[Any] = []
    if len(lattice["key_columns"]) == 1:
        # A lattice keyed on one column can be pointed at (a ref is a
        # foreign key to it). _key is unique and lowercases it, so the raw
        # values are unique too.
        args.append(sa.UniqueConstraint(lattice["key_columns"][0], name=f"ux_{name}_key"[:63]))
    for key, (target_id, target_col) in (refs or {}).items():
        t_schema, t_name = located(engine, customer_id, target_id)
        ref = f"{t_schema}.{t_name}.{target_col}" if t_schema else f"{t_name}.{target_col}"
        args.append(sa.ForeignKeyConstraint([key], [ref], name=f"fk_{name}_{key}"[:63], ondelete="RESTRICT"))
    return sa.Table(name, md, *cols, *args, schema=schema)


def stub_target(engine: Any, customer_id: str, target_id: str, target_col: str) -> sa.Table:
    """Enough of a referenced table for a foreign key to resolve against."""
    schema, name = located(engine, customer_id, target_id)
    md = sa.MetaData()
    return sa.Table(name, md, sa.Column(target_col, sa.Text), schema=schema)


def quote(engine: Any, ident: str) -> str:
    return engine.dialect.identifier_preparer.quote(ident)


def qualified(engine: Any, schema: Optional[str], name: str) -> str:
    return f"{quote(engine, schema)}.{quote(engine, name)}" if schema else quote(engine, name)


def index_ddl(engine: Any, table: sa.Table, lattice: dict[str, Any]) -> list[str]:
    """Browse order, case-insensitive lookups on keys and entities, and on
    Postgres a full-text index over _search."""
    q = qualified(engine, table.schema, table.name)
    out = [f"CREATE INDEX IF NOT EXISTS {quote(engine, ('ix_' + table.name + '_order')[:63])} ON {q} (_created_at, _key)"]
    wanted = set(lattice["key_columns"]) | {c["key"] for c in lattice["columns"] if c.get("entity") or c["type"] == "ref"}
    for c in lattice["columns"]:
        if c["key"] not in wanted:
            continue
        ix = quote(engine, f"ix_{table.name}_{c['key']}"[:63])
        expr = f"lower({quote(engine, c['key'])})" if c["type"] in ("text", "url") else quote(engine, c["key"])
        out.append(f"CREATE INDEX IF NOT EXISTS {ix} ON {q} (({expr}))")
    if is_pg(engine):
        out.append(f"CREATE INDEX IF NOT EXISTS {quote(engine, ('ix_' + table.name + '_search')[:63])} ON {q} "
                   "USING gin (to_tsvector('simple', coalesce(_search, '')))")
    return out


def view_columns(lattice: dict[str, Any]) -> list[str]:
    return [c["key"] for c in lattice["columns"]]


# ------------------------------------------------------------- values

def to_db(col: dict[str, Any], value: Any) -> Any:
    if value is None:
        return None
    if col["type"] == "date":
        return value if isinstance(value, date) else date.fromisoformat(str(value)[:10])
    if col["type"] == "number":
        return float(value)
    return value


def from_db(col_type: str, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if col_type == "number" and isinstance(value, float) and value.is_integer() and abs(value) < 2**53:
        return int(value)
    return value


def search_text(data: dict[str, Any]) -> Optional[str]:
    return " ".join(str(v) for v in data.values() if isinstance(v, str)).lower() or None


# ------------------------------------------------------------- the SQL tool

class SqlRefused(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def check_sql(query: Any) -> str:
    """One read-only statement, or a refusal. The database is the real
    wall (a role that may only read this customer's views, a read-only
    transaction); this keeps plain mistakes from reaching it."""
    q = str(query or "").strip().rstrip(";").strip()
    if not q:
        raise SqlRefused("sql_required", "Give one SELECT statement.")
    if len(q) > SQL_MAX_CHARS:
        raise SqlRefused("sql_too_long", f"Keep the query under {SQL_MAX_CHARS} characters.")
    bare = re.sub(r"'(?:[^']|'')*'", "''", q)
    bare = re.sub(r"--[^\n]*|/\*.*?\*/", " ", bare, flags=re.S)
    if ";" in bare:
        raise SqlRefused("sql_one_statement", "One statement at a time.")
    first = (bare.split(None, 1) or [""])[0].lower()
    if first not in ("select", "with"):
        raise SqlRefused("sql_read_only", "Only SELECT (or WITH ... SELECT) queries; lattices are written through their tools.")
    if _DENIED.search(bare):
        raise SqlRefused("sql_denied", "System catalogs and settings are off limits; query the lattices by name.")
    return q
