"""SQL for the sqlite-vec self-host vector backend (R9: all sqlite-vec SQL here).

This module is the SINGLE home for every sqlite-vec statement, mirroring the
project rule that raw SQL lives only in ``metadata_store*`` files.
``SqliteVecIndex`` (infrastructure/sqlite_vec_index.py) orchestrates — deciding
WHEN to rebuild and HOW to score/merge — but issues no SQL itself; it calls the
helpers here.

TWO LANES, TWO STRATEGIES (forced by a sqlite-vec limit)
--------------------------------------------------------
sqlite-vec 0.1.9 caps a ``vec0`` virtual-table vector COLUMN at 8192 dimensions.

  • fact_768   — 768 ≤ 8192, so facts live in a real ``vec0`` KNN index
                 (``vec_facts``), mirrored from the ``facts`` table and rebuilt
                 lazily per scope on invalidation (see below). This is the
                 backend's value-add: a true SIMD vector index in one container.
  • routing_10k — 10000 > 8192, so routing CANNOT use a vec0 column. Instead the
                 routing lane is a brute-force cosine scan computed in SQL with
                 sqlite-vec's scalar ``vec_distance_cosine(vec_f32(...), ...)``
                 (no dimension cap), read LIVE from ``crystals``. Routing banks
                 are small (one row per crystal, filtered by type), so this is
                 cheap and — crucially — it is exactly what the in-memory
                 ``VectorStore`` does, so parity is exact. Because it reads
                 ``crystals`` directly, the routing lane needs no mirror, no
                 rebuild, and no invalidation.

IN-SQLITE FACT REBUILD
----------------------
The fact index lives in the SAME SQLite file as the metadata store. A "rebuild"
of a scope's vectors never round-trips through Python — it is one
``INSERT ... SELECT`` that reads the JSON-text ``vector`` column straight out of
``facts`` and lets sqlite-vec parse it. Vectors never leave the DB.

PARTITION-KEY SENTINELS (fact lane only)
----------------------------------------
vec0 metadata columns are NOT NULL, and the only delete shape sqlite-vec makes
cheap is a pure PARTITION-KEY delete. So the general fact bank folds its
``crystal_type`` INTO the partition key: general rows store
``customer_id = "__general__::<crystal_type>"`` (see ``general_scope``).
Customer rows store the real customer_id. "Delete general fact bank for type T"
and "delete customer C's facts" are then both single-partition deletes. (The
routing lane has no vec0 table and so no sentinel.)

DISTANCE / SCORE
----------------
Cosine everywhere; sqlite-vec normalizes internally, so we store RAW vectors.
cosine similarity = ``1 - distance``. The fact lane uses that directly
(FactVectorStore normalizes both sides → cosine). The routing lane multiplies by
``‖query‖`` in the orchestrator to reproduce ``VectorStore.search``'s
``‖query‖·cos`` (unit-normalized rows, raw query).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import structlog
from sqlalchemy import event, text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from .metadata_store import MetadataStore

logger = structlog.get_logger(__name__)

# The fixed native + routing dimensions (gtr-t5-base native = 768; the bipolar
# HDC projection = 10000). The fact vec0 table hardcodes 768 in its DDL; the
# routing scan filters rows to 10000 via json_array_length so a stray-dim stored
# vector is skipped rather than crashing vec_distance_cosine.
VEC_FACTS_DIM = 768
VEC_ROUTING_DIM = 10000

# sqlite-vec hard-caps a vec0 KNN ``k`` at 4096 ("limit is 4096"). Plain
# (operator=None) searches fetch exactly k; when an ACL operator filter may
# reject top-ranked rows we over-fetch up to this cap. A bank with more than
# 4096 in-scope candidates AND heavy top-ranked ACL rejection is a documented
# self-host caveat (use a Qdrant server). The same cap bounds the routing scan's
# LIMIT for symmetry (routing banks are far smaller than this in practice).
VEC_KNN_MAX = 4096

_GENERAL_PREFIX = "__general__::"

_FACTS_TABLE = "vec_facts"


def general_scope(crystal_type: str) -> str:
    """Partition-key sentinel for the general FACT bank of ``crystal_type``."""
    return f"{_GENERAL_PREFIX}{crystal_type}"


# ---------------------------------------------------------------------------
# Extension loading
# ---------------------------------------------------------------------------

def register_sqlite_vec_loader(engine: "AsyncEngine") -> None:
    """Load the sqlite-vec extension on every pooled connection of ``engine``.

    No-op (with a one-time warning) when ``sqlite_vec`` isn't installed — the
    memory/qdrant backends never touch sqlite-vec, so a missing extension must
    not break store construction.

    aiosqlite runs sqlite3 in a worker thread and exposes only ASYNC
    enable_load_extension/load_extension coroutines on its Connection; calling
    those synchronously from the connect event is a no-op. So we reach through
    to the REAL sqlite3 connection (``driver_connection._conn``) and load on it.
    SQLAlchemy's aiosqlite dialect opens with ``check_same_thread=False``, so
    this cross-thread sync load is tolerated. The event re-fires per pooled
    connection, so file-DB deployments (AsyncAdaptedQueuePool, many connections)
    get the extension on each.
    """
    try:
        import sqlite_vec
    except ImportError:
        logger.warning(
            "sqlite_vec.not_installed",
            note="sqlite_vec backend unavailable; install crystal-cache[sqlite-vec]",
        )
        return

    @event.listens_for(engine.sync_engine, "connect")
    def _load_sqlite_vec(dbapi_conn, _rec):  # noqa: ANN001
        # dbapi_conn is SQLAlchemy's adapted connection. For aiosqlite it wraps
        # an aiosqlite.Connection at .driver_connection, whose real sqlite3
        # connection is at ._conn. For pysqlite, driver_connection IS the
        # sqlite3 connection and ._conn is absent (getattr falls through).
        raw = getattr(dbapi_conn, "driver_connection", dbapi_conn)
        raw = getattr(raw, "_conn", raw)
        try:
            raw.enable_load_extension(True)
            sqlite_vec.load(raw)
            raw.enable_load_extension(False)
        except Exception as exc:  # noqa: BLE001 — never break connection setup.
            # A broken load is surfaced loudly at first use (CREATE VIRTUAL
            # TABLE / vec_distance_cosine fail), not here, so non-vec backends
            # sharing this engine keep working.
            logger.warning(
                "sqlite_vec.load_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )


# ---------------------------------------------------------------------------
# Fact schema (idempotent) — only the fact lane has a vec0 table.
# ---------------------------------------------------------------------------

_CREATE_FACTS = text(
    f"CREATE VIRTUAL TABLE IF NOT EXISTS {_FACTS_TABLE} USING vec0("
    "customer_id TEXT partition key, "
    "crystal_type TEXT, "
    "pair_type TEXT, "
    "fact_id TEXT, "
    "crystal_id TEXT, "
    f"embedding float[{VEC_FACTS_DIM}] distance_metric=cosine)"
)


async def ensure_vec0_schema(store: "MetadataStore") -> None:
    """Create the fact vec0 virtual table if absent. Idempotent (IF NOT EXISTS).

    The routing lane has no vec0 table (it scans ``crystals`` live), so there is
    nothing to create for it.
    """
    async with store.engine.begin() as conn:
        await conn.execute(_CREATE_FACTS)


# ---------------------------------------------------------------------------
# Fact rebuilds — DELETE one partition, then INSERT...SELECT straight from the
# source tables. The json_valid + json_array_length guards skip rows whose
# stored vector is absent/empty/wrong-dim, matching the in-memory stores'
# `if f.vector and len(f.vector) > 0` + dimension-mismatch skip.
# ---------------------------------------------------------------------------

_DELETE_FACTS_PARTITION = text(
    f"DELETE FROM {_FACTS_TABLE} WHERE customer_id = :scope"
)

# Customer facts: partition = real customer_id; crystal_type col unused for the
# fact lane (facts are scoped by customer partition + pair_type), stored as ''.
# Membership mirrors list_all_facts_for_customer:
#   facts JOIN crystals ON facts.crystal_id = crystals.id WHERE customer_id = :cid
_REBUILD_FACTS_CUSTOMER = text(
    f"INSERT INTO {_FACTS_TABLE}"
    "(customer_id, crystal_type, pair_type, fact_id, crystal_id, embedding) "
    "SELECT c.customer_id, '', f.pair_type, f.id, f.crystal_id, f.vector "
    "FROM facts f JOIN crystals c ON f.crystal_id = c.id "
    "WHERE c.customer_id = :cid "
    "AND json_valid(f.vector) AND json_array_length(f.vector) = :dim"
)

# General facts: partition = general_scope(type). Membership mirrors
# list_all_facts_general: customer_id IS NULL AND crystal_type = :ctype.
_REBUILD_FACTS_GENERAL = text(
    f"INSERT INTO {_FACTS_TABLE}"
    "(customer_id, crystal_type, pair_type, fact_id, crystal_id, embedding) "
    "SELECT :scope, '', f.pair_type, f.id, f.crystal_id, f.vector "
    "FROM facts f JOIN crystals c ON f.crystal_id = c.id "
    "WHERE c.customer_id IS NULL AND c.crystal_type = :ctype "
    "AND json_valid(f.vector) AND json_array_length(f.vector) = :dim"
)


async def rebuild_facts_customer(store: "MetadataStore", customer_id: str) -> None:
    async with store.engine.begin() as conn:
        await conn.execute(_DELETE_FACTS_PARTITION, {"scope": customer_id})
        await conn.execute(
            _REBUILD_FACTS_CUSTOMER, {"cid": customer_id, "dim": VEC_FACTS_DIM}
        )


async def rebuild_facts_general(store: "MetadataStore", crystal_type: str) -> None:
    scope = general_scope(crystal_type)
    async with store.engine.begin() as conn:
        await conn.execute(_DELETE_FACTS_PARTITION, {"scope": scope})
        await conn.execute(
            _REBUILD_FACTS_GENERAL,
            {"scope": scope, "ctype": crystal_type, "dim": VEC_FACTS_DIM},
        )


# ---------------------------------------------------------------------------
# Fact KNN — filtered nearest-neighbour over one partition scope. sqlite-vec
# applies the metadata/partition filters BEFORE the k cut (exact filtered KNN),
# and returns the auto `distance` column. We over-fetch up to VEC_KNN_MAX.
# ---------------------------------------------------------------------------

async def knn_facts(
    store: "MetadataStore",
    *,
    scope: str,
    query_json: str,
    k: int,
    pair_types: Optional[list[str]] = None,
    with_keys: bool = False,
) -> list[tuple]:
    """KNN over one fact partition (a customer_id or a general_scope sentinel).

    Returns (fact_id, crystal_id, pair_type, distance) rows, or 5-tuples with a
    trailing prompt_text (joined back from ``facts``) when ``with_keys``.
    """
    fetch_k = min(int(k), VEC_KNN_MAX)
    params: dict[str, object] = {"q": query_json, "k": fetch_k, "scope": scope}
    where = ["embedding MATCH :q", "k = :k", "customer_id = :scope"]
    if pair_types:
        names = []
        for i, pt in enumerate(pair_types):
            key = f"pt{i}"
            params[key] = pt
            names.append(f":{key}")
        where.append(f"pair_type IN ({', '.join(names)})")

    if with_keys:
        # Alias the vec0 table so the join column references are unambiguous.
        where_j = [w.replace("embedding MATCH", "v.embedding MATCH")
                   .replace("customer_id =", "v.customer_id =")
                   .replace("pair_type IN", "v.pair_type IN")
                   for w in where]
        sql = (
            f"SELECT v.fact_id, v.crystal_id, v.pair_type, v.distance, f.prompt_text "
            f"FROM {_FACTS_TABLE} v LEFT JOIN facts f ON v.fact_id = f.id "
            f"WHERE {' AND '.join(where_j)}"
        )
    else:
        sql = (
            f"SELECT fact_id, crystal_id, pair_type, distance "
            f"FROM {_FACTS_TABLE} "
            f"WHERE {' AND '.join(where)}"
        )
    async with store.session() as session:
        result = await session.execute(text(sql), params)
        return [tuple(row) for row in result.all()]


# ---------------------------------------------------------------------------
# Routing search — brute-force cosine over `crystals` via the scalar
# vec_distance_cosine (no vec0 table; no 8192 cap). Read live from `crystals`,
# so always fresh. ORDER BY ascending distance == descending cosine; per-scope
# LIMIT k is sufficient because the global top-k is a subset of each scope's
# top-k (customer/general crystal_ids are disjoint, so customer-precedence never
# discards a row that would otherwise rank). The json guards skip stored vectors
# that are absent/invalid/wrong-dim (mirrors VectorStore's `usable` + dim skip).
# k is interpolated as a validated int (parameterised LIMIT is brittle across
# sqlite drivers); every other value is a bound parameter.
# ---------------------------------------------------------------------------

_ROUTING_DISTANCE = (
    "vec_distance_cosine(vec_f32(c.routing_vector), vec_f32(:q)) AS dist"
)
_ROUTING_USABLE = (
    "c.routing_vector IS NOT NULL "
    "AND json_valid(c.routing_vector) "
    "AND json_array_length(c.routing_vector) = :dim"
)


async def routing_search_customer(
    store: "MetadataStore",
    *,
    customer_id: str,
    crystal_type: str,
    query_json: str,
    k: int,
) -> list[tuple]:
    """Brute-force routing scan over one customer's bank of ``crystal_type``.

    Mirrors VectorStore's customer bank (list_crystals_for_customer_and_type).
    Returns (crystal_id, distance) rows, nearest first.
    """
    limit = min(int(k), VEC_KNN_MAX)
    sql = (
        f"SELECT c.id, {_ROUTING_DISTANCE} FROM crystals c "
        f"WHERE c.customer_id = :cid AND c.crystal_type = :ctype "
        f"AND {_ROUTING_USABLE} ORDER BY dist ASC LIMIT {limit}"
    )
    params = {"q": query_json, "cid": customer_id, "ctype": crystal_type,
              "dim": VEC_ROUTING_DIM}
    async with store.session() as session:
        result = await session.execute(text(sql), params)
        return [tuple(row) for row in result.all()]


async def routing_search_general(
    store: "MetadataStore",
    *,
    crystal_type: str,
    query_json: str,
    k: int,
) -> list[tuple]:
    """Brute-force routing scan over the general bank of ``crystal_type``.

    Mirrors VectorStore's general bank (list_general_crystals).
    Returns (crystal_id, distance) rows, nearest first.
    """
    limit = min(int(k), VEC_KNN_MAX)
    sql = (
        f"SELECT c.id, {_ROUTING_DISTANCE} FROM crystals c "
        f"WHERE c.customer_id IS NULL AND c.crystal_type = :ctype "
        f"AND {_ROUTING_USABLE} ORDER BY dist ASC LIMIT {limit}"
    )
    params = {"q": query_json, "ctype": crystal_type, "dim": VEC_ROUTING_DIM}
    async with store.session() as session:
        result = await session.execute(text(sql), params)
        return [tuple(row) for row in result.all()]
