"""SqliteVecIndex — the zero-ops, single-container self-host VectorIndex.

WHAT THIS IS (docs/VECTOR_STORE_RESEARCH.md §5/§8)
--------------------------------------------------
The sqlite-vec backend keeps the vector index in the SAME SQLite file as the
metadata store, so a deployment needs ONE container and no extra service. This
class is the orchestration half: it decides WHEN a scope's vec0 rows need
rebuilding and HOW to score / merge / ACL-filter results, reproducing the
in-memory ``FactVectorStore`` / ``VectorStore`` behavior exactly. Every line of
vec0 SQL lives in ``metadata_store_vec`` (R9); this file issues none.

FACT LANE — LAZY REBUILD ON INVALIDATE (ratified 2026-06-30)
------------------------------------------------------------
The fact lane is a vec0 KNN index mirrored from ``facts``. The write path stays
untouched: a write goes through ``MetadataStore`` (the DB is the source of
truth), then the existing call site fires the SYNCHRONOUS ``invalidate*`` —
which here only flips a freshness flag (no I/O). The next fact SEARCH that
touches a stale scope does one bulk ``INSERT ... SELECT`` rebuild of just that
scope's vec0 partition, in-SQLite, then queries it. A burst of N writes to one
scope therefore costs ONE rebuild on the next read, not N. This mirrors the
Qdrant backend's "mark stale on invalidate, reload lazily on search" contract.

Fact rebuild + KNN run under a single ``asyncio.Lock`` so concurrent searches
can't race a half-rebuilt partition; ``invalidate*`` stay lock-free (a flag
flip), exactly as ``FactVectorStore.invalidate*`` are synchronous. Worst case
under a race is one redundant rebuild on the next search — never a stale or
partial result. (Single-container modest scale is the advertised envelope; the
Qdrant server backend is the power option for high concurrency.)

ROUTING LANE — LIVE BRUTE-FORCE (no mirror, no rebuild)
-------------------------------------------------------
sqlite-vec 0.1.9 caps a vec0 vector column at 8192 dims, and routing vectors are
10000-d, so routing cannot use a vec0 index. Instead it is a brute-force cosine
scan computed in SQL with the scalar ``vec_distance_cosine`` (no dimension cap),
read LIVE from ``crystals``. Routing banks are small (one row per crystal,
filtered by type), so this is cheap and exactly reproduces ``VectorStore``.
Reading ``crystals`` directly means the routing lane is ALWAYS fresh: it keeps
no mirror, needs no rebuild, and ``invalidate*`` are no-ops for it (they only
affect the fact mirror). Routing searches therefore take no lock.

SCORING (identical to the in-memory stores)
--------------------------------------------
Cosine everywhere; sqlite-vec normalizes internally, so we store RAW vectors.
cosine similarity = ``1 - distance``.
  • fact_768   — score = (1 - distance) * weight; customer weight 1.0, each
                 subscribed general bank ``GENERAL_TIE_BREAK`` (0.995). Reproduces
                 ``FactVectorStore.search`` (normalize both sides → cosine).
  • routing_10k — score = (1 - distance) * ‖query‖. Reproduces
                 ``VectorStore.search``'s ``‖query‖·cos`` (unit-normalized rows,
                 raw query), customer-precedence on crystal_id collision, NO
                 tie-break factor. ``general_crystal_types`` is the caller's
                 explicit subscription arg (matches VectorStore.search).
"""
from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Optional

import numpy as np

from . import metadata_store_vec as msv
from .fact_vector_store import GENERAL_TIE_BREAK
from .permissions import can_read

if TYPE_CHECKING:
    from ..models import Operator
    from .metadata_store import MetadataStore


def _vector_json(query_vector) -> str:
    """Encode a query vector as the JSON-text array sqlite-vec MATCH accepts."""
    arr = np.asarray(query_vector, dtype=np.float32).ravel()
    return json.dumps(arr.tolist())


class SqliteVecIndex:
    """VectorIndex backed by sqlite-vec ``vec0`` tables in the metadata DB.

    Constructed only for a SQLite metadata store (``build_vector_index`` guards
    the Postgres case). Holds no vectors itself — the vec0 tables live in the
    same ``.db`` and are (re)built straight from ``facts`` / ``crystals``.
    """

    def __init__(self, *, metadata_store: "MetadataStore") -> None:
        self._meta = metadata_store
        self._schema_ready = False
        self._lock = asyncio.Lock()
        # Fact-lane freshness sets: a scope present here has up-to-date vec0
        # rows; absent (or invalidated) → rebuild on next fact search. The
        # routing lane keeps no freshness state — it reads `crystals` live.
        self._fresh_facts_customer: set[str] = set()
        self._fresh_facts_general: set[str] = set()

    # -- schema ---------------------------------------------------------------

    async def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        await msv.ensure_vec0_schema(self._meta)
        self._schema_ready = True

    # -- subscription (fact-lane general banks) -------------------------------

    async def _subscribed_types(self, customer_id: str) -> list[str]:
        # Read fresh each search (matches InMemoryVectorIndex →
        # FactVectorStore._subscribed_types); fail-closed so a subs lookup error
        # never breaks search.
        try:
            return await self._meta.get_customer_general_types(customer_id)
        except Exception:  # noqa: BLE001
            return []

    # -- fact lane (768) ------------------------------------------------------

    async def search_facts(
        self,
        *,
        customer_id: str,
        query_vector: np.ndarray,
        pair_types: Optional[list[str]] = None,
        k: int = 10,
        operator: Optional["Operator"] = None,
        with_keys: bool = False,
    ) -> list[tuple]:
        q_json = _vector_json(query_vector)
        subscribed = await self._subscribed_types(customer_id)
        # Plain search fetches exactly k; an ACL filter may reject top-ranked
        # rows, so over-fetch up to the sqlite-vec cap and trim post-filter.
        fetch_k = min(int(k), msv.VEC_KNN_MAX) if operator is None else msv.VEC_KNN_MAX

        results: list[tuple] = []
        async with self._lock:
            await self._ensure_schema()

            if customer_id not in self._fresh_facts_customer:
                await msv.rebuild_facts_customer(self._meta, customer_id)
                self._fresh_facts_customer.add(customer_id)
            cust_rows = await msv.knn_facts(
                self._meta,
                scope=customer_id,
                query_json=q_json,
                k=fetch_k,
                pair_types=pair_types,
                with_keys=with_keys,
            )
            self._emit(cust_rows, 1.0, with_keys, results)

            for crystal_type in subscribed:
                if crystal_type not in self._fresh_facts_general:
                    await msv.rebuild_facts_general(self._meta, crystal_type)
                    self._fresh_facts_general.add(crystal_type)
                gen_rows = await msv.knn_facts(
                    self._meta,
                    scope=msv.general_scope(crystal_type),
                    query_json=q_json,
                    k=fetch_k,
                    pair_types=pair_types,
                    with_keys=with_keys,
                )
                self._emit(gen_rows, GENERAL_TIE_BREAK, with_keys, results)

        results.sort(key=lambda x: x[3], reverse=True)

        if operator is None:
            return results[:k]
        return await self._acl_filter_facts(results, operator, k)

    @staticmethod
    def _emit(rows, weight: float, with_keys: bool, out: list[tuple]) -> None:
        # knn_facts rows: (fact_id, crystal_id, pair_type, distance[, prompt_text]).
        for r in rows:
            score = (1.0 - float(r[3])) * weight
            if with_keys:
                out.append((r[0], r[1], r[2], score, r[4]))
            else:
                out.append((r[0], r[1], r[2], score))

    async def _acl_filter_facts(
        self, results: list[tuple], operator: "Operator", k: int
    ) -> list[tuple]:
        # Mirrors FactVectorStore.search / QdrantVectorIndex.search_facts: lazy
        # can_read, per-crystal verdict cache, capped at k.
        # P3: fetch the operator's group memberships ONCE so 'group' grants
        # resolve (can_read fail-closes group grants without the set).
        group_ids = await self._meta.list_group_ids_for_operator(operator.id)
        allowed: list[tuple] = []
        verdicts: dict[str, bool] = {}
        for row in results:
            crystal_id = row[1]
            verdict = verdicts.get(crystal_id)
            if verdict is None:
                crystal = await self._meta.get_crystal(crystal_id)
                if crystal is None:
                    verdict = False
                else:
                    acls = await self._meta.list_acls_for_crystal(crystal_id)
                    verdict = can_read(crystal, operator, acls, group_ids)
                verdicts[crystal_id] = verdict
            if verdict:
                allowed.append(row)
                if len(allowed) >= k:
                    break
        return allowed[:k]

    # -- routing lane (10k) ---------------------------------------------------

    async def search_routing(
        self,
        *,
        customer_id: str,
        query_vector: np.ndarray,
        k: int = 5,
        crystal_type: str,
        general_crystal_types: Optional[list[str]] = None,
        operator: Optional["Operator"] = None,
    ) -> list[tuple[str, float]]:
        q = np.asarray(query_vector, dtype=np.float32)
        # ‖query‖ recreates VectorStore.search's |query|*cos (cosine distance
        # gives cos = 1 - distance regardless of query magnitude; reapply norm).
        q_norm = float(np.linalg.norm(q))
        q_json = _vector_json(q)
        fetch_k = min(int(k), msv.VEC_KNN_MAX) if operator is None else msv.VEC_KNN_MAX

        # Live brute-force over `crystals` — no lock, no mirror, no rebuild.
        # Merge keyed by crystal_id: customer bank fills first, a general bank
        # contributes a crystal_id only if the customer bank didn't
        # (customer-precedence; VectorStore.search's `if cid not in all_hits`).
        merged: dict[str, float] = {}

        cust_rows = await msv.routing_search_customer(
            self._meta,
            customer_id=customer_id,
            crystal_type=crystal_type,
            query_json=q_json,
            k=fetch_k,
        )
        for cid, distance in cust_rows:
            if cid is not None:
                merged[cid] = (1.0 - float(distance)) * q_norm

        for gen_type in general_crystal_types or []:
            gen_rows = await msv.routing_search_general(
                self._meta,
                crystal_type=gen_type,
                query_json=q_json,
                k=fetch_k,
            )
            for cid, distance in gen_rows:
                if cid is not None and cid not in merged:
                    merged[cid] = (1.0 - float(distance)) * q_norm

        if not merged:
            return []
        ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)

        if operator is None:
            return ranked[:k]
        return await self._acl_filter_routing(ranked, operator, k)

    async def _acl_filter_routing(
        self, ranked: list[tuple[str, float]], operator: "Operator", k: int
    ) -> list[tuple[str, float]]:
        # P3: group memberships fetched once (see _acl_filter_facts).
        group_ids = await self._meta.list_group_ids_for_operator(operator.id)
        allowed: list[tuple[str, float]] = []
        verdicts: dict[str, bool] = {}
        for cid, score in ranked:
            verdict = verdicts.get(cid)
            if verdict is None:
                crystal = await self._meta.get_crystal(cid)
                if crystal is None:
                    verdict = False
                else:
                    acls = await self._meta.list_acls_for_crystal(cid)
                    verdict = can_read(crystal, operator, acls, group_ids)
                verdicts[cid] = verdict
            if verdict:
                allowed.append((cid, score))
                if len(allowed) >= k:
                    break
        return allowed[:k]

    # -- summary lane (fetch-by-id; not a search) -----------------------------

    async def get_summary_vector(
        self,
        *,
        customer_id: str,
        crystal_id: str,
    ) -> Optional[list[float]]:
        crystal = await self._meta.get_crystal(crystal_id)
        if crystal is None or not crystal.summary_vector:
            return None
        # Cross-tenant guard (identical to InMemoryVectorIndex): routing never
        # crosses customers, so this only rejects misuse. General crystals
        # (customer_id None) are world-readable and pass.
        if crystal.customer_id is not None and crystal.customer_id != customer_id:
            return None
        return list(crystal.summary_vector)

    # -- invalidation (synchronous flag flips; lazy rebuild on next fact search;
    #    routing reads `crystals` live, so these are no-ops for routing) --------

    def invalidate(self, customer_id: str) -> None:
        self._fresh_facts_customer.discard(customer_id)

    def invalidate_general(self, crystal_type: Optional[str] = None) -> None:
        if crystal_type is None:
            self._fresh_facts_general.clear()
        else:
            self._fresh_facts_general.discard(crystal_type)

    def invalidate_all(self) -> None:
        self._fresh_facts_customer.clear()
        self._fresh_facts_general.clear()
