"""QdrantVectorIndex — the fact_768 + routing_10k lanes on Qdrant (Step 2).

WHY (docs/VECTOR_STORE_RESEARCH.md §5/§6, decisions 2026-06-28/29)
------------------------------------------------------------------
The fact lane was the first lane to move off the in-memory ``FactVectorStore``
matrix onto Qdrant (the read-performance pick); the 10k crystal routing lane
follows it here (Step 2b). This backend implements the ``VectorIndex`` seam's
``search_facts`` (native-768) AND ``search_routing`` (10k routing) against
Qdrant. The fact lane is a plain HNSW collection; the routing lane is a SECOND
collection under BINARY QUANTIZATION + float rescore — the over-parameterized
10k vector is exactly the regime binary quant holds recall in (Step-0 proved
recall@1 = 1.000 at 10k, and the live Qdrant latency bench confirmed it). The
summary_10k lane stays in Postgres, fetched by id for the HDC unbind
(``get_summary_vector``) — storage + fetch, not a search.

SOURCE OF TRUTH = THE DATABASE (decision #2, 2026-06-29)
-------------------------------------------------------
Qdrant is a REBUILDABLE index, never an independent source of truth. Rather
than dual-writing each fact at save time (and then reconciling drift), this
backend LAZILY MIRRORS the DB: a customer's facts (and each subscribed general
bank) are loaded from ``MetadataStore`` into Qdrant on first ``search_facts``,
exactly as ``FactVectorStore`` lazily builds its matrix. ``invalidate(
customer_id)`` is synchronous and just marks the customer stale (no I/O on the
write path); the next ``search_facts`` drops that customer's mirrored points
and reloads from the DB. The index therefore cannot diverge from the DB, and
"rebuild" is just ``invalidate_all()`` followed by lazy reload. (Per-point
upsert on ingest is a later performance optimization; correctness first.)

PARITY BAR
----------
``search_facts`` reproduces ``FactVectorStore.search`` (infrastructure/
fact_vector_store.py): the customer bank (weight 1.0) merged with every
subscribed general bank (weight ``GENERAL_TIE_BREAK``, imported from there so
the two can't drift), a ``pair_type`` membership filter, score-descending sort,
the optional ACL ``operator`` post-filter (lazy ``can_read``, capped at k), and
``with_keys`` sparse-key passthrough. Qdrant's COSINE distance over
unit-normalized vectors equals the brute-force cosine the matrix computes
(validated to < 1e-5).

SINGLE-DIM ASSUMPTION
---------------------
One collection holds all fact points (payload-partitioned by scope /
customer_id / crystal_type — Qdrant's multi-tenancy pattern), so a single
vector dimension is assumed (one encoder per deployment, true today: native
768). Facts whose vector length differs from the collection's are skipped on
load (mirroring ``_build_bank``'s dim-mismatch skip); a query whose dimension
differs raises ValueError (a wrong-encoder misconfiguration should be loud).
This is a deliberate, narrow divergence from FactVectorStore, which tolerates a
general bank of a different dim by fail-soft skipping at query time — here a
genuine encoder mismatch is surfaced rather than silently degraded.

LAZINESS NOTE FOR THE FACTORY
-----------------------------
Import this module lazily (inside the qdrant backend branch), never at the top
of a runtime module — ``qdrant_client`` is an optional dependency, so importing
this unconditionally would break memory-backend deployments that don't install
it.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Optional

import numpy as np
import structlog

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    BinaryQuantization,
    BinaryQuantizationConfig,
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    MatchValue,
    PointStruct,
    QuantizationSearchParams,
    SearchParams,
    VectorParams,
)

from .fact_vector_store import GENERAL_TIE_BREAK
from .permissions import can_read

if TYPE_CHECKING:
    from ..models import Operator
    from .metadata_store import MetadataStore

logger = structlog.get_logger(__name__)

# Namespace for deriving deterministic Qdrant point IDs from string fact_ids
# (Qdrant requires uint/UUID ids). uuid5(NS, fact_id) is stable across reloads,
# so re-mirroring a fact upserts the same point rather than duplicating it.
_POINT_NS = uuid.UUID("6f4c7d2e-1b3a-5c6d-8e9f-0a1b2c3d4e5f")

# Per-scope candidate fetch ceiling for the merge. FactVectorStore scores every
# fact then takes top-k AFTER the customer+general merge; Qdrant returns top-N
# per query, so we over-fetch per scope and merge. For banks up to this size the
# merge is exact (the parity bar). A true oversample knob for very large banks
# is the performance follow-up (lands with per-point upsert).
_MERGE_FETCH = 256


class QdrantVectorIndex:
    """fact_768 + routing_10k lanes on Qdrant; summary_10k fetched from Postgres.

    Mirrors the DB lazily (load-on-search, drop-on-invalidate) so Qdrant stays a
    rebuildable index that cannot diverge from ``MetadataStore``. Reproduces
    ``FactVectorStore.search`` for the fact lane (see module docstring).
    """

    def __init__(
        self,
        *,
        client: AsyncQdrantClient,
        metadata_store: "MetadataStore",
        collection_name: str = "crys_facts",
        routing_collection_name: str = "crys_routing",
        routing_oversampling: float = 2.0,
    ) -> None:
        self._client = client
        self._meta = metadata_store
        self._collection = collection_name
        self._routing_collection = routing_collection_name
        self._routing_oversampling = float(routing_oversampling)

        # fact_768 collection state.
        self._dim: Optional[int] = None
        self._collection_ready = False
        self._needs_full_reset = False
        self._init_lock = asyncio.Lock()

        # routing_10k collection state (binary-quantized; see
        # _ensure_routing_collection).
        self._routing_dim: Optional[int] = None
        self._routing_collection_ready = False
        self._routing_needs_full_reset = False
        self._routing_init_lock = asyncio.Lock()

        # Fact lazy-load bookkeeping (mirrors FactVectorStore's caches).
        self._loaded: set[str] = set()           # customer_ids mirrored
        self._loaded_locks: dict[str, asyncio.Lock] = {}
        self._loaded_general: set[str] = set()    # crystal_types mirrored
        self._general_locks: dict[str, asyncio.Lock] = {}
        self._subs: dict[str, list[str]] = {}     # customer_id -> general types

        # Routing lazy-load bookkeeping (mirrors VectorStore's per-customer /
        # per-general caches). Routing takes its general types as a search arg,
        # so there is no subscription lookup here (unlike the fact lane).
        self._routing_loaded: set[str] = set()          # customer_ids mirrored
        self._routing_loaded_locks: dict[str, asyncio.Lock] = {}
        self._routing_loaded_general: set[str] = set()  # crystal_types mirrored
        self._routing_general_locks: dict[str, asyncio.Lock] = {}

    # ---- id / vector helpers -------------------------------------------------

    @staticmethod
    def _pid(fact_id: str) -> str:
        return str(uuid.uuid5(_POINT_NS, fact_id))

    @staticmethod
    def _norm(vec) -> list[float]:
        v = np.asarray(vec, dtype=np.float32)
        n = float(np.linalg.norm(v))
        if n > 0.0:
            v = v / n
        return v.tolist()

    # ---- collection / load ---------------------------------------------------

    async def _ensure_collection(self, dim: int) -> None:
        if self._collection_ready and not self._needs_full_reset:
            return
        async with self._init_lock:
            if self._needs_full_reset:
                # Deferred invalidate_all(): drop the whole collection, then
                # rebuild it fresh below. Done lazily here so invalidate_all
                # stays synchronous and off the write path.
                if await self._client.collection_exists(self._collection):
                    await self._client.delete_collection(self._collection)
                self._collection_ready = False
                self._dim = None
                self._needs_full_reset = False
            if self._collection_ready:
                return
            if not await self._client.collection_exists(self._collection):
                await self._client.create_collection(
                    self._collection,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
            self._dim = dim
            self._collection_ready = True

    async def _load_facts(self, facts: list, base_payload: dict) -> None:
        """Upsert one bank's facts as points. Skips empty banks and dim-mismatch
        facts — mirrors FactVectorStore._build_bank."""
        usable = [f for f in facts if f.vector and len(f.vector) > 0]
        if not usable:
            return
        await self._ensure_collection(len(usable[0].vector))
        dim = self._dim
        points: list[PointStruct] = []
        for f in usable:
            if len(f.vector) != dim:
                continue  # dimension mismatch, skip
            points.append(
                PointStruct(
                    id=self._pid(f.id),
                    vector=self._norm(f.vector),
                    payload={
                        **base_payload,
                        "fact_id": f.id,
                        "crystal_id": f.crystal_id,
                        "pair_type": f.pair_type,
                        "prompt_text": f.prompt_text or "",
                    },
                )
            )
        if points:
            await self._client.upsert(self._collection, points=points)

    def _loaded_lock_for(self, customer_id: str) -> asyncio.Lock:
        lock = self._loaded_locks.get(customer_id)
        if lock is None:
            lock = asyncio.Lock()
            self._loaded_locks[customer_id] = lock
        return lock

    async def _ensure_loaded(self, customer_id: str) -> None:
        if customer_id in self._loaded:
            return
        async with self._loaded_lock_for(customer_id):
            if customer_id in self._loaded:
                return
            # Clear any prior points for this customer before reloading so a
            # reload after invalidate() reflects DELETIONS — re-upserting alone
            # (deterministic point ids) would leave a deleted fact's point
            # behind. No-op on a cold first load (collection not yet created).
            await self._delete_customer_points(customer_id)
            facts = await self._meta.list_all_facts_for_customer(customer_id)
            await self._load_facts(
                facts, {"scope": "customer", "customer_id": customer_id}
            )
            self._loaded.add(customer_id)
            logger.info(
                "qdrant_vector_index.loaded_customer",
                customer_id=customer_id, total_facts=len(facts),
            )

    def _general_lock_for(self, crystal_type: str) -> asyncio.Lock:
        lock = self._general_locks.get(crystal_type)
        if lock is None:
            lock = asyncio.Lock()
            self._general_locks[crystal_type] = lock
        return lock

    async def _ensure_general_loaded(self, crystal_type: str) -> None:
        if crystal_type in self._loaded_general:
            return
        async with self._general_lock_for(crystal_type):
            if crystal_type in self._loaded_general:
                return
            await self._delete_general_points(crystal_type)
            facts = await self._meta.list_all_facts_general(crystal_type)
            await self._load_facts(
                facts, {"scope": "general", "crystal_type": crystal_type}
            )
            self._loaded_general.add(crystal_type)
            logger.info(
                "qdrant_vector_index.loaded_general",
                crystal_type=crystal_type, total_facts=len(facts),
            )

    async def _subscribed_types(self, customer_id: str) -> list[str]:
        subs = self._subs.get(customer_id)
        if subs is None:
            try:
                subs = await self._meta.get_customer_general_types(customer_id)
            except Exception:  # noqa: BLE001 — never break search on a subs lookup
                subs = []
            self._subs[customer_id] = subs
        return subs

    # ---- point deletion (used by the lazy reload path) -----------------------

    async def _delete_customer_points(self, customer_id: str) -> None:
        if not self._collection_ready:
            return
        await self._client.delete(
            self._collection,
            points_selector=FilterSelector(filter=Filter(must=[
                FieldCondition(key="scope", match=MatchValue(value="customer")),
                FieldCondition(key="customer_id", match=MatchValue(value=customer_id)),
            ])),
        )

    async def _delete_general_points(self, crystal_type: str) -> None:
        if not self._collection_ready:
            return
        await self._client.delete(
            self._collection,
            points_selector=FilterSelector(filter=Filter(must=[
                FieldCondition(key="scope", match=MatchValue(value="general")),
                FieldCondition(key="crystal_type", match=MatchValue(value=crystal_type)),
            ])),
        )

    # ---- invalidation (mirrors FactVectorStore; SYNCHRONOUS) -----------------
    # invalidate* only flip in-memory flags — no Qdrant I/O on the write path.
    # The deferred delete happens on the next search via _ensure_loaded /
    # _ensure_general_loaded (per customer / per type), or via _ensure_collection
    # for the full reset. This makes the seam a drop-in for FactVectorStore's
    # synchronous invalidate*, so write-path call sites migrate unchanged.

    def invalidate(self, customer_id: str) -> None:
        """Mark a customer stale on BOTH lanes; the next search drops that
        customer's mirrored points (facts and routing) and reloads from the DB.
        Synchronous + non-blocking (the deletes are deferred to the _ensure_*
        loaders, off the write path)."""
        self._loaded.discard(customer_id)
        self._loaded_locks.pop(customer_id, None)
        self._subs.pop(customer_id, None)
        self._routing_loaded.discard(customer_id)
        self._routing_loaded_locks.pop(customer_id, None)

    def invalidate_general(self, crystal_type: Optional[str] = None) -> None:
        """Mark general bank(s) stale on BOTH lanes; the next search of a given
        type drops its points and reloads. None marks every general bank stale."""
        if crystal_type is None:
            self._loaded_general.clear()
            self._general_locks.clear()
            self._routing_loaded_general.clear()
            self._routing_general_locks.clear()
            return
        self._loaded_general.discard(crystal_type)
        self._general_locks.pop(crystal_type, None)
        self._routing_loaded_general.discard(crystal_type)
        self._routing_general_locks.pop(crystal_type, None)

    def invalidate_all(self) -> None:
        """Drop all cached state on BOTH lanes. The whole-collection deletes are
        deferred to the next load (via the _ensure_*_collection reset flags),
        keeping this sync."""
        self._loaded.clear()
        self._loaded_locks.clear()
        self._loaded_general.clear()
        self._general_locks.clear()
        self._subs.clear()
        self._needs_full_reset = True
        self._routing_loaded.clear()
        self._routing_loaded_locks.clear()
        self._routing_loaded_general.clear()
        self._routing_general_locks.clear()
        self._routing_needs_full_reset = True

    # ---- fact lane search ----------------------------------------------------

    @staticmethod
    def _pair_type_condition(pair_types: Optional[list[str]]) -> list:
        if not pair_types:
            return []
        return [FieldCondition(key="pair_type", match=MatchAny(any=list(pair_types)))]

    async def _query_scope(self, must: list, q: list[float]) -> list:
        resp = await self._client.query_points(
            self._collection, query=q, query_filter=Filter(must=must),
            limit=_MERGE_FETCH, with_payload=True,
        )
        return resp.points

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
        await self._ensure_loaded(customer_id)

        q = np.asarray(query_vector, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm > 0.0:
            q = q / q_norm

        # A genuine encoder/dim mismatch should be loud (see module docstring).
        # When nothing is mirrored yet (self._dim is None) there is nothing to
        # check against — the searches below simply return empty.
        if self._dim is not None and q.shape[0] != self._dim:
            raise ValueError(
                f"Query vector dim {q.shape[0]} does not match fact collection "
                f"dim {self._dim} for customer {customer_id}."
            )

        q_list = q.tolist()
        pt_cond = self._pair_type_condition(pair_types)
        results: list[tuple] = []

        def _emit(points, weight: float) -> None:
            for p in points:
                pl = p.payload or {}
                score = float(p.score) * weight
                if with_keys:
                    results.append((
                        pl["fact_id"], pl["crystal_id"], pl["pair_type"],
                        score, pl.get("prompt_text", ""),
                    ))
                else:
                    results.append((
                        pl["fact_id"], pl["crystal_id"], pl["pair_type"], score,
                    ))

        if self._collection_ready:
            cust_pts = await self._query_scope(
                [FieldCondition(key="scope", match=MatchValue(value="customer")),
                 FieldCondition(key="customer_id", match=MatchValue(value=customer_id)),
                 *pt_cond],
                q_list,
            )
            _emit(cust_pts, 1.0)

        for crystal_type in await self._subscribed_types(customer_id):
            await self._ensure_general_loaded(crystal_type)
            if not self._collection_ready:
                continue
            gen_pts = await self._query_scope(
                [FieldCondition(key="scope", match=MatchValue(value="general")),
                 FieldCondition(key="crystal_type", match=MatchValue(value=crystal_type)),
                 *pt_cond],
                q_list,
            )
            _emit(gen_pts, GENERAL_TIE_BREAK)

        results.sort(key=lambda x: x[3], reverse=True)

        if operator is None:
            return results[:k]

        # Foundation F2 ACL post-filter — mirrors FactVectorStore.search (lazy
        # can_read, per-crystal verdict cache, capped at k). Duplicated rather
        # than shared to avoid perturbing the parity-bar store mid-slice; a
        # dedupe into a shared helper is a candidate once both backends settle.
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
                    verdict = can_read(crystal, operator, acls)
                verdicts[crystal_id] = verdict
            if verdict:
                allowed.append(row)
                if len(allowed) >= k:
                    break
        return allowed[:k]

    # ---- routing lane (10k, binary-quantized) --------------------------------
    # A SECOND Qdrant collection, parallel to the fact collection but under
    # binary quantization + float rescore (docs/VECTOR_STORE_RESEARCH.md §5).
    # Lazily DB-mirrored from each crystal's routing_vector exactly as the fact
    # lane mirrors FactRow.vector; reproduces VectorStore.search. Methods are
    # parallel to the fact-lane ones rather than a shared generalization, to
    # avoid perturbing the fact parity path mid-slice (dedupe is a candidate
    # once both lanes settle).

    async def _ensure_routing_collection(self, dim: int) -> None:
        if self._routing_collection_ready and not self._routing_needs_full_reset:
            return
        async with self._routing_init_lock:
            if self._routing_needs_full_reset:
                # Deferred invalidate_all(): drop + rebuild fresh below. Done
                # lazily here so invalidate_all stays synchronous.
                if await self._client.collection_exists(self._routing_collection):
                    await self._client.delete_collection(self._routing_collection)
                self._routing_collection_ready = False
                self._routing_dim = None
                self._routing_needs_full_reset = False
            if self._routing_collection_ready:
                return
            if not await self._client.collection_exists(self._routing_collection):
                await self._client.create_collection(
                    self._routing_collection,
                    # DOT (not COSINE): VectorStore.search scores unit-normalized
                    # routing rows against the RAW (un-normalized) query, i.e.
                    # |query| * cos. Stored vectors are unit-normalized in
                    # _load_routing and the query is sent raw in search_routing,
                    # so DOT reproduces that score exactly. Ranking is identical
                    # to cosine (|query| is constant per query), so the Step-0 /
                    # bench recall result transfers unchanged.
                    vectors_config=VectorParams(size=dim, distance=Distance.DOT),
                    # Binary quantization (1-bit codes + HNSW graph resident,
                    # float vectors mmap'd): the over-parameterized 10k routing
                    # vector is the regime BQ holds recall in, and the float
                    # rescore (QuantizationSearchParams in _query_routing_scope)
                    # resolves near-ties. Step-0 + the live bench proved
                    # recall@1 = 1.000 at 10k.
                    quantization_config=BinaryQuantization(
                        binary=BinaryQuantizationConfig(always_ram=True)
                    ),
                )
            self._routing_dim = dim
            self._routing_collection_ready = True

    async def _load_routing(self, crystals: list, base_payload: dict) -> None:
        """Upsert one bank's crystals as routing points. Skips crystals with no
        routing_vector and dim-mismatch rows — mirrors VectorStore._ensure_loaded
        (Phase 6.3: routing is on routing_vector, not summary_vector)."""
        usable = [c for c in crystals if c.routing_vector]
        if not usable:
            return
        await self._ensure_routing_collection(len(usable[0].routing_vector))
        dim = self._routing_dim
        points: list[PointStruct] = []
        for c in usable:
            if len(c.routing_vector) != dim:
                continue  # dimension mismatch within a bank, skip
            points.append(
                PointStruct(
                    id=self._pid(c.id),
                    vector=self._norm(c.routing_vector),
                    payload={
                        **base_payload,
                        "crystal_id": c.id,
                        "crystal_type": c.crystal_type,
                    },
                )
            )
        if points:
            await self._client.upsert(self._routing_collection, points=points)

    def _routing_loaded_lock_for(self, customer_id: str) -> asyncio.Lock:
        lock = self._routing_loaded_locks.get(customer_id)
        if lock is None:
            lock = asyncio.Lock()
            self._routing_loaded_locks[customer_id] = lock
        return lock

    async def _ensure_routing_loaded(self, customer_id: str) -> None:
        if customer_id in self._routing_loaded:
            return
        async with self._routing_loaded_lock_for(customer_id):
            if customer_id in self._routing_loaded:
                return
            # Drop any prior routing points for this customer before reloading so
            # a reload after invalidate() reflects DELETIONS / routing_vector
            # changes (deterministic ids alone would leave stale points). No-op
            # on a cold first load (collection not yet created).
            await self._delete_routing_customer_points(customer_id)
            crystals = await self._meta.list_crystals_for_customer(
                customer_id, include_recall_gated=False,
            )
            await self._load_routing(
                crystals, {"scope": "customer", "customer_id": customer_id}
            )
            self._routing_loaded.add(customer_id)
            logger.info(
                "qdrant_vector_index.routing_loaded_customer",
                customer_id=customer_id, total_crystals=len(crystals),
            )

    def _routing_general_lock_for(self, crystal_type: str) -> asyncio.Lock:
        lock = self._routing_general_locks.get(crystal_type)
        if lock is None:
            lock = asyncio.Lock()
            self._routing_general_locks[crystal_type] = lock
        return lock

    async def _ensure_routing_general_loaded(self, crystal_type: str) -> None:
        if crystal_type in self._routing_loaded_general:
            return
        async with self._routing_general_lock_for(crystal_type):
            if crystal_type in self._routing_loaded_general:
                return
            await self._delete_routing_general_points(crystal_type)
            crystals = await self._meta.list_general_crystals(crystal_type)
            await self._load_routing(
                crystals, {"scope": "general", "crystal_type": crystal_type}
            )
            self._routing_loaded_general.add(crystal_type)
            logger.info(
                "qdrant_vector_index.routing_loaded_general",
                crystal_type=crystal_type, total_crystals=len(crystals),
            )

    async def _delete_routing_customer_points(self, customer_id: str) -> None:
        if not self._routing_collection_ready:
            return
        await self._client.delete(
            self._routing_collection,
            points_selector=FilterSelector(filter=Filter(must=[
                FieldCondition(key="scope", match=MatchValue(value="customer")),
                FieldCondition(key="customer_id", match=MatchValue(value=customer_id)),
            ])),
        )

    async def _delete_routing_general_points(self, crystal_type: str) -> None:
        if not self._routing_collection_ready:
            return
        await self._client.delete(
            self._routing_collection,
            points_selector=FilterSelector(filter=Filter(must=[
                FieldCondition(key="scope", match=MatchValue(value="general")),
                FieldCondition(key="crystal_type", match=MatchValue(value=crystal_type)),
            ])),
        )

    async def _query_routing_scope(self, must: list, q: list[float]) -> list:
        resp = await self._client.query_points(
            self._routing_collection, query=q, query_filter=Filter(must=must),
            limit=_MERGE_FETCH,
            search_params=SearchParams(
                quantization=QuantizationSearchParams(
                    rescore=True, oversampling=self._routing_oversampling
                )
            ),
            with_payload=True,
        )
        return resp.points

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
        """Reproduce VectorStore.search on Qdrant's binary-quantized routing
        collection: the customer bank filtered to ``crystal_type`` merged with
        each ``general_crystal_types`` bank at RAW cosine (no tie-break factor —
        unlike the fact lane), customer-precedence on crystal_id collision,
        score-descending, top-k, then the optional ACL ``operator`` post-filter.
        General types are an explicit arg (the caller's subscription set), not
        looked up — matching VectorStore.search's signature."""
        await self._ensure_routing_loaded(customer_id)

        # NOTE: the query is sent RAW (NOT normalized). VectorStore.search
        # scores unit-normalized rows against the raw query (|query| * cos);
        # with DOT distance + unit-normalized stored vectors (see
        # _ensure_routing_collection / _load_routing), sending the raw query
        # reproduces that score. search_facts normalizes because FactVectorStore
        # does — the two lanes legitimately differ here.
        q = np.asarray(query_vector, dtype=np.float32)
        # A genuine encoder/dim mismatch should be loud (mirrors search_facts and
        # VectorStore.search's customer-bank ValueError). Nothing mirrored yet
        # (routing_dim is None) -> the searches below simply return empty.
        if self._routing_dim is not None and q.shape[0] != self._routing_dim:
            raise ValueError(
                f"Query vector dim {q.shape[0]} does not match routing collection "
                f"dim {self._routing_dim} for customer {customer_id}."
            )
        q_list = q.tolist()

        # Merge keyed by crystal_id: the customer bank fills first, a general
        # bank contributes a crystal_id only if the customer bank didn't — i.e.
        # the customer's version wins (VectorStore.search's `if cid not in
        # all_hits`). Raw cosine on both; routing has no GENERAL_TIE_BREAK.
        merged: dict[str, float] = {}

        if self._routing_collection_ready:
            cust_pts = await self._query_routing_scope(
                [FieldCondition(key="scope", match=MatchValue(value="customer")),
                 FieldCondition(key="customer_id", match=MatchValue(value=customer_id)),
                 FieldCondition(key="crystal_type", match=MatchValue(value=crystal_type))],
                q_list,
            )
            for p in cust_pts:
                cid = (p.payload or {}).get("crystal_id")
                if cid is not None:
                    merged[cid] = float(p.score)

        for gen_type in general_crystal_types or []:
            await self._ensure_routing_general_loaded(gen_type)
            if not self._routing_collection_ready:
                continue
            gen_pts = await self._query_routing_scope(
                [FieldCondition(key="scope", match=MatchValue(value="general")),
                 FieldCondition(key="crystal_type", match=MatchValue(value=gen_type))],
                q_list,
            )
            for p in gen_pts:
                cid = (p.payload or {}).get("crystal_id")
                if cid is not None and cid not in merged:
                    merged[cid] = float(p.score)

        if not merged:
            return []
        ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)

        if operator is None:
            return ranked[:k]

        # ACL post-filter — mirrors VectorStore.search / search_facts (lazy
        # can_read, per-crystal verdict cache, capped at k).
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
                    verdict = can_read(crystal, operator, acls)
                verdicts[cid] = verdict
            if verdict:
                allowed.append((cid, score))
                if len(allowed) >= k:
                    break
        return allowed[:k]

    # ---- summary lane (storage + fetch-by-id; not a search) ------------------

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
