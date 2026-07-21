"""VectorIndex — the backend-agnostic seam over CRYS's three vector lanes.

WHY THIS EXISTS (docs/VECTOR_STORE_RESEARCH.md §5/§6, decision 2026-06-28)
--------------------------------------------------------------------------
Retrieval today talks to two concrete in-memory stores (`FactVectorStore`,
`VectorStore`) and reads `Crystal.summary_vector` inline for recall. To move
the routing/fact lanes onto Qdrant (read-performance pick) without forking
every call site, those call sites must talk to ONE interface; swapping the
backend (in-memory → Qdrant → pgvectorscale) then becomes a construction
change, not a code-wide rewrite.

THE THREE LANES (each a different job; see §1 of the research doc)
------------------------------------------------------------------
  • fact_768   — agent / MCP retrieval. `FactVectorStore.search` today: a
                 per-customer in-memory matrix with pair_type filtering, an
                 optional ACL `operator` filter, `with_keys` sparse-key
                 passthrough, and a general-bank merge. ANN-indexed lane.
  • routing_10k — proxy / SDK "which crystal" routing. `VectorStore.search`
                 today: per-(customer, crystal_type) matrices, cosine over the
                 `Σ encode(prompt) @ P` routing accumulator. ANN-indexed lane.
  • summary_10k — recall's HDC unbind. NOT a similarity search: routing picks
                 the crystal, then recall fetches THAT crystal's
                 `summary_vector` by id and unbinds. Storage + fetch-by-id only.

GROUNDED, NOT IDEALIZED
-----------------------
The §5 sketch used a generic `search(lane=...)`. Grounding against the real
stores, the two ANN lanes have incompatible search signatures
(`pair_types`/`with_keys` vs `crystal_type`/`general_crystal_types`), so a
generic signature would degrade to a `**kwargs` dict and lose type-safety.
Hence three lane-specific methods. Lane semantics (pair_type filter, ACL
filter, general-bank merge) are part of the read CONTRACT every backend must
honor, so they live on the interface — a Qdrant backend maps pair_types →
payload filter, operator → ACL post-filter, etc.

SCOPE (Step 2 — fact lane on Qdrant)
------------------------------------
This is the read seam plus cache INVALIDATION. There is still no `upsert`/
`delete` of individual points: a write goes through `MetadataStore.add_pair_*`
(the DB is the source of truth), then `invalidate()` tells the backend that a
customer's facts changed. The in-memory backend drops its cached matrix; the
Qdrant backend (`QdrantVectorIndex`) drops that customer's mirrored points and
re-loads from the DB lazily on the next search — so Qdrant stays a rebuildable
DB mirror, never an independent store. `invalidate*` is SYNCHRONOUS — a drop-in
match for `FactVectorStore.invalidate*`, so existing write-path call sites keep
their exact semantics and the fact lane can migrate to the seam incrementally.
The Qdrant backend's `invalidate` just marks the customer stale (no I/O on the
write path); the actual point delete + reload happens lazily on the next
search. `InMemoryVectorIndex` wraps the SAME store instances the app already
builds and delegates to their `invalidate*`.

These methods cover the FACT lane (the only lane on a non-in-memory backend so
far). The routing store keeps its own direct `invalidate()` at the routing
write sites until `routing_10k` also moves to Qdrant; summary is fetch-by-id
(nothing to invalidate).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from ..models import Operator
    from .fact_vector_store import FactVectorStore
    from .metadata_store import MetadataStore
    from .vector_store import VectorStore


@runtime_checkable
class VectorIndex(Protocol):
    """Read surface every vector backend implements. Method signatures mirror
    the concrete stores they replace, so call sites are backend-agnostic."""

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
        """fact_768 lane. Mirrors `FactVectorStore.search`: returns
        (fact_id, crystal_id, pair_type, score) tuples sorted desc, or
        5-tuples (+prompt_text) when with_keys=True; general-bank merge and
        the optional ACL operator filter are applied by the backend."""
        ...

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
        """routing_10k lane. Mirrors `VectorStore.search`: returns
        (crystal_id, cosine) sorted desc, merging subscribed general banks and
        applying the optional ACL operator filter."""
        ...

    async def get_summary_vector(
        self,
        *,
        customer_id: str,
        crystal_id: str,
    ) -> Optional[list[float]]:
        """summary_10k lane: fetch one crystal's `summary_vector` by id for the
        recall unbind (not a search). None if the crystal is absent, has no
        summary_vector, or belongs to a different customer."""
        ...

    def invalidate(self, customer_id: str) -> None:
        """Fact lane: signal that this customer's facts changed. The in-memory
        backend drops its cached matrix; the Qdrant backend marks the customer
        stale so the next search drops its mirrored points and reloads from the
        DB. Synchronous — a drop-in match for FactVectorStore.invalidate, so the
        Qdrant delete is deferred to the next search, off the write path."""
        ...

    def invalidate_general(self, crystal_type: Optional[str] = None) -> None:
        """Fact lane: drop general-bank fact cache(s). None drops every general
        bank; a crystal_type drops just that one (after the seed importer
        writes)."""
        ...

    def invalidate_all(self) -> None:
        """Fact lane: drop all cached fact state."""
        ...


class InMemoryVectorIndex:
    """The current behavior, behind the interface.

    Wraps the same `FactVectorStore`, `VectorStore`, and `MetadataStore`
    instances the app/runtime already construct, and delegates verbatim — so
    introducing the seam changes no behavior. Kept permanently as the
    tests/offline backend even after Qdrant lands (the research doc's
    `InMemoryVectorIndex`).
    """

    def __init__(
        self,
        *,
        fact_store: "FactVectorStore",
        vector_store: "VectorStore",
        metadata_store: "MetadataStore",
    ) -> None:
        self._facts = fact_store
        self._routing = vector_store
        self._meta = metadata_store

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
        return await self._facts.search(
            customer_id,
            query_vector,
            pair_types=pair_types,
            k=k,
            operator=operator,
            with_keys=with_keys,
        )

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
        return await self._routing.search(
            customer_id,
            query_vector,
            k,
            crystal_type=crystal_type,
            general_crystal_types=general_crystal_types,
            operator=operator,
        )

    async def get_summary_vector(
        self,
        *,
        customer_id: str,
        crystal_id: str,
    ) -> Optional[list[float]]:
        crystal = await self._meta.get_crystal(crystal_id)
        if crystal is None or not crystal.summary_vector:
            return None
        # Cross-tenant guard: routing never routes across customers, so this
        # only ever rejects misuse — it can't change correct behavior. General
        # crystals (customer_id None) are world-readable and pass.
        if crystal.customer_id is not None and crystal.customer_id != customer_id:
            return None
        return list(crystal.summary_vector)

    def invalidate(self, customer_id: str) -> None:
        # Both lanes: facts (FactVectorStore) AND routing (VectorStore). Mirrors
        # QdrantVectorIndex.invalidate, which clears both — so one
        # vector_index.invalidate(customer) refreshes routing and facts in
        # either backend (the bonding / delete / promotion write sites rely on
        # this for the routing lane once they invalidate via the seam).
        self._facts.invalidate(customer_id)
        self._routing.invalidate(customer_id)

    def invalidate_general(self, crystal_type: Optional[str] = None) -> None:
        self._facts.invalidate_general(crystal_type)
        self._routing.invalidate_general(crystal_type)

    def invalidate_all(self) -> None:
        self._facts.invalidate_all()
        self._routing.invalidate_all()


def build_vector_index(
    *,
    backend: str,
    fact_store: "FactVectorStore",
    vector_store: "VectorStore",
    metadata_store: "MetadataStore",
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    qdrant_location: Optional[str] = None,
    qdrant_collection: str = "crys_facts",
    qdrant_routing_collection: str = "crys_routing",
    qdrant_routing_oversampling: float = 2.0,
) -> VectorIndex:
    """Construct the configured VectorIndex backend.

    backend="memory" (the default everywhere today) -> InMemoryVectorIndex,
    wrapping the in-memory stores the caller already built — zero behavior
    change.

    backend="qdrant" -> QdrantVectorIndex: the fact_768 lane (plain HNSW) AND
    the routing_10k lane (a second collection, binary-quantized + float
    rescore) on Qdrant; summary_10k is fetched by id from the DB. The Qdrant
    client targets ``qdrant_url`` (a running server) when given, else
    ``qdrant_location`` (an embedded path or
    ":memory:", defaulting to an ephemeral in-memory Qdrant). ``qdrant_client``
    and ``QdrantVectorIndex`` are imported HERE, lazily, so a memory-backend
    deployment never needs qdrant-client installed.

    backend="sqlite_vec" -> SqliteVecIndex: the fact_768 lane as a vec0 KNN
    index AND the routing_10k lane as a live vec_distance_cosine scan, both in
    the SAME SQLite metadata DB (one container, no extra service). Requires a
    SQLite metadata_store; raises for Postgres (use the Qdrant server there).
    ``SqliteVecIndex`` is imported HERE, lazily, like the Qdrant backend.
    """
    if backend == "sqlite_vec":
        # Zero-ops single-container self-host: the vec0 fact index + the routing
        # scan live in the SAME SQLite metadata DB. REQUIRES a SQLite store —
        # there is no Postgres equivalent of the vec0 tables, so a Postgres
        # deployment must use the Qdrant server backend instead.
        if metadata_store.engine.dialect.name != "sqlite":
            raise ValueError(
                "vector_backend 'sqlite_vec' requires a SQLite metadata store "
                f"(got dialect {metadata_store.engine.dialect.name!r}); use "
                "backend 'qdrant' with a Qdrant server for Postgres deployments."
            )
        from .sqlite_vec_index import SqliteVecIndex

        return SqliteVecIndex(metadata_store=metadata_store)
    if backend == "qdrant":
        from qdrant_client import AsyncQdrantClient

        from .qdrant_vector_index import QdrantVectorIndex

        if qdrant_url:
            client = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            client = AsyncQdrantClient(location=qdrant_location or ":memory:")
        return QdrantVectorIndex(
            client=client,
            metadata_store=metadata_store,
            collection_name=qdrant_collection,
            routing_collection_name=qdrant_routing_collection,
            routing_oversampling=qdrant_routing_oversampling,
        )
    if backend != "memory":
        raise ValueError(
            f"Unknown vector_backend {backend!r} "
            "(expected 'memory', 'qdrant', or 'sqlite_vec')."
        )
    return InMemoryVectorIndex(
        fact_store=fact_store,
        vector_store=vector_store,
        metadata_store=metadata_store,
    )
