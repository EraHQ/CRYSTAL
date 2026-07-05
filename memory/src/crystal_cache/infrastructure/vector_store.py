"""Vector store — in-memory cache for crystal routing vectors.

This is the retrieval-hot-path backend. It answers one question:
"given a query vector and a customer, which crystals rank highest
by cosine similarity?"

DESIGN NOTES:
  - Crystals live in the main DB (metadata_store). Phase 6.3
    (May 2026): the routing primitive operates on `routing_vector`
    (`Σ encode(prompt_i) @ P`), NOT on `summary_vector`
    (`Σ bind(P_i, A_i)`). See models/crystal.py for the geometric
    argument. Recall (recall_from_crystal) still operates on
    summary_vector — the two vectors serve different jobs.
  - For v0 we load every crystal's routing_vector into a per-customer
    matrix on first access and cache it. Refreshed on explicit
    invalidation (e.g. after a bank bootstrap).
  - "Cache" is process-local. That's fine for a single replica. Multi-
    replica deployments will need to either share the cache (Redis) or
    accept per-replica warm-up.
  - For banks up to ~10k crystals per customer and 10k-dim vectors,
    the matrix is ~400MB per customer. That's the practical upper
    bound for this implementation.
  - When the bank gets larger or customer count gets higher, swap for
    pgvector. The VectorStore interface is small enough that the swap
    is mechanical.

NOT IN SCOPE HERE:
  - Fact vectors (per-fact retrieval). That's Group E+.
  - Approximate nearest neighbor. Exact search over per-customer
    banks is fine at this scale.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from .permissions import can_read

if TYPE_CHECKING:
    from ..infrastructure.metadata_store import MetadataStore
    from ..models import Operator


@dataclass
class _CustomerBank:
    """Cached per-(customer, crystal_type) crystal vectors.

    Phase 3 (April 2026): the cache key is (customer_id, crystal_type)
    rather than just customer_id. Each (customer, type) pair gets its
    own pre-loaded matrix; `VectorStore.search` filters routing to
    crystals of one type. The crystal_type=None special case still
    exists for legacy callers that want all-types search and is keyed
    as (customer_id, None) internally.
    """
    crystal_ids: list[str] = field(default_factory=list)
    # Matrix of shape (n_crystals, d_hdc). Phase 1.3 (April 2026):
    # rows are L2-normalized at load time in `_ensure_loaded` so the
    # cosine-as-dot-product math in `search` is correct regardless of
    # how the source summary_vector was persisted. Bind-storage
    # crystals (whose summary_vectors are raw bundles, not unit-norm)
    # are routed correctly through this matrix; legacy unit-norm
    # crystals normalize to themselves and are unaffected.
    matrix: Optional[np.ndarray] = None


class VectorStore:
    """Per-customer in-memory crystal vector cache.

    Loads lazily from the MetadataStore. Not thread-safe for concurrent
    loads of the SAME customer (simultaneous first-access would both trigger
    a load); used from a single FastAPI worker loop this is fine.

    Phase 3 (April 2026): cache is keyed by (customer_id, crystal_type)
    so type-scoped routing has its own pre-loaded matrix per type.
    crystal_type=None is the legacy/unfiltered path — kept for
    back-compat with callers that want all-types search.
    """

    def __init__(self, store: "MetadataStore") -> None:
        self._store = store
        self._banks: dict[tuple[str, Optional[str]], _CustomerBank] = {}
        self._locks: dict[tuple[str, Optional[str]], asyncio.Lock] = {}
        # General crystal caches — loaded once, shared across customers.
        # Key: crystal_type (e.g., "general:legacy")
        self._general_banks: dict[str, _CustomerBank] = {}
        self._general_locks: dict[str, asyncio.Lock] = {}

    # -----------------------------------------------------------------
    # Cache management
    # -----------------------------------------------------------------

    def _lock_for(
        self, customer_id: str, crystal_type: Optional[str]
    ) -> asyncio.Lock:
        key = (customer_id, crystal_type)
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    async def _ensure_loaded(
        self,
        customer_id: str,
        crystal_type: Optional[str] = None,
    ) -> _CustomerBank:
        """Load this (customer, type)'s crystals from the DB if not cached yet.

        crystal_type=None loads the customer's full bank across all
        types. crystal_type=str loads the type-scoped subset via
        `list_crystals_for_customer_and_type` (Phase 3, indexed via
        ix_crystals_customer_type).
        """
        key = (customer_id, crystal_type)
        bank = self._banks.get(key)
        if bank is not None and bank.matrix is not None:
            return bank

        async with self._lock_for(customer_id, crystal_type):
            # Re-check under lock
            bank = self._banks.get(key)
            if bank is not None and bank.matrix is not None:
                return bank

            if crystal_type is None:
                crystals = await self._store.list_crystals_for_customer(
                    customer_id, include_recall_gated=False,
                )
            else:
                crystals = await self._store.list_crystals_for_customer_and_type(
                    customer_id, crystal_type, include_recall_gated=False,
                )
            bank = _CustomerBank()
            if not crystals:
                # Cache the empty result so we don't re-query on every miss
                bank.matrix = np.empty((0, 0), dtype=np.float32)
                self._banks[key] = bank
                return bank

            # Filter out crystals whose ROUTING vector is missing or empty.
            # Phase 6.3 (May 2026): VectorStore.search routes on
            # crystal.routing_vector, not crystal.summary_vector.
            # See Finding 16 — the bind-bundle stored in summary_vector
            # is geometrically near-orthogonal to its component
            # prompt-projections, so cosine routing against it returns
            # near-zero scores even for matching queries.
            # routing_vector = `Σ encode(prompt_i) @ P` IS
            # cosine-compatible with `encode(query) @ P`.
            #
            # Pre-Phase-6.3 crystals have routing_vector=None and are
            # invisible to routing until backfilled via
            # scripts/backfill_routing_vectors.py. That's the
            # Phase 6.3 clean-break: no silent fallback to the
            # broken-at-scale summary_vector path. If a customer's
            # bank routes nothing, the diagnostic is "backfill
            # routing_vectors" rather than "routing returns garbage."
            usable = [c for c in crystals if c.routing_vector]
            if not usable:
                bank.matrix = np.empty((0, 0), dtype=np.float32)
                self._banks[key] = bank
                return bank

            d_hdc = len(usable[0].routing_vector)
            ids: list[str] = []
            rows: list[np.ndarray] = []
            for c in usable:
                if len(c.routing_vector) != d_hdc:
                    # Dimension mismatch within a single bank is a bug;
                    # skip rather than crash and move on.
                    continue
                ids.append(c.id)
                # Phase 6.3 (May 2026): L2-normalize routing_vector on
                # load. Same contract as the prior summary_vector
                # path — raw bundles persisted by add_pair_to_crystal,
                # unit-norm enforced at the read boundary so cosine-
                # as-dot-product is valid in `search`.
                #
                # Edge case: zero-norm rows (degenerate, shouldn't
                # happen in practice but possible if an accumulator
                # cancelled exactly to zero). We leave those as
                # zeros rather than dividing by zero; the row will
                # simply never win a cosine comparison against a
                # non-zero query, which is correct.
                row = np.asarray(c.routing_vector, dtype=np.float32)
                norm = float(np.linalg.norm(row))
                if norm > 0.0:
                    row = row / norm
                rows.append(row.astype(np.float32))

            bank.crystal_ids = ids
            bank.matrix = np.vstack(rows) if rows else np.empty((0, d_hdc), dtype=np.float32)
            self._banks[key] = bank
            return bank

    def invalidate(self, customer_id: str) -> None:
        """Drop ALL cache entries for one customer. Phase 3: walks every
        cache key and removes those matching this customer, since a
        write may have changed multiple type-scoped banks at once
        (and the safe move is to drop them all rather than guess which).

        Call after a bank bootstrap or after add_pair_for_customer
        writes. For v0 nothing mutates crystals during a request, so
        invalidation is a manual operator action plus the
        post-write call inside add_pair_for_customer.
        """
        # Walk and collect keys to drop, then drop them. Modifying
        # the dict during iteration would error.
        to_drop = [
            key for key in self._banks
            if key[0] == customer_id
        ]
        for key in to_drop:
            self._banks.pop(key, None)
        # Locks are kept around — they're cheap and a re-load would
        # need them again. Their absence wouldn't cause incorrect
        # behavior, just a tiny re-allocation.

    def invalidate_all(self) -> None:
        self._banks.clear()
        self._general_banks.clear()

    async def _ensure_general_loaded(
        self, crystal_type: str
    ) -> _CustomerBank:
        """Load general crystals (customer_id IS NULL) for this type.

        Unlike customer banks, general banks are loaded ONCE and shared
        across all customer searches. Only invalidated when an admin
        imports new general crystals.
        """
        bank = self._general_banks.get(crystal_type)
        if bank is not None and bank.matrix is not None:
            return bank

        lock = self._general_locks.get(crystal_type)
        if lock is None:
            lock = asyncio.Lock()
            self._general_locks[crystal_type] = lock

        async with lock:
            bank = self._general_banks.get(crystal_type)
            if bank is not None and bank.matrix is not None:
                return bank

            crystals = await self._store.list_general_crystals(crystal_type)
            bank = _CustomerBank()
            usable = [c for c in crystals if c.routing_vector]
            if not usable:
                bank.matrix = np.empty((0, 0), dtype=np.float32)
                self._general_banks[crystal_type] = bank
                return bank

            d_hdc = len(usable[0].routing_vector)
            ids: list[str] = []
            rows: list[np.ndarray] = []
            for c in usable:
                if len(c.routing_vector) != d_hdc:
                    continue
                ids.append(c.id)
                row = np.asarray(c.routing_vector, dtype=np.float32)
                norm = float(np.linalg.norm(row))
                if norm > 0.0:
                    row = row / norm
                rows.append(row)

            bank.crystal_ids = ids
            bank.matrix = np.vstack(rows) if rows else np.empty((0, d_hdc), dtype=np.float32)
            self._general_banks[crystal_type] = bank
            return bank

    def invalidate_general(self, crystal_type: Optional[str] = None) -> None:
        """Invalidate general bank cache(s) after admin import. None clears ALL
        general banks (mirrors FactVectorStore.invalidate_general and the
        VectorIndex seam contract); a concrete type clears just that one."""
        if crystal_type is None:
            self._general_banks.clear()
            self._general_locks.clear()
        else:
            self._general_banks.pop(crystal_type, None)
            self._general_locks.pop(crystal_type, None)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    async def search(
        self,
        customer_id: str,
        query_vector: np.ndarray,
        k: int = 5,
        *,
        crystal_type: str,
        general_crystal_types: Optional[list[str]] = None,
        operator: Optional["Operator"] = None,
    ) -> list[tuple[str, float]]:
        """Return the top-K (crystal_id, cosine_similarity).

        Searches the customer's bank AND any subscribed general banks.
        Results are merged by cosine score. Customer crystals win ties
        over general crystals (customer may have specialized knowledge).

        Foundation F2: when `operator` is provided, results are filtered
        through `infrastructure.permissions.can_read` so an operator only
        sees crystals it's permitted to read (its own private crystals,
        team-readable ones, ACL-granted ones, and — as an admin — anything
        grouped to its team). General crystals (customer_id None) are
        world-readable and always pass. `operator=None` preserves today's
        unfiltered behavior exactly (no extra queries).
        """
        # 1. Search customer's own bank
        bank = await self._ensure_loaded(customer_id, crystal_type)
        all_hits: dict[str, float] = {}

        if bank.matrix is not None and bank.matrix.size > 0:
            if query_vector.shape[0] != bank.matrix.shape[1]:
                raise ValueError(
                    f"Query vector dim {query_vector.shape[0]} does not match "
                    f"bank dim {bank.matrix.shape[1]} for customer {customer_id}."
                )
            scores = bank.matrix @ query_vector.astype(np.float32)
            for i, cid in enumerate(bank.crystal_ids):
                all_hits[cid] = float(scores[i])

        # 2. Search each subscribed general bank
        if general_crystal_types:
            for gen_type in general_crystal_types:
                gen_bank = await self._ensure_general_loaded(gen_type)
                if gen_bank.matrix is None or gen_bank.matrix.size == 0:
                    continue
                if query_vector.shape[0] != gen_bank.matrix.shape[1]:
                    continue  # dim mismatch, skip silently
                gen_scores = gen_bank.matrix @ query_vector.astype(np.float32)
                for i, cid in enumerate(gen_bank.crystal_ids):
                    # Customer-specific version wins over general
                    if cid not in all_hits:
                        all_hits[cid] = float(gen_scores[i])

        # 3. Sort by score descending.
        if not all_hits:
            return []
        sorted_hits = sorted(all_hits.items(), key=lambda x: x[1], reverse=True)

        # 4. Permission filter (Foundation F2). operator=None → behavior
        # preserved exactly: take the top-K as before, no crystal/ACL
        # fetches. With an operator, walk in score order keeping only
        # crystals can_read permits, fetching the crystal + its ACLs lazily
        # (per-id verdict cache), and stop once k have passed.
        if operator is None:
            return sorted_hits[:k]

        # P3: group memberships fetched once so 'group' grants resolve
        # (can_read fail-closes group grants without the set).
        group_ids = await self._store.list_group_ids_for_operator(operator.id)
        permitted: list[tuple[str, float]] = []
        verdict_cache: dict[str, bool] = {}
        for cid, score in sorted_hits:
            allowed = verdict_cache.get(cid)
            if allowed is None:
                crystal = await self._store.get_crystal(cid)
                if crystal is None:
                    allowed = False
                else:
                    acls = await self._store.list_acls_for_crystal(cid)
                    allowed = can_read(
                        crystal, operator, acls=acls,
                        operator_group_ids=group_ids,
                    )
                verdict_cache[cid] = allowed
            if allowed:
                permitted.append((cid, score))
                if len(permitted) >= k:
                    break
        return permitted
