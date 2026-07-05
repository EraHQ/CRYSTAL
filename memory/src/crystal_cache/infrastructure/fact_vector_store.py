"""Fact-level vector store — Phase 1 of V3 Cognitive Routing.

Adds fact-level search alongside the existing crystal-level search.
Each fact's `vector` field (encoded prompt_text) becomes searchable
independently, filtered by pair_type.

This is the foundation that all V3 routers build on. Instead of
matching queries to crystals (which average across many facts),
routers match queries to individual facts (precise, type-filtered).

Usage:
    fact_store = FactVectorStore(store=metadata_store)

    # Content router: search only content chunks
    results = await fact_store.search(
        customer_id="cus_xxx",
        query_vector=encoded_query,
        pair_types=["content_chunk"],
        k=5,
    )

    # Knowledge router: search entity/qa facts
    results = await fact_store.search(
        customer_id="cus_xxx",
        query_vector=encoded_query,
        pair_types=["entity_attribute", "question_answer"],
        k=10,
    )
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
import structlog

from .permissions import can_read

if TYPE_CHECKING:
    from ..infrastructure.metadata_store import MetadataStore
    from ..models import Operator

# structlog, not stdlib logging: this module's keyword-style log calls
# (customer_id=..., crystal_type=...) are structlog's calling
# convention. Under stdlib logging they only "worked" because INFO is
# suppressed by default — the first enabled-level call with kwargs
# raised TypeError (found by the general-crystals harness, 2026-06-12).
logger = structlog.get_logger(__name__)

# Accommodation tie-break: general facts compete on merit but the
# customer's own knowledge wins ties. A multiplicative nudge rather
# than hard precedence — a strong general fact can still beat a weak
# customer fact, but at equal relevance the tenant's bank speaks first.
GENERAL_TIE_BREAK = 0.995


@dataclass
class _FactEntry:
    """One fact's vector data for search."""
    fact_id: str
    crystal_id: str
    pair_type: str
    prompt_text: str


@dataclass
class _FactBank:
    """Cached per-customer fact vectors.

    All facts for a customer are loaded once, then filtered
    by pair_type at search time. This avoids loading separate
    indexes per pair_type while keeping search efficient.
    """
    entries: list[_FactEntry] = field(default_factory=list)
    matrix: Optional[np.ndarray] = None  # (n_facts, d) normalized


class FactVectorStore:
    """Per-customer in-memory fact vector cache.

    Loads all facts for a customer on first access, caches the
    matrix, and filters by pair_type at search time.
    """

    def __init__(self, store: "MetadataStore") -> None:
        self._store = store
        self._banks: dict[str, _FactBank] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        # General banks: keyed by crystal_type, loaded ONCE and shared
        # across every customer's searches (general knowledge is
        # system-level; per-customer copies were v1's "992 identical
        # pairs per customer" waste this design exists to kill).
        self._general_banks: dict[str, _FactBank] = {}
        self._general_locks: dict[str, asyncio.Lock] = {}
        # Subscription cache: customer_id -> general types. Invalidated
        # with the customer's bank so a subscription change takes
        # effect on the next search after invalidate().
        self._subs: dict[str, list[str]] = {}

    def _lock_for(self, customer_id: str) -> asyncio.Lock:
        lock = self._locks.get(customer_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[customer_id] = lock
        return lock

    async def _ensure_loaded(self, customer_id: str) -> _FactBank:
        """Load all facts for a customer into the cache."""
        bank = self._banks.get(customer_id)
        if bank is not None and bank.matrix is not None:
            return bank

        async with self._lock_for(customer_id):
            # Re-check under lock
            bank = self._banks.get(customer_id)
            if bank is not None and bank.matrix is not None:
                return bank

            # Load all facts for this customer's crystals
            facts = await self._store.list_all_facts_for_customer(customer_id)
            bank = self._build_bank(facts)
            self._banks[customer_id] = bank

            logger.info(
                "fact_vector_store.loaded",
                customer_id=customer_id,
                total_facts=len(bank.entries),
                pair_types=list(set(e.pair_type for e in bank.entries)),
            )
            return bank

    @staticmethod
    def _build_bank(facts: list) -> _FactBank:
        """Facts -> normalized matrix bank. Shared by customer + general loaders."""
        bank = _FactBank()
        usable = [f for f in facts if f.vector and len(f.vector) > 0]
        if not usable:
            bank.matrix = np.empty((0, 0), dtype=np.float32)
            return bank
        d = len(usable[0].vector)
        entries: list[_FactEntry] = []
        rows: list[np.ndarray] = []
        for f in usable:
            if len(f.vector) != d:
                continue  # dimension mismatch, skip
            entries.append(_FactEntry(
                fact_id=f.id,
                crystal_id=f.crystal_id,
                pair_type=f.pair_type,
                prompt_text=f.prompt_text or "",
            ))
            row = np.asarray(f.vector, dtype=np.float32)
            norm = float(np.linalg.norm(row))
            if norm > 0.0:
                row = row / norm
            rows.append(row)
        bank.entries = entries
        bank.matrix = np.vstack(rows) if rows else np.empty((0, d), dtype=np.float32)
        return bank

    def _general_lock_for(self, crystal_type: str) -> asyncio.Lock:
        lock = self._general_locks.get(crystal_type)
        if lock is None:
            lock = asyncio.Lock()
            self._general_locks[crystal_type] = lock
        return lock

    async def _ensure_general_loaded(self, crystal_type: str) -> _FactBank:
        """Load one general bank (customer_id NULL, this type) once."""
        bank = self._general_banks.get(crystal_type)
        if bank is not None and bank.matrix is not None:
            return bank
        async with self._general_lock_for(crystal_type):
            bank = self._general_banks.get(crystal_type)
            if bank is not None and bank.matrix is not None:
                return bank
            facts = await self._store.list_all_facts_general(crystal_type)
            bank = self._build_bank(facts)
            self._general_banks[crystal_type] = bank
            logger.info(
                "fact_vector_store.general_loaded",
                crystal_type=crystal_type, total_facts=len(bank.entries),
            )
            return bank

    async def _subscribed_types(self, customer_id: str) -> list[str]:
        subs = self._subs.get(customer_id)
        if subs is None:
            try:
                subs = await self._store.get_customer_general_types(customer_id)
            except Exception:  # noqa: BLE001 — subscription lookup must
                subs = []      # never break customer search; fail closed.
            self._subs[customer_id] = subs
        return subs

    @staticmethod
    def _score_bank(
        bank: _FactBank,
        q: np.ndarray,
        pair_type_set: Optional[set[str]],
        weight: float,
        out: list[tuple],
        with_keys: bool = False,
    ) -> None:
        """Append (fact_id, crystal_id, pair_type, weighted score) rows.

        When with_keys is True each row carries a 5th element — the
        fact's prompt_text (sparse key) — for callers that rerank after
        search (the hybrid-rank ContentRouter path). The sort, top-k,
        and operator-filter logic index [3]/[1] only, so 5-tuples flow
        through unchanged.
        """
        scores = bank.matrix @ q
        for i, entry in enumerate(bank.entries):
            if pair_type_set and entry.pair_type not in pair_type_set:
                continue
            score = float(scores[i]) * weight
            if with_keys:
                out.append((
                    entry.fact_id,
                    entry.crystal_id,
                    entry.pair_type,
                    score,
                    entry.prompt_text,
                ))
            else:
                out.append((
                    entry.fact_id,
                    entry.crystal_id,
                    entry.pair_type,
                    score,
                ))

    def invalidate(self, customer_id: str) -> None:
        """Drop cache for a customer. Call after writing new facts."""
        keys_to_drop = [k for k in self._banks if k == customer_id]
        for k in keys_to_drop:
            del self._banks[k]
            self._locks.pop(k, None)
        self._subs.pop(customer_id, None)

    def invalidate_general(self, crystal_type: Optional[str] = None) -> None:
        """Drop general bank cache(s). Call after the seed importer writes."""
        if crystal_type is None:
            self._general_banks.clear()
            self._general_locks.clear()
        else:
            self._general_banks.pop(crystal_type, None)
            self._general_locks.pop(crystal_type, None)

    def invalidate_all(self) -> None:
        """Drop all caches."""
        self._banks.clear()
        self._locks.clear()
        self._general_banks.clear()
        self._general_locks.clear()
        self._subs.clear()

    async def search(
        self,
        customer_id: str,
        query_vector: np.ndarray,
        *,
        pair_types: Optional[list[str]] = None,
        k: int = 10,
        operator: Optional["Operator"] = None,
        with_keys: bool = False,
    ) -> list[tuple]:
        """Search facts by cosine similarity, optionally filtered by pair_type.

        Args:
            customer_id: which customer's facts to search
            query_vector: encoded query (will be L2-normalized)
            pair_types: if provided, only search facts with these pair_types
            k: number of results to return
            operator: if provided (Foundation F2), the results are filtered
                to crystals this operator may read (POSIX mode bits + group
                + owner + named ACL grants). None (the default) skips the
                filter entirely — today's tenancy/subscription behavior,
                with no extra queries.

        Returns:
            List of (fact_id, crystal_id, pair_type, cosine_score)
            sorted by score descending. When with_keys=True, each tuple
            carries a 5th element: the fact's prompt_text (sparse key),
            for callers that rerank after search (hybrid-rank).

        General-crystals merge (2026-06-12): results come from the
        customer's bank PLUS every general bank the customer subscribes
        to (customers.general_crystal_types), ranked together. General
        scores carry GENERAL_TIE_BREAK so the tenant's own knowledge
        wins ties (accommodation thesis) without hard precedence. The
        merge lives HERE — the single point below every consumer
        (agent retrieval tools, v3 routers, proxy injection) — so all
        surfaces inherit general knowledge with zero call-site changes.
        A general bank whose vector dimension doesn't match the query
        is skipped with a warning, never an error: a half-built seed
        bank must not break customer search.
        """
        bank = await self._ensure_loaded(customer_id)

        # Normalize query
        q = np.asarray(query_vector, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm > 0.0:
            q = q / q_norm

        results: list[tuple] = []
        pair_type_set = set(pair_types) if pair_types else None

        if bank.matrix is not None and bank.matrix.size > 0:
            if q.shape[0] != bank.matrix.shape[1]:
                raise ValueError(
                    f"Query vector dim {q.shape[0]} does not match "
                    f"fact bank dim {bank.matrix.shape[1]} for customer {customer_id}."
                )
            self._score_bank(bank, q, pair_type_set, 1.0, results, with_keys=with_keys)

        for crystal_type in await self._subscribed_types(customer_id):
            gbank = await self._ensure_general_loaded(crystal_type)
            if gbank.matrix is None or gbank.matrix.size == 0:
                continue
            if q.shape[0] != gbank.matrix.shape[1]:
                logger.warning(
                    "fact_vector_store.general_dim_mismatch",
                    crystal_type=crystal_type,
                    bank_dim=int(gbank.matrix.shape[1]), query_dim=int(q.shape[0]),
                )
                continue  # fail-soft: skip this general bank
            self._score_bank(
                gbank, q, pair_type_set, GENERAL_TIE_BREAK, results,
                with_keys=with_keys,
            )

        # Sort by score descending
        results.sort(key=lambda x: x[3], reverse=True)

        # Foundation F2: permission filter. With no operator context this is
        # today's behavior exactly — return the top-k unchanged, no extra
        # queries. With an operator, keep only crystals the operator may
        # read, walking in score order and fetching crystal + ACLs lazily
        # (cached per crystal_id within this call) until k candidates pass.
        # The candidate set is already tenancy-scoped, so the filter only
        # ever removes a teammate's owner-private crystals; general /
        # subscribed facts are world-shared and pass (see permissions.
        # can_read). The lazy fetch means low-ranked candidates we'd never
        # return cost nothing.
        if operator is None:
            return results[:k]

        # P3: group memberships fetched once so 'group' grants resolve
        # (can_read fail-closes group grants without the set).
        group_ids = await self._store.list_group_ids_for_operator(operator.id)
        allowed: list[tuple] = []
        verdicts: dict[str, bool] = {}
        for row in results:
            crystal_id = row[1]
            verdict = verdicts.get(crystal_id)
            if verdict is None:
                crystal = await self._store.get_crystal(crystal_id)
                if crystal is None:
                    verdict = False
                else:
                    acls = await self._store.list_acls_for_crystal(crystal_id)
                    verdict = can_read(crystal, operator, acls, group_ids)
                verdicts[crystal_id] = verdict
            if verdict:
                allowed.append(row)
                if len(allowed) >= k:
                    break
        return allowed[:k]

    async def get_pair_type_stats(self, customer_id: str) -> dict[str, int]:
        """Get count of facts per pair_type for a customer.

        Useful for the Index of Indexes and router pre-filtering.
        """
        bank = await self._ensure_loaded(customer_id)
        stats: dict[str, int] = {}
        for entry in bank.entries:
            stats[entry.pair_type] = stats.get(entry.pair_type, 0) + 1
        return stats
