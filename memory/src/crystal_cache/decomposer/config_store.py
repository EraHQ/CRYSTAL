"""DslConfigStore — in-memory store for compiled DSL configs, per tenant.

A tenant's compiled DSL configs live here. The concept-path router queries
this store to find the best-matching config for a decomposed query.

v0.4 SCOPE — LAZY DB-BACKED
--------------------------
Source text is persisted in the `dsl_configs` SQLAlchemy table. On first
access for a tenant, the store queries all rows for that tenant,
concatenates the source texts (ordered by name for determinism), compiles
them as a single DSL program, and caches the resulting RuntimeEnv.

`invalidate(tenant_id)` drops the cache so the next access re-queries.
Call after persisting a new/updated config via the admin endpoint.

The v0.2 interface — `register_source(tenant_id, source)` for programmatic,
in-memory-only use — remains available for tests and offline tooling.

Why in-memory + DB rather than pure DB?
  - Source text is the authoritative representation; compiled vectors
    are a cache we can always rebuild.
  - Re-compile is cheap (milliseconds) so cold-start penalties are low.
  - A per-process numpy-backed cache is far faster than querying + recompiling
    on every request.

INTERFACE DESIGN
----------------
Tenant-scoped. Each tenant gets its own RuntimeEnv under the hood, which
means concept vectors for the same name in different tenants are
orthogonal — matches the DSL's tenant-isolation guarantee.

THREAD SAFETY
-------------
Not thread-safe for concurrent WRITES to the same tenant. A per-tenant
asyncio.Lock serializes first-load so two concurrent first-access requests
don't both trigger a compile. Reads after load are safe.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Optional

import numpy as np

from crystal_cache.dsl import RuntimeEnv, run

if TYPE_CHECKING:
    from crystal_cache.infrastructure.metadata_store import MetadataStore


class DslConfigStoreError(Exception):
    pass


class DslConfigStore:
    """Per-tenant store of compiled DSL configs.

    Two use patterns:

      1. In-memory (tests, offline tools): call `register_source()` to
         compile and stash a tenant's source, call `rank()` or `get_env()`.
         No database involvement.

      2. DB-backed (production): construct with a `metadata_store=`
         argument. On first access for a tenant, the store queries all
         dsl_configs rows for that tenant, concatenates them, compiles,
         and caches. `invalidate(tenant_id)` forces a re-load on next
         access.

    The two patterns coexist: `register_source` always shadows the DB
    for the tenants it touches. That's what tests want.
    """

    def __init__(self, metadata_store: Optional["MetadataStore"] = None) -> None:
        # tenant_id -> RuntimeEnv
        self._envs: dict[str, RuntimeEnv] = {}
        # tenant_id -> source (for debug + fallback)
        self._sources: dict[str, str] = {}
        # Tenants we've asked the DB about. Present whether or not the
        # tenant had any configs — avoids re-querying empty tenants
        # once per request.
        self._db_loaded: set[str] = set()
        # Per-tenant lock to serialize first-load from DB.
        self._locks: dict[str, asyncio.Lock] = {}
        self._metadata_store: Optional["MetadataStore"] = metadata_store

    def _lock_for(self, tenant_id: str) -> asyncio.Lock:
        lock = self._locks.get(tenant_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[tenant_id] = lock
        return lock

    # -----------------------------------------------------------------
    # Synchronous registration (tests, offline tooling)
    # -----------------------------------------------------------------

    def register_source(self, tenant_id: str, source: str) -> RuntimeEnv:
        """Compile DSL source for a tenant. Replaces any existing env.

        Returns the compiled env for callers that want to inspect it.
        This bypasses the DB — use `upsert_source_and_reload` for the
        admin path that both persists and recompiles.
        """
        if not tenant_id:
            raise DslConfigStoreError("tenant_id must be non-empty")
        env = run(source, tenant_id=tenant_id)
        self._envs[tenant_id] = env
        self._sources[tenant_id] = source
        # Mark as "loaded" so DB lookup doesn't race ahead and overwrite.
        self._db_loaded.add(tenant_id)
        return env

    # -----------------------------------------------------------------
    # Async DB-backed load
    # -----------------------------------------------------------------

    async def ensure_loaded(self, tenant_id: str) -> None:
        """Load this tenant's configs from the metadata store if not cached.

        Idempotent: repeated calls are free. If no metadata_store was
        configured, this is a no-op (register_source is the only path).
        """
        if tenant_id in self._db_loaded:
            return
        if self._metadata_store is None:
            # No DB configured — nothing to load.
            self._db_loaded.add(tenant_id)
            return

        async with self._lock_for(tenant_id):
            if tenant_id in self._db_loaded:
                return
            rows = await self._metadata_store.list_dsl_configs_for_customer(
                tenant_id
            )
            if rows:
                # Concatenate the named sources in name order. Separator
                # ensures identifiers don't accidentally merge across
                # source boundaries.
                combined = "\n\n".join(src for _name, src in rows)
                env = run(combined, tenant_id=tenant_id)
                self._envs[tenant_id] = env
                self._sources[tenant_id] = combined
            # Mark tenant as checked regardless of whether rows existed.
            # Avoids re-querying a tenant with no configs on every request.
            self._db_loaded.add(tenant_id)

    def invalidate(self, tenant_id: str) -> None:
        """Drop this tenant's compiled env. Next access re-loads from DB.

        Call after upserting/deleting a dsl_configs row via the admin API.
        """
        self._envs.pop(tenant_id, None)
        self._sources.pop(tenant_id, None)
        self._db_loaded.discard(tenant_id)

    async def upsert_source_and_reload(
        self, tenant_id: str, name: str, source_text: str
    ) -> RuntimeEnv:
        """Admin helper: persist to DB, invalidate cache, force reload.

        Returns the freshly-compiled env. Raises DslConfigStoreError if
        no metadata_store was configured (the store is operating in
        in-memory-only mode, so there's nothing to persist to).
        """
        if self._metadata_store is None:
            raise DslConfigStoreError(
                "upsert_source_and_reload requires a metadata_store "
                "(pass metadata_store=... to DslConfigStore)"
            )
        await self._metadata_store.upsert_dsl_config(
            customer_id=tenant_id, name=name, source_text=source_text
        )
        self.invalidate(tenant_id)
        await self.ensure_loaded(tenant_id)
        env = self._envs.get(tenant_id)
        if env is None:
            # Source compiled to an empty program — unusual but not an error.
            # Return a freshly-registered empty env so callers have something.
            env = run("", tenant_id=tenant_id)
            self._envs[tenant_id] = env
        return env

    # -----------------------------------------------------------------
    # Read-side API
    # -----------------------------------------------------------------

    def has_tenant(self, tenant_id: str) -> bool:
        """True if the tenant has a compiled env in cache.

        This is SYNCHRONOUS — callers that need to force a DB load first
        should call `await ensure_loaded(tenant_id)` then `has_tenant`.
        The ConceptRouter does exactly this.
        """
        return tenant_id in self._envs

    def get_env(self, tenant_id: str) -> Optional[RuntimeEnv]:
        """Return the compiled env for a tenant, or None if not cached."""
        return self._envs.get(tenant_id)

    def rank(
        self,
        tenant_id: str,
        query_hv: np.ndarray,
    ) -> list[tuple[str, float]]:
        """Rank this tenant's configs against a query vector.

        Returns empty list if the tenant has no configs or isn't cached.
        No error — retrieval simply doesn't contribute, same pattern as
        CrystalRouter.
        """
        env = self._envs.get(tenant_id)
        if env is None:
            return []
        return env.rank_configs(query_hv)

    def list_configs(self, tenant_id: str) -> list[str]:
        """Return config names registered for this tenant, empty if unknown."""
        env = self._envs.get(tenant_id)
        if env is None:
            return []
        return sorted(env.configs.keys())

    def source_of(self, tenant_id: str) -> Optional[str]:
        """Return the concatenated DSL source this tenant was compiled from."""
        return self._sources.get(tenant_id)

    def clear(self) -> None:
        """Drop all tenants. Mostly for tests."""
        self._envs.clear()
        self._sources.clear()
        self._db_loaded.clear()
        self._locks.clear()
