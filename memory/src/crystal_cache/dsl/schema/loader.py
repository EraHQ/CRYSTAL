"""Schema DSL loader (Phase 4.4).

Bridges \"DSL source persisted in `crystal_types.pair_schema_dsl`\" to
\"compiled object available at runtime.\" Caches per-type compiled
objects so request-time hooks (write validation, route boost, ACL
defaults) don't pay parse + validate + compile on every call.

CACHING MODEL
-------------
Keyed by `crystal_type_id` (e.g. 'general:python_debugging'), NOT by
customer. Schema definitions are global \u2014 every customer sees the
same compiled object for a given type. A customer-specific *crystal*
of type `customer:medical_records` reuses the type-level compiled
object; the customer-specific bit is the crystal row's owning
customer_id, not anything the schema DSL knows about.

Compare to `crystal_cache.decomposer.config_store.DslConfigStore`,
which is per-tenant because the concept DSL is per-tenant. Different
DSL, different cache key.

INVALIDATION
------------
Call `invalidate(crystal_type_id)` after upserting a `crystal_types`
row via the admin endpoint (Phase 4.8). The next access re-fetches,
re-compiles, and re-caches. `invalidate_all()` drops everything \u2014
useful for tests and for blunt cache busting.

EMPTY pair_schema_dsl
---------------------
Migration 0012 seeds 'general:legacy' and 'customer:legacy' with
empty `pair_schema_dsl`. The loader treats empty source as a
permissive default: one open `pair_type \"question_answer\"`
accepting `text` for both fields, no route hint, ACL defaults from
the row's scope. This preserves back-compat with every existing
FAQ-bank write and any other call that uses the default pair_type.

THREAD SAFETY
-------------
Per-type asyncio.Lock serializes first-load. Reads after load are
safe (the cache map is read-mostly). Concurrent writes to the same
type_id (i.e. simultaneous admin upserts) are NOT serialized at this
layer \u2014 the DB serializes them, and `invalidate` is fast enough that
a brief inconsistency window is acceptable.

ERROR SURFACE
-------------
- get(unknown_type_id)  -> returns None (not registered in the DB).
- get(empty_dsl)        -> returns the permissive default object.
- get(invalid_dsl)      -> raises SchemaLoadError carrying the
                           diagnostics from validation. The DB row's
                           pair_schema_dsl was corrupted somehow
                           (admin endpoint should have rejected the
                           upsert; if you see this in production, the
                           validator and admin layer have drifted).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from crystal_cache.dsl.schema.compiler import (
    CompiledAclGrant,
    CompiledCrystalType,
    CompiledField,
    CompiledPairType,
    SchemaCompileError,
    compile_program,
)
from crystal_cache.dsl.schema.parser import SchemaParseError, parse
from crystal_cache.dsl.schema.validator import (
    Diagnostic,
    SchemaValidationError,
    validate,
)

if TYPE_CHECKING:
    from crystal_cache.infrastructure.metadata_store import MetadataStore
    from crystal_cache.models import CrystalAcl, CrystalType


class SchemaLoadError(Exception):
    """Raised when a `crystal_types.pair_schema_dsl` row fails to parse,
    validate, or compile.

    Carries the type_id and (where available) diagnostics. Callers that
    want to surface the underlying issue can inspect `.diagnostics`.
    """

    def __init__(
        self,
        type_id: str,
        message: str,
        diagnostics: Optional[list[Diagnostic]] = None,
        cause: Optional[Exception] = None,
    ):
        self.type_id = type_id
        self.diagnostics = diagnostics or []
        self.cause = cause
        full_msg = f"failed to load schema for crystal_type {type_id!r}: {message}"
        if cause is not None:
            full_msg += f"\n  caused by: {type(cause).__name__}: {cause}"
        super().__init__(full_msg)


# ---------------------------------------------------------------------------
# SchemaLoader
# ---------------------------------------------------------------------------


class SchemaLoader:
    """Per-type schema DSL cache, DB-backed.

    Two use patterns mirror DslConfigStore:

      1. In-memory (tests, offline tools): construct without
         metadata_store, call `register_source(type_id, dsl_source)`
         to compile and stash a type. No DB involvement.

      2. DB-backed (production): construct with metadata_store=...
         On first access for a type_id, the loader fetches the
         crystal_types row, compiles its pair_schema_dsl, and caches.
         `invalidate(type_id)` forces a re-load on next access.

    register_source always shadows the DB for the type_ids it
    touches \u2014 that's what tests want.
    """

    def __init__(
        self,
        metadata_store: Optional["MetadataStore"] = None,
    ) -> None:
        # type_id -> CompiledCrystalType
        self._cache: dict[str, CompiledCrystalType] = {}
        # Type ids we've asked the DB about. Present whether or not
        # the row existed \u2014 avoids re-querying missing types every
        # request.
        self._db_checked: set[str] = set()
        # Type ids known to be missing from the DB. Distinct from
        # _db_checked so we can return None for these without a
        # second cache lookup.
        self._db_missing: set[str] = set()
        # Per-type lock to serialize first-load.
        self._locks: dict[str, asyncio.Lock] = {}
        self._metadata_store: Optional["MetadataStore"] = metadata_store

    def _lock_for(self, type_id: str) -> asyncio.Lock:
        lock = self._locks.get(type_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[type_id] = lock
        return lock

    # -----------------------------------------------------------------
    # Synchronous registration (tests, offline tooling)
    # -----------------------------------------------------------------

    def register_source(
        self, type_id: str, dsl_source: str
    ) -> CompiledCrystalType:
        """Compile DSL source for a type. Replaces any existing entry.

        Returns the compiled object for caller inspection. Empty source
        is allowed and produces the permissive default object \u2014 same
        rule as DB-loaded empty rows.

        This bypasses the DB. Use `upsert_source_and_reload` for the
        admin path that both persists and recompiles.
        """
        if not type_id:
            raise SchemaLoadError(type_id, "type_id must be non-empty")
        if not dsl_source.strip():
            compiled = _empty_dsl_default(type_id)
        else:
            compiled = self._compile_or_raise_load(type_id, dsl_source)
        self._cache[type_id] = compiled
        # Mark as DB-checked so an ensure_loaded won't overwrite this
        # in-memory registration with whatever the DB has.
        self._db_checked.add(type_id)
        # The type IS present in cache; clear any prior missing flag.
        self._db_missing.discard(type_id)
        return compiled

    # -----------------------------------------------------------------
    # Async DB-backed access
    # -----------------------------------------------------------------

    async def get(self, type_id: str) -> Optional[CompiledCrystalType]:
        """Return the compiled object for a type_id, fetching from DB
        if not cached. None means \"not registered in the DB.\"

        Raises SchemaLoadError if the type IS registered but its
        pair_schema_dsl fails to parse, validate, or compile.
        """
        # Fast path: cached and present.
        cached = self._cache.get(type_id)
        if cached is not None:
            return cached
        # Fast path: known-missing.
        if type_id in self._db_missing:
            return None
        # Slow path: DB lookup, compile, cache.
        await self._ensure_loaded(type_id)
        return self._cache.get(type_id)

    async def _ensure_loaded(self, type_id: str) -> None:
        """Internal: load this type from the DB if not yet checked.

        Idempotent. Holds a per-type lock to keep two concurrent first-
        access requests from both compiling.
        """
        if type_id in self._db_checked:
            return
        if self._metadata_store is None:
            # No DB configured \u2014 mark as checked-missing so we don't
            # spin on the lock indefinitely on repeated calls.
            self._db_checked.add(type_id)
            self._db_missing.add(type_id)
            return

        async with self._lock_for(type_id):
            # Re-check inside the lock in case another waiter already
            # did the work.
            if type_id in self._db_checked:
                return
            row = await self._metadata_store.get_crystal_type(type_id)
            if row is None:
                self._db_checked.add(type_id)
                self._db_missing.add(type_id)
                return

            compiled = self._compile_row(row)
            self._cache[type_id] = compiled
            self._db_checked.add(type_id)
            # Was previously missing? Clear the flag.
            self._db_missing.discard(type_id)

    def _compile_row(self, row: "CrystalType") -> CompiledCrystalType:
        """Compile a CrystalType row's pair_schema_dsl, applying the
        empty-source default when needed.

        The compiled object is enriched with the row's metadata
        (capacity, autosplit, thresholds, display_name) when the DSL
        omits them \u2014 since the DSL author may rely on registry-row
        defaults rather than restating in DSL. This is the inverse of
        the compile path: there, the DSL is the source of truth and
        defaults come from the spec; here, the row's columns are
        authoritative for fields the DSL doesn't override.
        """
        if not row.pair_schema_dsl.strip():
            # Empty DSL \u2014 permissive default with the row's metadata
            # baked in.
            return _empty_dsl_default(
                type_id=row.id,
                display_name=row.display_name,
                scope=row.scope,
                capacity_default=row.capacity_default,
                autosplit_policy=row.autosplit_policy,
                routing_threshold=row.routing_threshold,
                cleanup_threshold=row.cleanup_threshold,
            )
        compiled = self._compile_or_raise_load(row.id, row.pair_schema_dsl)
        # Reconcile row-level metadata with compiled-from-DSL values.
        # The DSL is authoritative when it specifies a value; the row
        # is authoritative when the DSL is silent. This handles the
        # case where the DSL declares only pair_types/route/acl and
        # leaves the headers to the row.
        return _merge_with_row_metadata(compiled, row)

    def _compile_or_raise_load(
        self, type_id: str, dsl_source: str
    ) -> CompiledCrystalType:
        """Run parse + validate + compile, wrapping errors as
        SchemaLoadError so callers don't need to know about the three
        underlying error types.

        Multi-type sources are NOT supported here \u2014 a single
        `crystal_types.pair_schema_dsl` row should declare exactly
        one crystal_type, and that type's id must match the row's id.
        Multi-type sources at this layer would create ambiguity about
        which compiled object to return.
        """
        try:
            program = parse(dsl_source)
        except SchemaParseError as e:
            raise SchemaLoadError(
                type_id,
                f"DSL parse failed: {e}",
                cause=e,
            ) from e

        diagnostics = validate(program)
        errors = [d for d in diagnostics if d.level == "error"]
        if errors:
            raise SchemaLoadError(
                type_id,
                f"DSL validation failed with {len(errors)} error(s)",
                diagnostics=diagnostics,
            )

        try:
            compiled_program = compile_program(program)
        except SchemaCompileError as e:
            raise SchemaLoadError(
                type_id,
                f"DSL compilation failed: {e}",
                diagnostics=diagnostics,
                cause=e,
            ) from e

        # The DSL must declare exactly one crystal_type, and its id
        # must match the row's id.
        if len(compiled_program.crystal_types) != 1:
            raise SchemaLoadError(
                type_id,
                f"pair_schema_dsl must declare exactly one crystal_type; "
                f"this row's DSL declares "
                f"{len(compiled_program.crystal_types)}",
            )
        compiled = next(iter(compiled_program.crystal_types.values()))
        if compiled.type_id != type_id:
            raise SchemaLoadError(
                type_id,
                f"pair_schema_dsl declares type_id "
                f"{compiled.type_id!r} but the row's id is "
                f"{type_id!r}; the DSL's crystal_type id must match "
                f"the row's id",
            )
        return compiled

    def invalidate(self, type_id: str) -> None:
        """Drop a type's cached compiled object. Next access re-loads.

        Call after upserting/deleting a `crystal_types` row via the
        admin endpoint.
        """
        self._cache.pop(type_id, None)
        self._db_checked.discard(type_id)
        self._db_missing.discard(type_id)

    def invalidate_all(self) -> None:
        """Drop everything. Useful for tests; usable at runtime if a
        bulk schema reload is needed (e.g. after a migration)."""
        self._cache.clear()
        self._db_checked.clear()
        self._db_missing.clear()
        self._locks.clear()

    async def upsert_source_and_reload(
        self, type_id: str, dsl_source: str
    ) -> CompiledCrystalType:
        """Admin path: persist DSL to the row, invalidate, force reload.

        Returns the freshly-compiled object. Validates BEFORE persisting:
        a malformed DSL never lands in the DB. This is the inverse of
        the empty-source case \u2014 if the author wrote bad DSL, we want
        them to see the error and fix it before the row gets corrupted.

        Raises SchemaLoadError if no metadata_store was configured
        (the loader is in-memory-only mode), or if the DSL fails
        parse / validate / compile.
        """
        if self._metadata_store is None:
            raise SchemaLoadError(
                type_id,
                "upsert_source_and_reload requires a metadata_store",
            )

        # Validate before writing. compile_or_raise_load handles parse
        # + validate + compile; if it raises, the DB row stays clean.
        # Empty source is a special case \u2014 valid by definition (the
        # loader's empty-source default applies).
        if dsl_source.strip():
            self._compile_or_raise_load(type_id, dsl_source)

        # Persist. We do this by reading the existing row, updating the
        # pair_schema_dsl field, and upserting. If the row doesn't
        # exist yet, the upsert creates it with the necessary metadata
        # \u2014 but that path requires more than just DSL, so we error
        # rather than guess at scope/display_name. Use upsert_crystal_type
        # to create new types; this method is for editing the DSL on
        # existing types.
        existing_row = await self._metadata_store.get_crystal_type(type_id)
        if existing_row is None:
            raise SchemaLoadError(
                type_id,
                "cannot upsert DSL on a crystal_type row that does not "
                "exist; create the row first via upsert_crystal_type",
            )

        # Mutate and re-upsert. CrystalType is a Pydantic model; produce
        # a copy with the new DSL.
        updated = existing_row.model_copy(update={"pair_schema_dsl": dsl_source})
        await self._metadata_store.upsert_crystal_type(updated)

        # Invalidate cache and force reload so the returned compiled
        # object is post-write state.
        self.invalidate(type_id)
        await self._ensure_loaded(type_id)
        compiled = self._cache.get(type_id)
        if compiled is None:
            # Should be unreachable \u2014 we just persisted the row and
            # ensure_loaded should have populated the cache.
            raise SchemaLoadError(
                type_id,
                "post-upsert cache miss \u2014 upsert succeeded but reload "
                "produced no compiled object",
            )
        return compiled

    # -----------------------------------------------------------------
    # Read helpers
    # -----------------------------------------------------------------

    def has(self, type_id: str) -> bool:
        """True if the type is in the cache. Synchronous; doesn't
        trigger a DB load. For the question \"does this type exist
        in the registry,\" use `await get(type_id) is not None`.
        """
        return type_id in self._cache

    def cached_ids(self) -> list[str]:
        """All type_ids currently in cache, sorted. Mostly for tests
        and inspector debug.
        """
        return sorted(self._cache.keys())


# ---------------------------------------------------------------------------
# Empty-DSL default
# ---------------------------------------------------------------------------


def _empty_dsl_default(
    type_id: str,
    *,
    display_name: Optional[str] = None,
    scope: Optional[str] = None,
    capacity_default: int = 50,
    autosplit_policy: str = "split",
    routing_threshold: Optional[float] = None,
    cleanup_threshold: Optional[float] = None,
) -> CompiledCrystalType:
    """The compiled object returned for empty pair_schema_dsl rows.

    Permissive default:
      - one open pair_type \"question_answer\" accepting `text` for
        both fields, no attribute requirements.
      - no route hint.
      - ACL defaults from the scope (general -> world read;
        customer/document/personal -> owner read).

    Inputs default to spec defaults if not supplied (test path); the
    DB-loaded path passes the row's metadata so the compiled object
    reflects what the operator actually configured.
    """
    # Scope inference: if the caller didn't supply scope, derive from
    # the type_id prefix. Fall through to 'general' if the id format
    # is malformed (the validator catches malformed ids; here we just
    # need something).
    if scope is None:
        if ":" in type_id:
            scope = type_id.split(":", 1)[0]
        else:
            scope = "general"

    if display_name is None:
        slug = type_id.split(":", 1)[1] if ":" in type_id else type_id
        display_name = slug.replace("_", " ").title() if slug else type_id

    pair_type = CompiledPairType(
        name="question_answer",
        prompt_field=CompiledField(
            role="prompt_field",
            field_name="prompt",
            type_tag="text",
            attrs=(),
        ),
        answer_field=CompiledField(
            role="answer_field",
            field_name="answer",
            type_tag="text",
            attrs=(),
        ),
    )

    if scope == "general":
        acl = (
            CompiledAclGrant(
                principal_kind="world",
                principal_id=None,
                grant_kind="read",
            ),
        )
    else:
        acl = (
            CompiledAclGrant(
                principal_kind="owner",
                principal_id=None,
                grant_kind="read",
            ),
        )

    return CompiledCrystalType(
        type_id=type_id,
        display_name=display_name,
        scope=scope,
        capacity_default=capacity_default,
        autosplit_policy=autosplit_policy,
        routing_threshold=routing_threshold,
        cleanup_threshold=cleanup_threshold,
        pair_types={"question_answer": pair_type},
        route_hint=None,
        acl_defaults=acl,
    )


def _merge_with_row_metadata(
    compiled: CompiledCrystalType,
    row: "CrystalType",
) -> CompiledCrystalType:
    """Reconcile compiled-from-DSL values with row-level metadata.

    Where the DSL specifies a value, the DSL wins. Where the DSL is
    silent (i.e. the compiler used its built-in default), we fall back
    to the row's column. This handles the case where the DSL declares
    only pair_types/route_when/acl and leaves headers to the row \u2014 a
    valid authoring style, especially for types where the operator
    has tuned capacity / thresholds via the inspector and doesn't want
    those values restated in DSL.

    The signal for \"DSL is silent\" is per-field:
      - capacity_default: spec default is 50; if compiled is 50,
        prefer the row's value. Author can override by writing
        `capacity 50` explicitly in DSL (we'll still see 50 either
        way; the row should also be 50 in this case so no drift).
      - autosplit_policy: spec default is 'split'; if compiled is
        'split', prefer the row's value (with the same caveat).
      - routing_threshold / cleanup_threshold: nullable. If the
        compiled value is None (DSL omitted), use the row's value
        (also nullable; both-None is fine).
      - display_name: derived-from-id default is title-cased slug.
        We can't reliably distinguish \"author wrote display_name
        explicitly with title-cased value\" from \"compiler derived
        the default,\" so the DSL always wins here. Authors who want
        the row's display_name should NOT write display_name in DSL.

    This is a small reconciliation surface; it could grow if more
    headers land. Keep it small and explicit so changes are
    inspectable.
    """
    capacity = (
        compiled.capacity_default
        if compiled.capacity_default != 50
        else row.capacity_default
    )
    autosplit = (
        compiled.autosplit_policy
        if compiled.autosplit_policy != "split"
        else row.autosplit_policy
    )
    routing = (
        compiled.routing_threshold
        if compiled.routing_threshold is not None
        else row.routing_threshold
    )
    cleanup = (
        compiled.cleanup_threshold
        if compiled.cleanup_threshold is not None
        else row.cleanup_threshold
    )
    return CompiledCrystalType(
        type_id=compiled.type_id,
        display_name=compiled.display_name,
        scope=compiled.scope,
        capacity_default=capacity,
        autosplit_policy=autosplit,
        routing_threshold=routing,
        cleanup_threshold=cleanup,
        pair_types=compiled.pair_types,
        route_hint=compiled.route_hint,
        acl_defaults=compiled.acl_defaults,
    )


# ---------------------------------------------------------------------------
# ACL default resolution (Phase 4.7)
# ---------------------------------------------------------------------------


def resolve_acl_defaults(
    acl_defaults: tuple[CompiledAclGrant, ...],
    *,
    crystal_id: str,
    owning_customer_id: str,
) -> list["CrystalAcl"]:
    """Translate a compiled type's ACL defaults into concrete CrystalAcl rows
    ready to persist for a freshly-created crystal.

    The compiler emits CompiledAclGrant objects with three principal
    kinds. This function resolves each to the (principal_type,
    principal_id) pair that crystal_acls expects:

      world   -> ("global", "world")
      owner   -> ("customer", owning_customer_id)
      literal -> ("customer", grant.principal_id)  [from the DSL string]

    The resolution from `owner` to a concrete customer_id is what
    couldn't happen at compile time — the compiled object is reused
    across customers, so the customer_id only enters at instantiation.

    Args:
        acl_defaults: tuple from CompiledCrystalType.acl_defaults.
        crystal_id: the freshly-created crystal these grants attach to.
        owning_customer_id: resolves the `owner` sentinel.

    Returns:
        List of CrystalAcl Pydantic models. Each has granted_at set
        to the current UTC time. Caller persists via
        MetadataStore.add_acl(); add_acl is idempotent so re-applying
        defaults to a crystal that already has matching grants is a
        no-op.

    Raises:
        ValueError: on an unknown principal_kind. Should be unreachable
            given the compiler emits only the three known kinds; the
            check is defensive against future drift.
    """
    # Local import to avoid a top-level cycle: models -> infrastructure
    # -> dsl/schema/loader -> models.
    from crystal_cache.models import CrystalAcl

    now = datetime.now(timezone.utc)
    rows: list["CrystalAcl"] = []
    for grant in acl_defaults:
        if grant.principal_kind == "world":
            principal_type = "global"
            principal_id = "world"
        elif grant.principal_kind == "owner":
            principal_type = "customer"
            principal_id = owning_customer_id
        elif grant.principal_kind == "literal":
            if grant.principal_id is None:
                # Compiler invariant: literal grants must carry a
                # non-None principal_id. If we see None here, the
                # compiler drifted.
                raise ValueError(
                    f"compiled literal ACL grant has no principal_id; "
                    f"compiler-loader drift"
                )
            principal_type = "customer"
            principal_id = grant.principal_id
        else:
            raise ValueError(
                f"unknown ACL principal_kind "
                f"{grant.principal_kind!r}; valid kinds are "
                f"'world', 'owner', 'literal'"
            )

        rows.append(CrystalAcl(
            crystal_id=crystal_id,
            principal_type=principal_type,
            principal_id=principal_id,
            grant=grant.grant_kind,
            granted_at=now,
        ))
    return rows


# ---------------------------------------------------------------------------
# FastAPI dependency hook (Phase 4.8)
# ---------------------------------------------------------------------------
#
# Mirrors the get_metadata_store / set_metadata_store pattern from
# infrastructure.metadata_store. The app lifespan installs the loader
# at startup; admin endpoints depend on get_schema_loader.

_loader: Optional[SchemaLoader] = None


def set_schema_loader(loader: Optional[SchemaLoader]) -> None:
    """Install (or clear) the process-wide SchemaLoader.

    Called in app lifespan. Passing None clears the reference on shutdown.
    """
    global _loader
    _loader = loader


def get_schema_loader() -> SchemaLoader:
    """FastAPI dependency: returns the active SchemaLoader.

    Raises RuntimeError if the loader hasn't been initialized — this is a
    startup configuration error, not a request error.
    """
    if _loader is None:
        raise RuntimeError(
            "SchemaLoader not initialized. "
            "Call set_schema_loader() in app lifespan."
        )
    return _loader
