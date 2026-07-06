"""Metadata store — async session management and targeted CRUD.

Wraps SQLAlchemy's async engine + async session factory. Everything in
the app goes through this to talk to Postgres (or SQLite in dev).

Design notes:
  - One MetadataStore per process, constructed in the FastAPI lifespan
  - Helpers for specific domain operations (create_customer, write_query_log)
    rather than generic CRUD — generic methods would add abstraction before
    the access patterns are proven.
  - All helpers return Pydantic entities, never ORM rows. The ORM layer
    is an internal detail.
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

import numpy as np
import structlog
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ..config import Settings, get_settings
from ..encoding.base import BindCapableEncoder, TextEncoder
from ..encoding.executor import encode_async, encode_native_async
from ..models import (
    User,
    Crystal,
    CrystalAcl,
    CrystalChain,
    CrystalDiagnostic,
    CrystalEdit,
    CrystalType,
    Customer,
    Fact,
    Feedback,
    ModelRoutingConfig,
    Operator,
    QueryLog,
)
from .schema import (
    UserRow,
    Base,
    CrystalAclRow,
    CrystalChainRow,
    CrystalDiagnosticRow,
    CrystalEditRow,
    CrystalRow,
    CrystalTypeRow,
    CustomerRow,
    DslConfigRow,
    FactRow,
    FeedbackRow,
    GroupMemberRow,
    GroupRow,
    OperatorRow,
    QueryLogRow,
)
from .metadata_store_vec import register_sqlite_vec_loader

if TYPE_CHECKING:
    from ..decomposer.base import Decomposer
    from ..dsl.schema.loader import SchemaLoader
    from ..learning.bonder import Bonder
    from .fact_vector_store import FactVectorStore
    from .vector_index import VectorIndex
    from .vector_store import VectorStore

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Phase 1.1 / 1.2 capacity constants
# ---------------------------------------------------------------------------

class CrystalCapacityError(Exception):
    """Raised when a write would exceed a crystal's hard capacity AND the
    crystal_type's autosplit_policy is 'refuse'.

    Phase 1.2's default behavior is auto-split (the registry rows seeded
    by migration 0012 use autosplit_policy='split'). The 'refuse' policy
    is opt-in for crystal types where the operator wants explicit
    control over partitioning rather than implicit sibling spawning —
    e.g. medical-records or audit-log types where lineage matters and
    auto-spawning would surprise downstream consumers.

    The error carries the crystal_id and the resolved hard ceiling so
    inspector and operator UIs can surface what hit the limit.
    """

    def __init__(
        self,
        crystal_id: str,
        crystal_type: str,
        hard_ceiling: int,
        fact_count: int,
    ) -> None:
        self.crystal_id = crystal_id
        self.crystal_type = crystal_type
        self.hard_ceiling = hard_ceiling
        self.fact_count = fact_count
        super().__init__(
            f"crystal {crystal_id!r} of type {crystal_type!r} is at "
            f"capacity ({fact_count}/{hard_ceiling}) and its type's "
            f"autosplit_policy is 'refuse'. The write was rejected. "
            f"Either raise the type's capacity_default, switch its "
            f"autosplit_policy to 'split', or route the write to a "
            f"different crystal."
        )


class PairTypeValidationError(Exception):
    """Raised when a write's `pair_type` is not declared by the crystal
    type's compiled schema (Phase 4.5).

    This fires when a SchemaLoader is supplied to add_pair_to_crystal /
    add_pair_for_customer AND the crystal's crystal_type has a compiled
    schema in the loader AND the requested pair_type isn't in that
    schema's `pair_types` map. Validation is opt-in via the loader
    keyword — callers that don't pass one (existing tests, the legacy
    import path) skip validation and write whatever pair_type string
    they want.

    Catches typo'd pair_types at write time ("questoin_answer" instead
    of "question_answer") and prevents writers from inventing pair_type
    names that recall code can't interpret. The crystal's compiled
    schema is the source of truth for what's a valid pair_type.

    Carries the crystal_type id and the list of valid pair_types so
    inspector / admin UIs can render the rejection with a fix
    suggestion.
    """

    def __init__(
        self,
        crystal_id: str,
        crystal_type: str,
        attempted_pair_type: str,
        valid_pair_types: list[str],
    ) -> None:
        self.crystal_id = crystal_id
        self.crystal_type = crystal_type
        self.attempted_pair_type = attempted_pair_type
        self.valid_pair_types = list(valid_pair_types)
        valid_str = (
            ", ".join(repr(p) for p in self.valid_pair_types)
            if self.valid_pair_types
            else "<none declared>"
        )
        super().__init__(
            f"crystal {crystal_id!r} of type {crystal_type!r} rejects "
            f"pair_type {attempted_pair_type!r}; the type's compiled "
            f"schema declares: {valid_str}. Either declare "
            f"{attempted_pair_type!r} in the type's pair_schema_dsl, "
            f"correct the typo at the call site, or omit the "
            f"schema_loader argument to skip validation."
        )


# Per-crystal soft capacity ceiling. Empirically validated in the v2
# spike (research module's KnowledgeCrystal): bind-storage recall stays
# clean to roughly 30 pairs and degrades steadily past that. 50 is the
# point where we start warning; the hard ceiling kicks in just above.
#
# Implementation note: the warning fires EXACTLY ONCE per crystal,
# at the write that pushes fact_count to this value. Firing on every
# write past 50 would flood the inspector logs for any healthy
# long-lived crystal.
CRYSTAL_CAPACITY_SOFT_CEILING: int = 50

# Per-crystal HARD capacity ceiling (Phase 1.2). When a write would
# push fact_count past this number, `add_pair_to_crystal` does NOT
# refuse — refusing would be terrible UX. Instead it auto-splits:
# spawn a sibling crystal (parent_crystal_id = the original,
# inheriting customer_id, source_kind, summary_text, encoder_fingerprint)
# and write the pair there. The returned Fact's crystal_id is the
# sibling's id; callers that compare against the requested crystal_id
# can detect the redirect.
#
# Today this equals the soft ceiling (50). Phase 3 introduces
# crystal_type registry; per-type ceilings (default 50, customer-tier
# may raise) lift the global into a config lookup. Until then,
# everything is global.
#
# The ceiling is a number-of-pairs counter, not a math correctness
# boundary. Past this point, recall accuracy degrades but doesn't
# break catastrophically; the auto-split keeps individual crystals
# in the recoverable regime.
CRYSTAL_CAPACITY_HARD_CEILING: int = 50


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class MetadataStore:
    """Async SQLAlchemy session factory + domain CRUD."""

    def __init__(self, settings_override: Settings | None = None) -> None:
        # Resolve the active settings at construction time, not at import
        # time. This lets tests that override CC_DATABASE_URL via environment
        # (after clearing the get_settings cache) actually take effect.
        cfg = settings_override or get_settings()
        self._engine: AsyncEngine = create_async_engine(
            cfg.database_url,
            echo=cfg.database_echo,
            future=True,
        )
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        # Self-host vector backend (CC_VECTOR_BACKEND=sqlite_vec). Load the
        # sqlite-vec extension on every pooled connection of this engine so the
        # vec0 fact index + the routing vec_distance_cosine scan work. No-op for
        # non-SQLite engines, and broad-except guarded inside the loader, so it
        # can never break store construction for the memory/qdrant backends.
        if self._engine.dialect.name == "sqlite":
            register_sqlite_vec_loader(self._engine)

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    async def init(self) -> None:
        """Create tables if they don't exist. For dev/test only —
        production uses Alembic migrations.

        This method DOES NOT seed any registry data. Production seeding
        of the legacy crystal_type rows happens in migration 0012's
        bulk_insert step. Tests that need the legacy registry rows
        seeded against an in-memory DB should call
        `_seed_legacy_crystal_types_for_tests()` from their fixture.
        Keeping seeding out of init() avoids quietly running migration-
        equivalent writes against any process that calls init() —
        production startup hooks, dev scripts, etc.
        """
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def _seed_legacy_crystal_types_for_tests(self) -> None:
        """Test-only helper: seed the two legacy registry rows that
        migration 0012 inserts in production.

        Use ONLY from test fixtures that bypass Alembic by calling
        `init()` against an in-memory SQLite DB. Production code paths
        must not call this; they go through Alembic's bulk_insert in
        migration 0012 instead.

        Idempotent: re-seeding an already-seeded DB is a no-op via
        `upsert_crystal_type`'s upsert semantics.
        """
        await self.upsert_crystal_type(CrystalType(
            id="general:legacy",
            display_name="General (legacy catch-all)",
            scope="general",
        ))
        await self.upsert_crystal_type(CrystalType(
            id="customer:legacy",
            display_name="Customer (legacy catch-all)",
            scope="customer",
        ))

    async def dispose(self) -> None:
        """Close the engine and its connection pool. Call on app shutdown."""
        await self._engine.dispose()

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield an AsyncSession in a transaction.

        Commits on success; rolls back on exception.
        """
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    # -----------------------------------------------------------------
    # Customer CRUD (minimum needed for Part A)
    # -----------------------------------------------------------------

    async def create_customer(
        self,
        provider: str,
        model_id: str,
        api_key_ref: str,
        base_url: Optional[str] = None,
        injection_preference: str = "text",
        shadow_sample_rate: float = 0.05,
    ) -> Customer:
        """Create a new customer with a server-generated api_key (Key A).

        Returns the full Customer including the api_key. This is the
        ONLY time the api_key is returned — subsequent reads do not
        expose it (see get_customer_by_id).
        """
        from .credentials import generate_api_key, hash_api_key
        from .token_crypto import encrypt_secret

        customer_id = f"cus_{uuid.uuid4().hex[:16]}"
        api_key = generate_api_key()  # raw key, returned once below
        # Key B is encrypted at rest UNCONDITIONALLY (launch-prep security
        # pass) — requires CC_TOKEN_ENCRYPTION_KEY; there is no plaintext
        # fallback. Empty refs (nothing to protect) pass through.
        routing = ModelRoutingConfig(
            provider=provider,
            model_id=model_id,
            api_key_ref=encrypt_secret(api_key_ref) if api_key_ref else api_key_ref,
            base_url=base_url,
        )
        customer = Customer(
            id=customer_id,
            api_key=api_key,
            model_routing_config=routing,
            injection_preference=injection_preference,  # type: ignore[arg-type]
            shadow_sample_rate=shadow_sample_rate,
        )

        async with self.session() as session:
            row = CustomerRow(
                id=customer.id,
                api_key_hash=hash_api_key(api_key),
                model_routing_config=routing.model_dump(),
                injection_preference=customer.injection_preference,
                shadow_sample_rate=customer.shadow_sample_rate,
                routing_context_window=customer.routing_context_window,
                shadow_max_per_day=customer.shadow_max_per_day,
                retention_policy=customer.retention_policy,
                billing_config=customer.billing_config,
                created_at=customer.created_at,
            )
            session.add(row)

        # P1 identity chain (ratified 2026-07-02): every team is born with
        # its default admin operator, so the team key resolves to an owner
        # from the very first request.
        await self.ensure_default_admin(customer.id)

        return customer

    async def get_customer_by_id(self, customer_id: str) -> Optional[Customer]:
        async with self.session() as session:
            row = await session.get(CustomerRow, customer_id)
            if row is None:
                return None
            return _customer_from_row(row)

    # ------------------------------------------------------------------
    # Task-scoped keys (Phase 3 G3, 2026-07-03, ratified): the disposable
    # box's only credential. Hash at rest; single key per task; budget +
    # expiry + revocation are all enforced at resolve time by callers.
    # ------------------------------------------------------------------

    async def mint_task_key(
        self,
        customer_id: str,
        task_id: str,
        *,
        budget_micro_usd: int,
        ttl_seconds: int,
    ) -> tuple[str, "TaskKey"]:
        """Create the one key for a task. Returns (raw_key, record) — the
        raw key exists only in this return value; at rest there is only
        the hash. Re-minting an existing task_id is an integrity error
        (one task, one key, one lifecycle)."""
        from ..models.task_key import TaskKey
        from .credentials import generate_api_key, hash_api_key
        from .schema import TaskKeyRow

        raw = "ck_task_" + generate_api_key().split("_")[-1]
        now = datetime.now(timezone.utc)
        row = TaskKeyRow(
            task_id=task_id,
            key_hash=hash_api_key(raw),
            customer_id=customer_id,
            budget_micro_usd=int(budget_micro_usd),
            expires_at=now + timedelta(seconds=int(ttl_seconds)),
            revoked_at=None,
            created_at=now,
        )
        async with self.session() as session:
            session.add(row)
        return raw, TaskKey(
            task_id=task_id, customer_id=customer_id,
            budget_micro_usd=int(budget_micro_usd),
            expires_at=row.expires_at, revoked_at=None, created_at=now,
        )

    async def resolve_task_key(self, raw_key: str) -> Optional["TaskKey"]:
        """Hash the presented key and return the LIVE record: unknown,
        revoked, and expired keys all resolve to None identically (no
        distinguishing oracle for an attacker holding a dead key)."""
        from ..models.task_key import TaskKey
        from .credentials import hash_api_key
        from .schema import TaskKeyRow

        async with self.session() as session:
            stmt = select(TaskKeyRow).where(
                TaskKeyRow.key_hash == hash_api_key(raw_key)
            )
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row is None or row.revoked_at is not None:
                return None
            expires = row.expires_at
            if expires is not None and expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
            if expires is None or expires <= datetime.now(timezone.utc):
                return None
            return TaskKey(
                task_id=row.task_id, customer_id=row.customer_id,
                budget_micro_usd=row.budget_micro_usd,
                expires_at=row.expires_at, revoked_at=row.revoked_at,
                created_at=row.created_at,
            )

    async def revoke_task_key(self, task_id: str) -> bool:
        """The kill switch's twin (teardown revokes). Idempotent."""
        from .schema import TaskKeyRow

        async with self.session() as session:
            row = await session.get(TaskKeyRow, task_id)
            if row is None:
                return False
            if row.revoked_at is None:
                row.revoked_at = datetime.now(timezone.utc)
            return True

    async def task_spend_micro_usd(self, task_id: str) -> int:
        """Cumulative ledger spend attributed to a task (session_id =
        task_id) — the CostReader for budget enforcement, both at the
        proxy door and in the remote-task monitor."""
        from .schema import LlmCallRow

        async with self.session() as session:
            stmt = select(
                func.coalesce(func.sum(LlmCallRow.computed_cost_micro_usd), 0)
            ).where(LlmCallRow.session_id == task_id)
            return int((await session.execute(stmt)).scalar_one())

    async def set_customer_inference_mode(
        self, customer_id: str, mode: str
    ) -> Optional[Customer]:
        """Flip a tenant between managed and byok inference (E4 / Phase C
        settings surface, 2026-07-06). Caller validates the mode string
        and the byok-requires-Key-B rule; this just persists."""
        async with self.session() as session:
            row = await session.get(CustomerRow, customer_id)
            if row is None:
                return None
            row.inference_mode = mode
            return _customer_from_row(row)

    async def get_customer_by_api_key(self, api_key: str) -> Optional[Customer]:
        """Used by the auth dependency. Hashes the presented key and matches
        the stored hash (no plaintext at rest, 2026-06-13)."""
        from .credentials import hash_api_key

        async with self.session() as session:
            stmt = select(CustomerRow).where(
                CustomerRow.api_key_hash == hash_api_key(api_key)
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return _customer_from_row(row)

    # -----------------------------------------------------------------
    # Operator CRUD (Foundation F1 — team identity layer)
    # -----------------------------------------------------------------
    #
    # Security posture (2026-06-13): raw keys are never stored. A
    # server-generated key is returned ONCE from create_operator; only
    # its hash (credentials.hash_api_key) is persisted, and auth looks
    # operators up by hashing the presented key. See
    # infrastructure/credentials.py.

    async def ensure_default_admin(self, team_id: str) -> Operator:
        """The identity-chain keystone (P1, ratified 2026-07-02): every team
        has exactly one DEFAULT ADMIN operator, and the bare team/customer
        API key ACTS AS that operator — so every request has an operator
        and every crystal can have an owner.

        Deterministic id → idempotent get-or-create, which also SELF-HEALS
        customers created before this existed (no migration; the first
        resolution creates it). Carries NO api_key_hash: it authenticates
        only through customer-key resolution (ingress/auth.resolve_actor),
        never with a key of its own.
        """
        import hashlib

        operator_id = (
            "opdef_" + hashlib.sha1(team_id.encode("utf-8")).hexdigest()[:16]
        )
        existing = await self.get_operator_by_id(operator_id)
        if existing is not None:
            return existing
        operator = Operator(
            id=operator_id,
            team_id=team_id,
            display_name="Default Admin",
            role="admin",
            status="active",
            api_key_hash=None,
        )
        try:
            async with self.session() as session:
                session.add(OperatorRow(
                    id=operator.id,
                    team_id=operator.team_id,
                    display_name=operator.display_name,
                    role=operator.role,
                    status=operator.status,
                    api_key_hash=None,
                    credential_public_key=None,
                    created_at=operator.created_at,
                ))
        except Exception:
            # Concurrent first-resolution race: the deterministic PK makes
            # the loser's insert collide — re-read and return the winner's.
            raced = await self.get_operator_by_id(operator_id)
            if raced is not None:
                return raced
            raise
        return operator

    async def create_operator(
        self,
        team_id: str,
        display_name: str,
        role: str = "operator",
    ) -> tuple[Operator, str]:
        """Create an operator under a team; return (Operator, raw_api_key).

        The raw key is returned ONLY here — it is hashed for storage and
        not recoverable afterward. The returned Operator carries the hash
        in `api_key_hash`, never the raw key.
        """
        from .credentials import generate_api_key, hash_api_key

        operator_id = f"op_{uuid.uuid4().hex[:16]}"
        raw_key = generate_api_key()
        operator = Operator(
            id=operator_id,
            team_id=team_id,
            display_name=display_name,
            role=role,  # type: ignore[arg-type]
            status="active",
            api_key_hash=hash_api_key(raw_key),
        )
        async with self.session() as session:
            row = OperatorRow(
                id=operator.id,
                team_id=operator.team_id,
                display_name=operator.display_name,
                role=operator.role,
                status=operator.status,
                api_key_hash=operator.api_key_hash,
                credential_public_key=operator.credential_public_key,
                created_at=operator.created_at,
            )
            session.add(row)
        return operator, raw_key

    async def get_operator_by_id(self, operator_id: str) -> Optional[Operator]:
        async with self.session() as session:
            row = await session.get(OperatorRow, operator_id)
            return _operator_from_row(row) if row else None

    async def get_operator_by_api_key(
        self, api_key: str
    ) -> Optional[Operator]:
        """Auth lookup: hash the presented key, match the stored hash.

        Returns the operator regardless of status (active/suspended) so
        the auth dependency — not the store — decides to reject a
        suspended one, keeping 'suspended' distinguishable from 'unknown
        key'.
        """
        from .credentials import hash_api_key

        key_hash = hash_api_key(api_key)
        if not key_hash:
            return None
        async with self.session() as session:
            stmt = select(OperatorRow).where(
                OperatorRow.api_key_hash == key_hash
            )
            row = (await session.execute(stmt)).scalar_one_or_none()
            return _operator_from_row(row) if row else None

    # -- Users (hosted-platform accounts, Accounts Phase A 2026-07-06) ----
    # The IdP-anchored sign-in layer. id = Identity Platform uid, so JWT
    # resolution is a primary-key get. R9: this file owns the SQL.

    async def create_user(
        self,
        user_id: str,
        email: str,
        customer_id: Optional[str] = None,
        role: str = "owner",
    ) -> User:
        """Create a platform account (uid comes from the IdP, never minted
        here). platform_admin accounts carry customer_id=None."""
        user = User(
            id=user_id,
            email=email,
            customer_id=customer_id,
            role=role,  # type: ignore[arg-type]
        )
        async with self.session() as session:
            session.add(UserRow(
                id=user.id,
                email=user.email,
                customer_id=user.customer_id,
                role=user.role,
                industry=user.industry,
                building=user.building,
                experience=user.experience,
                created_at=user.created_at,
                updated_at=user.updated_at,
            ))
        return user

    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        async with self.session() as session:
            row = await session.get(UserRow, user_id)
            return _user_from_row(row) if row else None

    async def get_user_by_email(self, email: str) -> Optional[User]:
        async with self.session() as session:
            stmt = select(UserRow).where(UserRow.email == email)
            row = (await session.execute(stmt)).scalar_one_or_none()
            return _user_from_row(row) if row else None

    async def update_user_onboarding(
        self,
        user_id: str,
        industry: Optional[str] = None,
        building: Optional[str] = None,
        experience: Optional[str] = None,
    ) -> Optional[User]:
        """Record onboarding answers; only provided fields change."""
        async with self.session() as session:
            row = await session.get(UserRow, user_id)
            if row is None:
                return None
            if industry is not None:
                row.industry = industry
            if building is not None:
                row.building = building
            if experience is not None:
                row.experience = experience
            return _user_from_row(row)

    async def list_operators_for_team(
        self, team_id: str
    ) -> list[Operator]:
        """All operators under a team, newest first."""
        async with self.session() as session:
            stmt = (
                select(OperatorRow)
                .where(OperatorRow.team_id == team_id)
                .order_by(OperatorRow.created_at.desc())
            )
            result = await session.execute(stmt)
            return [_operator_from_row(r) for r in result.scalars().all()]

    async def set_operator_role(self, operator_id: str, role: str) -> bool:
        """Change an operator's role. Returns False if not found."""
        async with self.session() as session:
            stmt = (
                update(OperatorRow)
                .where(OperatorRow.id == operator_id)
                .values(role=role)
            )
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount > 0

    async def set_operator_status(
        self, operator_id: str, status: str
    ) -> bool:
        """Activate / suspend an operator. Returns False if not found.

        Suspension preserves the row (and the operator's owned crystals +
        provenance); auth denies a suspended operator at the boundary.
        """
        async with self.session() as session:
            stmt = (
                update(OperatorRow)
                .where(OperatorRow.id == operator_id)
                .values(status=status)
            )
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount > 0

    async def list_customers(self, limit: int = 1000) -> list[Customer]:
        """List all customers across all tenants. Admin-only — there is
        no scoping. Used by the inspector UI's customer selector.

        Returns customers ordered by created_at descending so the most
        recently created customer is first. Limit defaults to 1000
        which is way more than any realistic dev environment will have;
        if the inspector ever runs against a real production DB we'll
        need pagination.
        """
        async with self.session() as session:
            stmt = (
                select(CustomerRow)
                .order_by(CustomerRow.created_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            return [_customer_from_row(r) for r in result.scalars().all()]

    async def count_crystals_for_customer(self, customer_id: str) -> int:
        """Count crystals for a single customer. Used by the inspector
        to render `crystal_count` on the customer list without sending
        the full crystal payload.
        """
        from sqlalchemy import func
        async with self.session() as session:
            stmt = (
                select(func.count(CrystalRow.id))
                .where(CrystalRow.customer_id == customer_id)
            )
            result = await session.execute(stmt)
            return int(result.scalar_one())

    async def list_query_logs_for_customer(
        self,
        customer_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[int, list[QueryLog]]:
        """Return (total_count, recent_rows) for a customer's query logs.

        Ordered by timestamp descending. The total is computed against
        the same WHERE clause so the inspector can show "showing 100 of
        5,234" without a second roundtrip.
        """
        from sqlalchemy import func
        async with self.session() as session:
            count_stmt = (
                select(func.count(QueryLogRow.id))
                .where(QueryLogRow.customer_id == customer_id)
            )
            total = int((await session.execute(count_stmt)).scalar_one())

            stmt = (
                select(QueryLogRow)
                .where(QueryLogRow.customer_id == customer_id)
                .order_by(QueryLogRow.timestamp.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await session.execute(stmt)
            items = [_query_log_from_row(r) for r in result.scalars().all()]
            return total, items

    async def list_crystals_for_customer_paginated(
        self,
        customer_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[int, list[Crystal]]:
        """Paginated crystal listing for the inspector's bank browser.

        Ordered by created_at descending. Returns (total_count, page).
        """
        from sqlalchemy import func
        async with self.session() as session:
            count_stmt = (
                select(func.count(CrystalRow.id))
                .where(CrystalRow.customer_id == customer_id)
            )
            total = int((await session.execute(count_stmt)).scalar_one())

            stmt = (
                select(CrystalRow)
                .where(CrystalRow.customer_id == customer_id)
                .order_by(CrystalRow.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await session.execute(stmt)
            items = [_crystal_from_row(r) for r in result.scalars().all()]
            return total, items

    # -----------------------------------------------------------------
    # QueryLog CRUD (minimum needed for Part A)
    # -----------------------------------------------------------------

    async def write_query_log(self, log: QueryLog) -> None:
        """Append a query log entry. Called on every chat completion."""
        async with self.session() as session:
            row = QueryLogRow(
                id=log.id,
                customer_id=log.customer_id,
                query_text=log.query_text,
                query_vector=log.query_vector,
                match_type=log.match_type,
                injection_method=log.injection_method,
                confidence_gate_fires=log.confidence_gate_fires,
                matched_facts=log.matched_facts,
                response_text=log.response_text,
                response_confidence_at_commit=log.response_confidence_at_commit,
                upstream_call_made=log.upstream_call_made,
                shadow_ran=log.shadow_ran,
                shadow_delta=log.shadow_delta,
                prompt_tokens=log.prompt_tokens,
                completion_tokens=log.completion_tokens,
                shadow_prompt_tokens=log.shadow_prompt_tokens,
                shadow_completion_tokens=log.shadow_completion_tokens,
                prompt_token_overhead=log.prompt_token_overhead,
                concept_top_config=log.concept_top_config,
                concept_top_score=log.concept_top_score,
                concept_payload=log.concept_payload,
                sequence_id=log.sequence_id,
                turn_index=log.turn_index,
                routed_crystal_id=log.routed_crystal_id,
                top1_score=log.top1_score,
                top2_score=log.top2_score,
                latency_ms=log.latency_ms,
                timestamp=log.timestamp,
            )
            session.add(row)

    async def next_turn_index(
        self, customer_id: str, sequence_id: str
    ) -> int:
        """Return the next turn_index to assign for (customer_id, sequence_id).

        Counts existing QueryLog rows for that sequence and returns the
        count — i.e. the next 0-based position. If no rows exist yet,
        returns 0.

        Race condition note: under concurrent requests for the same
        sequence, two requests can both read N and both write a row
        with turn_index=N. Acceptable for v0 — sequence_id collisions
        across concurrent users for the same conversation are rare,
        and turn_index is a soft grouping rather than a uniqueness
        constraint. If we ever need strict ordering we can switch to a
        per-sequence counter row with row-level locking.
        """
        from sqlalchemy import func
        async with self.session() as session:
            stmt = (
                select(func.count(QueryLogRow.id))
                .where(QueryLogRow.customer_id == customer_id)
                .where(QueryLogRow.sequence_id == sequence_id)
            )
            result = await session.execute(stmt)
            return int(result.scalar_one())

    async def list_query_logs_for_crystal(
        self,
        crystal_id: str,
        window_hours: int = 168,
        limit: int = 1000,
    ) -> list[QueryLog]:
        """Fetch query logs that touched a given crystal within window_hours.

        A query "touched" a crystal if the crystal's id appears in
        matched_facts (which for v0 stores crystal ids, not fact ids —
        this is a near-term schema evolution point). Since today's
        chat_completions doesn't populate matched_facts, this returns []
        for most customers until retrieval is wired.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        async with self.session() as session:
            stmt = (
                select(QueryLogRow)
                .where(QueryLogRow.timestamp >= cutoff)
                .order_by(QueryLogRow.timestamp.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                _query_log_from_row(r)
                for r in rows
                if crystal_id in (r.matched_facts or [])
            ]

    # -----------------------------------------------------------------
    # Crystal CRUD
    # -----------------------------------------------------------------

    async def upsert_crystal(self, crystal: Crystal) -> None:
        """Insert or update a crystal. The hot path doesn't use this
        today; it's the persistence target for offline bank builds."""
        async with self.session() as session:
            existing = await session.get(CrystalRow, crystal.id)
            data = {
                "customer_id": crystal.customer_id,
                "summary_vector": crystal.summary_vector,
                "routing_vector": crystal.routing_vector,
                "answer_embedding_native": crystal.answer_embedding_native,
                "encoder_fingerprint": crystal.encoder_fingerprint,
                "source_kind": crystal.source_kind,
                "answer_value": crystal.answer_value,
                "decay_rate": crystal.decay_rate,
                "fact_count": crystal.fact_count,
                "quality_tier": crystal.quality_tier,
                # Recall gate + birth attribution (2026-07-03). In the data
                # dict so both the insert and the setattr-update carry them.
                "recall_gated": crystal.recall_gated,
                "origin": crystal.origin,
                "eval_helped_count": crystal.eval_helped_count,
                "eval_hurt_count": crystal.eval_hurt_count,
                "live_shadow_helped_count": crystal.live_shadow_helped_count,
                "live_shadow_hurt_count": crystal.live_shadow_hurt_count,
                "keyword_fingerprint": crystal.keyword_fingerprint,
                "cluster_tightness": crystal.cluster_tightness,
                "attribution_spread": crystal.attribution_spread,
                "summary_text": crystal.summary_text,
                "build_method": crystal.build_method,
                "parent_crystal_id": crystal.parent_crystal_id,
                # V2 source versioning (VS-D2). The data dict feeds both
                # the insert (CrystalRow(**data)) and the update (setattr
                # loop), so adding the four keys here covers both paths.
                "source_path": crystal.source_path,
                "content_hash": crystal.content_hash,
                "source_modified_at": crystal.source_modified_at,
                "crystal_type": crystal.crystal_type,
                # Foundation F2 (POSIX permissions). In the data dict so
                # both the insert (CrystalRow(**data)) and the update
                # (setattr loop) carry owner/group/mode.
                "owner_operator_id": crystal.owner_operator_id,
                "group_team_id": crystal.group_team_id,
                "mode": crystal.mode,
                "decomposer_payload": crystal.decomposer_payload,
                "diagnostic_tags": crystal.diagnostic_tags,
                "last_eval_at": crystal.last_eval_at,
                "last_activity": crystal.last_activity,
            }
            if existing is None:
                session.add(CrystalRow(id=crystal.id, created_at=crystal.created_at, **data))
            else:
                for key, value in data.items():
                    setattr(existing, key, value)

    # ------------------------------------------------------------------
    # Topology-exact export/import — verdict 5, ratified 2026-07-02. The
    # bank arrives with earned trust intact: crystal identity, chains,
    # edges, tiers, scope stamps, conflicts, and citation provenance all
    # survive a round trip verbatim. Rows are serialized GENERICALLY via
    # table inspection, so the format tracks schema evolution instead of a
    # hand-maintained field list; unknown keys on import are dropped and
    # counted rather than fatal.
    # ------------------------------------------------------------------

    @staticmethod
    def _topology_row_to_dict(row) -> dict:
        out = {}
        for col in row.__table__.columns:
            val = getattr(row, col.name)
            if isinstance(val, datetime):
                val = val.isoformat()
            out[col.name] = val
        return out

    @staticmethod
    def _topology_dict_to_row(row_cls, data: dict):
        import sqlalchemy as _sa
        kwargs = {}
        dropped = 0
        cols = {c.name: c for c in row_cls.__table__.columns}
        for key, val in data.items():
            col = cols.get(key)
            if col is None:
                dropped += 1
                continue
            if (
                isinstance(col.type, _sa.DateTime)
                and isinstance(val, str)
            ):
                val = datetime.fromisoformat(val)
            kwargs[key] = val
        return row_cls(**kwargs), dropped

    async def export_bank_topology(self, customer_id: str) -> dict:
        """The topology-exact dump: every crystal row (vectors, tier, scope
        stamps, provenance), every fact, chains, co-query edges, knowledge
        conflicts, and citations — verbatim, id-preserving."""
        from .schema import CitationRow, CrystalEdgeRow, KnowledgeConflictRow

        async with self.session() as session:
            crystals = (await session.execute(
                select(CrystalRow).where(CrystalRow.customer_id == customer_id)
            )).scalars().all()
            crystal_ids = [c.id for c in crystals]
            facts = []
            chains = []
            edges = []
            if crystal_ids:
                facts = (await session.execute(
                    select(FactRow).where(FactRow.crystal_id.in_(crystal_ids))
                )).scalars().all()
                chains = (await session.execute(
                    select(CrystalChainRow).where(
                        CrystalChainRow.source_crystal_id.in_(crystal_ids)
                    )
                )).scalars().all()
                edges = (await session.execute(
                    select(CrystalEdgeRow).where(
                        CrystalEdgeRow.crystal_a_id.in_(crystal_ids)
                    )
                )).scalars().all()
            conflicts = (await session.execute(
                select(KnowledgeConflictRow).where(
                    KnowledgeConflictRow.customer_id == customer_id
                )
            )).scalars().all()
            citations = (await session.execute(
                select(CitationRow).where(
                    CitationRow.customer_id == customer_id
                )
            )).scalars().all()

            d = self._topology_row_to_dict
            return {
                "format": "crystal_topology_v2",
                "crystals": [d(r) for r in crystals],
                "facts": [d(r) for r in facts],
                "chains": [d(r) for r in chains],
                "edges": [d(r) for r in edges],
                "conflicts": [d(r) for r in conflicts],
                "citations": [d(r) for r in citations],
            }

    async def import_bank_topology(
        self, customer_id: str, payload: dict,
    ) -> dict:
        """Id-preserving restore of an export_bank_topology payload.

        Policies (documented, not silent):
          - customer_id and group_team_id are REWRITTEN to the importing
            team (a bank moves between tenants; its group is the new team).
          - Primary-key collisions are SKIPPED and counted — restore into a
            wiped or fresh bank for an exact copy.
          - An owner_operator_id unknown to the importing team is CLEARED
            (counted in owners_cleared): scope stays, ownership can't point
            at a ghost. Re-create operators first to keep ownership exact.
          - Unknown columns from a different schema version are dropped and
            counted, never fatal.
        """
        from .schema import CitationRow, CrystalEdgeRow, KnowledgeConflictRow

        counts = {
            "crystals": 0, "facts": 0, "chains": 0, "edges": 0,
            "conflicts": 0, "citations": 0, "skipped_collisions": 0,
            "owners_cleared": 0, "dropped_fields": 0,
        }
        exporter_team_ids = {
            c.get("customer_id") for c in payload.get("crystals", [])
        } - {None}

        async with self.session() as session:
            valid_ops = set((await session.execute(
                select(OperatorRow.id).where(
                    OperatorRow.team_id == customer_id
                )
            )).scalars().all())

            async def _insert(row_cls, data: dict, pk) -> bool:
                existing = await session.get(row_cls, pk)
                if existing is not None:
                    counts["skipped_collisions"] += 1
                    return False
                row, dropped = self._topology_dict_to_row(row_cls, data)
                counts["dropped_fields"] += dropped
                session.add(row)
                return True

            imported_crystals: set[str] = set()
            for c in payload.get("crystals", []):
                data = dict(c)
                data["customer_id"] = customer_id
                if data.get("group_team_id") in exporter_team_ids:
                    data["group_team_id"] = customer_id
                owner = data.get("owner_operator_id")
                if owner and owner not in valid_ops:
                    data["owner_operator_id"] = None
                    counts["owners_cleared"] += 1
                if await _insert(CrystalRow, data, data["id"]):
                    counts["crystals"] += 1
                    imported_crystals.add(data["id"])

            for f in payload.get("facts", []):
                if f.get("crystal_id") not in imported_crystals:
                    counts["skipped_collisions"] += 1
                    continue
                if await _insert(FactRow, dict(f), f["id"]):
                    counts["facts"] += 1

            for ch in payload.get("chains", []):
                if ch.get("source_crystal_id") not in imported_crystals:
                    counts["skipped_collisions"] += 1
                    continue
                if await _insert(
                    CrystalChainRow, dict(ch),
                    (ch["source_crystal_id"], ch["target_crystal_id"]),
                ):
                    counts["chains"] += 1

            for e in payload.get("edges", []):
                if e.get("crystal_a_id") not in imported_crystals:
                    counts["skipped_collisions"] += 1
                    continue
                if await _insert(
                    CrystalEdgeRow, dict(e),
                    (e["crystal_a_id"], e["crystal_b_id"],
                     e.get("edge_type", "co_queried")),
                ):
                    counts["edges"] += 1

            for k in payload.get("conflicts", []):
                data = dict(k)
                data["customer_id"] = customer_id
                if await _insert(KnowledgeConflictRow, data, data["id"]):
                    counts["conflicts"] += 1

            for ct in payload.get("citations", []):
                data = dict(ct)
                data["customer_id"] = customer_id
                if await _insert(CitationRow, data, data["id"]):
                    counts["citations"] += 1

        return counts

    async def list_crystal_ids_for_source_paths(
        self, customer_id: str, source_paths: list[str],
    ) -> list[str]:
        """Crystal ids stamped with any of the given source paths — the
        content-chunk half of share-source resolution (P4, ratified
        2026-07-02). Customer-guarded."""
        if not source_paths:
            return []
        async with self.session() as session:
            rows = (await session.execute(
                select(CrystalRow.id).where(
                    CrystalRow.customer_id == customer_id,
                    CrystalRow.source_path.in_(source_paths),
                )
            )).scalars().all()
            return list(rows)

    async def set_crystal_scope(
        self, crystal_id: str, customer_id: Optional[str], scope: str,
    ) -> bool:
        """The SHARE primitive (P2/P4, ratified 2026-07-02): move one
        crystal between personal (0o600) and team (0o640) by rewriting its
        POSIX mode. One reversible write — no copy, ownership unchanged.
        Customer-guarded; returns False when the crystal is unknown or
        foreign. Authorization (owner-or-admin) is the ENDPOINT's job."""
        from .permissions import mode_for_scope

        mode = mode_for_scope(scope)
        async with self.session() as session:
            row = await session.get(CrystalRow, crystal_id)
            if row is None:
                return False
            if customer_id is not None and row.customer_id != customer_id:
                return False
            row.mode = mode
        return True

    async def get_quality_tiers(
        self, crystal_ids: list[str], *, customer_id: Optional[str] = None,
    ) -> dict[str, str]:
        """{crystal_id: quality_tier} for the given ids (tier-signal read —
        retrieval/tier_signal.py). Customer-guarded when customer_id given;
        unknown ids simply absent from the map."""
        if not crystal_ids:
            return {}
        async with self.session() as session:
            stmt = select(CrystalRow.id, CrystalRow.quality_tier).where(
                CrystalRow.id.in_(crystal_ids)
            )
            if customer_id is not None:
                stmt = stmt.where(CrystalRow.customer_id == customer_id)
            rows = (await session.execute(stmt)).all()
            return {r[0]: r[1] for r in rows}

    async def list_thin_crystals_for_customer(
        self, customer_id: str, *, max_facts: int, limit: int = 50,
    ) -> list[dict]:
        """Crystals with 1..max_facts facts — the thin-coverage research
        signal for topic seeding (scan/topic_seeding.py). Excludes
        blacklisted crystals. Returns [{crystal_id, fact_count, sample_key}]
        where sample_key is one fact's sparse key (min() — a deterministic
        representative for subject/domain parsing)."""
        async with self.session() as session:
            stmt = (
                select(
                    FactRow.crystal_id,
                    func.count(FactRow.id),
                    func.min(FactRow.prompt_text),
                )
                .join(CrystalRow, CrystalRow.id == FactRow.crystal_id)
                .where(
                    CrystalRow.customer_id == customer_id,
                    CrystalRow.quality_tier != "blacklist",
                )
                .group_by(FactRow.crystal_id)
                .having(func.count(FactRow.id) <= max_facts)
                .order_by(func.count(FactRow.id).asc())
                .limit(limit)
            )
            rows = (await session.execute(stmt)).all()
            return [
                {
                    "crystal_id": r[0],
                    "fact_count": int(r[1]),
                    "sample_key": r[2] or "",
                }
                for r in rows
            ]

    async def set_crystal_quality_tier(
        self, crystal_id: str, customer_id: Optional[str], tier: str,
    ) -> bool:
        """Move one crystal's quality_tier (the tier-promotion writer).

        Customer-guarded when customer_id is given; returns False when no
        row matched (deleted crystal, wrong customer) so the scan can skip
        quietly."""
        async with self.session() as session:
            stmt = (
                update(CrystalRow)
                .where(CrystalRow.id == crystal_id)
                .values(quality_tier=tier)
            )
            if customer_id is not None:
                stmt = stmt.where(CrystalRow.customer_id == customer_id)
            result = await session.execute(stmt)
            return bool(result.rowcount)

    async def set_crystal_recall_gate(
        self, crystal_id: str, customer_id: Optional[str], gated: bool,
    ) -> bool:
        """Set (or clear) a crystal's recall gate (2026-07-03).

        Clearing the gate (gated=False) is PROMOTION: it makes a
        background-worker crystal usable in recall. Called by a human
        approval or a system_rules promotion rule. Customer-guarded when
        customer_id is given; returns False when no row matched.
        """
        async with self.session() as session:
            stmt = (
                update(CrystalRow)
                .where(CrystalRow.id == crystal_id)
                .values(recall_gated=gated)
            )
            if customer_id is not None:
                stmt = stmt.where(CrystalRow.customer_id == customer_id)
            result = await session.execute(stmt)
            return bool(result.rowcount)

    async def append_crystal_diagnostic_tags(
        self, crystal_id: str, customer_id: Optional[str], tags: list[str],
    ) -> bool:
        """Merge tags into a crystal's diagnostic_tags (2026-07-03, outbound
        review). Idempotent: already-present tags are not duplicated.
        Customer-guarded when customer_id is given; returns False when no
        row matched."""
        async with self.session() as session:
            row = await session.get(CrystalRow, crystal_id)
            if row is None:
                return False
            if customer_id is not None and row.customer_id != customer_id:
                return False
            current = list(row.diagnostic_tags or [])
            merged = current + [t for t in tags if t not in current]
            if merged != current:
                row.diagnostic_tags = merged
            return True

    async def list_recall_gated_crystals(
        self, customer_id: str, *, origin: Optional[str] = None,
    ) -> list[Crystal]:
        """List a customer's recall-gated crystals — the review/approval
        queue for un-promoted memory. Optionally filter by origin (e.g.
        'background_worker'). This is an ADMIN/review read, so it returns
        the gated rows the recall path hides.
        """
        async with self.session() as session:
            stmt = (
                select(CrystalRow)
                .where(CrystalRow.customer_id == customer_id)
                .where(CrystalRow.recall_gated.is_(True))
            )
            if origin is not None:
                stmt = stmt.where(CrystalRow.origin == origin)
            result = await session.execute(stmt)
            return [_crystal_from_row(r) for r in result.scalars().all()]

    async def get_crystal(self, crystal_id: str) -> Optional[Crystal]:
        async with self.session() as session:
            row = await session.get(CrystalRow, crystal_id)
            return _crystal_from_row(row) if row else None

    async def list_crystals_for_customer(
        self, customer_id: str, *, include_recall_gated: bool = True,
    ) -> list[Crystal]:
        # include_recall_gated defaults True so every existing caller (admin,
        # inspector, promotion, consolidation) is UNCHANGED and still sees
        # gated crystals — promotion in particular MUST see them to clear the
        # gate. Only the recall path passes False, so recall-gated crystals
        # never enter the candidate set until approved (2026-07-03).
        async with self.session() as session:
            stmt = select(CrystalRow).where(CrystalRow.customer_id == customer_id)
            if not include_recall_gated:
                stmt = stmt.where(CrystalRow.recall_gated.is_(False))
            result = await session.execute(stmt)
            return [_crystal_from_row(r) for r in result.scalars().all()]

    async def list_general_crystals(
        self, crystal_type: str, *, include_recall_gated: bool = True,
    ) -> list[Crystal]:
        """List general crystals (customer_id IS NULL) of a given type.

        Used by VectorStore to build general knowledge banks that
        customers can subscribe to. include_recall_gated defaults True to
        preserve existing behavior; the recall path passes False so gated
        general crystals are held out of the candidate set (2026-07-03).
        """
        async with self.session() as session:
            stmt = (
                select(CrystalRow)
                .where(CrystalRow.customer_id.is_(None))
                .where(CrystalRow.crystal_type == crystal_type)
            )
            if not include_recall_gated:
                stmt = stmt.where(CrystalRow.recall_gated.is_(False))
            result = await session.execute(stmt)
            return [_crystal_from_row(r) for r in result.scalars().all()]

    async def list_all_crystals(self, limit: int = 10000) -> list[Crystal]:
        """List crystals across all customers. Admin/dev tool, not for
        per-request use — there is no tenancy scoping here."""
        async with self.session() as session:
            stmt = select(CrystalRow).limit(limit)
            result = await session.execute(stmt)
            return [_crystal_from_row(r) for r in result.scalars().all()]

    # -----------------------------------------------------------------
    # Fact CRUD (Phase 1.1, April 2026 — bind-storage write API)
    # -----------------------------------------------------------------
    #
    # add_pair_to_crystal is the production write API for crystal-shaped
    # storage. Each call:
    #
    #   1. Encodes (prompt_text, answer_text) into HDC (10k-dim, unit-norm)
    #      via the supplied encoder.
    #   2. Computes the grating g = encode(prompt) * encode(answer)
    #      (elementwise bind — the storage primitive per Finding 15).
    #   3. Bundles g into the crystal's summary_vector via RAW additive
    #      accumulation. NO per-write renormalization — see the long
    #      note below.
    #   4. Encodes answer_text to native dim (768 for gtr-t5-base) and
    #      stores it on the new Fact row's `vector` field for codebook
    #      cleanup at recall time.
    #   5. Increments crystal.fact_count.
    #   6. Returns the new Fact.
    #
    # ---- On renormalization (write-time vs read-time) ----
    #
    # Earlier in Phase 1.1's design (see docs/PHASE_1_NOTES.md) we
    # planned to renormalize summary_vector after every accumulation.
    # That decision was wrong; the corrected approach is RAW
    # accumulation at write, normalization at read.
    #
    # Why renormalize-per-write breaks bind recall: when summary is
    # unit-norm before adding a new grating g_i (which has tiny norm,
    # ~0.01 with these projections), the bundle of [unit summary,
    # tiny g_i] renormalizes to weight g_i at ~0.01 of its addition.
    # The bundle of [unit summary, unit g_i+1, unit g_i+2, ...] then
    # has the most recent grating dominating older ones whose weights
    # were diluted by every subsequent renormalization. Late pairs
    # are buried in floating-point noise. Recovery via unbind +
    # reverse-project + cleanup loses signal entirely past a handful
    # of writes.
    #
    # Why raw accumulation works: each grating contributes at its
    # native bind magnitude (~0.01); after N writes every pair has
    # equal weight in the bundle, and unbind+reverse-project recovers
    # the right native answer with cosine well above noise. This
    # matches the research module's `KnowledgeCrystal.write` exactly
    # (memory += grating, no per-write normalization). Validated on
    # the FAQ bank in the v2 spike at 36/36 routing.
    #
    # Why downstream still gets unit-norm vectors: VectorStore loads
    # rows from the database into an in-memory matrix and now
    # normalizes each row at load time before computing cosine. That's
    # where the unit-norm contract is enforced — at the boundary
    # between persistence and the routing primitive, not in storage.
    # The four-way classifier consumes cosine scores produced by the
    # normalized matrix; its thresholds remain calibrated.
    #
    # ---- Capacity & encoder constraints ----
    #
    # Capacity ceiling enforcement is Phase 1.2's job; Phase 1.1 logs
    # a structlog warning at fact_count == 50 but does not refuse
    # writes (per resolved decision 3: "soft 50, raisable").
    #
    # Hash encoder is not supported. Bind-storage requires the
    # answer's native embedding for codebook cleanup, which only the
    # semantic encoder produces. The runtime check below catches this
    # with a clear error rather than letting a silent vector=[]
    # regression land. Per CLAUDE.md Hard Rule 15, hash encoder is
    # back-compat only.

    async def add_pair_to_crystal(
        self,
        crystal_id: str,
        prompt_text: str,
        answer_text: str,
        pair_type: str = "question_answer",
        *,
        encoder: BindCapableEncoder,
        source_kind: str = "model_reasoning",
        answer_value: Optional[str] = None,
        schema_loader: Optional["SchemaLoader"] = None,
        embed_text: Optional[str] = None,
    ) -> Fact:
        """Append one (prompt, answer) pair to a crystal's bind-storage memory.

        Phase 1.1 implementation, Phase 1.1-mitigations hardening,
        Phase 2 Fact-level provenance fields.
        Computes the grating, accumulates additively into summary_vector
        (no per-write normalization — see the module-level note),
        persists the answer's native embedding to the codebook, and
        stamps/verifies the encoder fingerprint to catch geometry drift.

        Args:
            crystal_id: existing crystal to write into. Raises ValueError
                if not found.
            prompt_text: the question / key text. Encoded to HDC; the
                grating's prompt component. Phase 2: also persisted on
                the Fact row's prompt_text column for the per-crystal
                cleanup_threshold calibrator (Phase 6.3) and inspector
                display.
            answer_text: the answer / value text. Encoded to HDC for
                the grating's answer component AND to native dim for
                the codebook entry.
            pair_type: DSL-declared pair type. Default "question_answer"
                for back-compat with Phase 0.1 callers; new write paths
                should specify explicitly. The default is preserved for
                pragmatic reasons — removing it now would break every
                existing test — but new callsites should pass an
                explicit pair_type. Phase 1.5+ will deprecate the
                default formally.
            encoder: bind-capable encoder (typed `BindCapableEncoder`,
                which the SemanticTextEncoder satisfies and the hash
                encoder does not). Required keyword-only argument. The
                same encoder fingerprint must be used across all writes
                to a single crystal; mismatches raise ValueError.
            source_kind: what kind of evidence this Fact carries
                (Phase 2, migration 0011). Default "model_reasoning"
                preserves existing callers' semantics: every Fact
                authored from verified (question, answer) pairs is
                a success Fact unless explicitly tagged otherwise.
                The pipeline's cache-hit short-circuit fires when
                this is "model_reasoning" AND answer_value is set.
            answer_value: canonical short answer for the cache-hit
                short-circuit. Default None means "no separate cache-
                shaped value worth storing; claim_text is the answer".
                Pass an explicit value when the Fact is authored from
                a verified short-answer pair (FAQ knowledge base,
                model_reasoning crystallizer outputs) so the pipeline
                can return the answer directly without invoking
                upstream on PERFECT-decision matches.
            schema_loader: optional Phase 4 SchemaLoader (default None
                = skip validation). When provided, the crystal's
                crystal_type is looked up in the loader; if a compiled
                schema is present, `pair_type` must be one of its
                declared pair_types. Mismatch raises
                PairTypeValidationError. When the loader returns None
                for the type (not registered), validation is silently
                skipped — unregistered types are tolerated for back-
                compat. Existing test fixtures and the legacy import
                path don't pass a loader and continue to write any
                pair_type string. Production code paths going through
                add_pair_for_customer thread the loader through, so
                the gateway path gets validation.

        Returns:
            The persisted Fact, with `vector` populated as the answer's
            native (typically 768-dim) unit-norm embedding, and the
            three Phase 2 provenance fields populated as passed.

        Raises:
            ValueError: if crystal_id does not exist; if the encoder
                does not expose encode_native (hash-encoder rejection);
                or if the encoder's fingerprint does not match the
                fingerprint stamped on the crystal by a prior write.
            PairTypeValidationError: if `schema_loader` is provided AND
                the crystal's crystal_type has a compiled schema in
                the loader AND `pair_type` is not in that schema's
                declared pair_types. Skipped silently when the loader
                is omitted or the type isn't registered.
        """
        # Hash-encoder rejection. The BindCapableEncoder Protocol
        # signals at type-check time that the encoder must expose
        # encode_native + fingerprint; the runtime hasattr check below
        # is belt-and-suspenders for callers that bypass the type
        # system (e.g. test doubles, dynamic dispatch). Bind-storage
        # requires the answer's native embedding for codebook cleanup,
        # which only the semantic encoder provides. Per CLAUDE.md
        # Hard Rule 15, hash encoder is back-compat only.
        if not hasattr(encoder, "encode_native"):
            raise ValueError(
                "add_pair_to_crystal requires a BindCapableEncoder "
                "(must expose `encode_native`). The hash encoder is "
                "not supported for bind-storage writes — it has no "
                "native dim, and codebook cleanup at recall time "
                "requires native answer embeddings. Pass a "
                "SemanticTextEncoder instance instead. Per CLAUDE.md "
                "Hard Rule 15, hash encoder is back-compat only."
            )
        if not hasattr(encoder, "fingerprint"):
            raise ValueError(
                "add_pair_to_crystal requires a BindCapableEncoder "
                "(must expose `fingerprint()`). Without a fingerprint "
                "there's no way to detect encoder-geometry drift "
                "between writes, and bind-v1 silently misdecodes "
                "out-of-distribution recovered vectors. Pass a "
                "SemanticTextEncoder instance instead."
            )

        # Capture the fingerprint up-front so we can compare it against
        # the crystal's stamped fingerprint inside the session. Cheap
        # operation — just a string format.
        encoder_fp = encoder.fingerprint()

        # Compute the bind grating in HDC space. Both encodes return
        # unit-norm float32 vectors of shape (d_hdc,); their elementwise
        # product is the storage primitive (Finding 15: storage uses
        # bind, synthesis at recall uses bundle).
        p_hdc = await encode_async(encoder, prompt_text)
        a_hdc = await encode_async(encoder, answer_text)
        if p_hdc.shape != a_hdc.shape:
            raise ValueError(
                f"encoder produced mismatched shapes: prompt {p_hdc.shape} "
                f"vs answer {a_hdc.shape}. encoder bug."
            )
        grating = p_hdc * a_hdc  # (d_hdc,) NOT unit-norm in general
        d_hdc = grating.shape[0]

        # Native embedding for the codebook. encode_native returns the
        # raw 768-dim sentence-transformer output (unit-norm) — same
        # input the bind-v1 decoder expects at recall time.
        #
        # This same vector is what FactVectorStore ranks on (Fact.vector),
        # so it is also the fact's SEARCH key. embed_text (default None)
        # lets a caller index the fact by text OTHER than the stored
        # answer — e.g. a natural-language description of a code symbol —
        # so a conceptual query matches the description while claim_text
        # still returns the verbatim body. None preserves the historical
        # behavior exactly: index == stored answer.
        _embed_source = embed_text if embed_text is not None else answer_text
        a_native = await encode_native_async(encoder, _embed_source)

        async with self.session() as session:
            crystal_row = await session.get(CrystalRow, crystal_id)
            if crystal_row is None:
                raise ValueError(f"crystal {crystal_id!r} not found")

            # Phase 3 (April 2026): resolve per-type capacity ceiling.
            # The crystal_type registry row carries `capacity_default`
            # which overrides the global CRYSTAL_CAPACITY_HARD_CEILING.
            # If the registry row is missing (orphan crystal_type id
            # — shouldn't happen since add_pair_for_customer validates,
            # but a direct upsert_crystal could create one), fall back
            # to the global default. This keeps the write working
            # rather than refusing on a registry miss.
            #
            # The lookup is one PK fetch — cheap. We do it inside the
            # session so we read a consistent snapshot with the
            # crystal row.
            type_row = None
            if crystal_row.crystal_type:
                type_row = await session.get(
                    CrystalTypeRow, crystal_row.crystal_type
                )
            if type_row is not None:
                hard_ceiling = int(type_row.capacity_default)
            else:
                hard_ceiling = CRYSTAL_CAPACITY_HARD_CEILING

            # Phase 4.5 (April 2026): pair_type validation against the
            # crystal type's compiled schema, if a SchemaLoader was
            # supplied. Validation is opt-in; callers that don't pass
            # a loader (existing tests, the legacy import path) skip
            # this entirely. When provided, the loader looks up the
            # crystal's crystal_type; if a compiled schema exists,
            # `pair_type` must be in `compiled.pair_types`. We do this
            # BEFORE the auto-split branch so a bad pair_type fails
            # without spawning a sibling crystal whose only purpose
            # was to absorb a write that wouldn't validate. Validation
            # is silently skipped when the type isn't registered in
            # the loader — unregistered types are tolerated for
            # back-compat, matching the type_row fallback behavior
            # above.
            if schema_loader is not None and crystal_row.crystal_type:
                compiled = await schema_loader.get(
                    crystal_row.crystal_type
                )
                if compiled is not None:
                    if pair_type not in compiled.pair_types:
                        raise PairTypeValidationError(
                            crystal_id=crystal_row.id,
                            crystal_type=crystal_row.crystal_type,
                            attempted_pair_type=pair_type,
                            valid_pair_types=sorted(
                                compiled.pair_types.keys()
                            ),
                        )

            # Phase 1.2: capacity hard-enforcement via auto-split.
            #
            # If this write would push fact_count past the hard ceiling,
            # we DO NOT refuse — refusing produces terrible UX (writes
            # silently fail at a number the customer didn't pick). Instead
            # we spawn a sibling crystal that inherits the parent's
            # identity (customer_id, source_kind, summary_text,
            # encoder_fingerprint) and redirect this write to it.
            #
            # The sibling's parent_crystal_id points back at the original,
            # preserving lineage for the inspector and for future
            # cross-sibling recall (Phase 3 chains).
            #
            # The returned Fact's crystal_id is the SIBLING's id, not the
            # requested crystal_id. Callers that care about redirect
            # detection can compare fact.crystal_id against the requested
            # id; callers that don't care just see the Fact persist
            # successfully like any other write.
            #
            # The sibling itself starts with the encoder fingerprint
            # already stamped (inherited from the parent at spawn). The
            # rest of this function then runs against the sibling row
            # exactly as it would against any other crystal — fingerprint
            # verify passes trivially, the grating becomes the sibling's
            # first summary_vector, fact_count goes 0 → 1.
            current_count = crystal_row.fact_count or 0
            if current_count + 1 > hard_ceiling:
                # Phase 3 (April 2026): consult autosplit_policy.
                # 'split' (default in seeded legacy rows) → spawn a
                #   sibling and redirect the write (existing Phase 1.2
                #   behavior).
                # 'refuse' → raise CrystalCapacityError. The caller
                #   (operator UI, smart router, etc.) decides what to
                #   do next: route to a different crystal, raise the
                #   capacity, or surface the error to the user.
                #
                # If the registry row was missing (orphan crystal_type;
                # type_row is None), fall back to 'split' as the safe
                # default — refusing on a missing registry would
                # surprise existing callers more than implicitly
                # auto-splitting does.
                policy = (
                    type_row.autosplit_policy
                    if type_row is not None
                    else "split"
                )
                if policy == "refuse":
                    raise CrystalCapacityError(
                        crystal_id=crystal_row.id,
                        crystal_type=crystal_row.crystal_type or "",
                        hard_ceiling=hard_ceiling,
                        fact_count=current_count,
                    )
                # policy == 'split' (or any unknown future policy that
                # falls through here — we treat unknown as split for
                # safety; raising on unknown would brick legacy banks
                # whose registry rows were updated by a future Alembic
                # version this code doesn't yet know about).
                sibling_id = f"crys_{uuid.uuid4().hex[:16]}"
                now_split = datetime.now(timezone.utc)
                sibling_row = CrystalRow(
                    id=sibling_id,
                    customer_id=crystal_row.customer_id,
                    summary_vector=[],  # populated below by the
                                        # accumulation branch (empty
                                        # current → grating becomes
                                        # the starting bundle).
                    answer_embedding_native=None,
                    encoder_fingerprint=crystal_row.encoder_fingerprint,
                    source_kind=crystal_row.source_kind,
                    answer_value=None,
                    decay_rate=crystal_row.decay_rate,
                    fact_count=0,
                    quality_tier=crystal_row.quality_tier,
                    eval_helped_count=0,
                    eval_hurt_count=0,
                    live_shadow_helped_count=0,
                    live_shadow_hurt_count=0,
                    keyword_fingerprint=list(crystal_row.keyword_fingerprint or []),
                    cluster_tightness=None,
                    attribution_spread=None,
                    summary_text=crystal_row.summary_text,
                    build_method=crystal_row.build_method,
                    parent_crystal_id=crystal_row.id,
                    # Phase 3 (April 2026): inherit crystal_type from parent.
                    # Without this the sibling lands in 'customer:legacy'
                    # (the column default) instead of the parent's type,
                    # silently breaking type-scoped routing across
                    # auto-split lineage. Per spec §1.2: "spawn a sibling
                    # crystal (same crystal_type, same ACL, ...)".
                    crystal_type=crystal_row.crystal_type,
                    # Foundation F2: inherit POSIX ownership from the parent.
                    # Without this, auto-splitting a private crystal (mode
                    # 0o600) would spawn a team-readable (0o640), unowned
                    # sibling — a privacy leak on split.
                    owner_operator_id=crystal_row.owner_operator_id,
                    group_team_id=crystal_row.group_team_id,
                    mode=crystal_row.mode,
                    diagnostic_tags=list(crystal_row.diagnostic_tags or []),
                    last_eval_at=None,
                    last_activity=now_split,
                    created_at=now_split,
                )
                session.add(sibling_row)

                # Phase 3 audit fix #5 (April 2026): copy parent's ACL
                # rows verbatim to the sibling. Spec §1.2 says auto-split
                # spawns "(same crystal_type, same ACL, linked via
                # parent_crystal_id)". The implicit owner-customer ACL
                # is already covered by inheriting customer_id; this
                # branch handles the EXPLICIT crystal_acls rows the
                # parent may carry (e.g. read_codebook grants to other
                # customers, world-read on general-tier crystals).
                #
                # Without this copy, a parent's explicit grants silently
                # don't carry over to siblings: a customer who had
                # access to the parent loses access on the sibling.
                # Auto-splits would partition access in surprising ways.
                #
                # Inline SELECT (rather than calling list_acls_for_crystal)
                # so the read shares this session's transaction with the
                # sibling-row INSERT. Either both land or neither does.
                # The Phase 3 ACL CRUD methods (list_acls_for_crystal,
                # add_acl) each open their own session, which would
                # produce a partial state on rollback.
                acl_select = (
                    select(CrystalAclRow)
                    .where(CrystalAclRow.crystal_id == crystal_row.id)
                )
                parent_acls = (
                    await session.execute(acl_select)
                ).scalars().all()
                for parent_acl in parent_acls:
                    session.add(CrystalAclRow(
                        crystal_id=sibling_id,
                        principal_type=parent_acl.principal_type,
                        principal_id=parent_acl.principal_id,
                        grant=parent_acl.grant,
                        # Sibling's grant timestamp is the spawn moment,
                        # not the parent's original grant time. The
                        # parent's row keeps its own granted_at; the
                        # sibling's grant is fresh as of this auto-split.
                        granted_at=now_split,
                    ))

                logger.info(
                    "crystal_auto_split",
                    parent_crystal_id=crystal_row.id,
                    sibling_crystal_id=sibling_id,
                    customer_id=crystal_row.customer_id,
                    parent_fact_count=current_count,
                    hard_ceiling=hard_ceiling,
                    crystal_type=crystal_row.crystal_type,
                    note=(
                        "Hard capacity ceiling reached on parent. Spawned "
                        "sibling crystal to absorb this and subsequent "
                        "writes targeted at the parent. Returned "
                        "Fact.crystal_id is the sibling's id; callers "
                        "comparing against the requested crystal_id can "
                        "detect the redirect."
                    ),
                )
                # From this point forward, all operations target the
                # sibling. crystal_id local is rebound so the rest of
                # this method (fact_row.crystal_id, the warning, the
                # returned Fact) all reference the sibling consistently.
                crystal_row = sibling_row
                crystal_id = sibling_id

            # Fingerprint stamp/verify.
            #
            # On first bind-storage write: stamp this encoder's
            # fingerprint. On subsequent writes: verify it matches.
            # Mismatch means a different encoder is trying to write
            # into a crystal whose existing summary_vector was built
            # from a different geometry — the bundle would be
            # incoherent and recall would silently degrade.
            existing_fp = crystal_row.encoder_fingerprint
            if existing_fp is None:
                crystal_row.encoder_fingerprint = encoder_fp
            elif existing_fp != encoder_fp:
                raise ValueError(
                    f"encoder fingerprint mismatch on crystal "
                    f"{crystal_id!r}: previously written with "
                    f"{existing_fp!r}, now being written with "
                    f"{encoder_fp!r}. Bind-storage requires the same "
                    f"encoder geometry across all writes to a single "
                    f"crystal; mixing geometries silently corrupts the "
                    f"recovered-vector distribution at recall time. "
                    f"Either use the original encoder, or rebuild the "
                    f"crystal from scratch with the new one."
                )

            # Accumulate grating into summary_vector. RAW addition, no
            # renormalization. Two branches:
            #   - Existing summary_vector with matching dim: add directly.
            #   - Empty summary_vector OR dim mismatch (e.g. a pre-Phase-1
            #     crystal whose summary_vector was a single-encode of
            #     summary_text, possibly different shape): treat the
            #     grating as the starting point. Per the clean-break
            #     decision in PHASE_1_NOTES.md — no migration of
            #     pre-Phase-1 crystals.
            current = crystal_row.summary_vector
            if current and len(current) == d_hdc:
                summary_vec = np.asarray(current, dtype=np.float32) + grating
            else:
                summary_vec = grating.astype(np.float32, copy=True)

            # Store as raw float32 list. Unit-norm enforcement happens
            # at the read boundary (VectorStore loads rows and
            # normalizes the matrix before cosine search). See the
            # module-level note above on the renormalize-at-read,
            # not-at-write design.
            crystal_row.summary_vector = summary_vec.astype(np.float32).tolist()

            # Phase 6.3 (May 2026): accumulate the prompt-only
            # superposition into routing_vector alongside the bind-
            # bundle into summary_vector. See Finding 16 + the
            # routing_vector field doc on Crystal for why this is
            # geometrically necessary — cosine routing on the
            # bind-bundle returns near-zero scores by construction;
            # cosine routing on `Σ P_i` returns the right ~0.4-0.6
            # range for related text.
            #
            # Same RAW-accumulation contract as summary_vector: no
            # per-write normalization, unit-norm enforcement is
            # read-side in VectorStore._ensure_loaded. Hard Rule 16
            # extends to routing_vector identically.
            #
            # Branch logic mirrors summary_vector: existing dim-matching
            # routing_vector accumulates additively; missing or
            # dim-mismatched starts fresh from p_hdc. The crystal_row's
            # routing_vector may be None on the very first bind-storage
            # write (legacy crystals, freshly-spawned siblings) — None
            # is treated as empty for accumulation purposes.
            current_routing = crystal_row.routing_vector
            if current_routing and len(current_routing) == d_hdc:
                routing_vec = (
                    np.asarray(current_routing, dtype=np.float32) + p_hdc
                )
            else:
                routing_vec = p_hdc.astype(np.float32, copy=True)
            crystal_row.routing_vector = routing_vec.astype(
                np.float32
            ).tolist()

            # Persist the codebook Fact with the answer's native embedding.
            # Phase 2 (migration 0011): also persist source_kind, answer_value,
            # prompt_text. The Fact-level fields drive the cache-hit short-
            # circuit and per-pair recoverability calibration that the
            # Crystal-level fields (added in 0006) covered before; the
            # Crystal-level versions stay one release as a back-compat
            # fallback for the legacy injection path (dropped in Phase 7.4).
            fact_id = f"fact_{uuid.uuid4().hex[:16]}"
            now = datetime.now(timezone.utc)
            fact_row = FactRow(
                id=fact_id,
                crystal_id=crystal_id,
                claim_text=answer_text,
                pair_type=pair_type,
                source_kind=source_kind,
                answer_value=answer_value,
                prompt_text=prompt_text,
                vector=a_native.astype(np.float32).tolist(),
                created_at=now,
            )
            session.add(fact_row)

            # Increment fact_count. Capacity enforcement (Phase 1.2)
            # consumes this counter; Phase 1.1 just logs an observatory
            # warning at the soft ceiling — EXACTLY ONCE per crystal,
            # at the write that crosses the threshold. Firing every
            # write past 50 would flood inspector logs.
            new_count = (crystal_row.fact_count or 0) + 1
            crystal_row.fact_count = new_count
            crystal_row.last_activity = now

            if new_count == CRYSTAL_CAPACITY_SOFT_CEILING:
                # Soft-ceiling observability warning. Today this fires
                # at the global SOFT constant (50), independent of the
                # per-type hard ceiling. Rationale: the soft ceiling
                # is research-derived from bind-storage degradation
                # past ~50 pairs in 768/10k geometry. A type with a
                # higher hard ceiling (e.g. customer:medical_records
                # raised to 200) still benefits from observability at
                # the 50-pair degradation point — the warning is
                # "recall accuracy starts to degrade," not "this
                # crystal is full." Per-type soft ceilings can land
                # in a future phase if we measure different capacity
                # ceilings empirically per type.
                logger.warning(
                    "crystal_capacity_soft_ceiling_reached",
                    crystal_id=crystal_id,
                    fact_count=new_count,
                    note=(
                        f"Crystal reached the {CRYSTAL_CAPACITY_SOFT_CEILING}-pair "
                        f"soft ceiling. Recall accuracy degrades empirically "
                        f"past this point per CLAUDE.md research. Phase 1.2 "
                        f"will enforce; Phase 1.1 observes only. This warning "
                        f"fires once per crystal, at the threshold-crossing "
                        f"write."
                    ),
                )

            return Fact(
                id=fact_id,
                crystal_id=crystal_id,
                claim_text=answer_text,
                pair_type=pair_type,
                source_kind=source_kind,  # type: ignore[arg-type]
                answer_value=answer_value,
                prompt_text=prompt_text,
                vector=a_native.astype(np.float32).tolist(),
                created_at=now,
            )

    # -----------------------------------------------------------------
    # Phase 1.3 (April 2026) — write-side routing
    # -----------------------------------------------------------------
    #
    # `add_pair_for_customer` is the smart write API. Where
    # `add_pair_to_crystal` requires the caller to supply a target
    # crystal_id, this method routes by content: "add this pair
    # somewhere appropriate in this customer's bank."
    #
    # This is the fix for Phase 1.2's "per-write spawns N siblings"
    # footgun. When a caller writes pairs P1, P2, P3 in a loop,
    # `add_pair_to_crystal(parent_id, ...)` against a full parent
    # spawns three separate siblings (no caller-side memory of the
    # previous redirect). `add_pair_for_customer` instead routes
    # each pair through VectorStore.search, so semantically related
    # pairs land in the same crystal naturally.
    #
    # Decision logic per write:
    #   1. Encode prompt to HDC.
    #   2. VectorStore.search(customer_id, query_hdc, k=1) → top-1.
    #   3. If empty bank: spawn fresh crystal, bind into it.
    #   4. If top-1 cosine >= bond_threshold AND top-1.fact_count <
    #      hard ceiling: bind into top-1.
    #   5. If top-1 cosine >= bond_threshold AND top-1 is at hard
    #      ceiling: spawn FRESH crystal (no parent_crystal_id link).
    #      This is option (α) per the locked Phase 1.3 plan: a full
    #      top-1 is treated as not-bondable, falling through to
    #      spawn fresh. Auto-split's parent-linked sibling is for
    #      callers who supply an explicit crystal_id (Phase 1.2);
    #      this router takes a different path because the routing
    #      decision was about content-similarity, not about
    #      "continue this crystal's lineage."
    #   6. If top-1 cosine < bond_threshold: spawn fresh.
    #
    # Deferrals to Phase 3 (crystal_type registry):
    #   - Per-type bond_threshold. Today there's one global setting.
    #   - crystal_type filter (route only within crystals of matching
    #     type). Today the router considers all of customer's
    #     crystals.
    #
    # Encoder fingerprint: same contract as add_pair_to_crystal.
    # When bonding to top-1, the existing fingerprint is verified;
    # when spawning fresh, the encoder's fingerprint is stamped on
    # the new crystal's first write.

    async def add_pair_for_customer(
        self,
        customer_id: str,
        prompt_text: str,
        answer_text: str,
        pair_type: str = "question_answer",
        *,
        encoder: BindCapableEncoder,
        vector_store: "VectorStore",
        vector_index: Optional["VectorIndex"] = None,
        crystal_type: str = "customer:legacy",
        bond_threshold: Optional[float] = None,
        source_kind: str = "model_reasoning",
        answer_value: Optional[str] = None,
        owner_operator_id: Optional[str] = None,
        group_team_id: Optional[str] = None,
        mode: int = 0o640,
        recall_gated: bool = False,
        origin: str = "direct",
        schema_loader: Optional["SchemaLoader"] = None,
        embed_text: Optional[str] = None,
        bonder: Optional["Bonder"] = None,
        decomposer: Optional["Decomposer"] = None,
    ) -> tuple[Crystal, Fact]:
        """Route (prompt, answer) into the customer's bank by content.

        Phase 1.3 implementation, Phase 3-extended for type scoping.
        The recommended write path for bulk pair-writes — callers
        don't have to know crystal ids, the router decides which
        existing-or-new crystal absorbs the pair based on
        prompt-vector similarity within the requested type.

        Args:
            customer_id: tenant scope. Routing is restricted to this
                customer's crystals.
            prompt_text: the question / key text. Used both as the
                routing query (against existing crystals) and as the
                grating's prompt component on bond/spawn.
            answer_text: the answer / value text. Encoded for the
                grating and stored in the codebook.
            pair_type: DSL-declared pair type. Default
                "question_answer" matches `add_pair_to_crystal`.
            encoder: bind-capable encoder. Same contract as
                `add_pair_to_crystal`. Required keyword-only.
            vector_store: the in-memory VectorStore — used for the bond
                search + invalidation ONLY on the fallback path, i.e.
                when `vector_index` is not passed. Required for
                back-compat (tests + any caller not yet migrated).
            vector_index: optional VectorIndex seam (Step 2b-ii-b). When
                passed, the bond search AND the post-write invalidate go
                through it instead of `vector_store`: in qdrant mode the
                routing lookup hits Qdrant (no in-memory routing matrix
                loaded at write time) and the invalidate refreshes BOTH
                lanes; in memory mode it delegates to `vector_store`, so
                behavior is unchanged. None -> identical to before.
            crystal_type: which type of crystal to route into
                (Phase 3). Default 'customer:legacy' matches the
                migration 0012 seeded type for back-compat with
                callers that don't specify. Routing is filtered to
                crystals of this type — a customer's
                'customer:medical_records' write never bonds with
                their 'customer:billing_records' crystals.
            bond_threshold: cosine threshold for bonding to top-1.
                Resolution order: caller arg > registry's per-type
                `routing_threshold` > settings.bond_threshold global.
                Pass an explicit value for testing.
            schema_loader: optional Phase 4 SchemaLoader. When provided,
                threaded into add_pair_to_crystal for write-time pair_type
                validation against the type's compiled schema. None
                (the default) skips validation and writes whatever
                pair_type string the caller supplies. See
                add_pair_to_crystal docs for the full semantic.
            embed_text: optional text to index the fact by INSTEAD of
                answer_text (threaded to add_pair_to_crystal). When set,
                the fact's native search vector is encode_native(embed_text)
                while claim_text still stores answer_text — e.g. index a
                code chunk by a natural-language description while
                returning the verbatim body. None (default) indexes by
                answer_text, unchanged.

        Returns:
            tuple[Crystal, Fact]: the crystal the pair landed in
            (existing if bonded, freshly-spawned otherwise) and
            the persisted Fact row.

        Raises:
            ValueError: same conditions as add_pair_to_crystal
                (encoder shape, fingerprint mismatch). Also raises
                if `crystal_type` is not registered — catches typos
                at write time rather than producing rows pointing
                at orphan type ids.
        """
        # Phase 3: validate the requested type exists in the registry.
        # Lookup is one indexed PK fetch — cheap. We do it before any
        # encode work so a typo'd type id fails fast without burning
        # encoder time.
        type_row = await self.get_crystal_type(crystal_type)
        if type_row is None:
            raise ValueError(
                f"crystal_type {crystal_type!r} is not registered. "
                f"Seed it via upsert_crystal_type() before writing "
                f"crystals of this type. Migration 0012 seeds "
                f"'general:legacy' and 'customer:legacy' as the "
                f"catch-all defaults."
            )

        # Content chunks bypass the bonder entirely. Each chunk is a
        # standalone piece of verbatim document text that should NOT
        # be bonded to existing knowledge crystals. Always spawn a
        # fresh crystal with one fact.
        if pair_type == "content_chunk":
            new_crystal_id = f"crys_{uuid.uuid4().hex[:16]}"
            now = datetime.now(timezone.utc)
            new_crystal = Crystal(
                id=new_crystal_id,
                customer_id=customer_id,
                summary_vector=[],
                summary_text=None,
                build_method="content_chunk",
                crystal_type=crystal_type,
                source_kind="document_chunk",
                # Recall gate + birth attribution (2026-07-03).
                recall_gated=recall_gated,
                origin=origin,
                # Foundation F2: POSIX ownership (None/None/0o640 = today's
                # unowned, team-readable crystal when no operator authored it).
                owner_operator_id=owner_operator_id,
                group_team_id=group_team_id,
                mode=mode,
                created_at=now,
                last_activity=now,
            )
            await self.upsert_crystal(new_crystal)
            fact = await self.add_pair_to_crystal(
                crystal_id=new_crystal_id,
                prompt_text=prompt_text,
                answer_text=answer_text,
                pair_type=pair_type,
                encoder=encoder,
                answer_value=answer_value,
                schema_loader=schema_loader,
                embed_text=embed_text,
            )
            (vector_index or vector_store).invalidate(customer_id)
            crystal = await self.get_crystal(new_crystal_id)
            logger.info(
                "add_pair_for_customer.content_chunk_spawned",
                customer_id=customer_id,
                crystal_id=new_crystal_id,
                crystal_type=crystal_type,
                prompt_text=prompt_text[:60],
            )
            return crystal, fact

        # Resolve threshold: caller arg > per-type override > global.
        # The per-type override is the Phase 3 surface for tuning
        # routing tightness per use case (e.g. medical records may
        # want a higher bond_threshold so paraphrases of distinct
        # symptoms don't accidentally collapse).
        if bond_threshold is None:
            if type_row.routing_threshold is not None:
                bond_threshold = float(type_row.routing_threshold)
            else:
                from ..config import get_settings
                bond_threshold = float(get_settings().bond_threshold)

        # Resolve capacity ceiling per type. Falls back to the global
        # CRYSTAL_CAPACITY_HARD_CEILING if the registry row left
        # capacity_default at the default 50.
        capacity_ceiling = int(type_row.capacity_default)

        # Encode the prompt for the routing lookup. We pay this
        # encode cost once and reuse the result inside
        # add_pair_to_crystal indirectly (it re-encodes; small
        # waste, but keeps the routing logic decoupled from the
        # write logic).
        query_hdc = await encode_async(encoder, prompt_text)

        # Step 1: search for a bond candidate, filtered to this type.
        # Phase 3: VectorStore.search accepts crystal_type so the
        # routing only considers candidates of the matching type.
        # crystal_type=None falls back to all-types search (legacy);
        # crystal_type=str filters to the given type.
        # Through the seam when a vector_index is passed (Step 2b-ii-b),
        # so qdrant mode hits Qdrant routing with no in-memory matrix;
        # else the in-memory VectorStore. Both return the same
        # magnitude*cosine score the bond_threshold is calibrated to.
        if vector_index is not None:
            candidates = await vector_index.search_routing(
                customer_id=customer_id,
                query_vector=query_hdc,
                k=1,
                crystal_type=crystal_type,
            )
        else:
            candidates = await vector_store.search(
                customer_id=customer_id,
                query_vector=query_hdc,
                k=1,
                crystal_type=crystal_type,
            )

        # Phase 6.3 follow-up #2 (May 2026): bonder dispatch.
        # When the caller passes no bonder, we construct a
        # _CosineOnlyBonder that exactly preserves the pre-followup-2
        # behavior (bond iff top1_score >= bond_threshold). When a
        # bonder is passed, it owns the full bond-vs-spawn decision
        # using its own thresholds and axes. Either way, the capacity
        # check below the bonder still applies — the bonder decides
        # whether to BOND, not whether the candidate has room.
        from crystal_cache.learning.bonder import (
            BondDecision, _CosineOnlyBonder,
        )

        effective_bonder: "Bonder"
        if bonder is not None:
            effective_bonder = bonder
        else:
            effective_bonder = _CosineOnlyBonder(threshold=bond_threshold)

        # Decompose the incoming prompt UPFRONT iff a decomposer is
        # wired. We do it once per add_pair_for_customer call rather
        # than letting the bonder lazily decompose because we may
        # also need the payload on spawn-fresh (to populate the new
        # crystal's decomposer_payload field per scope doc Decision
        # 4 / migration 0016). One LLM call serves both consumers.
        #
        # Cost note: this lands ~50-200ms on every add_pair_for_customer
        # call when the decomposer is wired. The 36-pair FAQ import
        # becomes ~7s of wall-clock on Groq Llama 3.1 8B. Bulk imports
        # at 1000+ pairs would feel this; if/when that becomes a real
        # workload, the optimization is gray-zone-only invocation per
        # the scope doc's Risk 5 mitigation. Out of scope for #2.
        #
        # DecomposerError (and any other unexpected exception) is
        # caught here so a transient LLM failure doesn't block the
        # write. The bonder receives incoming_payload=None and the
        # spawn path stores decomposer_payload=None. Same conservative
        # behavior as if no decomposer were wired.
        incoming_payload: Optional[dict[str, Any]] = None
        if decomposer is not None:
            try:
                result = await decomposer.decompose(prompt_text)
                incoming_payload = result.payload or None
            except Exception as e:
                logger.warning(
                    "add_pair_for_customer.decomposer_failed",
                    customer_id=customer_id,
                    error=str(e),
                    error_type=type(e).__name__,
                    note=(
                        "Decomposer call failed; proceeding with "
                        "axis-3 disabled for this write. The bonder "
                        "falls back to conservative-spawn in the "
                        "gray zone, and any spawn-fresh crystal lands "
                        "with decomposer_payload=NULL. Not a failure "
                        "of the write itself — graceful degradation."
                    ),
                )

        target_crystal_id: Optional[str] = None
        bond_decision: Optional[BondDecision] = None
        if candidates:
            top1_id, top1_score = candidates[0]
            top1_crystal = await self.get_crystal(top1_id)
            # KEYSTONE (P2, ratified 2026-07-02): scope is a merge
            # boundary. If the routing top-1 has a different scope
            # identity than the incoming pair's stamps, it is NOT a join
            # candidate — treat as empty bank and spawn fresh, so
            # personal and team facts can never share a crystal (and two
            # operators' personal facts can never share one either).
            if top1_crystal is not None:
                from .permissions import may_join

                if not may_join(
                    top1_crystal,
                    owner_operator_id=owner_operator_id,
                    group_team_id=group_team_id or customer_id,
                    mode=mode,
                ):
                    logger.info(
                        "add_pair_for_customer.scope_boundary_spawn",
                        customer_id=customer_id,
                        top1_crystal_id=top1_id,
                        top1_mode=top1_crystal.mode,
                        incoming_mode=mode,
                    )
                    top1_crystal = None
            top1_facts: list[Fact] = []
            top1_payload: Optional[dict[str, Any]] = None
            if top1_crystal is not None:
                top1_payload = top1_crystal.decomposer_payload
                # The bonder only needs facts when it might fire axis 2.
                # The bonder Protocol takes the list anyway — let the
                # bonder decide whether to look at it. Reading them is
                # one indexed query.
                top1_facts = await self.list_facts_for_crystal(top1_id)

            bond_decision = await effective_bonder.should_bond(
                candidate_crystal=top1_crystal,
                candidate_score=top1_score,
                candidate_facts=top1_facts,
                candidate_payload=top1_payload,
                incoming_prompt=prompt_text,
                incoming_payload=incoming_payload,
                encoder=encoder,
            )

            logger.info(
                "add_pair_for_customer.bonder_decision",
                customer_id=customer_id,
                crystal_type=crystal_type,
                top1_crystal_id=top1_id,
                bond=bond_decision.bond,
                reason=bond_decision.reason,
                candidate_score=bond_decision.candidate_score,
                best_fact_cosine=bond_decision.best_fact_cosine,
                payload_agreement=bond_decision.payload_agreement,
                decomposer_called=bond_decision.decomposer_called,
            )

            if bond_decision.bond:
                # Bonder approved. Now check capacity — the bonder
                # decides similarity, the metadata store decides
                # whether the chosen crystal has room. If full, fall
                # through to spawn-fresh per the locked Phase 1.3 plan
                # (option α).
                if (
                    top1_crystal is not None
                    and (top1_crystal.fact_count or 0) <
                    capacity_ceiling
                ):
                    target_crystal_id = top1_id
                else:
                    # Top-1 bonder-approved but at capacity. Spawn
                    # fresh; log distinctly from auto-split.
                    logger.info(
                        "add_pair_for_customer.bond_target_full",
                        customer_id=customer_id,
                        crystal_type=crystal_type,
                        top1_crystal_id=top1_id,
                        top1_score=top1_score,
                        bond_threshold=bond_threshold,
                        bonder_reason=bond_decision.reason,
                        capacity_ceiling=capacity_ceiling,
                        note=(
                            "Bonder approved bond to top-1 but the "
                            "candidate is at hard capacity ceiling; "
                            "spawning fresh crystal per Phase 1.3 "
                            "option α (no parent linkage). Phase 1.2 "
                            "auto-split is for direct "
                            "add_pair_to_crystal calls; the router "
                            "path takes a different decision because "
                            "semantic similarity to a full crystal "
                            "shouldn't force lineage."
                        ),
                    )

        # Step 2: spawn or bond.
        if target_crystal_id is None:
            # Spawn fresh. The new crystal's encoder_fingerprint stays
            # None at upsert time; add_pair_to_crystal stamps it on
            # the first write below. crystal_type is stamped here so
            # the next routing call sees the new crystal in its
            # type-scoped bank.
            #
            # Phase 6.3 follow-up #2 (May 2026): if a decomposer was
            # wired AND it returned a payload for this prompt, the
            # payload is stamped on the new crystal as its "concept
            # identity" (option α from scope doc Decision 4: first-
            # bonded fact's payload, written ONCE on spawn). Future
            # pairs that route to this crystal in the gray zone will
            # be measured against this stored payload via concept-HV
            # cosine. None is legitimate — the bonder treats absent
            # payloads as a conservative-spawn signal in the gray
            # zone, never as a bond approval.
            new_crystal_id = f"crys_{uuid.uuid4().hex[:16]}"
            now = datetime.now(timezone.utc)
            new_crystal = Crystal(
                id=new_crystal_id,
                customer_id=customer_id,
                summary_vector=[],  # add_pair_to_crystal seeds it from
                                    # the first grating.
                summary_text=None,
                build_method="router",
                crystal_type=crystal_type,
                decomposer_payload=incoming_payload,
                # Recall gate + birth attribution (2026-07-03).
                recall_gated=recall_gated,
                origin=origin,
                # Foundation F2: POSIX ownership stamped on spawn-fresh. On
                # bond (target_crystal_id already set) the existing crystal
                # keeps its owner — writing into a crystal doesn't reassign it.
                owner_operator_id=owner_operator_id,
                group_team_id=group_team_id,
                mode=mode,
                created_at=now,
                last_activity=now,
            )
            await self.upsert_crystal(new_crystal)
            target_crystal_id = new_crystal_id

            # Phase 4.7 (April 2026): instantiate ACL defaults from the
            # type's compiled schema. Opt-in via the same schema_loader
            # parameter that drives Phase 4.5 pair_type validation.
            # When provided AND the type has a compiled schema in the
            # loader, the schema's `acl_defaults` get materialized as
            # crystal_acls rows on this freshly-spawned crystal.
            #
            # Why only on spawn-fresh: bond-existing reuses an
            # existing crystal whose ACLs were instantiated at its
            # original creation. Re-applying defaults on every bond
            # would either be a no-op (add_acl is idempotent) or
            # silently override grants the operator manually adjusted
            # via the inspector. Idempotent no-op is safe but wasteful
            # at scale; the cleaner contract is "defaults apply at
            # creation only, manual changes after that."
            #
            # Why not on auto-split: the auto-split branch in
            # add_pair_to_crystal explicitly copies the parent's ACL
            # rows to the sibling (Phase 3 audit fix #5). The sibling
            # continues the parent's access policy verbatim. Schema
            # defaults would be redundant at best (matching what the
            # parent already had) or surprising at worst (overriding
            # operator-customized parent grants).
            #
            # Why not on direct upsert_crystal: existing callers
            # (tests, offline tooling) don't pass a schema_loader and
            # don't expect their crystal-creation calls to side-effect
            # ACL rows. Hooking ACL instantiation there would surprise
            # them. The router path is the only path where the
            # operator's intent ("create a fresh crystal of type X")
            # cleanly maps to "and apply X's ACL defaults."
            if schema_loader is not None:
                compiled = await schema_loader.get(crystal_type)
                if compiled is not None and compiled.acl_defaults:
                    from ..dsl.schema.loader import resolve_acl_defaults
                    acl_rows = resolve_acl_defaults(
                        compiled.acl_defaults,
                        crystal_id=new_crystal_id,
                        owning_customer_id=customer_id,
                    )
                    for acl_row in acl_rows:
                        await self.add_acl(acl_row)
                    logger.debug(
                        "add_pair_for_customer.applied_acl_defaults",
                        customer_id=customer_id,
                        crystal_id=new_crystal_id,
                        crystal_type=crystal_type,
                        acl_count=len(acl_rows),
                    )

            logger.info(
                "add_pair_for_customer.spawned_fresh",
                customer_id=customer_id,
                crystal_type=crystal_type,
                new_crystal_id=new_crystal_id,
                top1_score=(
                    candidates[0][1] if candidates else None
                ),
                bond_threshold=bond_threshold,
            )
        else:
            logger.debug(
                "add_pair_for_customer.bonded",
                customer_id=customer_id,
                crystal_type=crystal_type,
                crystal_id=target_crystal_id,
                top1_score=candidates[0][1] if candidates else None,
                bond_threshold=bond_threshold,
            )

        # Step 3: write the pair via the existing primitive.
        # Phase 4.5: thread schema_loader through. The loader hits its
        # own cache after the first lookup, so the validation cost is
        # one cache lookup per write — negligible.
        fact = await self.add_pair_to_crystal(
            crystal_id=target_crystal_id,
            prompt_text=prompt_text,
            answer_text=answer_text,
            pair_type=pair_type,
            encoder=encoder,
            source_kind=source_kind,
            answer_value=answer_value,
            schema_loader=schema_loader,
            embed_text=embed_text,
        )

        # Step 4: invalidate the VectorStore cache so subsequent
        # routing calls see the new/updated crystal. This is the
        # critical line that makes the bonding pattern work for bulk
        # writes — without it, the second pair-write would still see
        # the pre-write bank and might fail to bond to the crystal
        # we just spawned/updated.
        (vector_index or vector_store).invalidate(customer_id)

        # Step 5: refetch the (possibly redirected) crystal so caller
        # sees current state. add_pair_to_crystal's auto-split could
        # have redirected fact.crystal_id to a sibling if we ended up
        # bonding to a parent that crossed the ceiling on this write
        # (shouldn't happen — we checked capacity above — but the
        # capacity check and the actual write are not atomic, and a
        # concurrent writer could push the parent over the ceiling
        # in between). Refetching by fact.crystal_id is the safe
        # read.
        result_crystal = await self.get_crystal(fact.crystal_id)
        if result_crystal is None:
            # Should be unreachable — add_pair_to_crystal returned a
            # Fact whose crystal_id resolves to nothing. Fail loudly.
            raise RuntimeError(
                f"add_pair_for_customer: fact.crystal_id "
                f"{fact.crystal_id!r} does not resolve to a crystal "
                f"after write. This is a database integrity issue."
            )

        return result_crystal, fact

    async def list_all_facts_for_customer(
        self,
        customer_id: str,
    ) -> list[Fact]:
        """All facts across all crystals for a customer.

        Used by FactVectorStore to build the per-customer fact index.
        Returns facts joined through crystals.customer_id.
        """
        async with self.session() as session:
            stmt = (
                select(FactRow)
                .join(CrystalRow, FactRow.crystal_id == CrystalRow.id)
                .where(CrystalRow.customer_id == customer_id)
            )
            result = await session.execute(stmt)
            return [_fact_from_row(r) for r in result.scalars().all()]

    async def list_all_facts_general(
        self,
        crystal_type: str,
    ) -> list[Fact]:
        """All facts in the GENERAL bank for one crystal type.

        General crystals are rows with customer_id NULL (the nullable
        column existed since the v1 design doc; this is the first
        consumer). Used by FactVectorStore to build per-type general
        banks that are loaded once and shared across every customer's
        searches — general knowledge is system-level, read-only to
        customers, written only by the operator's seed importer.
        """
        async with self.session() as session:
            stmt = (
                select(FactRow)
                .join(CrystalRow, FactRow.crystal_id == CrystalRow.id)
                .where(
                    CrystalRow.customer_id.is_(None),
                    CrystalRow.crystal_type == crystal_type,
                )
            )
            result = await session.execute(stmt)
            return [_fact_from_row(r) for r in result.scalars().all()]

    async def get_customer_general_types(self, customer_id: str) -> list[str]:
        """The general crystal types this customer is subscribed to.

        Single source of truth for BOTH merge points (FactVectorStore
        and list_facts_by_key_prefix). Empty list = opted out entirely.
        Returns [] for unknown customers — a missing tenant must never
        widen scope.
        """
        async with self.session() as session:
            stmt = select(CustomerRow.general_crystal_types).where(
                CustomerRow.id == customer_id
            )
            row = (await session.execute(stmt)).scalar_one_or_none()
        if row is None:
            return []
        return list(row or [])

    async def set_customer_general_types(
        self, customer_id: str, types: list[str]
    ) -> bool:
        """Replace a customer's general-bank subscriptions.

        Used by the seed importer (subscribe the operator's customer to
        a newly imported type) and admin tooling. Returns False for
        unknown customers.
        """
        async with self.session() as session:
            stmt = (
                update(CustomerRow)
                .where(CustomerRow.id == customer_id)
                .values(general_crystal_types=list(types))
            )
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount > 0

    async def ensure_customer_row(
        self,
        customer_id: str,
        *,
        api_key: str,
        provider: str = "anthropic",
        model_id: str = "local",
    ) -> bool:
        """Idempotently ensure a customers row with a FIXED id exists.

        Exists for the coding agent's local customer: runtime builds it
        in-memory only (SQLite's unenforced FKs let bank writes succeed
        without the row), but subscription lookups
        (get_customer_general_types) need a real row. Returns True if
        created, False if it already existed. Never overwrites.
        """
        async with self.session() as session:
            existing = await session.get(CustomerRow, customer_id)
            if existing is not None:
                return False
            from .credentials import hash_api_key
            from .token_crypto import encrypt_secret
            session.add(CustomerRow(
                id=customer_id,
                api_key_hash=hash_api_key(api_key),
                model_routing_config={
                    "provider": provider,
                    "model_id": model_id,
                    "api_key_ref": encrypt_secret("local"),
                },
            ))
            await session.commit()
            return True

    async def import_general_bank(
        self,
        *,
        crystal_type: str,
        entries: list[dict],
        encoder: Any,
        display_name: Optional[str] = None,
    ) -> dict[str, int]:
        """Replace the general bank for one crystal type with `entries`.

        Each entry: {"key": wide→specific sparse key, "claim": the
        pattern text}. Pattern-form by design — the BCB findings
        (docs in crystal-cache-v1: BCB_BENCHMARK_FINDINGS.md) showed
        imperative pattern rules lift model performance (+17.6pp,
        zero regressions) where raw code examples showed none, so the
        general bank stores patterns, not code.

        Layout: one general crystal (customer_id NULL) per top domain
        (second key segment, e.g. 'Python' in 'General|Python|…'),
        facts as question_answer pairs with prompt_text = the sparse
        key (searchable by key_scan AND by vector — the key's
        wide→specific words carry the semantics) and claim_text = the
        pattern. Vectors ride the encoder executor lane.

        REPLACE semantics: existing general crystals of this type are
        deleted first — re-importing a seed file is a sync, not an
        accumulation, same contract as document re-ingestion.
        """
        from ..encoding.executor import encode_async, encode_native_async

        # Make the bank DISCOVERABLE, not just retrievable: ensure a
        # crystal_types registry row (scope='general') exists for this
        # type. The retrieval merge reads type STRINGS off crystals,
        # but every surface that LISTS subscribable banks (the
        # inspector's General Knowledge panel, the admin API) reads
        # the registry — found live 2026-06-12: five seeded banks were
        # fully retrievable and fully invisible, because import never
        # registered them. Create-if-missing only: a reseed must never
        # overwrite an operator-customized display_name.
        if await self.get_crystal_type(crystal_type) is None:
            derived = display_name or " ".join(
                w.upper() if len(w) <= 4 else w.capitalize()
                for w in crystal_type.split(":", 1)[-1].split("_") if w
            )
            await self.upsert_crystal_type(CrystalType(
                id=crystal_type,
                display_name=derived or crystal_type,
                scope="general",
            ))
            logger.info(
                "general_bank.type_registered",
                crystal_type=crystal_type, display_name=derived,
            )

        # Replace: drop the old bank for this type.
        async with self.session() as session:
            old_ids = (await session.execute(
                select(CrystalRow.id).where(
                    CrystalRow.customer_id.is_(None),
                    CrystalRow.crystal_type == crystal_type,
                )
            )).scalars().all()
            if old_ids:
                await session.execute(
                    FactRow.__table__.delete().where(FactRow.crystal_id.in_(old_ids))
                )
                await session.execute(
                    CrystalRow.__table__.delete().where(CrystalRow.id.in_(old_ids))
                )
                await session.commit()

        # Group by domain (segment 2 of the key; fallback bucket for
        # malformed keys so one bad line never sinks the import).
        domains: dict[str, list[dict]] = {}
        for e in entries:
            parts = (e.get("key") or "").split("|")
            domain = parts[1].strip() if len(parts) >= 2 and parts[1].strip() else "Misc"
            domains.setdefault(domain, []).append(e)

        crystals_written = 0
        facts_written = 0
        async with self.session() as session:
            for domain, items in sorted(domains.items()):
                summary = f"General {domain} engineering patterns"
                summary_vec = await encode_async(self._require_encoder(encoder), summary)
                crystal = CrystalRow(
                    id=f"gcr_{uuid.uuid4().hex}",
                    customer_id=None,
                    crystal_type=crystal_type,
                    summary_text=summary,
                    summary_vector=[float(x) for x in summary_vec],
                )
                session.add(crystal)
                crystals_written += 1
                for e in items:
                    key = e["key"].strip()
                    claim = e["claim"].strip()
                    vec = await encode_native_async(self._require_encoder(encoder), key)
                    session.add(FactRow(
                        id=f"gft_{uuid.uuid4().hex}",
                        crystal_id=crystal.id,
                        pair_type="question_answer",
                        prompt_text=key,
                        claim_text=claim,
                        vector=[float(x) for x in vec],
                    ))
                    facts_written += 1
            await session.commit()
        logger.info(
            "general_bank.imported",
            crystal_type=crystal_type,
            crystals=crystals_written, facts=facts_written,
        )
        return {"crystals": crystals_written, "facts": facts_written}

    @staticmethod
    def _require_encoder(encoder: Any) -> Any:
        if encoder is None:
            raise ValueError(
                "import_general_bank requires an encoder — general facts "
                "must be vector-searchable from the moment they land."
            )
        return encoder

    async def list_crystal_types_by_scope(
        self, scope: Optional[str] = None
    ) -> list[CrystalType]:
        """Registry listing for discovery surfaces — the inspector's
        General Knowledge panel and the admin API. scope=None returns
        every registered type; scope='general' is what the subscription
        UI shows. Named distinctly (not list_crystal_types) so it can't
        silently shadow or be shadowed by any other registry reader in
        this 3k-line class."""
        async with self.session() as session:
            stmt = select(CrystalTypeRow).order_by(CrystalTypeRow.id)
            if scope:
                stmt = stmt.where(CrystalTypeRow.scope == scope)
            rows = (await session.execute(stmt)).scalars().all()
            return [_crystal_type_from_row(r) for r in rows]

    async def add_reflection_fact(
        self,
        customer_id: str,
        *,
        key: str,
        claim: str,
        encoder: Any,
    ) -> dict[str, str]:
        """Write one reflection fact into the CUSTOMER'S OWN bank.

        Phase C (the reflection loop): lessons the agent earned from a
        fail→pass verify cycle land here as ordinary facts — key under
        the `Reflections|` namespace, pair_type question_answer, vector
        from the encoder lane — in one accumulating crystal of type
        'reflection' per customer. Same fact shape as the seed banks,
        so retrieval, key_scan, and the general-bank merge all see them
        with zero consumer changes; being customer facts, they sit on
        the winning side of the 0.995 tie-break.

        These are PRIVATE. There is no code path from here to the
        general bank, by design — generalization is operator authorship
        through the seeds pipeline, never promotion.
        """
        from ..encoding.executor import encode_async, encode_native_async

        enc = self._require_encoder(encoder)
        async with self.session() as session:
            crystal_id = (await session.execute(
                select(CrystalRow.id).where(
                    CrystalRow.customer_id == customer_id,
                    CrystalRow.crystal_type == "reflection",
                ).limit(1)
            )).scalar_one_or_none()
            if crystal_id is None:
                summary = "Reflections — lessons this agent learned from its own failed-then-fixed runs"
                summary_vec = await encode_async(enc, summary)
                crystal = CrystalRow(
                    id=f"rcr_{uuid.uuid4().hex}",
                    customer_id=customer_id,
                    crystal_type="reflection",
                    summary_text=summary,
                    summary_vector=[float(x) for x in summary_vec],
                )
                session.add(crystal)
                crystal_id = crystal.id
            vec = await encode_native_async(enc, key)
            fact = FactRow(
                id=f"rft_{uuid.uuid4().hex}",
                crystal_id=crystal_id,
                pair_type="question_answer",
                prompt_text=key,
                claim_text=claim,
                vector=[float(x) for x in vec],
            )
            session.add(fact)
            await session.commit()
            logger.info("reflection.stored", customer_id=customer_id, key=key)
            return {"fact_id": fact.id, "crystal_id": crystal_id}

    async def delete_crystal(
        self,
        crystal_id: str,
        customer_id: Optional[str] = None,
        *,
        vector_store: Optional["VectorStore"] = None,
        fact_vector_store: Optional["FactVectorStore"] = None,
    ) -> bool:
        """Delete a crystal and ALL its facts. Returns True if deleted.

        CU-9. Whole-crystal deletion is the clean removal primitive:
        the HDC codebook (summary_vector) dies with the row, so no
        grating subtraction is needed — unlike per-fact removal from a
        shared crystal, which would require unbinding individual
        gratings from the accumulated codebook.

        Args:
            crystal_id: the crystal to delete.
            customer_id: when given, the delete is tenancy-scoped — a
                crystal belonging to a different customer is NOT
                deleted and False is returned.
            vector_store: optional; invalidated for the crystal's
                customer on success so routing can't bond into a
                deleted crystal.
            fact_vector_store: optional; invalidated likewise so
                deleted facts stop surfacing in fact search.
        """
        async with self.session() as session:
            row = await session.get(CrystalRow, crystal_id)
            if row is None:
                return False
            if customer_id is not None and row.customer_id != customer_id:
                logger.warning(
                    "metadata_store.delete_crystal.tenancy_mismatch",
                    crystal_id=crystal_id,
                    requested_by=customer_id,
                )
                return False
            owner = row.customer_id
            fact_result = await session.execute(
                select(FactRow).where(FactRow.crystal_id == crystal_id)
            )
            fact_rows = fact_result.scalars().all()
            for fr in fact_rows:
                await session.delete(fr)
            await session.delete(row)

        if owner:
            if vector_store is not None:
                vector_store.invalidate(owner)
            if fact_vector_store is not None:
                fact_vector_store.invalidate(owner)

        logger.info(
            "metadata_store.crystal_deleted",
            crystal_id=crystal_id,
            customer_id=owner,
            facts_deleted=len(fact_rows),
        )
        return True

    async def delete_fact(
        self,
        fact_id: str,
        customer_id: Optional[str] = None,
        *,
        encoder: Any,
        vector_store: Optional["VectorStore"] = None,
        fact_vector_store: Optional["FactVectorStore"] = None,
    ) -> bool:
        """Delete a single fact and recompute its crystal's vectors. True if deleted.

        Per-fact deletion is the costly-but-correct counterpart to
        delete_crystal. A crystal's summary_vector and routing_vector are a
        RAW ADDITIVE accumulation of per-fact contributions
        (add_pair_to_crystal: summary += encode(prompt)*encode(answer);
        routing += encode(prompt); Hard Rule 16 — no per-write
        normalization), so removing one fact's contribution means rebuilding
        those two vectors from the SURVIVING facts. We REBUILD from survivors
        rather than subtract the one grating because rebuild is
        self-correcting: it doesn't trust the accumulator's prior integrity
        (a historical dim-mismatch reset, fp drift) the way subtraction
        would. The fact row stores prompt_text + claim_text (the exact texts
        that were encoded; embed_text only ever affected the native search
        vector, never the grating), so re-encoding survivors replays the
        accumulation exactly.

        The fact's native search vector (Fact.vector) lives only on the row,
        so deleting the row + invalidating fact_vector_store is all the
        fact-level (768-dim) search needs; only the crystal-level HDC vectors
        are recomputed. When the last fact goes, the crystal is deleted whole
        (its vectors would be all-zero and it could never match).

        Args:
            fact_id: the fact to delete.
            customer_id: when given, tenancy-scoped — a fact whose crystal
                belongs to another customer (or a general crystal) is NOT
                deleted and False is returned.
            encoder: a BindCapableEncoder (encode + encode_native +
                fingerprint) used to re-encode survivors. Its fingerprint
                must match the crystal's stamped encoder_fingerprint —
                recomputing in a different geometry would corrupt the
                codebook (same contract as add_pair_to_crystal).
            vector_store / fact_vector_store: invalidated for the owner on
                success so stale crystal/fact vectors stop surfacing.

        Raises:
            ValueError: if the encoder lacks encode_native/fingerprint, or
                its fingerprint doesn't match the crystal's stamped one.
        """
        if not hasattr(encoder, "encode_native") or not hasattr(
            encoder, "fingerprint"
        ):
            raise ValueError(
                "delete_fact requires a BindCapableEncoder (encode_native + "
                "fingerprint) to recompute the crystal's vectors from the "
                "surviving facts. Pass a SemanticTextEncoder."
            )
        encoder_fp = encoder.fingerprint()
        crystal_id: Optional[str] = None
        owner: Optional[str] = None

        async with self.session() as session:
            fact_row = await session.get(FactRow, fact_id)
            if fact_row is None:
                return False
            crystal_id = fact_row.crystal_id
            crystal_row = await session.get(CrystalRow, crystal_id)

            if crystal_row is None:
                # Orphan fact (its crystal is already gone): just drop it.
                await session.delete(fact_row)
            else:
                if (
                    customer_id is not None
                    and crystal_row.customer_id != customer_id
                ):
                    logger.warning(
                        "metadata_store.delete_fact.tenancy_mismatch",
                        fact_id=fact_id,
                        crystal_id=crystal_id,
                        requested_by=customer_id,
                    )
                    return False
                owner = crystal_row.customer_id
                stamped_fp = crystal_row.encoder_fingerprint
                if stamped_fp and stamped_fp != encoder_fp:
                    raise ValueError(
                        f"encoder fingerprint mismatch on crystal "
                        f"{crystal_id!r}: crystal stamped {stamped_fp!r} but "
                        f"encoder is {encoder_fp!r}. Recomputing in a "
                        "different geometry would corrupt the codebook — use "
                        "the original encoder."
                    )

                # Drop the fact, then rebuild from whatever survives.
                await session.delete(fact_row)
                await session.flush()  # exclude it from the survivors query

                survivor_result = await session.execute(
                    select(FactRow)
                    .where(FactRow.crystal_id == crystal_id)
                    .order_by(FactRow.created_at.asc())
                )
                survivors = survivor_result.scalars().all()

                if not survivors:
                    # Last fact gone — delete the now-empty crystal whole.
                    await session.delete(crystal_row)
                else:
                    # Replay the additive accumulation over the survivors,
                    # in insertion order (created_at asc) to match how the
                    # vectors were originally built.
                    summary_vec: Optional[np.ndarray] = None
                    routing_vec: Optional[np.ndarray] = None
                    for fr in survivors:
                        p_hdc = await encode_async(encoder, fr.prompt_text or "")
                        a_hdc = await encode_async(encoder, fr.claim_text or "")
                        grating = p_hdc * a_hdc
                        if summary_vec is None:
                            summary_vec = grating.astype(np.float32, copy=True)
                            routing_vec = p_hdc.astype(np.float32, copy=True)
                        else:
                            summary_vec = summary_vec + grating
                            routing_vec = routing_vec + p_hdc
                    crystal_row.summary_vector = summary_vec.astype(
                        np.float32
                    ).tolist()
                    crystal_row.routing_vector = routing_vec.astype(
                        np.float32
                    ).tolist()
                    crystal_row.fact_count = len(survivors)

        if owner:
            if vector_store is not None:
                vector_store.invalidate(owner)
            if fact_vector_store is not None:
                fact_vector_store.invalidate(owner)

        logger.info(
            "metadata_store.fact_deleted",
            fact_id=fact_id,
            crystal_id=crystal_id,
            customer_id=owner,
        )
        return True

    async def list_facts_for_crystal(
        self,
        crystal_id: str,
        pair_type: Optional[str] = None,
    ) -> list[Fact]:
        """All Facts in a crystal's codebook, optionally filtered by pair_type.

        At recall time, cleanup will load the full unfiltered codebook
        (pair_type=None) and walk it for nearest-neighbor matching.
        The pair_type filter is for diagnostics and the inspector —
        e.g., "show me all date_progress_note Facts in this crystal."

        Ordered by created_at ascending so codebook iteration matches
        write order.
        """
        async with self.session() as session:
            stmt = select(FactRow).where(FactRow.crystal_id == crystal_id)
            if pair_type is not None:
                stmt = stmt.where(FactRow.pair_type == pair_type)
            stmt = stmt.order_by(FactRow.created_at)
            result = await session.execute(stmt)
            return [_fact_from_row(r) for r in result.scalars().all()]

    async def headline_facts_for_crystals(
        self, crystal_ids: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Representative fact per crystal for the inspector's bank list.

        Returns {crystal_id: {"key": prompt_text, "claim": claim_text,
        "source_kind": source_kind}} using each crystal's EARLIEST fact
        (created_at asc = write order). The sparse key (prompt_text) is
        what lets the bank render a human breadcrumb + title instead of a
        raw crystal id, and its leading segment classifies the crystal
        (Reflections|… / General|… / Code|…).

        One column-targeted query for the whole page — it selects only
        the three text columns, never the 768-dim `vector`, so it stays
        cheap even at 50 crystals × many facts each. Crystals with no
        facts are simply absent from the result (callers default).
        """
        if not crystal_ids:
            return {}
        out: dict[str, dict[str, Any]] = {}
        async with self.session() as session:
            stmt = (
                select(
                    FactRow.crystal_id,
                    FactRow.prompt_text,
                    FactRow.claim_text,
                    FactRow.source_kind,
                )
                .where(FactRow.crystal_id.in_(crystal_ids))
                .order_by(FactRow.created_at)
            )
            for cid, prompt_text, claim_text, source_kind in (
                await session.execute(stmt)
            ).all():
                if cid in out:
                    continue  # earliest fact wins (ordered asc)
                out[cid] = {
                    "key": prompt_text or "",
                    "claim": claim_text or "",
                    "source_kind": source_kind or "",
                }
        return out

    async def get_fact(self, fact_id: str) -> Optional[Fact]:
        async with self.session() as session:
            row = await session.get(FactRow, fact_id)
            return _fact_from_row(row) if row else None

    # -----------------------------------------------------------------
    # Phase 3 (April 2026): CrystalType registry CRUD
    # -----------------------------------------------------------------
    #
    # Migration 0012 seeds two registry rows ('general:legacy',
    # 'customer:legacy'). New types land via `upsert_crystal_type`;
    # an unknown type passed to `add_pair_for_customer` raises
    # ValueError at write time so a missing registry entry surfaces
    # immediately rather than silently producing rows with an
    # orphan type id.

    async def upsert_crystal_type(self, crystal_type: CrystalType) -> None:
        """Insert or update a registry entry.

        No FK constraint on Crystal.crystal_type today (per migration
        0012's rationale), so updating a type doesn't cascade to
        existing crystals. That's intentional — changes to capacity
        ceilings, thresholds, or autosplit policies should affect
        future writes only, not retroactively rewrite past behavior.
        """
        async with self.session() as session:
            existing = await session.get(CrystalTypeRow, crystal_type.id)
            data = {
                "display_name": crystal_type.display_name,
                "scope": crystal_type.scope,
                "capacity_default": crystal_type.capacity_default,
                "autosplit_policy": crystal_type.autosplit_policy,
                "routing_threshold": crystal_type.routing_threshold,
                "cleanup_threshold": crystal_type.cleanup_threshold,
                "pair_schema_dsl": crystal_type.pair_schema_dsl,
            }
            if existing is None:
                session.add(CrystalTypeRow(
                    id=crystal_type.id,
                    created_at=crystal_type.created_at,
                    **data,
                ))
            else:
                for key, value in data.items():
                    setattr(existing, key, value)

    async def get_crystal_type(
        self, type_id: str
    ) -> Optional[CrystalType]:
        async with self.session() as session:
            row = await session.get(CrystalTypeRow, type_id)
            return _crystal_type_from_row(row) if row else None

    async def list_crystal_types(self) -> list[CrystalType]:
        """List all registered crystal types.

        Cross-tenant by design: the registry is a global table
        (general:* types are world-shared; customer:* types are also
        in the same id pool, governed by ACLs not by table-level
        isolation).
        """
        async with self.session() as session:
            stmt = select(CrystalTypeRow).order_by(CrystalTypeRow.id)
            result = await session.execute(stmt)
            return [
                _crystal_type_from_row(r)
                for r in result.scalars().all()
            ]

    # -----------------------------------------------------------------
    # Phase 3: type-scoped crystal lookup
    # -----------------------------------------------------------------

    async def list_crystals_for_customer_and_type(
        self,
        customer_id: str,
        crystal_type: str,
        *,
        include_recall_gated: bool = True,
    ) -> list[Crystal]:
        """Filter a customer's crystals to one type.

        The Phase 3 routing path uses this when add_pair_for_customer
        narrows write-side bonding to crystals of the same type, and
        when the read-side router restricts a query to one type.
        Today VectorStore.search consumes this list to rebuild its
        cached matrix per (customer, type) pair.

        Indexed on (customer_id, crystal_type) via
        ix_crystals_customer_type from migration 0012.
        """
        async with self.session() as session:
            stmt = (
                select(CrystalRow)
                .where(CrystalRow.customer_id == customer_id)
                .where(CrystalRow.crystal_type == crystal_type)
            )
            if not include_recall_gated:
                stmt = stmt.where(CrystalRow.recall_gated.is_(False))
            result = await session.execute(stmt)
            return [_crystal_from_row(r) for r in result.scalars().all()]

    # -----------------------------------------------------------------
    # Phase 3: CrystalAcl CRUD
    # -----------------------------------------------------------------
    #
    # ACL rows are OPT-IN ADDITIVE. The presence of a row grants
    # access to its principal; the absence of rows means "use the
    # crystal's scope default" (customer-scope crystals are readable
    # by their owning customer; general-scope by world). There is
    # no deny-grant in Phase 3.

    # ------------------------------------------------------------------
    # Groups — P3, ratified 2026-07-02. Named sub-teams as grant targets:
    # a crystal_acls 'group' grant lets every member read without touching
    # the crystal's POSIX mode. Dict-row returns per the
    # list_thin_crystals_for_customer precedent.
    # ------------------------------------------------------------------

    async def create_group(self, customer_id: str, name: str) -> dict:
        """Create a named group. Names are unique per team (the CLI says
        'share with backend' unambiguously); a duplicate name raises."""
        group_id = f"grp_{uuid.uuid4().hex[:16]}"
        now = datetime.now(timezone.utc)
        async with self.session() as session:
            session.add(GroupRow(
                id=group_id, customer_id=customer_id,
                name=name, created_at=now,
            ))
        return {"id": group_id, "customer_id": customer_id, "name": name,
                "created_at": now.isoformat()}

    async def add_group_member(
        self, group_id: str, operator_id: str, customer_id: str,
    ) -> bool:
        """Add an operator to a group. Customer-guarded on BOTH sides (the
        group and the operator must belong to the team). Idempotent.
        Returns False when either side is unknown or foreign."""
        async with self.session() as session:
            group = await session.get(GroupRow, group_id)
            if group is None or group.customer_id != customer_id:
                return False
            operator = await session.get(OperatorRow, operator_id)
            if operator is None or operator.team_id != customer_id:
                return False
            existing = await session.get(
                GroupMemberRow, (group_id, operator_id),
            )
            if existing is None:
                session.add(GroupMemberRow(
                    group_id=group_id, operator_id=operator_id,
                ))
        return True

    async def remove_group_member(
        self, group_id: str, operator_id: str, customer_id: str,
    ) -> bool:
        async with self.session() as session:
            group = await session.get(GroupRow, group_id)
            if group is None or group.customer_id != customer_id:
                return False
            existing = await session.get(
                GroupMemberRow, (group_id, operator_id),
            )
            if existing is None:
                return False
            await session.delete(existing)
            return True

    async def list_groups_for_customer(self, customer_id: str) -> list[dict]:
        """Groups with their member ids, name-ordered."""
        async with self.session() as session:
            groups = (await session.execute(
                select(GroupRow)
                .where(GroupRow.customer_id == customer_id)
                .order_by(GroupRow.name)
            )).scalars().all()
            if not groups:
                return []
            members = (await session.execute(
                select(GroupMemberRow).where(
                    GroupMemberRow.group_id.in_([g.id for g in groups])
                )
            )).scalars().all()
            by_group: dict[str, list[str]] = {}
            for m in members:
                by_group.setdefault(m.group_id, []).append(m.operator_id)
            return [
                {"id": g.id, "name": g.name,
                 "created_at": g.created_at.isoformat() if g.created_at else None,
                 "member_ids": sorted(by_group.get(g.id, []))}
                for g in groups
            ]

    async def list_group_ids_for_operator(self, operator_id: str) -> frozenset:
        """The membership set can_read consumes for 'group' grants — one
        indexed query, fetched once per search by the retrieval filters."""
        async with self.session() as session:
            rows = (await session.execute(
                select(GroupMemberRow.group_id).where(
                    GroupMemberRow.operator_id == operator_id
                )
            )).scalars().all()
            return frozenset(rows)

    async def add_acl(self, acl: CrystalAcl) -> None:
        """Add a grant. Idempotent: re-adding the same composite key
        is a no-op (silently does nothing rather than raising).
        """
        async with self.session() as session:
            existing = await session.get(
                CrystalAclRow,
                (acl.crystal_id, acl.principal_type, acl.principal_id, acl.grant),
            )
            if existing is not None:
                # Idempotent re-grant. Don't bump granted_at — the
                # caller may want "first granted at" preserved.
                return
            session.add(CrystalAclRow(
                crystal_id=acl.crystal_id,
                principal_type=acl.principal_type,
                principal_id=acl.principal_id,
                grant=acl.grant,
                granted_at=acl.granted_at,
            ))

    async def remove_acl(
        self,
        *,
        crystal_id: str,
        principal_type: str,
        principal_id: str,
        grant: str,
    ) -> bool:
        """Revoke a specific grant. Returns True if a row was deleted,
        False if no matching row existed.
        """
        async with self.session() as session:
            existing = await session.get(
                CrystalAclRow,
                (crystal_id, principal_type, principal_id, grant),
            )
            if existing is None:
                return False
            await session.delete(existing)
            return True

    async def list_acls_for_crystal(
        self, crystal_id: str
    ) -> list[CrystalAcl]:
        async with self.session() as session:
            stmt = (
                select(CrystalAclRow)
                .where(CrystalAclRow.crystal_id == crystal_id)
                .order_by(CrystalAclRow.granted_at)
            )
            result = await session.execute(stmt)
            return [_acl_from_row(r) for r in result.scalars().all()]

    async def list_acls_for_principal(
        self,
        principal_type: str,
        principal_id: str,
    ) -> list[CrystalAcl]:
        """Reverse-direction lookup: "what crystals can this principal
        access?" Used by the ACL resolver during chain expansion.
        Indexed via ix_crystal_acls_principal.
        """
        async with self.session() as session:
            stmt = (
                select(CrystalAclRow)
                .where(CrystalAclRow.principal_type == principal_type)
                .where(CrystalAclRow.principal_id == principal_id)
            )
            result = await session.execute(stmt)
            return [_acl_from_row(r) for r in result.scalars().all()]

    # -----------------------------------------------------------------
    # Phase 3: CrystalChain CRUD
    # -----------------------------------------------------------------

    async def add_chain(self, chain: CrystalChain) -> None:
        """Add a chain edge. Self-loops rejected at write time as a
        defensive guard: a crystal already includes its own facts
        in cleanup, chaining to itself would just waste a DB lookup
        at recall time.

        Phase 3 audit fix #7 (April 2026): bidirectional chains are
        stored as TWO rows (one per direction), each with
        direction='source_uses_target'. The resolver only ever
        forward-walks; this lets us drop the reverse-walk logic that
        the original one-row-per-bidirectional model required.

        Idempotency:
          - For a `source_uses_target` edge, re-adding the same (src,
            tgt) is a no-op (PK conflict resolved by leaving existing
            row alone). If the existing row was previously bidirectional
            (i.e. (tgt, src) row also exists), the reverse row is
            REMOVED — changing direction from bidirectional to one-way
            should reflect the new authoring intent.
          - For a `bidirectional` edge, both (src, tgt) and (tgt, src)
            rows are upserted as `source_uses_target`. Calling this
            against a pre-existing one-way edge upgrades it to
            bidirectional by adding the missing reverse row.
        """
        if chain.source_crystal_id == chain.target_crystal_id:
            raise ValueError(
                "Self-loop chain rejected: source and target are "
                f"both {chain.source_crystal_id!r}. A crystal already "
                "covers its own facts in cleanup; chaining to itself "
                "adds no value."
            )

        src = chain.source_crystal_id
        tgt = chain.target_crystal_id
        now = chain.created_at

        async with self.session() as session:
            forward = await session.get(CrystalChainRow, (src, tgt))
            reverse = await session.get(CrystalChainRow, (tgt, src))

            if chain.direction == "bidirectional":
                # Two-row representation: both directions exist as
                # source_uses_target.
                if forward is None:
                    session.add(CrystalChainRow(
                        source_crystal_id=src,
                        target_crystal_id=tgt,
                        direction="source_uses_target",
                        created_at=now,
                    ))
                # else: leave forward row alone, including its
                # original created_at. The direction column on a
                # two-row representation is always source_uses_target;
                # if it's something else from a legacy single-row
                # bidirectional, normalize it.
                elif forward.direction != "source_uses_target":
                    forward.direction = "source_uses_target"

                if reverse is None:
                    session.add(CrystalChainRow(
                        source_crystal_id=tgt,
                        target_crystal_id=src,
                        direction="source_uses_target",
                        created_at=now,
                    ))
                elif reverse.direction != "source_uses_target":
                    reverse.direction = "source_uses_target"
            else:
                # Direction is source_uses_target (one-way).
                if forward is None:
                    session.add(CrystalChainRow(
                        source_crystal_id=src,
                        target_crystal_id=tgt,
                        direction="source_uses_target",
                        created_at=now,
                    ))
                else:
                    # Forward row exists; ensure it's tagged correctly.
                    if forward.direction != "source_uses_target":
                        forward.direction = "source_uses_target"
                # If a reverse row exists from a prior bidirectional
                # add, REMOVE it. The author has now declared this
                # edge as one-way; the reverse-side grant should be
                # revoked to match that declaration. Same-call
                # session ensures atomicity — either the forward
                # exists and the reverse is gone, or rollback leaves
                # both in their prior state.
                if reverse is not None:
                    await session.delete(reverse)

    async def remove_chain(
        self,
        *,
        source_crystal_id: str,
        target_crystal_id: str,
    ) -> bool:
        """Remove a chain edge.

        Phase 3 audit fix #7: under the two-row representation, a
        bidirectional edge is stored as two rows (src→tgt) and
        (tgt→src). `remove_chain` removes EITHER direction if it
        exists — callers shouldn't have to know whether the edge was
        authored as bidirectional or one-way to revoke it. Returns
        True if any row was deleted.

        If you want to remove only one direction of a bidirectional
        edge (turn it into a one-way edge in the other direction),
        call `add_chain` with the desired one-way direction instead;
        `add_chain`'s direction-change logic will revoke the
        unwanted reverse row.
        """
        async with self.session() as session:
            removed = False
            forward = await session.get(
                CrystalChainRow,
                (source_crystal_id, target_crystal_id),
            )
            if forward is not None:
                await session.delete(forward)
                removed = True
            reverse = await session.get(
                CrystalChainRow,
                (target_crystal_id, source_crystal_id),
            )
            if reverse is not None:
                await session.delete(reverse)
                removed = True
            return removed

    async def list_chains_from_source(
        self, source_crystal_id: str
    ) -> list[CrystalChain]:
        """Outgoing chains from a source crystal. Includes
        bidirectional rows (the source is on the source-side of the
        edge and pulls from target). Used by the chain resolver's
        forward walk.
        """
        async with self.session() as session:
            stmt = (
                select(CrystalChainRow)
                .where(CrystalChainRow.source_crystal_id == source_crystal_id)
            )
            result = await session.execute(stmt)
            return [_chain_from_row(r) for r in result.scalars().all()]

    async def list_chains_into_target(
        self, target_crystal_id: str
    ) -> list[CrystalChain]:
        """Incoming chains where the target is THIS crystal.

        Phase 3 audit fix #7 (April 2026): bidirectional chains are
        stored as TWO rows under the two-row representation (one
        per direction, each with `direction='source_uses_target'`),
        so the resolver never reverse-walks. It only ever asks
        `list_chains_from_source(my_id)` and forward-walks the
        results. See `chain_resolver.py` for the resolver contract.

        That makes this method NOT a resolver primitive. Its
        remaining uses are:

          - Inspector / admin queries: "what other crystals chain
            INTO this one?" Useful for surfacing dependencies
            before the operator deletes a crystal that's a chain
            target elsewhere.
          - Audit tooling: detecting orphan chains where the
            target was deleted out from under the source row.
          - Future Phase 3 follow-ups around chain governance
            (e.g. "revoke all chains pointing at a deprecated
            crystal") that need the reverse-direction view.

        Direction is NOT filtered — the caller sees every row whose
        target_crystal_id matches, including bidirectional partners
        and pure one-way `source_uses_target` rows. Callers that want
        only one shape filter the returned list themselves.

        Indexed via ix_crystal_chains_target from migration 0012.
        """
        async with self.session() as session:
            stmt = (
                select(CrystalChainRow)
                .where(CrystalChainRow.target_crystal_id == target_crystal_id)
            )
            result = await session.execute(stmt)
            return [_chain_from_row(r) for r in result.scalars().all()]

    # -----------------------------------------------------------------
    # CrystalDiagnostic CRUD
    # -----------------------------------------------------------------

    async def write_diagnostic(self, diagnostic: CrystalDiagnostic) -> None:
        async with self.session() as session:
            row = CrystalDiagnosticRow(
                id=diagnostic.id,
                crystal_id=diagnostic.crystal_id,
                observed_at=diagnostic.observed_at,
                failure_mode_distribution=diagnostic.failure_mode_distribution,
                top_help_query_exemplars=diagnostic.top_help_query_exemplars,
                top_hurt_query_exemplars=diagnostic.top_hurt_query_exemplars,
                compression_ratio_p25=diagnostic.compression_ratio_p25,
                compression_ratio_p50=diagnostic.compression_ratio_p50,
                compression_ratio_p75=diagnostic.compression_ratio_p75,
                query_distribution_drift=diagnostic.query_distribution_drift,
                proposed_edit_ids=diagnostic.proposed_edit_ids,
            )
            session.add(row)

    async def get_latest_diagnostic(
        self, crystal_id: str
    ) -> Optional[CrystalDiagnostic]:
        async with self.session() as session:
            stmt = (
                select(CrystalDiagnosticRow)
                .where(CrystalDiagnosticRow.crystal_id == crystal_id)
                .order_by(CrystalDiagnosticRow.observed_at.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            return _diagnostic_from_row(row) if row else None

    # -----------------------------------------------------------------
    # CrystalEdit CRUD
    # -----------------------------------------------------------------

    async def write_edit(self, edit: CrystalEdit) -> None:
        async with self.session() as session:
            row = CrystalEditRow(
                id=edit.id,
                crystal_id=edit.crystal_id,
                edit_type=edit.edit_type,
                proposed_by=edit.proposed_by,
                rationale=edit.rationale,
                affected_facts=edit.affected_facts,
                expected_impact=edit.expected_impact,
                status=edit.status,
                executed_at=edit.executed_at,
                created_at=edit.created_at,
                actual_impact=edit.actual_impact,
            )
            session.add(row)

    async def list_edits(
        self,
        customer_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 200,
    ) -> list[CrystalEdit]:
        """List edits, optionally scoped by customer (via crystal join) and status."""
        async with self.session() as session:
            stmt = select(CrystalEditRow)
            if status is not None:
                stmt = stmt.where(CrystalEditRow.status == status)
            if customer_id is not None:
                # Join via crystal_id → crystals.customer_id
                stmt = stmt.join(
                    CrystalRow, CrystalRow.id == CrystalEditRow.crystal_id
                ).where(CrystalRow.customer_id == customer_id)
            stmt = stmt.order_by(CrystalEditRow.created_at.desc()).limit(limit)
            result = await session.execute(stmt)
            return [_edit_from_row(r) for r in result.scalars().all()]

    # -----------------------------------------------------------------
    # Feedback CRUD (Stage 2b — thumbs signal)
    # -----------------------------------------------------------------

    async def write_feedback(self, feedback: Feedback) -> None:
        """Persist a feedback row.

        We do NOT validate that the (customer_id, sequence_id, turn_index)
        triple resolves to an existing QueryLog row. Reasons:
          - The QueryLog write is async/best-effort in chat_completions;
            the user might thumbs the response before the row lands.
          - Customer-supplied sequence_id values may pre-exist in the
            customer's app before the user ever sent a message through
            us. Rejecting on "unknown sequence" creates surprising
            failures for legitimate flows.
        Joins happen at read time. Orphan feedback rows (no matching
        QueryLog) are tolerable and visible in the inspector for
        offline triage.
        """
        async with self.session() as session:
            row = FeedbackRow(
                id=feedback.id,
                customer_id=feedback.customer_id,
                sequence_id=feedback.sequence_id,
                turn_index=feedback.turn_index,
                signal=feedback.signal,
                comment=feedback.comment,
                created_at=feedback.created_at,
            )
            session.add(row)

    async def find_query_log_by_sequence(
        self,
        customer_id: str,
        sequence_id: str,
        turn_index: int,
    ) -> Optional[QueryLog]:
        """Look up a QueryLog by (customer_id, sequence_id, turn_index).

        Used by the feedback endpoint to retrieve the original prompt
        and response for learning. Returns None if the QueryLog hasn't
        been written yet (race between feedback and log write) or
        doesn't exist.

        Indexed via ix_query_logs_sequence.
        """
        async with self.session() as session:
            stmt = (
                select(QueryLogRow)
                .where(QueryLogRow.customer_id == customer_id)
                .where(QueryLogRow.sequence_id == sequence_id)
                .where(QueryLogRow.turn_index == turn_index)
                .limit(1)
            )
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row is None:
                return None
            return _query_log_from_row(row)

    async def get_last_query_log_for_sequence(
        self,
        customer_id: str,
        sequence_id: str,
    ) -> Optional[QueryLog]:
        """Return the most recent QueryLog for a conversation, or None.

        Used by the session-dispatch layer (memory blend, D-MB3) to read
        the prior turn's outcome — what was matched/routed last — so a
        vague follow-up can be recognized without reintroducing v1's
        module-global session dict. Ordered by turn_index desc, with
        timestamp as a tiebreaker. Indexed via ix_query_logs_sequence.
        """
        async with self.session() as session:
            stmt = (
                select(QueryLogRow)
                .where(QueryLogRow.customer_id == customer_id)
                .where(QueryLogRow.sequence_id == sequence_id)
                .order_by(
                    QueryLogRow.turn_index.desc(),
                    QueryLogRow.timestamp.desc(),
                )
                .limit(1)
            )
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row is None:
                return None
            return _query_log_from_row(row)

    async def list_feedback_for_customer(
        self,
        customer_id: str,
        signal: Optional[str] = None,
        limit: int = 200,
    ) -> list[Feedback]:
        """List feedback rows for a customer, optionally filtered by signal.

        Ordered by created_at descending. Used by the inspector and by
        the future batch-distillation worker (which scans for thumbs-down
        rows that haven't been processed into failure crystals yet).
        """
        async with self.session() as session:
            stmt = select(FeedbackRow).where(
                FeedbackRow.customer_id == customer_id
            )
            if signal is not None:
                stmt = stmt.where(FeedbackRow.signal == signal)
            stmt = stmt.order_by(FeedbackRow.created_at.desc()).limit(limit)
            result = await session.execute(stmt)
            return [_feedback_from_row(r) for r in result.scalars().all()]

    async def list_feedback_for_turn(
        self,
        customer_id: str,
        sequence_id: str,
        turn_index: int,
    ) -> list[Feedback]:
        """All feedback rows for one specific assistant turn.

        A turn can carry multiple feedback rows (thumbs first, comment
        later). Ordered by created_at ascending so the chronological
        feedback flow is preserved.
        """
        async with self.session() as session:
            stmt = (
                select(FeedbackRow)
                .where(FeedbackRow.customer_id == customer_id)
                .where(FeedbackRow.sequence_id == sequence_id)
                .where(FeedbackRow.turn_index == turn_index)
                .order_by(FeedbackRow.created_at)
            )
            result = await session.execute(stmt)
            return [_feedback_from_row(r) for r in result.scalars().all()]

    # -----------------------------------------------------------------
    # DslConfig CRUD (v0.4 — concept-path persistence)
    # -----------------------------------------------------------------

    async def upsert_dsl_config(
        self, customer_id: str, name: str, source_text: str
    ) -> None:
        """Insert or update a DSL config source for a customer.

        The in-memory DslConfigStore is NOT updated by this method —
        callers must invalidate their in-memory env for this customer
        separately. We don't cross that concern here because the store
        is an app-level object, not a MetadataStore concern.
        """
        async with self.session() as session:
            existing = await session.get(DslConfigRow, (customer_id, name))
            now = datetime.now(timezone.utc)
            if existing is None:
                session.add(
                    DslConfigRow(
                        customer_id=customer_id,
                        name=name,
                        source_text=source_text,
                        created_at=now,
                        updated_at=now,
                    )
                )
            else:
                existing.source_text = source_text
                existing.updated_at = now

    async def list_dsl_configs_for_customer(
        self, customer_id: str
    ) -> list[tuple[str, str]]:
        """Return [(name, source_text), ...] ordered by name for a customer.

        Returns [] if the customer has no configs. Stable order so that
        concatenation at compile time is deterministic.
        """
        async with self.session() as session:
            stmt = (
                select(DslConfigRow)
                .where(DslConfigRow.customer_id == customer_id)
                .order_by(DslConfigRow.name)
            )
            result = await session.execute(stmt)
            return [(r.name, r.source_text) for r in result.scalars().all()]

    async def delete_dsl_config(self, customer_id: str, name: str) -> bool:
        """Delete one named config. Returns True if the row existed."""
        async with self.session() as session:
            existing = await session.get(DslConfigRow, (customer_id, name))
            if existing is None:
                return False
            await session.delete(existing)
            return True


# ---------------------------------------------------------------------------
# Row → Pydantic conversion
# ---------------------------------------------------------------------------

def _customer_from_row(row: CustomerRow) -> Customer:
    routing_data = row.model_routing_config or {}
    routing = ModelRoutingConfig(**routing_data)
    return Customer(
        id=row.id,
        # Raw key is never stored — only api_key_hash. Reads expose no key.
        api_key=None,
        inference_mode=getattr(row, "inference_mode", None) or "byok",
        subscription_tier=row.subscription_tier,
        model_routing_config=routing,
        injection_preference=row.injection_preference,  # type: ignore[arg-type]
        shadow_sample_rate=row.shadow_sample_rate,
        routing_context_window=row.routing_context_window,
        shadow_max_per_day=row.shadow_max_per_day,
        retention_policy=row.retention_policy,
        billing_config=row.billing_config,
        # Column truth, verbatim (opt-out fix, 2026-06-12). The previous
        # `or ["general:legacy"]` fallback double-defaulted: the column's
        # server_default already subscribes new rows to legacy at INSERT,
        # so the only rows this fallback ever touched were explicit
        # opt-outs — a customer who unsubscribed from everything ([])
        # silently snapped back to legacy. Defaulting lives in exactly
        # one place now: the column. NULL (edge rows predating the
        # column) maps to [] — the same answer
        # get_customer_general_types, the declared source of truth for
        # both retrieval merge points, has always given.
        general_crystal_types=row.general_crystal_types or [],
        created_at=row.created_at,
    )


def _operator_from_row(row: OperatorRow) -> Operator:
    return Operator(
        id=row.id,
        team_id=row.team_id,
        display_name=row.display_name,
        role=row.role,  # type: ignore[arg-type]
        status=row.status,  # type: ignore[arg-type]
        api_key_hash=row.api_key_hash,
        credential_public_key=row.credential_public_key,
        created_at=row.created_at,
    )


def _user_from_row(row: UserRow) -> User:
    return User(
        id=row.id,
        email=row.email,
        customer_id=row.customer_id,
        role=row.role,
        industry=row.industry,
        building=row.building,
        experience=row.experience,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _crystal_from_row(row: CrystalRow) -> Crystal:
    return Crystal(
        id=row.id,
        customer_id=row.customer_id,
        summary_vector=row.summary_vector or [],
        # routing_vector is nullable. Phase 6.3 (May 2026): empty-list-
        # vs-None matters downstream — VectorStore._ensure_loaded skips
        # rows whose routing_vector is None or empty. Don't coerce None
        # to [] here; the distinction is load-bearing.
        routing_vector=row.routing_vector,
        # answer_embedding_native is nullable. Empty-list-vs-None matters
        # downstream (synthesis branch checks `is None` to decide whether
        # the crystal can participate). Don't coerce to [] here.
        answer_embedding_native=row.answer_embedding_native,
        # encoder_fingerprint is nullable on legacy crystals (built
        # before migration 0009). Round-trip as-is; recall paths gate
        # bind-storage operations on it being present.
        encoder_fingerprint=row.encoder_fingerprint,
        # source_kind / answer_value are populated by migration 0006's
        # server default + nullable column. Pass through verbatim;
        # validation lives in the Pydantic Crystal model.
        source_kind=row.source_kind,  # type: ignore[arg-type]
        answer_value=row.answer_value,
        decay_rate=row.decay_rate,
        fact_count=row.fact_count,
        quality_tier=row.quality_tier,  # type: ignore[arg-type]
        recall_gated=bool(getattr(row, "recall_gated", False)),
        origin=getattr(row, "origin", "direct") or "direct",
        eval_helped_count=row.eval_helped_count,
        eval_hurt_count=row.eval_hurt_count,
        live_shadow_helped_count=row.live_shadow_helped_count,
        live_shadow_hurt_count=row.live_shadow_hurt_count,
        keyword_fingerprint=row.keyword_fingerprint or [],
        cluster_tightness=row.cluster_tightness,
        attribution_spread=row.attribution_spread,
        summary_text=row.summary_text,
        build_method=row.build_method,
        parent_crystal_id=row.parent_crystal_id,
        # V2 source versioning (VS-D2). Nullable on every pre-versioning
        # row; round-trip verbatim. Replace semantics (VS-D3, locked
        # 2026-06-10): no is_current flag — stale crystals are deleted,
        # never kept.
        source_path=row.source_path,
        content_hash=row.content_hash,
        source_modified_at=row.source_modified_at,
        crystal_type=row.crystal_type,
        # Foundation F2 (POSIX permissions). Pass through verbatim; the
        # resolver (infrastructure/permissions.can_read) interprets NULL
        # owner/group as legacy (group falls back to the owning tenant).
        owner_operator_id=row.owner_operator_id,
        group_team_id=row.group_team_id,
        mode=row.mode,
        decomposer_payload=row.decomposer_payload,
        diagnostic_tags=row.diagnostic_tags or [],
        last_eval_at=row.last_eval_at,
        created_at=row.created_at,
        last_activity=row.last_activity,
    )


def _crystal_type_from_row(row: CrystalTypeRow) -> CrystalType:
    """Convert CrystalTypeRow -> CrystalType.

    String columns for `scope` and `autosplit_policy` round-trip
    verbatim; Pydantic validates against the Literal types at
    construction. Migration 0012 only seeds 'general' and 'customer'
    scopes; later phases (5 = document, future = personal) will add
    rows in those scopes.
    """
    return CrystalType(
        id=row.id,
        display_name=row.display_name,
        scope=row.scope,  # type: ignore[arg-type]
        capacity_default=row.capacity_default,
        autosplit_policy=row.autosplit_policy,  # type: ignore[arg-type]
        routing_threshold=row.routing_threshold,
        cleanup_threshold=row.cleanup_threshold,
        pair_schema_dsl=row.pair_schema_dsl,
        created_at=row.created_at,
    )


def _acl_from_row(row: CrystalAclRow) -> CrystalAcl:
    return CrystalAcl(
        crystal_id=row.crystal_id,
        principal_type=row.principal_type,  # type: ignore[arg-type]
        principal_id=row.principal_id,
        grant=row.grant,  # type: ignore[arg-type]
        granted_at=row.granted_at,
    )


def _chain_from_row(row: CrystalChainRow) -> CrystalChain:
    return CrystalChain(
        source_crystal_id=row.source_crystal_id,
        target_crystal_id=row.target_crystal_id,
        direction=row.direction,  # type: ignore[arg-type]
        created_at=row.created_at,
    )


def _diagnostic_from_row(row: CrystalDiagnosticRow) -> CrystalDiagnostic:
    return CrystalDiagnostic(
        id=row.id,
        crystal_id=row.crystal_id,
        observed_at=row.observed_at,
        failure_mode_distribution=row.failure_mode_distribution or {},
        top_help_query_exemplars=row.top_help_query_exemplars or [],
        top_hurt_query_exemplars=row.top_hurt_query_exemplars or [],
        compression_ratio_p25=row.compression_ratio_p25,
        compression_ratio_p50=row.compression_ratio_p50,
        compression_ratio_p75=row.compression_ratio_p75,
        query_distribution_drift=row.query_distribution_drift,
        proposed_edit_ids=row.proposed_edit_ids or [],
    )


def _edit_from_row(row: CrystalEditRow) -> CrystalEdit:
    return CrystalEdit(
        id=row.id,
        crystal_id=row.crystal_id,
        edit_type=row.edit_type,  # type: ignore[arg-type]
        proposed_by=row.proposed_by,
        rationale=row.rationale,
        affected_facts=row.affected_facts or [],
        expected_impact=row.expected_impact,
        status=row.status,  # type: ignore[arg-type]
        executed_at=row.executed_at,
        created_at=row.created_at,
        actual_impact=row.actual_impact,
    )


def _query_log_from_row(row: QueryLogRow) -> QueryLog:
    return QueryLog(
        id=row.id,
        customer_id=row.customer_id,
        query_text=row.query_text,
        query_vector=row.query_vector or [],
        match_type=row.match_type,  # type: ignore[arg-type]
        injection_method=row.injection_method,  # type: ignore[arg-type]
        confidence_gate_fires=row.confidence_gate_fires,
        matched_facts=row.matched_facts or [],
        response_text=row.response_text,
        response_confidence_at_commit=row.response_confidence_at_commit,
        upstream_call_made=row.upstream_call_made,
        shadow_ran=row.shadow_ran,
        shadow_delta=row.shadow_delta,
        prompt_tokens=row.prompt_tokens,
        completion_tokens=row.completion_tokens,
        shadow_prompt_tokens=row.shadow_prompt_tokens,
        shadow_completion_tokens=row.shadow_completion_tokens,
        prompt_token_overhead=row.prompt_token_overhead,
        concept_top_config=row.concept_top_config,
        concept_top_score=row.concept_top_score,
        concept_payload=row.concept_payload,
        sequence_id=row.sequence_id,
        turn_index=row.turn_index,
        routed_crystal_id=row.routed_crystal_id,
        top1_score=row.top1_score,
        top2_score=row.top2_score,
        sparse_key=getattr(row, "sparse_key", None),
        latency_ms=row.latency_ms,
        timestamp=row.timestamp,
    )


def _feedback_from_row(row: FeedbackRow) -> Feedback:
    return Feedback(
        id=row.id,
        customer_id=row.customer_id,
        sequence_id=row.sequence_id,
        turn_index=row.turn_index,
        signal=row.signal,  # type: ignore[arg-type]
        comment=row.comment,
        created_at=row.created_at,
    )


def _fact_from_row(row: FactRow) -> Fact:
    return Fact(
        id=row.id,
        crystal_id=row.crystal_id,
        claim_text=row.claim_text,
        pair_type=row.pair_type,
        # Phase 2 fields (migration 0011). Server defaults guarantee
        # non-NULL for source_kind and prompt_text on legacy rows;
        # answer_value is nullable. Pass through verbatim; the
        # Pydantic Fact model owns the SourceKind validation.
        source_kind=row.source_kind,  # type: ignore[arg-type]
        answer_value=row.answer_value,
        prompt_text=row.prompt_text or "",
        vector=row.vector or [],
        source_doc_id=row.source_doc_id,
        extracted_by=row.extracted_by,
        verified_by=row.verified_by,
        grating_strength=row.grating_strength,
        hit_count=row.hit_count,
        last_hit_at=row.last_hit_at,
        created_at=row.created_at,
    )


# ---------------------------------------------------------------------------
# FastAPI dependency hook
# ---------------------------------------------------------------------------

_store: Optional[MetadataStore] = None


def set_metadata_store(store: Optional[MetadataStore]) -> None:
    """Install (or clear) the process-wide MetadataStore.

    Called in app lifespan. Passing None clears the reference on shutdown.
    """
    global _store
    _store = store


def get_metadata_store() -> MetadataStore:
    """FastAPI dependency: returns the active MetadataStore.

    Raises RuntimeError if the store hasn't been initialized — this is a
    startup configuration error, not a request error.
    """
    if _store is None:
        raise RuntimeError(
            "MetadataStore not initialized. "
            "Call set_metadata_store() in app lifespan."
        )
    return _store
