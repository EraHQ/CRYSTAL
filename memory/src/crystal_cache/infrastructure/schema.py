"""SQLAlchemy ORM table definitions.

Mirrors the Pydantic models in crystal_cache.models 1:1. Used by the
metadata store, Alembic migrations, and any code that needs to query
the database directly.

Design notes:
- Using SQLAlchemy 2.x typed Mapped[] style
- All timestamps are UTC (use `datetime.now(timezone.utc)`)
- Vector columns: for MVP we store as JSON (list[float]). When we switch
  to pgvector in production, these become pgvector.sqlalchemy.Vector(d_hdc).
- Every customer-scoped table has an index on customer_id
- FKs are present but CASCADE is deliberately not used — we'd rather get
  a foreign key violation than silently delete telemetry or diagnostics.

Entity → table mapping:
  Customer          → customers
  Crystal           → crystals
  CrystalEdge       → crystal_edges
  Fact              → facts
  QueryLog          → query_logs
  VerificationTask  → verification_tasks
  Document          → documents
  CrystalDiagnostic → crystal_diagnostics
  CrystalEdit       → crystal_edits
  DslConfig         → dsl_configs     (v0.4 — concept-path persistence)
  ReasoningTrace    → reasoning_traces (Phase 8.5 — MCR artifact 1)
  Critique          → critiques        (Phase 8.5 — MCR artifact 2)
  ActionItem        → action_items     (Phase 8.5 — MCR artifact 3)
  ItemAlignment     → item_alignments  (Phase 10A — metacognitive artifact)
  CritiqueSynthesis → critique_syntheses (Phase 10A — metacognitive artifact)
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base class for all ORM models."""


# ---------------------------------------------------------------------------
# Customer
# ---------------------------------------------------------------------------

class CustomerRow(Base):
    __tablename__ = "customers"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # Hashed Crystal Cache API key (Key A). No plaintext at rest
    # (2026-06-13): auth hashes the presented key and matches this column
    # (see infrastructure/credentials.py). The raw key is returned once at
    # creation and never persisted.
    api_key_hash: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)

    # Serialized as JSON: ModelRoutingConfig
    model_routing_config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    injection_preference: Mapped[str] = mapped_column(String(32), default="text")
    shadow_sample_rate: Mapped[float] = mapped_column(Float, default=0.05)

    # Hosted-plane subscription tier (Phase 3 G6, 2026-07-03, ratified).
    # NULL = no tier = self-host / unlimited: the admission module treats
    # a missing tier as the uncapped policy, so self-host deployments are
    # untouched by design. The managed platform sets this at signup.
    subscription_tier: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True, default=None,
    )
    retention_policy: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    billing_config: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Phase 1.5.3: per-customer override for multi-turn routing window.
    # NULL = use system default (3 user turns). An explicit integer
    # overrides the window size for this customer's queries.
    routing_context_window: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )

    # Phase 12 (CU-27 / P0.111): per-customer override for the daily
    # shadow-critique cost cap. NULL = use the global default
    # (settings.shadow_max_per_customer_per_day). An explicit integer
    # caps THIS customer's shadow critiques per rolling 24h window,
    # letting the operator tune metacognition (R&D) spend per customer
    # tier. Mirrors routing_context_window's nullable-override shape.
    shadow_max_per_day: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )

    # Subscription tier (Phase 3 G6, 2026-07-03, ratified): names the row
    # in the admission tier table (control/admission.py) that caps this
    # tenant's disposable-task deadline, budget, queue depth, concurrency,
    # and GPU access on the HOSTED plane. NULL = the deployment default
    # (settings.default_subscription_tier). Self-host ignores this — the
    # operator sets limits directly.
    subscription_tier: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True
    )
    # byok (default; API-created back-compat) | managed (signup default —
    # Era-keyed inference, ledger-flagged, capped). E4, Accounts Phase B
    # 2026-07-06. String, not enum, per convention.
    inference_mode: Mapped[str] = mapped_column(
        String(16), nullable=False, default="byok", server_default="byok"
    )

    # General crystal subscriptions (V2). JSON list of crystal_type IDs
    # the customer is subscribed to for general knowledge retrieval.
    general_crystal_types: Mapped[Optional[list[str]]] = mapped_column(
        JSON, nullable=True, server_default='["general:legacy"]'
    )

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# ---------------------------------------------------------------------------
# Operator (Foundation F1 — team identity layer)
# ---------------------------------------------------------------------------

class OperatorRow(Base):
    """A human user under a team (= customer). Foundation F1.

    The identity layer beneath the customer/team entity
    (`docs/FOUNDATION_AND_GROWTH.md` F1). One shared API key can't carry
    multiple humans — no per-person attribution, revocation, or roles —
    so operators add an authenticated-human layer beneath the team that
    already owns everything. Roles (D1, locked) are the default POSIX
    posture: admin = root (team-scoped), operator = regular user,
    viewer = read-only; per-resource mode bits + named-grant ACLs
    (crystal_acls) refine specifics on top.

    Additive + backward-compatible: the pre-existing lone customer keeps
    working as a one-operator team; nothing here is required by current
    single-customer flows. Per the AgentTaskRow precedent, local default
    stores create this table free via store.init()'s create-missing-
    tables; the Alembic-managed dev DB needs a migration
    (alembic revision --autogenerate -m "operators").
    """
    __tablename__ = "operators"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # The team this operator belongs to (= customers.id). v1: exactly one
    # team per operator (many-to-many is a future team_memberships table).
    team_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False, index=True
    )
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)

    # Role posture (D1). String, not an enum column, so a future role
    # (e.g. billing-only) lands without an Alembic step — matching the
    # codebase's String-over-enum convention (crystal scope/grant).
    role: Mapped[str] = mapped_column(
        String(32), nullable=False, default="operator", server_default="operator"
    )
    # Lifecycle. 'suspended' is denied at the auth boundary WITHOUT
    # deleting the row, so owned crystals + provenance survive.
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="active", server_default="active"
    )

    # Per-operator scoped credential (D2). Stored HASHED — auth hashes the
    # presented key and looks up by hash; the raw key is never persisted.
    # Nullable: an operator may exist before a key is issued. Unique when
    # present (one key -> one operator). Confirmed 2026-06-13: hashed, no
    # plaintext anywhere (the legacy plaintext Customer.api_key migrates
    # to the same scheme next).
    api_key_hash: Mapped[Optional[str]] = mapped_column(
        String(128), unique=True, nullable=True
    )

    # Passkey / public-key anchor (D2). The signing identity G2's control
    # plane reuses for end-to-end signed authorization (WebAuthn/passkeys).
    # Provisioned here so the control plane gets it for free; NULL until a
    # passkey is registered.
    credential_public_key: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class UserRow(Base):
    """An account holder on the hosted platform (Phase A, 2026-07-06).

    The IdP-anchored identity: `id` IS the GCP Identity Platform uid, so
    JWT resolution is a primary-key get with no mapping table. Distinct
    from OperatorRow (team-internal humans under Key-A auth): users are
    the SIGN-IN layer; operators remain the in-team identity. One user ->
    one tenant in v1 (multi-seat is a later platform feature; this schema
    does not block it).

    role: 'owner' (a tenant's account) | 'platform_admin' (Era staff;
    customer_id NULL — the platform root sits above tenants). String,
    not enum, per codebase convention. Onboarding fields land in OUR
    database, queryable from day one (ratified plan).
    """
    __tablename__ = "users"

    # GCP Identity Platform uid (verbatim). PK => JWT resolution is a get().
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    # NULL only for platform_admin (no home tenant).
    customer_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=True, index=True
    )
    role: Mapped[str] = mapped_column(
        String(32), nullable=False, default="owner", server_default="owner"
    )
    # Onboarding signal (industry / what you're building / experience).
    industry: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    building: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    experience: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


# ---------------------------------------------------------------------------
# Crystal
# ---------------------------------------------------------------------------

class CrystalRow(Base):
    __tablename__ = "crystals"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=True, index=True
    )  # None = general crystal

    # 10k-dim vector — JSON for MVP, pgvector.Vector(10000) in production
    summary_vector: Mapped[list[float]] = mapped_column(JSON, nullable=False)

    # Phase 6.3 (migration 0014, May 2026): routing-vector parallel
    # storage. `Σ encode(prompt_i) @ P` per-write accumulator that lives
    # alongside summary_vector. VectorStore.search routes on this
    # vector; recall_from_crystal continues to unbind against
    # summary_vector.
    #
    # Nullable for the same reason answer_embedding_native is nullable:
    # pre-migration crystals haven't been backfilled. The Pydantic
    # Crystal model and the round-trip in _crystal_from_row preserve
    # None, and VectorStore._ensure_loaded skips rows with None or
    # empty routing_vector at load time.
    routing_vector: Mapped[Optional[list[float]]] = mapped_column(
        JSON, nullable=True
    )

    # Native-dim (e.g. 768 for gtr-t5-base) embedding of the canonical
    # answer. Optional/nullable: crystals created before migration 0005
    # do not have this populated and use the M2-only routing path.
    # Populated for new crystals by import_bank under
    # the semantic encoder. Consumed by the synthesis path (bind-v1
    # decoder input) on SPREAD-decision queries.
    answer_embedding_native: Mapped[Optional[list[float]]] = mapped_column(
        JSON, nullable=True
    )

    # Encoder geometry fingerprint (migration 0009, Phase 1.1
    # mitigations, April 2026).
    #
    # Stamped on first bind-storage write to a crystal; re-checked on
    # every subsequent write and at recall time. A mismatch between
    # the encoder used to write and the encoder used to read produces
    # out-of-distribution recovered vectors that bind-v1 silently
    # misdecodes. The fingerprint catches that with a clear
    # ValueError instead.
    #
    # Format: produced by `BindCapableEncoder.fingerprint()`. For the
    # production semantic encoder, looks like:
    #   "semantic:sentence-transformers/gtr-t5-base/native=768/hdc=10000/seed=42"
    #
    # Nullable because:
    #   - Pre-Phase-1.1 crystals (built via the legacy import_bank.py
    #     direct-upsert path) have no fingerprint and continue to work
    #     for legacy M2-only routing.
    #   - Crystals built via add_pair_to_crystal AFTER migration 0009
    #     get the field populated automatically.
    encoder_fingerprint: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )

    # Provenance + cache-hit support (migration 0006, April 2026).
    #
    # source_kind tags what kind of evidence this crystal carries:
    #   'model_reasoning'      — verified-correct prior answer (default)
    #   'failed_reasoning'     — imperative rule from a wrong attempt
    #   'web_search_result'    — advisory reference from upstream search
    #   'code_execution_result'— advisory reference from upstream code exec
    # See models/crystal.py SourceKind for the full enum + rationale.
    # Server default 'model_reasoning' preserves pre-0006 rows as
    # success crystals on SELECT.
    #
    # answer_value carries the canonical short answer for cache-hit
    # short-circuiting. NULL for failure rules and legacy rows; populated
    # by import_bank for new success crystals.
    source_kind: Mapped[str] = mapped_column(
        String(32), default="model_reasoning", server_default="model_reasoning"
    )
    answer_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    decay_rate: Mapped[float] = mapped_column(Float, default=0.01)
    fact_count: Mapped[int] = mapped_column(Integer, default=0)

    # Quality gate
    quality_tier: Mapped[str] = mapped_column(String(32), default="quarantine", index=True)

    # Recall gate + birth attribution (2026-07-03, recall-gated memory).
    #
    # recall_gated is the "can this crystal be USED at all" bit, ORTHOGONAL
    # to quality_tier (which is only an epistemic SIGNAL and never gates
    # recall). Default False = normal: recall behaves exactly as before and
    # tier stays a pure signal. True = the crystal is held OUT of the recall
    # candidate set entirely until the gate is cleared (by human approval or
    # a system_rules promotion rule). This is how autonomous background
    # workers write memory that cannot be relied on until reviewed, without
    # changing what any tier means.
    #
    # origin records WHAT created the crystal (distinct from source_kind,
    # which records the KIND of evidence). 'direct' = foreground/user
    # ingest (the default, unchanged behavior). 'background_worker' =
    # autonomous task output (born recall_gated). Rules and audits key on
    # this axis; it also lands the birth-attribution the ACL work needs.
    recall_gated: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="0", index=True,
    )
    origin: Mapped[str] = mapped_column(
        String(32), default="direct", server_default="direct", index=True,
    )

    eval_helped_count: Mapped[int] = mapped_column(Integer, default=0)
    eval_hurt_count: Mapped[int] = mapped_column(Integer, default=0)
    live_shadow_helped_count: Mapped[int] = mapped_column(Integer, default=0)
    live_shadow_hurt_count: Mapped[int] = mapped_column(Integer, default=0)

    # Diagnostic fingerprint (research §2.2)
    keyword_fingerprint: Mapped[list[str]] = mapped_column(JSON, default=list)
    cluster_tightness: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    attribution_spread: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Human-readable summary used by CrystalReader (Group D injection)
    summary_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Construction + lineage
    build_method: Mapped[str] = mapped_column(String(32), default="kmeans")
    parent_crystal_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("crystals.id"), nullable=True
    )

    # V2 source versioning (VS-D2). For crystals ingested from a file
    # (code or documents): which file the crystal came from, a content
    # fingerprint for change-detection / dedup, and the file's modified
    # time. Replace semantics (VS-D3, locked 2026-06-10): re-ingest of a
    # changed source DELETES the prior crystals for that source_path and
    # writes fresh ones — no is_current flag, no stale crystals, ever.
    # All nullable so non-file crystals are untouched.
    source_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    content_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    source_modified_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Phase 3 (migration 0012, April 2026): crystal type registry FK.
    #
    # Every crystal carries a type (e.g. 'general:legacy',
    # 'customer:medical_records'). The type carries scope, capacity
    # default, autosplit policy, and per-type bond_threshold /
    # cleanup_threshold overrides via the `crystal_types` table.
    #
    # No SQL FK to crystal_types.id today (per migration 0012's
    # rationale: SQLite Alembic batch-mode FKs to fresh tables are
    # fragile, and the column is application-validated). The MetadataStore
    # checks the type exists before write; an unknown type is a
    # programming error caught at runtime.
    #
    # NOT NULL with server default 'customer:legacy' so any code path
    # that hasn't been updated for Phase 3 yet still produces valid
    # rows (they land in the legacy bucket, which is the intended
    # back-compat target).
    crystal_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        default="customer:legacy",
        server_default="customer:legacy",
    )

    # Foundation F2 (POSIX permissions). The crystal as an owned resource:
    #   owner_operator_id — the operator who authored it (FK operators.id;
    #     NULL for legacy / non-operator-authored crystals).
    #   group_team_id     — the POSIX group, i.e. the owning team (FK
    #     customers.id; NULL for legacy, where the resolver falls back to
    #     customer_id). Distinct from customer_id (the owning TENANT) so a
    #     future regrouping can move a crystal's group without restating
    #     tenancy.
    #   mode              — POSIX mode bits as an int (0o640 == 416). Only
    #     the READ bits are consumed today (retrieval gating in
    #     infrastructure/permissions.can_read); write/execute are reserved.
    # Default 0o640 (owner rw, group r, other none) = team-readable, which
    # preserves "a team reads its own crystals" for pre-F2 rows on SELECT.
    # No index: the bits are consumed as an in-memory filter over an
    # already-tenancy-scoped candidate set (FactVectorStore.search), not a
    # WHERE clause — YAGNI until a "crystals I own" query needs one.
    owner_operator_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("operators.id"), nullable=True
    )
    group_team_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=True
    )
    mode: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0o640, server_default="416"
    )

    # Phase 6.3 follow-up #2 (migration 0016, May 2026): the
    # decomposer payload that established this crystal's "concept
    # identity" at spawn time. Populated ONLY on spawn-fresh by
    # add_pair_for_customer when a wired decomposer returned a
    # payload; bond writes do NOT update this field. Consumed by
    # ThreeAxisBonder for axis-3 (payload agreement) gray-zone
    # decisions. NULL is legitimate for three groups: pre-followup-2
    # crystals, crystals spawned without a wired decomposer, and
    # crystals whose first-bond decomposer call raised
    # DecomposerError. The bonder treats None as "no axis-3 signal,
    # conservative spawn in the gray zone."
    #
    # Shape is the decomposer's `DecompositionResult.payload` dict
    # (typically {intent, topic, domain, optional tone, optional
    # urgency}). Schema isn't pinned in the DB layer because the
    # decomposer protocol explicitly does not pin it (see
    # decomposer/base.py: "PAYLOAD SHAPE IS LOOSE").
    decomposer_payload: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSON, nullable=True
    )

    # Populated by diagnostic engine
    diagnostic_tags: Mapped[list[str]] = mapped_column(JSON, default=list)

    last_eval_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_activity: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        Index("ix_crystals_customer_tier", "customer_id", "quality_tier"),
        # Phase 3 (migration 0012): composite index for type-scoped
        # routing. The access pattern is "list customer X's crystals
        # of type Y" — e.g. add_pair_for_customer routing through
        # `customer:medical_records` only, ignoring the customer's
        # `customer:billing_records` crystals at routing time.
        Index("ix_crystals_customer_type", "customer_id", "crystal_type"),
    )


# ---------------------------------------------------------------------------
# Phase 3 (migration 0012): Crystal type registry, ACLs, chains.
# ---------------------------------------------------------------------------

class CrystalTypeRow(Base):
    """Per-tenant (or global) registry entry describing a kind of crystal.

    Every CrystalRow.crystal_type points at one of these. The type
    governs routing/cleanup threshold defaults, capacity ceiling, and
    autosplit policy. Phase 4 lands `pair_schema_dsl` consumption
    (validating writes against declared pair-types).

    No customer FK: the type id namespace IS the scope marker.
    'general:*' types are world-shared; 'customer:*' are conceptually
    per-tenant but implemented as a shared id pool keyed by convention.
    Actual access governance is in `crystal_acls`, not here.
    """
    __tablename__ = "crystal_types"

    # String PK matches the convention from existing String enum
    # columns. Format by convention: 'scope:slug'
    # (e.g. 'general:math', 'customer:medical_records').
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)

    # 'general' | 'customer' | 'document' | 'personal'. Validated at
    # the Pydantic layer; column stays a String so future scopes can
    # land without an Alembic step.
    scope: Mapped[str] = mapped_column(String(32), nullable=False)

    # Per-type capacity ceiling. Default 50 matches the global
    # CRYSTAL_CAPACITY_HARD_CEILING. Customer-tier types may raise it;
    # document-tier defaults to 50 with autosplit on.
    capacity_default: Mapped[int] = mapped_column(
        Integer, nullable=False, default=50, server_default="50"
    )

    # 'split' | 'refuse'. 'split' (default) auto-spawns a sibling
    # crystal when capacity is hit. 'refuse' raises CrystalCapacityError
    # so the operator picks the partition explicitly.
    autosplit_policy: Mapped[str] = mapped_column(
        String(32), nullable=False, default="split", server_default="split"
    )

    # Per-type override of the global bond_threshold (write-side
    # routing in add_pair_for_customer) and cleanup_threshold (read-
    # side recall in recall_from_crystal). NULL = use the global from
    # `settings`. Keeping these nullable rather than snapshot-defaulting
    # to the global values lets us tell "this type was explicitly
    # tuned" apart from "this type was created with whatever the
    # global was at the time" — important for Phase 6.3's calibrator.
    routing_threshold: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    cleanup_threshold: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )

    # Phase 4 hook. The DSL source declaring valid pair_types for
    # crystals of this type, default ACL grants, routing concept-paths,
    # etc. Empty string today; Phase 4 lands the parser/compiler.
    pair_schema_dsl: Mapped[str] = mapped_column(
        Text, nullable=False, default="", server_default=""
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )


class CrystalAclRow(Base):
    """Per-crystal access grant.

    Composite PK on (crystal_id, principal_type, principal_id, grant)
    so a single crystal can carry multiple distinct grants without
    update-on-conflict semantics:

      (crys_X, 'customer',     'cus_A',  'read')
      (crys_X, 'crystal_chain','crys_Y', 'read_codebook')

    Both rows coexist; each is an independent grant. Removing one
    (single DELETE) doesn't disturb the other.

    The ACL system is OPT-IN ADDITIVE: missing rows mean "use the
    default for the crystal's scope." Customer-scope crystals default
    to (customer_id, read); general-scope to (world, read). The
    resolver consults rows when present and falls back to scope
    defaults when absent. See chain_resolver.py for the order.
    """
    __tablename__ = "crystal_acls"

    crystal_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("crystals.id"), primary_key=True
    )
    # 'customer' | 'global' | 'crystal_chain'.
    #   - 'customer' + customer_id: a tenant-scoped grant
    #   - 'global'   + 'world':     a public grant (general-tier)
    #   - 'crystal_chain' + crystal_id: another crystal can borrow
    #     this one's facts via chain (granted at the target's ACL,
    #     not the source's).
    principal_type: Mapped[str] = mapped_column(
        String(32), primary_key=True
    )
    principal_id: Mapped[str] = mapped_column(
        String(64), primary_key=True
    )
    # 'read' | 'read_codebook'.
    #   - 'read': principal can route INTO this crystal AND consume
    #     its facts.
    #   - 'read_codebook': principal cannot route in but CAN extend
    #     their cleanup codebook with this crystal's Facts via chain
    #     resolution. The chaining primitive's load-bearing grant.
    grant: Mapped[str] = mapped_column(
        String(32), primary_key=True
    )

    granted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    __table_args__ = (
        # Principal-side lookup: "what crystals can this principal
        # access?" Used by the ACL resolver. The PK covers the
        # crystal-side lookup ("what grants does this crystal carry?")
        # already.
        Index(
            "ix_crystal_acls_principal",
            "principal_type", "principal_id",
        ),
    )


class CrystalChainRow(Base):
    """Directed edge between two crystals for cleanup-codebook extension.

    Composite PK on (source_crystal_id, target_crystal_id). A chain
    edge is uniquely identified by its endpoints; direction is a
    property of that single edge.

    When recall walks out from `source_crystal_id`, it pulls
    `target_crystal_id`'s Facts into the cleanup codebook (subject to
    target's `read_codebook` ACL granted to the source's customer).
    direction='source_uses_target' is one-way (default);
    'bidirectional' is both ways. Bidirectional is implemented as a
    single row, not two — the resolver checks the direction column
    when traversing target -> sources.

    Self-loops (source == target) are schema-allowed but resolver-
    skipped — a crystal already includes its own facts in cleanup,
    chaining to itself adds nothing.
    """
    __tablename__ = "crystal_chains"

    source_crystal_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("crystals.id"), primary_key=True
    )
    target_crystal_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("crystals.id"), primary_key=True
    )
    # 'source_uses_target' | 'bidirectional'.
    direction: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="source_uses_target",
        server_default="source_uses_target",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    __table_args__ = (
        # Reverse-direction lookup: "which crystals chain INTO this
        # target?" Used by bidirectional chain resolution and by the
        # inspector. The PK covers the forward (source -> targets)
        # case directly via index-prefix scan.
        Index(
            "ix_crystal_chains_target",
            "target_crystal_id",
        ),
    )


# ---------------------------------------------------------------------------
# CrystalEdge
# ---------------------------------------------------------------------------

class CrystalEdgeRow(Base):
    __tablename__ = "crystal_edges"

    # Composite primary key: (crystal_a_id, crystal_b_id, edge_type)
    crystal_a_id: Mapped[str] = mapped_column(String(64), ForeignKey("crystals.id"), primary_key=True)
    crystal_b_id: Mapped[str] = mapped_column(String(64), ForeignKey("crystals.id"), primary_key=True)
    edge_type: Mapped[str] = mapped_column(String(32), primary_key=True, default="co_queried")

    weight: Mapped[float] = mapped_column(Float, default=0.0)
    last_reinforced_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# ---------------------------------------------------------------------------
# Fact
# ---------------------------------------------------------------------------

class FactRow(Base):
    __tablename__ = "facts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    crystal_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("crystals.id"), nullable=False, index=True
    )

    claim_text: Mapped[str] = mapped_column(Text, nullable=False)

    # Pair-type tag (Phase 0.1, April 2026). Set at write time, immutable.
    # The cleanup match at recall time returns one Fact, and that Fact's
    # pair_type is the inferred query type. Indexed for diagnostic lookups
    # like "how many medication_dosage Facts does this crystal hold?";
    # NOT used for filtering at recall time — cleanup walks the full
    # crystal codebook and the pair_type emerges from the match.
    pair_type: Mapped[str] = mapped_column(
        String(64), nullable=False, server_default="question_answer", index=True
    )

    # Phase 2 (April 2026, migration 0011): source_kind, answer_value,
    # prompt_text. The cache-hit short-circuit and source-kind-aware
    # injection move from the Crystal-level fields (added in 0006) to
    # the matched Fact: a crystal holds many pairs, and the right
    # voicing/answer comes from the SPECIFIC pair cleanup recovered,
    # not from the crystal's aggregate.
    #
    # source_kind: which kind of evidence this Fact carries
    # ('model_reasoning' / 'failed_reasoning' / 'web_search_result' /
    # 'code_execution_result'). Server default 'model_reasoning' so
    # pre-0011 rows SELECT cleanly as success Facts.
    #
    # answer_value: canonical short answer for the cache-hit path.
    # NULL when the Fact's claim_text is itself the full answer
    # (legacy imports, document-section Facts, anything non-cache-
    # shaped). The pipeline's cache-hit branch checks
    # `answer_value is not None and != ''` before short-circuiting.
    #
    # prompt_text: the prompt that was bind-paired into this Fact at
    # write time. Persisting it unlocks Phase 6.3's per-crystal
    # cleanup_threshold calibrator (replay the unbind per pair to
    # measure recoverability) and inspector display of "what queries
    # match this codebook entry?". Server default '' preserves
    # pre-0011 rows whose prompts were never written.
    source_kind: Mapped[str] = mapped_column(
        String(32), nullable=False,
        default="model_reasoning", server_default="model_reasoning",
    )
    answer_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    prompt_text: Mapped[str] = mapped_column(
        Text, nullable=False, default="", server_default="",
    )

    vector: Mapped[list[float]] = mapped_column(JSON, default=list)

    source_doc_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("documents.id"), nullable=True
    )
    extracted_by: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    verified_by: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    grating_strength: Mapped[float] = mapped_column(Float, default=1.0)
    hit_count: Mapped[int] = mapped_column(Integer, default=0)
    last_hit_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# ---------------------------------------------------------------------------
# QueryLog
# ---------------------------------------------------------------------------

class QueryLogRow(Base):
    __tablename__ = "query_logs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False, index=True
    )

    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    query_vector: Mapped[list[float]] = mapped_column(JSON, default=list)

    match_type: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    injection_method: Mapped[str] = mapped_column(String(32), default="none")
    confidence_gate_fires: Mapped[int] = mapped_column(Integer, default=0)

    matched_facts: Mapped[list[str]] = mapped_column(JSON, default=list)

    response_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    response_confidence_at_commit: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    upstream_call_made: Mapped[bool] = mapped_column(Boolean, default=True)

    shadow_ran: Mapped[bool] = mapped_column(Boolean, default=False)
    shadow_delta: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # v0.4 token accounting — populated after upstream response has usage
    # data. Null for rows written before the 0004 migration and for
    # requests where the upstream didn't return usage (errors, legacy
    # self-hosted endpoints that don't report token counts).
    #
    # shadow_* columns are null when shadow_ran is False. prompt_token_overhead
    # is null unless BOTH prompt_tokens AND shadow_prompt_tokens are set;
    # it's just prompt_tokens - shadow_prompt_tokens but persisted for
    # one-column dashboard aggregates.
    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    shadow_prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    shadow_completion_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    prompt_token_overhead: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # v0.3+ concept-path observations (read-only, not routing-authoritative)
    concept_top_config: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    concept_top_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    concept_payload: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # Stage 2a (April 2026, GAIA fold-back): sequence anchoring.
    sequence_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    turn_index: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Phase 1.2 (April 2026): routing-decision telemetry.
    routed_crystal_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("crystals.id"), nullable=True
    )
    top1_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    top2_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # V2: sparse key used for this query's retrieval.
    sparse_key: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    __table_args__ = (
        Index(
            "ix_query_logs_sequence",
            "customer_id", "sequence_id", "turn_index",
        ),
    )


# ---------------------------------------------------------------------------
# VerificationTask
# ---------------------------------------------------------------------------

class VerificationTaskRow(Base):
    __tablename__ = "verification_tasks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False, index=True
    )

    candidate_claim: Mapped[str] = mapped_column(Text, nullable=False)
    candidate_vector: Mapped[list[float]] = mapped_column(JSON, default=list)
    source: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)

    priority: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    assigned_to: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

class DocumentRow(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False, index=True
    )

    source: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    facts_extracted_count: Mapped[int] = mapped_column(Integer, default=0)
    facts_verified_count: Mapped[int] = mapped_column(Integer, default=0)
    facts_rejected_count: Mapped[int] = mapped_column(Integer, default=0)

    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# ---------------------------------------------------------------------------
# CrystalDiagnostic (research-grounded §4)
# ---------------------------------------------------------------------------

class CrystalDiagnosticRow(Base):
    __tablename__ = "crystal_diagnostics"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    crystal_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("crystals.id"), nullable=False, index=True
    )

    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    failure_mode_distribution: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)

    top_help_query_exemplars: Mapped[list[str]] = mapped_column(JSON, default=list)
    top_hurt_query_exemplars: Mapped[list[str]] = mapped_column(JSON, default=list)

    compression_ratio_p25: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    compression_ratio_p50: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    compression_ratio_p75: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    query_distribution_drift: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    proposed_edit_ids: Mapped[list[str]] = mapped_column(JSON, default=list)


# ---------------------------------------------------------------------------
# CrystalEdit (research-grounded §4)
# ---------------------------------------------------------------------------

class CrystalEditRow(Base):
    __tablename__ = "crystal_edits"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    crystal_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("crystals.id"), nullable=False, index=True
    )

    edit_type: Mapped[str] = mapped_column(String(32), nullable=False)
    proposed_by: Mapped[str] = mapped_column(String(32), default="diagnostic_engine")

    rationale: Mapped[str] = mapped_column(Text, default="")
    affected_facts: Mapped[list[str]] = mapped_column(JSON, default=list)

    expected_impact: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    executed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    actual_impact: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


# ---------------------------------------------------------------------------
# Feedback (Stage 2b — explicit thumbs signal for retrospective learning)
# ---------------------------------------------------------------------------

class FeedbackRow(Base):
    """User feedback (thumbs up/down) on a specific assistant turn."""
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False
    )

    sequence_id: Mapped[str] = mapped_column(String(64), nullable=False)
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False)

    signal: Mapped[str] = mapped_column(String(8), nullable=False)
    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    __table_args__ = (
        Index("ix_feedback_customer", "customer_id", "created_at"),
        Index("ix_feedback_sequence", "customer_id", "sequence_id", "turn_index"),
    )


# ---------------------------------------------------------------------------
# DslConfig (v0.4 — persisted source text for compiled DSL configs)
# ---------------------------------------------------------------------------

class DslConfigRow(Base):
    """Persisted DSL source for the concept-path ConfigStore."""
    __tablename__ = "dsl_configs"

    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), primary_key=True
    )
    name: Mapped[str] = mapped_column(String(128), primary_key=True)

    source_text: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


# ---------------------------------------------------------------------------
# V2 Learning State Tables (migration 0018)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# System rules — user-owned automation of their own judgment (2026-07-03).
#
# A declarative, per-tenant rule set: "WHEN <conditions>, DO <action> to
# things matching <selector>." Storage is generic (JSON columns) so new
# rule_types are code + validation, not migrations; execution is TYPED per
# rule_type (a validator/executor declares exactly which selectors,
# conditions, and actions it accepts), so a malformed or malicious rule
# can't do something unintended — the executor is the safety boundary,
# mirroring how the sandbox is the exec boundary. First rule_type shipped:
# 'promotion' (clears the recall gate on background-worker memory when the
# user's conditions hold). Designed to later hold 'sharing', 'approval',
# 'task_spawn'. Rules come ONLY from the user via the control plane — never
# from tool output or crystal content (same instruction-source boundary as
# everywhere else), so a poisoned crystal can't author a rule to promote
# itself.
# ---------------------------------------------------------------------------

class SystemRuleRow(Base):
    __tablename__ = "system_rules"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False
    )
    rule_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, server_default="1")
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    # WHAT the rule applies to / WHEN it fires / WHAT it does. JSON so the
    # shape generalizes across rule_types without per-type migrations; the
    # typed validator for each rule_type enforces the real schema.
    selector: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    conditions: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    action: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    priority: Mapped[int] = mapped_column(Integer, default=100)
    # Audit: why did this fire, and how often.
    last_fired_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    fire_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    __table_args__ = (
        Index("idx_system_rules_customer_type", "customer_id", "rule_type"),
    )


class MandatoryRuleRow(Base):
    __tablename__ = "mandatory_rules"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False
    )
    rule_text: Mapped[str] = mapped_column(Text, nullable=False)
    is_mandatory: Mapped[bool] = mapped_column(Boolean, default=True)
    unless_clause: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_round: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    __table_args__ = (
        Index("idx_mandatory_rules_customer", "customer_id"),
    )


class MetaPatternRow(Base):
    __tablename__ = "meta_patterns"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False
    )
    pattern_text: Mapped[str] = mapped_column(Text, nullable=False)
    affected_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        Index("idx_meta_patterns_customer", "customer_id"),
    )


class BlacklistedReflectionRow(Base):
    __tablename__ = "blacklisted_reflections"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False
    )
    reflection_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    reflection_text: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        Index("idx_blacklisted_customer", "customer_id"),
        Index(
            "idx_blacklisted_hash", "customer_id", "reflection_hash",
            unique=True,
        ),
    )


# ---------------------------------------------------------------------------
# Document Uploads (V2 — document ingestion pipeline)
# ---------------------------------------------------------------------------

class DocumentUploadRow(Base):
    __tablename__ = "document_uploads"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False
    )
    label: Mapped[str] = mapped_column(String(256), default="")
    text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), default="pending", server_default="pending"
    )
    crystal_type: Mapped[str] = mapped_column(
        String(128), default="customer:legacy"
    )
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    crystals_written: Mapped[int] = mapped_column(Integer, default=0)
    items_extracted: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    crystallized_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    source_file_id: Mapped[Optional[str]] = mapped_column(
        String(256), nullable=True
    )
    source_modified_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    source_connection_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )

    extracted_items: Mapped[Optional[list]] = mapped_column(
        JSON, nullable=True
    )
    detected_type: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )
    confirmed_type: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )
    content_chunks: Mapped[Optional[list]] = mapped_column(
        JSON, nullable=True
    )
    # P2 scope-on-sources (ratified 2026-07-02): a document is a SOURCE and
    # carries its own scope; every crystal born from it inherits these
    # stamps. NULL = legacy row → team-scoped unowned crystals (today's
    # behavior, so in-flight uploads are unaffected by the migration).
    scope: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    owner_operator_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )

    __table_args__ = (
        Index("idx_document_uploads_customer", "customer_id"),
        Index("idx_document_uploads_status", "customer_id", "status"),
        Index("idx_doc_uploads_source_file", "customer_id", "source_file_id"),
    )


# ---------------------------------------------------------------------------
# Drive Connections (Google Drive OAuth — encrypted tokens)
# ---------------------------------------------------------------------------

class DriveConnectionRow(Base):
    __tablename__ = "drive_connections"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False
    )
    provider: Mapped[str] = mapped_column(String(32), default="google_drive")
    email: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    encrypted_refresh_token: Mapped[str] = mapped_column(Text, nullable=False)
    token_nonce: Mapped[str] = mapped_column(String(64), nullable=False)
    scopes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="active")
    last_synced_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class WatchedFolderRow(Base):
    __tablename__ = "watched_folders"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    connection_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("drive_connections.id"), nullable=False
    )
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False
    )
    folder_id: Mapped[str] = mapped_column(String(256), nullable=False)
    folder_name: Mapped[str] = mapped_column(String(512), nullable=False)
    folder_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    contains_phi: Mapped[bool] = mapped_column(Boolean, default=False)
    sync_interval_minutes: Mapped[int] = mapped_column(Integer, default=60)
    last_checked_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_file_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class WatchedFileRow(Base):
    __tablename__ = "watched_files"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    connection_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("drive_connections.id"), nullable=False
    )
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False
    )
    file_id: Mapped[str] = mapped_column(String(256), nullable=False)
    file_name: Mapped[str] = mapped_column(String(512), nullable=False)
    mime_type: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    contains_phi: Mapped[bool] = mapped_column(Boolean, default=False)
    sync_interval_minutes: Mapped[int] = mapped_column(Integer, default=60)
    last_checked_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_modified_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    status: Mapped[str] = mapped_column(String(32), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# ---------------------------------------------------------------------------
# BAA Tracking (HIPAA compliance)
# ---------------------------------------------------------------------------

class BaaTrackingRow(Base):
    __tablename__ = "baa_tracking"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False, unique=True
    )
    baa_signed: Mapped[bool] = mapped_column(Boolean, default=False)
    baa_signed_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    baa_document_ref: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    phi_data_sources: Mapped[Optional[list[str]]] = mapped_column(JSON, nullable=True)
    hipaa_contact_email: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# ---------------------------------------------------------------------------
# PHI Access Log (HIPAA audit trail)
# ---------------------------------------------------------------------------

class PhiAccessLogRow(Base):
    __tablename__ = "phi_access_log"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(String(64), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(256), nullable=False)
    resource_name: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    contains_phi: Mapped[bool] = mapped_column(Boolean, default=False)
    source_connection_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        Index("idx_phi_access_customer", "customer_id"),
        Index("idx_phi_access_timestamp", "customer_id", "timestamp"),
    )


# ---------------------------------------------------------------------------
# V3 Cognition Tables
# ---------------------------------------------------------------------------

class PushReviewQueueRow(Base):
    __tablename__ = "push_review_queue"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    key: Mapped[str] = mapped_column(Text, nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(64), default="llm_observation")
    status: Mapped[str] = mapped_column(String(32), default="pending")
    crystal_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    source_query_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SpendBudgetRow(Base):
    """spend_budgets — the tenant-owned budget SUBSTRATE (S4, 2026-07-08;
    docs/GAP_ENGINE_AND_LEARN_REDESIGN.md). One row = one cap for one
    spend FUNCTION ('auto_research' first; shadow_critic, gap_fill,
    convergence_scan migrate later), optionally narrowed to one operator
    (team seats — F1). cap_micro_usd=0 or no row = the function is OFF
    for auto paths (manual-by-default, ratified B-1). Enforcement reads
    the llm_calls ledger by origin — the ledger IS the meter. Platform
    tier caps remain the ceiling; these allocate WITHIN them."""

    __tablename__ = "spend_budgets"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    function: Mapped[str] = mapped_column(String(64), nullable=False)
    operator_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    period: Mapped[str] = mapped_column(String(16), default="monthly")
    cap_micro_usd: Mapped[int] = mapped_column(BigInteger, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    __table_args__ = (
        UniqueConstraint(
            "customer_id", "function", "operator_id",
            name="uq_spend_budgets_scope",
        ),
    )


class KnowledgeGapRow(Base):
    __tablename__ = "knowledge_gaps"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    domain: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    subject: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    missing: Mapped[str] = mapped_column(Text, nullable=False)
    # S3 provenance (2026-07-08): full sparse key + the query that missed.
    full_key: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    triggering_query: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # S4 (2026-07-08): who can close this gap — researchable (agent, web
    # tools) | workable (agent, by doing) | needs_document (human only).
    # NULL = pre-S4 row (sweep treats as researchable for continuity).
    disposition: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    priority: Mapped[str] = mapped_column(String(32), default="medium")
    status: Mapped[str] = mapped_column(String(32), default="open")
    source: Mapped[str] = mapped_column(String(64), default="llm_observation")
    filled_by_crystal_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


# ---------------------------------------------------------------------------
# knowledge_conflicts — Never-Idle Convergence, 2026-06-16.
#
# The first-class peer of knowledge_gaps for the convergence half of the
# accommodation thesis (docs/NEVER_IDLE_CONVERGENCE.md). A gap = "we lack
# knowledge about X"; a conflict = "we hold two facts about X that can't both
# be true." The contradiction-scan generator writes `open` rows when its
# CONTRADICTS discriminator fires over subject-adjacent facts.
#
# Mirrors KnowledgeGapRow's columns + lifecycle. The deltas are intrinsic to a
# conflict being about a PAIR: two fact ids + two crystal ids (SOFT pointers,
# no FK — REPLACE deletes facts/crystals, the citations/shard_events
# precedent), two claim snapshots (so the row reads without joins and survives
# the underlying facts changing/being deleted), two provenance strings, and a
# `pair_key` idempotence hash. status/resolution are String (not enum) so a
# new lifecycle/verb lands without a migration (the codebase's String-over-
# enum convention).
#
# IDEMPOTENCE (D4): unique on (customer_id, pair_key) — a re-scan can't write a
# duplicate, and the unique index is the robust backstop behind the
# generator's pre-check (the shard_events ux_*_idempotent precedent). pair_key
# folds in a hash of both claim texts, so a fact whose claim CHANGED yields a
# new pair_key and is re-evaluated; terminal rows (resolved/dismissed) keep
# their pair_key and are never re-surfaced — the loop provably quiesces.
#
# Local default stores create this free via store.init()'s create-missing-
# tables; the Alembic-managed dev DB gets it via the knowledge_conflicts
# migration (down_revision e1a3c5b7d2f4).
# ---------------------------------------------------------------------------


class KnowledgeConflictRow(Base):
    __tablename__ = "knowledge_conflicts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # The two conflicting facts + their crystals. Soft pointers (plain
    # columns, no FK) — REPLACE semantics delete facts/crystals.
    fact_a_id: Mapped[str] = mapped_column(String(64), nullable=False)
    fact_b_id: Mapped[str] = mapped_column(String(64), nullable=False)
    crystal_a_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    crystal_b_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # The sparse-key Subject / region where the two facts collide.
    subject: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)

    # Claim snapshots captured at detection time (readable without joins;
    # survive the underlying facts changing or being deleted).
    claim_a: Mapped[str] = mapped_column(Text, nullable=False)
    claim_b: Mapped[str] = mapped_column(Text, nullable=False)

    # Per-side provenance, human-facing ("source_kind @ source_path").
    provenance_a: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    provenance_b: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    detector: Mapped[str] = mapped_column(
        String(64), default="contradiction_scan",
        server_default="contradiction_scan",
    )
    # open | resolved | dismissed.
    status: Mapped[str] = mapped_column(
        String(32), default="open", server_default="open"
    )
    # NULL while open; qualified | superseded | blacklisted | dismissed.
    resolution: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    # Idempotence key (D4): hash of sorted (fact_a_id, fact_b_id) + hash of
    # both claim texts.
    pair_key: Mapped[str] = mapped_column(String(128), nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        # Inspector / backlog read: "open conflicts for this customer."
        Index(
            "ix_knowledge_conflicts_customer_status",
            "customer_id", "status",
        ),
        # Idempotence: one (customer, pair_key) row, ever.
        Index(
            "ux_knowledge_conflicts_pair_key",
            "customer_id", "pair_key",
            unique=True,
        ),
    )


class CognitionTaskRow(Base):
    __tablename__ = "cognition_tasks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    task_type: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    priority: Mapped[str] = mapped_column(String(32), default="background")
    status: Mapped[str] = mapped_column(String(32), default="pending")
    result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    result_crystal_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    source_query_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


# ---------------------------------------------------------------------------
# MCR Tables (Phase 8.5 — Multiple Critique of Reasoning)
# ---------------------------------------------------------------------------

class ReasoningTraceRow(Base):
    """MCR artifact 1 — the agent's structured self-report of how it
    produced a response. See Phase 8.5 P0.34–P0.40. Soft-linked back
    to query_logs via (customer_id, sequence_id, turn_index); FK to
    query_logs.id when known.
    """
    __tablename__ = "reasoning_traces"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False, index=True
    )

    sequence_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )
    turn_index: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )

    query_log_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("query_logs.id"), nullable=True
    )

    events: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, default=list, nullable=False
    )
    crystals_used: Mapped[list[str]] = mapped_column(
        JSON, default=list, nullable=False
    )
    tool_calls: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, default=list, nullable=False
    )
    inferences: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, default=list, nullable=False
    )
    borders_crossed: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, default=list, nullable=False
    )
    gaps_felt: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, default=list, nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    __table_args__ = (
        Index(
            "ix_reasoning_traces_sequence",
            "customer_id", "sequence_id", "turn_index",
        ),
        Index(
            "ix_reasoning_traces_customer_created",
            "customer_id", "created_at",
        ),
    )


class CritiqueRow(Base):
    """MCR artifact 2 — the structured output of a single critic
    reviewing a single reasoning trace. Soft pointer + soft-join key
    per P0.35; critic identity in two columns per P0.36.
    """
    __tablename__ = "critiques"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False, index=True
    )

    trace_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )
    sequence_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )
    turn_index: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )

    critic_role: Mapped[str] = mapped_column(
        String(32), nullable=False
    )
    critic_model: Mapped[str] = mapped_column(
        String(128), nullable=False
    )

    observations: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, default=list, nullable=False
    )
    summary_text: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )
    total_action_items: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    __table_args__ = (
        Index(
            "ix_critiques_sequence",
            "customer_id", "sequence_id", "turn_index",
        ),
        Index("ix_critiques_trace", "trace_id"),
        Index(
            "ix_critiques_role_created",
            "customer_id", "critic_role", "created_at",
        ),
    )


class ActionItemRow(Base):
    """MCR artifact 3 — a proposed next action emerging from a critique.
    Hard FK to critiques.id per P0.35. Lifecycle (P0.40):
    pending → {promoted, deferred, dropped, acted}.
    """
    __tablename__ = "action_items"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    critique_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("critiques.id"), nullable=False, index=True
    )
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False, index=True
    )

    action_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )
    content: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    critic_confidence: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending",
        server_default="pending", index=True
    )
    metacog_decision_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    acted_artifact_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    __table_args__ = (
        Index(
            "ix_action_items_customer_status",
            "customer_id", "status",
        ),
        Index(
            "ix_action_items_type_status",
            "action_type", "status",
        ),
    )


# ---------------------------------------------------------------------------
# Metacognitive Tables (Phase 10A — MCR §4.4 + §4.5)
# ---------------------------------------------------------------------------
#
# Two new tables computed by the metacognitive layer per
# `docs/MCR_FRAMEWORK.md` §4.4 (item alignment) and §4.5
# (critique synthesis). Schema decisions (P0.71 + P0.72) lock the
# shape for Phase 10A.
#
# These tables are READ by humans + Phase 10.5's substrate review
# surface, and WRITTEN by `metacognition.engine.compute_alignment_and_
# synthesis_for_trace`. Per D-MCR-15 (harness boundary), the
# metacognitive layer's only runtime action is computing these
# rows and transitioning action_items.status — it never modifies
# the harness.
# ---------------------------------------------------------------------------


class ItemAlignmentRow(Base):
    """MCR artifact 4 — alignment class of one action item against
    items from other critics for the same trace.

    Per MCR doc §4.4: one row per (trace, focus_item). The schema
    choice (P0.71) follows §6's decision loop: walk action_items for
    a trace, look up the alignment record for each, decide. One row
    per item makes that O(1) per item.

    `alignment_class` values (P0.40-aligned vocabulary extension):
      same_action          — substantively the same proposal as
                             items from at least one other critic
      similar_action       — same action_type with similar but not
                             identical content
      divergent_action     — proposed by only one critic, or by
                             multiple critics with unrelated content
      contradictory_action — proposed in direct tension with another
                             critic's item (e.g. two edit_proposal
                             items with same crystal_id but
                             different proposed_change)

    Phase 10A v1 algorithm (P0.73): pure-function `classify_pair`
    in `metacognition/alignment.py`. Phase 10B may refine with
    semantic similarity or per-action-type contradiction rules.

    The `paired_item_ids` JSON list records WHICH items from other
    critics this row's classification refers to. Empty list for
    `divergent_action` items proposed by a single critic.
    """
    __tablename__ = "item_alignments"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False, index=True
    )

    # Soft pointer to reasoning_traces.id (matches CritiqueRow's
    # convention). Indexed for the "alignments for trace X" query.
    trace_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )

    # Hard FK to the action item this alignment is FOR. Mirrors the
    # ActionItemRow→CritiqueRow hard-FK pattern (P0.35).
    focus_item_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("action_items.id"), nullable=False
    )

    # The alignment class. Validated at the Pydantic layer; String
    # column so a future fifth class (or refinement) can land
    # without an Alembic step.
    alignment_class: Mapped[str] = mapped_column(
        String(32), nullable=False
    )

    # action_items.id list of the OTHER critics' items this row's
    # focus item aligned with. Empty for solo `divergent_action`.
    # JSON list so the read pattern doesn't need a join table.
    paired_item_ids: Mapped[list[str]] = mapped_column(
        JSON, default=list, nullable=False
    )

    # Optional: classification confidence. Phase 10A's v1 algorithm
    # is rule-based + deterministic, so confidence is always 1.0
    # when set (or NULL). Phase 10B may use this for a learned
    # classifier.
    confidence: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )

    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    __table_args__ = (
        # Primary read: "all alignments for this trace."
        Index(
            "ix_item_alignments_trace",
            "trace_id",
        ),
        # Per-item lookup: "what's the alignment for THIS item?"
        Index(
            "ix_item_alignments_focus_item",
            "focus_item_id",
        ),
    )


class CritiqueSynthesisRow(Base):
    """MCR artifact 5 — the metacognitive layer's review record for
    a trace's critiques.

    Per MCR doc §4.5: one row per (trace, review-window) — a trace
    may be re-reviewed later as critic calibrations shift, producing
    a NEW row, not an update. Phase 10A v1 algorithm (P0.74) walks
    the trace's action_items, classifies each via the alignment
    row, and assigns to promoted/deferred/dropped buckets per the
    locked rules.

    `promoted_item_ids` / `deferred_item_ids` / `dropped_item_ids`:
    JSON lists of action_items.id values. Mutually exclusive within
    a single synthesis row.

    `promotion_rationales`: JSON dict mapping action_item.id → short
    string explaining the decision (e.g. "both critics agreed",
    "shadow solo proposal"). The metacognitive layer's audit trail.

    `critic_calibration_updates`: empty for Phase 10A; Phase 10B's
    calibration layer populates it. Schema-forward-compatible.

    `cross_trace_patterns`: empty for Phase 10A; future work.
    Schema-forward-compatible.

    No SQL UNIQUE on (trace_id, review_window_start) — re-syntheses
    are allowed. The audit trail is preserved by appending rows.
    """
    __tablename__ = "critique_syntheses"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False, index=True
    )

    # Soft pointer to reasoning_traces.id.
    trace_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )

    # Review window. For Phase 10A both fields default to
    # created_at; Phase 10B's scheduler may set wider windows when
    # re-reviewing accumulated traces.
    review_window_start: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    review_window_end: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # The three decision buckets. Stored as JSON lists of
    # action_item.id. Mutually exclusive per synthesis row.
    promoted_item_ids: Mapped[list[str]] = mapped_column(
        JSON, default=list, nullable=False
    )
    deferred_item_ids: Mapped[list[str]] = mapped_column(
        JSON, default=list, nullable=False
    )
    dropped_item_ids: Mapped[list[str]] = mapped_column(
        JSON, default=list, nullable=False
    )

    # Audit trail: per-item rationale strings. JSON dict keyed by
    # action_item.id.
    promotion_rationales: Mapped[dict[str, str]] = mapped_column(
        JSON, default=dict, nullable=False
    )

    # Placeholders for Phase 10B (calibration) and future
    # (cross-trace patterns). Empty for Phase 10A; included in the
    # schema so Phase 10B doesn't need an Alembic step.
    critic_calibration_updates: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, default=list, nullable=False
    )
    cross_trace_patterns: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, default=list, nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    __table_args__ = (
        # Primary read: "syntheses for this trace, newest first."
        Index(
            "ix_critique_syntheses_trace_created",
            "trace_id", "created_at",
        ),
        # Per-customer chronological scan for the substrate review
        # surface (Phase 10.5).
        Index(
            "ix_critique_syntheses_customer_created",
            "customer_id", "created_at",
        ),
    )


# ---------------------------------------------------------------------------
# Critic Calibration (Phase 10B — MCR §7)
# ---------------------------------------------------------------------------
#
# One row per (customer_id, critic_role, critic_model). The metacognitive
# layer's synthesis step (Phase 10A) calls `update_calibrations_from_
# synthesis` after writing a synthesis row; that helper upserts these
# counters per critic identity.
#
# Phase 10B does NOT use these counters in the promotion decision
# (P0.74's rules from Phase 10A unchanged). The counters are written
# for future use — Phase 11+ may add drop-on-low-trust-critic logic
# that reads these rows. Cold-start (§11 Q6) = "row doesn't exist"
# per P0.81.
#
# Composite UNIQUE on (customer_id, critic_role, critic_model) lets
# `upsert_critic_calibration` use select-then-update-or-insert without
# race concerns inside a single-threaded worker cycle.
# ---------------------------------------------------------------------------


class CriticCalibrationRow(Base):
    """MCR artifact 6 — running estimates per critic identity per
    customer. Per MCR §7 "Critic calibration."
    """
    __tablename__ = "critic_calibrations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False, index=True
    )

    # Critic identity (P0.36 pattern from Phase 8.5 critiques).
    critic_role: Mapped[str] = mapped_column(
        String(32), nullable=False
    )
    critic_model: Mapped[str] = mapped_column(
        String(128), nullable=False
    )

    # Running counters. Incremented by
    # `update_calibrations_from_synthesis` after each synthesis row
    # is written. total_proposals = promoted + deferred + dropped
    # (Phase 10A produces no dropped, so dropped_count stays 0 in
    # 10B; Phase 11+ may populate).
    total_proposals: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )
    promoted_count: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )
    deferred_count: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )
    dropped_count: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )

    # When this calibration was last touched by a synthesis.
    # Useful for the calibration surfacing scan ("which critics
    # haven't been heard from recently?").
    last_synthesis_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow,
        nullable=False
    )

    __table_args__ = (
        # Upsert key: exactly one row per critic identity per customer.
        Index(
            "ux_critic_calibrations_identity",
            "customer_id", "critic_role", "critic_model",
            unique=True,
        ),
    )


# ---------------------------------------------------------------------------
# agent_tasks — the coding-agent daemon's work queue (2026-06-11).
#
# "Database tables ARE the message queues" (the cognition_tasks pattern,
# applied to coding work). Producers: the CRYS CLI (`--queue`) and the
# agent itself (the guarded `queue_task` tool — queueing IS approving a
# future auto-approved headless run, so the guard prompts). Consumer:
# the daemon (`python -m crystal_code --daemon`), which claims oldest-
# queued, executes the F8 composed background run (branch-quarantined,
# shell/browser denied, ground-truth verify), and writes the report
# back to the row.
#
# Local default stores pick this table up free via store.init()'s
# create-missing-tables; the Alembic-managed dev DB needs a migration
# (alembic revision --autogenerate -m "agent_tasks").
# ---------------------------------------------------------------------------


class AgentTaskRow(Base):
    """One queued/running/finished coding-agent background task."""
    __tablename__ = "agent_tasks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False, index=True
    )

    # What to do and where. project_dir is machine-local by nature —
    # the queue is meaningful only to daemons on the machine that
    # enqueued it (documented, not enforced).
    project_dir: Mapped[str] = mapped_column(Text, nullable=False)
    task: Mapped[str] = mapped_column(Text, nullable=False)
    branch: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    # queued | running | done | failed. The daemon claims by flipping
    # queued -> running (oldest first); on startup it marks any stale
    # 'running' rows failed rather than silently re-running half-done
    # git work.
    status: Mapped[str] = mapped_column(
        String(16), default="queued", server_default="queued",
        nullable=False, index=True,
    )
    # 'cli' (user enqueued via --queue) or 'agent' (the queue_task tool).
    source: Mapped[str] = mapped_column(
        String(16), default="cli", server_default="cli", nullable=False
    )

    # Scheduling. run_at NULL = ASAP. The daemon claims the oldest DUE
    # task (run_at IS NULL OR run_at <= now). Times are local machine
    # time — the queue is machine-local by nature (see project_dir).
    run_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Recurrence. NULL = one-shot. For recurring tasks, run_at is the
    # series ANCHOR: occurrences fire at anchor + k*recur_seconds,
    # fixed-rate against the wall clock (never finish-time + interval —
    # "daily at 09:00" must mean 09:00). On completion the daemon
    # enqueues the next FUTURE occurrence as a NEW row (immutable
    # history; lineage via parent_task_id), skipping any occurrences
    # missed while the daemon was down — schedules never pile up.
    recur_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    parent_task_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True, index=True
    )
    # Consecutive failures across the SERIES (carried to each child row,
    # reset on success). At the cap the daemon parks the series instead
    # of recurring — silent infinite failure burn is the failure mode
    # this prevents.
    series_failures: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )

    # Outcome. report is the tail of the run's output; log_path points
    # at the full per-task log under ~/.crystal-code/tasks/.
    report: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    log_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


# ---------------------------------------------------------------------------
# crystal_contributions — Foundation F3 promotion provenance (2026-06-14).
#
# When operator-private crystals are promoted (merged up) to the team tier,
# ONE survivor crystal remains and the rest are superseded. This table
# captures, at merge time, WHO contributed each source crystal and the
# credit share reserved for them — the forward-reference to G4's shard
# ledger (capture-at-merge is cheap; reconstruct-later is impossible, so the
# slots are reserved even though there is nothing to pay out yet).
#
# Local default stores create this free via store.init()'s create-missing-
# tables; the Alembic-managed dev DB gets it via the crystal_contributions
# migration (down_revision b2e7c9a4f1d3).
# ---------------------------------------------------------------------------


class CrystalContributionRow(Base):
    """One contributor's provenance + reserved credit share on a merged
    team crystal (Foundation F3).

    Grain: one row per (merged_crystal, source_crystal), including the
    survivor's own original. `merged_crystal_id` IS a live FK (the survivor
    persists). `source_crystal_id` is a HISTORICAL id, NOT a FK —
    superseded non-survivor crystals are deleted at merge, so a constraint
    there would dangle. `share_basis_points` is the source crystal's
    reserved share; the shares for one merged crystal sum to 10000 (equal
    split in F3 v1). Integer units — credit is never a float (G4 contract).
    G4 aggregates by `contributor_operator_id` for an operator's total
    claim.
    """
    __tablename__ = "crystal_contributions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    merged_crystal_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("crystals.id"), nullable=False, index=True
    )
    # The operator who owned the source crystal. NULL only if a source was
    # unowned (F3 detect scans operator-owned crystals, so this is set in
    # practice; nullable for robustness + future tiers).
    contributor_operator_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("operators.id"), nullable=True
    )
    # Historical id of a contributing source crystal (the survivor's own id,
    # or a now-deleted non-survivor). Plain column, no FK — see class docstring.
    source_crystal_id: Mapped[str] = mapped_column(String(64), nullable=False)
    share_basis_points: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    __table_args__ = (
        Index("ix_crystal_contributions_merged", "merged_crystal_id"),
        Index(
            "ix_crystal_contributions_contributor",
            "contributor_operator_id",
        ),
    )


# ---------------------------------------------------------------------------
# Foundation F4 — session registry (surface consolidation, 2026-06-14).
#
# A live registry every surface (CRYS terminal, future Inspector) writes to
# and the Inspector reads — the unified-surfaces law made real ("see CRYS
# activity in the Inspector"). A session heartbeats its status +
# current_action; LIVENESS IS INFERRED FROM STALENESS, never self-reported
# (a crashed agent can't report its own crash), so a row stale beyond the
# threshold is presumed crashed and its dependencies orphaned — the same
# logic as the daemon's stale window (coding-agent daemon.py), generalized
# to a DB row because sessions surface server-side across machines.
#
# Local default stores create these free via store.init()'s create-missing-
# tables; the Alembic-managed dev DB gets them via the agent_sessions
# migration (down_revision d5f1a2b3c4e6).
# ---------------------------------------------------------------------------


class AgentSessionRow(Base):
    """One agent session (CRYS run) registered against a team (Foundation F4).

    Carries team_id + operator_id so the Inspector can scope by team and
    filter by operator. `status` + `current_action` are self-reported on
    every state transition + on a timer; `last_heartbeat_at` is the liveness
    signal (staleness → presumed crashed, derived at read time and
    materialized by the sweep). `awaiting_payload` is a G2 forward-ref (the
    approval-gate request the control plane will relay); `cost_usd_cumulative`
    is a G3 forward-ref (integer micro-USD — 1e-6 USD — since money is never a
    float; unwired in F4). `parent_session_id` threads subagent lineage.
    """
    __tablename__ = "agent_sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    team_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False
    )
    # The operator running the session. NULL for a team-key (root) session
    # with no individual operator attached.
    operator_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("operators.id"), nullable=True
    )

    host: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    pid: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    project_dir: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    model: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    # Self-reported lifecycle: starting | running | awaiting_approval | idle |
    # exited | crashed. String (not enum) so a future state lands without a
    # migration. 'exited'/'crashed' are terminal (the staleness derivation
    # leaves them alone).
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="starting", server_default="starting"
    )
    current_action: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # G2 forward-ref: the human-readable pending-approval payload the control
    # plane will surface + relay a signed decision for. NULL until awaiting.
    awaiting_payload: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSON, nullable=True
    )
    # Subagent lineage (a session spawned by another session). NULL for a
    # top-level session.
    parent_session_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True, index=True
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    # The liveness signal. Stale-beyond-threshold ⇒ presumed crashed.
    last_heartbeat_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    # G3 forward-ref: integer micro-USD (1e-6 USD). Money is never a float;
    # G3's cost choke point wires this. Default 0, unwired in F4.
    cost_usd_cumulative: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )

    __table_args__ = (
        Index("ix_agent_sessions_team", "team_id", "last_heartbeat_at"),
    )


class SessionDependencyRow(Base):
    """A resource a session spawned (Foundation F4).

    kind ∈ mcp_server | subprocess | browser | queued_task | pip_env. The
    session registers these as they spawn (it knows their PIDs), so a crashed
    session's dependencies can be presumed orphaned (the staleness sweep flips
    active deps of a crashed session to 'orphaned'). `session_id` is a plain
    indexed column (no FK) matching the crystal_contributions precedent —
    SQLite doesn't enforce FKs and the join is application-level.
    """
    __tablename__ = "session_dependencies"

    dependency_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    descriptor: Mapped[str] = mapped_column(
        Text, nullable=False, default="", server_default=""
    )
    pid: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # active | exited | orphaned.
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="active", server_default="active"
    )
    spawned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    __table_args__ = (
        Index("ix_session_dependencies_session", "session_id"),
    )


# ---------------------------------------------------------------------------
# web_search_logs — launch-prep sweep, 2026-07-02.
#
# One row per web search the system runs (agent tool or cognition research
# step). This is the goldmine's raw side: the queries are unusually
# high-intent (generated to fill identified knowledge holes), and the
# derivative chain — which results fed an answer, what got crystallized,
# whether those crystals later earned citations or conflicts — joins to this
# table by URL through crystal provenance (source_kind=web_search_result).
# Results are stored WITHOUT extracted page content (title/url/snippet
# only) — the interaction structure is the asset, not the page text.
# Soft pointers, no FKs, per the F4 precedent.
# ---------------------------------------------------------------------------

class WebSearchLogRow(Base):
    __tablename__ = "web_search_logs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    query: Mapped[str] = mapped_column(Text, nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    n_results: Mapped[int] = mapped_column(Integer, default=0)

    # [{title, url, snippet}] — content deliberately excluded (see above).
    results: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    # "tool" (agent/cognition registry dispatch) for now; room for more
    # origins without a migration.
    origin: Mapped[str] = mapped_column(String(16), default="tool")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# ---------------------------------------------------------------------------
# citations — Growth G1 (trust + the metering rail), 2026-06-15.
#
# One row per cited claim in a response turn. The proxy's post-response step
# parses [[cc:N]] markers out of the model's answer, grounds each against the
# crystal it cites, and records the result here. G1 v1 cites the PRIMARY
# injected crystal only, so a grounded turn writes ≤1 row; the table is shaped
# for multi-source from the start.
#
# This is the RAW per-claim record. G4's shard ledger (append-only
# shard_events) is what mints credit, idempotent on (interaction, crystal) —
# so NO uniqueness here; a turn may legitimately cite one crystal in several
# claims, and G4 dedupes when it pays out. `grounded` gates credit: only
# grounded=True citations are load-bearing (a cited-but-ungrounded span is a
# spurious citation — recorded for telemetry, never paid).
#
# `crystal_id` + `query_log_id` are SOFT pointers (plain columns, no FK,
# app-level join) — the crystal_contributions / session precedent. Under
# REPLACE semantics a cited crystal can later be deleted; a FK would dangle.
# `crystal_version` pins content_hash for the audit trail.
#
# Local default stores create this free via store.init()'s create-missing-
# tables; the Alembic-managed dev DB gets it via the citations migration
# (down_revision e7a2c9d4b1f8).
# ---------------------------------------------------------------------------


class CitationRow(Base):
    """One cited (and grounding-checked) claim from a response turn (G1)."""
    __tablename__ = "citations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False, index=True
    )
    # Soft pointer to the interaction (query_logs.id). Plain column, no FK —
    # follows the F4 "no FK" migration style. NULL if the turn had no query
    # log id to hand.
    query_log_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # The cited crystal — HISTORICAL id, plain column (REPLACE deletes
    # crystals; a FK would dangle). crystal_version pins content_hash.
    crystal_id: Mapped[str] = mapped_column(String(64), nullable=False)
    crystal_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # The handle the model cited (the N in [[cc:N]]).
    handle: Mapped[str] = mapped_column(
        String(16), nullable=False, default="", server_default=""
    )
    # The claim span the citation was attached to (best-effort; for audit +
    # a future entailment-grade grounding pass).
    claim_span: Mapped[str] = mapped_column(
        Text, nullable=False, default="", server_default=""
    )
    # Grounding cosine (claim ↔ cited crystal content) + whether it cleared
    # the threshold. Only grounded=True citations accrue G4 credit.
    grounding_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    grounded: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    __table_args__ = (
        Index("ix_citations_customer_created", "customer_id", "created_at"),
        Index("ix_citations_query_log", "query_log_id"),
        # G4 aggregation: sum credit per (customer, crystal).
        Index("ix_citations_crystal", "customer_id", "crystal_id"),
    )


# ---------------------------------------------------------------------------
# control_commands — Growth G2 (control plane), 2026-06-15.
#
# The outbound-poll command channel: an operator's decision (approve/deny an
# approval gate, or terminate a session / dependency) is written here; the
# agent POLLS for commands targeting its session and acts. Nothing connects
# INBOUND to the agent (NAT-safe). The decision is SIGNED by the operator's
# key (signature/nonce/signed_at) and the AGENT verifies it before acting —
# the server is a courier that cannot forge a decision. `session_id` is a soft
# pointer (the session/agent precedent). First-wins is a compare-and-set on
# `status` (pending→consumed); F4 staleness voids pending commands of a
# crashed session.
# ---------------------------------------------------------------------------


class ControlCommandRow(Base):
    """One control-plane command targeting a session (Growth G2)."""
    __tablename__ = "control_commands"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # Soft pointer to agent_sessions.session_id (plain column, no FK — the
    # session/contribution precedent). Indexed for the agent's poll.
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False, index=True
    )
    # The approval request this answers (echoes the session's awaiting_payload
    # request_id). For terminate commands, a fresh id. First-wins is keyed on
    # this server-side via the consumed flag; the agent ignores decisions for
    # a request it is no longer awaiting.
    request_id: Mapped[str] = mapped_column(String(64), nullable=False)
    # approval_decision | terminate | terminate_dependency.
    command_type: Mapped[str] = mapped_column(String(32), nullable=False)
    # approve | deny (for approval_decision); NULL for terminate.
    decision: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    # The dependency to kill (for terminate_dependency); NULL otherwise.
    dependency_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    payload: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # Signed authorization (the agent verifies against the operator's pinned
    # public key). base64 detached signature over the canonical
    # {session_id, request_id, decision, nonce, signed_at} payload.
    signature: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    nonce: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    signed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    issued_by_operator_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )

    # pending | consumed | voided. The agent claims by flipping
    # pending→consumed (compare-and-set = first-wins).
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending", server_default="pending"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    consumed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        Index("ix_control_commands_session_status", "session_id", "status"),
    )


# ---------------------------------------------------------------------------
# llm_calls — Growth G3 (cost accounting), 2026-06-15.
#
# The single cost row every model invocation emits via record_llm_call(). Cost
# is computed from a per-model price table in config (prices move —
# externalized) and stored as INTEGER micro-USD (1e-6 USD; money is never a
# float). Attribution: session + parent_session (rollup) + team + operator +
# origin. Views are GROUP BYs (all-time / daily / weekly per agent / operator /
# team); average = per-agent (D6). Budgets read these and auto-pause via G2.
# ---------------------------------------------------------------------------


class LlmCallRow(Base):
    """One model invocation's cost + attribution (Growth G3)."""
    __tablename__ = "llm_calls"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False, index=True
    )
    # Soft pointers (a call may have no session, e.g. a bare proxy request).
    session_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    parent_session_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )
    operator_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # interactive | task | cognition | subagent | depth | metacognition |
    # inline_research. String (not enum) so a new origin lands without a
    # migration.
    origin: Mapped[str] = mapped_column(
        String(32), nullable=False, default="interactive",
        server_default="interactive",
    )
    model: Mapped[str] = mapped_column(String(128), nullable=False)

    input_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    output_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    cache_creation_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    cache_read_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    # Cost in INTEGER micro-USD (1e-6 USD). Money is never a float.
    computed_cost_micro_usd: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    # Billing dimension (E4, Accounts Phase B 2026-07-06): 'managed' =
    # served on Era's key, rebillable (SUM x markup); NULL = byok/internal
    # (not rebilled). Stamped PER CALL so mid-month inference_mode flips
    # stay accurate — the customer's current mode is never consulted at
    # rebill time. Orthogonal to `origin` (workload vs money dimensions).
    billing: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    __table_args__ = (
        Index("ix_llm_calls_team_created", "customer_id", "created_at"),
        Index("ix_llm_calls_session", "session_id"),
        Index("ix_llm_calls_operator_created", "operator_id", "created_at"),
    )


# ---------------------------------------------------------------------------
# shard_events / expert_authorizations — Growth G4 (marketplace), 2026-06-15.
#
# The append-only shard ledger. Metering is CITATIONS (the G1 rail): a *cited*
# crystal (grounding-gated, self-traffic excluded) accrues a shard — injection
# is cheap, citation means the model found it load-bearing, which kills
# key-stuffing. Never mutated; corrections are compensating entries
# (event_type='clawback'). IDEMPOTENT: unique on (interaction_id, crystal_id,
# event_type) so one interaction can't double-credit a crystal (a credit and
# its later clawback coexist via the distinct event_type). INTEGER shard
# units; balance = sum(shards_credited); spends are negative debits in the
# same ledger (closed-loop — subscription only, no cash-out). Shards are
# proportional claims on a bounded reward pool (D7 — DEFERRED placeholder;
# v1 credits a fixed weight per grounded citation). Convertibility stays OFF
# until metering survives adversarial traffic.
# ---------------------------------------------------------------------------


class ShardEventRow(Base):
    """One append-only shard-ledger event (Growth G4)."""
    __tablename__ = "shard_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # The expert/owner whose balance this moves (the crystal author /
    # contributor). NULL for events not tied to an operator.
    owner_operator_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True, index=True
    )
    # The crystal that earned (soft pointer — REPLACE deletes crystals). NULL
    # for spends/debits not tied to a crystal.
    crystal_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # The team whose traffic generated the credit (for consumer-diversity
    # checks). NULL for spends.
    consuming_team_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )
    # The interaction (query_log id) the credit derives from — the idempotency
    # anchor. NULL for spends.
    interaction_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # credit | debit | clawback.
    event_type: Mapped[str] = mapped_column(String(16), nullable=False)
    # citation | spend | correction (free-form provenance of the event).
    signal_type: Mapped[str] = mapped_column(
        String(32), nullable=False, default="citation", server_default="citation"
    )
    # The raw usefulness weight before pool apportionment (D7). Float is fine
    # — it is NOT money; shards_credited is the money-equivalent integer.
    raw_weight: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0, server_default="0"
    )
    # INTEGER shard units. Positive for credit, negative for debit/clawback.
    shards_credited: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    __table_args__ = (
        # Idempotency: one (interaction, crystal, event_type) credits once.
        # NULLs are distinct in SQLite/Postgres unique indexes, so multiple
        # spends (all-NULL keys) are allowed.
        Index(
            "ux_shard_events_idempotent",
            "interaction_id", "crystal_id", "event_type",
            unique=True,
        ),
        Index("ix_shard_events_owner", "owner_operator_id", "created_at"),
        Index("ix_shard_events_crystal", "crystal_id"),
    )


class ExpertAuthorizationRow(Base):
    """An operator vetted to author general crystals in a domain (Growth G4).

    The scope registry behind marketplace authoring: an expert is authorized
    per `general:<domain>`. Reputation/dispute rows are deferred; this is the
    minimal vetting substrate the team→general promotion gate checks.
    """
    __tablename__ = "expert_authorizations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    operator_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    team_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False
    )
    # The general:<domain> scope the operator may author into.
    domain: Mapped[str] = mapped_column(String(128), nullable=False)
    # active | revoked.
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="active", server_default="active"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    __table_args__ = (
        Index("ux_expert_auth_operator_domain", "operator_id", "domain", unique=True),
    )


# ---------------------------------------------------------------------------
# agent_events — Unify-Agents event stream, 2026-06-15.
#
# The append-only record of everything CRYS does, keyed by session_id and
# ordered by a per-session monotonic `seq`. This is the backbone the "Agents"
# Inspector surface, the unified interaction log, and cost rollups all read.
# CRYS is the product; this makes its every turn / tool call / delegated
# subagent / crystal / gap visible — nothing is proxy-only.
#
# APPEND-ONLY + a JSON `payload`: new event_types never need a migration.
# event_type is a free-form String (not an enum) for the same reason. All
# pointers are SOFT (plain columns, no FK) — session_id / parent_session_id
# follow the F4 session precedent; events outlive (and may predate the
# materialization of) the rows they reference, and REPLACE deletes crystals a
# crystal_written payload names. Rows are operational, not domain entities, so
# methods return plain dicts (the session/agent-task precedent).
#
# Recording is fail-safe and flows through SessionHandle.record_event
# (CRYS-side) like the heartbeat — it must never raise into or block a turn.
# `seq` is assigned MAX+1 per session: single-writer per session in practice
# (the REPL or the daemon owns its session), and append-only means a tie only
# affects ordering, broken by created_at then id.
#
# Local default stores get this free via store.init() create-missing-tables;
# the Alembic-managed dev DB gets it via the agent_events migration
# (down_revision d0f4b6c8e2a3).
# ---------------------------------------------------------------------------


class AgentEventRow(Base):
    """One append-only event in a CRYS session's activity stream."""
    __tablename__ = "agent_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # Soft pointer to agent_sessions.session_id (plain column, no FK).
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    # Denormalized team (customers.id) so the unified log + team rollups read
    # without a join. Soft/nullable — set by the recorder when known.
    team_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # Per-session monotonic order (MAX+1 at write). 0-based.
    seq: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    # Which turn this event belongs to (groups tool/subagent events into a
    # turn). NULL for lifecycle/queue events outside a turn.
    turn_index: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # The spawning session for subagent events — lets the timeline nest a
    # subagent's activity under the turn that delegated it.
    parent_session_id: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )
    # Free-form (no enum, so a new type lands without a migration). See the
    # taxonomy in docs/UNIFY_AGENTS_INSPECTOR.md.
    event_type: Mapped[str] = mapped_column(String(48), nullable=False)
    # Coarse grouping: lifecycle | turn | tool | subagent | knowledge | gap |
    # error. NULL allowed.
    phase: Mapped[Optional[str]] = mapped_column(String(24), nullable=True)
    # Short human line ("writing game.js", "delegating research: map module Y").
    label: Mapped[str] = mapped_column(
        Text, nullable=False, default="", server_default=""
    )
    # Event-specific structured detail.
    payload: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    # ok | error | denied | truncated (for events with an outcome). NULL else.
    status: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    # For wrapped work (tool call, subagent, turn).
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # For LLM-bearing events (turn, subagent).
    tokens_input: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    tokens_output: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # INTEGER micro-USD (1e-6 USD). Money is never a float — the G3 precedent.
    cost_micro_usd: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    __table_args__ = (
        # Timeline read + incremental poll (after_seq).
        Index("ix_agent_events_session_seq", "session_id", "seq"),
        # Unified interaction log + team rollups, newest-first.
        Index("ix_agent_events_team_created", "team_id", "created_at"),
        Index("ix_agent_events_type", "event_type"),
    )


# ---------------------------------------------------------------------------
# agent_conversations — CRYS session continuity (P5), 2026-06-16.
#
# Per-scope conversation persistence so context survives exit/relaunch. CRYS
# used to start every run with an empty `messages` and a fresh session id and
# never look back, so a relaunch in the same project had no memory of the
# prior conversation ("which file did I last work on?"). This table holds the
# resumable transcript keyed by a generic SCOPE.
#
# MODE-AGNOSTIC (coding is one mode among several): `conversation_key` is the
# scope, and it generalizes — the resolved project_dir for the CLI coding
# mode, a thread id for the future general/web mode (the Inspector chat
# playground becoming CRYS). Same persistence layer, different key, so the web
# playground inherits resume for free; `mode` tags which.
#
# DB-BACKED (not a local file): the local file the CLI could write doesn't
# exist server-side for the web playground, and doesn't follow a user across
# machines. The store IS the boundary (the F4 session precedent) — CRYS
# offline writes to the local default SQLite store, logged-in to the team DB,
# the web playground to the server DB. One mechanism, all three.
#
# ONE ROW PER SCOPE: unique (customer_id, conversation_key). The CLI reuses
# the project_dir key, so a relaunch resumes the SAME rolling conversation
# (upsert overwrites); the web mode uses unique thread ids for many threads.
# Soft customer_id (plain indexed column, no FK) — the recent-table
# convention (push_review / knowledge_gaps / knowledge_conflicts). transcript
# is the raw anthropic `messages` list (JSON); the CALLER caps it before
# writing (the store persists what it is given). Rows are operational state,
# not a domain entity, so the store methods return plain dicts (the
# agent_sessions / agent_tasks precedent).
#
# Local default stores create this free via store.init()'s create-missing-
# tables; the Alembic-managed dev DB gets it via the agent_conversations
# migration (down_revision f3b5d7c9e1a4).
# ---------------------------------------------------------------------------


class AgentConversationRow(Base):
    """One resumable conversation transcript, keyed by (customer, scope)."""
    __tablename__ = "agent_conversations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # The scope this conversation belongs to: a resolved project_dir (CLI
    # coding mode) or a thread id (future general/web mode). Long enough for
    # an absolute path (the source_path precedent).
    conversation_key: Mapped[str] = mapped_column(String(512), nullable=False)

    # Which CRYS mode owns it: 'coding' | 'general' | ... String (not enum)
    # so a new mode lands without a migration.
    mode: Mapped[str] = mapped_column(
        String(32), nullable=False, default="coding", server_default="coding"
    )
    # Optional human label (useful for named web threads; NULL for the CLI
    # project-scoped conversation).
    title: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)

    # The raw anthropic `messages` array (list of message dicts). Capped by
    # the caller before write.
    transcript: Mapped[list] = mapped_column(JSON, default=list)
    # Turns recorded so far (for the launch recap).
    turn_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    # Short recap of the last turn (for the launch line).
    last_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Mode-specific extras (e.g. coding mode's last_files). Generic JSON so
    # the core stays mode-agnostic.
    meta: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # The controlling model selected for this conversation (C6 — model
    # selection, 2026-06-17). Per-conversation STICKY model: the web client's
    # explicit choice is persisted here (last-writer-wins) and reused on later
    # turns from any device; NULL = no selection → fall back to the
    # CC_AGENT_MODEL house default → built-in DEFAULT_MODEL. Mirrors
    # AgentSessionRow.model (same String(128)). A typed column, NOT a key in
    # `meta`, so upsert_conversation's full-meta overwrite can never clobber it.
    model: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    __table_args__ = (
        # One resumable conversation per (customer, scope).
        Index(
            "ux_agent_conversations_scope",
            "customer_id", "conversation_key",
            unique=True,
        ),
        # "my recent conversations, newest first" (the future web thread list).
        Index(
            "ix_agent_conversations_customer_updated",
            "customer_id", "updated_at",
        ),
    )


class GroupRow(Base):
    """A named sub-team — P3, ratified 2026-07-02.

    Lightweight grant target: crystal_acls rows with principal_type
    'group' let every member read the crystal without touching its POSIX
    mode. Customer-scoped; names unique within a team so the CLI can say
    'share with backend' unambiguously.
    """
    __tablename__ = "groups"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    __table_args__ = (
        Index("ux_groups_customer_name", "customer_id", "name", unique=True),
    )


class GroupMemberRow(Base):
    """Membership edge for GroupRow (P3). Composite PK → idempotent adds
    collide instead of duplicating; removal is a single DELETE."""
    __tablename__ = "group_members"

    group_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("groups.id"), primary_key=True
    )
    operator_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("operators.id"), primary_key=True
    )

    __table_args__ = (
        Index("ix_group_members_operator", "operator_id"),
    )


class OAuthStateRow(Base):
    """Pending OAuth state nonces (F1 CSRF fix, 2026-07-03).

    One row per issued /auth-url state. The callback must present a state
    that exists here (single-use: consumed on read) and is younger than the
    TTL — otherwise the flow is rejected. Server-side persistence (not an
    in-process dict) so the check holds across multiple API instances.
    """

    __tablename__ = "oauth_states"

    state: Mapped[str] = mapped_column(String(128), primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )


class TaskKeyRow(Base):
    """Task-scoped API keys (Phase 3 G3, 2026-07-03, ratified).

    The ONLY credential a disposable box carries: acts as its tenant on the
    public chat proxy ONLY (never the SDK surface), metered in the ledger
    under session_id = task_id, budget-checked at the door, revocable, and
    expiring. One key per task. Hash at rest, same as customer keys.
    """

    __tablename__ = "task_keys"

    task_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    key_hash: Mapped[str] = mapped_column(
        String(128), nullable=False, unique=True, index=True,
    )
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.id"), nullable=False,
    )
    budget_micro_usd: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )
    revoked_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
    )
