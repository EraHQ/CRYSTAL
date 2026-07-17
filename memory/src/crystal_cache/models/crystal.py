"""Crystal + CrystalEdge entities — §6 of BUILD_PROPOSAL.md.

Research grounded: quality_tier is four-valued (not binary) per §3 merged
architecture. Added fields from §4 telemetry loop:
  - keyword_fingerprint (helpful vs hurtful diagnostic — see research §2.2)
  - cluster_tightness (measured at build)
  - parent_crystal_id (lineage across rebuilds)
  - diagnostic_tags (populated by diagnostic engine)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


QualityTier = Literal["whitelist", "neutral", "quarantine", "blacklist"]
# Lineage tag describing how this crystal came to exist.
#
# - "kmeans" / "hierarchical": legacy clustering build paths.
# - "manual": operator-authored.
# - "edited": came out of the diagnostic engine's edit loop.
# - "spawned": Phase 1.2 auto-split sibling (parent_crystal_id set).
# - "router": Phase 1.3 add_pair_for_customer fresh spawn
#   (parent_crystal_id is None; no lineage to a full top-1).
BuildMethod = Literal[
    "kmeans", "hierarchical", "manual", "edited", "spawned", "router",
    "content_chunk",  # Document content chunk (verbatim text, bypasses bonder)
]

# What kind of evidence the crystal carries.
#
# - "model_reasoning": a verified-correct prior answer (its `answer_value`
#   is the canonical short answer). On a high-cosine match, the
#   pipeline can serve `answer_value` directly without invoking the
#   upstream LLM — this is the cache-hit short-circuit.
# - "failed_reasoning": an imperative rule extracted from a wrong prior
#   attempt (one Haiku-reflection call per failure). Carries no
#   `answer_value`; injected with imperative voice as a constraint on
#   the next attempt.
# - "web_search_result": a deduplicated result row from an upstream
#   web search the model performed. Advisory reference material.
# - "code_execution_result": stdout / artifacts from upstream code
#   execution. Advisory reference material.
#
# Default "model_reasoning" preserves existing rows (created before the
# 0006 migration) as success crystals, which is correct — those crystals
# were authored from verified facts.
SourceKind = Literal[
    "model_reasoning",
    "failed_reasoning",
    "web_search_result",
    "code_execution_result",
    "document_chunk",  # Verbatim content chunk from document ingestion
]


class Crystal(BaseModel):
    id: str
    customer_id: Optional[str] = None  # None = general crystal (system-level knowledge)

    # The crystal's holographic memory: `Σ bind(P_i, A_i)` over every
    # pair written via add_pair_to_crystal. Used by the recall path
    # (recall_from_crystal unbinds against this with a query's HDC
    # projection to recover the answer). NOT used for routing — see
    # `routing_vector` below.
    summary_vector: list[float]  # 10k-dim (hot path; production will use a blob store ref)

    # Phase 6.3 (May 2026): the crystal's routing address.
    #
    # `Σ encode(prompt_i) @ P` accumulated alongside summary_vector on
    # every add_pair_to_crystal write. This is the prompt-side
    # superposition WITHOUT answer-binding — geometrically compatible
    # with `encode(query) @ P` so the M2 cosine routing primitive
    # actually works at scale.
    #
    # Why two vectors:
    #   - summary_vector is `Σ bind(P_i, A_i)` — a bundle of grating
    #     keys. Bipolar HDC algebra makes this near-orthogonal to its
    #     component prompt-projections, so cosine routing against it
    #     produces near-zero scores even for queries whose prompts
    #     WERE bound in. Validated empirically by Finding 16's smoke
    #     test (every query NO_MATCH on the rebuilt FAQ bank).
    #   - routing_vector is `Σ P_i` — a bundle of prompt-projections.
    #     Same vector geometry as `encode(query) @ P`, so cosine on
    #     related text lands at ~0.4-0.6 like the legacy bank shape.
    #
    # Both are persisted as RAW BUNDLES. Unit-norm enforcement is
    # read-side in VectorStore._ensure_loaded — same contract as
    # summary_vector (Hard Rule 16).
    #
    # Optional/nullable: pre-Phase-6.3 crystals (built before migration
    # 0014) have None here. VectorStore skips such rows at load time
    # (they're invisible to routing until backfilled via
    # scripts/backfill_routing_vectors.py). New crystals built via
    # add_pair_to_crystal AFTER migration 0014 get this populated
    # automatically.
    routing_vector: Optional[list[float]] = None

    # Native-dim (e.g. 768 for gtr-t5-base) embedding of this crystal's
    # canonical answer. Populated by import_bank via
    # encoder.encode_native(answer_text). Used as input to the bind/unbind
    # synthesis path — the routing decision table's SPREAD branch needs
    # the native answer embeddings of top-1 and top-2 to compute
    # bind(top1_ans, top2_ans) and decode that with bind-v1.
    #
    # Optional/nullable so existing crystals (created before the April 2026
    # schema change) don't break. A crystal without this field can still
    # be routed via M2 cosine on summary_vector — it just can't participate
    # in synthesis. The bank-import script populates this for new and
    # re-imported crystals.
    answer_embedding_native: Optional[list[float]] = None

    # Encoder geometry fingerprint (Phase 1.1 mitigations, April 2026).
    # Set by `BindCapableEncoder.fingerprint()` on the first bind-storage
    # write to this crystal. Re-checked on every subsequent write and at
    # recall time. Catches encoder geometry drift (different model, seed,
    # or dim) that would otherwise silently corrupt recovered vectors.
    #
    # None for pre-Phase-1.1 crystals (built via legacy import_bank.py
    # direct-upsert) and for crystals never written via add_pair_to_crystal.
    # Those still route fine via M2 cosine; they just can't safely
    # participate in bind-storage chain extension.
    encoder_fingerprint: Optional[str] = None

    # Maintenance
    decay_rate: float = 0.01
    fact_count: int = 0

    # Quality gate — produced by offline evaluator + shadow telemetry
    quality_tier: QualityTier = "quarantine"

    # Recall gate + birth attribution (2026-07-03, recall-gated memory).
    # recall_gated is the orthogonal "can this be USED at all" bit: True =
    # held out of recall until approved (human or a system_rules promotion
    # rule), independent of quality_tier which is only a signal. origin is
    # WHAT created the crystal ('direct' default vs 'background_worker'),
    # distinct from source_kind (KIND of evidence).
    recall_gated: bool = False
    origin: str = "direct"

    eval_helped_count: int = 0
    eval_hurt_count: int = 0
    live_shadow_helped_count: int = 0
    live_shadow_hurt_count: int = 0

    # Diagnostic fingerprint — research §2.2 finding
    keyword_fingerprint: list[str] = Field(default_factory=list)
    cluster_tightness: Optional[float] = None  # mean cos-to-centroid
    attribution_spread: Optional[float] = None  # std of projections

    # Optional human-readable summary of what this crystal is "about".
    # Populated by bank-construction writers when crystals are built from
    # verified facts. Used by CrystalReader for text-injection: if present,
    # it becomes the injection prefix; if absent, we fall back to the
    # keyword fingerprint.
    summary_text: Optional[str] = None

    # Provenance + cache-hit support (April 2026, GAIA fold-back).
    #
    # source_kind tags what kind of evidence this crystal carries
    # (success / failure rule / web result / code result). The
    # injection layer renders success-derived crystals as advisory
    # reference material and failure-derived crystals as imperative
    # constraints. Mixing voices regressed accuracy on the GAIA bench;
    # keeping them separate is load-bearing.
    #
    # answer_value carries the canonical short answer for cache-hit
    # short-circuiting on PERFECT-decision matches. When set on a
    # "model_reasoning" crystal, the pipeline can return this string
    # to the caller without invoking the upstream LLM. None for
    # failure rules and for the legacy bank (existing crystals were
    # imported with full answer text in summary_text but no separate
    # short-answer field).
    #
    # Both are optional/nullable; existing crystals (pre-0006) load
    # with source_kind defaulted to "model_reasoning" via the column
    # default and answer_value=None, so they participate normally in
    # routing and injection but never trigger the cache-hit path
    # (which requires a populated answer_value).
    source_kind: SourceKind = "model_reasoning"
    answer_value: Optional[str] = None

    # Lineage + construction
    build_method: BuildMethod = "kmeans"
    parent_crystal_id: Optional[str] = None  # set on split/merge

    # V2 source versioning (VS-D2). Populated for crystals ingested from a
    # file. Replace semantics (VS-D3, locked 2026-06-10): a changed source
    # DELETES its prior crystals and writes fresh ones — no is_current
    # flag, no stale crystals. See infrastructure/schema.py CrystalRow.
    source_path: Optional[str] = None
    # Gate D (C1/C2): canonical scheme-qualified location identity.
    source_uri: Optional[str] = None
    content_hash: Optional[str] = None
    source_modified_at: Optional[datetime] = None

    # Phase 3 (April 2026): crystal type registry.
    #
    # Every crystal carries a type id (e.g. 'general:legacy',
    # 'customer:medical_records'). The CrystalType registry row
    # for this id carries scope, capacity, autosplit policy, and
    # per-type threshold overrides.
    #
    # Default 'customer:legacy' lands any new crystal in the
    # back-compat bucket if the caller hasn't specified a type —
    # matches the migration 0012 server default. Phase 4 narrows
    # the customer-tier story when DSL-authored types take over.
    crystal_type: str = "customer:legacy"

    # Foundation F2 (POSIX permissions). A crystal is an owned resource:
    # an owner (the authoring operator), a group (the owning team), and
    # POSIX mode bits. Only the READ bits are consumed today — retrieval is
    # permission-checked at FactVectorStore.search() and VectorStore.search()
    # via infrastructure/permissions.can_read; write/execute bits are reserved
    # (the execute bit carries no crystal semantics, per the locked F2
    # axiom). All three are nullable/defaulted so today's crystals and
    # write paths are untouched: owner/group default None (the resolver
    # falls back to the owning tenant for group and treats general crystals
    # as world-shared), and mode defaults to 0o640 (owner rw, group r,
    # other none) — team-readable, matching "a team reads its own
    # crystals." Operator-authored writes set these explicitly in F2.2; the
    # scope tiers map to mode (operator-private 0o600, team 0o640, general
    # world-readable).
    owner_operator_id: Optional[str] = None
    group_team_id: Optional[str] = None
    mode: int = 0o640

    # Phase 6.3 follow-up #2 (migration 0016, May 2026): the
    # decomposer payload that established this crystal's "concept
    # identity" at spawn time.
    #
    # Populated ONLY on spawn-fresh, when add_pair_for_customer was
    # called with a wired decomposer that returned a payload. Bond
    # writes do NOT update this field — the first-bonded fact's
    # payload is the crystal's payload forever (option α from the
    # scope doc; see PHASE_6_3_FOLLOWUP_2_DECOMPOSER_BONDING_SCOPE.md
    # §Decision 4).
    #
    # Consumed by ThreeAxisBonder at bond time: when an incoming pair's
    # routing-vector cosine lands in the gray zone AND the per-fact-
    # prompt cosine doesn't fire either, the bonder compares the
    # incoming pair's freshly-decomposed payload against this stored
    # payload via concept-HV cosine. Agreement >= T_payload bonds;
    # disagreement spawns fresh.
    #
    # Nullable for three legitimate reasons:
    #   1. Pre-followup-2 crystals (built before migration 0016)
    #   2. Crystals spawned without a wired decomposer
    #   3. Crystals whose first-bond decomposer call raised
    #      DecomposerError (graceful degradation)
    # The bonder treats None as "no axis-3 signal, fall through to
    # conservative spawn in the gray zone."
    decomposer_payload: Optional[dict] = None

    # Populated by diagnostic engine
    diagnostic_tags: list[str] = Field(default_factory=list)
    # e.g. ["relational", "topical", "large", "drift-detected"]

    last_eval_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


CrystalEdgeType = Literal["co_queried", "sub_domain", "parent_child"]


class CrystalEdge(BaseModel):
    """Graph edge between two crystals — §6 of BUILD_PROPOSAL.md."""

    crystal_a_id: str
    crystal_b_id: str
    edge_type: CrystalEdgeType = "co_queried"
    weight: float = 0.0
    last_reinforced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
