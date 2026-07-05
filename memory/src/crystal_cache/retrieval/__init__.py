"""Retrieval layer — v2 incremental port.

v1's retrieval/ package was 24 files / ~278 KB covering both the V2
(legacy) and V3 pipelines. v2 ports a small subset incrementally:

  - Phase 6 Wave D (DONE): v3_push_pull, v3_signal_handler
    (proxy-mode push/pull protocol — what the LLM emits and how it
    gets processed)
  - Phase 7 Wave 7A (DONE): sparse_key, v3_routers, v3_navigation,
    v3_depth, v3_composer (the four V3 routers per D-A3 + composer
    per D-A4 — agent-reframe survivors that re-expose as agent tools
    in Phase 7.5)
  - Phase 7 Wave 7B (DONE): V2 legacy pipeline — pipeline, router,
    reader, recall, chain_resolver, match_classifier,
    concept_router, synthesis, composer (V2, distinct from
    v3_composer). Ported verbatim to make chat_proxy match v1
    behavior; retiring in Phase 9 once the agent owns retrieval
    orchestration. NOT ported: query_classifier (V3 routing logic;
    Phase 7.5 will decide) and v3_pipeline (DROPPED per agent
    reframe).
  - Phase 7 Wave 7F (DONE): Mem0 consolidation — `mem0_session.py`
    replaces v1's `v3_session_memory.py` + `v3_mem0.py` per D6 /
    P0.9. Only `v3_session_memory.py`'s behavior was actually used
    by v1's chat_completions; `v3_mem0.py` was dead code at runtime.
    Wave 7F also filled the three SDK 501 stubs (/v1/retrieve,
    /v1/learn, /v1/consolidate) and the chat_proxy + feedback
    endpoints — those changes live in endpoints/, not here.

This module exports the Wave D, Wave 7A, Wave 7B, and Wave 7F
retrieval-layer surfaces.

NAMING NOTE — TWO COMPOSERS
---------------------------
There are two composer-like things in this package, deliberately
kept distinct per the inventory in Wave 7B:

  - `v3_composer.Composer` (Wave 7A) — the V3 proxy-mode helper
    that merges per-router results into one injection string.
    Exported as `Composer` (the class name from v1).

  - `composer.{InstructionComposer,BayesianComposer,...}` (Wave 7B)
    — the V2 pluggable strategy classes that build sectioned
    injection text from a `ComposerContext`. Exported under their
    own class names; the abstract base is `ComposerStrategy`.

The two are NOT interchangeable; the V3 router pipeline calls the
former, the V2 retrieve_and_inject pipeline calls the latter.
Keep them on separate import paths to avoid conflation hazards.
"""
from .chain_resolver import ChainResolver
from .citations import (
    CITE_INSTRUCTION,
    CitationSource,
    assign_handles,
    build_primary_citation,
    extract_claim_span,
    map_citations,
    parse_citations,
    render_sources_footer,
    rewrite_markers,
)
from .citation_grounding import (
    CITATION_GROUNDING_THRESHOLD,
    ground_citations,
)
from .composer import (
    BayesianComposer,
    ComposerContext,
    ComposerStrategy,
    InstructionComposer,
    get_composer,
    register_composer,
)
from .concept_router import ConceptRouteOutcome, ConceptRouter
from .match_classifier import (
    MatchClassifier,
    MatchType,
    RoutingDecision,
    RoutingResult,
)
from .mem0_session import (
    add_conversation_turn,
    get_mem0,
    init_mem0,
    search_session_context,
)
from .pipeline import (
    DEFAULT_TOP_K,
    RetrievalOutcome,
    retrieve_and_inject,
)
from .reader import CrystalContext, CrystalReader, Voicing
from .recall import RecalledFact, recall_from_crystal
from .router import CrystalRouter
from .sparse_key import (
    DELIMITER,
    SparseKey,
    common_prefix,
    common_suffix,
    contains_segment,
    detect_gaps,
    format_key,
    is_structured_key,
    matches,
    parse_key,
    scan_keys,
    validate_key,
)
from .synthesis import synthesize_joint_statement
from .v3_composer import (
    Composer,
    determine_injection_method,
    determine_match_type,
)
from .v3_depth import DepthResult, DepthRouter
from .v3_navigation import NavigationResult, NavigationRouter
from .v3_push_pull import (
    CRYSTAL_TOOL_NAMES,
    CRYSTAL_TOOLS,
    ParsedSignals,
    extract_crystal_tool_calls,
    inject_crystal_tools,
    is_crystal_tool_call,
    parse_tool_calls,
)
from .v3_routers import (
    ContentRouter,
    KnowledgeRouter,
    RouterResult,
)
from .v3_signal_handler import (
    AUTO_COMMIT_THRESHOLD,
    REVIEW_QUEUE_THRESHOLD,
    handle_signals,
    run_inline_research,
)

__all__ = [
    # sparse_key (Wave 7A)
    "DELIMITER",
    "SparseKey",
    "detect_gaps",
    "common_prefix",
    "common_suffix",
    "contains_segment",
    "format_key",
    "is_structured_key",
    "matches",
    "parse_key",
    "scan_keys",
    "validate_key",
    # v3_routers (Wave 7A)
    "ContentRouter",
    "KnowledgeRouter",
    "RouterResult",
    # v3_navigation (Wave 7A)
    "NavigationResult",
    "NavigationRouter",
    # v3_depth (Wave 7A)
    "DepthResult",
    "DepthRouter",
    # v3_composer (Wave 7A)
    "Composer",
    "determine_injection_method",
    "determine_match_type",
    # v3_push_pull (Wave D)
    "CRYSTAL_TOOLS",
    "CRYSTAL_TOOL_NAMES",
    "ParsedSignals",
    "extract_crystal_tool_calls",
    "inject_crystal_tools",
    "is_crystal_tool_call",
    "parse_tool_calls",
    # v3_signal_handler (Wave D)
    "AUTO_COMMIT_THRESHOLD",
    "REVIEW_QUEUE_THRESHOLD",
    "handle_signals",
    "run_inline_research",
    # pipeline (Wave 7B) — V2 legacy entry point
    "DEFAULT_TOP_K",
    "RetrievalOutcome",
    "retrieve_and_inject",
    # router (Wave 7B) — V2 legacy top-K crystal routing
    "CrystalRouter",
    # reader (Wave 7B) — V2 legacy crystal → context snippet
    "CrystalContext",
    "CrystalReader",
    "Voicing",
    # recall (Wave 7B) — bind-storage read-side primitive
    "RecalledFact",
    "recall_from_crystal",
    # chain_resolver (Wave 7B) — ACL-checked one-hop chain walk
    "ChainResolver",
    # match_classifier (Wave 7B) — three-way + four-way classifiers
    "MatchClassifier",
    "MatchType",
    "RoutingDecision",
    "RoutingResult",
    # concept_router (Wave 7B) — decomposer → DSL config-rank path
    "ConceptRouter",
    "ConceptRouteOutcome",
    # synthesis (Wave 7B) — SPREAD-branch bind-v1 joint statement
    "synthesize_joint_statement",
    # composer (Wave 7B) — V2 pluggable composer strategies
    #   Note: distinct from v3_composer.Composer above
    "ComposerContext",
    "ComposerStrategy",
    "InstructionComposer",
    "BayesianComposer",
    "get_composer",
    "register_composer",
    # mem0_session (Wave 7F) — Mem0 session-memory wrapper (D6 / P0.9)
    "init_mem0",
    "get_mem0",
    "add_conversation_turn",
    "search_session_context",
    # citations (Growth G1) — trust + the metering rail
    "CITE_INSTRUCTION",
    "CitationSource",
    "assign_handles",
    "build_primary_citation",
    "extract_claim_span",
    "map_citations",
    "parse_citations",
    "render_sources_footer",
    "rewrite_markers",
    "CITATION_GROUNDING_THRESHOLD",
    "ground_citations",
]
