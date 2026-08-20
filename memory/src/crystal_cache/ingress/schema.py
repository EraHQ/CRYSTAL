"""OpenAI-compatible request/response schemas for /v1/chat/completions.

We accept OpenAI's wire format as the universal input. Internally, provider
adapters translate to/from each provider's native API (see
execution/upstream_client.py).

These schemas intentionally cover the common subset. Not everything in
OpenAI's spec is supported:
  - JSON mode / structured outputs — TODO, Phase 1.5.4
  - image/audio content — TODO, Phase 5.4 (placeholder type below)
  - logit_bias, user, seed — ignored, not forwarded

Supported via pass-through / translation:
  - streaming (SSE) — Phase 1.5.1
  - tool/function calling — Phase 1.5.2 (translated for Anthropic)
  - error envelope (OpenAI-shaped) — Phase 1.5.5

A plain "chat with messages, get a response" works. That's the minimum;
everything above is parity for OpenAI-SDK customers.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


# -----------------------------------------------------------------------------
# Request
# -----------------------------------------------------------------------------

MessageRole = Literal["system", "user", "assistant", "tool"]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: MessageRole
    # OpenAI allows content to be a string, a list of content parts, or
    # null. Null is required on assistant messages that only emit
    # tool_calls (no text) — the OpenAI SDK sends `content=None`
    # automatically in that case. Phase 1.5.2.
    content: Union[str, list[dict[str, Any]], None] = None
    name: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI POST /v1/chat/completions body."""
    # Accept extra fields (user, seed, logit_bias, response_format, etc.)
    # without validation error. We don't forward them in v0 but we don't
    # want to reject them either.
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)

    # Common sampling params
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    n: Optional[int] = Field(default=None, ge=1)
    stop: Optional[Union[str, list[str]]] = None
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)

    # SSE streaming. Phase 1.5.1.
    stream: Optional[bool] = None

    # Tool / function calling. Phase 1.5.2.
    #
    # `tools` is an OpenAI-shape list:
    #   [{"type": "function",
    #     "function": {"name": ..., "description": ..., "parameters": {...}}}]
    # We don't validate the inner shape here — passes through to OpenAI as-is,
    # gets translated to Anthropic's `tools` shape inside AnthropicClient.
    #
    # `tool_choice` accepts:
    #   - "none" | "auto" | "required" (string forms)
    #   - {"type": "function", "function": {"name": <name>}} (specific tool)
    # OpenAI-shape; AnthropicClient translates to Anthropic's tool_choice.
    #
    # `parallel_tool_calls` defaults true at OpenAI; we accept it explicitly
    # so customers can opt into single-call mode without it being silently
    # dropped. Anthropic doesn't have a direct equivalent — we ignore it on
    # Anthropic-routed requests (Anthropic always allows parallel) and the
    # field passes through to OpenAI unchanged.
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Optional[Union[str, dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None

    # Phase 1.5.4: JSON mode / structured outputs pass-through.
    #
    # OpenAI supports:
    #   {"type": "json_object"}                    — guaranteed valid JSON
    #   {"type": "json_schema", "json_schema": {}} — structured outputs
    # Anthropic supports JSON mode via a different mechanism (tooling or
    # system prompt). Self-hosted endpoints vary.
    #
    # We accept any dict shape and pass it through. No internal logic
    # changes — the upstream provider enforces or ignores as appropriate.
    response_format: Optional[dict[str, Any]] = None

    # OpenAI-compatible metadata. Free-form string-to-string map; OpenAI
    # uses it for tagging requests with caller-defined keys. We adopt
    # one well-known key:
    #
    #   metadata.sequence_id — conversation anchor (Stage 2a). When
    #   present, the gateway uses it as-is to group QueryLog rows.
    #   When absent, the gateway falls back to the X-Sequence-Id
    #   header, then to server-side inference from the message hash.
    #
    # Other keys flow through to telemetry via `extra="allow"` but are
    # not load-bearing for the pipeline.
    metadata: Optional[dict[str, str]] = None


# -----------------------------------------------------------------------------
# Customer CRUD
# -----------------------------------------------------------------------------

class CreateCustomerRequest(BaseModel):
    """Body for POST /v1/customers.

    The caller provides upstream routing config. We generate the Crystal Cache
    API key (Key A in the design doc) server-side and return it once.
    """
    provider: Literal["openai", "anthropic", "self_hosted"]
    model_id: str  # e.g. "gpt-4o", "claude-opus-4-7", "Qwen/Qwen3-0.6B"
    api_key_ref: str  # upstream key (Key B) — encrypted at rest by the store (enc:v1)
    base_url: Optional[str] = None  # required when provider == "self_hosted"
    injection_preference: Literal["text", "hidden_state", "none"] = "text"
    shadow_sample_rate: float = Field(default=0.05, ge=0.0, le=1.0)


class CreateCustomerResponse(BaseModel):
    """Response body for POST /v1/customers.

    api_key (Key A) is returned ONCE here. The caller must save it — the
    server stores only its hash (credentials.hash_api_key); the upstream
    key (Key B) is AES-256-GCM encrypted at rest.
    """
    id: str
    api_key: str
    provider: str
    model_id: str


class GetCustomerResponse(BaseModel):
    """Response body for GET /v1/customers/{id}.

    Does NOT include the upstream api_key (api_key_ref). Does NOT include
    the Crystal Cache api_key (shown only at creation).
    """
    id: str
    provider: str
    model_id: str
    base_url: Optional[str] = None
    injection_preference: str
    shadow_sample_rate: float
    # Assumptions funnel F3 (Q5=A): the tenant's explore toggle.
    # None = deployment default (explore ON).
    assumptions_explore: Optional[bool] = None
    created_at: str  # ISO 8601


# -----------------------------------------------------------------------------
# Operator CRUD (Foundation F1 — team identity layer)
# -----------------------------------------------------------------------------
#
# Operators are authenticated humans under a team (the customer = the
# team). Management is authed by the team's customer key (the team root
# provisions its operators); operator self-introspection (/v1/operators/me)
# is authed by the operator's own key. Role/status literals are inlined to
# match this module's convention (canonical types live on models.operator).

class CreateOperatorRequest(BaseModel):
    """Body for POST /v1/operators — create an operator under the team the
    Bearer customer key belongs to."""
    display_name: str = Field(min_length=1, max_length=256)
    role: Literal["admin", "operator", "viewer"] = "operator"


class CreateOperatorResponse(BaseModel):
    """Response for POST /v1/operators.

    api_key (the operator's Key A) is returned ONCE here — only its hash is
    persisted, never the raw key (no plaintext at rest).
    """
    id: str
    team_id: str
    display_name: str
    role: str
    status: str
    api_key: str  # raw, shown once


class OperatorResponse(BaseModel):
    """Response for operator reads. Never includes a key."""
    id: str
    team_id: str
    display_name: str
    role: str
    status: str
    created_at: str  # ISO 8601


class OperatorListResponse(BaseModel):
    """Response for GET /v1/operators."""
    total: int = 0
    operators: list[OperatorResponse] = Field(default_factory=list)


class SetOperatorRoleRequest(BaseModel):
    """Body for PATCH /v1/operators/{id}/role."""
    role: Literal["admin", "operator", "viewer"]


class SetOperatorStatusRequest(BaseModel):
    """Body for PATCH /v1/operators/{id}/status."""
    status: Literal["active", "suspended"]


# -----------------------------------------------------------------------------
# Feedback (Stage 2b)
# -----------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    """Body for POST /v1/feedback.

    The customer's frontend collects feedback in their UI; their
    backend forwards it here. We accept (sequence_id, turn_index, signal)
    and an optional comment. The (customer_id) is derived from the
    Bearer token, not the body — this prevents one customer from
    submitting feedback against another customer's conversations.

    We do NOT validate that the referenced (sequence_id, turn_index)
    actually exists as a QueryLog row. See MetadataStore.write_feedback
    for the rationale.
    """
    sequence_id: str = Field(min_length=1, max_length=64)
    turn_index: int = Field(ge=0)
    signal: Literal["up", "down"]
    comment: Optional[str] = Field(default=None, max_length=4000)


class FeedbackResponse(BaseModel):
    """Response body for POST /v1/feedback."""
    id: str
    customer_id: str
    sequence_id: str
    turn_index: int
    signal: Literal["up", "down"]
    comment: Optional[str]
    created_at: str  # ISO 8601
    # V2: learning outcome (populated when feedback triggers learning)
    learning_triggered: bool = False
    crystals_written: int = 0


# -----------------------------------------------------------------------------
# SDK Mode Schemas (V2)
# -----------------------------------------------------------------------------

class RetrieveRequest(BaseModel):
    """Body for POST /v1/retrieve.

    SDK mode: retrieve relevant crystals for a query without
    making an upstream LLM call. The developer decides how to
    use the retrieved context in their own prompt.
    """
    query: str = Field(min_length=1, max_length=50000)
    crystal_type: str = "customer:legacy"
    k: int = Field(default=5, ge=1, le=20)
    composer: str = Field(
        default="bayesian",
        description="Composer strategy: 'instruction' or 'bayesian'",
    )


class RetrieveResponse(BaseModel):
    """Response body for POST /v1/retrieve."""
    injection: str = ""  # Pre-composed injection text
    cache_hit: bool = False
    answer: Optional[str] = None  # Populated on cache hit
    score: float = 0.0
    routing: str = "no_match"  # PERFECT / SPREAD / LOW_CONFIDENCE / NO_MATCH
    matched_crystal_ids: list[str] = Field(default_factory=list)
    sparse_key: Optional[str] = None  # What sparse key was used


class LearnRequest(BaseModel):
    """Body for POST /v1/learn.

    Teach the system from a success or failure. On failure, the
    system generates a reflection + knowledge crystal. On success,
    the system caches the solution.
    """
    prompt: str = Field(min_length=1, max_length=50000)
    response: str = Field(min_length=1, max_length=100000)
    outcome: Literal["pass", "fail"]
    signal: Optional[str] = Field(
        default=None, max_length=10000,
        description="Failure signal (test errors, user feedback, etc.)",
    )
    crystal_type: str = "customer:legacy"


class LearnResponse(BaseModel):
    """Response body for POST /v1/learn."""
    crystals_written: int = 0
    reflection: Optional[str] = None
    knowledge: Optional[str] = None
    category: Optional[str] = None
    cached: bool = False
    error: Optional[str] = None


class StoreRequest(BaseModel):
    """Body for POST /v1/store.

    Directly store a fact or knowledge crystal without going
    through the learning pipeline. For domain experts.
    """
    key: str = Field(
        min_length=1, max_length=5000,
        description="The retrieval key (what query should match this)",
    )
    value: str = Field(
        min_length=1, max_length=50000,
        description="The knowledge content",
    )
    crystal_type: str = "customer:legacy"
    pair_type: str = "question_answer"
    source_kind: Literal[
        "model_reasoning", "failed_reasoning",
        "web_search_result", "code_execution_result",
    ] = "model_reasoning"
    answer_value: Optional[str] = Field(
        default=None,
        description="If set, enables cache-hit on exact match",
    )
    # Foundation F2: operator-authored writes only. True stamps the crystal
    # Ingest scope (P2, ratified 2026-07-02). Explicit scope wins; else the
    # legacy `private` flag; else the deployment default
    # (CC_DEFAULT_INGEST_SCOPE — personal). P1 gives every request an
    # operator (team keys act as the Default Admin), so every write is
    # owner-attributed and personal scope is always well-defined.
    scope: Optional[str] = Field(
        default=None,
        description=(
            "Crystal scope: 'personal' (owner-only, mode 0o600) or 'team' "
            "(group-readable, mode 0o640). Omitted = the deployment "
            "default (CC_DEFAULT_INGEST_SCOPE, ships as 'personal'). "
            "Supersedes `private`."
        ),
    )
    # mode 0o600 (owner-private); False (default) stamps 0o640 (team-
    # readable). Ignored for team-key writes (which create unowned crystals).
    private: bool = Field(
        default=False,
        description=(
            "Operator-authored writes only: True = owner-private crystal "
            "(mode 0o600); False = team-readable (mode 0o640)."
        ),
    )


class StoreResponse(BaseModel):
    """Response body for POST /v1/store."""
    crystal_id: str
    fact_id: str
    sparse_key: str


class ConsolidateRequest(BaseModel):
    """Body for POST /v1/consolidate.

    Trigger memory consolidation: merge duplicate rules, add
    UNLESS clauses from contradicting knowledge, find systemic
    failure patterns via meta-reflection.
    """
    crystal_type: str = "customer:legacy"
    run_meta: bool = Field(
        default=True,
        description="Run meta-reflection on systemic patterns (slower, more tokens)",
    )


class ConsolidateResponse(BaseModel):
    """Response body for POST /v1/consolidate."""
    mandatory_rules_written: int = 0
    advisory_rules_written: int = 0
    meta_patterns_written: int = 0
    contradictions_found: int = 0
    behavior_rules_found: int = 0
    error: Optional[str] = None


class BankStatsResponse(BaseModel):
    """Response body for GET /v1/stats."""
    crystal_count: int = 0
    fact_count: int = 0
    quality_distribution: dict[str, int] = Field(default_factory=dict)
    crystal_type_distribution: dict[str, int] = Field(default_factory=dict)
    pair_type_distribution: dict[str, int] = Field(default_factory=dict)
    source_kind_distribution: dict[str, int] = Field(default_factory=dict)
    cache_hit_eligible: int = 0
    total_query_logs: int = 0
    recent_cache_hit_rate: Optional[float] = None


class CrystalDetailResponse(BaseModel):
    """Response body for GET /v1/crystals/{id}/detail."""
    id: str
    customer_id: Optional[str] = None
    crystal_type: Optional[str] = None
    source_kind: Optional[str] = None
    quality_tier: str = "neutral"
    fact_count: int = 0
    created_at: str
    facts: list[dict[str, Any]] = Field(default_factory=list)


class CrystalListResponse(BaseModel):
    """Response body for GET /v1/crystals-list."""
    total: int = 0
    crystals: list[dict[str, Any]] = Field(default_factory=list)


class ExportResponse(BaseModel):
    """Response body for POST /v1/export."""
    record_count: int = 0
    export_format: str = "jsonl"
    data: list[dict[str, Any]] = Field(default_factory=list)


class ImportRequest(BaseModel):
    """Body for POST /v1/import."""
    records: list[dict[str, Any]] = Field(
        min_length=1,
        description="List of {key, value, pair_type?, source_kind?, answer_value?}",
    )
    crystal_type: str = "customer:legacy"
    wipe: bool = Field(
        default=False,
        description="Delete existing crystals before importing",
    )


class ImportResponse(BaseModel):
    """Response body for POST /v1/import."""
    records_processed: int = 0
    crystals_written: int = 0
    errors: int = 0


class SubscribeRequest(BaseModel):
    """Body for POST /v1/subscribe and POST /v1/unsubscribe.

    Accepts either shape — at least one must be present:
      {"crystal_type": "general:x"}           single (the inspector UI's toggles)
      {"crystal_types": ["general:x", ...]}   batch (SDK callers)

    Presence is validated in the endpoint helper (clean 422 detail)
    rather than a model validator, so a bad body produces the same
    error envelope as every other /v1 validation failure. Found live
    2026-06-12: the original list-only shape silently 422'd the UI's
    singular body and the subscription toggles appeared dead.
    """
    crystal_type: Optional[str] = None
    crystal_types: Optional[list[str]] = None


class QueryLogEntry(BaseModel):
    """One entry in the query log."""
    id: str
    query_text: str
    match_type: Optional[str] = None
    injection_method: Optional[str] = None
    top_score: Optional[float] = None
    cache_hit: bool = False
    upstream_call_made: bool = True
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    timestamp: str


class QueryLogResponse(BaseModel):
    """Response body for GET /v1/query-log."""
    total: int = 0
    entries: list[QueryLogEntry] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Document Ingestion Schemas (V2)
# -----------------------------------------------------------------------------

class DocumentUploadRequest(BaseModel):
    """Body for POST /v1/documents."""
    text: str = Field(min_length=1, max_length=500000)
    label: str = Field(default="", max_length=256)
    crystal_type: str = "customer:legacy"
    # P2 scope-on-sources: personal|team; omitted = deployment default.
    scope: Optional[str] = None
    auto_crystallize: bool = Field(
        default=False,
        description="If true, crystallize immediately",
    )


class DocumentResponse(BaseModel):
    """Response for a document."""
    id: str
    customer_id: str
    label: str
    status: str
    char_count: int = 0
    crystals_written: int = 0
    items_extracted: int = 0
    created_at: str
    crystallized_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Response for GET /v1/documents."""
    total: int = 0
    documents: list[DocumentResponse] = Field(default_factory=list)


class CrystallizeResponse(BaseModel):
    """Response for POST /v1/documents/{id}/crystallize."""
    document_id: str
    status: str
    chunks_processed: int = 0
    items_extracted: int = 0
    crystals_written: int = 0
    errors: int = 0
    items: list[dict[str, Any]] = Field(default_factory=list)
