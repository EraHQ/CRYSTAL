"""SDK mode endpoints — /v1/{retrieve, store, learn, consolidate, stats, crystals, query_logs}.

These give developers programmatic access to the cache without going
through the chat completions proxy. Useful when the customer's app
wants to compose its own prompt and just needs the cache to return
relevant context.

Phase 6 status by endpoint, with Phase 7 Wave 7F fill-ins:
  - /v1/stats               works today (uses ported store methods)
  - /v1/crystals/*          works today (uses ported store methods)
  - /v1/crystals-list       alias for /v1/crystals (Phase 6.5 P2.2, back-compat)
  - /v1/query_logs          works today (uses ported store methods)
  - /v1/store               works today (Wave 7F: now also generates
                            sparse_key per v1 verbatim)
  - /v1/retrieve            FILLED Wave 7F — consumes retrieve_and_inject
                            + composer; loads mandatory_rules +
                            meta_patterns via the Wave 7F mixin
                            methods (R9-clean, no inline SQL)
  - /v1/learn               FILLED Wave 7F — consumes LearningService
  - /v1/consolidate         FILLED Wave 7F — consumes ConsolidationService
  - /v1/export, /v1/import  STUB → 501 (require multi-table reads)
  - /v1/subscribe etc.      STUB → 501 (general crystal subscriptions)

The three Wave 7F fills are R9-clean: v1's `sdk_retrieve` had inline
SQLAlchemy reading MandatoryRuleRow + MetaPatternRow, which violates
R9. The v2 port routes those reads through new
`LearningExtensionsMixin` methods (list_mandatory_rules_for_customer,
list_meta_patterns_for_customer) — same mixin file that holds the
Wave 7E write-side methods for the same tables.
"""
from __future__ import annotations

from collections import Counter
from typing import Annotated, Any, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from ..infrastructure import MetadataStore
from ..infrastructure.metadata_store import get_metadata_store
from ..ingress.auth import require_customer, resolve_principal, require_customer_or_console
from ..ingress.schema import (
    BankStatsResponse,
    ConsolidateRequest,
    ConsolidateResponse,
    CrystalDetailResponse,
    CrystalListResponse,
    ExportResponse,
    ImportRequest,
    ImportResponse,
    LearnRequest,
    LearnResponse,
    QueryLogResponse,
    RetrieveRequest,
    RetrieveResponse,
    StoreRequest,
    StoreResponse,
    SubscribeRequest,
)
from ..models import Customer, Operator

logger = structlog.get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# /v1/retrieve — FILLED Wave 7F
# ---------------------------------------------------------------------------

@router.post("/v1/retrieve", response_model=RetrieveResponse)
async def sdk_retrieve(
    body: RetrieveRequest,
    request: Request,
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(resolve_principal)
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> RetrieveResponse:
    """SDK mode: retrieve relevant crystals for a query.

    Returns pre-composed injection text and crystal metadata without
    making an upstream LLM call. The developer decides how to use the
    retrieved context in their own prompt.

    Cache-hit short-circuit: if retrieve_and_inject returns a populated
    cache_hit_response, return that directly with cache_hit=True so the
    caller knows the bank has a verified answer.

    Wave 7F note on R9: v1 had inline SQLAlchemy reading
    MandatoryRuleRow + MetaPatternRow. v2 routes through
    `store.list_mandatory_rules_for_customer` +
    `store.list_meta_patterns_for_customer` (new Wave 7F read-side
    methods on LearningExtensionsMixin). Same tables, same access
    pattern, R9-clean.
    """
    from ..encoding.sparse_keys import generate_sparse_key
    from ..retrieval import (
        ComposerContext,
        get_composer,
        retrieve_and_inject,
    )

    customer, operator = principal

    # Generate sparse key for logging
    sparse_key: Optional[str] = None
    try:
        sparse_key = generate_sparse_key(body.query)
    except Exception:
        pass

    # Run retrieval pipeline
    messages = [{"role": "user", "content": body.query}]
    try:
        outcome = await retrieve_and_inject(
            customer=customer,
            messages=messages,
            store=store,
            vector_index=request.app.state.vector_index,
            encoder=request.app.state.prompt_encoder,
            crystal_type=body.crystal_type,
            operator=operator,
        )
    except Exception as e:
        logger.error("sdk.retrieve.failed", error=str(e))
        return RetrieveResponse()

    # Cache hit short-circuit
    if outcome.cache_hit_response is not None:
        return RetrieveResponse(
            injection="",
            cache_hit=True,
            answer=outcome.cache_hit_response,
            score=outcome.top_score,
            routing=(
                outcome.routing_decision.value
                if outcome.routing_decision
                else "no_match"
            ),
            matched_crystal_ids=outcome.matched_crystal_ids,
            sparse_key=sparse_key,
        )

    # Build injection via composer
    injection = ""
    if outcome.injected_text:
        composer = get_composer(body.composer)
        ctx = ComposerContext(
            failure_rules=[],
            knowledge_facts=[],
            reference_text=outcome.injected_text,
            reference_match_type=(
                outcome.routing_decision.value
                if outcome.routing_decision
                else None
            ),
            customer_id=customer.id,
            query_text=body.query,
        )
        # Load mandatory rules + meta patterns via Wave 7F mixin
        # methods (R9-clean; replaces v1's inline SQLAlchemy).
        try:
            ctx.mandatory_rules = list(
                await store.list_mandatory_rules_for_customer(customer.id)
            )
            ctx.meta_patterns = list(
                await store.list_meta_patterns_for_customer(customer.id)
            )
        except Exception:
            # Best-effort: failure to load rules/patterns degrades
            # composer to using just the injected_text. v1 had the
            # same try/except shape around the inline SQL.
            pass

        injection = await composer.compose(ctx)
        if not injection:
            injection = outcome.injected_text or ""

    return RetrieveResponse(
        injection=injection,
        cache_hit=False,
        score=outcome.top_score,
        routing=(
            outcome.routing_decision.value
            if outcome.routing_decision
            else "no_match"
        ),
        matched_crystal_ids=outcome.matched_crystal_ids,
        sparse_key=sparse_key,
    )


# ---------------------------------------------------------------------------
# /v1/store — works today (Wave 7F: now generates sparse_key per v1)
# ---------------------------------------------------------------------------

@router.post("/v1/store", response_model=StoreResponse)
async def sdk_store(
    body: StoreRequest,
    request: Request,
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(resolve_principal)
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> StoreResponse:
    """SDK mode: directly store a fact or knowledge crystal.

    Bypasses the learning pipeline. For domain experts who know
    exactly what knowledge to add.

    The sparse_key is a unified wide->specific PATH derived from the key
    AND value together (see encoding.sparse_keys). Folding the value in
    yields a deeper, more specific path, which widens the shared-segment
    surface that gap detection and the ratchet feed on. It is stored as the
    prompt_text and returned in the response; the value is separately stored
    as the answer and drives the fact-lane search vector. Keyless, the key
    degrades to depth-1. (Supersedes the prior depth-1 'first 8 words'
    behavior with a real path when an LLM key is present.)
    """
    from ..encoding.sparse_keys import generate_sparse_key

    customer, operator = principal

    # P1 + P2 (ratified 2026-07-02): every request resolves to an operator
    # (team keys act as the Default Admin), so every crystal is born owned.
    # Scope precedence: explicit body.scope > legacy private flag > the
    # deployment default (CC_DEFAULT_INGEST_SCOPE — ships as personal).
    # Viewers are read-only — they may not write.
    if operator is not None and operator.role == "viewer":
        raise HTTPException(
            status_code=403,
            detail="Viewers are read-only and cannot store crystals.",
        )
    if body.scope is not None and body.scope not in ("personal", "team"):
        raise HTTPException(
            status_code=422,
            detail="scope must be 'personal' or 'team'",
        )
    from ..config import get_settings
    from ..infrastructure.permissions import mode_for_scope

    scope = body.scope or (
        "personal" if body.private else get_settings().default_ingest_scope
    )
    owner_operator_id = operator.id if operator is not None else None
    group_team_id = operator.team_id if operator is not None else None
    mode = mode_for_scope(scope) if operator is not None else 0o640

    encoder = request.app.state.prompt_encoder
    vector_store = request.app.state.vector_store

    sparse_key = generate_sparse_key(f"{body.key}: {body.value}")

    crystal, fact = await store.add_pair_for_customer(
        customer_id=customer.id,
        prompt_text=sparse_key,
        answer_text=body.value,
        pair_type=body.pair_type or "question_answer",
        encoder=encoder,
        vector_store=vector_store,
        vector_index=getattr(request.app.state, "vector_index", None),
        crystal_type=body.crystal_type or "customer:legacy",
        source_kind=body.source_kind or "model_reasoning",
        answer_value=body.answer_value,
        owner_operator_id=owner_operator_id,
        group_team_id=group_team_id,
        mode=mode,
    )

    return StoreResponse(
        crystal_id=crystal.id,
        fact_id=fact.id,
        sparse_key=sparse_key,
    )


# ---------------------------------------------------------------------------
# /v1/learn — FILLED Wave 7F
# ---------------------------------------------------------------------------

async def run_learn(
    *,
    body: LearnRequest,
    request: Request,
    customer: Customer,
    store: MetadataStore,
) -> LearnResponse:
    """Teach the system from a success or failure for an already-resolved
    customer. The SDK /v1/learn route and the trusted-internal admin learn
    route both delegate here; only the customer-resolution differs.

    On failure: generates reflection + knowledge crystal via Level B+F
    (LearningService.learn_from_failure).
    On success: caches the solution for future retrieval
    (LearningService.cache_success).
    """
    from ..learning import LearningService

    encoder = request.app.state.prompt_encoder
    vector_store = request.app.state.vector_store
    svc = LearningService(
        store=store, encoder=encoder, vector_store=vector_store,
        vector_index=getattr(request.app.state, "vector_index", None),
    )

    if body.outcome == "fail":
        signal = body.signal or "User indicated this response was incorrect"
        result = await svc.learn_from_failure(
            customer_id=customer.id,
            prompt=body.prompt,
            response=body.response,
            failure_signal=signal,
            crystal_type=body.crystal_type,
        )
        return LearnResponse(
            crystals_written=result.crystals_written,
            reflection=result.reflection,
            knowledge=result.knowledge,
            category=result.category,
            error=result.error,
        )
    else:
        cached = await svc.cache_success(
            customer_id=customer.id,
            prompt=body.prompt,
            solution=body.response,
            crystal_type=body.crystal_type,
        )
        return LearnResponse(
            crystals_written=1 if cached else 0,
            cached=cached,
        )


@router.post("/v1/learn", response_model=LearnResponse)
async def sdk_learn(
    body: LearnRequest,
    request: Request,
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> LearnResponse:
    """SDK mode: teach the system from a success or failure (Bearer Key A).

    Thin wrapper over run_learn; the admin learn route delegates to the same
    helper with a path-resolved customer.
    """
    return await run_learn(
        body=body, request=request, customer=customer, store=store,
    )


# ---------------------------------------------------------------------------
# /v1/consolidate — FILLED Wave 7F
# ---------------------------------------------------------------------------

@router.post("/v1/consolidate", response_model=ConsolidateResponse)
async def sdk_consolidate(
    body: ConsolidateRequest,
    request: Request,
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> ConsolidateResponse:
    """SDK mode: trigger memory consolidation.

    Merges duplicate behavior rules, adds UNLESS clauses from
    contradicting knowledge, identifies systemic failure patterns.
    Call after a batch of learn() calls to clean up the bank.

    Wave 7F fill-in. Verbatim port from v1 except for the import.
    """
    from ..maintenance import ConsolidationService

    svc = ConsolidationService(store=store)
    result = await svc.consolidate(
        customer_id=customer.id,
        crystal_type=body.crystal_type,
        run_meta=body.run_meta,
    )

    return ConsolidateResponse(
        mandatory_rules_written=result.mandatory_rules_written,
        advisory_rules_written=result.advisory_rules_written,
        meta_patterns_written=result.meta_patterns_written,
        contradictions_found=result.contradictions_found,
        behavior_rules_found=result.behavior_rules_found,
        error=result.error,
    )


# ---------------------------------------------------------------------------
# /v1/stats — works today (Phase 6.5 P2.1 restored distributions)
# ---------------------------------------------------------------------------

@router.get("/v1/stats", response_model=BankStatsResponse)
async def sdk_stats(
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Return summary statistics for this customer's bank.

    Distributions (quality_tier, crystal_type, pair_type, source_kind)
    are computed by iterating the customer's crystals and their facts.
    Acceptable for inspector-scale customers (low-thousands of
    crystals); production scale would push these into SQL GROUP BY
    queries.

    Per Phase 6.5 P2.1, this restores v1's response fidelity. Wave B
    initially returned only crystal_count + fact_count.
    """
    crystal_count = await store.count_crystals_for_customer(customer.id)
    crystals = await store.list_crystals_for_customer(customer.id)

    # Crystal-level distributions
    quality_dist: Counter[str] = Counter()
    type_dist: Counter[str] = Counter()
    source_dist: Counter[str] = Counter()
    cache_hit_eligible = 0

    for c in crystals:
        if c.quality_tier:
            quality_dist[c.quality_tier] += 1
        if c.crystal_type:
            type_dist[c.crystal_type] += 1
        if c.source_kind:
            source_dist[c.source_kind] += 1
        # A crystal is cache-hit eligible if it has an answer_value
        # populated (the cache-hit short-circuit reads this directly).
        if c.answer_value:
            cache_hit_eligible += 1

    # Fact-level: pair_type distribution. Requires walking the facts
    # of every crystal. Acceptable for inspector scale.
    pair_type_dist: Counter[str] = Counter()
    total_facts = 0
    for c in crystals:
        facts = await store.list_facts_for_crystal(c.id)
        total_facts += len(facts)
        for f in facts:
            if f.pair_type:
                pair_type_dist[f.pair_type] += 1

    # Query log total. The list method returns (total, items); we
    # only need the total for the stats response.
    total_query_logs, _ = await store.list_query_logs_for_customer(
        customer_id=customer.id, limit=1, offset=0,
    )

    return JSONResponse(content={
        "crystal_count": crystal_count,
        "fact_count": total_facts,
        "quality_distribution": dict(quality_dist),
        "crystal_type_distribution": dict(type_dist),
        "pair_type_distribution": dict(pair_type_dist),
        "source_kind_distribution": dict(source_dist),
        "cache_hit_eligible": cache_hit_eligible,
        "total_query_logs": total_query_logs,
        "recent_cache_hit_rate": None,  # Phase 7+ — requires query log mining
    })


# ---------------------------------------------------------------------------
# /v1/crystals/* — works today
# ---------------------------------------------------------------------------

# Phase 6.5 P2.2: register both /v1/crystals (v2 URL) and the v1 alias
# /v1/crystals-list pointing at the same handler. The alias preserves
# back-compat for SDK consumers using v1's URL; Phase 11 cleanup will
# drop the alias once usage is confirmed migrated.

@router.get("/v1/crystals", response_model=CrystalListResponse)
@router.get("/v1/crystals-list", response_model=CrystalListResponse)
async def sdk_crystal_list(
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    limit: int = 50,
    offset: int = 0,
) -> JSONResponse:
    """List the customer's crystals (paginated).

    Two URL paths point here:
      - GET /v1/crystals       — v2 URL (preferred)
      - GET /v1/crystals-list  — v1 URL alias (back-compat; Phase 11
                                 cleanup will drop this)
    """
    total, crystals = await store.list_crystals_for_customer_paginated(
        customer_id=customer.id, limit=limit, offset=offset,
    )
    return JSONResponse(content={
        "total": total,
        "offset": offset,
        "limit": limit,
        "crystals": [
            {
                "id": c.id,
                "crystal_type": c.crystal_type,
                "summary_text": c.summary_text,
                "fact_count": c.fact_count,
                "quality_tier": c.quality_tier,
                "created_at": c.created_at.isoformat(),
            }
            for c in crystals
        ],
    })


@router.get("/v1/crystals/{crystal_id}", response_model=CrystalDetailResponse)
async def sdk_crystal_detail(
    crystal_id: str,
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    crystal = await store.get_crystal(crystal_id)
    if crystal is None or crystal.customer_id != customer.id:
        raise HTTPException(status_code=404, detail="Crystal not found")
    facts = await store.list_facts_for_crystal(crystal_id)
    return JSONResponse(content={
        "crystal": {
            "id": crystal.id,
            "crystal_type": crystal.crystal_type,
            "summary_text": crystal.summary_text,
            "fact_count": crystal.fact_count,
            "created_at": crystal.created_at.isoformat(),
        },
        "facts": [
            {
                "id": f.id,
                "claim_text": f.claim_text,
                "pair_type": f.pair_type,
                "prompt_text": f.prompt_text,
                "created_at": f.created_at.isoformat(),
            }
            for f in facts
        ],
    })


@router.delete("/v1/crystals/{crystal_id}")
async def sdk_crystal_delete(
    crystal_id: str,
    request: Request,
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Delete a crystal and all its facts (tenancy-scoped).

    Wired (CU-9) to store.delete_crystal, which cascades to the
    crystal's facts and invalidates this customer's in-memory vector
    stores so routing can't bond into a deleted crystal and deleted
    facts stop surfacing in fact search. 404 when the crystal doesn't
    exist or belongs to another customer.

    Per-fact deletion (DELETE /v1/facts/{id}) stays a stub on purpose:
    removing one fact from a shared crystal requires subtracting its
    grating from the accumulated HDC codebook (the per-fact
    grating-subtraction backlog item), so whole-crystal deletion is the
    clean removal primitive.
    """
    deleted = await store.delete_crystal(
        crystal_id,
        customer.id,
        vector_store=getattr(request.app.state, "vector_store", None),
        # Invalidate via the active vector index (Qdrant-aware) when present,
        # else the in-memory fact store. Both expose sync invalidate(customer).
        fact_vector_store=(getattr(request.app.state, "vector_index", None)
                           or getattr(request.app.state, "fact_vector_store", None)),
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Crystal not found")
    return JSONResponse(content={"deleted": True, "crystal_id": crystal_id})


@router.delete("/v1/facts/{fact_id}")
async def sdk_fact_delete(
    fact_id: str,
    request: Request,
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Delete a single fact and recompute its crystal's vectors (tenancy-scoped).

    Wired (CU-9) to store.delete_fact: removes the fact, rebuilds the parent
    crystal's summary/routing vectors from the surviving facts (or deletes
    the crystal if this was its last fact), and invalidates this customer's
    in-memory vector stores. 404 when the fact doesn't exist or its crystal
    belongs to another customer.
    """
    deleted = await store.delete_fact(
        fact_id,
        customer.id,
        encoder=request.app.state.prompt_encoder,
        vector_store=getattr(request.app.state, "vector_store", None),
        # Active vector index (Qdrant-aware) when present, else in-memory fact
        # store. Both expose sync invalidate(customer).
        fact_vector_store=(getattr(request.app.state, "vector_index", None)
                           or getattr(request.app.state, "fact_vector_store", None)),
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Fact not found")
    return JSONResponse(content={"deleted": True, "fact_id": fact_id})


# ---------------------------------------------------------------------------
# /v1/query_logs — works today
# ---------------------------------------------------------------------------

@router.get("/v1/query_logs", response_model=QueryLogResponse)
async def sdk_query_log(
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    limit: int = 100,
    offset: int = 0,
) -> JSONResponse:
    total, logs = await store.list_query_logs_for_customer(
        customer_id=customer.id, limit=limit, offset=offset,
    )
    return JSONResponse(content={
        "total": total,
        "offset": offset,
        "limit": limit,
        "query_logs": [
            {
                "id": q.id,
                "query_text": q.query_text,
                "match_type": q.match_type,
                "injection_method": q.injection_method,
                "upstream_call_made": q.upstream_call_made,
                "prompt_tokens": q.prompt_tokens,
                # S12: caching split.
                "cache_read_tokens": q.cache_read_tokens,
                "cache_creation_tokens": q.cache_creation_tokens,
                "completion_tokens": q.completion_tokens,
                "latency_ms": q.latency_ms,
                "sequence_id": q.sequence_id,
                "turn_index": q.turn_index,
                "timestamp": q.timestamp.isoformat(),
            }
            for q in logs
        ],
    })


# ---------------------------------------------------------------------------
# /v1/export, /v1/import — STUB (Phase 7+ cleanup)
# ---------------------------------------------------------------------------

@router.get("/v1/export", response_model=ExportResponse)
async def sdk_export(
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> ExportResponse:
    """Export the customer's bank as fact-level records (the inverse of /v1/import).

    Dumps one record per fact: {key, value, key_is_path, pair_type,
    source_kind, answer_value, crystal_type}. `key` is the fact's stored
    prompt_text (already a finished sparse path); `value` is its claim_text.
    `key_is_path=True` tells /v1/import the key is already a path so it stores
    it verbatim instead of re-deriving one (which would drift/compound the
    path on a restore). The parent crystal's metadata (crystal_type,
    source_kind, answer_value) rides on each record so a round-trip through
    /v1/import can restore the right types.

    Fact-faithful, not crystal-topology-exact (see BACKLOG §12): every fact
    survives, but facts that shared one multi-fact crystal are re-routed
    independently on import. Returns the whole bank in one response
    (inspector scale); production scale would paginate or stream.
    """
    crystals = await store.list_crystals_for_customer(customer.id)
    records: list[dict[str, Any]] = []
    for c in crystals:
        facts = await store.list_facts_for_crystal(c.id)
        for f in facts:
            records.append({
                "key": f.prompt_text,
                "value": f.claim_text,
                # Mark the key as an already-derived sparse path so a
                # round-trip import preserves it verbatim (see sdk_import).
                "key_is_path": True,
                "pair_type": f.pair_type,
                "source_kind": c.source_kind,
                "answer_value": c.answer_value,
                "crystal_type": c.crystal_type,
            })
    return ExportResponse(
        record_count=len(records),
        export_format="jsonl",
        data=records,
    )


@router.post("/v1/import", response_model=ImportResponse)
async def sdk_import(
    body: ImportRequest,
    request: Request,
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> ImportResponse:
    """Import fact-level records into the customer's bank (batch /v1/store).

    Each record {key, value, key_is_path?, pair_type?, source_kind?,
    answer_value?, crystal_type?} is written via add_pair_for_customer. Key
    handling mirrors /v1/store with one branch:
      - key_is_path=True (records from /v1/export): the key is already a
        finished sparse path — store it verbatim (sanitized via format_key),
        do NOT re-derive, so an export->import restore is path-stable.
      - otherwise (raw/seed records): derive a path from key + value via
        generate_sparse_key, exactly as /v1/store does.
    Per-record crystal_type / source_kind / pair_type / answer_value override
    the batch default (body.crystal_type) when present, so a multi-type export
    round-trips. Writes are team-level (unowned, team-readable mode 0o640);
    operator-scoped import is a future refinement.

    wipe=True deletes the customer's existing crystals first (each via
    store.delete_crystal, which cascades to facts + invalidates the vector
    stores). Per-record failures are counted, never fatal — one bad record
    can't abort the batch. A record with an empty key or value is counted as
    an error and skipped.

    Fact-faithful, not crystal-topology-exact — see BACKLOG §12.
    """
    from ..encoding.sparse_keys import generate_sparse_key
    from ..retrieval.sparse_key import format_key

    encoder = request.app.state.prompt_encoder
    vector_store = request.app.state.vector_store
    # Active vector index (Qdrant-aware) for invalidation; fall back to the
    # in-memory fact store. Passed below as delete_crystal's fact_vector_store.
    fact_vector_store = (getattr(request.app.state, "vector_index", None)
                         or getattr(request.app.state, "fact_vector_store", None))

    # Optional wipe: remove the existing bank before importing. Reuses the
    # tested delete_crystal primitive (cascades to facts + invalidates the
    # in-memory stores) rather than a separate bulk path.
    if body.wipe:
        existing = await store.list_crystals_for_customer(customer.id)
        for c in existing:
            try:
                await store.delete_crystal(
                    c.id,
                    customer.id,
                    vector_store=vector_store,
                    fact_vector_store=fact_vector_store,
                )
            except Exception as e:
                logger.warning(
                    "sdk.import.wipe_failed", crystal_id=c.id, error=str(e)
                )

    records_processed = 0
    errors = 0
    seen_crystal_ids: set[str] = set()

    for rec in body.records:
        try:
            key = (rec.get("key") or "").strip()
            value = rec.get("value") or ""
            if not key or not value:
                errors += 1
                continue
            # Exported records carry key_is_path=True: the key is already a
            # finished sparse path, so preserve it verbatim (format_key just
            # sanitizes; it's idempotent on an already-clean path). Raw/seed
            # records get a path derived from key + value, same as /v1/store.
            if rec.get("key_is_path"):
                sparse_key = format_key(key)
            else:
                sparse_key = generate_sparse_key(f"{key}: {value}")
            crystal, _fact = await store.add_pair_for_customer(
                customer_id=customer.id,
                prompt_text=sparse_key,
                answer_text=value,
                pair_type=rec.get("pair_type") or "question_answer",
                encoder=encoder,
                vector_store=vector_store,
                vector_index=getattr(request.app.state, "vector_index", None),
                crystal_type=(
                    rec.get("crystal_type")
                    or body.crystal_type
                    or "customer:legacy"
                ),
                source_kind=rec.get("source_kind") or "model_reasoning",
                answer_value=rec.get("answer_value"),
                owner_operator_id=None,
                group_team_id=None,
                mode=0o640,
            )
            records_processed += 1
            seen_crystal_ids.add(crystal.id)
        except Exception as e:
            errors += 1
            logger.warning("sdk.import.record_failed", error=str(e))

    return ImportResponse(
        records_processed=records_processed,
        crystals_written=len(seen_crystal_ids),
        errors=errors,
    )


# ---------------------------------------------------------------------------
# /v1/subscribe — general-bank subscriptions (FILLED 2026-06-12)
#
# Phase A built the store half (customer.general_crystal_types + the
# FactVectorStore merge + subscription-aware prefix scans) and the CRYS
# seed importer subscribed its own customer directly through the store.
# The SERVER surface stayed a 501 stub — found live when the inspector's
# General Knowledge panel showed nothing for a fully seeded bank. These
# four routes are thin maps onto get/set_customer_general_types; the
# registry (scope='general') is the validation source for what's
# subscribable.
# ---------------------------------------------------------------------------

def _requested_general_types(body: SubscribeRequest) -> list[str]:
    """Collect requested types from either body shape — the UI's singular
    {"crystal_type": ...} or the SDK's batch {"crystal_types": [...]} —
    deduped, order preserved. An honest 422 beats a silent shape
    mismatch: the original list-only schema 422'd the UI's singular
    body before the handler ran and the toggles appeared dead."""
    types: list[str] = []
    single = (getattr(body, "crystal_type", None) or "").strip()
    if single:
        types.append(single)
    for t in getattr(body, "crystal_types", None) or []:
        t = (t or "").strip()
        if t and t not in types:
            types.append(t)
    if not types:
        raise HTTPException(
            status_code=422,
            detail=(
                "body must carry a non-empty 'crystal_type' or "
                "'crystal_types' list"
            ),
        )
    return types


async def _validated_general_type(store: MetadataStore, ct: str) -> str:
    registered = await store.get_crystal_type(ct)
    if registered is None:
        raise HTTPException(
            status_code=404,
            detail=f"crystal_type {ct!r} is not registered",
        )
    if registered.scope != "general":
        raise HTTPException(
            status_code=400,
            detail=(
                f"crystal_type {ct!r} has scope {registered.scope!r} — "
                "only general-scope banks are subscribable"
            ),
        )
    return ct


@router.post("/v1/subscribe")
async def sdk_subscribe(
    body: SubscribeRequest,
    request: Request,
    customer: Annotated[Customer, Depends(require_customer_or_console)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Subscribe the calling customer to one or more general banks
    (idempotent union — existing subscriptions are never dropped)."""
    requested = [
        await _validated_general_type(store, t)
        for t in _requested_general_types(body)
    ]
    subs = await store.get_customer_general_types(customer.id)
    added = [t for t in requested if t not in subs]
    if added:
        subs = [*subs, *added]
        await store.set_customer_general_types(customer.id, subs)
        _invalidate_subscription_cache(request, customer.id)
    return JSONResponse(content={
        "subscribed": added,
        "general_crystal_types": subs,
    })


def _invalidate_subscription_cache(request: Request, customer_id: str) -> None:
    """Drop the FactVectorStore's cached subscription list so a toggle
    takes effect on the customer's NEXT search, not the next process
    restart — the FVS caches customer_id → general types and only
    refreshes after invalidate() (see FactVectorStore._subscribed_types).
    getattr-guarded: test apps that mount this router without a full
    lifespan have no vector index or fact store, and a subscription write
    must still succeed there."""
    idx = (getattr(request.app.state, "vector_index", None)
           or getattr(request.app.state, "fact_vector_store", None))
    if idx is not None:
        idx.invalidate(customer_id)


async def _unsubscribe(
    request: Request, store: MetadataStore, customer: Customer,
    types: list[str],
) -> JSONResponse:
    subs = await store.get_customer_general_types(customer.id)
    removed = [t for t in types if t in subs]
    if removed:
        subs = [t for t in subs if t not in removed]
        await store.set_customer_general_types(customer.id, subs)
        _invalidate_subscription_cache(request, customer.id)
    return JSONResponse(content={
        "unsubscribed": removed,
        "general_crystal_types": subs,
    })


@router.delete("/v1/subscribe/{crystal_type}")
async def sdk_unsubscribe_path(
    crystal_type: str,
    request: Request,
    customer: Annotated[Customer, Depends(require_customer_or_console)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """RESTful unsubscribe — the shape the inspector UI calls. No
    registry validation: removing a stale subscription to a type that
    no longer exists must always work."""
    return await _unsubscribe(request, store, customer, [crystal_type.strip()])


@router.post("/v1/unsubscribe")
async def sdk_unsubscribe(
    body: SubscribeRequest,
    request: Request,
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """v1-parity unsubscribe (body form; accepts singular or batch)."""
    return await _unsubscribe(
        request, store, customer, _requested_general_types(body)
    )


@router.get("/v1/subscriptions")
async def sdk_list_subscriptions(
    customer: Annotated[Customer, Depends(require_customer_or_console)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    return JSONResponse(content={
        "general_crystal_types": await store.get_customer_general_types(customer.id),
    })


# ---------------------------------------------------------------------------
# /v1/crystals/{id}/scope — the SHARE capability (P2/P4, ratified 2026-07-02)
# ---------------------------------------------------------------------------

@router.post("/v1/crystals/{crystal_id}/scope")
async def sdk_set_crystal_scope(
    crystal_id: str,
    body: dict,
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(resolve_principal)
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Share or unshare one crystal: {"scope": "team"} flips group-read on
    (0o640), {"scope": "personal"} flips it off (0o600). One reversible
    mode write — no copy, ownership unchanged.

    Authorization: the crystal's OWNER, or an ADMIN of the owning team
    (P1: the bare team key acts as the Default Admin, so team-key callers
    are admins). Everyone else gets 403. Deliberately a human/API surface
    only — there is NO agent-facing share tool (ratified exclusion: the
    agent never rewires access control on its own initiative).
    """
    customer, operator = principal
    scope = (body or {}).get("scope")
    if scope not in ("personal", "team"):
        raise HTTPException(
            status_code=422, detail="scope must be 'personal' or 'team'",
        )
    crystal = await store.get_crystal(crystal_id)
    if crystal is None or crystal.customer_id != customer.id:
        raise HTTPException(status_code=404, detail="Unknown crystal")

    is_owner = (
        operator is not None
        and crystal.owner_operator_id is not None
        and operator.id == crystal.owner_operator_id
    )
    is_admin = operator is not None and operator.role == "admin"
    if not (is_owner or is_admin):
        raise HTTPException(
            status_code=403,
            detail="Only the crystal's owner or a team admin may change its scope.",
        )

    changed = await store.set_crystal_scope(crystal_id, customer.id, scope)
    if not changed:
        raise HTTPException(status_code=404, detail="Unknown crystal")
    return JSONResponse(content={"crystal_id": crystal_id, "scope": scope})


# ---------------------------------------------------------------------------
# Topology-exact export/import — verdict 5, ratified 2026-07-02
# ---------------------------------------------------------------------------

@router.post("/v1/export/topology")
async def sdk_export_topology(
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Topology-exact export: the bank leaves with its earned trust intact
    — crystal identity, vectors, tiers, scope stamps, chains, co-query
    edges, conflicts, and citation provenance, all verbatim. The
    fact-level /v1/export remains for portable re-routing imports; this is
    the exact-restore format (see import policies on /v1/import/topology).
    """
    payload = await store.export_bank_topology(customer.id)
    return JSONResponse(content={
        "export_format": payload["format"],
        "crystal_count": len(payload["crystals"]),
        "fact_count": len(payload["facts"]),
        "data": payload,
    })


@router.post("/v1/import/topology")
async def sdk_import_topology(
    body: dict,
    request: Request,
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Id-preserving restore of a topology export. Body: the export's
    `data` object (or the full export response — both accepted).

    Policies: customer/group rewritten to the importing team; PK
    collisions skipped and counted (restore into a fresh or wiped bank
    for an exact copy); owners unknown to this team are cleared and
    counted; unknown schema fields dropped and counted.
    """
    payload = body.get("data") if isinstance(body.get("data"), dict) else body
    if not isinstance(payload, dict) or "crystals" not in payload:
        raise HTTPException(
            status_code=422,
            detail="Body must be a topology export (object with 'crystals').",
        )
    counts = await store.import_bank_topology(customer.id, payload)

    # Refresh the in-memory / vec indexes so imports are searchable now.
    for attr in ("vector_store", "vector_index", "fact_vector_store"):
        idx = getattr(request.app.state, attr, None)
        if idx is not None and hasattr(idx, "invalidate"):
            try:
                res = idx.invalidate(customer.id)
                if hasattr(res, "__await__"):
                    await res
            except Exception as e:  # noqa: BLE001 — import succeeded; log only
                logger.warning("sdk.import_topology.invalidate_failed",
                               index=attr, error=str(e))
    return JSONResponse(content=counts)


# ---------------------------------------------------------------------------
# /v1/crystals/{id}/grants — named grants to a group or an operator (P3)
# ---------------------------------------------------------------------------

@router.post("/v1/crystals/{crystal_id}/grants")
async def sdk_add_crystal_grant(
    crystal_id: str,
    body: dict,
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(resolve_principal)
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Grant read on one crystal to a group or an individual operator
    without touching its POSIX mode: {"principal_type": "group"|"operator",
    "principal_id": "..."}. Same owner-or-admin authorization as the scope
    endpoint; same human-surface-only exclusion (no agent tool)."""
    customer, operator = principal
    ptype = (body or {}).get("principal_type")
    pid = (body or {}).get("principal_id")
    if ptype not in ("group", "operator") or not pid:
        raise HTTPException(
            status_code=422,
            detail="principal_type must be 'group' or 'operator' with a principal_id",
        )
    crystal = await store.get_crystal(crystal_id)
    if crystal is None or crystal.customer_id != customer.id:
        raise HTTPException(status_code=404, detail="Unknown crystal")
    is_owner = (
        operator is not None
        and crystal.owner_operator_id is not None
        and operator.id == crystal.owner_operator_id
    )
    is_admin = operator is not None and operator.role == "admin"
    if not (is_owner or is_admin):
        raise HTTPException(
            status_code=403,
            detail="Only the crystal's owner or a team admin may grant access.",
        )
    # Guard the principal belongs to this team (a grant must never point
    # across tenants).
    if ptype == "group":
        groups = await store.list_groups_for_customer(customer.id)
        if pid not in {g["id"] for g in groups}:
            raise HTTPException(status_code=404, detail="Unknown group")
    else:
        target = await store.get_operator_by_id(pid)
        if target is None or target.team_id != customer.id:
            raise HTTPException(status_code=404, detail="Unknown operator")

    from ..models.crystal_type import CrystalAcl

    await store.add_acl(CrystalAcl(
        crystal_id=crystal_id, principal_type=ptype,
        principal_id=pid, grant="read",
    ))
    return JSONResponse(content={
        "crystal_id": crystal_id, "principal_type": ptype,
        "principal_id": pid, "grant": "read",
    })


@router.delete("/v1/crystals/{crystal_id}/grants")
async def sdk_remove_crystal_grant(
    crystal_id: str,
    body: dict,
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(resolve_principal)
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Revoke a named grant — same body and authorization as the grant."""
    customer, operator = principal
    ptype = (body or {}).get("principal_type")
    pid = (body or {}).get("principal_id")
    if ptype not in ("group", "operator") or not pid:
        raise HTTPException(
            status_code=422,
            detail="principal_type must be 'group' or 'operator' with a principal_id",
        )
    crystal = await store.get_crystal(crystal_id)
    if crystal is None or crystal.customer_id != customer.id:
        raise HTTPException(status_code=404, detail="Unknown crystal")
    is_owner = (
        operator is not None
        and crystal.owner_operator_id is not None
        and operator.id == crystal.owner_operator_id
    )
    is_admin = operator is not None and operator.role == "admin"
    if not (is_owner or is_admin):
        raise HTTPException(
            status_code=403,
            detail="Only the crystal's owner or a team admin may revoke access.",
        )
    removed = await store.remove_acl(
        crystal_id=crystal_id, principal_type=ptype,
        principal_id=pid, grant="read",
    )
    if not removed:
        raise HTTPException(status_code=404, detail="No such grant")
    return JSONResponse(content={"crystal_id": crystal_id, "revoked": True})
