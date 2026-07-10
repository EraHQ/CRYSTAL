"""Admin endpoints — /admin/api/*.

Operator-facing views over the audit tables. v1's admin surface
(in ingress/admin.py) included push review queue management,
knowledge-gap browsing, and cognition-task listing. Refactored to
use Phase 5 MetadataStore methods.

Endpoints:
  GET    /admin/api/push-queue                       list pending pushes
  POST   /admin/api/push-queue/{id}/approve          approve + write crystal
  POST   /admin/api/push-queue/{id}/reject           reject
  GET    /admin/api/knowledge-gaps                   list gaps (enriched)
  GET    /admin/api/cognition-tasks                  list tasks
  GET    /admin/api/customers                        list customers (cross-tenant)
  GET    /admin/api/customers/{id}/crystals          list a customer's crystals
  GET    /admin/api/crystals/{id}                    crystal detail
  GET    /admin/api/crystal_types                    crystal_type registry (?scope=)

Phase 6 status: read endpoints work end-to-end. The push-approve
endpoint writes a crystal via `add_pair_for_customer` (port-side this
needs the request app.state vector_store + encoder + decomposer to
exist, which they do post-lifespan).

Note: admin auth in v1 is intentionally simple ("if you can reach
this endpoint internally, you're authorized"). Production deployments
should put a separate Bearer scheme + IP allowlist in front. For
v2 we keep the v1 posture but flag CU-7 (new cleanup item) for proper
admin auth before any public deployment.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..infrastructure import MetadataStore
from ..infrastructure.metadata_store import get_metadata_store
from ..ingress.schema import ChatCompletionRequest, LearnRequest
from .agent import AgentRequest

import dataclasses
from ..config import settings
from ..llm import get_llm_client
from ..scan import scan_for_contradictions

logger = structlog.get_logger(__name__)

router = APIRouter()


class ApprovePushRequest(BaseModel):
    crystal_type: str = "customer:legacy"


class ResolveConflictRequest(BaseModel):
    """Body for the conflict curation gate. `resolution` is one of
    dismissed | qualified | superseded | blacklisted; `loser` ('a'|'b') names
    the losing fact and is required for superseded/blacklisted (ignored
    otherwise)."""
    resolution: str
    loser: Optional[str] = None


# --- Push review queue ---

@router.get("/admin/api/push-queue")
async def list_review_queue(
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    customer_id: str,  # required query param — admin scopes per customer
    status: Optional[str] = None,
    limit: int = 50,
) -> JSONResponse:
    """List push-review queue items for one customer.

    Returns pending items by default. Pass ?status=approved or rejected
    to see post-decision history.
    """
    # Pinned tenants see exactly their own queue (2026-07-07 sweep).
    customer_id = getattr(request.state, "tenant_pin", None) or customer_id
    items = await store.list_push_review_items(
        customer_id=customer_id,
        status=status,
        limit=limit,
    )
    return JSONResponse(content={
        "items": [
            {
                "id": it.id,
                "customer_id": it.customer_id,
                "key": it.key,
                "value": it.value,
                "confidence": it.confidence,
                "source": it.source,
                "status": it.status,
                "crystal_id": it.crystal_id,
                "source_query_id": it.source_query_id,
                "reviewed_at": it.reviewed_at.isoformat() if it.reviewed_at else None,
                "created_at": it.created_at.isoformat(),
            }
            for it in items
        ],
        "count": len(items),
    })


@router.post("/admin/api/push-queue/{item_id}/approve")
async def approve_review_item(
    item_id: str,
    body: ApprovePushRequest,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    customer_id: str,
) -> JSONResponse:
    """Approve a push: write the (key, value) as a crystal and mark approved."""
    item = await store.get_push_review_item(item_id, customer_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Review item not found")
    if item.status != "pending":
        raise HTTPException(
            status_code=409,
            detail=f"Item is in status {item.status!r}, not pending",
        )

    # Write the crystal
    crystal, _fact = await store.add_pair_for_customer(
        customer_id=customer_id,
        prompt_text=item.key,
        answer_text=item.value,
        pair_type="question_answer",
        encoder=request.app.state.prompt_encoder,
        vector_store=request.app.state.vector_store,
        vector_index=getattr(request.app.state, "vector_index", None),
        crystal_type=body.crystal_type,
        source_kind="model_reasoning",
    )

    await store.mark_push_review_approved(
        item_id=item_id,
        crystal_id=crystal.id,
        reviewed_at=datetime.now(timezone.utc),
    )
    logger.info(
        "admin.push.approved",
        item_id=item_id,
        crystal_id=crystal.id,
        customer_id=customer_id,
    )
    return JSONResponse(content={"approved": True, "crystal_id": crystal.id})


@router.post("/admin/api/push-queue/{item_id}/reject")
async def reject_review_item(
    item_id: str,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    customer_id: str,
) -> JSONResponse:
    item = await store.get_push_review_item(item_id, customer_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Review item not found")
    if item.status != "pending":
        raise HTTPException(
            status_code=409,
            detail=f"Item is in status {item.status!r}, not pending",
        )
    await store.mark_push_review_rejected(
        item_id=item_id,
        reviewed_at=datetime.now(timezone.utc),
    )
    return JSONResponse(content={"rejected": True})


# --- Knowledge gaps ---

@router.get("/admin/api/knowledge-gaps")
async def list_knowledge_gaps(
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    customer_id: str,
    limit: int = 50,
) -> JSONResponse:
    """List knowledge gaps with filling-crystal snippet enrichment.

    Uses the AN-6 resolution from Phase 5:
    `list_knowledge_gaps_with_filled_content` encapsulates the 1+N
    FactRow fetch inside the store. Acceptable at limit=50.
    """
    customer_id = getattr(request.state, "tenant_pin", None) or customer_id
    enriched = await store.list_knowledge_gaps_with_filled_content(
        customer_id=customer_id, limit=limit,
    )
    return JSONResponse(content={
        "gaps": [
            {
                "id": gap.id,
                "customer_id": gap.customer_id,
                "domain": gap.domain,
                "subject": gap.subject,
                "missing": gap.missing,
                "priority": gap.priority,
                "status": gap.status,
                "source": gap.source,
                "filled_by_crystal_id": gap.filled_by_crystal_id,
                "filled_snippet": snippet,
                "created_at": gap.created_at.isoformat(),
                "resolved_at": gap.resolved_at.isoformat() if gap.resolved_at else None,
            }
            for (gap, snippet) in enriched
        ],
        "count": len(enriched),
    })


# --- Cognition tasks ---

@router.get("/admin/api/chat/sessions")
async def list_chat_sessions_endpoint(
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    customer_id: str,
    limit: int = 50,
) -> JSONResponse:
    """S7: playground chat history — sessions for the sidebar. Tenant
    principals are force-scoped by the pin (same contract as every
    console read)."""
    customer_id = getattr(request.state, "tenant_pin", None) or customer_id
    sessions = await store.list_chat_sessions(
        customer_id, limit=max(1, min(limit, 100))
    )
    return JSONResponse(content={"sessions": sessions, "count": len(sessions)})


@router.get("/admin/api/chat/sessions/{sequence_id}")
async def get_chat_session_endpoint(
    sequence_id: str,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    customer_id: str,
) -> JSONResponse:
    """S7: one session's ordered transcript. Customer-scoped in the
    reader — a foreign sequence_id returns an empty list, uniform with
    the 404-shaped reads elsewhere."""
    customer_id = getattr(request.state, "tenant_pin", None) or customer_id
    turns = await store.get_session_transcript(customer_id, sequence_id)
    # S8: enrich each turn with its tool calls from the reasoning trace.
    # Alignment is positional (see get_session_tool_calls) — a missing
    # trace shifts nothing worse than that turn's chips.
    tool_calls = await store.get_session_tool_calls(customer_id, sequence_id)
    for i, turn in enumerate(turns):
        turn["tool_calls"] = tool_calls[i] if i < len(tool_calls) else []
    return JSONResponse(content={
        "sequence_id": sequence_id, "turns": turns, "count": len(turns),
    })


@router.get("/admin/api/cognition-tasks")
async def list_cognition_tasks(
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    customer_id: str,
    status: Optional[str] = None,
    limit: int = 50,
) -> JSONResponse:
    customer_id = getattr(request.state, "tenant_pin", None) or customer_id
    tasks = await store.list_cognition_tasks(
        customer_id=customer_id,
        status=status,
        limit=limit,
    )
    return JSONResponse(content={
        "tasks": [
            {
                "id": t.id,
                "customer_id": t.customer_id,
                "task_type": t.task_type,
                "payload": t.payload,
                "priority": t.priority,
                "status": t.status,
                "result": t.result,
                "result_crystal_id": t.result_crystal_id,
                "source_query_id": t.source_query_id,
                "error_message": t.error_message,
                "created_at": t.created_at.isoformat(),
                "started_at": t.started_at.isoformat() if t.started_at else None,
                "completed_at": t.completed_at.isoformat() if t.completed_at else None,
            }
            for t in tasks
        ],
        "count": len(tasks),
    })


# --- Customers + crystals (admin views) ---

@router.get("/admin/api/customers")
async def admin_list_customers(
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    limit: int = 1000,
) -> JSONResponse:
    """List all customers across tenants. Cross-tenant by design.

    Includes a `crystal_count` per customer so the inspector can show
    bank size at a glance.
    """
    customers = await store.list_customers(limit=limit)
    rows = []
    for c in customers:
        try:
            count = await store.count_crystals_for_customer(c.id)
        except Exception:
            count = None
        rows.append({
            "id": c.id,
            "provider": c.model_routing_config.provider,
            "model_id": c.model_routing_config.model_id,
            "crystal_count": count,
            "created_at": c.created_at.isoformat(),
        })
    return JSONResponse(content={"customers": rows, "count": len(rows)})


@router.get("/admin/api/customers/{customer_id}/spend")
async def get_customer_spend(
    customer_id: str,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
):
    """The tenant console's usage view (Phase C, 2026-07-06): ledger
    totals plus the managed month-to-date against the tier cap. Rides
    the tenant guard — a tenant principal reaches only its own id here
    (foreign ids 404 at the middleware).
    """
    from ..control.admission import resolve_tier

    customer = await store.get_customer_by_id(customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")

    totals = await store.cost_totals_for_team(customer_id)
    managed_mtd = await store.managed_spend_micro_usd_this_month(customer_id)
    cap = resolve_tier(
        customer.subscription_tier
    ).monthly_managed_budget_micro_usd
    return {
        "customer_id": customer_id,
        "inference_mode": customer.inference_mode,
        "subscription_tier": customer.subscription_tier,
        # BYOK UX (2026-07-10): boolean ONLY — the Settings page needs
        # "a provider key is stored" to render truthfully (its input
        # clears on save, which read as "didn't save"), and to explain
        # WHY the byok flip is refused when no key exists. Never the
        # ref itself.
        "has_upstream_key": bool(
            getattr(
                getattr(customer, "model_routing_config", None),
                "api_key_ref",
                None,
            )
        ),
        "totals": totals,
        "managed_month_to_date_micro_usd": managed_mtd,
        "managed_monthly_cap_micro_usd": cap,
    }


@router.get("/admin/api/customers/{customer_id}/crystals")
async def admin_list_customer_crystals(
    customer_id: str,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    limit: int = 50,
    offset: int = 0,
) -> JSONResponse:
    """Paginated crystal listing for the inspector's bank browser."""
    total, crystals = await store.list_crystals_for_customer_paginated(
        customer_id=customer_id, limit=limit, offset=offset,
    )
    # P3 (Unify-Agents bank readability): attach each crystal's
    # representative sparse key so the inspector can render a human
    # breadcrumb + title and classify the crystal (Reflections|… /
    # General|… / Code|…) without a per-row detail fetch. One cheap
    # vector-free query for the whole page.
    headlines = await store.headline_facts_for_crystals([c.id for c in crystals])
    return JSONResponse(content={
        "total": total,
        "offset": offset,
        "limit": limit,
        "crystals": [
            {
                "id": c.id,
                "customer_id": c.customer_id,
                "crystal_type": c.crystal_type,
                "build_method": c.build_method,
                "summary_text": c.summary_text,
                "fact_count": c.fact_count,
                "quality_tier": c.quality_tier,
                "headline_key": (headlines.get(c.id) or {}).get("key"),
                "headline_claim": (headlines.get(c.id) or {}).get("claim"),
                "headline_source_kind": (headlines.get(c.id) or {}).get("source_kind"),
                "created_at": c.created_at.isoformat(),
                "last_activity": c.last_activity.isoformat() if c.last_activity else None,
            }
            for c in crystals
        ],
    })


@router.get("/admin/api/crystals/{crystal_id}")
async def admin_get_crystal(
    request: Request,
    crystal_id: str,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Crystal detail + its facts."""
    crystal = await store.get_crystal(crystal_id)
    if crystal is None:
        raise HTTPException(status_code=404, detail="Crystal not found")
    # Pinned tenants may only open their OWN crystals (2026-07-07 sweep);
    # a foreign id gets the identical 404 — never an existence oracle.
    _pin = getattr(request.state, "tenant_pin", None)
    if _pin and crystal.customer_id != _pin:
        raise HTTPException(status_code=404, detail="Crystal not found")
    facts = await store.list_facts_for_crystal(crystal_id)
    return JSONResponse(content={
        "crystal": {
            "id": crystal.id,
            "customer_id": crystal.customer_id,
            "crystal_type": crystal.crystal_type,
            "summary_text": crystal.summary_text,
            "fact_count": crystal.fact_count,
            "quality_tier": crystal.quality_tier,
            "build_method": crystal.build_method,
            "parent_crystal_id": crystal.parent_crystal_id,
            "encoder_fingerprint": crystal.encoder_fingerprint,
            "created_at": crystal.created_at.isoformat(),
            "last_activity": crystal.last_activity.isoformat() if crystal.last_activity else None,
        },
        "facts": [
            {
                "id": f.id,
                "claim_text": f.claim_text,
                "pair_type": f.pair_type,
                "source_kind": f.source_kind,
                "prompt_text": f.prompt_text,
                "answer_value": f.answer_value,
                "created_at": f.created_at.isoformat(),
            }
            for f in facts
        ],
    })


# --- Inspector helpers: admin key + query logs (frontend port) ---
#
# The inspector UI needs two reads that v1's ingress/admin.py exposed but
# the v2 endpoints/ reorg hadn't restored:
#   - admin_key: lets the Chat playground act as the customer (Bearer) to
#     call /v1/chat/completions, /v1/learn, /v1/stats.
#   - query_logs (keyless admin): the Query Log page and the playground's
#     log-matching read this without holding the customer key.
# Both follow the existing admin posture (keyless, trusted-internal; CU-7
# tracks real admin auth before any public deploy). R9-clean: store methods
# only, no inline SQL.

@router.get("/admin/api/customers/{customer_id}/admin_key")
async def admin_get_customer_key(
    customer_id: str,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Deprecated by no-plaintext (2026-06-13): API keys are stored hashed
    and CANNOT be retrieved. Returns 410.

    The raw Key A is shown exactly once, at customer creation. The
    inspector's Chat playground used to fetch a customer's key here to
    call the Bearer-protected /v1/* endpoints; retrieving a key is
    fundamentally incompatible with no-plaintext-at-rest. The intended
    replacement is a keyless admin chat proxy (the admin surface is
    already trusted-internal), tracked as a follow-up. 404 still wins if
    the customer doesn't exist, so a missing customer stays
    distinguishable from the deprecation.
    """
    customer = await store.get_customer_by_id(customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    raise HTTPException(
        status_code=410,
        detail=(
            "Customer API keys are hashed and cannot be retrieved. Use the "
            "key shown once at creation. (A keyless admin chat proxy will "
            "replace this for the inspector playground.)"
        ),
    )


@router.post("/admin/api/customers/{customer_id}/chat")
async def admin_customer_chat(
    customer_id: str,
    body: ChatCompletionRequest,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
):
    """Keyless admin chat proxy (inspector playground) — the replacement for
    the deprecated admin_key fetch above.

    Runs the full /v1/chat/completions pipeline for a customer resolved by
    path id instead of by Bearer Key A (keys are hashed and unretrievable
    since no-plaintext, 2026-06-13). Same trusted-internal admin posture as
    the rest of /admin/api/* — CU-7 tracks real admin auth before any public
    deploy. The customer's retrieval / learning / logging run exactly as the
    public proxy would; only the auth source differs. 404 if the customer
    doesn't exist.
    """
    from .chat_proxy import run_chat_completion

    customer = await store.get_customer_by_id(customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return await run_chat_completion(
        body=body, request=request, customer=customer, store=store,
    )


@router.post("/admin/api/customers/{customer_id}/agent")
async def admin_customer_agent(
    customer_id: str,
    body: AgentRequest,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
):
    """Keyless admin agent run (inspector playground) — drives CRYS (the
    agent) on the customer's message history instead of the single-shot
    proxy. Same trusted-internal admin posture as the rest of /admin/api/*
    (CU-7 tracks real admin auth before any public deploy); resolves the
    customer by path id and delegates to the shared `run_agent_messages`
    pipeline so the playground reaches CRYS without the customer's Bearer
    key. 404 if the customer doesn't exist.

    Body is Anthropic Messages API-shaped (same as POST /v1/agent/messages):
    {messages: [...], model?, max_tokens?, system?, metadata?}.
    """
    from .agent import run_agent_messages

    customer = await store.get_customer_by_id(customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return await run_agent_messages(
        body=body, request=request, customer=customer, store=store,
    )


@router.post("/admin/api/customers/{customer_id}/learn")
async def admin_customer_learn(
    customer_id: str,
    body: LearnRequest,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
):
    """Keyless admin learn (inspector playground feedback). Same trusted-
    internal posture as the rest of /admin/api/*; resolves the customer by
    path id and delegates to the shared run_learn helper so the playground's
    thumbs-up / thumbs-down can teach the bank without the customer's Bearer
    key. 404 if the customer doesn't exist.
    """
    from .sdk import run_learn

    customer = await store.get_customer_by_id(customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return await run_learn(
        body=body, request=request, customer=customer, store=store,
    )


@router.get("/admin/api/customers/{customer_id}/query_logs")
async def admin_list_query_logs(
    customer_id: str,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    limit: int = 100,
    offset: int = 0,
) -> JSONResponse:
    """List a customer's query logs for the inspector (keyless admin).

    Mirrors the SDK /v1/query_logs shape but under the admin surface so
    the Query Log page doesn't need the customer's Bearer key. The
    inspector reads `items`; `matched_facts` is included (defaulted to an
    empty list) because the playground and log views index it. Optional
    fields are read via getattr so a leaner QueryLogRow can't 500 this.
    """
    total, logs = await store.list_query_logs_for_customer(
        customer_id=customer_id, limit=limit, offset=offset,
    )
    return JSONResponse(content={
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": [
            {
                "id": q.id,
                "timestamp": q.timestamp.isoformat(),
                "query_text": q.query_text,
                "match_type": q.match_type,
                "injection_method": q.injection_method,
                "matched_facts": getattr(q, "matched_facts", None) or [],
                "response_text": getattr(q, "response_text", None),
                "prompt_tokens": q.prompt_tokens,
                "completion_tokens": q.completion_tokens,
                # S12 fix (2026-07-09): the console Logs pane reads THIS
                # serializer, not sdk /v1/query_logs — the cache split
                # shipped in v12 but only on the SDK shape, so the pane
                # kept showing dashes. Getattr-safe like its neighbors.
                "cache_read_tokens": getattr(q, "cache_read_tokens", None),
                "cache_creation_tokens": getattr(q, "cache_creation_tokens", None),
                "sequence_id": getattr(q, "sequence_id", None),
                "turn_index": getattr(q, "turn_index", None),
                "prompt_token_overhead": getattr(q, "prompt_token_overhead", None),
                "shadow_ran": getattr(q, "shadow_ran", None),
                "shadow_delta": getattr(q, "shadow_delta", None),
                "latency_ms": q.latency_ms,
            }
            for q in logs
        ],
    })


# --- Inspector Activity view: agent sessions + dependencies + commands ---
#
# The Foundation F4 session registry (live agents) + G2 control commands,
# surfaced for the inspector. Keyless, scoped by ?customer_id=, same
# trusted-internal posture as the rest of /admin/api/* (CU-7 tracks real
# admin auth before any public deploy). The Bearer-authed /v1/sessions/*
# routes serve agents + operators; this is the inspector's read of the SAME
# rows. The session store methods already return plain dicts whose datetimes
# FastAPI's encoder serializes, so these return a dict directly rather than
# the manual-isoformat JSONResponse the older routes use. Liveness (is_stale
# / effective_status) is derived per row by the registry; the list route also
# sweeps stale sessions so the view self-heals — without a heartbeating client
# nothing else would flip a dead row to 'crashed'. R9-clean: store methods
# only.

@router.get("/admin/api/sessions")
async def admin_list_sessions(
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    customer_id: str,
    include_terminal: bool = True,
) -> dict[str, Any]:
    """List a team's agent sessions for the inspector Activity view.

    Newest-heartbeat first; each row carries derived liveness
    (effective_status reads 'crashed' for a stale non-terminal session).
    Sweeps stale sessions first so the materialized status + orphaned deps
    stay truthful even with no agent currently heartbeating.
    """
    await store.mark_stale_sessions()
    sessions = await store.list_sessions_for_team(
        customer_id, include_terminal=include_terminal,
    )
    return {"sessions": sessions, "count": len(sessions)}


@router.get("/admin/api/sessions/{session_id}/dependencies")
async def admin_list_session_dependencies(
    session_id: str,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    customer_id: str,
) -> dict[str, Any]:
    """A session's spawned dependencies (mcp_server | subprocess | browser |
    queued_task | pip_env). customer_id scopes the lookup so a session id
    outside the team 404s rather than leaking."""
    session = await store.get_session(session_id)
    if session is None or session["team_id"] != customer_id:
        raise HTTPException(status_code=404, detail="session not found")
    deps = await store.list_dependencies_for_session(session_id)
    return {"dependencies": deps, "count": len(deps)}


@router.get("/admin/api/sessions/{session_id}/commands")
async def admin_list_session_commands(
    session_id: str,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    customer_id: str,
    status: Optional[str] = None,
) -> dict[str, Any]:
    """A session's control-plane commands (G2): approval decisions + terminate
    commands, with their pending/consumed/voided status. Team-scoped via
    customer_id (cross-team 404)."""
    session = await store.get_session(session_id)
    if session is None or session["team_id"] != customer_id:
        raise HTTPException(status_code=404, detail="session not found")
    commands = await store.list_commands_for_session(session_id, status=status)
    return {"commands": commands, "count": len(commands)}


# --- Inspector Agents view: the CRYS event stream + daemon queue + gaps ---
#
# The Unify-Agents surface. agent_events is the per-session activity timeline
# (turns, tool calls, subagents, crystals, gaps); the daemon's agent_tasks
# queue + agent-run gaps are the background work. Keyless, scoped by
# ?customer_id=, same posture as the sessions routes above; store methods
# only (R9). Events are team-scoped via the owning session's team check
# (cross-team 404, like deps/commands); tasks scope natively by customer;
# gaps are filtered to the customer in-route (list_agent_gaps is
# source-scoped, not team-scoped).

@router.get("/admin/api/agents/events")
async def admin_list_agent_events(
    session_id: str,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    customer_id: str,
    after_seq: Optional[int] = None,
    limit: int = 500,
) -> dict[str, Any]:
    """A session's event stream (turns, tool calls, subagents, crystals,
    gaps) in seq order. `after_seq` returns only newer events — the live
    timeline's incremental poll. Team-scoped via the owning session."""
    session = await store.get_session(session_id)
    if session is None or session["team_id"] != customer_id:
        raise HTTPException(status_code=404, detail="session not found")
    events = await store.list_events_for_session(
        session_id, after_seq=after_seq, limit=limit,
    )
    return {"events": events, "count": len(events)}


@router.get("/admin/api/agents/tasks")
async def admin_list_agent_tasks(
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    customer_id: str,
    limit: int = 50,
) -> dict[str, Any]:
    """The daemon's background task queue for a team (queued/running/done/
    failed, recurrence, report/error), newest first."""
    tasks = await store.list_agent_tasks(customer_id, limit=limit)
    return {"tasks": tasks, "count": len(tasks)}


@router.get("/admin/api/agents/gaps")
async def admin_list_agent_gaps(
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    customer_id: str,
    limit: int = 50,
) -> dict[str, Any]:
    """Agent-run gaps (terminal background failures, retryable) for a team.
    list_agent_gaps is scoped to source='agent_run' globally, so filter to
    the customer in-route."""
    gaps = await store.list_agent_gaps(limit=limit)
    scoped = [g for g in gaps if g.get("customer_id") == customer_id]
    return {"gaps": scoped, "count": len(scoped)}


@router.get("/admin/api/crystal_types")
async def admin_list_crystal_types(
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    scope: Optional[str] = None,
) -> JSONResponse:
    """List the crystal_type registry for the inspector (keyless admin).

    The Knowledge page's General Knowledge Banks panel reads this to
    render subscribable banks (it filters scope === 'general' client-
    side; the optional ?scope= param does the same server-side for
    other consumers). Registry rows are created by migrations (the
    legacy pair) and by import_general_bank's create-if-missing upsert
    — the 2026-06-12 fix that made seeded banks discoverable, not just
    retrievable. R9-clean: store method only, no inline SQL.
    """
    types = await store.list_crystal_types_by_scope(scope)
    return JSONResponse(content={
        "count": len(types),
        "items": [
            {
                "id": t.id,
                "display_name": t.display_name,
                "scope": t.scope,
            }
            for t in types
        ],
    })


# --- Inspector Convergence view: knowledge conflicts + unified backlog ---
#
# Never-Idle Convergence (docs/NEVER_IDLE_CONVERGENCE.md). The contradiction
# scan surfaces knowledge_conflicts; list_backlog is the one ranked view over
# every waiting-work queue. Keyless, scoped by ?customer_id=, same trusted-
# internal posture as the rest of /admin/api/* (CU-7 tracks real admin auth
# before any public deploy). Conflicts are returned via model_dump(mode="json")
# so datetimes serialize; backlog items are plain dicts whose datetimes
# FastAPI's encoder serializes on a plain-dict return (the sessions/agents
# route precedent). R9-clean: store methods only.

@router.get("/admin/api/conflicts")
async def admin_list_conflicts(
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    customer_id: str,
    status: Optional[str] = None,
    limit: int = 50,
) -> dict[str, Any]:
    """List a customer's knowledge conflicts (open by default; pass
    ?status=resolved|dismissed for history)."""
    customer_id = getattr(request.state, "tenant_pin", None) or customer_id
    conflicts = await store.list_knowledge_conflicts(
        customer_id, status=status, limit=limit,
    )
    return {
        "conflicts": [c.model_dump(mode="json") for c in conflicts],
        "count": len(conflicts),
    }


@router.get("/admin/api/backlog")
async def admin_list_backlog(
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    customer_id: str,
    limit: int = 100,
) -> dict[str, Any]:
    """The unified backlog: one ranked view over the customer's waiting work
    across gaps, conflicts, cognition/agent tasks, review, and verification.
    Highest priority first, oldest-first within a priority."""
    customer_id = getattr(request.state, "tenant_pin", None) or customer_id
    items = await store.list_backlog(customer_id, limit=limit)
    return {"items": items, "count": len(items)}


@router.post("/admin/api/conflicts/scan")
async def admin_scan_conflicts(
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    customer_id: str,
    max_calls: Optional[int] = None,
    max_pairs: Optional[int] = None,
) -> dict[str, Any]:
    """Run the contradiction scan for one customer on demand (the "audit my
    bank now" path). Surfacing-only; budget-bounded by max_calls / max_pairs
    (defaulting to the convergence settings). Runs regardless of
    enable_convergence_scan — this is an explicit operator action. 503 if no
    LLM provider is configured.
    """
    if not get_llm_client().is_ready():
        raise HTTPException(
            status_code=503,
            detail="No LLM provider configured (set CC_LLM_API_KEY or ANTHROPIC_API_KEY)",
        )
    result = await scan_for_contradictions(
        store=store,
        customer_id=customer_id,
        max_candidate_pairs=(
            max_pairs if max_pairs is not None
            else settings.convergence_max_pairs_per_scan
        ),
        max_discriminator_calls=(
            max_calls if max_calls is not None
            else settings.convergence_max_calls_per_cycle
        ),
    )
    return {"scan": dataclasses.asdict(result)}


@router.post("/admin/api/conflicts/{conflict_id}/resolve")
async def admin_resolve_conflict(
    conflict_id: str,
    body: ResolveConflictRequest,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> dict[str, Any]:
    """Curation gate: settle a conflict and apply its effect. superseded /
    blacklisted deactivate the losing fact (grating→0; blacklisted also
    records the wrong claim); qualified keeps both; dismissed is a no-op.
    Non-destructive. 400 on a bad resolution or a missing loser where one is
    required; 404 if the conflict doesn't exist."""
    try:
        updated = await store.apply_conflict_resolution(
            conflict_id,
            resolution=body.resolution,
            loser=body.loser,
            resolved_at=datetime.now(timezone.utc),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if updated is None:
        raise HTTPException(status_code=404, detail="Conflict not found")
    return {"conflict": updated.model_dump(mode="json")}
