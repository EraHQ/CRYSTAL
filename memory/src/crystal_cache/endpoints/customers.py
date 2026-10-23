"""Customer CRUD — /v1/customers/*

Verbatim port from v1's app.py.

Phase 6.5 P4.1 / CU-8: `update_upstream_key` now calls
`store.update_customer_upstream_key`, closing the only remaining
inline-SQL violation in v2 endpoints.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import StringConstraints

from ..infrastructure import MetadataStore
from ..hygiene import SecretFormatError, clean_secret
from ..infrastructure.metadata_store import get_metadata_store
from ..ingress.auth import require_customer_self_or_admin
from ..ingress.errors import build_error_envelope
from ..ingress.limits import DAYS, ID64, Bound, require_bounds
from ..ingress.schema import (
    CreateCustomerRequest,
    CreateCustomerResponse,
    GetCustomerResponse,
)

logger = structlog.get_logger(__name__)

# Item 2 (2026-10-22): the providers a tenant's own key can belong to.
# "nebius" is a Token Factory key (checked on save, served by the lane).
UPSTREAM_PROVIDERS: tuple[str, ...] = ("anthropic", "openai", "nebius", "self_hosted")


router = APIRouter()


@router.post(
    "/v1/customers",
    response_model=CreateCustomerResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_customer(
    body: CreateCustomerRequest,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> CreateCustomerResponse:
    """Create a customer and return their Crystal Cache API key (Key A).

    The caller provides upstream routing config including their upstream
    provider's API key (Key B). We generate Key A server-side; it's
    returned ONCE and must be stored by the caller.
    """
    if body.provider == "self_hosted" and not body.base_url:
        raise HTTPException(
            status_code=400,
            detail="self_hosted provider requires base_url",
        )
    # B1 (2026-09-30): refuse a Key B carrying whitespace or control
    # characters BEFORE the row exists, so a bad paste never lands.
    try:
        api_key_ref = (
            clean_secret(body.api_key_ref, field="api_key_ref")
            if body.api_key_ref else body.api_key_ref
        )
    except SecretFormatError:
        raise HTTPException(
            status_code=400,
            detail="api_key_ref contains whitespace or control characters; paste the key alone",
        )

    customer = await store.create_customer(
        provider=body.provider,
        model_id=body.model_id,
        api_key_ref=api_key_ref,
        base_url=body.base_url,
        injection_preference=body.injection_preference,
        shadow_sample_rate=body.shadow_sample_rate,
    )
    logger.info(
        "customer.created",
        customer_id=customer.id,
        provider=body.provider,
        model_id=body.model_id,
    )
    return CreateCustomerResponse(
        id=customer.id,
        api_key=customer.api_key,
        provider=customer.model_routing_config.provider,
        model_id=customer.model_routing_config.model_id,
    )


@router.get("/v1/customers/{customer_id}", response_model=GetCustomerResponse)
async def get_customer(
    customer_id: ID64,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> GetCustomerResponse:
    # B1 (2026-07-03): self-or-admin only. Was unauthenticated.
    customer = await require_customer_self_or_admin(customer_id, request, store)
    return GetCustomerResponse(
        id=customer.id,
        provider=customer.model_routing_config.provider,
        model_id=customer.model_routing_config.model_id,
        background_model_id=customer.model_routing_config.background_model_id,
        base_url=customer.model_routing_config.base_url,
        injection_preference=customer.injection_preference,
        shadow_sample_rate=customer.shadow_sample_rate,
        assumptions_explore=customer.assumptions_explore,
        created_at=customer.created_at.isoformat(),
    )


@router.patch("/v1/customers/{customer_id}/upstream_key")
async def update_upstream_key(
    customer_id: ID64,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Update a customer's upstream API key (Key B).

    Used by the chat playground to set/change the Anthropic/OpenAI
    key without recreating the customer.

    B1 (2026-07-03): self-or-admin authorization. This route was
    UNAUTHENTICATED — anyone who knew a customer_id could overwrite the
    upstream key. Now the caller must be the customer itself (Key A) or
    the platform admin.

    Per Phase 6.5 P4.1, this uses `update_customer_upstream_key` rather
    than inline SQLAlchemy. CU-8 closed.
    """
    # Authorize BEFORE reading the body or touching the key.
    await require_customer_self_or_admin(customer_id, request, store)

    body = await request.json()
    require_bounds(body, {"api_key_ref": Bound(max_len=512), "provider": Bound(max_len=32),
                          "base_url": Bound(max_len=256)})  # Q46 (PR-6)
    new_key = body.get("api_key_ref", "")
    if not new_key:
        raise HTTPException(status_code=400, detail="api_key_ref is required")
    # B1 (2026-09-30): a key with a trailing space or newline used to be
    # stored as-is, then echoed by httpx in every upstream error.
    try:
        new_key = clean_secret(new_key, field="api_key_ref")
    except SecretFormatError:
        raise HTTPException(
            status_code=400,
            detail="api_key_ref contains whitespace or control characters; paste the key alone",
        )
    if not new_key:
        raise HTTPException(status_code=400, detail="api_key_ref is required")
    # Item 2 (2026-10-22): the key's provider rides along. "nebius" is a
    # Token Factory key: checked against /v1/models before it is stored,
    # and the tenant's picks are moved onto the lane if they were not
    # Token Factory models already.
    provider = (body.get("provider") or "").strip().lower() or None
    base_url = (body.get("base_url") or "").strip() or None
    if provider is not None and provider not in UPSTREAM_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=build_error_envelope(
                "provider must be one of " + ", ".join(UPSTREAM_PROVIDERS),
                error_type="invalid_request_error", param="provider", code="provider_unknown",
            ),
        )
    if base_url is not None and (provider != "self_hosted" or not base_url.startswith("https://")):
        raise HTTPException(
            status_code=400,
            detail=build_error_envelope(
                "base_url is only for a self-hosted endpoint and must start with https://",
                error_type="invalid_request_error", param="base_url", code="base_url_invalid",
            ),
        )
    models_seen: Optional[int] = None
    if provider == "nebius":
        import asyncio

        from ..llm.catalog import verify_key

        try:
            models_seen = await asyncio.to_thread(verify_key, new_key)
        except Exception as e:  # noqa: BLE001 — the reason is logged redacted; the caller gets a code
            logger.warning("customer.upstream_key_rejected", customer_id=customer_id,
                           provider=provider, error=type(e).__name__)
            raise HTTPException(
                status_code=400,
                detail=build_error_envelope(
                    "Token Factory did not accept that key. Check it in the Nebius console and paste it again.",
                    error_type="invalid_request_error", param="api_key_ref", code="provider_key_rejected",
                ),
            )

    updated = await store.update_customer_upstream_key(
        customer_id=customer_id,
        new_api_key_ref=new_key,
        provider=provider,
        base_url=base_url,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Customer not found")
    moved: dict[str, Optional[str]] = {}
    if provider == "nebius":
        from ..config import settings
        from ..llm.lanes import family_of

        customer = await store.get_customer_by_id(customer_id)
        cfg = getattr(customer, "model_routing_config", None)
        if family_of(getattr(cfg, "model_id", None)) != "nemotron":
            await store.set_customer_model(customer_id, settings.nemotron_model_large)
            moved["model_id"] = settings.nemotron_model_large
        bg = getattr(cfg, "background_model_id", None)
        if bg and family_of(bg) != "nemotron":
            await store.set_customer_background_model(customer_id, None)
            moved["background_model_id"] = None

    logger.info(
        "customer.upstream_key_updated",
        customer_id=customer_id,
        provider=provider,
        models_seen=models_seen,
    )
    return JSONResponse(content={"updated": True, "customer_id": customer_id,
                                 "provider": provider, "models_seen": models_seen, "moved": moved})


@router.patch("/v1/customers/{customer_id}/assumptions_explore")
async def set_assumptions_explore(
    customer_id: ID64,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """The tenant's assumptions explore toggle (Q5=A, funnel F3).

    Body: {"assumptions_explore": true | false | null} — tri-state:
    true/false are explicit choices; null REVERTS to the deployment
    default (explore ON, "assumptions on everything"). False withholds
    STRUCTURAL-tier funnel edges (key_adjacent / vector_similar) at
    verdict-spend time, so only demand-evidenced pairs cost model
    calls. Self-or-admin — it's the tenant's spend posture.
    """
    await require_customer_self_or_admin(customer_id, request, store)

    body = await request.json()
    if "assumptions_explore" not in body:
        raise HTTPException(
            status_code=400,
            detail="assumptions_explore is required (true, false, or null)",
        )
    value = body["assumptions_explore"]
    if value is not None and not isinstance(value, bool):
        raise HTTPException(
            status_code=400,
            detail="assumptions_explore must be true, false, or null",
        )

    updated = await store.set_customer_assumptions_explore(
        customer_id, value,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Customer not found")

    logger.info(
        "customer.assumptions_explore_set",
        customer_id=customer_id,
        explore=value,
    )
    return JSONResponse(content={
        "customer_id": customer_id,
        "assumptions_explore": value,
    })


# Step-up auth window for key rotation: a JWT principal's last sign-in
# (auth_time) must be within this many seconds. 5 minutes = long enough
# for the reauth popup round-trip, short enough that an idle hijacked
# session cannot rotate.
ROTATE_MAX_AUTH_AGE_SECONDS = 300


@router.get("/v1/customers/{customer_id}/spend/origins")
async def customer_spend_origins(
    customer_id: ID64,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    days: Optional[DAYS] = None,
) -> JSONResponse:
    """Spend grouped by ledger origin (S12) — the console's
    where-is-the-money-going view. Self-or-admin, like the other
    money routes: it's the tenant's own ledger. ?days=N bounds the
    window; default all time."""
    await require_customer_self_or_admin(customer_id, request, store)
    since = None
    if days is not None:
        if days < 1:
            raise HTTPException(
                status_code=400, detail="days must be >= 1"
            )
        from datetime import datetime, timedelta, timezone
        since = datetime.now(timezone.utc) - timedelta(days=days)
    origins = await store.cost_by_origin(customer_id, since=since)
    return JSONResponse(content={"origins": origins})


@router.get("/v1/customers/{customer_id}/budgets")
async def list_budgets(
    customer_id: ID64,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """The tenant's spend-budget rows (S4 substrate). Self-or-admin."""
    await require_customer_self_or_admin(customer_id, request, store)
    budgets = await store.list_spend_budgets(customer_id)
    # S12 (2026-07-09): the shadow critic's REAL cap is count-based
    # (customers.shadow_max_per_day override, else the global default) —
    # not an S4 spend row. Surface it read-only alongside the budgets,
    # with last-24h usage read off the ledger (origin='shadow_critic')
    # so the card never lies about enforcement it doesn't have.
    from datetime import datetime, timedelta, timezone

    from ..config import get_settings

    shadow_cap = await store.get_customer_shadow_cap_override(customer_id)
    _default_cap = getattr(
        get_settings(), "shadow_max_per_customer_per_day", None
    )
    _since = datetime.now(timezone.utc) - timedelta(hours=24)
    _by_origin = await store.cost_by_origin(customer_id, since=_since)
    _shadow_row = next(
        (r for r in _by_origin if r["origin"] == "shadow_critic"), None
    )
    return JSONResponse(content={
        "budgets": [b.model_dump(mode="json") for b in budgets],
        "shadow_critic": {
            "max_per_day": shadow_cap,
            "default_max_per_day": _default_cap,
            "effective_max_per_day": (
                shadow_cap if shadow_cap is not None else _default_cap
            ),
            "runs_last_24h": (
                _shadow_row["call_count"] if _shadow_row else 0
            ),
            "cost_micro_usd_last_24h": (
                _shadow_row["cost_micro_usd"] if _shadow_row else 0
            ),
        },
    })


@router.put("/v1/customers/{customer_id}/budgets/{function}")
async def upsert_budget(
    customer_id: ID64,
    function: Annotated[str, StringConstraints(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")],
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Create/update one budget row (S4). Body: {cap_micro_usd, period?}.
    cap_micro_usd=0 turns the function OFF for auto paths. v1 exposes
    'auto_research'; the substrate accepts any function name so later
    functions (shadow_critic, gap_fill) are row-inserts, not schema
    work. Self-or-admin: it's the tenant's money."""
    await require_customer_self_or_admin(customer_id, request, store)
    body = await request.json()
    require_bounds(body, {  # Q46 (PR-6)
        "cap_micro_usd": Bound(max_int=10**12),
        "period": Bound(max_len=16),
    })
    try:
        cap = int(body.get("cap_micro_usd", 0))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="cap_micro_usd must be an integer")
    if cap < 0:
        raise HTTPException(status_code=400, detail="cap_micro_usd must be >= 0")
    period = str(body.get("period") or "monthly")
    if period not in ("daily", "monthly"):
        raise HTTPException(status_code=400, detail="period must be daily|monthly")
    budget = await store.upsert_spend_budget(
        customer_id, function=function, cap_micro_usd=cap, period=period,
    )
    logger.info(
        "customer.budget_upserted",
        customer_id=customer_id, function=function, cap_micro_usd=cap,
    )
    return JSONResponse(content=budget.model_dump(mode="json"))


@router.post("/v1/gaps/{gap_id}/research")
async def promote_gap_to_research(
    gap_id: ID64,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """MANUAL promotion (S4): enqueue a cognition task for one gap — the
    click-path counterpart to the budget-gated auto sweep. Self-or-admin
    on the gap's OWNER; foreign/unknown gap = uniform 404. The task's
    spend runs under the tenant's normal inference doors (E4), not the
    auto_research budget — a human clicked; that IS the authorization."""
    gap = await store.get_knowledge_gap(gap_id)
    if gap is None:
        raise HTTPException(status_code=404, detail="Gap not found")
    await require_customer_self_or_admin(gap.customer_id, request, store)
    # RC-09 (2026-10-04): research is a paid cognition run; fact cap +
    # daily door for the gap's tenant.
    gap_owner = await store.get_customer_by_id(gap.customer_id)
    if gap_owner is not None:
        from ..ingress.auth import require_write_capacity

        await require_write_capacity(gap_owner, store, spend=True)

    task = await store.create_cognition_task(
        gap.customer_id,
        task_type="research",
        payload={
            "gap_id": gap.id,
            "topic": gap.missing,  # the worker's goal field
            "full_key": gap.full_key,
            "triggering_query": gap.triggering_query,
        },
        priority="background",
    )
    logger.info("gap.promoted", gap_id=gap.id, task_id=task.id)
    return JSONResponse(content={"task_id": task.id, "gap_id": gap.id})


# Console Refine PR 6 (docs/CONSOLE_UX_AUDIT.md 5.4 item 5): every gap row
# has a verb. "Type the answer" writes the answer as a fact and fills the
# gap with it; Dismiss closes a gap the operator does not want researched,
# and Undo reopens it. Self-or-admin on the gap's owner; foreign/unknown
# gap = uniform 404, like /research above.

async def _owned_gap(gap_id: str, request: Request, store: MetadataStore):
    gap = await store.get_knowledge_gap(gap_id)
    if gap is None:
        raise HTTPException(status_code=404, detail="Gap not found")
    await require_customer_self_or_admin(gap.customer_id, request, store)
    return gap


@router.post("/v1/gaps/{gap_id}/answer")
async def answer_gap(
    gap_id: ID64,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Fill a gap by hand: body {text}. The text becomes a fact under the
    gap's key (full_key, else domain|subject, else the missing text), born
    operator_stated, and the gap is marked filled by that crystal. Pays
    the fact-cap door, not the daily AI door: no model runs."""
    gap = await _owned_gap(gap_id, request, store)
    if gap.status != "open":
        raise HTTPException(status_code=409, detail="This gap is no longer open.")
    body = await request.json()
    require_bounds(body, {"text": Bound(max_len=50_000)})  # Q46 (PR-6)
    text = str(body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="text is required")
    owner = await store.get_customer_by_id(gap.customer_id)
    if owner is None:
        raise HTTPException(status_code=404, detail="Gap not found")
    from ..ingress.auth import require_write_capacity

    await require_write_capacity(owner, store)
    key = (gap.full_key or "").strip() or "|".join(
        x for x in (gap.domain, gap.subject) if x
    ) or gap.missing[:200]
    crystal, fact = await store.add_pair_for_customer(
        customer_id=gap.customer_id,
        prompt_text=key,
        answer_text=text,
        pair_type="question_answer",
        encoder=request.app.state.prompt_encoder,
        vector_store=request.app.state.vector_store,
        vector_index=getattr(request.app.state, "vector_index", None),
        crystal_type="customer:legacy",
        source_kind="operator_stated",
    )
    await store.mark_knowledge_gap_filled(
        gap.id, filled_by_crystal_id=crystal.id,
        resolved_at=datetime.now(timezone.utc),
    )
    logger.info("gap.answered", gap_id=gap.id, crystal_id=crystal.id)
    return JSONResponse(content={
        "gap_id": gap.id, "crystal_id": crystal.id, "fact_id": fact.id, "status": "filled",
    })


@router.post("/v1/gaps/{gap_id}/dismiss")
async def dismiss_gap(
    gap_id: ID64,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Close an open gap without filling it (status 'closed'). Reversible
    with /reopen; nothing is deleted."""
    gap = await _owned_gap(gap_id, request, store)
    if gap.status != "open":
        raise HTTPException(status_code=409, detail="Only an open gap can be dismissed.")
    await store.set_knowledge_gap_status(gap.id, gap.customer_id, "closed")
    logger.info("gap.dismissed", gap_id=gap.id)
    return JSONResponse(content={"gap_id": gap.id, "status": "closed"})


@router.post("/v1/gaps/{gap_id}/reopen")
async def reopen_gap(
    gap_id: ID64,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """The Undo for /dismiss: a closed gap goes back to open."""
    gap = await _owned_gap(gap_id, request, store)
    if gap.status != "closed":
        raise HTTPException(status_code=409, detail="Only a dismissed gap can be reopened.")
    await store.set_knowledge_gap_status(gap.id, gap.customer_id, "open")
    logger.info("gap.reopened", gap_id=gap.id)
    return JSONResponse(content={"gap_id": gap.id, "status": "open"})


@router.post("/v1/customers/{customer_id}/api_key")
async def rotate_api_key(
    customer_id: ID64,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Regenerate the customer's Key A (Phase C follow-on, 2026-07-07).

    Self-or-admin — critically including the OWNER's session JWT, because
    the primary caller is someone who LOST the key (never copied the
    one-time reveal) and holds only a console session. The old key stops
    working the instant this commits: one key per customer, no grace
    window. The new raw key is returned exactly once, mirroring signup.

    STEP-UP AUTH (2026-07-07, ratified): JWT principals must have
    authenticated RECENTLY (auth_time within ROTATE_MAX_AUTH_AGE_SECONDS)
    — a hijacked idle session cannot mint itself a fresh Key A; the
    console re-prompts for credentials before calling this. Key A callers
    are exempt: presenting the current key IS the fresh credential, and
    rotation kills it regardless.
    """
    await require_customer_self_or_admin(customer_id, request, store)

    from ..config import get_settings
    from ..ingress import auth as auth_mod
    from ..ingress.auth import _bearer_token_from_header, _looks_like_firebase_jwt

    auth_header = (
        request.headers.get("authorization")
        or request.headers.get("Authorization")
    )
    bearer = _bearer_token_from_header(auth_header)
    if bearer and _looks_like_firebase_jwt(bearer):
        settings = get_settings()
        claims = auth_mod._verify_firebase_jwt(
            bearer, (settings.firebase_project_id or "").strip()
        ) or {}
        auth_time = int(claims.get("auth_time") or 0)
        import time as _time
        if _time.time() - auth_time > ROTATE_MAX_AUTH_AGE_SECONDS:
            raise HTTPException(
                status_code=401,
                detail=(
                    "Please verify your sign-in again to regenerate the "
                    "API key."
                ),
            )

    rotated = await store.rotate_customer_api_key(customer_id)
    if rotated is None:
        raise HTTPException(status_code=404, detail="Customer not found")

    logger.info("customer.api_key_rotated", customer_id=customer_id)
    return JSONResponse(
        content={
            "customer_id": customer_id,
            "api_key": rotated.api_key,  # the one-time reveal
        }
    )


@router.patch("/v1/customers/{customer_id}/model")
async def update_model(
    customer_id: ID64,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Update the customer's upstream model (Phase C settings surface,
    2026-07-06). Self-or-admin. Hosted parity principle: every
    customization self-host has, hosted has — model choice is the first.

    managed customers pick from the platform's servable set (Era's key
    serves the call, so the platform must recognize the model); byok
    customers may set any non-empty model string — their key, their
    model.
    """
    from ..control.admission import canonical_managed_model, enforce_managed_model

    await require_customer_self_or_admin(customer_id, request, store)

    body = await request.json()
    require_bounds(body, {"model_id": Bound(max_len=128)})  # Q46 (PR-6)
    model_id = (body.get("model_id") or "").strip()
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")

    customer = await store.get_customer_by_id(customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    # v108 (Q13=A): the allow-list is per tier (free: Haiku + Sonnet).
    enforce_managed_model(customer, model_id)
    # Q88=A: a managed tenant's legacy id is stored as its 5.5 successor.
    if getattr(customer, "inference_mode", "byok") == "managed":
        model_id = canonical_managed_model(model_id)
    # Hackathon item 1: a Token Factory pick is probed before it is
    # stored. Answers need tool calling (every connected tool rides it);
    # a model that will not make a tool call is refused at the door.
    caps = await _probe_token_factory_pick(customer, model_id, store)
    if caps is not None and not caps.get("tools"):
        raise HTTPException(
            status_code=400,
            detail=build_error_envelope(
                f"{model_id} did not make a tool call when asked, so it "
                "cannot answer with your connected tools. Pick it for "
                "background curation instead, or choose another model.",
                error_type="invalid_request_error",
                param="model_id",
                code="model_no_tools",
            ),
        )

    updated = await store.set_customer_model(customer_id, model_id)
    if updated is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    if caps is not None:
        await store.set_customer_model_caps(customer_id, model_id, caps)

    logger.info("customer.model_updated", customer_id=customer_id,
                model_id=model_id)
    return JSONResponse(content={"updated": True, "customer_id": customer_id,
                                 "model_id": model_id, "caps": caps})


@router.patch("/v1/customers/{customer_id}/background_model")
async def update_background_model(
    customer_id: ID64,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Q87 (2026-10-17): the tenant's BACKGROUND CURATION model (research,
    composition, extraction, curation judgments). Same door as the
    answers model: self-or-admin, the plan's allow-list for managed
    tenants, legacy ids stored as their successor. An empty model_id
    clears the pick (the platform tier map applies again)."""
    from ..control.admission import canonical_managed_model, enforce_managed_model

    await require_customer_self_or_admin(customer_id, request, store)

    body = await request.json()
    require_bounds(body, {"model_id": Bound(max_len=128)})  # Q46 (PR-6)
    model_id = (body.get("model_id") or "").strip() or None

    customer = await store.get_customer_by_id(customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    caps = None
    if model_id:
        enforce_managed_model(customer, model_id)
        if getattr(customer, "inference_mode", "byok") == "managed":
            model_id = canonical_managed_model(model_id)
        # Hackathon item 1: probed, never refused. A background model
        # without json_schema still works: the seam sends json_object
        # with the schema in the prompt (llm/client._complete_openai).
        caps = await _probe_token_factory_pick(customer, model_id, store)

    updated = await store.set_customer_background_model(customer_id, model_id)
    if updated is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    if caps is not None and model_id:
        await store.set_customer_model_caps(customer_id, model_id, caps)

    logger.info("customer.background_model_updated", customer_id=customer_id,
                model_id=model_id)
    return JSONResponse(content={"updated": True, "customer_id": customer_id,
                                 "background_model_id": model_id, "caps": caps})


async def _probe_token_factory_pick(customer, model_id: str, store=None) -> Optional[dict]:
    """Run the two-call probe for a tenant's Token Factory pick and return
    its record, or None when the pick is not a Token Factory model (an
    Anthropic id, a byok tenant on another provider) or the lane is off.
    A managed tenant is probed on the platform's client; a byok tenant
    whose key is a Token Factory key (provider nebius, item 2) is probed
    on their own. A probe that cannot run at all (transport) records both
    flags False with the error, so the console shows why; it never raises."""
    import asyncio

    from ..llm.catalog import probe_model
    from ..llm.client import get_llm_client_for_customer, get_nemotron_client
    from ..llm.lanes import family_of, nemotron_enabled

    mode = getattr(customer, "inference_mode", "byok")
    cfg = getattr(customer, "model_routing_config", None)
    if mode == "byok":
        if (getattr(cfg, "provider", "") or "").lower() != "nebius" or store is None:
            return None
        if family_of(model_id) != "nemotron":
            return None
        try:
            client = await get_llm_client_for_customer(customer, store)
        except Exception:  # noqa: BLE001 — no usable key: the pick stands unprobed
            return None
        caps = await asyncio.to_thread(probe_model, client, model_id)
        logger.info("customer.model_probed", customer_id=getattr(customer, "id", None),
                    model_id=model_id, json_schema=caps.json_schema, tools=caps.tools, on="tenant_key")
        return caps.as_dict()
    if family_of(model_id) != "nemotron" or not nemotron_enabled():
        return None
    caps = await asyncio.to_thread(probe_model, get_nemotron_client(), model_id)
    logger.info("customer.model_probed", customer_id=getattr(customer, "id", None),
                model_id=model_id, json_schema=caps.json_schema, tools=caps.tools)
    return caps.as_dict()


@router.get("/v1/customers/{customer_id}/models/catalog")
async def model_catalog(
    customer_id: ID64,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Hackathon item 1 (2026-10-22): every TEXT model Token Factory lists,
    for the Settings picker. Context length and a cost band only (tenants
    never see dollars, Q58=A); the tenant's own probe records ride along
    so the console can show JSON / Tools chips. Self-or-admin."""
    from ..llm.catalog import text_models
    from ..llm.lanes import nemotron_enabled

    await require_customer_self_or_admin(customer_id, request, store)
    customer = await store.get_customer_by_id(customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    cfg = getattr(customer, "model_routing_config", None)
    caps = dict(getattr(cfg, "model_caps", None) or {})
    enabled = nemotron_enabled()
    models = [
        {
            "id": m.id,
            "name": m.name,
            "context_length": m.context_length,
            "band": m.band,
            "fast": m.is_fast,
            "nvidia": m.id.lower().startswith("nvidia/"),
            "caps": caps.get(m.id.lower()),
        }
        for m in (text_models() if enabled else ())
    ]
    models.sort(key=lambda m: (not m["nvidia"], m["id"].lower()))
    return JSONResponse(content={"enabled": enabled, "models": models})


@router.patch("/v1/customers/{customer_id}/inference_mode")
async def update_inference_mode(
    customer_id: ID64,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Flip a tenant between managed and byok inference (E4 / Phase C
    Settings surface, 2026-07-06). Self-or-admin, like the sibling
    upstream_key route.

    Rule: flipping to byok REQUIRES a stored Key B — otherwise the very
    next proxy call would fail on an empty upstream key. The Settings
    page enforces the order (paste key, then flip); the API enforces it
    for everyone else.
    """
    await require_customer_self_or_admin(customer_id, request, store)

    body = await request.json()
    require_bounds(body, {"inference_mode": Bound(max_len=32)})  # Q46 (PR-6)
    mode = (body.get("inference_mode") or "").strip().lower()
    if mode not in ("managed", "byok"):
        raise HTTPException(
            status_code=400,
            detail="inference_mode must be 'managed' or 'byok'",
        )

    customer = await store.get_customer_by_id(customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    if mode == "byok" and not (customer.model_routing_config.api_key_ref or ""):
        raise HTTPException(
            status_code=400,
            detail=(
                "Switching to your own key requires one on file — add your "
                "provider API key first."
            ),
        )

    updated = await store.set_customer_inference_mode(customer_id, mode)
    if updated is None:
        raise HTTPException(status_code=404, detail="Customer not found")

    logger.info(
        "customer.inference_mode_updated",
        customer_id=customer_id,
        inference_mode=mode,
    )
    return JSONResponse(
        content={"updated": True, "customer_id": customer_id,
                 "inference_mode": mode}
    )


# ---------------------------------------------------------------------------
# Automation: the tenant's system rules (Q99 C + the first approval rule,
# 2026-10-23). Anthony: "control how automated you want the system to be
# and create approval rules." The system_rules package (strict, typed)
# gets its routes here, tenant-pathed and self-or-admin; the console's
# Settings "Automation" section is the only writer. Rules are settings,
# not bank writes (test_http_write_wall allow-list).
# ---------------------------------------------------------------------------

def _rule_json(rule) -> dict:
    from ..system_rules.curation import rule_text

    return {
        "id": rule.id,
        "rule_type": rule.rule_type,
        "name": rule.name,
        "selector": dict(rule.selector or {}),
        "conditions": dict(rule.conditions or {}),
        "action": dict(rule.action or {}),
        "enabled": bool(rule.enabled),
        "priority": int(rule.priority),
        "fire_count": int(rule.fire_count or 0),
        "last_fired_at": rule.last_fired_at.isoformat() if rule.last_fired_at else None,
        "created_at": rule.created_at.isoformat() if rule.created_at else None,
        "text": rule_text(rule),
    }


def _rules_json(rules) -> dict:
    from ..system_rules.curation import DEFAULT_LINE

    return {"rules": [_rule_json(r) for r in rules], "count": len(rules), "default": DEFAULT_LINE}


def _rule_invalid(e: Exception) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail=build_error_envelope(
            str(e), error_type="invalid_request_error", param="rule", code="rule_invalid",
        ),
    )


@router.get("/v1/customers/{customer_id}/rules")
async def list_customer_rules(
    customer_id: ID64,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Every rule this tenant wrote, in order, with its fire audit and its
    sentence. Q101: rules written under the older promotion/approval
    shapes are folded into curation rules first, so the list is one kind."""
    from ..system_rules.store import convert_legacy_rules, list_rules

    await require_customer_self_or_admin(customer_id, request, store)
    await convert_legacy_rules(store, customer_id)
    rules = await list_rules(store, customer_id)
    return JSONResponse(content=_rules_json(rules))


@router.post("/v1/customers/{customer_id}/rules", status_code=status.HTTP_201_CREATED)
async def create_customer_rule(
    customer_id: ID64,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Write one rule. Body: {rule_type, name, selector, conditions, action,
    enabled?, priority?}. STRICT validation (an unknown key refuses with
    code rule_invalid) before any write."""
    from ..system_rules import RuleValidationError
    from ..system_rules.store import create_rule

    await require_customer_self_or_admin(customer_id, request, store)
    body = await request.json()
    if not isinstance(body, dict):
        raise _rule_invalid(ValueError("body must be an object"))
    rule_type = str(body.get("rule_type") or "").strip()
    name = str(body.get("name") or "").strip()[:120]
    if not rule_type or not name:
        raise _rule_invalid(ValueError("rule_type and name are required"))
    for key in ("selector", "conditions", "action"):
        if not isinstance(body.get(key, {}), dict):
            raise _rule_invalid(ValueError(f"{key} must be an object"))
    enabled = body.get("enabled", True)
    priority = body.get("priority")
    if not isinstance(enabled, bool) or isinstance(priority, bool) or not isinstance(priority, (int, type(None))):
        raise _rule_invalid(ValueError("enabled must be a boolean and priority an integer"))
    try:
        rule = await create_rule(
            store, customer_id, rule_type, name,
            selector=dict(body.get("selector") or {}),
            conditions=dict(body.get("conditions") or {}),
            action=dict(body.get("action") or {}),
            enabled=enabled,
            priority=(max(0, min(priority, 10_000)) if priority is not None else None),
        )
    except RuleValidationError as e:
        raise _rule_invalid(e)
    logger.info("customer.rule_created", customer_id=customer_id, rule_id=rule.id, rule_type=rule_type)
    return JSONResponse(status_code=status.HTTP_201_CREATED, content={"rule": _rule_json(rule)})


@router.patch("/v1/customers/{customer_id}/rules/{rule_id}")
async def update_customer_rule(
    customer_id: ID64,
    rule_id: ID64,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Turn a rule on or off, rename it, or change what it applies to, its
    conditions or its decision. The changed rule is re-validated strictly
    before the write."""
    from ..system_rules import RuleValidationError
    from ..system_rules.store import update_rule

    await require_customer_self_or_admin(customer_id, request, store)
    body = await request.json()
    if not isinstance(body, dict):
        raise _rule_invalid(ValueError("body must be an object"))
    enabled = body.get("enabled")
    name = body.get("name")
    if enabled is not None and not isinstance(enabled, bool):
        raise _rule_invalid(ValueError("enabled must be a boolean"))
    if name is not None and not isinstance(name, str):
        raise _rule_invalid(ValueError("name must be a string"))
    for key in ("selector", "conditions", "action"):
        if body.get(key) is not None and not isinstance(body.get(key), dict):
            raise _rule_invalid(ValueError(f"{key} must be an object"))
    try:
        rule = await update_rule(
            store, customer_id, rule_id,
            enabled=enabled, name=(name.strip()[:120] if isinstance(name, str) else None),
            selector=body.get("selector"), conditions=body.get("conditions"), action=body.get("action"),
        )
    except RuleValidationError as e:
        raise _rule_invalid(e)
    if rule is None:
        raise HTTPException(status_code=404, detail="Rule not found")
    return JSONResponse(content={"rule": _rule_json(rule)})


@router.post("/v1/customers/{customer_id}/rules/{rule_id}/move")
async def move_customer_rule(
    customer_id: ID64,
    rule_id: ID64,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Q101: move a rule one line up or down. Body: {direction: "up" |
    "down"}. Returns the whole reordered list, since order is what the
    list means (first match decides)."""
    from ..system_rules.store import move_rule

    await require_customer_self_or_admin(customer_id, request, store)
    body = await request.json()
    direction = body.get("direction") if isinstance(body, dict) else None
    if direction not in ("up", "down"):
        raise _rule_invalid(ValueError("direction must be 'up' or 'down'"))
    rules = await move_rule(store, customer_id, rule_id, direction)
    if rules is None:
        raise HTTPException(status_code=404, detail="Rule not found")
    return JSONResponse(content=_rules_json(rules))


@router.delete("/v1/customers/{customer_id}/rules/{rule_id}")
async def delete_customer_rule(
    customer_id: ID64,
    rule_id: ID64,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    from ..system_rules.store import delete_rule

    await require_customer_self_or_admin(customer_id, request, store)
    if not await delete_rule(store, customer_id, rule_id):
        raise HTTPException(status_code=404, detail="Rule not found")
    logger.info("customer.rule_deleted", customer_id=customer_id, rule_id=rule_id)
    return JSONResponse(content={"deleted": True, "rule_id": rule_id})
