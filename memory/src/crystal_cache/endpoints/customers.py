"""Customer CRUD — /v1/customers/*

Verbatim port from v1's app.py.

Phase 6.5 P4.1 / CU-8: `update_upstream_key` now calls
`store.update_customer_upstream_key`, closing the only remaining
inline-SQL violation in v2 endpoints.
"""
from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse

from ..infrastructure import MetadataStore
from ..infrastructure.metadata_store import get_metadata_store
from ..ingress.auth import require_customer_self_or_admin
from ..ingress.schema import (
    CreateCustomerRequest,
    CreateCustomerResponse,
    GetCustomerResponse,
)

logger = structlog.get_logger(__name__)

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

    customer = await store.create_customer(
        provider=body.provider,
        model_id=body.model_id,
        api_key_ref=body.api_key_ref,
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
    customer_id: str,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> GetCustomerResponse:
    # B1 (2026-07-03): self-or-admin only. Was unauthenticated.
    customer = await require_customer_self_or_admin(customer_id, request, store)
    return GetCustomerResponse(
        id=customer.id,
        provider=customer.model_routing_config.provider,
        model_id=customer.model_routing_config.model_id,
        base_url=customer.model_routing_config.base_url,
        injection_preference=customer.injection_preference,
        shadow_sample_rate=customer.shadow_sample_rate,
        created_at=customer.created_at.isoformat(),
    )


@router.patch("/v1/customers/{customer_id}/upstream_key")
async def update_upstream_key(
    customer_id: str,
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
    new_key = body.get("api_key_ref", "")
    if not new_key:
        raise HTTPException(status_code=400, detail="api_key_ref is required")

    updated = await store.update_customer_upstream_key(
        customer_id=customer_id,
        new_api_key_ref=new_key,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Customer not found")

    logger.info(
        "customer.upstream_key_updated",
        customer_id=customer_id,
    )
    return JSONResponse(content={"updated": True, "customer_id": customer_id})


# Step-up auth window for key rotation: a JWT principal's last sign-in
# (auth_time) must be within this many seconds. 5 minutes = long enough
# for the reauth popup round-trip, short enough that an idle hijacked
# session cannot rotate.
ROTATE_MAX_AUTH_AGE_SECONDS = 300


@router.get("/v1/customers/{customer_id}/budgets")
async def list_budgets(
    customer_id: str,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """The tenant's spend-budget rows (S4 substrate). Self-or-admin."""
    await require_customer_self_or_admin(customer_id, request, store)
    budgets = await store.list_spend_budgets(customer_id)
    return JSONResponse(
        content={"budgets": [b.model_dump(mode="json") for b in budgets]}
    )


@router.put("/v1/customers/{customer_id}/budgets/{function}")
async def upsert_budget(
    customer_id: str,
    function: str,
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
    gap_id: str,
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


@router.post("/v1/customers/{customer_id}/api_key")
async def rotate_api_key(
    customer_id: str,
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
    customer_id: str,
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
    from .me import MANAGED_ALLOWED_MODELS

    await require_customer_self_or_admin(customer_id, request, store)

    body = await request.json()
    model_id = (body.get("model_id") or "").strip()
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")

    customer = await store.get_customer_by_id(customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    if (customer.inference_mode == "managed"
            and model_id not in MANAGED_ALLOWED_MODELS):
        raise HTTPException(
            status_code=400,
            detail=(
                "Managed inference supports: "
                + ", ".join(sorted(MANAGED_ALLOWED_MODELS))
                + ". Switch to your own key for other models."
            ),
        )

    updated = await store.set_customer_model(customer_id, model_id)
    if updated is None:
        raise HTTPException(status_code=404, detail="Customer not found")

    logger.info("customer.model_updated", customer_id=customer_id,
                model_id=model_id)
    return JSONResponse(content={"updated": True, "customer_id": customer_id,
                                 "model_id": model_id})


@router.patch("/v1/customers/{customer_id}/inference_mode")
async def update_inference_mode(
    customer_id: str,
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
