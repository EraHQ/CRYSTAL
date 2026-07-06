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
