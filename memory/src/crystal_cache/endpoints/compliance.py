"""Compliance endpoints — HIPAA BAA status.

GET  /v1/compliance/baa     — read current BAA tracking
PUT  /v1/compliance/baa     — upsert BAA tracking

Verbatim port from v1's app.py, refactored to use Phase 5's
`get_baa_record` / `upsert_baa_record` store methods.
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from ..infrastructure import MetadataStore
from ..infrastructure.metadata_store import get_metadata_store
from ..ingress.auth import require_customer
from ..models import Customer

logger = structlog.get_logger(__name__)

router = APIRouter()


class BaaUpdateRequest(BaseModel):
    """Partial-update request body. All fields optional; only provided
    fields are updated."""
    model_config = ConfigDict(extra="forbid")  # Q46 (PR-6)

    baa_signed: Optional[bool] = None
    baa_signed_date: Optional[str] = Field(default=None, max_length=64)  # ISO 8601
    baa_document_ref: Optional[str] = Field(default=None, max_length=512)
    phi_data_sources: Optional[list[Annotated[str, Field(max_length=256)]]] = Field(default=None, max_length=50)
    # One address, bounded; email-validator is not a dependency, so the
    # shape check is the mailbox@host form (Q46).
    hipaa_contact_email: Optional[str] = Field(
        default=None, max_length=254, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
    )
    notes: Optional[str] = Field(default=None, max_length=4000)


def _baa_to_dict(baa) -> dict[str, Any]:
    """Serialize a BaaTracking model to the v1 response shape."""
    return {
        "id": baa.id,
        "customer_id": baa.customer_id,
        "baa_signed": baa.baa_signed,
        "baa_signed_date": baa.baa_signed_date.isoformat() if baa.baa_signed_date else None,
        "baa_document_ref": baa.baa_document_ref,
        "phi_data_sources": baa.phi_data_sources or [],
        "hipaa_contact_email": baa.hipaa_contact_email,
        "notes": baa.notes,
        "created_at": baa.created_at.isoformat(),
        "updated_at": baa.updated_at.isoformat(),
    }


@router.get("/v1/compliance/baa")
async def get_baa_status(
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Read this customer's BAA tracking record.

    Returns 404 if no record exists yet. Customers without a record
    are treated as not-yet-onboarded for HIPAA purposes — calling
    PUT creates the record.
    """
    baa = await store.get_baa_record(customer.id)
    if baa is None:
        raise HTTPException(
            status_code=404,
            detail="No BAA record on file for this customer",
        )
    return JSONResponse(content=_baa_to_dict(baa))


@router.put("/v1/compliance/baa")
async def update_baa_status(
    body: BaaUpdateRequest,
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Upsert this customer's BAA tracking record.

    Creates the record if missing; otherwise patches the provided
    fields. Returns the post-state.
    """
    # Parse ISO date if provided
    signed_date_dt: Optional[datetime] = None
    if body.baa_signed_date:
        try:
            signed_date_dt = datetime.fromisoformat(
                body.baa_signed_date.replace("Z", "+00:00")
            )
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="baa_signed_date must be ISO 8601",
            )

    baa = await store.upsert_baa_record(
        customer.id,
        baa_signed=body.baa_signed,
        baa_signed_date=signed_date_dt,
        baa_document_ref=body.baa_document_ref,
        phi_data_sources=body.phi_data_sources,
        hipaa_contact_email=body.hipaa_contact_email,
        notes=body.notes,
    )

    logger.info(
        "baa.upserted",
        customer_id=customer.id,
        baa_signed=baa.baa_signed,
    )
    return JSONResponse(content=_baa_to_dict(baa))
