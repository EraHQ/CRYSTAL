"""Diagnostic API endpoints — /api/crystals, /api/edits.

Inspector-tier endpoints that surface per-crystal diagnostic
information (latest CrystalDiagnostic row) and the proposed-edit
queue. Ported verbatim from v1 per Phase 6.5 P1.3.

Endpoints:
  GET    /api/crystals                          list customer's crystals
                                                 with diagnostic shape
  GET    /api/crystals/{crystal_id}/diagnostic  latest diagnostic row
  GET    /api/edits                             proposed crystal edits

The history endpoint (`/api/crystals/{id}/history`) lives in
`stubs.py` since it's a 501 stub in v1.
"""
from __future__ import annotations

from typing import Annotated, Any, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from ..infrastructure import MetadataStore
from ..infrastructure.metadata_store import get_metadata_store
from ..ingress.auth import require_customer
from ..models import Customer

logger = structlog.get_logger(__name__)

router = APIRouter()


def _crystal_to_inspector_shape(c) -> dict[str, Any]:
    """v1 inspector list shape — keyword_fingerprint truncated."""
    fp = c.keyword_fingerprint or []
    return {
        "id": c.id,
        "fact_count": c.fact_count,
        "quality_tier": c.quality_tier,
        "source_kind": c.source_kind,
        "keyword_fingerprint": fp[:10],  # v1 truncates at 10
        "summary_text": c.summary_text,
        "created_at": c.created_at.isoformat(),
        "last_activity": c.last_activity.isoformat() if c.last_activity else None,
    }


@router.get("/api/crystals")
async def list_crystals_for_inspector(
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """List this customer's crystals in inspector shape.

    Distinct from `/v1/crystals` (SDK shape, paginated): this returns
    every crystal for inspector display with the v1 truncated
    keyword_fingerprint.
    """
    crystals = await store.list_crystals_for_customer(customer.id)
    return JSONResponse(content={
        "crystals": [_crystal_to_inspector_shape(c) for c in crystals],
        "count": len(crystals),
    })


@router.get("/api/crystals/{crystal_id}/diagnostic")
async def get_crystal_diagnostic(
    crystal_id: str,
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Latest CrystalDiagnostic row for a crystal.

    Tenancy check: confirms the crystal belongs to this customer
    before returning. Returns 404 if the crystal isn't ours.
    Returns a "no diagnostic" payload if the crystal exists but
    has never been evaluated.
    """
    crystal = await store.get_crystal(crystal_id)
    if crystal is None or crystal.customer_id != customer.id:
        raise HTTPException(status_code=404, detail="Crystal not found")

    diag = await store.get_latest_diagnostic(crystal_id)
    if diag is None:
        return JSONResponse(content={
            "crystal_id": crystal_id,
            "diagnostic": None,
            "message": "No diagnostic available for this crystal yet.",
        })

    return JSONResponse(content={
        "crystal_id": crystal_id,
        "diagnostic": {
            "id": diag.id,
            "observed_at": diag.observed_at.isoformat(),
            "failure_mode_distribution": diag.failure_mode_distribution,
            "top_help_query_exemplars": diag.top_help_query_exemplars,
            "top_hurt_query_exemplars": diag.top_hurt_query_exemplars,
            "compression_ratio_p25": diag.compression_ratio_p25,
            "compression_ratio_p50": diag.compression_ratio_p50,
            "compression_ratio_p75": diag.compression_ratio_p75,
            "query_distribution_drift": diag.query_distribution_drift,
            "proposed_edit_ids": diag.proposed_edit_ids,
        },
    })


@router.get("/api/edits")
async def list_edits_for_inspector(
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    status_filter: Optional[str] = None,
) -> JSONResponse:
    """List proposed crystal edits.

    Query param `status_filter` filters by edit status (e.g.
    'proposed', 'approved', 'executed', 'rejected'). Default
    returns all. Limit hardcoded at 200 per v1.
    """
    edits = await store.list_edits(
        customer_id=customer.id,
        status=status_filter,
        limit=200,
    )
    return JSONResponse(content={
        "edits": [
            {
                "id": e.id,
                "crystal_id": e.crystal_id,
                "edit_type": e.edit_type,
                "proposed_by": e.proposed_by,
                "rationale": e.rationale,
                "affected_facts": e.affected_facts,
                "expected_impact": e.expected_impact,
                "status": e.status,
                "executed_at": e.executed_at.isoformat() if e.executed_at else None,
                "created_at": e.created_at.isoformat(),
                "actual_impact": e.actual_impact,
            }
            for e in edits
        ],
        "count": len(edits),
    })
