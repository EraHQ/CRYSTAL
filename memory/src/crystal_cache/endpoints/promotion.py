"""Promotion endpoints — Foundation F3 (detect → curate → merge up).

The HTTP surface over the promotion engine (maintenance/promotion_service.py).
Two admin-gated routes on the operator→team rung:

  GET  /v1/promotion/candidates  — run detect live; return near-duplicate
       clusters that span >= 2 operators (what an admin could promote).
  POST /v1/promotion/merge       — promote a chosen cluster into one team
       crystal (chgrp/chmod-up the survivor + supersede the rest + record
       contributor provenance).

Both gated by require_role("admin"): curation is the admin's (= root's) call,
and D3 makes operator-private crystal content admin-readable so the survey is
legitimate. The team is the principal's customer; detect and merge scope to
it. The merge route invalidates the app's vector + fact-vector caches so the
superseded crystals stop surfacing and the survivor's new team-tier grouping
takes effect immediately.

The Inspector's promotion-candidate queue (a Growth surface) reads these;
F3 lands the API, not the React view. Request/response models are
endpoint-local (specific to this surface, not shared like the chat schema).
"""
from __future__ import annotations

from typing import Annotated, Any, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ..infrastructure import MetadataStore
from ..infrastructure.metadata_store import get_metadata_store
from ..ingress.auth import require_role
from ..maintenance.promotion_service import PromotionError, PromotionService
from ..models import Customer, Operator

logger = structlog.get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Endpoint-local request/response models
# ---------------------------------------------------------------------------

class PromotionCandidateOut(BaseModel):
    """One detected near-duplicate cluster proposed for promotion."""
    crystal_ids: list[str]
    operator_ids: list[str]
    mean_similarity: float
    size: int
    previews: dict[str, Optional[str]] = Field(default_factory=dict)


class PromotionCandidatesResponse(BaseModel):
    candidates: list[PromotionCandidateOut]


class MergeRequest(BaseModel):
    """The chosen cluster's crystal ids. The engine validates the rest
    (belong to the team, operator-owned, >= 2 distinct operators)."""
    source_crystal_ids: list[str]


class MergeResponse(BaseModel):
    merged_crystal_id: str
    superseded_crystal_ids: list[str]
    contributions: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get(
    "/v1/promotion/candidates",
    response_model=PromotionCandidatesResponse,
)
async def list_promotion_candidates(
    request: Request,
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(require_role("admin"))
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    threshold: float = 0.95,
) -> PromotionCandidatesResponse:
    """Run detect live for the admin's team and return promotion candidates.

    Admin-gated (require_role): only a team admin (= root) or the team key
    surveys promotable duplicates — D3 makes operator-private content
    admin-readable. Read-only; no mutation.
    """
    customer, _actor = principal
    svc = PromotionService(store)
    candidates = await svc.detect_candidates(customer.id, threshold=threshold)
    return PromotionCandidatesResponse(
        candidates=[
            PromotionCandidateOut(
                crystal_ids=c.crystal_ids,
                operator_ids=c.operator_ids,
                mean_similarity=c.mean_similarity,
                size=c.size,
                previews=c.previews,
            )
            for c in candidates
        ]
    )


@router.post("/v1/promotion/merge", response_model=MergeResponse)
async def merge_promotion_candidate(
    body: MergeRequest,
    request: Request,
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(require_role("admin"))
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> MergeResponse:
    """Promote a chosen cluster into one team crystal (curate → merge).

    Admin-gated. The team is the principal's customer. After the merge the
    app's vector + fact-vector caches are invalidated so the superseded
    crystals stop surfacing and the survivor's team-tier grouping takes
    effect. An invalid request (unknown / cross-team / non-operator-owned
    sources, or fewer than two distinct operators) returns 400.
    """
    customer, _actor = principal
    svc = PromotionService(store)
    try:
        result = await svc.merge(
            customer.id,
            body.source_crystal_ids,
            vector_store=getattr(request.app.state, "vector_store", None),
            # Active vector index (Qdrant-aware) for invalidation; fall back to
            # the in-memory fact store. merge uses it only to invalidate.
            fact_vector_store=(
                getattr(request.app.state, "vector_index", None)
                or getattr(request.app.state, "fact_vector_store", None)
            ),
        )
    except PromotionError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(
        "promotion.merge_api",
        team_id=customer.id,
        merged_crystal_id=result.merged_crystal_id,
        superseded=len(result.superseded_crystal_ids),
    )
    return MergeResponse(
        merged_crystal_id=result.merged_crystal_id,
        superseded_crystal_ids=result.superseded_crystal_ids,
        contributions=result.contributions,
    )
