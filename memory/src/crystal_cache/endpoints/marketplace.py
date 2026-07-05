"""Marketplace endpoints — Growth G4 (shard ledger + vetting).

The Inspector's read API over the shard ledger (shard_events) plus the expert
vetting surface (expert_authorizations), both via
metadata_store_shard_ext.py. Shard balance is **financial** — reads are
role-gated to operator+; authorizing/revoking an expert is an **admin**
action (the curate authority, like F3 merge). The team root key is always
admitted.

Routes (all team-scoped; an operator id outside the team 404s):

  GET    /v1/marketplace/balance?operator_id=  — an expert's shard balance
         (integer shards).
  GET    /v1/marketplace/ledger?operator_id=   — an expert's ledger entries,
         newest first.
  GET    /v1/marketplace/experts               — the team's expert
         authorizations.
  POST   /v1/marketplace/experts               — authorize an operator to
         author general crystals in a domain (admin).
  POST   /v1/marketplace/experts/revoke        — revoke an authorization
         (admin).

This is G4's read + vetting API. Crediting is automatic: a grounded citation
of a general crystal mints a shard via record_citation_credit, wired into the
proxy citation block behind CC_ENABLE_MARKETPLACE_METERING. Convertibility
(shards offsetting subscription) is OFF at launch — the spend substrate exists
but is not wired to billing. Reputation / dispute / clawback economy + the
bounded reward pool (D7) are deferred.
"""
from __future__ import annotations

from typing import Annotated, Any, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..infrastructure import MetadataStore
from ..infrastructure.metadata_store import get_metadata_store
from ..ingress.auth import require_role
from ..models import Customer, Operator

logger = structlog.get_logger(__name__)

router = APIRouter()


async def _require_team_operator(
    store: MetadataStore, operator_id: str, team_id: str
) -> None:
    """Assert the operator belongs to the team, else 404 (no cross-team
    balance/ledger leak). Uses the F1 operator lookup."""
    operator = await store.get_operator_by_id(operator_id)
    if operator is None or operator.team_id != team_id:
        raise HTTPException(status_code=404, detail="operator not found")


class AuthorizeExpertRequest(BaseModel):
    """Authorize an operator to author general crystals in a domain."""
    operator_id: str
    domain: str  # general:<domain>


class RevokeExpertRequest(BaseModel):
    operator_id: str
    domain: str


@router.get("/v1/marketplace/balance")
async def shard_balance(
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(require_role("operator"))
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    operator_id: str,
) -> dict[str, Any]:
    """An expert's shard balance (integer shards = credits − debits)."""
    customer, _actor = principal
    await _require_team_operator(store, operator_id, customer.id)
    balance = await store.shard_balance(operator_id)
    return {"operator_id": operator_id, "shard_balance": balance}


@router.get("/v1/marketplace/ledger")
async def shard_ledger(
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(require_role("operator"))
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    operator_id: str,
    limit: int = 100,
) -> dict[str, Any]:
    """An expert's ledger entries, newest first."""
    customer, _actor = principal
    await _require_team_operator(store, operator_id, customer.id)
    events = await store.list_shard_events(operator_id, limit=limit)
    return {"operator_id": operator_id, "events": events}


@router.get("/v1/marketplace/experts")
async def list_experts(
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(require_role("operator"))
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    active_only: bool = True,
) -> dict[str, Any]:
    """The team's expert authorizations."""
    customer, _actor = principal
    rows = await store.list_expert_authorizations(
        customer.id, active_only=active_only
    )
    return {"experts": rows}


@router.post("/v1/marketplace/experts")
async def authorize_expert(
    body: AuthorizeExpertRequest,
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(require_role("admin"))
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> dict[str, Any]:
    """Authorize an operator to author general crystals in a domain (admin)."""
    customer, _actor = principal
    await _require_team_operator(store, body.operator_id, customer.id)
    authorization = await store.authorize_expert(
        body.operator_id, customer.id, body.domain
    )
    return {"authorization": authorization}


@router.post("/v1/marketplace/experts/revoke")
async def revoke_expert(
    body: RevokeExpertRequest,
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(require_role("admin"))
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> dict[str, Any]:
    """Revoke an operator's authorization in a domain (admin)."""
    customer, _actor = principal
    await _require_team_operator(store, body.operator_id, customer.id)
    revoked = await store.revoke_expert(body.operator_id, body.domain)
    if not revoked:
        raise HTTPException(status_code=404, detail="authorization not found")
    return {"revoked": True, "operator_id": body.operator_id, "domain": body.domain}
