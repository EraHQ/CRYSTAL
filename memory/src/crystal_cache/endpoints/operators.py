"""Operator CRUD — /v1/operators/*  (Foundation F1).

Operators are authenticated humans under a team (the customer row IS the
team). Two auth surfaces:

  - Team management (create / list / get / role / status) is authed by the
    team's customer key (require_customer). The team's Key A is the team
    root credential — whoever holds it provisions and manages the team's
    operators. This is the bootstrap path: it needs no pre-existing
    operator, and it matches the POSIX posture (root provisions users).
  - Self-introspection (/v1/operators/me) is authed by the operator's own
    key (require_operator), so a client holding an operator key can confirm
    who it is and what role it has.

Operator-on-operator management (create / role / status) is admin-gated
via require_role("admin"): the team root key is always admitted (it
outranks every operator role, preserving the bootstrap path), an admin
operator key is admitted, and viewer/operator keys get a 403. The read
paths (list / get / me) keep their lighter team-key / operator-key auth.

R9-clean: store methods only, no inline SQL.
"""
from __future__ import annotations

from typing import Annotated, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, status

from ..infrastructure import MetadataStore
from ..infrastructure.metadata_store import get_metadata_store
from ..ingress.auth import require_customer, require_operator, require_role
from ..ingress.schema import (
    CreateOperatorRequest,
    CreateOperatorResponse,
    OperatorListResponse,
    OperatorResponse,
    SetOperatorRoleRequest,
    SetOperatorStatusRequest,
)
from ..models import Customer, Operator

logger = structlog.get_logger(__name__)

router = APIRouter()


def _operator_response(op: Operator) -> OperatorResponse:
    return OperatorResponse(
        id=op.id,
        team_id=op.team_id,
        display_name=op.display_name,
        role=op.role,
        status=op.status,
        created_at=op.created_at.isoformat(),
    )


async def _team_scoped_operator(
    operator_id: str, customer: Customer, store: MetadataStore
) -> Operator:
    """Fetch an operator and assert it belongs to the authenticated team.

    A cross-team (or missing) id returns 404 rather than leaking that the
    operator exists on some other team.
    """
    op = await store.get_operator_by_id(operator_id)
    if op is None or op.team_id != customer.id:
        raise HTTPException(status_code=404, detail="Operator not found")
    return op


@router.post(
    "/v1/operators",
    response_model=CreateOperatorResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_operator(
    body: CreateOperatorRequest,
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(require_role("admin"))
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> CreateOperatorResponse:
    """Create an operator under the authenticated team.

    Admin-gated (require_role): the team root key OR an admin operator key
    may provision operators; viewer/operator keys get a 403. The operator's
    raw API key is returned ONCE here; only its hash is stored. The caller
    must save it.
    """
    customer, _actor = principal
    operator, raw_key = await store.create_operator(
        team_id=customer.id,
        display_name=body.display_name,
        role=body.role,
    )
    logger.info(
        "operator.created",
        operator_id=operator.id,
        team_id=customer.id,
        role=operator.role,
    )
    return CreateOperatorResponse(
        id=operator.id,
        team_id=operator.team_id,
        display_name=operator.display_name,
        role=operator.role,
        status=operator.status,
        api_key=raw_key,
    )


@router.get("/v1/operators", response_model=OperatorListResponse)
async def list_operators(
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> OperatorListResponse:
    """List operators on the authenticated team (newest first)."""
    ops = await store.list_operators_for_team(customer.id)
    return OperatorListResponse(
        total=len(ops),
        operators=[_operator_response(o) for o in ops],
    )


# NOTE: /me is declared BEFORE /{operator_id} so the literal path wins —
# otherwise "me" would bind to operator_id.
@router.get("/v1/operators/me", response_model=OperatorResponse)
async def get_me(
    operator: Annotated[Operator, Depends(require_operator)],
) -> OperatorResponse:
    """Identify the operator behind the presented operator key."""
    return _operator_response(operator)


@router.get("/v1/operators/{operator_id}", response_model=OperatorResponse)
async def get_operator(
    operator_id: str,
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> OperatorResponse:
    """Get one operator on the authenticated team."""
    op = await _team_scoped_operator(operator_id, customer, store)
    return _operator_response(op)


@router.patch("/v1/operators/{operator_id}/role", response_model=OperatorResponse)
async def set_operator_role(
    operator_id: str,
    body: SetOperatorRoleRequest,
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(require_role("admin"))
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> OperatorResponse:
    """Change an operator's role (team-scoped). Admin-gated (require_role)."""
    customer, _actor = principal
    await _team_scoped_operator(operator_id, customer, store)
    await store.set_operator_role(operator_id, body.role)
    logger.info(
        "operator.role_changed",
        operator_id=operator_id,
        team_id=customer.id,
        role=body.role,
    )
    updated = await store.get_operator_by_id(operator_id)
    return _operator_response(updated)


@router.patch("/v1/operators/{operator_id}/status", response_model=OperatorResponse)
async def set_operator_status(
    operator_id: str,
    body: SetOperatorStatusRequest,
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(require_role("admin"))
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> OperatorResponse:
    """Activate or suspend an operator (team-scoped). Admin-gated
    (require_role). Suspension preserves the row; auth denies a suspended
    operator at the boundary."""
    customer, _actor = principal
    await _team_scoped_operator(operator_id, customer, store)
    await store.set_operator_status(operator_id, body.status)
    logger.info(
        "operator.status_changed",
        operator_id=operator_id,
        team_id=customer.id,
        status=body.status,
    )
    updated = await store.get_operator_by_id(operator_id)
    return _operator_response(updated)
