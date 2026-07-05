"""Groups API — P3, ratified 2026-07-02.

Named sub-teams as grant targets. A group is the lightweight thing the
CLI needs so "share with backend" is one call: create a group, add
members, then grant crystals to it (POST /v1/crystals/{id}/grants in
sdk.py). Group management is ADMIN-only (it shapes who can be granted
what); listing is open to any team principal so share targets are
discoverable. Deliberately a human/API surface — no agent-facing tool.
"""
from __future__ import annotations

from typing import Annotated, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..infrastructure import MetadataStore
from ..infrastructure.metadata_store import get_metadata_store
from ..ingress.auth import resolve_principal
from ..models import Customer, Operator

logger = structlog.get_logger(__name__)

router = APIRouter()


class CreateGroupRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    member_ids: list[str] = Field(default_factory=list)


class GroupMemberRequest(BaseModel):
    operator_id: str = Field(min_length=1, max_length=64)


def _require_admin(operator: Optional[Operator]) -> None:
    if operator is None or operator.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Only a team admin may manage groups.",
        )


@router.post("/v1/groups")
async def create_group(
    body: CreateGroupRequest,
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(resolve_principal)
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Create a named group, optionally seeding members in the same call."""
    customer, operator = principal
    _require_admin(operator)
    try:
        group = await store.create_group(customer.id, body.name.strip())
    except Exception:
        # The (customer_id, name) unique index makes duplicates collide.
        raise HTTPException(
            status_code=409,
            detail=f"A group named {body.name.strip()!r} already exists.",
        )
    skipped = []
    for member_id in body.member_ids:
        ok = await store.add_group_member(group["id"], member_id, customer.id)
        if not ok:
            skipped.append(member_id)
    logger.info(
        "group.created", customer_id=customer.id,
        group_id=group["id"], name=group["name"], skipped=skipped,
    )
    return JSONResponse(content={**group, "skipped_member_ids": skipped})


@router.post("/v1/groups/{group_id}/members")
async def add_group_member(
    group_id: str,
    body: GroupMemberRequest,
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(resolve_principal)
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    customer, operator = principal
    _require_admin(operator)
    ok = await store.add_group_member(group_id, body.operator_id, customer.id)
    if not ok:
        raise HTTPException(
            status_code=404, detail="Unknown group or operator for this team",
        )
    return JSONResponse(content={"group_id": group_id,
                                 "operator_id": body.operator_id})


@router.delete("/v1/groups/{group_id}/members/{operator_id}")
async def remove_group_member(
    group_id: str,
    operator_id: str,
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(resolve_principal)
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    customer, operator = principal
    _require_admin(operator)
    ok = await store.remove_group_member(group_id, operator_id, customer.id)
    if not ok:
        raise HTTPException(status_code=404, detail="No such membership")
    return JSONResponse(content={"group_id": group_id,
                                 "operator_id": operator_id,
                                 "removed": True})


@router.get("/v1/groups")
async def list_groups(
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(resolve_principal)
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Groups with member ids — open to any team principal so share
    targets are discoverable."""
    customer, _operator = principal
    groups = await store.list_groups_for_customer(customer.id)
    return JSONResponse(content={"groups": groups, "total": len(groups)})
