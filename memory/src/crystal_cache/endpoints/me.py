"""GET /v1/me — who am I? (Accounts Phase A, 2026-07-06, ratified plan).

The frontend pins itself with this: one call resolves the bearer to its
principal kind, tenant, and role. Accepts all three principal kinds —
the static platform-admin key, a hosted Firebase JWT (users table), and
a tenant Key A (plus operator keys, which resolve to their team). 401
for anything else; the response never guesses.
"""
from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from ..infrastructure.metadata_store import MetadataStore, get_metadata_store
from ..ingress.auth import (
    _bearer_token_from_header,
    _looks_like_firebase_jwt,
    is_platform_admin_token,
    resolve_firebase_user,
)

router = APIRouter(tags=["identity"])


@router.get("/v1/me")
async def get_me(
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> dict:
    auth = (
        request.headers.get("authorization")
        or request.headers.get("Authorization")
    )
    bearer = _bearer_token_from_header(auth)
    if not bearer:
        raise HTTPException(status_code=401, detail="Missing bearer token")

    # 1) The static platform-admin key: the platform root.
    if is_platform_admin_token(bearer):
        return {
            "kind": "platform_admin_key",
            "role": "platform_admin",
            "customer_id": None,
            "user_id": None,
            "email": None,
        }

    # 2) Hosted identity (Firebase JWT -> users table).
    if _looks_like_firebase_jwt(bearer):
        user = await resolve_firebase_user(store, bearer)
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {
            "kind": "user",
            "role": user.role,
            "customer_id": user.customer_id,
            "user_id": user.id,
            "email": user.email,
        }

    # 3) Tenant credentials: operator key first (never falls through to
    # the team path — same ordering as resolve_principal), then Key A.
    operator = await store.get_operator_by_api_key(bearer)
    if operator is not None:
        if operator.status != "active":
            raise HTTPException(status_code=403, detail="Operator is suspended")
        return {
            "kind": "operator",
            "role": operator.role,
            "customer_id": operator.team_id,
            "user_id": operator.id,
            "email": None,
        }

    customer = await store.get_customer_by_api_key(bearer)
    if customer is not None:
        return {
            "kind": "customer_key",
            "role": "owner",
            "customer_id": customer.id,
            "user_id": None,
            "email": None,
        }

    raise HTTPException(status_code=401, detail="Invalid token")
