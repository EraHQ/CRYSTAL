"""Identity surface: GET /v1/me, POST /v1/auth/signup, POST /v1/me/onboarding.

Phase A (2026-07-06): /v1/me — who am I? (all principal kinds).
Phase C (2026-07-06): signup provisioning + onboarding capture.

The frontend pins itself with this: one call resolves the bearer to its
principal kind, tenant, and role. Accepts all three principal kinds —
the static platform-admin key, a hosted Firebase JWT (users table), and
a tenant Key A (plus operator keys, which resolve to their team). 401
for anything else; the response never guesses.
"""
from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from ..config import get_settings
from ..infrastructure.metadata_store import MetadataStore, get_metadata_store
from ..ingress import auth as auth_mod
from ..ingress.auth import (
    _bearer_token_from_header,
    _looks_like_firebase_jwt,
    is_platform_admin_token,
    resolve_firebase_user,
)

# Managed-signup provisioning defaults (Phase C, decided 2026-07-06):
# the first impression runs on the good model — the tier's monthly cap
# bounds the spend risk. Scans stay OFF (spend-bearing curation is
# opt-in); chat is ON by virtue of inference_mode=managed.
SIGNUP_DEFAULT_PROVIDER = "anthropic"
SIGNUP_DEFAULT_MODEL = "claude-sonnet-5"

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


@router.post("/v1/auth/signup")
async def signup(
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> dict:
    """Provision a tenant for a first-time hosted sign-in (Phase C).

    Auth = a VALID Firebase JWT (verified directly — the resolver would
    reject an unprovisioned user, which is exactly the state signup
    exists to fix). Creates the Customer (inference_mode=managed, no
    Key B — the platform key serves; default model per the launch
    decision) and the owner User row. Returns identity + Key A — the
    ONE time the raw key is shown (hashed at rest).

    Idempotent for an already-provisioned uid (returns the existing
    identity, no new tenant, key=None). Allowlisted admin emails don't
    come through here — the resolver bootstraps them platform_admin at
    first touch of any authed route.

    Optional body: {"industry", "building", "experience"} — the
    onboarding signal, captured in OUR database at signup time.
    """
    settings = get_settings()
    if not (getattr(settings, "firebase_project_id", "") or "").strip():
        raise HTTPException(
            status_code=404, detail="Hosted signup is not enabled")

    auth = (
        request.headers.get("authorization")
        or request.headers.get("Authorization")
    )
    bearer = _bearer_token_from_header(auth)
    if not bearer or not _looks_like_firebase_jwt(bearer):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    # Through the MODULE attribute — the verification seam lives in
    # ingress.auth and must stay monkeypatchable from there.
    claims = auth_mod._verify_firebase_jwt(
        bearer, settings.firebase_project_id.strip()
    )
    if not claims:
        raise HTTPException(status_code=401, detail="Invalid token")
    uid = claims.get("sub") or claims.get("user_id") or ""
    email = (claims.get("email") or "").strip().lower()
    if not uid or not email:
        raise HTTPException(status_code=401, detail="Invalid token")

    existing = await store.get_user_by_id(uid)
    if existing is not None:
        return {
            "created": False,
            "user_id": existing.id,
            "email": existing.email,
            "role": existing.role,
            "customer_id": existing.customer_id,
            "api_key": None,  # shown once, at creation only
        }
    if email in auth_mod._admin_bootstrap_emails():
        # The resolver owns admin bootstrap; signup never mints a tenant
        # for the platform root.
        user = await resolve_firebase_user(store, bearer)
        return {
            "created": True,
            "user_id": user.id,
            "email": user.email,
            "role": user.role,
            "customer_id": None,
            "api_key": None,
        }

    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 — body is optional
        body = {}
    body = body if isinstance(body, dict) else {}

    customer = await store.create_customer(
        provider=SIGNUP_DEFAULT_PROVIDER,
        model_id=SIGNUP_DEFAULT_MODEL,
        api_key_ref="",  # managed: the platform key serves; no Key B
    )
    await store.set_customer_inference_mode(customer.id, "managed")
    user = await store.create_user(uid, email, customer.id, "owner")
    if any(body.get(k) for k in ("industry", "building", "experience")):
        await store.update_user_onboarding(
            uid,
            industry=body.get("industry"),
            building=body.get("building"),
            experience=body.get("experience"),
        )
    return {
        "created": True,
        "user_id": user.id,
        "email": user.email,
        "role": user.role,
        "customer_id": customer.id,
        "api_key": customer.api_key,  # THE one-time reveal
    }


@router.post("/v1/me/onboarding")
async def update_onboarding(
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> dict:
    """Capture/refresh the onboarding signal for the signed-in user
    (Phase C). JWT principals only — key-based callers have no user row."""
    auth = (
        request.headers.get("authorization")
        or request.headers.get("Authorization")
    )
    bearer = _bearer_token_from_header(auth)
    if not bearer or not _looks_like_firebase_jwt(bearer):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    user = await resolve_firebase_user(store, bearer)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    body = await request.json()
    body = body if isinstance(body, dict) else {}
    updated = await store.update_user_onboarding(
        user.id,
        industry=body.get("industry"),
        building=body.get("building"),
        experience=body.get("experience"),
    )
    return {
        "user_id": updated.id,
        "industry": updated.industry,
        "building": updated.building,
        "experience": updated.experience,
    }
