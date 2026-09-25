"""OAuth consent bridge (L2-S5b, 2026-09-24).

Two endpoints behind the Inspector's /oauth/consent page (Q4=A):

  GET  /v1/oauth/client/{client_id}  — public client metadata for the
       consent card ("Claude wants to connect"). Client metadata is
       public by construction (the client supplied it at registration).
  POST /v1/oauth/approve             — the signed-in user approves; we
       resolve their seat, mint the single-use code via the provider,
       and hand back the redirect target.

SECURITY INVARIANT: the redirect_uri is honored ONLY if it exactly
matches one the client registered. A consent page can be phished into
forwarding query params; the registered-URI check is what makes that
harmless.
"""
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ..config import get_settings
from ..control.oauth_provider import CrystalOAuthProvider
from ..infrastructure.metadata_store import MetadataStore, get_metadata_store
from ..ingress.auth import (
    _bearer_token_from_header,
    _looks_like_firebase_jwt,
    resolve_firebase_user,
)

router = APIRouter()


def _require_enabled() -> None:
    if not get_settings().oauth_enabled:
        raise HTTPException(status_code=404, detail="Not found")


@router.get("/v1/oauth/client/{client_id}")
async def oauth_client_info(
    client_id: str,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> dict:
    _require_enabled()
    provider = CrystalOAuthProvider(store)
    client = await provider.get_client(client_id)
    if client is None:
        raise HTTPException(status_code=404, detail="Unknown client")
    return {
        "client_id": client.client_id,
        "client_name": client.client_name or "An MCP client",
        "redirect_hosts": sorted(
            {str(u).split("/")[2] for u in client.redirect_uris}
        ),
    }


class ApproveRequest(BaseModel):
    client_id: str = Field(max_length=128)
    redirect_uri: str = Field(max_length=1024)
    state: str = Field(default="", max_length=1024)
    code_challenge: str = Field(max_length=256)
    scopes: str = Field(default="", max_length=512)
    explicit_redirect: bool = True
    resource: Optional[str] = Field(default=None, max_length=1024)


@router.post("/v1/oauth/approve")
async def oauth_approve(
    body: ApproveRequest,
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> dict:
    _require_enabled()
    auth = (
        request.headers.get("authorization")
        or request.headers.get("Authorization")
    )
    bearer = _bearer_token_from_header(auth)
    if not bearer or not _looks_like_firebase_jwt(bearer):
        raise HTTPException(status_code=401, detail="Session required")
    user = await resolve_firebase_user(store, bearer)
    if user is None or not user.customer_id:
        raise HTTPException(status_code=401, detail="Invalid session")

    provider = CrystalOAuthProvider(store)
    client = await provider.get_client(body.client_id)
    if client is None:
        raise HTTPException(status_code=404, detail="Unknown client")
    # THE invariant: only registered redirect targets are honored.
    registered = {str(u).rstrip("/") for u in client.redirect_uris}
    if body.redirect_uri.rstrip("/") not in registered:
        raise HTTPException(
            status_code=400, detail="redirect_uri is not registered"
        )

    # The user's seat — same resolution signup uses (solo tenants have
    # exactly one, the default admin; pinned same-seat in T2a).
    operator = await store.ensure_default_admin(user.customer_id)

    code = await provider.create_authorization_code(
        client_id=body.client_id,
        operator_id=operator.id,
        redirect_uri=body.redirect_uri,
        code_challenge=body.code_challenge,
        scopes=[s for s in body.scopes.split(" ") if s],
        redirect_uri_provided_explicitly=body.explicit_redirect,
        resource=body.resource,
    )
    sep = "&" if "?" in body.redirect_uri else "?"
    redirect_to = f"{body.redirect_uri}{sep}code={code}"
    if body.state:
        from urllib.parse import quote

        redirect_to += f"&state={quote(body.state)}"
    return {"redirect_to": redirect_to}
