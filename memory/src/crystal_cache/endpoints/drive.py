"""Google Drive endpoints — DRIVE-Q1=B (2026-07-24).

The unification cut: Drive is a source-watch scheme, not a parallel
ingestion system. This module now holds exactly two concerns:

1. OAuth — minting connections. The console-facing auth-url is a
   KEYLESS ADMIN route (the admin_customer_chat pattern: customer
   resolved from the path, no Bearer — the console has no customer
   key to hold since no-plaintext). The Google-facing callback stays
   on /v1 verbatim: Google's redirect carries no auth by design; the
   F1 single-use server-side state is the entire trust chain.

2. Connection CRUD — list + disconnect, keyless admin. Disconnect
   also removes any gdrive source_watches riding the connection.

Everything else the legacy module did — watched folders/files CRUD,
browse, client- and server-side imports — retired 2026-07-24 with the
watched_folders/watched_files tables. Folder watching is a normal
source_watch (scheme=gdrive) served by the standard watches API;
syncing is DriveSourceHandler under the one sync loop.
"""
from __future__ import annotations

import uuid
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

from ..infrastructure import MetadataStore
from ..infrastructure.metadata_store import get_metadata_store

logger = structlog.get_logger(__name__)

router = APIRouter()


def _connection_to_dict(conn) -> dict[str, Any]:
    return {
        "id": conn.id,
        "customer_id": conn.customer_id,
        "provider": conn.provider,
        "email": conn.email,
        "scopes": conn.scopes,
        "status": conn.status,
        "last_synced_at": conn.last_synced_at.isoformat() if conn.last_synced_at else None,
        "error_message": conn.error_message,
        "created_at": conn.created_at.isoformat(),
        # NEVER expose encrypted_refresh_token / token_nonce to the client.
    }


# --- OAuth ---

@router.get("/admin/api/customers/{customer_id}/gdrive/auth-url")
async def admin_gdrive_auth_url(
    request: Request,
    customer_id: str,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Keyless admin (2026-07-24): the console's path to connecting
    Drive. Customer resolved from the path — the tenant-pathed
    allowlist admits own-id reads, and no customer Bearer exists in
    the accounts world to demand.

    F1 CSRF posture unchanged: the state is an OPAQUE single-use nonce
    persisted server-side (oauth_states) mapped to the customer there;
    the callback proceeds only for a fresh, unredeemed state.
    """
    import secrets

    from ..infrastructure.drive_connector import build_auth_url

    customer = await store.get_customer_by_id(customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")

    redirect_uri = str(request.base_url).rstrip("/") + "/v1/connectors/gdrive/callback"
    state = secrets.token_urlsafe(32)
    await store.create_oauth_state(state, customer.id)

    try:
        url = build_auth_url(redirect_uri=redirect_uri, state=state)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content={"auth_url": url, "state": state})


@router.get("/v1/connectors/gdrive/callback")
async def gdrive_callback(
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
):
    """OAuth callback — exchanges authorization code for tokens.

    Google redirects here after the user grants consent. We exchange
    the code for tokens, encrypt the refresh token, persist the
    connection, and redirect back to the inspector UI. Matches v1
    verbatim including the redirect target.

    No Bearer auth on this endpoint: the user's browser is the
    requester, not the customer's backend. Customer identity comes
    from the `state` query param set in /auth-url.
    """
    from ..infrastructure.drive_connector import (
        exchange_code, get_user_email,
    )

    code = request.query_params.get("code")
    state = request.query_params.get("state", "")
    error = request.query_params.get("error")

    if error:
        logger.warning("gdrive.auth_denied", error=error)
        return JSONResponse(
            status_code=400,
            content={"error": f"OAuth denied: {error}"},
        )

    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code")

    # F1 CSRF fix: redeem the server-stored single-use state. Unknown,
    # already-used, or stale (>10 min) states are all rejected the same
    # way — the callback never trusts anything the state string claims.
    customer_id = await store.consume_oauth_state(state, max_age_seconds=600)
    if customer_id is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid, expired, or already-used state parameter",
        )

    customer = await store.get_customer_by_id(customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer in state not found")

    redirect_uri = str(request.base_url).rstrip("/") + "/v1/connectors/gdrive/callback"

    try:
        tokens = await exchange_code(code, redirect_uri)
    except Exception as e:
        logger.error("gdrive.token_exchange_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Token exchange failed: {e}")

    refresh_token = tokens.get("refresh_token")
    access_token = tokens.get("access_token")

    if not refresh_token:
        raise HTTPException(
            status_code=400,
            detail="No refresh token received. User may need to re-authorize with prompt=consent.",
        )

    # Get user email
    email = None
    if access_token:
        email = await get_user_email(access_token)

    # P4 (2026-07-10): enc:v2 under the tenant's DEK, family
    # "drive_oauth"; the composite string lives in the token column and
    # the nonce column carries the "v2" sentinel.
    encrypted = await store.encrypt_tenant_secret(
        customer_id, "drive_oauth", refresh_token
    )
    nonce = "v2"
    conn_id = f"drv_{uuid.uuid4().hex[:16]}"

    await store.create_drive_connection(
        customer_id=customer_id,
        connection_id=conn_id,
        email=email,
        encrypted_refresh_token=encrypted,
        token_nonce=nonce,
        scopes=tokens.get("scope"),
    )

    logger.info("gdrive.connected", customer_id=customer_id, connection_id=conn_id, email=email)

    # Redirect back to the inspector UI (matches v1).
    return RedirectResponse(url="/admin/knowledge?drive=connected")


# --- Connection CRUD (keyless admin, 2026-07-24) ---

@router.get("/admin/api/customers/{customer_id}/gdrive/connections")
async def admin_gdrive_list_connections(
    customer_id: str,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    conns = await store.list_drive_connections(customer_id)
    return JSONResponse(content={
        "connections": [_connection_to_dict(c) for c in conns],
    })


@router.delete("/admin/api/customers/{customer_id}/gdrive/connections/{connection_id}")
async def admin_gdrive_disconnect(
    customer_id: str,
    connection_id: str,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Disconnect: removes the connection AND any gdrive source_watches
    riding it (their crystals stay — retiring knowledge is a curation
    act, not a plumbing side effect)."""
    conn = await store.get_drive_connection(connection_id, customer_id)
    if conn is None:
        raise HTTPException(status_code=404, detail="Connection not found")

    removed_watches = 0
    watches = await store.list_source_watches(customer_id)
    for w in watches:
        if w.scheme == "gdrive" and (w.config or {}).get("connection_id") == connection_id:
            await store.delete_source_watch(w.id, customer_id)
            removed_watches += 1

    await store.delete_drive_connection(connection_id, customer_id)
    logger.info(
        "gdrive.disconnected",
        customer_id=customer_id,
        connection_id=connection_id,
        removed_watches=removed_watches,
    )
    return JSONResponse(content={
        "deleted": True,
        "connection_id": connection_id,
        "removed_watches": removed_watches,
    })
