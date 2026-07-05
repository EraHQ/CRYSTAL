"""Google Drive endpoints — /v1/connectors/gdrive/*.

OAuth flow, connection management, browse folders, watched folders +
files CRUD. Refactored to use Phase 5 MetadataStore methods.

Integration shape (per Phase 6.5 P0.2): both v1 client-side and v1
server-side import paths supported, with verbatim v1 URLs.

Endpoints (matching v1 verbatim):
  GET    /v1/connectors/gdrive/auth-url               return URL user visits
  GET    /v1/connectors/gdrive/callback               OAuth callback → redirect
  GET    /v1/connectors/gdrive/connections            list connections
  DELETE /v1/connectors/gdrive/{connection_id}        disconnect (cascading)
  GET    /v1/connectors/gdrive/{connection_id}/browse browse folders/files
  POST   /v1/connectors/gdrive/{connection_id}/folders        watch folder
  GET    /v1/connectors/gdrive/{connection_id}/folders        list watched
  DELETE /v1/connectors/gdrive/folders/{watch_id}             unwatch folder
  POST   /v1/connectors/gdrive/{connection_id}/files          watch file
  GET    /v1/connectors/gdrive/{connection_id}/watched-files  list watched
  DELETE /v1/connectors/gdrive/watched-files/{watch_id}       unwatch file
  POST   /v1/connectors/gdrive/import                 client-side import
  POST   /v1/connectors/gdrive/import-from-drive      server-side import (v2)

Two import models supported:

  /import accepts `{files: [{drive_id, title, text, mime_type, ...}]}`
  where the frontend has already fetched text via Claude.ai's MCP
  connector. Matches v1 verbatim.

  /import-from-drive accepts `{connection_id, file_ids: [...]}` and
  uses stored OAuth tokens to fetch text server-side. Added in v2
  for clients that don't have an MCP Drive integration.

The OAuth callback redirects to `/admin/knowledge?drive=connected`
(matches v1) rather than returning JSON, so the user lands back in
the inspector UI after authorizing.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

from ..config import settings
from ..infrastructure import MetadataStore
from ..infrastructure.metadata_store import get_metadata_store
from ..ingress.auth import require_customer
from ..models import Customer

logger = structlog.get_logger(__name__)

router = APIRouter()


# --- Request bodies ---

class AddWatchedFolderRequest(BaseModel):
    folder_id: str
    folder_name: str
    folder_path: Optional[str] = None
    contains_phi: bool = False
    sync_interval_minutes: int = 60


class AddWatchedFileRequest(BaseModel):
    file_id: str
    file_name: str
    mime_type: Optional[str] = None
    contains_phi: bool = False
    sync_interval_minutes: int = 60


class ImportFromDriveRequest(BaseModel):
    """Server-side bulk import: fetch text via stored OAuth token."""
    connection_id: str
    file_ids: list[str]
    crystal_type: str = "customer:legacy"


# --- Helpers ---

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


def _folder_to_dict(f) -> dict[str, Any]:
    return {
        "id": f.id,
        "connection_id": f.connection_id,
        "folder_id": f.folder_id,
        "folder_name": f.folder_name,
        "folder_path": f.folder_path,
        "contains_phi": f.contains_phi,
        "sync_interval_minutes": f.sync_interval_minutes,
        "last_checked_at": f.last_checked_at.isoformat() if f.last_checked_at else None,
        "last_file_count": f.last_file_count,
        "status": f.status,
        "created_at": f.created_at.isoformat(),
    }


def _file_to_dict(wf) -> dict[str, Any]:
    return {
        "id": wf.id,
        "connection_id": wf.connection_id,
        "file_id": wf.file_id,
        "file_name": wf.file_name,
        "mime_type": wf.mime_type,
        "contains_phi": wf.contains_phi,
        "sync_interval_minutes": wf.sync_interval_minutes,
        "last_checked_at": wf.last_checked_at.isoformat() if wf.last_checked_at else None,
        "last_modified_at": wf.last_modified_at.isoformat() if wf.last_modified_at else None,
        "status": wf.status,
        "created_at": wf.created_at.isoformat(),
    }


# --- OAuth ---

@router.get("/v1/connectors/gdrive/auth-url")
async def gdrive_auth_url(
    request: Request,
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Return the URL the user should visit to authorize Drive access.

    F1 CSRF fix (2026-07-03): the state is an OPAQUE single-use nonce
    persisted server-side (oauth_states table) and mapped to the
    customer there — nothing about the customer is encoded in the state
    itself, and the callback only proceeds when the presented state
    exists, is fresh (10-minute TTL), and has not been redeemed before.
    This kills both directions of OAuth CSRF: an attacker can neither
    forge a state for a victim customer nor replay a captured one.
    """
    import secrets

    from ..infrastructure.drive_connector import build_auth_url

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
    from ..infrastructure.token_crypto import encrypt_token

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

    # Encrypt the refresh token at rest
    encrypted, nonce = encrypt_token(refresh_token)
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


# --- Connection CRUD ---

@router.get("/v1/connectors/gdrive/connections")
async def gdrive_list_connections(
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    conns = await store.list_drive_connections(customer.id)
    return JSONResponse(content={
        "connections": [_connection_to_dict(c) for c in conns],
    })


@router.delete("/v1/connectors/gdrive/{connection_id}")
async def gdrive_disconnect(
    connection_id: str,
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Cascade: removes watched folders + files + the connection.

    Matches v1's URL shape `/v1/connectors/gdrive/{connection_id}`
    (no extra `/connections/` segment).
    """
    conn = await store.get_drive_connection(connection_id, customer.id)
    if conn is None:
        raise HTTPException(status_code=404, detail="Connection not found")
    await store.delete_drive_connection_with_watches(connection_id, customer.id)
    return JSONResponse(content={"deleted": True, "connection_id": connection_id})


@router.get("/v1/connectors/gdrive/{connection_id}/browse")
async def gdrive_browse_folders(
    connection_id: str,
    request: Request,
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Browse folders + files for the folder picker UI.

    Query param: parent_id (default 'root'). Matches v1's response
    shape: `{parent_id, folders: [...], files: [...]}` where folders
    is `[{id, name, type: 'folder'}]` and files include modifiedTime
    and size.
    """
    from ..infrastructure.drive_connector import (
        refresh_access_token, list_folders, list_folder_files,
    )

    conn = await store.get_drive_connection(connection_id, customer.id)
    if conn is None:
        raise HTTPException(status_code=404, detail="Connection not found")

    try:
        access_token = await refresh_access_token(
            conn.encrypted_refresh_token, conn.token_nonce,
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token refresh failed: {e}")

    parent_id = request.query_params.get("parent_id", "root")

    folders = await list_folders(access_token, parent_id)
    files = await list_folder_files(access_token, parent_id)

    return JSONResponse(content={
        "parent_id": parent_id,
        "folders": [
            {"id": f["id"], "name": f["name"], "type": "folder"}
            for f in folders
        ],
        "files": [
            {
                "id": f["id"],
                "name": f.get("name", "Untitled"),
                "mimeType": f.get("mimeType", ""),
                "modifiedTime": f.get("modifiedTime", ""),
                "size": f.get("size"),
            }
            for f in files
        ],
    })


# --- Watched folders ---

@router.post("/v1/connectors/gdrive/{connection_id}/folders")
async def gdrive_add_watched_folder(
    connection_id: str,
    body: AddWatchedFolderRequest,
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    conn = await store.get_drive_connection(connection_id, customer.id)
    if conn is None:
        raise HTTPException(status_code=404, detail="Connection not found")

    folder = await store.create_watched_folder(
        watch_id=f"wf_{uuid.uuid4().hex[:16]}",
        connection_id=connection_id,
        customer_id=customer.id,
        folder_id=body.folder_id,
        folder_name=body.folder_name,
        folder_path=body.folder_path,
        contains_phi=body.contains_phi,
        sync_interval_minutes=body.sync_interval_minutes,
    )
    return JSONResponse(content=_folder_to_dict(folder))


@router.get("/v1/connectors/gdrive/{connection_id}/folders")
async def gdrive_list_watched_folders(
    connection_id: str,
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    folders = await store.list_watched_folders_for_connection(connection_id, customer.id)
    return JSONResponse(content={"folders": [_folder_to_dict(f) for f in folders]})


@router.delete("/v1/connectors/gdrive/folders/{watch_id}")
async def gdrive_remove_watched_folder(
    watch_id: str,
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    folder = await store.get_watched_folder(watch_id, customer.id)
    if folder is None:
        raise HTTPException(status_code=404, detail="Watched folder not found")
    await store.delete_watched_folder(watch_id, customer.id)
    return JSONResponse(content={"deleted": True, "watch_id": watch_id})


# --- Watched files ---

@router.post("/v1/connectors/gdrive/{connection_id}/files")
async def gdrive_add_watched_file(
    connection_id: str,
    body: AddWatchedFileRequest,
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    conn = await store.get_drive_connection(connection_id, customer.id)
    if conn is None:
        raise HTTPException(status_code=404, detail="Connection not found")

    wf = await store.create_watched_file(
        watch_id=f"wfl_{uuid.uuid4().hex[:16]}",
        connection_id=connection_id,
        customer_id=customer.id,
        file_id=body.file_id,
        file_name=body.file_name,
        mime_type=body.mime_type,
        contains_phi=body.contains_phi,
        sync_interval_minutes=body.sync_interval_minutes,
    )
    return JSONResponse(content=_file_to_dict(wf))


@router.get("/v1/connectors/gdrive/{connection_id}/watched-files")
async def gdrive_list_watched_files(
    connection_id: str,
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """List watched files. URL matches v1 verbatim (`/watched-files`,
    not `/files` to distinguish from POST `/files` which adds one)."""
    files = await store.list_watched_files_for_connection(connection_id, customer.id)
    return JSONResponse(content={"files": [_file_to_dict(f) for f in files]})


@router.delete("/v1/connectors/gdrive/watched-files/{watch_id}")
async def gdrive_remove_watched_file(
    watch_id: str,
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    wf = await store.get_watched_file(watch_id, customer.id)
    if wf is None:
        raise HTTPException(status_code=404, detail="Watched file not found")
    await store.delete_watched_file(watch_id, customer.id)
    return JSONResponse(content={"deleted": True, "watch_id": watch_id})


# --- Client-side import (v1 verbatim) ---

@router.post("/v1/connectors/gdrive/import")
async def gdrive_import_files(
    request: Request,
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Import files whose text was already extracted client-side.

    The frontend uses Claude.ai's MCP Drive connector to fetch and
    extract file text in the browser, then POSTs the extracted text
    here. Each file becomes a pending DocumentUpload; the
    crystallization worker picks them up.

    Body shape (matches v1):
      {
        "files": [
          {"drive_id": "...", "title": "...", "text": "...",
           "mime_type": "...", "crystal_type": "..."},
          ...
        ]
      }
    """
    body = await request.json()
    files = body.get("files", [])

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    imported: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for f in files:
        drive_id = f.get("drive_id", "")
        title = f.get("title", "Untitled")
        text = f.get("text", "")
        crystal_type = f.get("crystal_type", "customer:legacy")

        if not text or not text.strip():
            skipped.append({"drive_id": drive_id, "title": title, "reason": "no_text_content"})
            continue

        doc = await store.create_document_upload(
            customer_id=customer.id,
            label=title,
            text=text,
            crystal_type=crystal_type,
            source_file_id=drive_id or None,
        )
        imported.append({
            "document_id": doc.id,
            "drive_id": drive_id,
            "title": title,
            "char_count": doc.char_count,
            "status": doc.status,
        })

    logger.info(
        "gdrive.import",
        customer_id=customer.id,
        imported=len(imported),
        skipped=len(skipped),
    )

    msg = f"Imported {len(imported)} files. Background worker will crystallize them automatically."
    if skipped:
        msg += f" Skipped {len(skipped)} files (no text content)."

    return JSONResponse(content={
        "imported": imported,
        "skipped": skipped,
        "total_imported": len(imported),
        "total_skipped": len(skipped),
        "message": msg,
    })


# --- Server-side import (v2 addition; uses stored OAuth) ---

@router.post("/v1/connectors/gdrive/import-from-drive")
async def gdrive_import_from_drive(
    body: ImportFromDriveRequest,
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Server-side bulk import using stored OAuth tokens.

    Added in v2 for SDK clients that don't have an MCP Drive
    integration in the browser. Equivalent functionality to the
    client-side `/import` endpoint but takes file IDs and fetches
    text server-side.

    Body shape:
      {"connection_id": "...", "file_ids": [...], "crystal_type": "..."}
    """
    from ..infrastructure.drive_connector import (
        refresh_access_token, get_file_metadata, read_file_text,
    )

    conn = await store.get_drive_connection(body.connection_id, customer.id)
    if conn is None:
        raise HTTPException(status_code=404, detail="Connection not found")

    try:
        access_token = await refresh_access_token(
            conn.encrypted_refresh_token, conn.token_nonce,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Token refresh failed: {e}")

    imported = 0
    failed = 0
    for file_id in body.file_ids:
        try:
            meta = await get_file_metadata(access_token, file_id)
            mime = meta.get("mimeType", "")
            file_name = meta.get("name", "Untitled")
            modified_str = meta.get("modifiedTime", "")
            file_modified_at = None
            if modified_str:
                try:
                    file_modified_at = datetime.fromisoformat(
                        modified_str.replace("Z", "+00:00")
                    )
                except ValueError:
                    pass

            text = await read_file_text(access_token, file_id, mime)
            if not text or not text.strip():
                continue

            await store.create_document_upload(
                customer_id=customer.id,
                label=file_name,
                text=text,
                crystal_type=body.crystal_type,
                source_file_id=file_id,
                source_modified_at=file_modified_at,
                source_connection_id=body.connection_id,
            )
            imported += 1
        except Exception as e:
            logger.warning("gdrive.import_from_drive.one_failed", file_id=file_id, error=str(e))
            failed += 1

    return JSONResponse(content={
        "imported": imported,
        "failed": failed,
        "total_requested": len(body.file_ids),
    })
