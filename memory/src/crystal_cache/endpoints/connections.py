"""Connected apps and the Beta list: console routes (2026-10-24).

Tenant-pathed under /admin/api/customers/{customer_id}/apps (the tenant
itself and a platform admin reach them; ingress/auth _TENANT_PATH_RE):

GET    .../apps                          the catalog for this tenant + its connections
POST   .../apps/session {integration}    a Nango Connect UI session token
POST   .../apps/{id}/reconnect           a reconnect session (expired sign-in)
PATCH  .../apps/{id}                     {uses, rules, cadence_minutes, paused}
POST   .../apps/{id}/sync                pull now
DELETE .../apps/{id}                     disconnect (memory made from it stays)
GET    .../apps/{id}/items               what it showed in the last 7 days

Platform admin only (/admin/api/beta, not tenant-pathed):

GET    /admin/api/beta                   features and invites
PATCH  /admin/api/beta/features/{key}    {status: open | invite | closed}
POST   /admin/api/beta/invites           {feature_key, email | customer_id, note}
DELETE /admin/api/beta/invites/{id}

Public, Nango-signed (X-Nango-Hmac-Sha256), no other auth BY DESIGN:

POST   /v1/connectors/nango/webhook
"""
from __future__ import annotations

import json
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from .. import beta
from ..connectors import nango, service
from ..infrastructure import MetadataStore
from ..infrastructure.metadata_store import get_metadata_store
from ..ingress.errors import build_error_envelope
from ..ingress.limits import ID64

logger = structlog.get_logger(__name__)
router = APIRouter()


def _refused(e: service.ConnectionRefused) -> HTTPException:
    return HTTPException(status_code=e.status, detail=build_error_envelope(
        e.message, error_type="invalid_request_error", param="connection", code=e.code))


def _nango_failed(e: nango.NangoError) -> HTTPException:
    status = 503 if e.code in ("connections_unavailable", "nango_unreachable") else 502
    return HTTPException(status_code=status, detail=build_error_envelope(
        "The connection service didn't answer; try again in a minute.", error_type="api_error",
        param="connection", code=e.code))


async def _body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    return body


@router.get("/admin/api/customers/{customer_id}/apps")
async def list_apps(customer_id: ID64, store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    if await store.get_customer_by_id(customer_id) is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return JSONResponse(content=await service.catalog_view(store, customer_id))


@router.post("/admin/api/customers/{customer_id}/apps/session")
async def start_app_session(customer_id: ID64, request: Request,
                            store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    body = await _body(request)
    integration = str(body.get("integration") or "")
    try:
        out = await service.start_session(store, customer_id, integration)
    except service.ConnectionRefused as e:
        raise _refused(e)
    except nango.NangoError as e:
        raise _nango_failed(e)
    return JSONResponse(content=out)


@router.post("/admin/api/customers/{customer_id}/apps/{connection_id}/reconnect")
async def reconnect_app(customer_id: ID64, connection_id: ID64,
                        store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    try:
        out = await service.reconnect(store, customer_id, connection_id)
    except service.ConnectionRefused as e:
        raise _refused(e)
    except nango.NangoError as e:
        raise _nango_failed(e)
    return JSONResponse(content=out)


@router.patch("/admin/api/customers/{customer_id}/apps/{connection_id}")
async def update_app(customer_id: ID64, connection_id: ID64, request: Request,
                     store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    body = await _body(request)
    unknown = set(body) - {"uses", "rules", "cadence_minutes", "paused"}
    if unknown:
        raise _refused(service.ConnectionRefused("invalid_request", f"unknown field(s) {sorted(unknown)}"))
    paused = body.get("paused")
    if paused is not None and not isinstance(paused, bool):
        raise _refused(service.ConnectionRefused("invalid_request", "paused must be true or false"))
    try:
        out = await service.set_uses(store, customer_id, connection_id, uses=body.get("uses"), rules=body.get("rules"),
                                     cadence_minutes=body.get("cadence_minutes"), paused=paused)
    except service.ConnectionRefused as e:
        raise _refused(e)
    return JSONResponse(content={"connection": out})


@router.post("/admin/api/customers/{customer_id}/apps/{connection_id}/sync")
async def sync_app_now(customer_id: ID64, connection_id: ID64,
                       store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    out = await store.update_app_connection(customer_id, connection_id, sync_due=True)
    if out is None:
        raise HTTPException(status_code=404, detail="Connection not found")
    return JSONResponse(content={"queued": True, "connection_id": connection_id})


@router.delete("/admin/api/customers/{customer_id}/apps/{connection_id}")
async def disconnect_app(customer_id: ID64, connection_id: ID64,
                         store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    try:
        await service.remove(store, customer_id, connection_id)
    except service.ConnectionRefused as e:
        raise _refused(e)
    return JSONResponse(content={"deleted": True, "connection_id": connection_id})


@router.get("/admin/api/customers/{customer_id}/apps/{connection_id}/items")
async def list_app_items(customer_id: ID64, connection_id: ID64, request: Request,
                         store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    if await store.get_app_connection(customer_id, connection_id) is None:
        raise HTTPException(status_code=404, detail="Connection not found")
    try:
        limit = max(1, min(int(request.query_params.get("limit") or 50), 200))
    except ValueError:
        limit = 50
    items = await store.list_watched_items(customer_id, connection_id=connection_id, limit=limit)
    for it in items:
        it.pop("body", None)  # the list shows snippets; bodies stay server-side
        it["remembered"] = bool(it.pop("remembered_document_id", None))
    return JSONResponse(content={"items": items, "count": len(items)})


# ------------------------------------------------------------------ webhook

@router.post("/v1/connectors/nango/webhook")
async def nango_webhook(request: Request, store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> dict:
    """No auth dependency BY DESIGN: Nango calls this, and the verified
    HMAC (X-Nango-Hmac-Sha256 over the raw body) is the authentication.
    Nothing in the body is read before the check passes."""
    raw = await request.body()
    if not nango.verify_webhook(raw, request.headers.get("x-nango-hmac-sha256")):
        raise HTTPException(status_code=401, detail="invalid signature")
    try:
        payload = json.loads(raw.decode("utf-8") or "{}")
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="invalid body")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="invalid body")
    out = await service.handle_webhook(store, payload)
    logger.info("connections.webhook", type=payload.get("type"), action=out.get("action"))
    return {"received": True}


# --------------------------------------------------------------- beta list

@router.get("/admin/api/beta")
async def beta_list(store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    await beta.ensure_defaults(store)
    return JSONResponse(content={"features": await store.list_beta_features(),
                                 "invites": await store.list_beta_invites()})


@router.patch("/admin/api/beta/features/{key}")
async def beta_feature_status(key: ID64, request: Request,
                              store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    await beta.ensure_defaults(store)
    body = await _body(request)
    status = body.get("status")
    if status not in beta.STATUSES:
        raise HTTPException(status_code=400, detail=build_error_envelope(
            f"status must be one of {list(beta.STATUSES)}", error_type="invalid_request_error", param="status",
            code="invalid_status"))
    out = await store.set_beta_feature_status(key, status)
    if out is None:
        raise HTTPException(status_code=404, detail="No such feature")
    return JSONResponse(content={"feature": out})


@router.post("/admin/api/beta/invites")
async def beta_invite(request: Request, store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    await beta.ensure_defaults(store)
    body = await _body(request)
    key = str(body.get("feature_key") or "")
    if key not in {f["key"] for f in await store.list_beta_features()}:
        raise HTTPException(status_code=404, detail="No such feature")
    email = str(body.get("email") or "").strip().lower() or None
    customer_id = str(body.get("customer_id") or "").strip() or None
    if email and ("@" not in email or len(email) > 320):
        raise HTTPException(status_code=400, detail=build_error_envelope(
            "That isn't an email address.", error_type="invalid_request_error", param="email", code="invalid_email"))
    if customer_id and await store.get_customer_by_id(customer_id) is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    if not email and not customer_id:
        raise HTTPException(status_code=400, detail=build_error_envelope(
            "Invite an email or a customer.", error_type="invalid_request_error", param="email", code="invite_target"))
    if email and not customer_id:
        user = await store.get_user_by_email(email)
        if user is not None and getattr(user, "customer_id", None):
            customer_id = user.customer_id
    invite = await store.create_beta_invite(key, customer_id=customer_id, email=email,
                                            invited_by="platform admin", note=str(body.get("note") or "")[:256])
    return JSONResponse(status_code=201, content={"invite": invite})


@router.delete("/admin/api/beta/invites/{invite_id}")
async def beta_uninvite(invite_id: ID64, store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    if not await store.delete_beta_invite(invite_id):
        raise HTTPException(status_code=404, detail="No such invite")
    return JSONResponse(content={"deleted": True, "invite_id": invite_id})
