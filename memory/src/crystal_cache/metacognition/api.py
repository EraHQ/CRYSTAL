"""Metacognition admin endpoints — Phase 10.5.

Mirrors `cognition/api.py` shape but lives under
`/admin/api/metacognition/*`. Phase 10.5 ships one endpoint:

  GET /admin/api/metacognition/substrate-observations
      [?customer_id=...] [?since=ISO] [?limit=N]

Returns the deferred substrate_observation action items with
composed critique + trace context. Thin pass-through to the
library function `list_substrate_observations` so the CLI + the
future admin dashboard both consume the same composition logic.

D-MCR-13 V1 surface (MCR §9, §11 Q7). The framework requires that
substrate observations be RECORDED and SURFACED, NOT auto-acted
on. This endpoint is the surfacing path; per Principle 9 / D-MCR-
15 the harness is never modified based on what gets read here.

Mounted in `app.py` via `app.include_router(metacog_api.router)`.
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Optional

import structlog
from fastapi import APIRouter, Depends, Request
from starlette.responses import JSONResponse

from ..infrastructure import MetadataStore
from ..infrastructure.metadata_store import get_metadata_store
from .substrate_review import (
    group_substrate_observations,
    list_substrate_observations,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/admin/api/metacognition", tags=["metacognition"])


@router.get("/substrate-observations")
async def list_substrate_observations_endpoint(
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    customer_id: Optional[str] = None,
    since: Optional[str] = None,
    limit: int = 50,
) -> JSONResponse:
    """Return deferred substrate observations with composed context.

    Query parameters:
      - customer_id (optional): scope to one customer; omitted =
        cross-tenant view.
      - since (optional, ISO 8601 datetime string): only items with
        created_at >= since.
      - limit (default 50, max 200): cap result count.

    Returns:
      {
        "total": N,
        "observations": [
          {
            "action_item": {...},
            "critique": {...} | null,
            "trace_summary": {...} | null,
          },
          ...
        ]
      }
    """
    # Tenant pin (Accounts Phase A): the guard middleware force-scopes
    # tenant principals — the pin overrides any caller-supplied
    # customer_id. Platform admins arrive unpinned (cross-tenant view).
    _pin = getattr(request.state, "tenant_pin", None)
    if _pin:
        customer_id = _pin

    # Parse + clamp the limit defensively.
    if limit < 1:
        limit = 1
    elif limit > 200:
        limit = 200

    parsed_since: Optional[datetime] = None
    if since is not None:
        try:
            parsed_since = datetime.fromisoformat(since)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={
                    "error": (
                        f"Invalid `since` value {since!r}; "
                        "expected ISO 8601 datetime."
                    ),
                },
            )

    views = await list_substrate_observations(
        store=store,
        customer_id=customer_id,
        since=parsed_since,
        limit=limit,
    )

    return JSONResponse(content={
        "total": len(views),
        "observations": [v.model_dump(mode="json") for v in views],
    })


@router.post("/substrate-observations/{item_id}/dismiss")
async def dismiss_substrate_observation(
    item_id: str,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """C1 (2026-07-08): soft-hide one observation — status → 'dropped'.
    The row survives (calibration signal per the status machine); the
    review surface filters to 'deferred' so it stops rendering. Platform
    admin only (the guard's default for unlisted /admin/api POSTs)."""
    item = await store.update_action_item_status(item_id, "dropped")
    if item is None:
        return JSONResponse(status_code=404, content={"detail": "not found"})
    return JSONResponse(content={"id": item_id, "status": "dropped"})


@router.post("/substrate-observations/dismiss-all")
async def dismiss_all_substrate_observations(
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    customer_id: Optional[str] = None,
) -> JSONResponse:
    """C1: drop every currently-surfaced observation (optionally scoped
    to one customer). Same soft-hide semantics as single dismiss."""
    items = await store.list_substrate_action_items(
        customer_id=customer_id, limit=200
    )
    dropped = 0
    for item in items:
        try:
            await store.update_action_item_status(item.id, "dropped")
            dropped += 1
        except Exception:  # noqa: BLE001
            continue
    return JSONResponse(content={"dropped": dropped})


@router.get("/substrate-observations/grouped")
async def group_substrate_observations_endpoint(
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    customer_id: Optional[str] = None,
    since: Optional[str] = None,
    limit: int = 200,
) -> JSONResponse:
    """Return deferred substrate observations GROUPED by subsystem (CU-30).

    Same query parameters and defensive parsing as the flat surface;
    `limit` bounds the underlying rows the groups roll up (default 200).

    Returns:
      {
        "total_groups": N,
        "groups": [
          {"subsystem": ..., "count": ..., "severities": {...},
           "latest_at": ..., "latest_complaint": ..., "item_ids": [...]},
          ...
        ]
      }
    ordered most-frequent-first (ties newest-first).
    """
    # Tenant pin (Accounts Phase A): the guard middleware force-scopes
    # tenant principals — the pin overrides any caller-supplied
    # customer_id. Platform admins arrive unpinned (cross-tenant view).
    _pin = getattr(request.state, "tenant_pin", None)
    if _pin:
        customer_id = _pin

    if limit < 1:
        limit = 1
    elif limit > 200:
        limit = 200

    parsed_since: Optional[datetime] = None
    if since is not None:
        try:
            parsed_since = datetime.fromisoformat(since)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={
                    "error": (
                        f"Invalid `since` value {since!r}; "
                        "expected ISO 8601 datetime."
                    ),
                },
            )

    groups = await group_substrate_observations(
        store=store,
        customer_id=customer_id,
        since=parsed_since,
        limit=limit,
    )

    return JSONResponse(content={
        "total_groups": len(groups),
        "groups": [g.model_dump(mode="json") for g in groups],
    })
