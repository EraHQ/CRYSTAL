"""DSL config admin endpoints — /api/dsl_configs/*.

Per-tenant CRUD over the named DSL config sources. Each customer has
zero or more named sources stored in the `dsl_configs` table; on
first access by the ConceptRouter, the DslConfigStore concatenates a
tenant's sources in name-order and compiles them into a single
RuntimeEnv. These endpoints let the inspector edit those sources.

Endpoints (matching v1 verbatim):
  GET    /api/dsl_configs                list named sources for tenant
  PUT    /api/dsl_configs/{name}         create or replace a source
  DELETE /api/dsl_configs/{name}         remove a source

PUT performs synchronous DSL compile validation before persistence
so invalid sources never land in the DB — better UX than persisting
broken sources and surfacing compile errors at routing time.

Per Phase 6.5 P1.4. Uses `app.state.dsl_config_store` wired in
lifespan.
"""
from __future__ import annotations

from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from ..decomposer.config_store import DslConfigStoreError
from ..dsl import run as compile_dsl
from ..infrastructure import MetadataStore
from ..infrastructure.metadata_store import get_metadata_store
from ..ingress.auth import require_customer
from ..models import Customer

logger = structlog.get_logger(__name__)

router = APIRouter()


def _validate_dsl_source(tenant_id: str, source_text: str) -> None:
    """Compile-check a DSL source before persisting it.

    Raises HTTPException(400) on parse / compile errors so the
    caller (operator typing in the inspector) sees a 400 with the
    exact error rather than a 500 at routing time.
    """
    try:
        compile_dsl(source_text, tenant_id=tenant_id)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"DSL source failed to compile: {e}",
        )


@router.get("/api/dsl_configs")
async def list_dsl_configs(
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """List named DSL config sources for this customer.

    Returns the persisted (name, source) rows from the DB, plus the
    set of compiled config names visible in the in-memory store's
    cached env (for operator debugging — should match unless the
    cache is stale).
    """
    rows = await store.list_dsl_configs_for_customer(customer.id)
    return JSONResponse(content={
        "configs": [
            {"name": name, "source_text": src}
            for name, src in rows
        ],
        "count": len(rows),
    })


@router.put("/api/dsl_configs/{name}")
async def upsert_dsl_config(
    name: str,
    request: Request,
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Create or replace a DSL config source.

    Body: `{"source_text": "..."}`. Compile-validates before persisting
    so the inspector can show the user the exact error before any DB
    write happens. After persist, the in-memory store is invalidated
    so the next ConceptRouter call sees the updated env.
    """
    if not name or "/" in name or "\\" in name:
        raise HTTPException(
            status_code=400,
            detail="config name must be non-empty and contain no slashes",
        )

    body = await request.json()
    source_text = body.get("source_text", "")
    if not isinstance(source_text, str):
        raise HTTPException(
            status_code=400,
            detail="source_text must be a string",
        )

    # Compile against the combined sources for this tenant. For the
    # validation step we compile the candidate source IN ISOLATION
    # (matches v1) — full-program compile happens at next routing
    # call when the store concatenates with other sources.
    _validate_dsl_source(customer.id, source_text)

    # Persist via the in-memory store's helper, which also invalidates
    # and reloads the cached env in one go.
    dsl_store = request.app.state.dsl_config_store
    try:
        await dsl_store.upsert_source_and_reload(
            tenant_id=customer.id,
            name=name,
            source_text=source_text,
        )
    except DslConfigStoreError as e:
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(
        "dsl_config.upserted",
        customer_id=customer.id,
        name=name,
        source_chars=len(source_text),
    )
    return JSONResponse(content={
        "name": name,
        "updated": True,
    })


@router.delete("/api/dsl_configs/{name}")
async def delete_dsl_config(
    name: str,
    request: Request,
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Remove a named DSL config source.

    After deletion, the in-memory store is invalidated so the next
    ConceptRouter call recompiles the remaining sources.
    """
    deleted = await store.delete_dsl_config(customer.id, name)
    if not deleted:
        raise HTTPException(status_code=404, detail="Config not found")

    # Invalidate the in-memory env so next access recompiles
    dsl_store = request.app.state.dsl_config_store
    dsl_store.invalidate(customer.id)

    logger.info("dsl_config.deleted", customer_id=customer.id, name=name)
    return JSONResponse(content={
        "name": name,
        "deleted": True,
    })
