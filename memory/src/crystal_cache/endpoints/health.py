"""Health probes — /health (liveness) and /health/deep (readiness).

Verbatim port from v1's app.py.
"""
from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..config import settings
from ..infrastructure import MetadataStore
from ..llm import get_llm_client

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, Any]:
    """Liveness probe.

    L6 (2026-09-05): carries the self-curation readiness signal — the
    highest-value line in the product per the functionality audit. A keyless
    deployment boots and serves traffic, but every self-curating feature
    (contradiction/dedup/gap scans, gap filling, multi-segment keying) skips
    silently without an internal model key. This field makes that state
    visible where every operator already looks.
    """
    return {
        "status": "ok",
        "environment": settings.environment,
        "version": "0.2.0",
        "self_curation": (
            "active" if get_llm_client().is_ready() else "idle_no_model_key"
        ),
    }


@router.get("/health/deep")
async def health_deep(request: Request) -> JSONResponse:
    """Readiness probe — forces a DB connection acquisition."""
    store: MetadataStore = request.app.state.metadata_store
    try:
        async with store.session() as session:
            await session.connection()
        return JSONResponse({"status": "ok", "database": "reachable"})
    except Exception as e:
        logger.error("health_deep.db_error", error=str(e))
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "database": "unreachable", "error": str(e)},
        )
