"""OpenAI compatibility endpoints — /v1/models and /v1/completions.

These mirror v1's OpenAI-compat surface so customers using the
OpenAI Python SDK can call:
  client.models.list()
  client.completions.create(...)

`/v1/completions` is legacy text completions; v1 has it as a 501
stub (preferring /v1/chat/completions). We preserve that stub here
so the SDK gets a structured error rather than a 404.

`/v1/models` returns the single model configured on the customer's
ModelRoutingConfig. Multi-model permission lists are a future
enhancement.
"""
from __future__ import annotations

from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from ..ingress.auth import require_customer
from ..models import Customer
from . import not_implemented

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.get("/v1/models")
async def list_models(
    customer: Annotated[Customer, Depends(require_customer)],
) -> dict[str, Any]:
    """OpenAI-compatible model listing. Returns the customer's
    configured model. Matches v1 verbatim.

    A future version will enumerate all models the customer is
    permitted to use (multi-model routing).
    """
    cfg = customer.model_routing_config
    return {
        "object": "list",
        "data": [
            {
                "id": cfg.model_id,
                "object": "model",
                "owned_by": cfg.provider,
            }
        ],
    }


@router.post("/v1/completions")
async def completions() -> JSONResponse:
    """Legacy text completions — intentionally 501.

    Matches v1: prefer `/v1/chat/completions` for all new work.
    The stub preserves the URL so OpenAI-SDK callers using
    `client.completions.create(...)` get a structured error.
    """
    return not_implemented(
        feature="completions",
        doc_ref="Legacy text completions — prefer /v1/chat/completions",
    )
