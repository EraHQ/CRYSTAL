"""Inspector-tier 501 stubs from v1.

These endpoints exist as 501 stubs in v1 to give the inspector UI's
API client structured responses (not 404s) until their backing
modules are built. Preserved here verbatim per CLAUDE.md R4: the
"stub-and-fill" pattern keeps v1's URL surface stable so the
inspector code doesn't need to change.

Endpoints (all 501 stubs):
  GET   /api/dashboard/overview
  GET   /api/dashboard/crystals
  GET   /api/verify/queue
  POST  /api/verify/approve/{task_id}
  POST  /api/verify/reject/{task_id}
  POST  /api/documents          (admin variant; distinct from /v1/documents)
  GET   /api/documents          (admin variant)
  GET   /api/settings
  POST  /api/settings
  GET   /api/crystals/{crystal_id}/history

The doc_refs match v1's stubs so a future implementer can find the
intended source. None of these block production traffic; they just
populate the OpenAPI surface from day one.
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from . import not_implemented

router = APIRouter()


# --- Dashboard ---

@router.get("/api/dashboard/overview")
async def dashboard_overview() -> JSONResponse:
    return not_implemented(
        feature="dashboard_overview",
        doc_ref="SYSTEM_DIAGRAM.md §4 + BUILD_PROPOSAL.md §7 customer/dashboard",
    )


@router.get("/api/dashboard/crystals")
async def dashboard_crystals() -> JSONResponse:
    return not_implemented(
        feature="dashboard_crystals",
        doc_ref="SYSTEM_DIAGRAM.md §4",
    )


# --- Verification queue (admin) ---

@router.get("/api/verify/queue")
async def verify_queue() -> JSONResponse:
    return not_implemented(
        feature="verify_queue",
        doc_ref="BUILD_PROPOSAL.md §7 customer/verification_ui",
    )


@router.post("/api/verify/approve/{task_id}")
async def verify_approve(task_id: str) -> JSONResponse:
    return not_implemented(
        feature="verify_approve",
        doc_ref="BUILD_PROPOSAL.md §7 customer/verification_ui",
    )


@router.post("/api/verify/reject/{task_id}")
async def verify_reject(task_id: str) -> JSONResponse:
    return not_implemented(
        feature="verify_reject",
        doc_ref="BUILD_PROPOSAL.md §7 customer/verification_ui",
    )


# --- Documents (admin variant) ---

@router.post("/api/documents")
async def documents_upload() -> JSONResponse:
    return not_implemented(
        feature="documents_upload",
        doc_ref="SYSTEM_DIAGRAM.md §4 + BUILD_PROPOSAL.md §7 customer/document_upload",
    )


@router.get("/api/documents")
async def documents_list() -> JSONResponse:
    return not_implemented(
        feature="documents_list",
        doc_ref="BUILD_PROPOSAL.md §7 customer/document_upload",
    )


# --- Settings ---

@router.get("/api/settings")
async def get_settings_endpoint() -> JSONResponse:
    return not_implemented(
        feature="get_settings",
        doc_ref="BUILD_PROPOSAL.md §7 customer/admin_settings",
    )


@router.post("/api/settings")
async def update_settings() -> JSONResponse:
    return not_implemented(
        feature="update_settings",
        doc_ref="BUILD_PROPOSAL.md §7 customer/admin_settings",
    )


# --- Crystal history ---

@router.get("/api/crystals/{crystal_id}/history")
async def crystal_history(crystal_id: str) -> JSONResponse:
    return not_implemented(
        feature="crystal_history",
        doc_ref="BUILD_PROPOSAL.md §4",
    )
