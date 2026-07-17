"""Document endpoints — /v1/documents/* 

Document upload, listing, review, approval, manual crystallization,
and deletion. Refactored to use Phase 5 MetadataStore methods.

Endpoints:
  POST   /v1/documents/upload          multipart file upload
  POST   /v1/documents                 JSON body upload
  GET    /v1/documents                 list this customer's docs
  GET    /v1/documents/{id}/review     get extracted items for review
  PUT    /v1/documents/{id}/review     update extracted items pre-approval
  POST   /v1/documents/{id}/approve    approve + crystallize
  POST   /v1/documents/{id}/crystallize  manual crystallize (no review)
  POST   /v1/documents/crystallize-all   crystallize all pending for customer
  DELETE /v1/documents/{id}            hard delete

Phase 6 note: the approve + crystallize paths invoke
`DocumentPipeline.approve_and_crystallize`, which is ported in
ingestion/. The manual crystallize path delegates to
`workers.crystallization.crystallize_document` (Phase 6 Wave A).

Status strings (per Phase 6.5 P0.1) match v1 verbatim:
  pending → crystallizing → review → crystallized
  pending → crystallizing → error
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Optional

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from ..infrastructure import MetadataStore
from ..infrastructure.metadata_store import get_metadata_store
from ..ingress.auth import (
    require_customer_or_console,
    resolve_principal_or_console,
)
from ..ingress.schema import (
    CrystallizeResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentUploadRequest,
)
from ..ingestion.file_extract import extract_text_from_file
from ..models import Customer, Operator


def _resolve_source_scope(
    scope: Optional[str], operator: Optional[Operator],
) -> tuple[str, Optional[str]]:
    """(scope, owner_operator_id) for a new document source (P2, ratified
    2026-07-02). Explicit scope wins; else the deployment default
    (CC_DEFAULT_INGEST_SCOPE — personal). Viewers are read-only. P1 makes
    the operator always present (team keys act as the Default Admin), so
    the owner is always well-defined."""
    if operator is not None and operator.role == "viewer":
        raise HTTPException(
            status_code=403,
            detail="Viewers are read-only and cannot upload documents.",
        )
    if scope is not None and scope not in ("personal", "team"):
        raise HTTPException(
            status_code=422, detail="scope must be 'personal' or 'team'",
        )
    from ..config import get_settings

    resolved = scope or get_settings().default_ingest_scope
    return resolved, (operator.id if operator is not None else None)

logger = structlog.get_logger(__name__)

router = APIRouter()


def _doc_to_response(doc) -> dict[str, Any]:
    """Serialize DocumentUpload to v1 response shape."""
    return {
        "id": doc.id,
        "customer_id": doc.customer_id,
        "label": doc.label,
        "status": doc.status,
        "crystal_type": doc.crystal_type,
        "char_count": doc.char_count,
        "crystals_written": doc.crystals_written,
        "items_extracted": doc.items_extracted,
        "content_chunks_count": len(doc.content_chunks or []),
        "error_message": doc.error_message,
        "detected_type": doc.detected_type,
        "confirmed_type": doc.confirmed_type,
        "source_file_id": doc.source_file_id,
        "source_modified_at": doc.source_modified_at.isoformat() if doc.source_modified_at else None,
        "source_connection_id": doc.source_connection_id,
        "created_at": doc.created_at.isoformat(),
        "crystallized_at": doc.crystallized_at.isoformat() if doc.crystallized_at else None,
    }


@router.post("/v1/documents/upload", response_model=DocumentResponse)
async def sdk_upload_document_file(
    request: Request,
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(resolve_principal_or_console)
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    file: UploadFile = File(...),
    label: Optional[str] = Form(default=None),
    crystal_type: str = Form(default="customer:legacy"),
    scope: Optional[str] = Form(default=None),
) -> JSONResponse:
    """Upload a file (PDF / DOCX / TXT) for crystallization.

    Extracts text via `extract_text_from_file`, creates a pending
    `DocumentUpload`, and returns the response. The crystallization
    worker picks it up on next poll.

    P2 scope-on-sources (ratified 2026-07-02): the document is a SOURCE
    — it carries scope + owner, and every crystal born from it inherits
    them. `scope` (personal|team) defaults to the deployment knob.
    """
    customer, operator = principal
    doc_scope, doc_owner = _resolve_source_scope(scope, operator)
    contents = await file.read()
    try:
        text = extract_text_from_file(contents, file.filename or "")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to extract text: {e}",
        )

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="File contains no extractable text",
        )

    doc = await store.create_document_upload(
        customer_id=customer.id,
        label=label or file.filename or "Untitled",
        text=text,
        crystal_type=crystal_type,
        scope=doc_scope,
        owner_operator_id=doc_owner,
    )

    logger.info(
        "document.uploaded",
        customer_id=customer.id,
        document_id=doc.id,
        filename=file.filename,
        char_count=doc.char_count,
    )
    return JSONResponse(content=_doc_to_response(doc))


@router.post("/v1/documents", response_model=DocumentResponse)
async def sdk_upload_document(
    body: DocumentUploadRequest,
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(resolve_principal_or_console)
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Create a document upload from a JSON body containing the text directly.

    P2 scope-on-sources: see the file-upload route.
    """
    customer, operator = principal
    doc_scope, doc_owner = _resolve_source_scope(body.scope, operator)
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    doc = await store.create_document_upload(
        customer_id=customer.id,
        label=body.label or "Untitled",
        text=body.text,
        scope=doc_scope,
        owner_operator_id=doc_owner,
        crystal_type=body.crystal_type or "customer:legacy",
    )

    logger.info(
        "document.created",
        customer_id=customer.id,
        document_id=doc.id,
        char_count=doc.char_count,
    )
    return JSONResponse(content=_doc_to_response(doc))


@router.post("/v1/documents/{document_id}/scope")
async def sdk_set_document_scope(
    document_id: str,
    request: Request,
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(resolve_principal_or_console)
    ],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """SHARE-SOURCE (P4, ratified 2026-07-02): flip the scope of a document
    AND everything derived from it in one call — {"scope": "team"} shares,
    {"scope": "personal"} unshares. Resolution is provenance-based:
    content-chunk crystals via their source_path stamps, knowledge-item
    crystals via the crystal ids recorded on the row's extracted_items at
    approve. The document row is restamped too, so future crystallization
    inherits the new scope. Authorization: the document's owner or a team
    admin. Human/API surface only — no agent-facing share tool.

    Note: a knowledge crystal can hold same-scope facts from other
    documents (team-mode cross-bonding); flipping it shares those facts
    too. That's the crystal-grain reality the keystone makes sound —
    everything in the crystal is same-scope by construction.
    """
    customer, operator = principal
    body = await request.json() if request.headers.get(
        "content-type", ""
    ).startswith("application/json") else {}
    scope = (body or {}).get("scope")
    if scope not in ("personal", "team"):
        raise HTTPException(
            status_code=422, detail="scope must be 'personal' or 'team'",
        )
    doc = await store.get_document_upload(document_id, customer.id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    is_owner = (
        operator is not None
        and doc.owner_operator_id is not None
        and operator.id == doc.owner_operator_id
    )
    is_admin = operator is not None and operator.role == "admin"
    if not (is_owner or is_admin):
        raise HTTPException(
            status_code=403,
            detail="Only the document's owner or a team admin may change its scope.",
        )

    # Resolve the document's crystal set from provenance.
    item_ids = {
        item.get("crystal_id")
        for item in (doc.extracted_items or [])
        if item.get("crystal_id")
    }
    chunk_paths = sorted({
        (chunk.get("source_path") or doc.label)
        for chunk in (doc.content_chunks or [])
    })
    chunk_ids = set(await store.list_crystal_ids_for_source_paths(
        customer.id, chunk_paths,
    ))
    crystal_ids = sorted(item_ids | chunk_ids)

    flipped = []
    for cid in crystal_ids:
        if await store.set_crystal_scope(cid, customer.id, scope):
            flipped.append(cid)
    await store.set_document_scope(document_id, customer.id, scope)

    logger.info(
        "document.scope_changed",
        customer_id=customer.id, document_id=document_id,
        scope=scope, crystals_flipped=len(flipped),
    )
    return JSONResponse(content={
        "document_id": document_id,
        "scope": scope,
        "crystals_flipped": len(flipped),
        "crystal_ids": flipped,
    })


@router.get("/v1/documents", response_model=DocumentListResponse)
async def sdk_list_documents(
    customer: Annotated[Customer, Depends(require_customer_or_console)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    status: Optional[str] = None,
    limit: Optional[int] = None,
) -> JSONResponse:
    """List this customer's documents, optionally filtered by status."""
    docs = await store.list_document_uploads(
        customer_id=customer.id,
        status=status,
        limit=limit,
    )
    return JSONResponse(content={
        "documents": [_doc_to_response(d) for d in docs],
        "count": len(docs),
    })


@router.get("/v1/documents/{document_id}/review")
async def sdk_get_document_review(
    document_id: str,
    customer: Annotated[Customer, Depends(require_customer_or_console)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Get the extracted items + content chunks pending review."""
    doc = await store.get_document_upload(document_id, customer.id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    # Gate D3 (2026-07-17): the comprehension PREVIEW — the review
    # surface shows what ingest will know, before it becomes facts.
    # Mechanism (imports + in-bank resolution) is computed here by the
    # same code_structure module the approve pass uses — one source of
    # truth, display-only (no approve gate on deterministic facts).
    # Judgment (chunk descriptions) is editable on the chunks
    # themselves; the envelope is type-generic so tabular/schema lanes
    # (Gates E/G, C5) can add their own keys without surface churn.
    comprehension = None
    chunks = doc.content_chunks or []
    if any(c.get("doc_type") == "code" for c in chunks):
        from ..ingestion.code_structure import (
            extract_imports,
            resolve_import_target,
        )
        full_text = "\n\n".join((c.get("text") or "") for c in chunks)
        imports = extract_imports(full_text)
        if imports:
            candidates = [
                c for c in await store.list_crystals_for_customer(customer.id)
                if (getattr(c, "source_uri", "") or "").startswith("repo://")
            ]
            comprehension = {"imports": [
                {
                    "module": m,
                    "resolved_path": (
                        t.source_path
                        if (t := resolve_import_target(m, "", candidates))
                        is not None else None
                    ),
                }
                for m in imports
            ]}

    return JSONResponse(content={
        "id": doc.id,
        "label": doc.label,
        "char_count": doc.char_count,
        "status": doc.status,
        "detected_type": doc.detected_type,
        "confirmed_type": doc.confirmed_type,
        "extracted_items": doc.extracted_items or [],
        "content_chunks": chunks,
        "items_extracted": doc.items_extracted,
        "comprehension": comprehension,
    })


@router.put("/v1/documents/{document_id}/review")
async def sdk_update_document_review(
    document_id: str,
    request: Request,
    customer: Annotated[Customer, Depends(require_customer_or_console)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Update extracted items / content chunks / confirmed type during review."""
    body = await request.json()
    # Verify ownership before update
    doc = await store.get_document_upload(document_id, customer.id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    await store.update_document_review_edits(
        document_id=document_id,
        customer_id=customer.id,
        extracted_items=body.get("extracted_items"),
        content_chunks=body.get("content_chunks"),
        confirmed_type=body.get("confirmed_type"),
    )
    return JSONResponse(content={"updated": True, "document_id": document_id})


@router.post("/v1/documents/{document_id}/approve", response_model=CrystallizeResponse)
async def sdk_approve_document(
    document_id: str,
    request: Request,
    customer: Annotated[Customer, Depends(require_customer_or_console)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Approve a document under review and crystallize its items.

    Body may contain `items` and `content_chunks` representing the
    final approved set (post-edit). If omitted, uses whatever is on
    the document row.

    Atomic step: save edits AND transition to crystallizing in one
    call, then run `DocumentPipeline.approve_and_crystallize` against
    the saved state.
    """
    doc = await store.get_document_upload(document_id, customer.id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    items = body.get("items") or (doc.extracted_items or [])
    content_chunks = body.get("content_chunks") or (doc.content_chunks or [])

    # Atomic transition: save edits + flip status to crystallizing
    await store.save_approval_edits_and_mark_crystallizing(
        document_id=document_id,
        items=items,
        content_chunks=content_chunks,
    )

    # Run the crystallization pipeline
    from ..ingestion.document_pipeline import DocumentPipeline
    pipeline = DocumentPipeline(
        store=store,
        encoder=request.app.state.prompt_encoder,
        vector_store=request.app.state.vector_store,
        vector_index=getattr(request.app.state, "vector_index", None),
        # Active vector index (Qdrant-aware) for invalidation; fall back to the
        # in-memory fact store. DocumentPipeline uses it only to invalidate.
        fact_vector_store=(getattr(request.app.state, "vector_index", None)
                           or getattr(request.app.state, "fact_vector_store", None)),
    )
    try:
        # Recall-gate birth attribution (2026-07-03): cognition/background-
        # worker output is written to document_uploads with
        # detected_type='inferred_knowledge' (cognition engine). Crystals
        # born from it are recall_gated until reviewed; user-uploaded docs
        # remain 'direct' (born usable) exactly as before.
        _origin = (
            "background_worker"
            if doc.detected_type == "inferred_knowledge"
            else "direct"
        )
        result = await pipeline.approve_and_crystallize(
            customer_id=customer.id,
            document_id=document_id,
            items=items,
            content_chunks=content_chunks,
            crystal_type=doc.confirmed_type or doc.crystal_type,
            scope=doc.scope,
            owner_operator_id=doc.owner_operator_id,
            origin=_origin,
        )
        await store.mark_document_crystallized(
            document_id=document_id,
            crystals_written=result.crystals_written,
            items_extracted=result.items_extracted,
            crystallized_at=datetime.now(timezone.utc),
        )
        # Share-source provenance (P4): the pipeline stamped each item dict
        # with its crystal_id — persist the mutated items so the document
        # knows its crystal set.
        await store.update_document_review_edits(
            document_id, customer.id, extracted_items=items,
        )
        logger.info(
            "document.crystallized",
            customer_id=customer.id,
            document_id=document_id,
            crystals_written=result.crystals_written,
        )
        return JSONResponse(content={
            "document_id": document_id,
            "status": "crystallized",
            "crystals_written": result.crystals_written,
            "items_extracted": result.items_extracted,
            "errors": result.errors,
        })
    except Exception as e:
        await store.mark_document_error(document_id, str(e))
        raise HTTPException(status_code=500, detail=f"Crystallization failed: {e}")


@router.post("/v1/documents/{document_id}/crystallize", response_model=CrystallizeResponse)
async def sdk_crystallize_document(
    document_id: str,
    request: Request,
    customer: Annotated[Customer, Depends(require_customer_or_console)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Manually trigger crystallization for a pending document.

    Goes straight from pending → extract → review without needing
    the worker to pick it up. Used when a customer wants immediate
    feedback after upload rather than waiting for the next poll.
    """
    doc = await store.get_document_upload(document_id, customer.id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    from ..workers.crystallization import crystallize_document
    await crystallize_document(
        store=store,
        encoder=request.app.state.prompt_encoder,
        vector_store=request.app.state.vector_store,
        document_id=document_id,
    )

    # Re-read the doc to get the post-state
    doc_after = await store.get_document_upload(document_id, customer.id)
    return JSONResponse(content={
        "document_id": document_id,
        "status": doc_after.status if doc_after else "unknown",
        "items_extracted": doc_after.items_extracted if doc_after else 0,
        "error_message": doc_after.error_message if doc_after else None,
    })


@router.post("/v1/documents/crystallize-all")
async def sdk_crystallize_all(
    request: Request,
    customer: Annotated[Customer, Depends(require_customer_or_console)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Crystallize all pending documents for this customer.

    Returns counts of processed, succeeded, failed.
    """
    pending = await store.list_document_uploads(
        customer_id=customer.id,
        status="pending",
    )

    from ..workers.crystallization import crystallize_document
    succeeded = 0
    failed = 0
    for doc in pending:
        try:
            await crystallize_document(
                store=store,
                encoder=request.app.state.prompt_encoder,
                vector_store=request.app.state.vector_store,
                document_id=doc.id,
            )
            succeeded += 1
        except Exception as e:
            failed += 1
            logger.warning(
                "documents.crystallize_all.one_failed",
                document_id=doc.id,
                error=str(e),
            )

    return JSONResponse(content={
        "processed": len(pending),
        "succeeded": succeeded,
        "failed": failed,
    })


@router.delete("/v1/documents/{document_id}")
async def sdk_delete_document(
    document_id: str,
    customer: Annotated[Customer, Depends(require_customer_or_console)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Hard delete a document row.

    Does NOT cascade to crystals — once a document has been
    crystallized its content lives in the bank independent of the
    upload row.
    """
    doc = await store.get_document_upload(document_id, customer.id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    await store.delete_document_upload(document_id, customer.id)
    return JSONResponse(content={"deleted": True, "document_id": document_id})
