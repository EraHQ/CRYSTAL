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

from typing import Annotated, Any, Optional

import asyncio

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from ..infrastructure import MetadataStore
from ..infrastructure.metadata_store import get_metadata_store
from ..ingress.auth import (
    _bearer_token_from_header,
    _looks_like_firebase_jwt,
    require_active_subscription,
    require_customer_or_console,
    require_write_capacity,
    resolve_principal_or_console,
)
from ..ingress.schema import (
    CrystallizeResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentUploadRequest,
)
from ..ingestion.file_extract import (
    DocumentTooLarge,
    UnsupportedFileType,
    extract_text_from_file,
    label_from_filename,
    sanitize_label,
    valid_upload_crystal_type,
)
from ..models import Customer, Operator


async def read_upload_bounded(file: UploadFile, max_bytes: int) -> bytes:
    """Lockdown PR-1 (B2-5): read a multipart file in 1 MiB chunks and
    stop at `max_bytes` with 413. `await file.read()` used to pull the
    whole body (Cloud Run's 32 MiB edge was the only bound) and every
    byte of it was then extracted and paid for."""
    buf = bytearray()
    while True:
        chunk = await file.read(1 << 20)
        if not chunk:
            break
        buf += chunk
        if len(buf) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"The file is larger than {max_bytes // 2**20} MB. "
                    "Split it or upload a smaller file."
                ),
            )
    return bytes(buf)


def extraction_error_to_http(e: Exception, *, customer_id: str) -> HTTPException:
    """Map an extraction failure to the response the client gets.
    DocumentTooLarge -> 413 and UnsupportedFileType -> 415 carry their
    own user-facing text; anything else gets a fixed message plus a
    reference (S4, 2026-09-30) so library internals never leak."""
    if isinstance(e, DocumentTooLarge):
        return HTTPException(status_code=413, detail=e.public_message)
    if isinstance(e, UnsupportedFileType):
        return HTTPException(status_code=415, detail=e.public_message)
    from ..hygiene import safe_error
    ref, message = safe_error(
        "documents.extract_failed", e,
        user_message="We couldn't read that file. Check the format and try again.",
        customer_id=customer_id,
    )
    return HTTPException(status_code=400, detail=message)


def _require_upload_crystal_type(value: Optional[str]) -> None:
    """Lockdown PR-5 (Q45): an upload's bucket is a customer bucket,
    `customer:` plus lowercase letters, digits, `_ . -`, at most 64.
    general:*, assumption and reflection carry semantics of their own."""
    if not valid_upload_crystal_type(value):
        raise HTTPException(
            status_code=422,
            detail="crystal_type must match customer:<name> (lowercase letters, digits, _ . -, at most 64).",
        )


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


# ---------------------------------------------------------------------------
# Lockdown PR-3 (Q51=A, B2-1/B2-3, 2026-10-09): what a review edit may say.
#
# A review row's chunks carry fields the SERVER stamped at extraction
# that later drive provenance (source_path via locator/doc_type, the
# C4 fragment carve via sheet/window), dedup/replace (the same), and
# the injection verdict (injection_hits). A request body may delete a
# chunk or describe it; it may not rewrite where a chunk came from,
# what it said, or whether it was screened. Items are the curator's to
# edit, except crystal_id, which the write leg stamps.
# ---------------------------------------------------------------------------
CHUNK_CLIENT_FIELDS: frozenset[str] = frozenset({"description"})
# PR-5 (Q45 / B2-4): `injection_hits` on an item is the server's screen
# finding, re-attached from the stored item with the same `index`.
ITEM_SERVER_FIELDS: frozenset[str] = frozenset({"crystal_id", "injection_hits"})


def merge_review_chunks(
    stored: Optional[list[dict[str, Any]]], edited: Any,
) -> list[dict[str, Any]]:
    """The chunk list to persist for a review edit: each edited chunk is
    the STORED chunk with that index, with only CHUNK_CLIENT_FIELDS taken
    from the edit. Chunks the row does not have (no index, unknown index,
    not a dict) are dropped. Order and deletions follow the edit."""
    by_index: dict[int, dict[str, Any]] = {}
    for c in stored or []:
        if isinstance(c, dict) and isinstance(c.get("index"), int):
            by_index[c["index"]] = c
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for e in edited if isinstance(edited, list) else []:
        if not isinstance(e, dict):
            continue
        idx = e.get("index")
        if not isinstance(idx, int) or idx not in by_index or idx in seen:
            continue
        seen.add(idx)
        merged = dict(by_index[idx])
        for f in CHUNK_CLIENT_FIELDS:
            if f in e:
                v = e[f]
                merged[f] = v if (v is None or isinstance(v, str)) else str(v)
        out.append(merged)
    return out


def merge_review_items(
    stored: Optional[list[dict[str, Any]]], edited: Any,
) -> list[dict[str, Any]]:
    """The item list to persist for a review edit: the edit's dicts with
    ITEM_SERVER_FIELDS stripped, then the server's screen findings
    re-attached from the stored item with the same `index` (PR-5). An
    item with no stored counterpart carries no findings: the write
    rescans it and quarantines on a hit. Non-dict entries are dropped."""
    by_index: dict[int, dict[str, Any]] = {}
    for it in stored or []:
        if isinstance(it, dict) and isinstance(it.get("index"), int):
            by_index[it["index"]] = it
    out: list[dict[str, Any]] = []
    for e in edited if isinstance(edited, list) else []:
        if not isinstance(e, dict):
            continue
        item = {k: v for k, v in e.items() if k not in ITEM_SERVER_FIELDS}
        src = by_index.get(item.get("index")) if isinstance(item.get("index"), int) else None
        if src is not None and "injection_hits" in src:
            item["injection_hits"] = src["injection_hits"]
        out.append(item)
    return out


def approved_via_for(request: Request) -> str:
    """Q50=A: the credential kind behind an approve. 'console' only for a
    signed-in console session (a Firebase JWT bearer); every key is
    'key'. The write leg treats only 'console' as a curator verdict."""
    bearer = _bearer_token_from_header(
        request.headers.get("authorization") or request.headers.get("Authorization")
    )
    return "console" if (bearer and _looks_like_firebase_jwt(bearer)) else "key"


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
    require_active_subscription(customer)  # L2-S2: 402 on expired trial
    _require_upload_crystal_type(crystal_type)
    await require_write_capacity(customer, store, spend=True)  # fact cap + daily door (Q31=A)
    doc_scope, doc_owner = _resolve_source_scope(scope, operator)
    from ..config import get_settings
    _settings = get_settings()
    max_chars = int(_settings.document_max_chars)
    contents = await read_upload_bounded(file, int(_settings.upload_max_bytes))
    try:
        # RC-14 (2026-10-06): PDF/docx extraction is CPU-bound and ran on
        # the event loop, stalling every other request for the duration.
        # PR-1: max_chars + 1 lets the PDF path stop parsing early while
        # the cap check below can still tell "over" from "exactly at".
        text = await asyncio.to_thread(
            extract_text_from_file,
            contents, file.filename or "",
            mime=file.content_type,
            max_chars=max_chars + 1,
        )
    except Exception as e:
        raise extraction_error_to_http(e, customer_id=customer.id)

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="File contains no extractable text",
        )
    if len(text) > max_chars:
        # Lockdown PR-1 (B2-5/B5-1): the JSON route has always capped at
        # 500,000 characters; the file route capped nothing, so one upload
        # could run thousands of paid extraction windows. Refused, never
        # silently truncated.
        raise HTTPException(
            status_code=413,
            detail=(
                f"Extracted text exceeds {max_chars:,} characters; "
                "split the file."
            ),
        )

    # Lockdown PR-5 (Q45): the filename is dispatch and a default label,
    # never a path; every label goes through the one sanitiser.
    doc = await store.create_document_upload(
        customer_id=customer.id,
        label=sanitize_label(label or label_from_filename(file.filename), text=text),
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
    require_active_subscription(customer)  # L2-S2: 402 on expired trial
    await require_write_capacity(customer, store, spend=True)  # fact cap + daily door (Q31=A)
    doc_scope, doc_owner = _resolve_source_scope(body.scope, operator)
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    doc = await store.create_document_upload(
        customer_id=customer.id,
        label=sanitize_label(body.label, text=body.text),  # Q45
        text=body.text,
        scope=doc_scope,
        owner_operator_id=doc_owner,
        crystal_type=body.crystal_type or "customer:legacy",
    )
    if body.auto_crystallize:
        # RC-17 (2026-10-08): the flag was accepted by the schema and
        # dropped here, so onboarding's upload sat pending forever. Stored
        # now; extraction marks the row approved and the worker writes it.
        await store.set_document_auto_approve(doc.id, True)

    logger.info(
        "document.created",
        customer_id=customer.id,
        document_id=doc.id,
        char_count=doc.char_count,
        auto_crystallize=bool(body.auto_crystallize),
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

    # Resolve the document's crystal set. Lockdown PR-3 (B2-1,
    # 2026-10-09): the write leg records the ids it produced on the row
    # (crystal_ids, server-only); that list is the set. Rows written
    # before the column exists fall back to provenance resolution (item
    # crystal_ids + chunk source paths) — and EITHER way every candidate
    # must belong to this document's owner (or be unowned, with the same
    # group) before it is flipped, so no id a client could have placed
    # on the row, and no colleague's same-named source, ever changes
    # scope through this call.
    recorded = getattr(doc, "crystal_ids", None)
    if recorded is not None:
        candidate_ids = sorted({c for c in recorded if isinstance(c, str) and c})
    else:
        item_ids = {
            item.get("crystal_id")
            for item in (doc.extracted_items or [])
            if isinstance(item, dict) and item.get("crystal_id")
        }
        chunk_paths = sorted({
            (chunk.get("source_path") or doc.label)
            for chunk in (doc.content_chunks or [])
            if isinstance(chunk, dict)
        })
        chunk_ids = set(await store.list_crystal_ids_for_source_paths(
            customer.id, chunk_paths,
        ))
        candidate_ids = sorted(item_ids | chunk_ids)

    flipped = []
    skipped_foreign = 0
    for cid in candidate_ids:
        crystal = await store.get_crystal(cid)
        if crystal is None or crystal.customer_id != customer.id:
            continue
        same_group = (crystal.group_team_id or crystal.customer_id) == customer.id
        owned_by_doc = (
            crystal.owner_operator_id == doc.owner_operator_id
            if doc.owner_operator_id is not None
            else crystal.owner_operator_id is None
        )
        if not (same_group and owned_by_doc):
            skipped_foreign += 1
            continue
        if await store.set_crystal_scope(cid, customer.id, scope):
            flipped.append(cid)
    await store.set_document_scope(document_id, customer.id, scope)
    if skipped_foreign:
        logger.warning(
            "document.scope_change_skipped_foreign",
            customer_id=customer.id, document_id=document_id,
            skipped=skipped_foreign,
        )

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


@router.get("/v1/documents/{document_id}", response_model=DocumentResponse)
async def sdk_get_document(
    document_id: str,
    customer: Annotated[Customer, Depends(require_customer_or_console)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """One document's status envelope (L7a gate 5, 2026-08-29). This is
    the poll target for CC_INGEST_MODE=worker, where /crystallize and
    /approve return 202 and the worker finishes the leg later; it is the
    same shape as one entry of the list. Tenancy-checked."""
    doc = await store.get_document_upload(document_id, customer.id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return JSONResponse(content=_doc_to_response(doc))


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
    """Update extracted items / content chunks / confirmed type during review.

    Lockdown PR-3 (B2-1/B2-3): only while the row is in review (a
    crystallized or queued row's items are provenance, not a draft), and
    the edit is MERGED over the stored row — see merge_review_chunks /
    merge_review_items for what a body may change.
    """
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="Body must be a JSON object")
    # Verify ownership before update
    doc = await store.get_document_upload(document_id, customer.id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.status != "review":
        raise HTTPException(
            status_code=409,
            detail="This document is not in review; its items can no longer be edited.",
        )

    items = (
        merge_review_items(doc.extracted_items, body["extracted_items"])
        if "extracted_items" in body else None
    )
    chunks = (
        merge_review_chunks(doc.content_chunks, body["content_chunks"])
        if "content_chunks" in body else None
    )
    confirmed_type = body.get("confirmed_type")
    if confirmed_type is not None and not isinstance(confirmed_type, str):
        raise HTTPException(status_code=422, detail="confirmed_type must be a string")
    await store.update_document_review_edits(
        document_id=document_id,
        customer_id=customer.id,
        extracted_items=items,
        content_chunks=chunks,
        confirmed_type=confirmed_type,
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

    CC_INGEST_MODE=inline (default): atomic step — save edits AND
    transition to crystallizing in one call — then run the write leg
    here and return the result.
    CC_INGEST_MODE=worker (L7a gate 5): atomic step — save edits AND
    mark 'approved' — and return 202; the crystallization worker claims
    the row and runs the same write leg. Poll GET /v1/documents/{id}.
    """
    # RC-09 (2026-10-04): approve runs extraction and writes crystals;
    # fact cap + daily door, whichever process does the work.
    await require_write_capacity(customer, store, spend=True)
    doc = await store.get_document_upload(document_id, customer.id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    # RC-16 / B6 (2026-10-08), regression closed by Lockdown PR-4
    # (2026-10-09): the PRESENCE of a key is the switch, never its
    # truthiness. `"items" in body` means the caller chose the items,
    # and an explicit `[]` is "no items" whatever `include_chunks` says;
    # an absent key means "as extracted". `include_chunks: false` drops
    # every chunk; otherwise a present `content_chunks` is the chosen
    # set and an absent one means "as extracted". The approve is refused
    # only when the caller's selection nets to nothing.
    # Lockdown PR-3 (B2-1/B2-3): the body's lists are merged over the
    # stored row the same way a review edit is — a client chooses WHICH
    # chunks and items to keep and may describe a chunk; the server's
    # provenance and screening fields on each chunk are kept from the
    # row, and crystal_id is never taken from a body.
    explicit = "items" in body or "content_chunks" in body or "include_chunks" in body
    items = (
        merge_review_items(doc.extracted_items, body["items"]) if "items" in body
        else (doc.extracted_items or [])
    )
    if "include_chunks" in body and not body["include_chunks"]:
        content_chunks: list[dict[str, Any]] = []
    elif "content_chunks" in body:
        content_chunks = merge_review_chunks(doc.content_chunks, body["content_chunks"])
    else:
        content_chunks = list(doc.content_chunks or [])
    if explicit and not items and not content_chunks:
        raise HTTPException(
            status_code=400,
            detail="Nothing is selected. Choose the items to keep, then approve.",
        )
    # Q50=A: record which credential approved; only a console session is
    # a curator verdict for the write leg.
    approved_via = approved_via_for(request)

    from ..config import get_settings
    if get_settings().ingest_mode == "worker":
        won = await store.mark_document_approved(
            document_id=document_id, items=items, content_chunks=content_chunks,
            approved_via=approved_via,
        )
        if not won:
            # RC-04 / E-S9: a second approve (double click, retry) must
            # not queue a second encode.
            raise HTTPException(
                status_code=409,
                detail="This document was already approved or is being processed.",
            )
        logger.info("document.approved_queued", customer_id=customer.id, document_id=document_id)
        return JSONResponse(status_code=202, content={
            "document_id": document_id,
            "status": "approved",
            "queued": True,
        })

    # Atomic transition: save edits + flip status to crystallizing
    won = await store.save_approval_edits_and_mark_crystallizing(
        document_id=document_id,
        items=items,
        content_chunks=content_chunks,
        approved_via=approved_via,
    )
    if not won:
        # RC-04 / E-S9: the other approve won the transition; do not
        # encode a second time.
        raise HTTPException(
            status_code=409,
            detail="This document was already approved or is being processed.",
        )

    # Run the write leg here — the same workflow the worker runs.
    from ..workers.crystallization import write_approved_document
    try:
        result = await write_approved_document(
            store=store,
            encoder=request.app.state.prompt_encoder,
            vector_store=request.app.state.vector_store,
            vector_index=getattr(request.app.state, "vector_index", None),
            fact_vector_store=getattr(request.app.state, "fact_vector_store", None),
            document_id=document_id,
            customer_id=customer.id,
        )
    except Exception as e:
        from ..hygiene import safe_error
        ref, message = safe_error(
            "documents.crystallize_failed", e,
            user_message="Crystallization failed.",
            customer_id=customer.id, document_id=document_id,
        )
        raise HTTPException(status_code=500, detail=message)
    return JSONResponse(content=result)


@router.post("/v1/documents/{document_id}/crystallize", response_model=CrystallizeResponse)
async def sdk_crystallize_document(
    document_id: str,
    request: Request,
    customer: Annotated[Customer, Depends(require_customer_or_console)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Manually trigger crystallization for a pending document.

    CC_INGEST_MODE=inline (default): goes straight from pending →
    extract → review here, without needing the worker to pick it up.
    CC_INGEST_MODE=worker (L7a gate 5): returns 202 — the row is already
    'pending', which the worker claims on its next poll. Poll
    GET /v1/documents/{id} for 'review'.
    """
    # RC-09: crystallize runs extraction; fact cap + daily door.
    await require_write_capacity(customer, store, spend=True)
    doc = await store.get_document_upload(document_id, customer.id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    from ..config import get_settings
    if get_settings().ingest_mode == "worker":
        return JSONResponse(status_code=202, content={
            "document_id": document_id,
            "status": doc.status,
            "queued": doc.status == "pending",
            "items_extracted": doc.items_extracted,
            "error_message": doc.error_message,
        })

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

    Returns counts of processed, succeeded, failed. Under
    CC_INGEST_MODE=worker (L7a gate 5) the rows are already 'pending' —
    the worker claims them; this returns 202 with the count queued.
    """
    # RC-09: crystallize-all runs extraction per document; fact cap +
    # daily door, once at the door (Q17=A: admitted work finishes).
    await require_write_capacity(customer, store, spend=True)
    pending = await store.list_document_uploads(
        customer_id=customer.id,
        status="pending",
    )

    from ..config import get_settings
    if get_settings().ingest_mode == "worker":
        return JSONResponse(status_code=202, content={
            "processed": len(pending), "queued": len(pending),
            "succeeded": 0, "failed": 0,
        })

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
