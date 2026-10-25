"""Lattices: console routes (2026-10-25; Q115 A, Q116 A, Q117 A).

Tenant-pathed under /admin/api/customers/{customer_id}/lattices (the tenant
itself and a platform admin reach them; ingress/auth _TENANT_PATH_RE).
Writes are the person's own hands: on the ledger as "You", undoable for
24 hours, and a cell they edit is a correction (snapshot cells change too).

GET    .../lattices                               every lattice with its row count
POST   .../lattices {name, description, columns, key_columns}
GET    .../lattices/{id}                          one lattice with its entity lists
PATCH  .../lattices/{id} {name, description, add_columns, column_changes}
DELETE .../lattices/{id}                          the lattice and every row
GET    .../lattices/{id}/rows?search&sort&desc&limit&offset&column&value&filters
POST   .../lattices/{id}/rows {rows, source}      add or update rows by key
POST   .../lattices/{id}/rows/delete {row_ids}
GET    .../lattices/{id}/rows/{row_id}            one row with each cell's history
PATCH  .../lattices/{id}/rows/{row_id} {column, value}
GET    .../lattices/{id}/entities?column&search   browse by entity
"""
from __future__ import annotations

import json
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import StringConstraints

from ..infrastructure import MetadataStore
from ..infrastructure.metadata_store import get_metadata_store
from ..ingress.errors import build_error_envelope
from ..ingress.limits import ID64, LIMIT, OFFSET
from ..lattices import service

router = APIRouter()
# Bounded query parameters (Lockdown PR-6a: every scalar parameter has a bound).
COLKEY = Annotated[str, StringConstraints(min_length=1, max_length=40, pattern=r"^[a-z][a-z0-9_]*$")]
NEEDLE = Annotated[str, StringConstraints(max_length=200)]
CELL = Annotated[str, StringConstraints(max_length=2000)]
FILTERS = Annotated[str, StringConstraints(max_length=4000)]
_BASE = "/admin/api/customers/{customer_id}/lattices"
_CONSOLE_REASON = "Changed by the person in the console"


def _refused(e: service.LatticeRefused) -> HTTPException:
    return HTTPException(status_code=e.status, detail=build_error_envelope(
        e.message, error_type="invalid_request_error", param="lattice", code=e.code))


async def _body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    return body


def _unknown(body: dict[str, Any], allowed: set[str]) -> None:
    extra = set(body) - allowed
    if extra:
        raise _refused(service.LatticeRefused("invalid_request", f"unknown field(s) {sorted(extra)}"))


async def _customer(store: MetadataStore, customer_id: str) -> Any:
    customer = await store.get_customer_by_id(customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer


@router.get(_BASE)
async def list_lattices(customer_id: ID64, store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    await _customer(store, customer_id)
    view = await service.list_view(store, customer_id)
    customer = await store.get_customer_by_id(customer_id)
    from ..control.admission import resolve_tier
    tier = getattr(customer, "subscription_tier", None)
    cap = resolve_tier(tier).lattice_row_cap if tier else None
    return JSONResponse(content={"lattices": view, "count": len(view),
                                 "rows": {"used": await store.count_lattice_rows(customer_id), "cap": cap}})


@router.post(_BASE)
async def create_lattice(customer_id: ID64, request: Request,
                         store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    await _customer(store, customer_id)
    body = await _body(request)
    _unknown(body, {"name", "description", "columns", "key_columns"})
    try:
        out = await service.create(store, customer_id, name=body.get("name"), columns=body.get("columns"),
                                   key_columns=body.get("key_columns"), description=body.get("description", ""))
    except service.LatticeRefused as e:
        raise _refused(e)
    return JSONResponse(status_code=201, content={"lattice": service.describe(out)})


@router.get(_BASE + "/{lattice_id}")
async def get_lattice(customer_id: ID64, lattice_id: ID64,
                      store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    try:
        lat = await service.get(store, customer_id, lattice_id)
        ent = await service.entities(store, customer_id, lattice_id, limit=12)
    except service.LatticeRefused as e:
        raise _refused(e)
    return JSONResponse(content={"lattice": service.describe(lat), "entities": ent["entities"]})


@router.patch(_BASE + "/{lattice_id}")
async def update_lattice(customer_id: ID64, lattice_id: ID64, request: Request,
                         store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    body = await _body(request)
    _unknown(body, {"name", "description", "add_columns", "column_changes"})
    try:
        out = await service.update(store, customer_id, lattice_id, **{k: body.get(k) for k in
                                                                      ("name", "description", "add_columns", "column_changes")})
    except service.LatticeRefused as e:
        raise _refused(e)
    return JSONResponse(content={"lattice": service.describe(out)})


@router.delete(_BASE + "/{lattice_id}")
async def delete_lattice(customer_id: ID64, lattice_id: ID64,
                         store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    try:
        out = await service.delete(store, customer_id, lattice_id)
    except service.LatticeRefused as e:
        raise _refused(e)
    return JSONResponse(content=out)


@router.get(_BASE + "/{lattice_id}/rows")
async def lattice_rows(customer_id: ID64, lattice_id: ID64,
                       store: Annotated[MetadataStore, Depends(get_metadata_store)],
                       search: Optional[NEEDLE] = None, sort: Optional[COLKEY] = None, desc: bool = False,
                       limit: LIMIT = 50, offset: OFFSET = 0, column: Optional[COLKEY] = None,
                       value: Optional[CELL] = None, filters: Optional[FILTERS] = None) -> JSONResponse:
    parsed: Any = None
    if filters:
        try:
            parsed = json.loads(filters)
        except ValueError:
            raise _refused(service.LatticeRefused("bad_filter", "filters is JSON: [{column, op, value}]."))
    if column and value is not None:
        parsed = (parsed if isinstance(parsed, list) else []) + [{"column": column, "op": "eq", "value": value}]
    try:
        out = await service.query(store, customer_id, lattice_id, filters=parsed, search=search, sort=sort,
                                  descending=desc, limit=limit, offset=offset)
    except service.LatticeRefused as e:
        raise _refused(e)
    return JSONResponse(content=out)


@router.post(_BASE + "/{lattice_id}/rows")
async def write_lattice_rows(customer_id: ID64, lattice_id: ID64, request: Request,
                             store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    customer = await _customer(store, customer_id)
    body = await _body(request)
    _unknown(body, {"rows", "source"})
    try:
        out = await service.write_rows(store, customer_id, lattice_id, body.get("rows"),
                                       source=str(body.get("source") or "added by you"), actor="console",
                                       customer=customer, correct=True)
    except service.LatticeRefused as e:
        raise _refused(e)
    action_id = await service.ledger(store, customer_id, out, actor="console", reason=_CONSOLE_REASON)
    return JSONResponse(content={**service.public(out), "action_id": action_id})


@router.post(_BASE + "/{lattice_id}/rows/delete")
async def delete_lattice_rows(customer_id: ID64, lattice_id: ID64, request: Request,
                              store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    body = await _body(request)
    _unknown(body, {"row_ids"})
    try:
        out = await service.delete_rows(store, customer_id, lattice_id, body.get("row_ids"))
    except service.LatticeRefused as e:
        raise _refused(e)
    return JSONResponse(content=out)


@router.get(_BASE + "/{lattice_id}/rows/{row_id}")
async def lattice_row(customer_id: ID64, lattice_id: ID64, row_id: ID64,
                      store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    try:
        lat = await service.get(store, customer_id, lattice_id)
    except service.LatticeRefused as e:
        raise _refused(e)
    rows = await store.get_lattice_rows(customer_id, lat["id"], [row_id])
    if not rows:
        raise HTTPException(status_code=404, detail="Row not found")
    return JSONResponse(content={"row": service.row_view(lat, rows[0], history=True)})


@router.patch(_BASE + "/{lattice_id}/rows/{row_id}")
async def edit_lattice_cell(customer_id: ID64, lattice_id: ID64, row_id: ID64, request: Request,
                            store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    customer = await _customer(store, customer_id)
    body = await _body(request)
    _unknown(body, {"column", "value"})
    try:
        out = await service.set_cell(store, customer_id, lattice_id, row_id, str(body.get("column") or ""),
                                     body.get("value"), actor="console", customer=customer)
    except service.LatticeRefused as e:
        raise _refused(e)
    action_id = await service.ledger(store, customer_id, out, actor="console", reason=_CONSOLE_REASON)
    rows = await store.get_lattice_rows(customer_id, out["lattice_id"], [row_id])
    lat = await service.get(store, customer_id, out["lattice_id"])
    return JSONResponse(content={"row": service.row_view(lat, rows[0], history=True) if rows else None,
                                 "action_id": action_id})


@router.get(_BASE + "/{lattice_id}/entities")
async def lattice_entities(customer_id: ID64, lattice_id: ID64,
                           store: Annotated[MetadataStore, Depends(get_metadata_store)],
                           column: Optional[COLKEY] = None, search: Optional[NEEDLE] = None,
                           limit: LIMIT = 30) -> JSONResponse:
    try:
        out = await service.entities(store, customer_id, lattice_id, column, search=search, limit=limit)
    except service.LatticeRefused as e:
        raise _refused(e)
    return JSONResponse(content=out)
