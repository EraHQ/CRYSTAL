"""Goals: console routes (2026-10-24, Q109 A).

Tenant-pathed under /admin/api/customers/{customer_id}/goals (the tenant
itself and a platform admin reach them; ingress/auth _TENANT_PATH_RE):

GET    .../goals                           goals, with open item counts
POST   .../goals {title, directive, may_send, may_post}
GET    .../goals/{id}                      one goal with its items
PATCH  .../goals/{id} {title, directive, status, may_send, may_post}
DELETE .../goals/{id}                      the goal and its items
POST   .../goals/{id}/replan               a fresh plan from the frontier tier
POST   .../goals/{id}/run                  check in on the next pass
POST   .../goals/{id}/run-anyway           a goal held for setup runs with what it has (Q112 A)
POST   .../goals/{id}/decide {answers}     answer the planner's questions; it then writes the plan
GET    .../goals/{id}/runs                 each planner and check-in run, counted (2026-10-27)
GET    .../goals/{id}/progress             each step's criteria measured now
GET    .../goals/{id}/comments             what the person said, and which check-in read it
POST   .../goals/{id}/comments {body, always}   every check-in reads it until it is handled (Series 385)
PATCH  .../goals/{id}/comments/{note} {status, always}   close, reopen or dismiss a note, or make it standing
PATCH  .../goals/{id}/steps/{n} {criteria} edit a step's completion criteria (Q136 A)
POST   .../goals/{id}/steps/{n}/review {approve, note}   a step waiting on the person
POST   .../goals/{id}/rollback {to_step, reason, preview}   back to any step (Q137 A); preview changes nothing
POST   .../goal-items/{item_id} {action: send|done|dismiss|reopen|answer, title?, body?}
GET    .../pulse                           the Command center's stats (Q110 B), read-only
"""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import Field
from fastapi.responses import JSONResponse

from ..goals import service
from ..infrastructure import MetadataStore
from ..infrastructure.metadata_store import get_metadata_store
from ..ingress.errors import build_error_envelope
from ..ingress.limits import ID64

router = APIRouter()
STEP_N = Annotated[int, Field(ge=1, le=50)]


def _refused(e: service.GoalRefused) -> HTTPException:
    return HTTPException(status_code=e.status, detail=build_error_envelope(
        e.message, error_type="invalid_request_error", param="goal", code=e.code))


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
        raise _refused(service.GoalRefused("invalid_request", f"unknown field(s) {sorted(extra)}"))


async def _customer(store: MetadataStore, customer_id: str) -> None:
    if await store.get_customer_by_id(customer_id) is None:
        raise HTTPException(status_code=404, detail="Customer not found")


@router.get("/admin/api/customers/{customer_id}/goals")
async def list_goals(customer_id: ID64, store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    await _customer(store, customer_id)
    goals = await store.list_goals(customer_id)
    open_items = await store.list_goal_items(customer_id, status="open", limit=500)
    for g in goals:
        mine = [i for i in open_items if i["goal_id"] == g["id"]]
        g["open"] = {k: sum(1 for i in mine if i["kind"] == k) for k in ("action", "draft", "note")}
    return JSONResponse(content={"goals": goals, "count": len(goals)})


@router.post("/admin/api/customers/{customer_id}/goals")
async def create_goal(customer_id: ID64, request: Request,
                      store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    await _customer(store, customer_id)
    body = await _body(request)
    _unknown(body, {"title", "directive", "may_send", "may_post"})
    try:
        goal = await service.create(store, customer_id, title=body.get("title"), directive=body.get("directive"),
                                    may_send=body.get("may_send", False), may_post=body.get("may_post", False))
    except service.GoalRefused as e:
        raise _refused(e)
    return JSONResponse(status_code=201, content={"goal": goal})


@router.get("/admin/api/customers/{customer_id}/goals/{goal_id}")
async def get_goal(customer_id: ID64, goal_id: ID64,
                   store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    goal = await store.get_goal(customer_id, goal_id)
    if goal is None:
        raise HTTPException(status_code=404, detail="Goal not found")
    items = await store.list_goal_items(customer_id, goal_id, limit=200)
    return JSONResponse(content={"goal": goal, "items": items})


@router.patch("/admin/api/customers/{customer_id}/goals/{goal_id}")
async def update_goal(customer_id: ID64, goal_id: ID64, request: Request,
                      store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    body = await _body(request)
    _unknown(body, {"title", "directive", "status", "may_send", "may_post"})
    try:
        goal = await service.update(store, customer_id, goal_id, **{k: body.get(k) for k in
                                                                    ("title", "directive", "status", "may_send", "may_post")})
    except service.GoalRefused as e:
        raise _refused(e)
    return JSONResponse(content={"goal": goal})


@router.delete("/admin/api/customers/{customer_id}/goals/{goal_id}")
async def delete_goal(customer_id: ID64, goal_id: ID64,
                      store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    if not await store.delete_goal(customer_id, goal_id):
        raise HTTPException(status_code=404, detail="Goal not found")
    return JSONResponse(content={"deleted": True, "goal_id": goal_id})


@router.post("/admin/api/customers/{customer_id}/goals/{goal_id}/replan")
async def replan_goal(customer_id: ID64, goal_id: ID64,
                      store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    try:
        goal = await service.replan(store, customer_id, goal_id)
    except service.GoalRefused as e:
        raise _refused(e)
    return JSONResponse(content={"goal": goal, "queued": True})


@router.post("/admin/api/customers/{customer_id}/goals/{goal_id}/run")
async def run_goal(customer_id: ID64, goal_id: ID64,
                   store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    try:
        goal = await service.run_now(store, customer_id, goal_id)
    except service.GoalRefused as e:
        raise _refused(e)
    return JSONResponse(content={"goal": goal, "queued": True})


@router.post("/admin/api/customers/{customer_id}/goals/{goal_id}/decide")
async def decide_goal(customer_id: ID64, goal_id: ID64, request: Request,
                      store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    body = await _body(request)
    _unknown(body, {"answers"})
    try:
        goal = await service.decide(store, customer_id, goal_id, body.get("answers"))
    except service.GoalRefused as e:
        raise _refused(e)
    return JSONResponse(content={"goal": goal, "queued": True})


@router.post("/admin/api/customers/{customer_id}/goals/{goal_id}/run-anyway")
async def run_goal_anyway(customer_id: ID64, goal_id: ID64,
                          store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    try:
        goal = await service.run_anyway(store, customer_id, goal_id)
    except service.GoalRefused as e:
        raise _refused(e)
    return JSONResponse(content={"goal": goal, "queued": True})


@router.get("/admin/api/customers/{customer_id}/goals/{goal_id}/runs")
async def goal_runs(customer_id: ID64, goal_id: ID64,
                    store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    try:
        runs = await service.runs_view(store, customer_id, goal_id)
    except service.GoalRefused as e:
        raise _refused(e)
    return JSONResponse(content={"runs": runs, "count": len(runs)})


@router.get("/admin/api/customers/{customer_id}/goals/{goal_id}/progress")
async def goal_progress(customer_id: ID64, goal_id: ID64,
                        store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    from ..goals.criteria import progress
    goal = await store.get_goal(customer_id, goal_id)
    if goal is None:
        raise HTTPException(status_code=404, detail="Goal not found")
    snaps = await store.list_goal_snapshots(customer_id, goal_id)
    return JSONResponse(content={"steps": await progress(store, customer_id, goal),
                                 "snapshots": [{"step_n": x["step_n"], "at": x["created_at"], "rows": x["rows"]}
                                               for x in snaps]})


@router.get("/admin/api/customers/{customer_id}/goals/{goal_id}/comments")
async def goal_comments(customer_id: ID64, goal_id: ID64,
                        store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    if await store.get_goal(customer_id, goal_id) is None:
        raise HTTPException(status_code=404, detail="Goal not found")
    comments = await store.list_goal_comments(customer_id, goal_id)
    return JSONResponse(content={"comments": comments, "count": len(comments)})


@router.post("/admin/api/customers/{customer_id}/goals/{goal_id}/comments")
async def add_goal_comment(customer_id: ID64, goal_id: ID64, request: Request,
                           store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    body = await _body(request)
    _unknown(body, {"body", "always"})
    try:
        c = await service.comment(store, customer_id, goal_id, body.get("body"), always=body.get("always", False))
    except service.GoalRefused as e:
        raise _refused(e)
    return JSONResponse(status_code=201, content={"comment": c})


@router.patch("/admin/api/customers/{customer_id}/goals/{goal_id}/comments/{note_id}")
async def set_goal_note(customer_id: ID64, goal_id: ID64, note_id: ID64, request: Request,
                        store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    body = await _body(request)
    _unknown(body, {"status", "always"})
    try:
        c = await service.set_note(store, customer_id, goal_id, note_id, status=body.get("status"), always=body.get("always"))
    except service.GoalRefused as e:
        raise _refused(e)
    return JSONResponse(content={"comment": c})


@router.patch("/admin/api/customers/{customer_id}/goals/{goal_id}/steps/{n}")
async def set_step_criteria(customer_id: ID64, goal_id: ID64, n: STEP_N, request: Request,
                            store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    body = await _body(request)
    _unknown(body, {"criteria"})
    try:
        goal = await service.set_criteria(store, customer_id, goal_id, n, body.get("criteria"))
    except service.GoalRefused as e:
        raise _refused(e)
    return JSONResponse(content={"goal": goal})


@router.post("/admin/api/customers/{customer_id}/goals/{goal_id}/steps/{n}/review")
async def review_goal_step(customer_id: ID64, goal_id: ID64, n: STEP_N, request: Request,
                           store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    body = await _body(request)
    _unknown(body, {"approve", "note"})
    try:
        goal = await service.review_step(store, customer_id, goal_id, n, approve=body.get("approve"), note=body.get("note"))
    except service.GoalRefused as e:
        raise _refused(e)
    return JSONResponse(content={"goal": goal})


@router.post("/admin/api/customers/{customer_id}/goals/{goal_id}/rollback")
async def rollback_goal(customer_id: ID64, goal_id: ID64, request: Request,
                        store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    body = await _body(request)
    _unknown(body, {"to_step", "reason", "preview"})
    app_state = getattr(getattr(request, "app", None), "state", None)
    state = {"encoder": getattr(app_state, "prompt_encoder", None),
             "vector_store": getattr(app_state, "vector_store", None),
             "fact_vector_store": getattr(app_state, "fact_vector_store", None)} if app_state is not None else None
    try:
        out = await service.rollback(store, customer_id, goal_id, body.get("to_step"), reason=body.get("reason"),
                                     state=state, actor="console", preview=body.get("preview") is True)
    except service.GoalRefused as e:
        raise _refused(e)
    return JSONResponse(content=out)


@router.post("/admin/api/customers/{customer_id}/goal-items/{item_id}")
async def goal_item_action(customer_id: ID64, item_id: ID64, request: Request,
                           store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    body = await _body(request)
    _unknown(body, {"action", "title", "body"})
    try:
        out = await service.item_action(store, customer_id, item_id, str(body.get("action") or ""), actor="console",
                                        title=body.get("title"), body=body.get("body"))
    except service.GoalRefused as e:
        raise _refused(e)
    return JSONResponse(content={"item": out.get("item"), "ref": out.get("ref")})


@router.get("/admin/api/customers/{customer_id}/pulse")
async def command_center_pulse(customer_id: ID64,
                               store: Annotated[MetadataStore, Depends(get_metadata_store)]) -> JSONResponse:
    from ..pulse import build_pulse
    customer = await store.get_customer_by_id(customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return JSONResponse(content=await build_pulse(store, customer))
