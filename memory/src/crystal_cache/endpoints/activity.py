"""Activity trace routes (2026-10-19). Anthony: "I need to be able to
review every command, skill, tool, mcp used. I need to know every little
detail about any run, chat, or background process."

Three reads under the tenant-pathed admin prefix (the guard scopes
/admin/api/customers/{cid}/* to the tenant's own id; a platform admin
reaches any):

  GET .../query_logs/{id}/trace   one question's full trace: the agent
                                  turn's self-report (every tool call
                                  with its input, output and duration;
                                  inferences, borders crossed, gaps
                                  felt), the model calls of that turn,
                                  and the connected-tool call that
                                  produced it (recall rows).
  GET .../model-calls             every model call (the llm_calls
                                  ledger), newest first; optional
                                  session_id narrows to one run or one
                                  conversation.
  GET .../tool-calls              every tool call from a connected
                                  client (tool_invocations), newest first.

Dollars reach platform admins only (admin_route_principal_kind, the
Lockdown PR-4 rule the spend view follows); tenants get tokens.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from ..infrastructure import MetadataStore
from ..infrastructure.metadata_store import get_metadata_store
from ..ingress.auth import admin_route_principal_kind
from ..ingress.limits import ID64, LIMIT

router = APIRouter()

# A chat turn's model calls carry the conversation's sequence_id; the
# turn itself is the window from "the request started" (timestamp minus
# latency) to a couple of minutes after (the self-critique runs then).
_TURN_BEFORE = timedelta(seconds=5)
_TURN_AFTER = timedelta(minutes=2)


def _strip_dollars(rows: list[dict[str, Any]], *, admin: bool) -> list[dict[str, Any]]:
    if admin:
        return rows
    return [{k: v for k, v in r.items() if k != "cost_micro_usd"} for r in rows]


@router.get("/admin/api/customers/{customer_id}/query_logs/{query_log_id}/trace")
async def query_log_trace(
    request: Request,
    customer_id: ID64,
    query_log_id: ID64,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    ql = await store.get_query_log(query_log_id, customer_id=customer_id)
    if ql is None:
        raise HTTPException(status_code=404, detail="Query log row not found")
    admin = await admin_route_principal_kind(request, store) == "platform_admin"
    trace = await store.get_reasoning_trace_for_query_log(customer_id, query_log_id)
    model_calls: list[dict[str, Any]] = []
    if ql.sequence_id:
        ts: datetime = ql.timestamp if ql.timestamp.tzinfo else ql.timestamp.replace(tzinfo=timezone.utc)
        started = ts - timedelta(milliseconds=int(ql.latency_ms or 0)) - _TURN_BEFORE
        model_calls = await store.list_llm_calls(
            customer_id, session_id=ql.sequence_id, since=started, until=ts + _TURN_AFTER, limit=50,
        )
    tool_calls = await store.list_tool_invocations(customer_id, query_log_id=query_log_id, limit=5)
    return JSONResponse(content={
        "query_log_id": ql.id,
        "sequence_id": ql.sequence_id,
        "turn_index": ql.turn_index,
        "model": getattr(ql, "model", None),
        "trace": None if trace is None else {
            "id": trace.id,
            "tool_calls": trace.tool_calls,
            "events": trace.events,
            "crystals_used": trace.crystals_used,
            "inferences": trace.inferences,
            "borders_crossed": trace.borders_crossed,
            "gaps_felt": trace.gaps_felt,
            "created_at": trace.created_at.isoformat() if getattr(trace, "created_at", None) else None,
        },
        "model_calls": _strip_dollars(model_calls, admin=admin),
        "tool_calls": tool_calls,
    })


@router.get("/admin/api/customers/{customer_id}/model-calls")
async def list_model_calls(
    request: Request,
    customer_id: ID64,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    session_id: Optional[ID64] = None,
    limit: LIMIT = 200,
) -> JSONResponse:
    admin = await admin_route_principal_kind(request, store) == "platform_admin"
    rows = await store.list_llm_calls(customer_id, session_id=session_id, limit=limit)
    return JSONResponse(content={"calls": _strip_dollars(rows, admin=admin), "count": len(rows)})


@router.get("/admin/api/customers/{customer_id}/tool-calls")
async def list_tool_calls(
    customer_id: ID64,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    limit: LIMIT = 200,
) -> JSONResponse:
    rows = await store.list_tool_invocations(customer_id, limit=limit)
    return JSONResponse(content={"calls": rows, "count": len(rows)})
