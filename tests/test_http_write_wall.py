"""RC-09 (2026-10-04): the HTTP lane pays the same walls as MCP.

Only document upload called require_write_capacity; store, learn, import,
topology import, approve, crystallize, crystallize-all, gap research, the
console spenders and the cognition requeue wrote or spent with no tenant
door, so a free Key A could write past 2,500 facts and run platform money
through /v1. Two pins:

1. STRUCTURAL, over app.routes: every POST/PUT/PATCH under /v1 that is
   not on the explicit allow-list below must call the tenant door
   (require_write_capacity or enforce_managed_budget) in its handler. A
   new write route cannot ship unclassified.
2. BEHAVIORAL, through the real app with Key A: a free managed tenant at
   the fact cap and out of daily capacity gets 402/429 with X-Plan-Wall
   on store, import, approve and crystallize-all, before any body is
   acted on.

RC-10 rides along: the MCP door and the HTTP door answer identically at
the monthly edge, the daily edge, and with carry; byok rows never move
the platform meter; the carry is capped at one day.
"""
import inspect
import re
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException

from crystal_cache.control.admission import (
    TIER_TABLE, daily_capacity, daily_capacity_block, enforce_managed_budget,
)

# Routes under /v1 that mutate but are NOT bank writes or spenders: account,
# billing, access, configuration, reads that use POST, and the Your-data
# actions that must keep working while walled.
_NOT_A_BANK_WRITE = {
    "/v1/customers",
    "/v1/customers/{customer_id}/api_key",
    "/v1/customers/{customer_id}/model",
    "/v1/customers/{customer_id}/inference_mode",
    "/v1/customers/{customer_id}/upstream_key",
    "/v1/customers/{customer_id}/assumptions_explore",
    "/v1/customers/{customer_id}/budgets/{function}",
    "/v1/customers/{customer_id}/settings",
    "/v1/customers/{customer_id}/operators",
    "/v1/retrieve",
    "/v1/export",
    "/v1/export/topology",
    "/v1/subscribe",
    "/v1/unsubscribe",
    "/v1/crystals/{crystal_id}/scope",
    "/v1/crystals/{crystal_id}/grants",
    "/v1/documents/{document_id}/scope",
    "/v1/documents/{document_id}/review",
    "/v1/me",
    "/v1/me/erase",
    "/v1/me/delete",
    "/v1/me/restore",
    "/v1/me/onboarding",
    "/v1/billing/checkout",
    "/v1/billing/portal",
    "/v1/billing/webhook",
    "/v1/oauth/token",
    "/v1/oauth/register",
    "/v1/oauth/authorize",
    "/v1/oauth/revoke",
    "/v1/feedback",  # the row is always recorded; learning pays the door inside
    # Classified 2026-10-04 from reading each handler (R18):
    "/v1/operators",                           # team management, no bank write
    "/v1/operators/{operator_id}/role",        # team management
    "/v1/operators/{operator_id}/status",      # team management
    "/v1/promotion/merge",                     # admin-gated merge of EXISTING crystals; no model call, no new facts
    "/v1/sessions/heartbeat",                  # telemetry
    "/v1/control/decisions",                   # agent control plane (approval decision)
    "/v1/control/terminate",                   # agent control plane
    "/v1/sessions/{session_id}/commands/claim",  # agent control plane
    "/v1/marketplace/experts",                 # authorization config
    "/v1/marketplace/experts/revoke",          # authorization config
    "/v1/compliance/baa",                      # compliance config
    "/v1/completions",                         # deliberate 501 (legacy)
}

# Routes whose door lives in the helper the endpoint delegates to. The pin
# follows the delegation and checks the helper's source (verified: the
# agent door is in run_agent_messages, the proxy door in
# run_chat_completion).
_DOOR_IN_HELPER = {
    "/v1/agent/messages": ("crystal_cache.endpoints.agent", "run_agent_messages"),
    "/v1/chat/completions": ("crystal_cache.endpoints.chat_proxy", "run_chat_completion"),
}

_DOOR = re.compile(r"require_write_capacity\(|enforce_managed_budget\(")


def _walk(routes):
    """FastAPI 0.142 keeps an included router in app.routes as an
    _IncludedRouter whose `original_router` is the APIRouter; its
    `.routes` are flat APIRoute objects with path, methods and endpoint
    (verified against fastapi 0.142.2 on a nested, prefixed app)."""
    for route in routes:
        orig = getattr(route, "original_router", None)
        if orig is not None:
            yield from _walk(orig.routes)
        else:
            yield route


def _write_routes(app):
    seen = set()
    for route in _walk(app.routes):
        if not hasattr(route, "endpoint"):
            continue
        methods = set(getattr(route, "methods", None) or set()) & {"POST", "PUT", "PATCH"}
        path = getattr(route, "path", "")
        key = (tuple(sorted(methods)), path)
        if methods and path.startswith("/v1") and key not in seen:
            seen.add(key)
            yield route, sorted(methods), path


def test_every_v1_write_route_calls_the_tenant_door(store, customer, semantic_encoder_stub,
                                                     vector_store, fact_vector_store):
    try:
        from tests.test_endpoint_smoke import _build_app
    except ModuleNotFoundError:
        from test_endpoint_smoke import _build_app

    app = _build_app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    missing = []
    inspected = []
    for route, methods, path in _write_routes(app):
        if path in _NOT_A_BANK_WRITE:
            continue
        inspected.append(path)
        if path in _DOOR_IN_HELPER:
            import importlib

            mod_name, fn_name = _DOOR_IN_HELPER[path]
            src = inspect.getsource(getattr(importlib.import_module(mod_name), fn_name))
        else:
            src = inspect.getsource(route.endpoint)
        if not _DOOR.search(src):
            missing.append(f"{'/'.join(methods)} {path}")
    # The walk must have found the routes this pin exists for, or a
    # trivially green run proves nothing.
    for expected in ("/v1/store", "/v1/learn", "/v1/import", "/v1/import/topology",
                     "/v1/documents/{document_id}/approve",
                     "/v1/documents/{document_id}/crystallize",
                     "/v1/documents/crystallize-all", "/v1/gaps/{gap_id}/research"):
        assert expected in inspected, (
            f"route walk did not see {expected}: inspected={inspected}; "
            f"app.routes={[(type(r).__name__, getattr(r, 'path', '?'), sorted(getattr(r, 'methods', None) or [])) for r in list(app.routes)[:12]]}"
        )
    assert missing == [], (
        "Write routes under /v1 with no tenant door. Add require_write_capacity "
        "(spend=True when the route runs a model) or list the route in "
        f"_NOT_A_BANK_WRITE with a reason: {missing}"
    )


async def _free_managed_at_the_wall(store, customer, monkeypatch):
    await store.set_customer_subscription(customer.id, "free", None)
    await store.set_customer_inference_mode(customer.id, "managed")

    async def _facts(cid):
        return 10**6

    async def _spent(cid, *, since, until=None):
        return 10**9

    monkeypatch.setattr(store, "count_billable_facts", _facts)
    monkeypatch.setattr(store, "platform_spend_micro_usd", _spent)


@pytest.mark.asyncio
async def test_walled_tenant_is_refused_on_the_real_routes(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store, monkeypatch
):
    from httpx import ASGITransport, AsyncClient

    try:
        from tests.test_endpoint_smoke import _build_app
    except ModuleNotFoundError:
        from test_endpoint_smoke import _build_app

    app = _build_app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    await _free_managed_at_the_wall(store, customer, monkeypatch)
    headers = {"Authorization": f"Bearer {customer.api_key}"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        for method, path, body in (
            ("POST", "/v1/store", {"key": "k", "value": "v"}),
            ("POST", "/v1/import", {"records": [{"key": "k", "value": "v"}]}),
            ("POST", "/v1/documents/crystallize-all", None),
            ("POST", "/v1/documents/doc_sample/approve", {}),
            ("POST", "/v1/documents/doc_sample/crystallize", None),
        ):
            r = await c.request(method, path, headers=headers, json=body)
            assert r.status_code in (402, 429), f"{method} {path}: {r.status_code} {r.text[:200]}"
            assert r.headers.get("x-plan-wall") in ("memory_full", "daily_capacity", "monthly_budget"), path


# ----------------------------------------------------------------------------
# RC-10: the doors agree
# ----------------------------------------------------------------------------

class _Cust:
    def __init__(self, tier="free", mode="managed"):
        self.id = "cus_x"
        self.subscription_tier = tier
        self.inference_mode = mode


class _Ledger:
    """A fake ledger keyed by window, with exactly the real store's
    platform_spend_micro_usd signature."""

    def __init__(self, today=0, yesterday=0, month=0):
        self.today, self.yesterday, self.month = today, yesterday, month

    async def platform_spend_micro_usd(self, cid, *, since, until=None):
        midnight = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        if until is not None:
            return self.yesterday
        if since == midnight:
            return self.today
        return self.month


@pytest.mark.asyncio
@pytest.mark.parametrize("today,yesterday,month,expect", [
    (0, 0, 0, None),
    (TIER_TABLE["free"].daily_managed_budget_micro_usd, 0, 0, "daily_capacity"),
    (0, TIER_TABLE["free"].daily_managed_budget_micro_usd * 2, 0, "daily_capacity"),  # carry
    (0, 0, TIER_TABLE["free"].monthly_managed_budget_micro_usd, "monthly_budget"),
])
async def test_mcp_door_and_http_door_give_the_same_answer(today, yesterday, month, expect):
    ledger = _Ledger(today, yesterday, month)
    cust = _Cust()
    mcp = await daily_capacity_block(ledger, cust)
    try:
        await enforce_managed_budget(ledger, cust)
        http = None
    except HTTPException as e:
        http = e.headers["X-Plan-Wall"]
    assert (mcp or {}).get("code") == expect
    assert http == expect


@pytest.mark.asyncio
async def test_carry_is_capped_at_one_day():
    free = TIER_TABLE["free"].daily_managed_budget_micro_usd
    state = await daily_capacity(_Ledger(today=0, yesterday=free * 10), _Cust())
    assert state["carried"] == free  # not free * 9
    assert state["state"] == "blocked"
    # Tomorrow starts clean: a fresh day with yesterday at exactly the
    # allowance carries nothing.
    state = await daily_capacity(_Ledger(today=0, yesterday=free), _Cust())
    assert state["carried"] == 0 and state["state"] == "ok"


def test_byok_rows_are_stamped_byok_everywhere_the_agent_and_shadow_write_the_ledger():
    from crystal_cache.agent import turn_finalize
    from crystal_cache.execution import shadow_evaluator

    for mod in (turn_finalize, shadow_evaluator):
        src = inspect.getsource(mod)
        assert 'else "byok"' in src, mod.__name__
        assert "else None" not in src.split("billing=")[1][:200], mod.__name__


def test_admit_task_is_gone():
    from crystal_cache.control import admission

    assert not hasattr(admission, "admit_task")
