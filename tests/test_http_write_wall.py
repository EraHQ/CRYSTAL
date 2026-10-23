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

# Routes that mutate but are NOT bank writes or spenders: account,
# billing, access, configuration, reads that use POST, deletes (RC-05:
# the Your-data actions must keep working while walled), team and agent
# control plane. Every entry names a reason. Lockdown PR-8 (Q57 class 5):
# the walk covers /admin/api and DELETE too, and an entry that matches
# no route is itself a failure (a dead entry would silently exempt a
# future route with the same path).
_NOT_A_BANK_WRITE = {
    # account / identity
    "/v1/customers": "tenant minting (platform-admin gated)",
    "/v1/customers/{customer_id}/api_key": "key rotation",
    "/v1/customers/{customer_id}/model": "settings",
    "/v1/customers/{customer_id}/background_model": "settings (Q87)",
    "/v1/customers/{customer_id}/inference_mode": "settings",
    "/v1/customers/{customer_id}/upstream_key": "settings",
    "/v1/customers/{customer_id}/assumptions_explore": "settings",
    "/v1/customers/{customer_id}/rules": "settings (automation rules, Q99 C 2026-10-23)",
    "/v1/customers/{customer_id}/rules/{rule_id}": "settings (automation rules)",
    "/v1/customers/{customer_id}/budgets/{function}": "settings",
    "/v1/auth/signup": "mints a trial tenant; its allowance is the door from then on",
    "/v1/oauth/approve": "identity (consent → code)",
    "/v1/me/erase": "Your-data action; must work while walled",
    "/v1/me/delete": "Your-data action",
    "/v1/me/restore": "Your-data action",
    "/v1/me/onboarding": "profile fields, no bank write",
    "/v1/billing/checkout": "billing",
    "/v1/billing/portal": "billing",
    "/v1/billing/webhook": "billing (Stripe-signed)",
    # reads that use POST / configuration
    "/v1/retrieve": "a read",
    "/v1/export/topology": "a read (export)",
    "/v1/subscribe": "subscription config (general bank)",
    "/v1/unsubscribe": "subscription config",
    "/v1/subscribe/{crystal_type}": "subscription config (DELETE)",
    "/v1/crystals/{crystal_id}/scope": "access control on an existing crystal",
    "/v1/crystals/{crystal_id}/grants": "access control",
    "/v1/documents/{document_id}/scope": "access control",
    "/v1/documents/{document_id}/review": "edits a parked row; the approve pays the door",
    # Console Refine PR 6 (5.4 item 5): status flips on an existing gap row; /answer pays the fact door.
    "/v1/gaps/{gap_id}/dismiss": "closes a gap; no bank write",
    "/v1/gaps/{gap_id}/reopen": "the Undo of dismiss; no bank write",
    "/v1/feedback": "the row is always recorded; learning pays the door inside",
    # deletes (RC-05: forgetting is always possible)
    "/v1/crystals/{crystal_id}": "delete",
    "/v1/documents/{document_id}": "delete",
    "/v1/facts/{fact_id}": "delete",
    # team and agent control plane (classified 2026-10-04 by reading each handler, R18)
    "/v1/operators": "team management, no bank write",
    "/v1/operators/{operator_id}/role": "team management",
    "/v1/operators/{operator_id}/status": "team management",
    "/v1/groups": "team management",
    "/v1/groups/{group_id}/members": "team management",
    "/v1/groups/{group_id}/members/{operator_id}": "team management (DELETE)",
    "/v1/promotion/merge": "admin-gated merge of EXISTING crystals; no model call, no new facts",
    "/v1/sessions/heartbeat": "telemetry",
    "/v1/control/decisions": "agent control plane (approval decision)",
    "/v1/control/terminate": "agent control plane",
    "/v1/sessions/{session_id}/commands/claim": "agent control plane",
    "/v1/marketplace/experts": "authorization config",
    "/v1/marketplace/experts/revoke": "authorization config",
    "/v1/compliance/baa": "compliance config",
    "/v1/completions": "deliberate 501 (legacy)",
    # /admin/api (Lockdown PR-8): console actions that create no billable
    # facts and spend no model money, or whose door is paid elsewhere.
    "/admin/api/push-queue/{item_id}/reject": "discards a pending push",
    "/admin/api/conflicts/{conflict_id}/resolve": "marks a resolution on existing facts; no model call",
    "/admin/api/conflicts/{conflict_id}/reopen": "the Undo of resolve: restores existing facts; no model call",
    "/admin/api/crystals/{crystal_id}/facts/{fact_id}/supersede": "replaces a fact in place (net zero facts)",
    "/admin/api/crystals/{crystal_id}/facts/{fact_id}/retire": "delete",
    "/admin/api/crystals/{crystal_id}/tier": "curation flag",
    "/admin/api/crystals/{crystal_id}": "delete",
    "/admin/api/assumptions/{crystal_id}/approve": "clears a recall gate; assumption origin rides free",
    "/admin/api/watches": "the door runs in the sync loop (RC-12, source_sync.py) before any write",
    "/admin/api/watches/{watch_id}": "pause / sync-now / delete of a watch",
    "/admin/api/source-schemas/{schema_id}/approve": "releases documents that paid the upload door; the worker's per-tenant budget gate covers extraction",
    "/admin/api/source-schemas/{schema_id}/reject": "parks documents",
    "/admin/api/source-schemas/{schema_id}/mapping": "edits a mapping",
    "/admin/api/source-schemas/preview": "stateless preview",
    "/admin/api/customers/{customer_id}/gdrive/connections/{connection_id}": "delete",
    "/admin/api/cognition/environments/{env_id}/critiques": "a note on a run; no model call",
    "/admin/api/cognition/tasks/{task_id}/cancel": "cancel",
    "/admin/api/cognition/critiques/{critique_id}": "status flip",
    "/admin/api/metacognition/substrate-observations/{item_id}/dismiss": "dismiss",
    "/admin/api/metacognition/substrate-observations/dismiss-all": "dismiss",
    "/admin/api/metacognition/substrate-observations/{item_id}/restore": "the Undo of dismiss; status flip",
    "/admin/api/customers/{customer_id}/schedules": "creates a standing instruction; the run pays the door when the cognition worker executes it",
    "/admin/api/customers/{customer_id}/schedules/{schedule_id}": "pause / resume / delete of a schedule",
    "/admin/api/agents/tasks/{task_id}/retry": "re-enqueues a failed daemon task; the agent door runs when the daemon executes it",
}

# Routes whose door lives in the helper the endpoint delegates to. The pin
# follows the delegation and checks the helper's source (verified: the
# agent door is in run_agent_messages, the proxy door in
# run_chat_completion).
_DOOR_IN_HELPER = {
    "/v1/agent/messages": ("crystal_cache.endpoints.agent", "run_agent_messages"),
    "/v1/chat/completions": ("crystal_cache.endpoints.chat_proxy", "run_chat_completion"),
    "/admin/api/customers/{customer_id}/chat": ("crystal_cache.endpoints.chat_proxy", "run_chat_completion"),
    "/admin/api/customers/{customer_id}/agent": ("crystal_cache.endpoints.agent", "run_agent_messages"),
}

_DOOR = re.compile(r"require_write_capacity\(|enforce_managed_budget\(")
_WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def _walk(routes):
    """FastAPI keeps an included router in app.routes as an
    _IncludedRouter whose `original_router` is the APIRouter; its
    `.routes` are flat APIRoute objects with path, methods and endpoint."""
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
        methods = set(getattr(route, "methods", None) or set()) & _WRITE_METHODS
        path = getattr(route, "path", "")
        key = (tuple(sorted(methods)), path)
        if methods and path.startswith(("/v1", "/admin/api")) and key not in seen:
            seen.add(key)
            yield route, sorted(methods), path


def test_every_write_route_calls_the_tenant_door():
    """PR-8 (Q57 class 5): the walk is the REAL app's routes (every router
    app.py mounts, /v1 and /admin/api, DELETE included), not the smoke
    app's subset."""
    import importlib

    from crystal_cache.app import app

    missing = []
    inspected = []
    seen_paths = set()
    for route, methods, path in _write_routes(app):
        seen_paths.add(path)
        if path in _NOT_A_BANK_WRITE:
            continue
        inspected.append(path)
        if path in _DOOR_IN_HELPER:
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
                     "/v1/documents/crystallize-all", "/v1/gaps/{gap_id}/research",
                     "/admin/api/push-queue/{item_id}/approve",
                     "/admin/api/cognition/tasks/{task_id}/requeue",
                     "/admin/api/customers/{customer_id}/operators/seed"):
        assert expected in inspected, f"route walk did not see {expected}: inspected={inspected}"
    assert len(seen_paths) >= 60, len(seen_paths)
    dead = sorted(p for p in _NOT_A_BANK_WRITE if p not in seen_paths)
    assert dead == [], f"allow-list entries that match no write route (prune them): {dead}"
    assert missing == [], (
        "Write routes with no tenant door. Add require_write_capacity "
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
#
# Lockdown PR-8 (Q57 class 7): the parity pin drives the REAL doors —
# mcp_server._write_admission_block (the MCP tools' door) and
# ingress.auth.require_write_capacity (the HTTP routes' door) — not the
# two shared admission functions underneath them, so the doors' own
# pre-logic (the fact cap, the tier gate, fail-closed) is in the
# comparison. The clock is injected (admission._now), so the fake
# ledger's windows are computed from a fixed mid-month date and the pin
# cannot flake on the 1st, when midnight and the month start coincide.

FIXED_NOW = datetime(2026, 10, 15, 12, 30, tzinfo=timezone.utc)


class _Cust:
    def __init__(self, tier="free", mode="managed"):
        self.id = "cus_x"
        self.subscription_tier = tier
        self.inference_mode = mode
        self.deletion_scheduled_at = None
        self.trial_ends_at = None


class _Ledger:
    """A fake store with exactly the methods the two doors call, keyed by
    the windows admission computes from the injected clock."""

    def __init__(self, today=0, yesterday=0, month=0, facts=0, customer=None):
        self.today, self.yesterday, self.month = today, yesterday, month
        self.facts = facts
        self.customer = customer or _Cust()

    async def platform_spend_micro_usd(self, cid, *, since, until=None):
        midnight = FIXED_NOW.replace(hour=0, minute=0, second=0, microsecond=0)
        month_start = midnight.replace(day=1)
        if until is not None:
            assert until == midnight and since == midnight - timedelta(days=1)
            return self.yesterday
        if since == midnight:
            return self.today
        assert since == month_start, since
        return self.month

    async def count_billable_facts(self, cid):
        return self.facts

    async def get_customer_by_id(self, cid):
        return self.customer


def _pin_clock(monkeypatch):
    from crystal_cache.control import admission

    monkeypatch.setattr(admission, "_now", lambda: FIXED_NOW)


async def _drive_both_doors(ledger, monkeypatch):
    """(mcp code, http code) from the two real doors for one ledger."""
    from crystal_cache.agent import mcp_server
    from crystal_cache.ingress.auth import require_write_capacity

    _pin_clock(monkeypatch)
    monkeypatch.setattr(mcp_server, "_get_state", lambda: {"store": ledger})
    token = mcp_server._current_customer_id.set(ledger.customer.id)
    try:
        mcp = await mcp_server._write_admission_block(spend=True)
    finally:
        mcp_server._current_customer_id.reset(token)
    try:
        await require_write_capacity(ledger.customer, ledger, spend=True)
        http = None
    except HTTPException as e:
        http = e.headers["X-Plan-Wall"]
    return (mcp or {}).get("code"), http


_FREE = TIER_TABLE["free"]


@pytest.mark.asyncio
@pytest.mark.parametrize("today,yesterday,month,facts,expect", [
    (0, 0, 0, 0, None),
    (_FREE.daily_managed_budget_micro_usd, 0, 0, 0, "daily_capacity"),
    (0, _FREE.daily_managed_budget_micro_usd * 2, 0, 0, "daily_capacity"),  # carry
    (0, 0, _FREE.monthly_managed_budget_micro_usd, 0, "monthly_budget"),
    (0, 0, 0, 10**6, "memory_full"),  # the fact cap, checked by the doors themselves
    (_FREE.daily_managed_budget_micro_usd, 0, 0, 10**6, "memory_full"),  # the cap comes first
])
async def test_mcp_door_and_http_door_give_the_same_answer(today, yesterday, month, facts, expect, monkeypatch):
    ledger = _Ledger(today, yesterday, month, facts)
    mcp, http = await _drive_both_doors(ledger, monkeypatch)
    assert mcp == expect, (mcp, expect)
    assert http == expect, (http, expect)


@pytest.mark.asyncio
async def test_the_doors_agree_on_the_first_of_the_month_too(monkeypatch):
    """The old fake ledger told 'today' from 'this month' by comparing
    `since` to a midnight computed from the wall clock: on the 1st the two
    windows start at the same instant and the pin flaked. With the clock
    injected the 1st is just another date."""
    from crystal_cache.control import admission

    first = datetime(2026, 11, 1, 0, 5, tzinfo=timezone.utc)

    class _FirstLedger(_Ledger):
        async def platform_spend_micro_usd(self, cid, *, since, until=None):
            if until is not None:
                return self.yesterday
            return self.today if since.day == 1 else self.month

    ledger = _FirstLedger(today=_FREE.daily_managed_budget_micro_usd)
    monkeypatch.setattr(admission, "_now", lambda: first)
    from crystal_cache.agent import mcp_server
    from crystal_cache.ingress.auth import require_write_capacity

    monkeypatch.setattr(mcp_server, "_get_state", lambda: {"store": ledger})
    token = mcp_server._current_customer_id.set("cus_x")
    try:
        mcp = await mcp_server._write_admission_block(spend=True)
    finally:
        mcp_server._current_customer_id.reset(token)
    with pytest.raises(HTTPException) as exc:
        await require_write_capacity(ledger.customer, ledger, spend=True)
    assert mcp["code"] == exc.value.headers["X-Plan-Wall"] == "daily_capacity"


@pytest.mark.asyncio
async def test_a_store_failure_refuses_at_both_doors(monkeypatch):
    """RC-05: a wall that cannot be evaluated refuses the write at the MCP
    door (admission_unavailable); the HTTP door raises, which the route
    turns into a 5xx rather than a free pass."""
    class _Broken(_Ledger):
        async def count_billable_facts(self, cid):
            raise RuntimeError("db down")

    ledger = _Broken()
    from crystal_cache.agent import mcp_server
    from crystal_cache.ingress.auth import require_write_capacity

    _pin_clock(monkeypatch)
    monkeypatch.setattr(mcp_server, "_get_state", lambda: {"store": ledger})
    token = mcp_server._current_customer_id.set("cus_x")
    try:
        mcp = await mcp_server._write_admission_block(spend=True)
    finally:
        mcp_server._current_customer_id.reset(token)
    assert mcp["code"] == "admission_unavailable"
    with pytest.raises(RuntimeError):
        await require_write_capacity(ledger.customer, ledger, spend=True)


@pytest.mark.asyncio
async def test_carry_is_capped_at_one_day(monkeypatch):
    _pin_clock(monkeypatch)
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
