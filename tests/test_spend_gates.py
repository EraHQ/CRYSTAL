"""v108 spend gates (docs/AUDIT_LLM_SPEND.md; Q11-Q15, Q17, Q22 ratified
2026-09-30 / 2026-10-01).

Q11=A  the daily allowance is ENFORCED at the door of every spend path.
Q12=A  MCP returns a structured `daily_capacity` error.
Q13=A  free gets Haiku + Sonnet; Opus from Starter.
Q14=A  monthly backstop = 30 x daily, counts all platform-paid origins.
Q15=A  max_tokens clamped per tier.
Q17=A  an admitted job finishes; overage carries into tomorrow.
Q22=A  the carry is computed from the ledger, one day, no column.
"""
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException

from crystal_cache.control.admission import (
    TIER_TABLE,
    clamp_max_tokens,
    daily_capacity,
    daily_capacity_block,
    enforce_managed_budget,
    enforce_managed_model,
)


class _Cust:
    def __init__(self, tier="free", mode="managed", cid="cus_spend"):
        self.id = cid
        self.subscription_tier = tier
        self.inference_mode = mode


def _spend_store(today: int, yesterday: int = 0, month: int | None = None):
    """A store whose ledger answers: [yesterday, midnight) -> yesterday,
    [midnight, now) -> today, anything earlier (the month window) ->
    month (defaults to today). On the 1st of a month the month window
    and the day window coincide, so tests that need the monthly wall
    set today == month."""
    class _S:
        async def platform_spend_micro_usd(self, cid, *, since, until=None):
            now = datetime.now(timezone.utc)
            midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
            if until is not None:
                return yesterday
            if since == midnight:
                return today
            return month if month is not None else today
    return _S()


# ---------------------------------------------------------------------------
# Tier table shape
# ---------------------------------------------------------------------------

def test_monthly_backstop_is_thirty_times_daily_everywhere():
    for name, t in TIER_TABLE.items():
        assert t.monthly_managed_budget_micro_usd == 30 * t.daily_managed_budget_micro_usd, name


def test_free_tier_has_no_opus_and_paid_tiers_do():
    assert "claude-opus-4-8" not in TIER_TABLE["free"].allowed_models
    assert {"claude-haiku-4-5", "claude-sonnet-5"} <= set(TIER_TABLE["free"].allowed_models)
    for name in ("starter_29", "scale_49_seat"):
        assert "claude-opus-4-8" in TIER_TABLE[name].allowed_models, name


def test_max_output_tokens_rise_with_tier():
    assert TIER_TABLE["free"].max_output_tokens < TIER_TABLE["starter_29"].max_output_tokens
    assert TIER_TABLE["starter_29"].max_output_tokens < TIER_TABLE["scale_49_seat"].max_output_tokens
    assert TIER_TABLE["free"].max_output_tokens <= 4_096


# ---------------------------------------------------------------------------
# daily_capacity: today + carried overage
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_capacity_states_at_the_free_allowance():
    free = TIER_TABLE["free"].daily_managed_budget_micro_usd  # $0.50
    for spent, state in ((0, "ok"), (free * 89 // 100, "ok"),
                         (free * 90 // 100, "warning"), (free - 1, "warning"),
                         (free, "blocked"), (free * 3, "blocked")):
        cap = await daily_capacity(_spend_store(spent), _Cust())
        assert cap["state"] == state, (spent, cap)
    assert cap["pct"] == 100


@pytest.mark.asyncio
async def test_yesterdays_overage_carries_into_today_once():
    """Q17=A + Q22=A: an ingest admitted at $0.49 that cost $1.00 overran
    by $0.50; today starts with that $0.50 already counted, so the next
    write is refused until midnight. Nothing from two days ago counts."""
    free = TIER_TABLE["free"].daily_managed_budget_micro_usd
    cap = await daily_capacity(_spend_store(today=0, yesterday=free * 2), _Cust())
    assert cap["carried"] == free
    assert cap["state"] == "blocked"
    cap = await daily_capacity(_spend_store(today=0, yesterday=free), _Cust())
    assert cap["carried"] == 0
    assert cap["state"] == "ok"


@pytest.mark.asyncio
async def test_unknown_tier_resolves_to_the_default_and_still_caps():
    cap = await daily_capacity(_spend_store(10**9), _Cust(tier="no_such_tier"))
    assert cap["state"] == "blocked"


# ---------------------------------------------------------------------------
# The door
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_door_refuses_managed_tenant_at_daily_allowance():
    free = TIER_TABLE["free"].daily_managed_budget_micro_usd
    # month defaults to today ($0.50), far under the $15 monthly backstop,
    # so the DAILY wall is the one that fires.
    with pytest.raises(HTTPException) as e:
        await enforce_managed_budget(_spend_store(free), _Cust())
    assert e.value.status_code == 429
    assert "Daily AI capacity" in e.value.detail
    assert "Remembering and recall keep working" in e.value.detail


@pytest.mark.asyncio
async def test_door_passes_below_allowance_and_never_touches_byok():
    free = TIER_TABLE["free"].daily_managed_budget_micro_usd
    await enforce_managed_budget(_spend_store(free - 1), _Cust())

    class _Boom:
        async def platform_spend_micro_usd(self, *a, **k):
            raise AssertionError("byok tenants never read the ledger")
    await enforce_managed_budget(_Boom(), _Cust(mode="byok"))


@pytest.mark.asyncio
async def test_monthly_backstop_counts_all_origins_and_fires_second():
    """Q14=A: a tenant inside every daily limit can only hit the monthly
    wall if the daily gate is broken, because 30 x daily == monthly.
    Here the ledger says the cap is spent; the monthly check runs first."""
    t = TIER_TABLE["starter_29"]
    cap = t.monthly_managed_budget_micro_usd
    with pytest.raises(HTTPException) as e:
        await enforce_managed_budget(
            _spend_store(today=cap, month=cap), _Cust("starter_29"),
        )
    assert "Monthly" in e.value.detail


@pytest.mark.asyncio
async def test_mcp_block_is_structured_and_names_the_reset():
    free = TIER_TABLE["free"].daily_managed_budget_micro_usd
    denied = await daily_capacity_block(_spend_store(free), _Cust())
    assert denied["code"] == "daily_capacity"
    assert "resets_at" in denied
    assert datetime.fromisoformat(denied["resets_at"]) > datetime.now(timezone.utc)
    assert await daily_capacity_block(_spend_store(0), _Cust()) is None
    assert await daily_capacity_block(_spend_store(free), _Cust(mode="byok")) is None


# ---------------------------------------------------------------------------
# Model policy and clamps
# ---------------------------------------------------------------------------

def test_free_tier_cannot_pick_opus_but_starter_can():
    with pytest.raises(HTTPException) as e:
        enforce_managed_model(_Cust("free"), "claude-opus-4-8")
    assert e.value.status_code == 400
    enforce_managed_model(_Cust("free"), "claude-sonnet-5")
    enforce_managed_model(_Cust("starter_29"), "claude-opus-4-8")
    enforce_managed_model(_Cust("free", mode="byok"), "anything-at-all")


def test_max_tokens_clamp_per_tier():
    assert clamp_max_tokens(_Cust("free"), 64_000) == TIER_TABLE["free"].max_output_tokens
    assert clamp_max_tokens(_Cust("free"), None) is None  # caller default applies
    assert clamp_max_tokens(_Cust("free"), 100) == 100
    assert clamp_max_tokens(_Cust("scale_49_seat"), 64_000) == TIER_TABLE["scale_49_seat"].max_output_tokens
    assert clamp_max_tokens(_Cust("free", mode="byok"), 64_000) == 64_000


# ---------------------------------------------------------------------------
# Wiring: every HTTP write surface and every MCP write tool passes the door
# ---------------------------------------------------------------------------

def test_http_write_wall_calls_the_spend_door():
    import inspect

    from crystal_cache.ingress import auth

    src = inspect.getsource(auth.require_write_capacity)
    assert "enforce_managed_budget" in src


def test_mcp_write_gate_calls_the_spend_door_and_synthesize_has_its_own():
    import inspect

    from crystal_cache.agent import mcp_server

    assert "daily_capacity_block" in inspect.getsource(mcp_server._write_admission_block)
    fn = mcp_server.memory_synthesize
    assert "_spend_admission_block" in inspect.getsource(getattr(fn, "fn", fn))


@pytest.mark.asyncio
async def test_plain_remember_passes_with_zero_capacity_but_ingest_does_not(
    store, customer, monkeypatch
):
    """Q30=A / Q31=A (2026-10-01): remembering is the promise and the
    fact cap bounds it; only the tools that run models pay the daily
    door. Proven through the real MCP gate with a store whose ledger
    says today's allowance is gone."""
    from crystal_cache.agent import mcp_server

    await store.set_customer_subscription(customer.id, "free", None)
    await store.set_customer_inference_mode(customer.id, "managed")
    monkeypatch.setattr(mcp_server, "_get_state", lambda: {"store": store})

    async def _facts(cid):
        return 0

    async def _spent(cid, *, since, until=None):
        # Today and yesterday exhausted; the month window stays under the
        # backstop so the DAILY door is the one under test (RC-10: the MCP
        # door checks the monthly backstop first, as the HTTP door does).
        midnight = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        return 10**9 if (until is not None or since == midnight) else 0

    monkeypatch.setattr(store, "count_billable_facts", _facts)
    monkeypatch.setattr(store, "platform_spend_micro_usd", _spent)
    token = mcp_server._current_customer_id.set(customer.id)
    try:
        assert await mcp_server._write_admission_block() is None
        assert await mcp_server._write_admission_block(spend=False) is None
        denied = await mcp_server._write_admission_block(spend=True)
    finally:
        mcp_server._current_customer_id.reset(token)
    assert denied is not None
    assert denied["code"] == "daily_capacity"


def test_which_mcp_tools_pay_the_daily_door():
    """Pinned by source: ingest and import opt in; store, remember,
    learn, forget and record_gap never do."""
    import inspect

    from crystal_cache.agent import mcp_server

    def src(name):
        fn = getattr(mcp_server, name)
        return inspect.getsource(getattr(fn, "fn", fn))

    for name in ("memory_ingest", "memory_import"):
        assert "_write_admission_block(spend=True)" in src(name), name
    for name in ("memory_store", "remember", "memory_learn", "memory_record_gap"):
        assert "_write_admission_block()" in src(name), name
        assert "spend=True" not in src(name), name
    # RC-05 (2026-10-04): deletes call NO wall at all; forgetting is always
    # possible, including over the cap and past the trial.
    for name in ("memory_forget", "forget"):
        assert "_write_admission_block" not in src(name), name


@pytest.mark.asyncio
async def test_http_store_wall_only_pays_the_daily_door_with_spend(
    store, customer, monkeypatch
):
    from crystal_cache.ingress.auth import require_write_capacity

    await store.set_customer_subscription(customer.id, "free", None)
    await store.set_customer_inference_mode(customer.id, "managed")
    c = await store.get_customer_by_id(customer.id)

    async def _facts(cid):
        return 0

    async def _spent(cid, *, since, until=None):
        # Today and yesterday are exhausted; the month window stays under
        # the backstop so the DAILY wall is the one under test.
        midnight = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        return 10**9 if (until is not None or since == midnight) else 0

    monkeypatch.setattr(store, "count_billable_facts", _facts)
    monkeypatch.setattr(store, "platform_spend_micro_usd", _spent)
    await require_write_capacity(c, store)  # plain store: passes
    with pytest.raises(HTTPException) as e:
        await require_write_capacity(c, store, spend=True)  # ingest: walled
    assert e.value.status_code == 429
    assert e.value.headers["X-Plan-Wall"] == "daily_capacity"


@pytest.mark.asyncio
async def test_every_plan_wall_carries_its_code_in_a_header():
    """Q28 groundwork: the console reads X-Plan-Wall to open the right
    upgrade modal; the detail stays the human message."""
    free = TIER_TABLE["free"].daily_managed_budget_micro_usd
    with pytest.raises(HTTPException) as e:
        await enforce_managed_budget(_spend_store(free), _Cust())
    assert e.value.headers["X-Plan-Wall"] == "daily_capacity"
    assert isinstance(e.value.detail, str)

    cap = TIER_TABLE["starter_29"].monthly_managed_budget_micro_usd
    with pytest.raises(HTTPException) as e:
        await enforce_managed_budget(_spend_store(today=cap, month=cap), _Cust("starter_29"))
    assert e.value.headers["X-Plan-Wall"] == "monthly_budget"

    with pytest.raises(HTTPException) as e:
        enforce_managed_model(_Cust("free"), "claude-opus-4-8")
    assert e.value.headers["X-Plan-Wall"] == "model_not_in_plan"
    assert e.value.status_code == 400


def test_daily_capacity_copy_names_ingest_and_keeps_remembering():
    from crystal_cache.control.admission import DAILY_CAPACITY_MESSAGE

    assert "ingesting documents" in DAILY_CAPACITY_MESSAGE
    assert "Remembering and recall keep working" in DAILY_CAPACITY_MESSAGE


def test_proxy_and_agent_call_the_door_before_any_model_call():
    import inspect
    import re

    from crystal_cache.endpoints import agent, chat_proxy

    for mod in (chat_proxy, agent):
        src = inspect.getsource(mod)
        assert "enforce_managed_budget(store, customer)" in src, mod.__name__
        assert re.search(r"clamp_max_tokens\(\s*customer", src), mod.__name__


def test_consolidate_and_feedback_are_rate_limited_and_gated():
    import inspect

    from crystal_cache.endpoints import feedback, sdk
    from crystal_cache.ingress import rate_limit

    assert "/v1/consolidate" in rate_limit._EXPENSIVE_PREFIXES
    assert "/v1/feedback" in rate_limit._EXPENSIVE_PREFIXES
    assert "enforce_managed_budget(store, customer)" in inspect.getsource(sdk.sdk_consolidate)
    assert "enforce_managed_budget(store, customer)" in inspect.getsource(feedback.submit_feedback)


@pytest.mark.asyncio
async def test_memory_import_refuses_oversized_batches(monkeypatch):
    from crystal_cache.agent import mcp_server

    async def _open(**kwargs):
        return None

    monkeypatch.setattr(mcp_server, "_write_admission_block", _open)
    fn = getattr(mcp_server.memory_import, "fn", mcp_server.memory_import)
    out = await fn(
        records=[{"key": f"k{i}", "value": "v"} for i in range(mcp_server.MEMORY_IMPORT_MAX_RECORDS + 1)],
    )
    assert out["code"] == "import_too_large"
    assert out["max_records"] == mcp_server.MEMORY_IMPORT_MAX_RECORDS


def test_cognition_attempts_are_clamped_at_both_ends():
    import inspect

    from crystal_cache.agent.tools import cognition as tool
    from crystal_cache.workers import cognition as worker

    assert tool.COGNITION_MAX_ATTEMPTS == 3
    assert "COGNITION_MAX_ATTEMPTS" in inspect.getsource(tool)
    assert "COGNITION_MAX_ATTEMPTS" in inspect.getsource(worker)


def test_shadow_critic_uses_the_real_usage_method_and_its_configured_model():
    import inspect

    from crystal_cache.agent import shadow_critic

    src = inspect.getsource(shadow_critic)
    assert 'getattr(client, "complete_detailed", None)' in src
    assert "model=chosen_model" in src
