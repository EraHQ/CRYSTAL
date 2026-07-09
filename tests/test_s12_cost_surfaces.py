"""S12 (2026-07-09): origin breakdown, shadow budget block, cache columns."""
from datetime import datetime, timedelta, timezone

import pytest

from crystal_cache.models import QueryLog


async def test_cost_by_origin_groups_and_orders(store, customer):
    from crystal_cache.cost.pricing import ModelPrice
    # price a known model so costs are non-zero and ordering is real:
    # interactive gets two calls (larger output), cognition one small.
    table = {"m": ModelPrice(
        input_micro_per_mtok=3_000_000,
        output_micro_per_mtok=15_000_000,
        cache_creation_micro_per_mtok=3_750_000,
        cache_read_micro_per_mtok=300_000,
    )}
    for origin, out_toks in (("interactive", 1000), ("shadow_critic", 200),
                             ("interactive", 600), ("cognition", 100)):
        await store.record_llm_call(
            customer.id, origin=origin, model="m",
            input_tokens=10, output_tokens=out_toks,
            cache_creation_tokens=2, cache_read_tokens=8,
            price_table=table,
        )
    rows = await store.cost_by_origin(customer.id)
    assert rows[0]["origin"] == "interactive"
    assert rows[0]["call_count"] == 2
    assert rows[0]["cache_read_tokens"] == 16
    assert rows[0]["cost_micro_usd"] > rows[1]["cost_micro_usd"]
    origins = {r["origin"] for r in rows}
    assert origins == {"interactive", "shadow_critic", "cognition"}


async def test_cost_by_origin_since_window(store, customer):
    await store.record_llm_call(
        customer.id, origin="task", model="m",
        input_tokens=1, output_tokens=1,
    )
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    assert await store.cost_by_origin(customer.id, since=future) == []
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    rows = await store.cost_by_origin(customer.id, since=past)
    assert rows and rows[0]["origin"] == "task"


async def test_query_log_cache_tokens_roundtrip(store, customer):
    log = QueryLog(
        id="ql_s12test0000000",
        customer_id=customer.id,
        query_text="q",
        match_type="none",
        injection_method="agent_tools",
        prompt_tokens=20,
        completion_tokens=40,
        cache_creation_tokens=1200,
        cache_read_tokens=9000,
    )
    await store.write_query_log(log)
    _total, logs = await store.list_query_logs_for_customer(
        customer.id, limit=5)
    row = next(l for l in logs if l.id == "ql_s12test0000000")
    assert row.cache_creation_tokens == 1200
    assert row.cache_read_tokens == 9000
