"""The Command center's pulse (Q110 B, 2026-10-24): growth, recall, what
Crystal did on its own, bank health, goals at work with email reply time,
and an 84-day heatmap, all from data already kept. A person's own clicks
are not autonomy; days with nothing are zeros, never missing.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from crystal_cache.infrastructure.schema import CrystalRow, FactRow, QueryLogRow
from crystal_cache.pulse import build_pulse


@pytest.mark.asyncio
async def test_the_pulse_counts_what_happened(store, customer):
    cid = customer.id
    now = datetime.now(timezone.utc).replace(hour=15, minute=0, second=0, microsecond=0)
    async with store.session() as s:
        s.add(CrystalRow(id="c_w", customer_id=cid, crystal_type="customer:legacy", summary_vector=[], quality_tier="whitelist"))
        s.add(CrystalRow(id="c_q", customer_id=cid, crystal_type="customer:legacy", summary_vector=[], quality_tier="quarantine"))
        for i, (cr, days_ago) in enumerate([("c_w", 0), ("c_w", 0), ("c_w", 3), ("c_q", 10), ("c_q", 40)]):
            s.add(FactRow(id=f"f{i}", crystal_id=cr, pair_type="question_answer", prompt_text="A|B", claim_text=f"claim {i}",
                          source_kind="document_chunk", created_at=now - timedelta(days=days_ago)))
        for i, (mt, days_ago) in enumerate([("high", 0), ("low", 0), ("none", 0), ("high", 2)]):
            s.add(QueryLogRow(id=f"q{i}", customer_id=cid, query_text="?", match_type=mt, timestamp=now - timedelta(days=days_ago)))
    # Autonomy: the Steward and a scan count; a person in chat does not.
    for actor, action in (("steward", "decide_conflict"), ("system", "decide_conflict"), ("steward", "research_gap"),
                          ("goal:goal_x", "send_email"), ("chat", "dismiss_gap"), ("console", "send_email")):
        await store.record_steward_action(cid, run_id=None, action=action, subject_kind="x", subject_id="y", label="l",
                                          reason="because it was", inverse=None, undoable=False, actor=actor)
    await store.record_curation_event(cid, event_type="memory_allowed", subject_id="c_q", label="rule")
    # Goals: one email drafted 30 minutes after it arrived and sent 50 minutes after.
    conn = await store.upsert_app_connection(cid, integration="google-mail", nango_connection_id="nc")
    wi = await store.upsert_watched_item(cid, conn["id"], kind="email", external_id="m1", title="Hi",
                                         occurred_at=datetime.now(timezone.utc) - timedelta(minutes=30))
    g = await store.create_goal(cid, title="Inbox", directive="Reply within 3 hours.")
    d = await store.add_goal_item(cid, g["id"], kind="draft", title="Re: Hi", body="Yes.", watched_item_id=wi["id"])
    await store.update_goal_item(cid, d["id"], status="sent")
    await store.add_goal_item(cid, g["id"], kind="action", title="Call")

    out = await build_pulse(store, customer, now=now)
    growth = out["growth"]
    assert growth["total"] == 5 and growth["week_added"] == 3 and len(growth["days"]) == 30
    assert growth["days"][-1]["facts"] == 5 and growth["days"][-2]["facts"] == 3  # today's two came today
    assert growth["days"][0]["facts"] == 1  # 29 days ago: only the 40-day-old fact
    assert out["recall"]["asked"] == 4 and out["recall"]["found"] == 3 and out["recall"]["found_pct"] == 75.0
    assert len(out["recall"]["days"]) == 14 and out["recall"]["days"][0] == {"day": out["recall"]["days"][0]["day"], "asked": 0, "found": 0}
    week = {k: sum(day[k] for day in out["autonomy"]["days"]) for k in out["autonomy"]["kinds"]}
    assert week == {"conflicts": 2, "assumptions": 1, "gaps": 1, "facts": 0, "goals": 1}
    assert out["autonomy"]["week"] == 5
    assert out["health"]["tiers"] == {"whitelist": 3, "quarantine": 2}
    assert out["health"]["open_conflicts"] == 0 and out["health"]["open_gaps"] == 0
    assert out["goals"]["week"]["sent"] == 1 and out["goals"]["week"]["drafted"] == 1 and out["goals"]["week"]["filed"] == 1
    assert out["goals"]["replies"] == 1 and 25 <= out["goals"]["median_to_draft_minutes"] <= 35
    assert out["goals"]["median_to_sent_minutes"] is not None
    assert out["capacity"]["pct"] is None or isinstance(out["capacity"]["pct"], (int, float))
    heat = out["heatmap"]
    assert len(heat) == 84 and heat[-1]["writes"] == 2 and heat[-1]["recalls"] == 3 and heat[-1]["acts"] == 5


@pytest.mark.asyncio
async def test_an_empty_bank_reads_as_zeros(store, customer):
    out = await build_pulse(store, customer)
    assert out["growth"]["total"] == 0 and all(d["facts"] == 0 for d in out["growth"]["days"])
    assert out["recall"]["found_pct"] is None
    assert out["goals"]["median_to_draft_minutes"] is None
    assert sum(h["writes"] + h["recalls"] + h["acts"] for h in out["heatmap"]) == 0
