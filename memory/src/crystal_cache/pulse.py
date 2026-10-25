"""The Command center's pulse (2026-10-24, Q110 B). Anthony: "I want to
think of cool stats and graphs we can show for the system as a whole."
One read, six panels and a heatmap, all from data Crystal already keeps:

  growth     facts in the bank on each of the last 30 days, and the week's change
  recall     questions per day (14 days) and the share that found memory
  autonomy   what Crystal did on its own per day (14 days), by kind
  health     facts by trust tier, open conflicts, open gaps
  goals      goal items handled this week, and email reply time (arrived ->
             drafted, arrived -> sent), medians in minutes
  capacity   today's AI capacity used, a percent (never dollars; null when
             the tenant brings its own key)
  heatmap    84 days of activity: writes, recalls, autonomous acts
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from statistics import median
from typing import Any, Optional

# What the ledger's action names mean on the chart.
AUTONOMY_KIND = {
    "decide_conflict": "conflicts", "reopen_conflict": "conflicts",
    "approve_assumption": "assumptions", "reject_assumption": "assumptions", "verify_assumption": "assumptions",
    "rule_approval": "assumptions",
    "research_gap": "gaps", "dismiss_gap": "gaps", "answer_gap": "gaps", "reopen_gap": "gaps",
    "edit_fact": "facts", "save_fact": "facts", "lattice_write": "facts", "set_tier": "facts", "retire_fact": "facts", "reprobe_fact": "facts",
    "send_email": "goals", "post_message": "goals", "create_event": "goals", "append_page": "goals",
}
AUTONOMY_KINDS = ("conflicts", "assumptions", "gaps", "facts", "goals")


def _days(n: int, today: date) -> list[str]:
    return [(today - timedelta(days=i)).isoformat() for i in range(n - 1, -1, -1)]


def _minutes(a: Optional[datetime], b: Optional[datetime]) -> Optional[float]:
    if a is None or b is None:
        return None
    if a.tzinfo is None:
        a = a.replace(tzinfo=timezone.utc)
    if b.tzinfo is None:
        b = b.replace(tzinfo=timezone.utc)
    m = (b - a).total_seconds() / 60.0
    return m if m >= 0 else None


async def build_pulse(store: Any, customer: Any, *, now: Optional[datetime] = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    today = now.date()
    start84 = datetime.combine(today - timedelta(days=83), datetime.min.time(), tzinfo=timezone.utc)
    week_ago = now - timedelta(days=7)
    cid = customer.id

    facts_daily = await store.daily_fact_counts(cid, start84)
    total = await store.fact_total(cid)
    recalls = await store.daily_recall_counts(cid, start84)
    acts = await store.daily_autonomous_acts(cid, start84)

    # growth: walk back from today's total, removing each later day's writes
    days30 = _days(30, today)
    growth, running = [], total
    for d in reversed(days30):
        growth.append({"day": d, "facts": max(0, running)})
        running -= facts_daily.get(d, 0)
    growth.reverse()
    week_added = sum(facts_daily.get(d, 0) for d in _days(7, today))

    days14 = _days(14, today)
    recall = [{"day": d, "asked": recalls.get(d, {}).get("asked", 0), "found": recalls.get(d, {}).get("found", 0)}
              for d in days14]
    asked14 = sum(r["asked"] for r in recall)
    found14 = sum(r["found"] for r in recall)

    by_day: dict[str, dict[str, int]] = {}
    acts_per_day: dict[str, int] = {}
    for d, action, n in acts:
        kind = AUTONOMY_KIND.get(action, "facts")
        by_day.setdefault(d, {k: 0 for k in AUTONOMY_KINDS})[kind] += n
        acts_per_day[d] = acts_per_day.get(d, 0) + n
    autonomy = [{"day": d, **by_day.get(d, {k: 0 for k in AUTONOMY_KINDS})} for d in days14]

    tiers = await store.facts_by_tier(cid)
    open_conflicts = await store.count_knowledge_conflicts(cid, status="open")
    open_gaps = len(await store.list_knowledge_gaps(cid, status="open", limit=1000))

    items = await store.goal_items_since(cid, week_ago)
    handled = {"sent": 0, "done": 0, "dismissed": 0, "drafted": 0, "filed": 0}
    for it in items:
        if it["status"] in ("sent", "done", "dismissed"):
            handled[it["status"]] += it["count"]
        if it["kind"] == "draft":
            handled["drafted"] += it["count"]
        elif it["kind"] == "action":
            handled["filed"] += it["count"]
    pairs = await store.reply_pairs(cid, week_ago)
    to_draft = [m for p in pairs if (m := _minutes(p["arrived"], p["drafted"])) is not None]
    to_sent = [m for p in pairs if (m := _minutes(p["arrived"], p["sent"])) is not None]

    capacity: Optional[float] = None
    if getattr(customer, "inference_mode", "byok") == "managed":
        try:
            from .control.admission import daily_capacity
            capacity = (await daily_capacity(store, customer))["pct"]
        except Exception:  # noqa: BLE001 — a meter hiccup never blanks the page
            capacity = None

    heat = [{"day": d, "writes": facts_daily.get(d, 0), "recalls": recalls.get(d, {}).get("asked", 0),
             "acts": acts_per_day.get(d, 0)} for d in _days(84, today)]

    return {
        "as_of": now.isoformat(),
        "growth": {"days": growth, "total": total, "week_added": week_added},
        "recall": {"days": recall, "asked": asked14, "found": found14,
                   "found_pct": round(100.0 * found14 / asked14, 1) if asked14 else None},
        "autonomy": {"days": autonomy, "kinds": list(AUTONOMY_KINDS),
                     "week": sum(acts_per_day.get(d, 0) for d in _days(7, today))},
        "health": {"tiers": tiers, "open_conflicts": int(open_conflicts), "open_gaps": open_gaps},
        "goals": {"week": handled, "replies": len(pairs),
                  "median_to_draft_minutes": round(median(to_draft), 1) if to_draft else None,
                  "median_to_sent_minutes": round(median(to_sent), 1) if to_sent else None},
        "capacity": {"pct": capacity},
        "heatmap": heat,
    }
