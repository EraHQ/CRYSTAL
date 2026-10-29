"""What a goal needs from the person, as questions (Series 386, Q154 C,
Q155, 2026-10-28).

2026-10-28 the Sports goal's "needs you" box held 18 reports of one
problem (pro-football-reference.com blocked), each worded differently,
with nowhere to answer. Anthony picked a question with answers to pick
(option C) and: "only three open problems per goal. At three the next
checkin cant run without resolving them all."

A problem is a goal item of kind note whose target holds `problem`:
{kind, about, step, question, options, recommended, count, first_seen,
last_seen}. Two reports are the same problem when they share step, kind
and what they are about (a site's domain, an app's name): the second
raises the count on the first, whatever its wording. At MAX_OPEN open
problems the goal waits: no check-in runs until the person has answered
or handled every one.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

KINDS = ("blocked_source", "app", "permission", "choice", "other")
KIND_WORDS = {"blocked_source": "Blocked source", "app": "App to connect", "permission": "Permission",
              "choice": "Your choice", "other": "Needs you"}
MAX_OPEN = 3
MAX_OPTIONS = 4


def about_key(about: Any) -> str:
    """What a problem is about, comparable: a URL or domain becomes its
    host without www; anything else lowercased words."""
    s = str(about or "").strip().lower()
    if not s:
        return ""
    host = urlsplit(s if "//" in s else f"//{s}").hostname or ""
    if host and "." in host and " " not in s:
        return host.removeprefix("www.")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()[:120]


def key(step: Any, kind: str, about: Any) -> str:
    try:
        n = int(step) if step not in (None, "") else 0
    except (TypeError, ValueError):
        n = 0
    return f"{n}|{kind}|{about_key(about)}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def fold_legacy(store: Any, customer_id: str, goal_id: str) -> None:
    """Reports from before Series 386 have no problem attached: they become
    one card (the newest, counting the rest), the others are dismissed."""
    items = await store.list_goal_items(customer_id, goal_id, status="open", kind="note", limit=200)
    legacy = [i for i in items if not (i.get("target") or {}).get("problem")]
    if not legacy:
        return
    legacy.sort(key=lambda i: i.get("created_at") or "", reverse=True)
    keep = legacy[0]
    problem = {"kind": "other", "about": "", "step": 0, "key": "legacy", "question": keep["title"][:300],
               "options": [], "recommended": -1, "count": len(legacy),
               "first_seen": legacy[-1].get("created_at") or _now(), "last_seen": keep.get("created_at") or _now()}
    await store.update_goal_item(customer_id, keep["id"], target={**(keep.get("target") or {}), "problem": problem})
    for i in legacy[1:]:
        await store.update_goal_item(customer_id, i["id"], status="dismissed")


async def open_problems(store: Any, customer_id: str, goal_id: str) -> list[dict[str, Any]]:
    await fold_legacy(store, customer_id, goal_id)
    items = await store.list_goal_items(customer_id, goal_id, status="open", kind="note", limit=200)
    return [i for i in items if (i.get("target") or {}).get("problem")]


async def report(store: Any, customer_id: str, goal: dict[str, Any], *, need: str, kind: str, about: str,
                 step: Any, question: str, options: list[str], recommended: int) -> dict[str, Any]:
    """File a problem, or count it on the open one it repeats."""
    kind = kind if kind in KINDS else "other"
    k = key(step, kind, about)
    open_ = await open_problems(store, customer_id, goal["id"])
    for item in open_:
        p = dict(item["target"]["problem"])
        if p.get("key") == k:
            p.update(count=int(p.get("count") or 1) + 1, last_seen=_now(), last_need=need[:500])
            await store.update_goal_item(customer_id, item["id"], target={**item["target"], "problem": p})
            return {"ok": True, "item_id": item["id"], "same_as_open": True, "count": p["count"],
                    "note": "The person already has this question open; it was counted, not asked again. Work on "
                            "something else until they answer."}
    if len(open_) >= MAX_OPEN:
        return {"ok": False, "code": "too_many_open",
                "error": f"{MAX_OPEN} questions already wait for the person; no more until they answer."}
    opts = [str(o).strip()[:200] for o in options or [] if str(o).strip()][:MAX_OPTIONS]
    problem = {"kind": kind, "about": str(about or "").strip()[:200], "step": int(k.split("|")[0]), "key": k,
               "question": (question or need).strip()[:300], "options": opts,
               "recommended": recommended if isinstance(recommended, int) and 0 <= recommended < len(opts) else -1,
               "count": 1, "first_seen": _now(), "last_seen": _now()}
    row = await store.add_goal_item(customer_id, goal["id"], kind="note", title=need[:500], body="",
                                    target={"problem": problem})
    held = len(open_) + 1 >= MAX_OPEN
    return {"ok": True, "item_id": row["id"], "open": len(open_) + 1,
            **({"note": f"That is {MAX_OPEN} open questions: check-ins pause until the person answers all of them."}
               if held else {})}


def waiting(goal: dict[str, Any]) -> bool:
    return bool((goal.get("plan") or {}).get("waiting_on_you"))


def open_text(problems: list[dict[str, Any]]) -> str:
    """For the check-in: what is already asked, so it isn't asked again."""
    if not problems:
        return ""
    lines = []
    for i in problems:
        p = i["target"]["problem"]
        step = f"step {p['step']}, " if p.get("step") else ""
        lines.append(f"- {p['question']} ({step}{KIND_WORDS.get(p['kind'], p['kind']).lower()}"
                     f"{', about ' + p['about'] if p.get('about') else ''})")
    return ("\n\nQUESTIONS ALREADY WAITING FOR THE PERSON. Never ask these again (report_stuck on the same step, kind "
            f"and subject only counts it); work on anything else until they answer. At {MAX_OPEN} open questions "
            "check-ins pause:\n" + "\n".join(lines))
