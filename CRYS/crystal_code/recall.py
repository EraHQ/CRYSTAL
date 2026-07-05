"""Pre-turn knowledge recall — CRYS consulting its OWN accumulated bank.

The library agent loop (`crystal_cache.agent.Agent.run`) has no implicit
retrieval the way the proxy's `retrieve_v3` does: a crystal only reaches
the model when the agent calls a search tool. In practice CRYS wrote
knowledge prolifically but rarely read it back DURING work (the
2026-06-15 "Realm of Aethermoor" sessions — it re-researched patterns it
had already saved). This step closes the accumulate->reuse loop WITHOUT
taking the decision away from the agent.

Before a turn, a fast model is asked one thing: given the user's
message, would consulting the agent's own bank help, and if so what
should it search for? It answers in its own words and may decline ("no,
I need different / fresh information"). When it recalls, the top bank
hits ride in THAT turn's system prompt as reusable context — retrieved
via `knowledge_search`, so the result (including the general-bank +
Reflections merge) is exactly what the agent would get if it searched
itself. The agent still owns the query; this just makes sure the
question gets ASKED.

Fully fail-safe: no client, a malformed reply, an empty bank, any
exception — all yield no recall. A bonus pass must never break or stall
a turn.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)

RECALL_DECISION_MAX_TOKENS = 150

RECALL_DECISION_SYSTEM = """You decide whether the agent should consult its OWN accumulated knowledge bank before acting on the user's message, and if so, what to search for.

The bank is the agent's MEMORY of past work — patterns and lessons it saved itself, keyed like "General|Game Development|Movement Systems|Delta Time Pattern" or "Reflections|<project>|<lesson>". It is NOT the current project's code or documents (separate tools handle those).

Reply with ONLY a JSON object, one of:
{"recall": true, "query": "<short search phrase, in your OWN words, for the patterns or lessons most relevant to this task>"}
{"recall": false}

recall=true when the task resembles work the agent may have done before — implementing a feature, fixing a class of bug, a design or balance decision — where a saved pattern or lesson would help.
recall=false for trivial or conversational messages, questions about THIS project's specific files (other tools handle those), or tasks needing fresh/external information the bank wouldn't hold.
The query is yours to shape — search for what would actually help, not necessarily the user's exact words. Reply {"recall": false} freely; a needless recall just wastes a step."""


async def maybe_recall(
    *,
    agent: Any,
    user_input: str,
    fast_model: str,
) -> Optional[str]:
    """Decide-then-recall against the agent's own bank for one turn.

    Returns a system-prompt block of recalled knowledge to append for
    THIS turn, or None to skip. Never raises — recall is a bonus pass.
    """
    try:
        client = getattr(agent, "llm", None)
        if client is None:
            return None

        # 1. The gate + the query — the agent's own judgment, one cheap
        #    fast-model call with a tiny output budget. Runs through the
        #    provider-neutral seam on the agent's own client; temperature
        #    1.0 preserves the historical API-default sampling.
        raw = await asyncio.to_thread(
            client.complete,
            system=RECALL_DECISION_SYSTEM,
            messages=[{"role": "user", "content": user_input[:2000]}],
            max_tokens=RECALL_DECISION_MAX_TOKENS,
            temperature=1.0,
            model=fast_model,
        )
        raw = raw.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        start = raw.find("{")
        if start < 0:
            return None
        parsed, _ = json.JSONDecoder().raw_decode(raw[start:])
        if not parsed.get("recall"):
            logger.info("recall.declined")
            return None
        query = " ".join(str(parsed.get("query", "")).split()).strip()
        if not query:
            return None

        # 2. Reuse the agent's own knowledge_search so retrieval (incl. the
        #    general-bank merge + Reflections) matches what the agent would
        #    get if it searched itself. injection_text is the composed,
        #    length-bounded result the proxy/agent already use.
        from crystal_cache.agent.tool_registry import get_registry

        tool = get_registry().get("knowledge_search")
        if tool is None:
            return None
        out = await tool.impl(customer_id=agent.customer.id, query=query)
        injection = (out or {}).get("injection_text")
        if not injection or not str(injection).strip():
            logger.info("recall.empty", query=query)
            return None

        logger.info("recall.injected", query=query, chars=len(str(injection)))
        return (
            "\n\nRECALLED KNOWLEDGE — from your OWN bank (you saved these on "
            f'past work; searched for "{query}"). Reuse what genuinely '
            "applies; ignore anything that doesn't fit this task:\n"
            f"{injection}"
        )
    except Exception as e:  # noqa: BLE001 — recall is a bonus pass, never fatal
        logger.warning("recall.failed", error=f"{type(e).__name__}: {e}")
        return None
