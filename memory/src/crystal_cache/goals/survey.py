"""The survey: research before a plan (Q138 A, 2026-10-27).

The Sports Predictions plan was sized from the planner's head: "~22K
games over 3 seasons" (about 3,900), "10 players per game" (22 to 26),
"12M player-games". Anthony: "A quick research about the questions it has
would answer all those questions." Before the frontier tier writes a
plan, the large tier lists the questions whose answers size and shape it
(how many records, how many participants each, which pages list them and
how those pages are split, which ids they carry) and answers each from
the web with a capped budget. The findings go to the planner as measured
facts (read on a page this survey, with its URL) or estimates, and the
step criteria take their targets from them. Kept on the plan (survey) so
the person sees what the plan was built on; a replan within a week reuses
it.
"""
from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)

SURVEY_TIER = "large"
SURVEY_ORIGIN = "goal_plan"
MAX_FINDINGS = 16
REUSE_FOR = timedelta(days=7)
_JSON = re.compile(r"\{.*\}", re.S)


def survey_system(goal: dict[str, Any], budget: int) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"""You research the facts a plan is sized and shaped on, before the plan is written. TODAY IS {today} (UTC). Nobody is watching; you only read.

The goal: {goal['title']}
In the person's words: {(goal.get('directive') or '')[:2000]}

1. List the questions a planner must answer with real numbers to plan this goal: how many records exist in the scope the person asked for (games in each season, products in a catalog, messages a week), how many participants or line items each record has (players in a game's box score), which public pages list the records many at a time, how those pages are split (by month, week, letter, page number) and how many there are, which ids the pages carry for each record, and which pages hold the details.
2. Answer each one from the pages themselves: web_search to find the source, web_tables to read a listing (it gives each table's row_count and related_pages, the other months or pages of the same listing), web_fetch for prose. Count from the page whenever you can; a count you read beats one you remember. Check one detail page to count its participants.
3. Budget: {budget} tool calls. Stop when the sizing questions are answered.

End with JSON only, no prose around it:
{{"findings": [{{"question": "...", "answer": "one or two sentences", "number": a number or null, "source": "the page's URL or empty", "measured": true only when you read it on a page in this survey}}]}}"""


def _parse(text: str) -> list[dict[str, Any]]:
    m = _JSON.search(text or "")
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return []
    out = []
    for f in (data.get("findings") if isinstance(data, dict) else None) or []:
        if not isinstance(f, dict) or not str(f.get("question") or "").strip():
            continue
        num = f.get("number")
        try:
            num = float(num) if num is not None and not isinstance(num, bool) else None
            if num is not None and num.is_integer():
                num = int(num)
        except (TypeError, ValueError):
            num = None
        source = str(f.get("source") or "").strip()[:500]
        out.append({"question": str(f["question"]).strip()[:300], "answer": str(f.get("answer") or "").strip()[:600],
                    "number": num, "source": source,
                    # Measured needs a page behind it.
                    "measured": bool(f.get("measured")) and source.startswith(("http://", "https://"))})
        if len(out) >= MAX_FINDINGS:
            break
    return out


def survey_text(survey: Optional[dict[str, Any]]) -> str:
    """The findings as the planner reads them."""
    findings = (survey or {}).get("findings") or []
    if not findings:
        return ""
    lines = []
    for f in findings:
        tag = f"measured, {f['source']}" if f.get("measured") else "estimated"
        lines.append(f"- {f['question']} -> {f['answer']} ({tag})")
    return ("\n\nThe survey (research done for this plan just now; size the plan and set criteria targets on these "
            "numbers; a measured number beats your own estimate, and any number you add yourself is an estimate):\n"
            + "\n".join(lines))


def reusable(previous: Optional[dict[str, Any]], directive: str) -> Optional[dict[str, Any]]:
    s = (previous or {}).get("survey") or None
    if not s or s.get("directive") != directive[:4000]:
        return None
    try:
        at = datetime.fromisoformat(s["at"])
    except (KeyError, TypeError, ValueError):
        return None
    return s if datetime.now(timezone.utc) - at < REUSE_FOR else None


def _registry(store: Any, encoder: Any, fact_store: Any):
    from ..agent.tool_registry import Tool, ToolRegistry, get_registry, import_all_tools
    from ..cognition.agentic import _WORKER_TOOLS
    from ..cognition.retrieval_adapter import dispatch_cognition_retrieval

    registry = ToolRegistry()
    web = get_registry()
    if web.get("web_search") is None:
        import_all_tools()
        web = get_registry()
    for name in ("web_search", "web_fetch", "web_tables"):
        spec = web.get(name)
        if spec is not None:
            registry.register(Tool(name=name, description=spec.description, contexts=frozenset({"agent"}),
                                   parameters_schema=spec.parameters_schema, impl=spec.impl,
                                   returns_description=spec.returns_description or ""))
    if encoder is not None and fact_store is not None:
        for spec in _WORKER_TOOLS:
            if spec["name"] != "crystal_search":
                continue

            async def _impl(customer_id: str, _a=spec["action"], **kwargs: Any) -> dict:
                return await dispatch_cognition_retrieval(action_value=_a, step_input=kwargs, customer_id=customer_id,
                                                          store=store, fact_store=fact_store, encoder=encoder)
            registry.register(Tool(name=spec["name"], description=spec["description"], contexts=frozenset({"agent"}),
                                   parameters_schema=spec["schema"], impl=_impl, returns_description=""))
    return registry


async def run_survey(store: Any, customer: Any, goal: dict[str, Any], *, llm: Any, budget: int, wall: float,
                     encoder: Any = None, fact_store: Any = None, run_id: Optional[str] = None) -> dict[str, Any]:
    """{at, directive, findings, calls, stop_reason}. Never raises: a
    survey that fails leaves the planner to estimate (and say so)."""
    from ..agent.agent import Agent
    from ..cost.emit import record_model_call

    started = datetime.now(timezone.utc)
    usage: dict[str, Any] = {}
    findings: list[dict[str, Any]] = []
    calls, stop = 0, None
    model = None
    try:
        try:
            model = llm._resolve_model(SURVEY_TIER, None) if hasattr(llm, "_resolve_model") else None
        except Exception:  # noqa: BLE001
            model = None
        agent = Agent(customer=SimpleNamespace(id=customer.id), llm=llm,
                      tool_state={"store": store, "fact_vector_store": fact_store, "encoder": encoder},
                      model=model, max_tokens=3000, max_iterations=budget, registry=_registry(store, encoder, fact_store))
        run = await asyncio.wait_for(
            agent.run(messages=[{"role": "user", "content": "Survey this goal, then answer with the JSON."}],
                      system=survey_system(goal, budget), deadline_seconds=wall, usage_sink=usage,
                      default_tool_timeout=60.0, model_timeout=150.0),
            timeout=wall + 60.0)
        findings = _parse(run.get("final_text") or "")
        calls = len(run.get("tool_calls") or [])
        stop = run.get("stop_reason")
    except Exception as e:  # noqa: BLE001
        stop = "error"
        logger.warning("goals.survey_failed", goal_id=goal["id"], error_type=type(e).__name__, error=str(e)[:200])
    pt, ct = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
    if pt or ct:
        try:
            await record_model_call(customer_id=customer.id, model=usage.get("model") or model or "unknown",
                                    input_tokens=pt, output_tokens=ct, origin=SURVEY_ORIGIN, session_id=run_id,
                                    store=store)
        except Exception:  # noqa: BLE001
            logger.debug("goals.survey_metering_failed", exc_info=True)
    logger.info("goals.surveyed", goal_id=goal["id"], findings=len(findings), calls=calls, stop_reason=stop,
                seconds=round((datetime.now(timezone.utc) - started).total_seconds(), 1))
    return {"at": started.isoformat(), "directive": (goal.get("directive") or "")[:4000], "findings": findings,
            "calls": calls, "stop_reason": stop}
