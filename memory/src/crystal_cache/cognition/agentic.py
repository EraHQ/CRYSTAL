"""Workers-as-CRYS — agentic composition steps (ratified 2026-07-13).

Rematch #9 closed the deterministic arc: every remaining failure was
DECISION-shaped — a wrong-org 404 nobody could react to, an empty
releases list nobody could pivot on, discovered projects nobody
verified. Rules live in code; these are judgments, so they get a
model IN A LOOP.

The ratified shape (Q1–Q5, all A):
  Q1A  Composition steps only (analyze/synthesize/format). Retrieval
       steps stay deterministic adapter calls — cheap primitives.
  Q2A  Fixed read-only toolset, ENGINE-enforced by construction: the
       worker's Agent gets a FRESH registry containing exactly
       web_search, web_fetch, crystal_search, crystal_key_scan,
       source_lookup. No write tools exist in its universe;
       cognition_run does not exist in its universe (no recursion).
  Q3A  Fixed platform caps: _AGENTIC_MAX_TOOL_CALLS iterations,
       _AGENTIC_WALL_SECONDS wall clock. Output budget = the flat
       composition cap.
  Q4A  In-process seam: the existing Agent loop instantiated inside
       the worker with the scoped registry. No HTTP hop.
  Q5A  Behind CC_COGNITION_AGENTIC_WORKERS (default False).

Information barriers hold exactly as ratified: the worker-agent sees
the step prompt (instruction + fair-share prior context + revision
block) — never goal.json, never the current attempt's validation.

Cost shape: an agentic composition step is 2–7 model calls instead of
one. The bet is attempt count — one approving agentic attempt beats
three rejecting classic ones (rematch #5 burned $0.34 on rejections a
single mid-composition fetch would have prevented).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import structlog

from ..cost.emit import record_model_call

logger = structlog.get_logger(__name__)

# Q3A platform caps — mechanism in code.
_AGENTIC_MAX_TOOL_CALLS = 6
_AGENTIC_WALL_SECONDS = 120.0

# The five read verbs (Q2A). Descriptions carry the DECISION guidance
# rematch #9 showed was missing — react to errors, verify before
# asserting.
_WORKER_TOOLS: list[dict[str, Any]] = [
    {
        "name": "web_search",
        "description": (
            "Search the web. Input {\"queries\": [...]} — one to five "
            "SEARCH-ENGINE KEYWORD queries (3-8 words each), fanned out "
            "concurrently. Use for DISCOVERY: finding the canonical "
            "repo/site for a name, finding what exists. If a fetch "
            "404'd, search the project's canonical name before "
            "guessing another URL."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 5,
                    "description": "Keyword queries, one target each.",
                },
            },
            "required": ["queries"],
        },
        "action": "web_search",
    },
    {
        "name": "web_fetch",
        "description": (
            "Retrieve exact page URLs you already know (rendered when "
            "the page needs JavaScript; github.com repo URLs are served "
            "from the GitHub API with releases + stats). Input "
            "{\"urls\": [...]}, up to five. React to what comes back: a "
            "404 means your URL guess is wrong — web_search the "
            "canonical name; an empty releases list means the project "
            "publishes elsewhere — fetch its homepage or tags instead."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 5,
                    "description": "Exact page URLs.",
                },
            },
            "required": ["urls"],
        },
        "action": "web_fetch",
    },
    {
        "name": "crystal_search",
        "description": (
            "Vector search over the customer's crystal bank. Good for "
            "meaning ('what do we know about X'); bad for counting or "
            "listing. Input {\"query\": \"...\"}."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
        "action": "crystal_search",
    },
    {
        "name": "crystal_key_scan",
        "description": (
            "Sparse-key prefix scan over the crystal bank "
            "(Source|Locator|Subject|Domain). Good for counting and "
            "listing what exists under a key prefix."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "key_prefix": {"type": "string"},
                "subject_contains": {"type": "string"},
            },
        },
        "action": "crystal_key_scan",
    },
    {
        "name": "source_lookup",
        "description": (
            "Look up registered source material (op/path/query/"
            "path_prefix passthrough) — read a known source document "
            "or list what sources exist under a path prefix."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "op": {"type": "string"},
                "path": {"type": "string"},
                "query": {"type": "string"},
                "path_prefix": {"type": "string"},
            },
        },
        "action": "source_lookup",
    },
]


def build_worker_registry(store: Any, fact_store: Any, encoder: Any):
    """A FRESH registry holding exactly the five read tools (Q2A).

    Enforcement is structural, not policy: write tools and
    cognition_run are not filtered out — they never exist in this
    registry, so the model cannot name them. Each impl routes through
    dispatch_cognition_retrieval, which is the SAME adapter the
    deterministic retrieval steps use — batch language, keywordize
    retry, GitHub API routing, SSRF guard, deadlines all included.
    """
    from ..agent.tool_registry import Tool, ToolRegistry
    from .retrieval_adapter import dispatch_cognition_retrieval

    registry = ToolRegistry()
    for spec in _WORKER_TOOLS:
        action = spec["action"]

        def _make_impl(action_value: str):
            async def _impl(customer_id: str, **kwargs: Any) -> dict:
                return await dispatch_cognition_retrieval(
                    action_value=action_value,
                    step_input=kwargs,
                    customer_id=customer_id,
                    store=store,
                    fact_store=fact_store,
                    encoder=encoder,
                )
            return _impl

        registry.register(Tool(
            name=spec["name"],
            description=spec["description"],
            contexts=frozenset({"agent"}),
            parameters_schema=spec["schema"],
            impl=_make_impl(action),
        ))
    return registry


def _worker_charter() -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"""You are a research worker executing ONE step of a larger plan. TODAY'S DATE IS {today} (UTC).

You have been given the step's instruction and source material gathered by earlier steps. Compose the step's output from that material. When the material is INSUFFICIENT or WRONG, use your tools to fix it before composing:
- A fetch that 404'd or errored means the URL guess was wrong. web_search the canonical name, then web_fetch the right page.
- An empty result (no releases, no data) means the information lives somewhere else. Pivot: fetch the project homepage, tags page, or changelog instead of accepting the gap.
- VERIFY claims that acceptance depends on (dates, version numbers, "newly launched") against a primary source before asserting them. A candidate discovered by search is not verified until its own page confirms the claim.
- CHECK IDENTITY before using repo/page data for a named project: the FETCHED REPOSITORY line (or the page's own name/description) must actually match the project. A rich, successful fetch of the WRONG repo looks exactly like a right one — a mismatch means wrong source, not "data found".
- Cite the ORIGINAL external URL for every factual claim — never internal step numbers.

Budget: you have a small number of tool calls ({_AGENTIC_MAX_TOOL_CALLS}). Spend them on the gaps that decide acceptance, not on re-checking what the material already establishes.

You are READ-ONLY: you cannot write memory or start workflows. When your research is done, write the step's complete output as your final message — structured text the next worker (or the deliverable) can use directly. Do not narrate your tool use in the final output."""


def _research_charter() -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"""You are a research worker. TODAY'S DATE IS {today} (UTC). Your ONLY job on this step: retrieve and VERIFY the listed targets from primary sources.

For each target:
1. Discover the canonical source (web_search the name if you don't have a verbatim URL — never guess URLs from memory).
2. Fetch it (web_fetch; github.com repo URLs return API data with a FETCHED REPOSITORY identity line).
3. CONFIRM IDENTITY: the fetched page/repo must actually BE the named target — check the name and description against the target. A rich result about the WRONG thing is worse than no result; if it mismatches, search the canonical name and re-fetch.
4. Verify the specific property asked for (version, release date, launch window, activity) against what the source itself says.
5. Some flagship projects do NOT publish GitHub releases (FFmpeg is the canonical example) — their version truth lives on the project's own site (e.g. ffmpeg.org/download) or in git tags. An empty GitHub releases list on an active repo means LOOK THERE, not "unknown". And never pair a version number with a different version's changelog — cite the changelog OF the version you name.

Budget: {_AGENTIC_MAX_TOOL_CALLS} tool calls — batch queries and URLs (both tools accept lists) so one call covers several targets.

Output: one section per target — canonical URL, the verified facts with the source URL for each, and an explicit IDENTITY CONFIRMED line naming the fetched source. If a target cannot be verified within budget, say exactly that for that target — never substitute a look-alike. You are READ-ONLY."""


async def run_agentic_composition(
    *,
    env: Any,
    step: Any,
    prompt: str,
    store: Any,
    fact_store: Any,
    encoder: Any,
    system: str = "",
) -> dict[str, Any]:
    """Run one composition step as a bounded agent session (Q4A).

    Returns {content, tool_calls, iterations, model, stop_reason}.
    Raises on timeout or loop error — the caller (roles) falls back to
    the classic single-call path so the new machinery can never lose
    an attempt.
    """
    from ..agent.agent import Agent
    from ..llm import get_llm_client
    from .roles import _COMPOSITION_MAX_TOKENS

    # Preserve the process-wide tool state: Agent.__init__ calls the
    # GLOBAL set_tool_state. Our scoped tools close over their own
    # store/fact_store/encoder, but the shared web tool (reached via
    # the dispatch adapter) may read the global state — pass the live
    # state through unchanged so instantiating a worker-agent never
    # clobbers the process.
    try:
        from ..agent.tools.retrievers import _get_state
        live_state = _get_state()
    except Exception:  # noqa: BLE001 — isolated tests have no state
        live_state = {}

    registry = build_worker_registry(store, fact_store, encoder)
    agent = Agent(
        customer=SimpleNamespace(id=env.customer_id),
        llm=get_llm_client(),
        tool_state=live_state,
        max_tokens=_COMPOSITION_MAX_TOKENS,
        max_iterations=_AGENTIC_MAX_TOOL_CALLS,
        registry=registry,
    )

    run = await asyncio.wait_for(
        agent.run(
            messages=[{"role": "user", "content": prompt}],
            system=system or _worker_charter(),
        ),
        timeout=_AGENTIC_WALL_SECONDS,
    )

    # Metering: the Agent loop does not meter (the chat endpoint does,
    # on its own path) — this seam lands ONE aggregated llm_calls row
    # for the whole session plus the env's UI estimate. Per-iteration
    # rows aren't available from Agent.run's aggregate return; the
    # total spend is what billing needs.
    prompt_tokens = int(run.get("prompt_tokens") or 0)
    completion_tokens = int(run.get("completion_tokens") or 0)
    env.record_tokens(prompt_tokens, completion_tokens, "sonnet")
    try:
        await record_model_call(
            customer_id=env.customer_id,
            model=run.get("model") or "unknown",
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            cache_read_tokens=run.get("cache_read_tokens"),
            cache_creation_tokens=run.get("cache_creation_tokens"),
            origin="cognition",
            session_id=env.id,
        )
    except Exception as e:  # noqa: BLE001 — metering must not fail work
        logger.warning("cognition.agentic_metering_failed",
                       env_id=env.id, error=str(e))

    # Trace for the pane + forensics: what the worker DID, outputs
    # trimmed (full outputs already flowed through the model turn).
    trace = [
        {
            "tool": c.get("tool_name"),
            "input": c.get("input"),
            "output_head": str(c.get("output"))[:500],
            "iteration": c.get("iteration"),
        }
        for c in (run.get("tool_calls") or [])
    ]

    logger.info(
        "cognition.agentic_step_complete",
        env_id=env.id,
        step_id=step.id,
        iterations=run.get("iterations"),
        tool_calls=len(trace),
        stop_reason=run.get("stop_reason"),
    )
    return {
        "content": run.get("final_text") or "",
        "tool_calls": trace,
        "iterations": run.get("iterations"),
        "model": run.get("model"),
        "stop_reason": run.get("stop_reason"),
    }


def _targets_from_input(step_input: dict) -> list[str]:
    raw = (step_input or {}).get("targets")
    if isinstance(raw, list):
        return [t.strip() for t in raw if isinstance(t, str) and t.strip()]
    single = (step_input or {}).get("target", "")
    return [single.strip()] if isinstance(single, str) and single.strip() else []


async def run_research_step(
    *,
    env: Any,
    step: Any,
    store: Any,
    fact_store: Any,
    encoder: Any,
) -> dict[str, Any]:
    """Execute a plannable RESEARCH step (ratified Q2A, 2026-07-14).

    Findings-shaped output so the answerability gate and carryover
    treat verified research as grounding. Requires the agentic flag;
    without it, degrades to a batch web_search over the target names
    (provenance-stamped) so a stale plan never strands a run.
    """
    targets = _targets_from_input(step.input or {})
    try:
        from ..config import get_settings
        agentic_on = bool(
            getattr(get_settings(), "cognition_agentic_workers", False)
        )
    except Exception:  # noqa: BLE001
        agentic_on = False

    if not agentic_on:
        from .retrieval_adapter import dispatch_cognition_retrieval
        logger.warning("cognition.research_degraded_to_search",
                       env_id=env.id, step_id=step.id,
                       targets=len(targets))
        out = await dispatch_cognition_retrieval(
            action_value="web_search",
            step_input={"queries": [
                t.split("—")[0].split(" - ")[0].strip()[:80]
                for t in targets[:5]
            ] or [""]},
            customer_id=env.customer_id,
            store=store, fact_store=fact_store, encoder=encoder,
        )
        out["degraded"] = "research_without_agentic_flag"
        return out

    prompt_lines = ["TARGETS TO RETRIEVE AND VERIFY:"]
    prompt_lines += [f"{i}. {t}" for i, t in enumerate(targets, 1)]
    if step.description:
        prompt_lines.append(f"\nStep instruction: {step.description}")
    result = await run_agentic_composition(
        env=env, step=step, prompt="\n".join(prompt_lines),
        store=store, fact_store=fact_store, encoder=encoder,
        system=_research_charter(),
    )
    text = result.get("content") or ""
    return {
        "targets": targets,
        "results_count": len(targets) if text.strip() else 0,
        "findings": (
            [{"title": f"Verified research: {len(targets)} targets",
              "url": "", "content": text}] if text.strip() else []
        ),
        "content_text": text,
        "agentic": True,
        "tool_calls": result.get("tool_calls") or [],
        "iterations": result.get("iterations"),
    }
