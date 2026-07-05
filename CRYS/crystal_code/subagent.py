"""F7 — Subagents: the main agent delegates research to scoped workers.

The cognition engine's orchestrator/worker pattern, applied to the
coding loop: the main agent (big model) hands a research task —
"find every caller of X and summarize the contract" — to a fresh
worker Agent on the FAST model (F6 routing). The worker has its own
context window and a hard read-only policy; the main agent gets back
just the synthesis, keeping its own context lean.

Containment, by construction rather than by trust:
  - read-only: the worker's interceptor denies writes, shell, and
    run_verify — research, not execution. No approval prompts ever
    reach the user from inside a worker.
  - depth 1: a worker cannot call the subagent tool. The registry is
    shared, so the tool is visible — the interceptor is the wall.
  - block_paths still hold: the PARENT guard's rules are enforced in
    the worker too, so a blocked `.env` can't be read by delegating
    the read one level down.
  - bounded: iterations are capped lower than the main agent's, and a
    semaphore caps concurrent workers at 3 process-wide.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

from crystal_cache.agent import Agent, Tool, get_registry
from crystal_cache.agent.system_prompt import build_system_prompt

from .guard import ALWAYS_ALLOWED, Guard, classify

SUBAGENT_TOOL_NAME = "subagent"
_MAX_CONCURRENT = 3
_MAX_ITERATIONS = 10

_semaphore = asyncio.Semaphore(_MAX_CONCURRENT)

_SUBAGENT_ADDENDUM = (
    "\n\nYou are a READ-ONLY research subagent. Investigate the task using "
    "read, list, search, and knowledge tools — mutating tools will refuse. "
    "Be thorough but economical, and end with a concise, self-contained "
    "report: the caller sees ONLY your final message, none of your "
    "intermediate steps."
)


def _make_policy(guard: Guard):
    """The worker's interceptor: parent block_paths + read-only + depth 1."""

    async def policy(tool_name: str, tool_input: dict[str, Any]) -> dict:
        if tool_name == SUBAGENT_TOOL_NAME:
            return {
                "action": "deny",
                "reason": "subagents cannot spawn subagents (depth limit is 1).",
            }
        # Parent project rules hold inside workers too.
        from .guard import _paths_from_input  # local import: same module family
        for p in _paths_from_input(tool_input):
            rule = guard._blocked_by(p)
            if rule:
                return {
                    "action": "deny",
                    "reason": (
                        f"project hook block_paths: '{p}' matches '{rule}' — "
                        "off-limits, including to subagents."
                    ),
                }
        if classify(tool_name) != "read" or tool_name in ALWAYS_ALLOWED:
            return {
                "action": "deny",
                "reason": (
                    "this is a read-only research subagent: read, list, "
                    "search, and knowledge tools only — no edits, no "
                    "commands, no verification runs."
                ),
            }
        return {"action": "allow"}

    return policy


def register_subagent_tool(
    parent_ref: dict,
    fast_model_fn: Callable[[], str],
    guard: Guard,
) -> bool:
    """Register the `subagent` tool (once per process).

    parent_ref: a {"agent": Agent|None} holder the CLI keeps pointed at
    the LIVE agent — /login swaps customers, and the worker must always
    inherit the current one (customer, client, tool_state).
    fast_model_fn: returns the current fast model (F6 routing).
    """
    registry = get_registry()
    if SUBAGENT_TOOL_NAME in registry:
        return False

    async def _impl(customer_id: str, task: str, context: Optional[str] = None, **kwargs: Any) -> dict[str, Any]:
        parent = parent_ref.get("agent")
        if parent is None:
            raise RuntimeError("subagent unavailable: no parent agent is running")
        prompt = task if not context else f"{task}\n\nContext from the caller:\n{context}"

        async with _semaphore:
            worker = Agent(
                customer=parent.customer,
                llm=parent.llm,
                tool_state=parent.tool_state,
                model=fast_model_fn(),
                max_iterations=_MAX_ITERATIONS,
                intercept=_make_policy(guard),
            )
            system = build_system_prompt(worker.customer, worker.tools) + _SUBAGENT_ADDENDUM
            result = await worker.run(
                messages=[{"role": "user", "content": prompt}],
                system=system,
            )
        return {
            "report": result.get("final_text") or "(the subagent produced no report)",
            "iterations": result.get("iterations"),
            "prompt_tokens": result.get("prompt_tokens"),
            "completion_tokens": result.get("completion_tokens"),
        }

    registry.register(Tool(
        name=SUBAGENT_TOOL_NAME,
        description=(
            "Delegate a READ-ONLY research task to a fast subagent with its "
            "own context window, and receive back only its final report. Use "
            "this to keep your own context lean on broad investigations — "
            "'find every caller of X and summarize the contract', 'map how "
            "module Y is wired', 'what does the knowledge base say about Z'. "
            "You may call it several times in one turn for independent "
            "questions (they run in parallel, capped at 3). Subagents cannot "
            "edit files, run commands, or spawn further subagents. Give each "
            "one a specific, self-contained task; include any context it "
            "needs, since it cannot see your conversation."
        ),
        contexts=frozenset({"agent"}),
        parameters_schema={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The specific, self-contained research task.",
                },
                "context": {
                    "type": "string",
                    "description": "Optional background the worker needs (it cannot see your conversation).",
                },
            },
            "required": ["task"],
        },
        impl=_impl,
    ))
    return True
