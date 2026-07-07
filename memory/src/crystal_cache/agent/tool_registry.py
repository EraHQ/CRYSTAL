"""Tool registry — single source of truth for the agent's tool surface.

Phase 7.5 implements D-A7 (independent packaging shapes) and §6.5
(curated overlap between agent and cognition) via this module.

DESIGN (locked in P0.22-P0.23):
- Tools are Python async functions with type-annotated signatures.
- Each tool is registered via `@register_tool(...)` at module import.
- Registration captures: name, contexts ({"agent","cognition"} or
  subset), description, JSON schema for params, schema for return,
  and the async callable.
- Schemas can be hand-supplied (more reliable for nested types) or
  reflected from the function signature.
- Adapters (`agent/adapters/anthropic.py`, `openai.py`, `mcp.py`)
  iterate the registry and translate the schemas to per-protocol
  formats. The Python signature is the source; the protocols are
  views over it.

CALLER CONTRACT (P0.23):
- Every tool accepts `customer_id: str` as the first parameter so
  the registry can pass identity uniformly. The agent loop injects
  the calling customer's id; cognition workers do the same. Tools
  never trust customer_id from LLM-emitted arguments.
- Every tool returns a JSON-serializable dict. Pydantic models live
  inside tool implementations; the dict serialization happens at
  the boundary so adapters can re-emit per protocol without
  knowing Python types.

CONTEXT FILTERING (D-A10 + §6.5.2):
- Tools tagged `contexts={"agent"}` are agent-only (write-side,
  llm_invoke, cognition_run).
- Tools tagged `contexts={"cognition"}` are cognition-worker-only
  (analyze, synthesize, format — composition primitives).
- Tools tagged `contexts={"agent", "cognition"}` are read-side
  shared (the four retrievers, both recalls, web_search, decompose).
- The agent's tool loop filters to `"agent" in contexts`.
- Cognition's worker dispatcher (after the §6.5.5 refactor) filters
  to `"cognition" in contexts`.

WIRE-FORMAT NAMING (R3, P0.26):
- Agent-side tool names are the DESIGN names from D-A3:
  content_search, knowledge_search, navigation_search, depth_search.
- Cognition's StepAction enum keeps v1 verbatim names:
  crystal_search, crystal_key_scan, web_search, analyze, synthesize,
  format. The enum values are persisted in cognition_tasks DB rows,
  so they are wire-format strings (R3).
- The registry holds both: each tool's `name` is the agent-side
  name; the optional `cognition_action_alias` maps the agent name
  to the StepAction enum value used internally by cognition. This
  lets the agent address tools by their design names while the
  cognition worker dispatcher addresses them by StepAction.value.

NOT IN THIS MODULE:
- Per-tool latency budgets, caching, rate limits (Phase 8
  operational concerns per §10).
- Streaming results back from cognition (Phase 8+ question per
  §6.5.7).
- MCR trace emission hooks (Phase 9 — extends each tool call with
  trace events).
"""
from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Tool metadata + registry
# ---------------------------------------------------------------------------

# Type alias for tool implementations. Every tool is an async function
# taking customer_id as its first param plus tool-specific kwargs, and
# returning a JSON-serializable dict.
ToolImpl = Callable[..., Awaitable[dict[str, Any]]]


@dataclass
class Tool:
    """A registered tool.

    Carries everything the agent loop, cognition dispatcher, and the
    three adapters (Anthropic, OpenAI, MCP) need to operate the tool.
    """

    name: str
    """Agent-side tool name (D-A3 design name). Examples:
    'content_search', 'knowledge_search', 'cognition_run'."""

    description: str
    """One-sentence description shown to the LLM in the tool list.
    Per D-A3/D-A6: descriptions are how the agent decides when to
    call a tool, so they must be precise about purpose and tradeoffs."""

    contexts: frozenset[str]
    """Set of execution contexts this tool can be called from.
    Allowed values: 'agent', 'cognition'. Most read-side tools have
    both; write-side and llm_invoke are agent-only; composition
    actions are cognition-only; cognition_run is agent-only."""

    parameters_schema: dict[str, Any]
    """JSON Schema for the tool's parameters. Used by adapters to
    generate per-protocol schemas. Should match the implementation's
    keyword arguments (excluding customer_id, which the registry
    injects)."""

    impl: ToolImpl
    """The async function that runs when this tool is called."""

    cognition_action_alias: Optional[str] = None
    """Optional alias for cognition's StepAction enum value. When set,
    cognition's worker dispatcher uses this name to look up the tool
    (preserves R3 wire-format compatibility with persisted
    cognition_tasks rows). When None, cognition uses `name`."""

    returns_description: str = ""
    """Free-text description of what the tool returns. Surfaced in
    adapter-generated schemas where the target protocol supports
    return-type docs."""

    available: Optional[Callable[[], bool]] = None
    """Optional runtime-availability predicate (2026-07-07). None =
    always available (the common case). When set, list_for_context
    HIDES the tool while the predicate is False — the model never sees
    it in the tool list or the system prompt, so it's never told to use
    a backend that isn't live (first user: the mem0 session tools,
    whose optional extra isn't installed on hosted). Registration is
    unconditional; visibility is what's dynamic."""


class ToolRegistry:
    """Holds all registered tools.

    Implemented as a regular class (not a singleton-via-globals)
    because tests want to be able to construct fresh registries.
    The module-level `get_registry()` exposes a process-wide
    singleton for normal usage.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        # Index by cognition_action_alias for fast worker dispatch.
        self._cognition_index: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Raises if the name is already taken."""
        if tool.name in self._tools:
            raise ValueError(
                f"Tool {tool.name!r} is already registered. "
                f"Tool names must be unique across the agent surface."
            )
        # Validate contexts is a non-empty subset of allowed values
        allowed = {"agent", "cognition"}
        if not tool.contexts:
            raise ValueError(
                f"Tool {tool.name!r}: contexts must be non-empty. "
                f"Pass at least one of {allowed}."
            )
        invalid = tool.contexts - allowed
        if invalid:
            raise ValueError(
                f"Tool {tool.name!r}: invalid contexts {invalid!r}. "
                f"Allowed values: {allowed}."
            )
        self._tools[tool.name] = tool
        # Build cognition index (StepAction.value → Tool)
        if "cognition" in tool.contexts:
            alias = tool.cognition_action_alias or tool.name
            if alias in self._cognition_index:
                # Two tools claim the same cognition alias — bug
                existing = self._cognition_index[alias]
                raise ValueError(
                    f"Tool {tool.name!r} cognition_action_alias "
                    f"{alias!r} collides with already-registered "
                    f"tool {existing.name!r}."
                )
            self._cognition_index[alias] = tool
        logger.debug(
            "tool_registry.registered",
            name=tool.name,
            contexts=sorted(tool.contexts),
            has_cognition_alias=tool.cognition_action_alias is not None,
        )

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by agent-side name. Returns None if not found."""
        return self._tools.get(name)

    def get_by_cognition_action(self, action_value: str) -> Optional[Tool]:
        """Get a tool by cognition StepAction enum value.

        Used by the §6.5.5-refactored worker dispatcher to look up
        which tool implementation backs a particular cognition step
        action. Returns None if no tool is registered with that alias
        or name in the cognition context (composition actions like
        ANALYZE/SYNTHESIZE/FORMAT are not in the tool registry — the
        worker dispatcher handles them separately).
        """
        return self._cognition_index.get(action_value)

    def list_for_context(self, context: str) -> list[Tool]:
        """List all tools available in a given context.

        Args:
            context: One of 'agent', 'cognition'.

        Returns:
            List of Tool instances, sorted by name for deterministic
            ordering (matters for prompt generation — the system
            prompt's tool list should be stable across runs).
        """
        return sorted(
            (
                t for t in self._tools.values()
                if context in t.contexts
                and (t.available is None or t.available())
            ),
            key=lambda t: t.name,
        )

    def names(self) -> list[str]:
        """Return all registered tool names, sorted."""
        return sorted(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Return the process-wide registry singleton.

    Constructed lazily on first access. Tool modules in
    `agent/tools/*.py` register their tools at import time by
    calling `get_registry().register(...)` (typically via the
    `@register_tool` decorator below).

    Tests that want isolation can construct fresh `ToolRegistry`
    instances directly; this function is the normal-usage entry
    point.
    """
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def reset_registry() -> None:
    """Clear the process-wide registry. ONLY for tests.

    Phase 8 fix (2026-05-26): also drops the agent.tools.* module
    cache from sys.modules so the next `import_all_tools()` call
    actually fires the @register_tool decorator side effects again.

    Without the sys.modules pop, Python's import system treats the
    second `from .tools import ...` as a no-op (modules are already
    imported), so the decorators don't re-run, so the freshly-empty
    registry stays empty. Tests that asserted "registry repopulates
    after reset_registry()" silently saw an empty registry.

    The sys.modules pop is scoped to the agent.tools subpackage so
    we don't perturb the rest of the import graph. The
    `agent.tool_registry` module itself is NOT cleared because
    that's THIS module — clearing it would orphan the globals
    `_registry` and `reset_registry` mid-call.
    """
    global _registry
    _registry = None
    # Pop the tool modules so the next import_all_tools re-fires
    # the @register_tool decorators. Order doesn't matter; popping
    # is idempotent on missing keys via .pop(name, None).
    for mod_name in [
        "crystal_cache.agent.tools.artifacts",
        "crystal_cache.agent.tools.cognition",
        "crystal_cache.agent.tools.curation",
        "crystal_cache.agent.tools.external",
        "crystal_cache.agent.tools.llm",
        "crystal_cache.agent.tools.memory",
        "crystal_cache.agent.tools.retrievers",
    ]:
        sys.modules.pop(mod_name, None)


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def register_tool(
    *,
    name: str,
    description: str,
    contexts: set[str],
    parameters_schema: dict[str, Any],
    cognition_action_alias: Optional[str] = None,
    returns_description: str = "",
    available: Optional[Callable[[], bool]] = None,
) -> Callable[[ToolImpl], ToolImpl]:
    """Decorator: registers an async function as an agent tool.

    Example:
        @register_tool(
            name="content_search",
            description="Find verbatim document chunks matching a query.",
            contexts={"agent", "cognition"},
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "..."},
                    "k": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        )
        async def content_search(
            customer_id: str,
            query: str,
            k: int = 10,
        ) -> dict[str, Any]:
            ...

    The decorator does NOT modify the wrapped function; it just
    registers it. The function remains callable directly for tests
    or for callers that already have a customer_id and tool name in
    hand.
    """

    def decorator(impl: ToolImpl) -> ToolImpl:
        # Sanity-check the signature: first param must be customer_id.
        try:
            sig = inspect.signature(impl)
            params = list(sig.parameters.values())
            if not params:
                raise ValueError(
                    f"Tool {name!r}: implementation has no parameters; "
                    f"must accept customer_id as the first argument."
                )
            if params[0].name != "customer_id":
                raise ValueError(
                    f"Tool {name!r}: first parameter must be named "
                    f"'customer_id' per P0.23. Got {params[0].name!r}."
                )
        except (ValueError, TypeError):
            # Re-raise validation errors; other introspection failures
            # shouldn't break import (e.g. C-extension async fns).
            raise

        tool = Tool(
            name=name,
            description=description,
            contexts=frozenset(contexts),
            parameters_schema=parameters_schema,
            impl=impl,
            cognition_action_alias=cognition_action_alias,
            returns_description=returns_description,
            available=available,
        )
        get_registry().register(tool)
        return impl

    return decorator


# ---------------------------------------------------------------------------
# Convenience: bulk import (called from agent/__init__.py)
# ---------------------------------------------------------------------------

def import_all_tools() -> None:
    """Trigger registration of every tool by importing every tool module.

    Called once during agent initialization (typically from
    `agent/__init__.py`). Import order doesn't matter — each tool
    module registers independently — but importing them all in one
    place gives a single point to debug "why isn't tool X in the
    registry."

    Idempotent at the Python-import level: re-importing already-
    cached modules is a no-op. To force re-registration (e.g. in
    tests after `reset_registry()`), call `reset_registry()` first
    — its sys.modules pop ensures the decorators fire again on the
    next import.
    """
    # These imports trigger @register_tool side effects in each module.
    # noqa: F401 because we don't reference the symbols directly.
    from .tools import artifacts as _artifacts  # noqa: F401
    from .tools import cognition as _cognition  # noqa: F401
    from .tools import curation as _curation  # noqa: F401
    from .tools import external as _external  # noqa: F401
    from .tools import llm as _llm  # noqa: F401
    from .tools import memory as _memory  # noqa: F401
    from .tools import retrievers as _retrievers  # noqa: F401
    logger.info(
        "tool_registry.imported_all",
        tools=get_registry().names(),
        count=len(get_registry()),
    )
