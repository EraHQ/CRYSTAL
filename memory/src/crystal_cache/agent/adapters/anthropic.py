"""Anthropic adapter — render tools for the Messages API.

Per P0.21 + P0.22: Python signatures are the source of truth, and
this adapter is the canonical implementation that the agent loop
relies on. Phase 7.5 ships only this adapter as fully-working
code; `openai.py` and `mcp.py` are skeletons whose wire format is
documented but whose runtime path lands in later phases.

ANTHROPIC TOOL SPEC SHAPE:
Each tool in the Anthropic Messages API request is a dict:
    {
        "name": "<tool name>",
        "description": "<one-line purpose>",
        "input_schema": {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
    }

The agent invokes the model with `tools=[...]`, and the model
emits `tool_use` content blocks with the tool name + input dict.
We sanitize the input (drop customer_id) and dispatch via the
registry — see Agent._dispatch_tool.

PARAMETER FILTERING:
Tools' `parameters_schema` already excludes `customer_id` (per
P0.23 — the registry injects it from the request context). So
the schema we hand to Anthropic is the tool's parameters_schema
verbatim, just wrapped as `input_schema`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from ..tool_registry import Tool

logger = structlog.get_logger(__name__)

# C1 (prompt caching): an ephemeral (5-min) cache breakpoint — CC-D2. The
# 5-min TTL needs no beta header; a back-to-back agent loop keeps it warm.
_EPHEMERAL_CACHE = {"type": "ephemeral"}


def render_tools_for_anthropic(tools: list["Tool"]) -> list[dict[str, Any]]:
    """Render a list of registered Tools as Anthropic tool spec dicts.

    Returns one dict per tool, in the order the tools were passed
    in (typically sorted alphabetically by name — the registry's
    `list_for_context` returns sorted results).

    C1 (prompt caching): the LAST tool carries an ephemeral `cache_control`
    breakpoint, so the whole tools-array prefix is cached. The tool set is
    identical across every CRYS request (and across customers — only the
    system prompt is customer-specific), so after the first write this prefix
    is a cache read within the 5-min window. Anthropic caches in
    tools → system → messages order, so this is the innermost cached prefix;
    `_call_model` layers the system + a moving conversation breakpoint on top.
    """
    rendered = [_render_one(t) for t in tools]
    if rendered:
        rendered[-1] = {**rendered[-1], "cache_control": _EPHEMERAL_CACHE}
    return rendered


def _render_one(tool: "Tool") -> dict[str, Any]:
    """Render a single Tool as an Anthropic tool spec."""
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters_schema,
    }
