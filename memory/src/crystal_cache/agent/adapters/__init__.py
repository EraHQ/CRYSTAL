"""Protocol adapters — render tools for per-protocol API shapes.

Three adapters live here:
- anthropic.py: Anthropic Messages API tool spec. FULL implementation —
  this is the path the agent loop uses at runtime per P0.21.
- openai.py: OpenAI Chat Completions tool-calling spec plus the
  request/response wire translation for the OpenAI-routed agent loop
  (Slice 5 of the provider-swap arc; see docs/LOCAL_MODELS_PLAN.md).
- mcp.py: MCP `tools/list` shape. Wire format rendered; MCP server
  process is a Phase 7.5+1 / Phase 11 task.

The three adapters share a contract: they take a list of registered
`Tool` instances from the tool registry and return a list of dicts
shaped per the target protocol. The Python signatures of the tool
implementations are the source of truth (P0.22); the adapters are
views over the registry, not separate definitions.
"""
from .anthropic import render_tools_for_anthropic
from .mcp import render_tools_for_mcp
from .openai import (
    OpenAIChatShim,
    messages_to_openai,
    parse_openai_response,
    render_tools_for_openai,
    tools_to_openai,
)

__all__ = [
    "OpenAIChatShim",
    "messages_to_openai",
    "parse_openai_response",
    "render_tools_for_anthropic",
    "render_tools_for_mcp",
    "render_tools_for_openai",
    "tools_to_openai",
]
