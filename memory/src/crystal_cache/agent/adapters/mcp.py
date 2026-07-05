"""MCP adapter — expose the tool registry as an MCP server.

Per P0.21: Python-functions internal; MCP only for external
consumers. The in-process MCP server SHIPPED (WS C —
`agent/mcp_server.py`, mounted at `/mcp` by app.py); it is built on
FastMCP, which generates tool schemas from the Python signatures
directly, so this renderer is not on its path. `render_tools_for_mcp`
remains the documented wire-format view for adapter parity with the
anthropic/openai renderers and for any external consumer that wants
the registry as MCP tool dicts without standing up the server. The
internal agent does NOT round-trip through MCP — that path would add
latency on the hot path with no clear benefit.

MCP TOOL SPEC SHAPE:
MCP tools are exposed via the server's `tools/list` and called via
`tools/call`. Each tool in the list has:

    {
        "name": "<tool name>",
        "description": "<one-line purpose>",
        "inputSchema": {  # camelCase per MCP spec
            "type": "object",
            "properties": {...},
            "required": [...]
        }
    }

The MCP `tools/call` request gives `{name, arguments}` where
arguments is already a parsed JSON object (not a string, unlike
OpenAI). The server dispatches via the same Agent._dispatch_tool
the in-process agent uses.

WHEN MCP LANDS:
- A new `agent/mcp_server.py` module wires the MCP server protocol
  (stdio or HTTP) and exposes the registry's "agent"-context tools.
- Customer-id resolution becomes a per-connection setting on the
  MCP server (passed at handshake or as an MCP capability
  parameter) — MCP itself doesn't have a built-in "customer
  identity" concept, so this is a Crystal-Cache-specific extension.
- The server can also expose per-tool resources (the tool's
  parameter schema as an MCP resource for richer client UX), but
  that's optional and lands later.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from ..tool_registry import Tool

logger = structlog.get_logger(__name__)


def render_tools_for_mcp(tools: list["Tool"]) -> list[dict[str, Any]]:
    """Render a list of registered Tools as MCP tool spec dicts.

    Each entry follows the MCP `tools/list` response format with
    `inputSchema` (camelCase) per the MCP spec.
    """
    return [_render_one(t) for t in tools]


def _render_one(tool: "Tool") -> dict[str, Any]:
    """Render a single Tool as an MCP tool spec."""
    return {
        "name": tool.name,
        "description": tool.description,
        "inputSchema": tool.parameters_schema,
    }
