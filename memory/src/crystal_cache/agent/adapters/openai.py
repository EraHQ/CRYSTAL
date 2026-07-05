"""OpenAI adapter — wire translation for the OpenAI-routed agent loop.

Anthropic message shape is CRYS's internal representation (see
docs/LOCAL_MODELS_PLAN.md, Architecture decision). This module translates at
the Agent._call_model boundary only:

- `render_tools_for_openai` — registry Tools -> OpenAI function specs.
- `tools_to_openai` — Anthropic-shaped tool DICTS -> OpenAI function specs
  (the loop renders tools once, Anthropic-shaped; the OpenAI path translates
  that rendering rather than re-rendering from the registry).
- `messages_to_openai` — Anthropic-shaped system + history -> OpenAI chat
  messages (tool_use -> tool_calls, tool_result -> role "tool";
  cache_control never crosses the boundary).
- `parse_openai_response` — OpenAI completion JSON -> a shim exposing
  `.content` (plain Anthropic-shaped block dicts — Agent._content_to_dict_list
  passes dicts through), `.stop_reason` (finish_reason mapped, preserving the
  H2 max_tokens truncation guard), and `.usage` with the four token fields the
  loop reads via getattr.

OPENAI FUNCTION-CALLING SPEC SHAPE (current v1 OpenAI SDK):
`tools=[{"type": "function", "function": {...}}]`; the model emits
`tool_calls` in the response message with shape
[{id, type: 'function', function: {name, arguments}}] — `arguments`
is a JSON-encoded string parsed defensively before dispatch (malformed
arguments become {} and the tool's own error path informs the model).

The loop reuses Agent._dispatch_tool (protocol-agnostic) unchanged.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from ..tool_registry import Tool

logger = structlog.get_logger(__name__)


def render_tools_for_openai(tools: list["Tool"]) -> list[dict[str, Any]]:
    """Render a list of registered Tools as OpenAI tool spec dicts.

    Each entry has the `{type: "function", function: {...}}` shape
    expected by the current OpenAI Chat Completions API tool-calling
    surface.
    """
    return [_render_one(t) for t in tools]


def _render_one(tool: "Tool") -> dict[str, Any]:
    """Render a single Tool as an OpenAI tool spec."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters_schema,
        },
    }


# ---------------------------------------------------------------------------
# Slice 5: request-side translation (Anthropic shape -> OpenAI wire)
# ---------------------------------------------------------------------------

def tools_to_openai(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Translate Anthropic-shaped tool dicts into OpenAI function specs.

    Input dicts carry name/description/input_schema (and possibly an
    Anthropic-only cache_control mark on the last tool, which is dropped).
    Returns None for an empty/None tool list so callers can omit the field.
    """
    if not tools:
        return None
    out: list[dict[str, Any]] = []
    for t in tools:
        out.append({
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("input_schema")
                or {"type": "object", "properties": {}},
            },
        })
    return out


def _tool_result_text(content: Any) -> str:
    """Stringify a tool_result's content for a role="tool" message.

    The agent loop builds tool_result content as a string; block-list and
    other shapes are handled defensively.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(str(b.get("text", "")))
            else:
                parts.append(json.dumps(b, default=str))
        return "\n".join(parts)
    return json.dumps(content, default=str)


def messages_to_openai(
    system: str | None,
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Translate an Anthropic-shaped history into OpenAI chat messages.

    - `system` becomes a leading {"role": "system"} message.
    - A user turn with string content passes through.
    - A user turn with blocks splits: tool_result blocks become role "tool"
      messages (tool_call_id = tool_use_id; is_error prefixes the content so
      the model still sees the failure), emitted FIRST so they directly
      follow the assistant tool_calls turn as OpenAI requires; any text
      blocks in the same turn follow as one user message.
    - An assistant turn with blocks: text blocks join into content;
      tool_use blocks become tool_calls with JSON-string arguments.
    - cache_control marks never cross the boundary.
    """
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            out.append({"role": role, "content": str(content)})
            continue

        if role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for b in content:
                if not isinstance(b, dict):
                    continue
                btype = b.get("type")
                if btype == "text":
                    text_parts.append(str(b.get("text", "")))
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": b.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": b.get("name", ""),
                            "arguments": json.dumps(
                                b.get("input") or {}, default=str,
                            ),
                        },
                    })
                else:
                    logger.warning(
                        "openai_adapter.unknown_assistant_block",
                        block_type=btype,
                    )
            wire: dict[str, Any] = {
                "role": "assistant",
                "content": "\n\n".join(text_parts) if text_parts else None,
            }
            if tool_calls:
                wire["tool_calls"] = tool_calls
            out.append(wire)
            continue

        # User turn with blocks: tool results first, then any text.
        text_parts = []
        for b in content:
            if not isinstance(b, dict):
                continue
            btype = b.get("type")
            if btype == "tool_result":
                body = _tool_result_text(b.get("content"))
                if b.get("is_error"):
                    body = f"[tool error] {body}"
                out.append({
                    "role": "tool",
                    "tool_call_id": b.get("tool_use_id", ""),
                    "content": body,
                })
            elif btype == "text":
                text_parts.append(str(b.get("text", "")))
            else:
                logger.warning(
                    "openai_adapter.unknown_user_block", block_type=btype,
                )
        if text_parts:
            out.append({"role": "user", "content": "\n\n".join(text_parts)})

    return out


# ---------------------------------------------------------------------------
# Slice 5: response-side translation (OpenAI wire -> Anthropic-shaped shim)
# ---------------------------------------------------------------------------

_FINISH_TO_STOP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",  # preserves the H2 truncation guard
}


@dataclass
class _UsageShim:
    """Duck-types the Anthropic usage object the loop reads via getattr."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class OpenAIChatShim:
    """Duck-types the Anthropic Messages response for the agent loop.

    `.content` holds plain Anthropic-shaped block dicts —
    Agent._content_to_dict_list passes dicts through unchanged, so the
    loop body, persistence, and trajectory stay identical either way.
    """

    content: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: _UsageShim = field(default_factory=_UsageShim)
    model: str = ""


def parse_openai_response(data: dict[str, Any]) -> OpenAIChatShim:
    """Translate an OpenAI chat completion into an Anthropic-shaped shim.

    Defensive by design (Phase 4 of the local-models plan): malformed
    tool-call arguments become {} with a warning — the tool's own error
    path then informs the model, which can retry. A response that carries
    tool_calls is treated as stop_reason "tool_use" even if the server
    reported finish_reason "stop" (some OpenAI-compatible servers do).
    A structurally empty response raises ValueError — the loop's error
    handling surfaces it.
    """
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("openai response has no choices")
    message = choices[0].get("message") or {}

    blocks: list[dict[str, Any]] = []
    text = message.get("content")
    if text:
        blocks.append({"type": "text", "text": text})

    for i, tc in enumerate(message.get("tool_calls") or []):
        fn = tc.get("function") or {}
        raw_args = fn.get("arguments") or "{}"
        try:
            parsed = json.loads(raw_args)
            if not isinstance(parsed, dict):
                raise ValueError("arguments not an object")
        except (ValueError, TypeError) as e:
            logger.warning(
                "openai_adapter.malformed_tool_arguments",
                tool=fn.get("name", ""),
                error=str(e),
            )
            parsed = {}
        blocks.append({
            "type": "tool_use",
            "id": tc.get("id") or f"toolcall_{i}",
            "name": fn.get("name", ""),
            "input": parsed,
        })

    finish = choices[0].get("finish_reason")
    stop_reason = _FINISH_TO_STOP.get(finish)
    if stop_reason is None:
        logger.warning(
            "openai_adapter.unknown_finish_reason", finish_reason=finish,
        )
        stop_reason = "end_turn"
    if any(b.get("type") == "tool_use" for b in blocks):
        stop_reason = "tool_use"

    usage_raw = data.get("usage") or {}
    details = usage_raw.get("prompt_tokens_details") or {}
    usage = _UsageShim(
        input_tokens=usage_raw.get("prompt_tokens") or 0,
        output_tokens=usage_raw.get("completion_tokens") or 0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=details.get("cached_tokens") or 0,
    )

    return OpenAIChatShim(
        content=blocks,
        stop_reason=stop_reason,
        usage=usage,
        model=data.get("model", ""),
    )
