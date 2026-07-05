"""V3 Push/Pull Protocol — Phase 9 (Tool-Based).

Instead of hacking structured text blocks into the LLM response and
parsing with regex, the push/pull protocol uses native tool calling.
The LLM calls crystal_* tools when it notices new knowledge, gaps,
corrections, or needs research.

Advantages over text-block approach:
  1. Structured by default — tool args are typed JSON, no regex
  2. Reliable — LLMs are trained to produce well-formed tool calls
  3. Invisible to user — tool calls are in tool_use blocks, not text
  4. Composable — add new tools without changing prompt format
  5. MCP-compatible — Crystal Cache can expose these as an MCP server

TOOL DEFINITIONS (passed to the upstream LLM alongside user's tools):

  crystal_push_store(key, value, confidence)
    — Store new knowledge the LLM noticed during the conversation

  crystal_push_gap(domain, subject, missing)
    — Flag missing knowledge the LLM couldn't answer about

  crystal_push_correct(key, old_value, new_value)
    — User corrected a fact, flag it for review

  crystal_pull_research(topic, scope, priority)
    — Request the SLM agent to analyze a topic asynchronously

  crystal_pull_expand(key_pattern, reason)
    — Preload related context for likely follow-up queries

The system intercepts these tool calls before they reach the user,
processes them, and returns confirmations. The user never sees them.

v2 port (Phase 6 Wave D): verbatim from v1. This module has no SQL
and no I/O — it's pure data definitions + dispatch helpers. The
side-effecting handlers live in v3_signal_handler.py alongside this.

Wire-format note (R3): the tool names, parameter names, and the
`priority` enum values (`immediate`, `background`, `idle`) are public
contracts the upstream LLM sees. Do not rename without an ADR.

Note on `priority` enum vs Pydantic TaskPriority:
  - This module's tool definition exposes
    `priority: enum=["immediate", "background", "idle"]` to the LLM.
  - The v2 Pydantic `TaskPriority` only allows
    `["urgent", "background"]`.
  - The mapping happens in `v3_signal_handler._handle_research` at
    persistence time per P0.4 (Phase 6 Wave D decision): `immediate
    → urgent`, `idle → background`, `background → background`. Tool
    surface stays v1-verbatim per R3; v2 Pydantic strictness stays
    per D1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


# --- Tool Definitions (OpenAI format) ---
# These get appended to whatever tools the customer already defined.

CRYSTAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "crystal_push_store",
            "description": (
                "Store new knowledge you noticed during this conversation. "
                "Use when the user mentions a fact, preference, or relationship "
                "that isn't in the provided context. Key format: an ordered "
                "path of segments from GENERAL to SPECIFIC, separated by '|' "
                "(wide on the left, specific on the right)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Path key, general to specific (e.g. 'Personal|Alice|Preferences|Favorite Color')",
                    },
                    "value": {
                        "type": "string",
                        "description": "The knowledge to store",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "0.0-1.0. Use 0.9+ when user explicitly stated it. Use 0.5-0.8 when inferred from conversation.",
                        "default": 0.7,
                    },
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crystal_push_gap",
            "description": (
                "Flag missing knowledge. Use when you couldn't fully answer "
                "the user's question because the knowledge base lacks information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "Knowledge domain (Film, Healthcare, Technical, etc.)",
                    },
                    "subject": {
                        "type": "string",
                        "description": "What the gap is about (e.g. 'Corporate Mistletoe')",
                    },
                    "missing": {
                        "type": "string",
                        "description": "What information is missing",
                    },
                },
                "required": ["subject", "missing"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crystal_push_correct",
            "description": (
                "Flag a correction. Use when the user says stored information "
                "is wrong. Do NOT auto-correct — flag for review."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Key of the fact being corrected (the general-to-specific '|' path)",
                    },
                    "old_value": {
                        "type": "string",
                        "description": "The current (incorrect) value",
                    },
                    "new_value": {
                        "type": "string",
                        "description": "The corrected value from the user",
                    },
                },
                "required": ["key", "new_value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crystal_pull_research",
            "description": (
                "Request background research on a topic. The system will "
                "analyze the topic asynchronously and store results for "
                "future queries. Use for complex analytical questions that "
                "need cross-referencing multiple sources."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "What to research",
                    },
                    "scope": {
                        "type": "string",
                        "description": "Where to search (e.g. 'all scenes', 'entity_attribute facts')",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["immediate", "background", "idle"],
                        "default": "background",
                    },
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crystal_pull_expand",
            "description": (
                "Preload related context for follow-up queries. Use when "
                "you expect the user will ask a follow-up about related "
                "content (e.g. nearby scenes, related entities)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key_pattern": {
                        "type": "string",
                        "description": "Pattern to match (e.g. 'Script|Scene 4*|Corporate Mistletoe|Film')",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this context would be useful for follow-ups",
                    },
                },
                "required": ["key_pattern"],
            },
        },
    },
]

# Tool names for quick lookup
CRYSTAL_TOOL_NAMES = {t["function"]["name"] for t in CRYSTAL_TOOLS}


def is_crystal_tool_call(tool_call: dict) -> bool:
    """Check if a tool call is a Crystal Cache push/pull tool."""
    name = tool_call.get("function", {}).get("name", "")
    return name in CRYSTAL_TOOL_NAMES


def inject_crystal_tools(tools: Optional[list[dict]]) -> list[dict]:
    """Append Crystal Cache tools to the customer's tool list.

    If the customer defined their own tools, Crystal tools are added
    alongside them. If no tools were defined, Crystal tools become
    the tool list.

    Returns a new list (doesn't mutate the input).
    """
    existing = list(tools) if tools else []
    # Don't double-add
    existing_names = {
        t.get("function", {}).get("name", "")
        for t in existing
    }
    for tool in CRYSTAL_TOOLS:
        if tool["function"]["name"] not in existing_names:
            existing.append(tool)
    return existing


def extract_crystal_tool_calls(
    response: dict,
) -> tuple[list[dict], list[dict]]:
    """Separate Crystal tool calls from regular tool calls in a response.

    Args:
        response: the upstream LLM response (OpenAI format)

    Returns:
        (crystal_calls, other_calls) — crystal calls are intercepted
        by the system, other calls pass through to the user.
    """
    crystal_calls = []
    other_calls = []

    choices = response.get("choices", [])
    for choice in choices:
        message = choice.get("message", {})
        tool_calls = message.get("tool_calls", [])
        for tc in tool_calls:
            if is_crystal_tool_call(tc):
                crystal_calls.append(tc)
            else:
                other_calls.append(tc)

    return crystal_calls, other_calls


@dataclass
class ParsedSignals:
    """All signals parsed from tool calls."""
    push_stores: list[dict] = field(default_factory=list)
    push_gaps: list[dict] = field(default_factory=list)
    push_corrections: list[dict] = field(default_factory=list)
    pull_research: list[dict] = field(default_factory=list)
    pull_expand: list[dict] = field(default_factory=list)
    raw_tool_calls: list[dict] = field(default_factory=list)

    @property
    def has_signals(self) -> bool:
        return bool(
            self.push_stores or self.push_gaps or self.push_corrections
            or self.pull_research or self.pull_expand
        )

    @property
    def total_count(self) -> int:
        return (
            len(self.push_stores) + len(self.push_gaps)
            + len(self.push_corrections) + len(self.pull_research)
            + len(self.pull_expand)
        )


def parse_tool_calls(tool_calls: list[dict]) -> ParsedSignals:
    """Parse Crystal tool calls into structured signals.

    Args:
        tool_calls: list of crystal tool call dicts from extract_crystal_tool_calls

    Returns:
        ParsedSignals with each signal type populated.
    """
    import json

    signals = ParsedSignals()
    signals.raw_tool_calls = tool_calls

    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        args_str = func.get("arguments", "{}")

        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            logger.warning("push_pull.bad_tool_args", name=name, args=args_str[:100])
            continue

        if name == "crystal_push_store":
            signals.push_stores.append(args)
        elif name == "crystal_push_gap":
            signals.push_gaps.append(args)
        elif name == "crystal_push_correct":
            signals.push_corrections.append(args)
        elif name == "crystal_pull_research":
            signals.pull_research.append(args)
        elif name == "crystal_pull_expand":
            signals.pull_expand.append(args)

    if signals.has_signals:
        logger.info(
            "push_pull.parsed",
            total=signals.total_count,
            stores=len(signals.push_stores),
            gaps=len(signals.push_gaps),
            corrections=len(signals.push_corrections),
            research=len(signals.pull_research),
            expand=len(signals.pull_expand),
        )

    return signals
