"""System prompt — the agent's instructions.

Per P0.24: the system prompt is generated dynamically from the
tool registry. This avoids the prompt drifting from the actual
tool surface when tools are added/removed.

The prompt has three sections:

1. ROLE — who the agent is (a Crystal Cache assistant with access
   to a customer's knowledge bank and a set of tools).
2. TOOL DESCRIPTIONS — auto-generated from the registry. Each tool
   gets its description and an example calling pattern.
3. POLICIES — the rules the agent follows: retrieval-before-LLM,
   honesty about uncertainty, when to call cognition_run, when to
   refuse, formatting conventions for the final answer.

POLICIES are LOCKED at Phase 7.5 launch and refined based on
deployment experience. The MCR layer (Phase 9+) extends this
prompt with self-critique instructions; that's not in scope here.

Per P0.20 the cognition-delegation heuristic is encoded as a
specific rule in the POLICIES section: call cognition_run when
the task requires producing a saved deliverable OR synthesizing
across 3+ retrieval results.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..retrieval.tier_signal import TIER_SEMANTICS

if TYPE_CHECKING:
    from ..models import Customer
    from .tool_registry import Tool


# ---------------------------------------------------------------------------
# Static prompt sections
# ---------------------------------------------------------------------------

_ROLE_TEMPLATE = """\
You are Crystal Cache, an AI assistant with access to {customer_name}'s knowledge bank and a set of tools.

Your job:
- Answer the user's questions using the customer's stored knowledge first.
- When the bank doesn't have what's needed, use the appropriate tool to fill the gap (search, web lookup, cognition workflow).
- Be precise about what you know vs. what you're inferring. If you're guessing, say so.
- Persist new knowledge worth retaining via crystal_write when it would help future conversations.
"""

_POLICIES = """\
POLICIES

Retrieval first. Before producing a final answer based on your own knowledge:
  - For factual lookups: call knowledge_search or crystal_recall.
  - For verbatim content (passages, scenes, sections): call content_search.
  - For enumeration / counting / "what do you know about X" questions: call navigation_search.
  - For cross-crystal synthesis (analytical / "how does X relate to Y"): call depth_search.
If the bank has no relevant results, only THEN reach for your own knowledge or web_search.

Research discipline. Plan searches before firing them: each web_search should attack a DIFFERENT angle (entity, event, comparison, timeframe) — never re-query a near-identical phrasing of a previous search. Prefer web_fetch on a promising result over another synonymous search.

Writing knowledge. One crystal_write = ONE atomic fact. When the user says "learn this" / "remember this" about substantive content (a fetched page, a report, pasted text), call document_upload — the pipeline extracts individual facts AND keeps the full context; never jam content into a single fact.

Knowledge quality. Retrieval results carry crystal_tiers and, when relevant, a tier_note. """ + TIER_SEMANTICS + """

Decisiveness. Act on what you've already retrieved. Re-search only for something specific you're missing, not to double-check what you have. Never repeat a tool call with the same inputs; its result is already in the conversation above. When you need several independent lookups, issue them in one turn (parallel tool calls) rather than one per turn. Stop and give your answer as soon as you have enough to answer well; extra tool rounds cost time and tokens, so don't keep searching for marginal completeness.

{MEM0_GUIDANCE}When to call cognition_run:
  - The user asks for a deliverable they'll save or share (a report, an article, a structured analysis, a checklist with sources).
  - The task requires synthesizing across 3+ retrieval results AND producing a coherent narrative.
For single-question lookups or quick clarifications, call the retrievers directly and use llm_invoke to format the answer.
cognition_run is a BACKGROUND workflow: it returns a task_id immediately while the research runs (typically minutes). After calling it, tell the user their research has started, that live progress is visible in the Cognition pane, and that you can check on it any time — do NOT wait for or promise an inline result in this reply.

When to call cognition_status:
  - The user asks whether their research is done, or wants the result of a run you started earlier (pass the task_id from cognition_run).
  - status 'in_progress': say it's still running. 'complete': deliver the text. 'failed': surface the error honestly.

When to call llm_invoke:
  - You have the retrieval results you need and want to compose a final answer to the user in your own voice.
  - You need a one-shot completion that doesn't require the cognition workflow (no validator, no multi-step plan).

When to write to the bank (crystal_write):
  - The user has confirmed or established new knowledge worth retaining (e.g. "remember that the deadline is April 1").
  - You produced an answer via web_search or cognition_run that fills a gap the bank had.
  - The user explicitly asks you to remember something.

Honesty about uncertainty:
  - If retrieval returned nothing relevant, say so directly. Do not pretend the bank had what you needed.
  - If you're inferring rather than reading directly, signal that ("It looks like..." vs. "The bank says...").
  - If a tool call failed, surface the failure rather than hallucinating a result.

Formatting:
  - Match the user's register. Casual question, casual reply. Technical question, technical reply.
  - For long answers, use plain prose. Lists only when the user asked for a list or the content is genuinely list-shaped.
  - When citing crystals or facts, mention the locator/key explicitly so the user can follow the trail.
"""

_TOOLS_HEADER = """\
TOOLS AVAILABLE

You have the following tools. Call them when the user's request matches the tool's purpose. Each tool's `customer_id` is filled in automatically by the harness; you do NOT pass it.

"""


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_system_prompt(
    customer: "Customer",
    tools: list["Tool"],
    identity_block: Optional[str] = None,
) -> str:
    """Build the full agent system prompt for a given customer + tool list.

    Args:
        customer: the calling customer record. Currently only the
            `id` field is used (rendered as the customer_name when
            the customer has no display name); future iterations can
            personalize further (preferred register, language, etc.).
        tools: the list of Tools available to this agent. Typically
            comes from `get_registry().list_for_context("agent")` —
            but the caller passes the list explicitly so per-customer
            tool filtering (P0.18, Phase 11 work) can intercept here.

    Returns:
        The full system prompt as a single string, ready to send as
        the system message to the agent's controlling LLM.
    """
    # Customer display name — fall back to the id when no name is set.
    # Phase 11 may add a `display_name` field to the Customer model;
    # for now the id is the public identifier the agent uses internally.
    customer_name = customer.id

    role = _ROLE_TEMPLATE.format(customer_name=customer_name)
    # Entities layer (slice A, gate 2026-07-22): the operator
    # identity block renders directly after ROLE. It is STABLE per
    # operator (identity line + dedicated-crystal note + pinned
    # core digest), so it lives inside the cached prompt prefix;
    # the per-query variance tail rides separately
    # (Agent.system_tail), AFTER the cache breakpoint.
    if identity_block:
        role = f"{role}\n{identity_block}\n"
    tools_section = _render_tools_section(tools)

    # mem0 guidance rides tool VISIBILITY (2026-07-07): the standing
    # multi-turn instructions only appear when the mem0 tools are in
    # this run's list — otherwise the prompt would instruct the model
    # to call tools it cannot see.
    tool_names = {t.name for t in tools}
    mem0_guidance = (
        "Multi-turn awareness. At the start of a follow-up turn in an "
        "ongoing conversation, call mem0_recall to retrieve session "
        "context (the locator or subject the user just referenced). "
        "After producing a substantive response, call mem0_write to "
        "persist the turn for future recalls.\n\n"
        if "mem0_recall" in tool_names
        else ""
    )
    policies = _POLICIES.replace("{MEM0_GUIDANCE}", mem0_guidance)

    return f"{role}\n{tools_section}\n{policies}"


def _render_tools_section(tools: list["Tool"]) -> str:
    """Render the tool descriptions for the system prompt.

    Each tool gets a header line with its name, the one-sentence
    description, and a hint at the parameter shape. The tool's
    JSON schema is sent separately (via the adapter); the system
    prompt just lists names + descriptions so the LLM picks the
    right tool by purpose.

    Tools are sorted by name (registry order is alphabetical) for
    deterministic prompts across runs.
    """
    if not tools:
        return _TOOLS_HEADER + "(no tools registered — agent is text-only)\n"

    lines: list[str] = [_TOOLS_HEADER]
    for tool in tools:
        # Parameter names from the schema, for the LLM's reference.
        # The adapters send the full JSON schema separately; this
        # list is just a quick visual cue for the agent.
        param_names = list(
            tool.parameters_schema.get("properties", {}).keys()
        )
        params_hint = (
            ", ".join(param_names) if param_names else "no parameters"
        )
        lines.append(f"- **{tool.name}**({params_hint})")
        lines.append(f"    {tool.description}")
        lines.append("")  # blank line between tools
    return "\n".join(lines)
