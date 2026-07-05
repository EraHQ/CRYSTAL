"""Conversation compaction — compress old turns for long conversations.

Short conversations (under COMPACT_THRESHOLD user turns): send full history.
Long conversations: compress older turns into a summary via Mem0, keep the
last KEEP_RECENT_TURNS turns verbatim.

This bounds upstream token cost on long conversations while preserving recent
context and pushing the overflow into long-term (Mem0) memory. The upstream
LLM sees:
  1. A system message with compressed session facts (Mem0 or rule-based)
  2. The last few turns verbatim (recent context)
  3. The Crystal Cache injection (if any)

Compaction is transparent to the rest of the pipeline: it returns a modified
messages list that replaces the full history.

PROVENANCE / blend note (2026-06-09)
------------------------------------
Ported from v1's `retrieval/v3_compaction.py`, which v1 ran as STEP 0 of
`retrieve_v3`. v2 dropped the v3 pipeline (P0.10) and with it this capability —
see docs/MEMORY_MANAGEMENT_V1_VS_V2.md and docs/MEMORY_BLEND_PLAN.md (D-MB1).
This is the same logic, re-added as a clean first-class module (the `v3_`
phase prefix is dropped) and pointed at v2's consolidated `mem0_session`
wrapper. Behavior and thresholds are unchanged from v1.

Wiring: `endpoints/chat_proxy.py` calls `should_compact` / `compact_conversation`
AFTER sequence-id resolution (so the conversation id stays stable across turns)
and BEFORE `retrieve_and_inject` (so the compacted list feeds both retrieval and
the upstream call).

Agent path (C3, 2026-06-17): `agent/agent.py::Agent.run` calls
`compact_agent_trajectory` (NOT `compact_conversation`) at the top of each loop
iteration when CC_AGENT_COMPACTION is on. Two Anthropic-Messages constraints
make it a separate entry point — see that function's docstring.

Config:
  CC_COMPACT_THRESHOLD=10     Compact after this many user turns
  CC_KEEP_RECENT_TURNS=4      Keep this many recent turns verbatim
"""
from __future__ import annotations

import os
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)

# Default thresholds (env-overridable, defaults preserved verbatim from v1).
COMPACT_THRESHOLD = int(os.environ.get("CC_COMPACT_THRESHOLD", "10"))
KEEP_RECENT_TURNS = int(os.environ.get("CC_KEEP_RECENT_TURNS", "4"))


def should_compact(messages: list[dict[str, Any]]) -> bool:
    """Check if the conversation is long enough to warrant compaction."""
    user_turns = sum(1 for m in messages if m.get("role") == "user")
    return user_turns >= COMPACT_THRESHOLD


def compact_conversation(
    messages: list[dict[str, Any]],
    customer_id: str,
    sequence_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Compact a long conversation by summarizing old turns.

    Strategy:
      1. If under threshold, return messages unchanged.
      2. Otherwise:
         a. Feed older turns to Mem0 for fact extraction (if enabled).
         b. Build a summary system message from the extracted memories.
         c. Keep only the last KEEP_RECENT_TURNS turns verbatim.
         d. Return [system msgs] + [summary system msg] + [recent turns].

    Falls back to a rule-based summary when Mem0 is unavailable, so the
    capability works whether or not CC_MEM0_ENABLED is set.
    """
    if not should_compact(messages):
        return messages

    # Separate system messages from conversation turns.
    system_msgs = [m for m in messages if m.get("role") == "system"]
    conv_msgs = [m for m in messages if m.get("role") != "system"]

    total_msgs = len(conv_msgs)

    # Keep the last KEEP_RECENT_TURNS pairs (each pair = user + assistant).
    keep_count = KEEP_RECENT_TURNS * 2
    if keep_count >= total_msgs:
        return messages  # Not enough to compact.

    old_msgs = conv_msgs[:-keep_count]
    recent_msgs = conv_msgs[-keep_count:]

    logger.info(
        "compaction.triggered",
        customer_id=customer_id,
        total_messages=total_msgs,
        old_messages=len(old_msgs),
        recent_messages=len(recent_msgs),
        threshold=COMPACT_THRESHOLD,
    )

    # Try Mem0-based compaction first; fall back to rule-based.
    summary = _mem0_compact(old_msgs, customer_id, sequence_id)
    if not summary:
        summary = _rule_based_compact(old_msgs)

    compacted: list[dict[str, Any]] = []
    compacted.extend(system_msgs)
    compacted.append({
        "role": "system",
        "content": (
            "The following is a summary of the earlier part of this "
            "conversation. Use it for context but focus on the user's most "
            "recent questions.\n\n"
            f"{summary}"
        ),
    })
    compacted.extend(recent_msgs)

    logger.info(
        "compaction.complete",
        customer_id=customer_id,
        original_messages=len(messages),
        compacted_messages=len(compacted),
        summary_chars=len(summary),
    )

    return compacted


def _mem0_compact(
    old_msgs: list[dict[str, Any]],
    customer_id: str,
    sequence_id: Optional[str] = None,
) -> Optional[str]:
    """Use Mem0 to extract key facts from old turns and assemble a summary.

    Feeds the old turns to Mem0 for extraction, then searches for the
    extracted memories and assembles them into a summary string. Returns
    None when Mem0 is disabled or extraction yields nothing — the caller
    then falls back to the rule-based summary.
    """
    try:
        from .mem0_session import get_mem0
        mem0 = get_mem0()
        if mem0 is None:
            return None

        mem0_messages = [
            {
                "role": m.get("role", "user"),
                "content": str(m.get("content", ""))[:1000],  # cap per message
            }
            for m in old_msgs
        ]

        mem0.add(
            mem0_messages,
            user_id=customer_id,
            run_id=sequence_id,
            metadata={"customer_id": customer_id, "source": "compaction"},
        )

        filters = {"user_id": customer_id}
        if sequence_id:
            filters["run_id"] = sequence_id

        results = mem0.search(
            "conversation summary",
            filters=filters,
            top_k=20,
        )

        if not results or not results.get("results"):
            return None

        facts: list[str] = []
        for mem in results["results"]:
            text = mem.get("memory", "")
            if text and text not in facts:
                facts.append(text)

        if not facts:
            return None

        summary = "Earlier in this conversation:\n" + "\n".join(
            f"- {f}" for f in facts
        )

        logger.info(
            "compaction.mem0_summary",
            customer_id=customer_id,
            facts_extracted=len(facts),
            summary_chars=len(summary),
        )

        return summary

    except Exception as e:
        logger.warning("compaction.mem0_failed", error=str(e))
        return None


def _rule_based_compact(old_msgs: list[dict[str, Any]]) -> str:
    """Rule-based summary when Mem0 is unavailable.

    Extracts each user question and the first meaningful line of the
    matching assistant response to build a quick overview of what was
    discussed.
    """
    lines: list[str] = []
    i = 0
    while i < len(old_msgs):
        msg = old_msgs[i]
        if msg.get("role") == "user":
            question = str(msg.get("content", ""))[:100]
            answer_preview = ""
            if i + 1 < len(old_msgs) and old_msgs[i + 1].get("role") == "assistant":
                full_answer = str(old_msgs[i + 1].get("content", ""))
                for line in full_answer.split("\n"):
                    stripped = line.strip()
                    if stripped and len(stripped) > 10:
                        answer_preview = stripped[:120]
                        break
                i += 2
            else:
                i += 1

            if answer_preview:
                lines.append(f"Q: {question}")
                lines.append(f"A: {answer_preview}...")
            else:
                lines.append(f"Q: {question}")
        else:
            i += 1

    if not lines:
        return (
            "Earlier conversation context was discussed but details are "
            "summarized."
        )

    return "Earlier in this conversation:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent-loop compaction (C3, 2026-06-17) — Anthropic Messages shape
# ---------------------------------------------------------------------------


def _summarize_agent_turns(old_msgs: list[dict[str, Any]]) -> str:
    """Headerless rule-based summary of dropped agent-trajectory turns.

    Unlike `_rule_based_compact` (which assumes plain user/assistant *text*),
    an agent trajectory is content BLOCKS: assistant turns carry text +
    `tool_use` blocks, and tool results are `user`-role turns of `tool_result`
    blocks. Passing those to `_rule_based_compact` would stringify the block
    lists into JSON noise, so this walks the blocks and records the agent's
    narrative — its reasoning text and which tools it called (with a tool-error
    marker). v1 records tool *names*, not their outputs: stale old results cost
    tokens without continuity value, and the results that still matter live in
    the verbatim recent window.
    """
    def _first_line(s: str) -> str:
        s = s.strip()
        nl = s.find(chr(10))
        return (s if nl < 0 else s[:nl])[:160]

    lines: list[str] = []
    for msg in old_msgs:
        role = msg.get("role")
        content = msg.get("content")
        blocks = content if isinstance(content, list) else None

        if role == "assistant":
            said = ""
            called: list[str] = []
            if blocks:
                for b in blocks:
                    if not isinstance(b, dict):
                        continue
                    btype = b.get("type")
                    if btype == "text" and not said:
                        said = _first_line(b.get("text") or "")
                    elif btype == "tool_use":
                        called.append(b.get("name", "?"))
            elif isinstance(content, str):
                said = _first_line(content)
            if said:
                lines.append(f"- {said}")
            if called:
                lines.append(f"  -> called {', '.join(called)}")

        elif role == "user":
            if isinstance(content, str) and content.strip():
                lines.append(f"- user: {_first_line(content)}")
            elif blocks:
                errs = sum(
                    1 for b in blocks
                    if isinstance(b, dict)
                    and b.get("type") == "tool_result"
                    and b.get("is_error")
                )
                if errs:
                    lines.append(f"  -> {errs} tool error(s)")

    if not lines:
        return "(earlier steps produced no summarizable detail)"
    return chr(10).join(lines)


def compact_agent_trajectory(
    working: list[dict[str, Any]],
    customer_id: str,
    *,
    prior_summary: Optional[str] = None,
    keep_recent_msgs: Optional[int] = None,
) -> Optional[tuple[str, list[dict[str, Any]]]]:
    """Compact an agent tool-loop trajectory (Anthropic Messages shape).

    Returns `(summary_text, new_working)` when compaction happened, or `None`
    when the trajectory is too short to compact (caller keeps `working` as-is).

    A SEPARATE entry point from `compact_conversation` (the proxy / OpenAI-
    shaped path) because the Anthropic Messages API forces two things that path
    never faces — see docs/CRYS_COST_AND_PARITY_PLAN.md (C3):

      1. No `role:"system"` message is emitted. The summary is RETURNED for the
         caller to fold into the `system` param; Anthropic rejects system-role
         entries inside the `messages` array.
      2. The kept window respects tool_use/tool_result adjacency. The agent's
         tool results are `user`-role turns of `tool_result` blocks, and the
         API requires each `tool_use` to be immediately followed by its
         `tool_result`. A naive tail slice would routinely start on a
         `tool_result` whose `tool_use` was dropped (orphan -> 400), so the
         window is advanced to open on an assistant turn and the leading task
         turn is preserved verbatim as the anchor (so the list still opens on a
         user message).

    The summary ACCUMULATES across repeated compactions: `prior_summary` (the
    running summary from earlier events) is extended with a fresh chunk that
    covers only the newly-overflowed turns, so nothing earlier than the current
    window is forgotten while the trajectory itself stays bounded. Rule-based
    only (CC-D7): `_mem0_compact` does add-then-search-all, which returns the
    full accumulated set and would double-count against per-chunk accumulation.
    """
    if not should_compact(working):
        return None

    keep = (
        keep_recent_msgs
        if keep_recent_msgs is not None
        else KEEP_RECENT_TURNS * 2
    )

    # An agent trajectory carries no system messages (the agent sends its
    # system prompt via the `system` param), but filter defensively.
    conv = [m for m in working if m.get("role") != "system"]
    if len(conv) <= 1:
        return None

    # conv[0] is the task — a clean user turn that anchors the window so the
    # post-compaction list still opens on a user message.
    task_anchor = conv[0]
    body = conv[1:]

    # Open the kept window on an assistant turn: [task (user), assistant, ...]
    # is valid, and never starts on a tool_result whose tool_use we dropped.
    cut = max(0, len(body) - keep)
    while cut < len(body) and body[cut].get("role") != "assistant":
        cut += 1

    recent = body[cut:]
    old = body[:cut]
    if not old or not recent:
        # Nothing safely drops — no benefit; leave the trajectory unchanged.
        return None

    chunk = _summarize_agent_turns(old)
    if prior_summary:
        summary = prior_summary + chr(10) + chunk
    else:
        summary = chunk
    new_working = [task_anchor, *recent]

    logger.info(
        "compaction.agent_trajectory",
        customer_id=customer_id,
        original_messages=len(working),
        compacted_messages=len(new_working),
        old_messages=len(old),
        summary_chars=len(summary),
    )
    return summary, new_working
