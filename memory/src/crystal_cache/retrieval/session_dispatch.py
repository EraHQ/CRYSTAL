"""Session dispatch — decide whether a turn needs retrieval at all.

Part of the memory blend (see docs/MEMORY_BLEND_PLAN.md, D-MB2 / D-MB3).

v2's chat proxy retrieves on every turn. That is wasteful and error-prone when
the user's message is a follow-up referencing conversation history the model
already has — a fresh vector search there often injects a mismatched crystal.
v1's `retrieve_v3` guarded against this with a follow-up detector that skipped
retrieval and let the model pull on demand via `crystal_pull_research`. This
module ports that decision onto v2's path, generalized past the original
GAIA-specific signals.

  Increment 2 (this landing): follow-up detection.
  Increment 3 (later): session consumption — seed hints from the last
  query_log for the conversation, which also strengthens this detector via
  the `session_subject` argument.
"""
from __future__ import annotations

import os
import re
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)

# A short follow-up free of explicit retrieval intent is treated as "the model
# already has what it needs." Env-overridable.
FOLLOWUP_MAX_WORDS = int(os.environ.get("CC_FOLLOWUP_MAX_WORDS", "15"))

# Domain-agnostic signals that the user is asking for NEW material — retrieval
# SHOULD run even mid-conversation. Generalized from v1 (the scene-N /
# "corporate mistletoe" benchmark specifics are intentionally dropped).
# "where is/are" is included so identity/location lookups always retrieve —
# the identity-vs-resemblance principle.
_RETRIEVAL_INTENT_PATTERNS = [
    r"\bwhat\s+is\b",
    r"\bwhat\s+are\b",
    r"\bwhat\s+do\s+you\s+know\b",
    r"\bwhere\s+is\b",
    r"\bwhere\s+are\b",
    r"\btell\s+me\s+about\b",
    r"\bshow\s+me\b",
    r"\blook\s+up\b",
    r"\bsearch\b",
    r"\bfind\b",
    r"\bexplain\b",
    r"\bdefine\b",
    r"\blist\b",
]
_RETRIEVAL_INTENT_RE = re.compile(
    "|".join(_RETRIEVAL_INTENT_PATTERNS), re.IGNORECASE
)


def _extract_last_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                return "\n".join(parts)
    return ""


def is_followup_no_retrieval_needed(
    messages: list[dict[str, Any]],
    query_text: Optional[str] = None,
    *,
    session_subject: Optional[str] = None,
) -> bool:
    """Return True when this turn is a follow-up that needs no new retrieval.

    The model already has the conversation history, and `crystal_pull_research`
    is injected, so it can request more context on demand. Skipping retrieval
    here avoids a wasteful (and often mismatched) vector search on a turn like
    "yes, give me the exact one."

    Conditions (all must hold):
      1. There is at least one assistant turn (a conversation is underway).
      2. There are at least two user turns.
      3. The query carries no explicit retrieval intent (what is / where is /
         show me / find / search / explain / ...). Those always retrieve.
      4. Either a subject was carried forward from the prior turn (Increment 3),
         or the query is short (<= FOLLOWUP_MAX_WORDS words).

    Conservative by design: when unsure it returns False (retrieve), the
    safe-but-wasteful default rather than risking a missing-context answer.
    """
    if query_text is None:
        query_text = _extract_last_user_text(messages)

    # 1. Need an assistant turn.
    if not any(m.get("role") == "assistant" for m in messages):
        return False

    # 2. Need prior user turns.
    user_turns = [m for m in messages if m.get("role") == "user"]
    if len(user_turns) < 2:
        return False

    q = (query_text or "").strip()
    if not q:
        return False

    # 3. Explicit retrieval intent → retrieve.
    if _RETRIEVAL_INTENT_RE.search(q):
        return False

    # 4a. A carried-forward subject → almost certainly a follow-up (Inc 3).
    if session_subject:
        return True

    # 4b. Short query without retrieval intent → follow-up.
    return len(q.split()) <= FOLLOWUP_MAX_WORDS


def session_subject_from_last_log(last_log: Any) -> Optional[str]:
    """Derive a carry-forward subject signal from the prior turn's QueryLog.

    Session consumption (memory blend, Inc 3; D-MB3), the DB-backed
    replacement for v1's module-global session dict. The signal is
    intentionally coarse: if the previous turn actually matched or routed
    a crystal, the conversation has an established subject, and a vague
    follow-up is probably about it. We return the prior turn's query_text
    as the subject marker — used as a truthy signal by
    is_followup_no_retrieval_needed, and available for future hint seeding.

    Returns None when there was no prior turn, or the prior turn matched
    nothing (so there's no established subject to carry forward).
    """
    if last_log is None:
        return None
    matched = bool(getattr(last_log, "matched_facts", None)) or bool(
        getattr(last_log, "routed_crystal_id", None)
    )
    if not matched:
        return None
    subject = (getattr(last_log, "query_text", "") or "").strip()
    return subject or None
