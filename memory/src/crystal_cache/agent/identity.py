"""Operator identity composition — Entities layer slice A.

Resolves WHO an agent run is speaking with and composes the two prompt
contributions the design ratified (gate 2026-07-22, SESSION_HANDOFF 0c):

- identity_block: STABLE per operator — the identity line, the
  dedicated-crystal note, and a pinned core digest of stated/verified
  facts (~500ch). Rendered into the cached prompt prefix by
  build_system_prompt, so it rides the C1 breakpoint.
- system_tail: VARIES per run — query-relevance facts selected by
  in-memory cosine over the operator crystal's fact vectors (~300ch,
  top-4). Appended as a second system block AFTER the breakpoint by
  the agent loop, so cross-run prefix caching survives.

Q1=B+C resolution: an explicit operator_id (validated to the team and
active) wins; otherwise a tenant with EXACTLY ONE active operator IS
that operator; zero or several active operators resolve nothing —
exactly the pre-Entities behavior.

Q4 read posture (the write-path amendment is pending ratification):
the pinned digest carries only operator-stated or human-verified
facts. Agent inferences stay out of the always-on prefix and surface
only through the relevance tail.

FastAPI-free by design (the turn_finalize precedent): the endpoint
extracts the query text and passes plain values, so any surface —
HTTP, CLI, workers — can compose identity the same way, and tests
need no web machinery.

Identity must never break a run (P0.44 posture): every failure path
logs and returns (None, None).
"""

from __future__ import annotations

import math
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)

# Prompt budgets (Q3). The digest is deliberately small: identity
# context earns its place in every prompt only by staying terse.
_DIGEST_BUDGET_CHARS = 500
_TAIL_BUDGET_CHARS = 300
_TAIL_TOP_K = 4


def _cosine(a: list[float], b: list[float]) -> float:
    """Plain cosine similarity for the relevance scan.

    In-memory over an operator crystal's few dozen fact vectors — no
    index machinery by design (the gate's honest-thin ruling).
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


async def resolve_operator(
    *,
    store: Any,
    customer_id: str,
    operator_id: Optional[str],
) -> Optional[Any]:
    """Q1=B+C: explicit id (team-checked, active) or sole-active fallback."""
    if operator_id:
        candidate = await store.get_operator_by_id(operator_id)
        if (
            candidate is not None
            and candidate.team_id == customer_id
            and candidate.status == "active"
        ):
            return candidate
        return None
    ops = await store.list_operators_for_team(customer_id)
    active = [o for o in ops if o.status == "active"]
    if len(active) == 1:
        return active[0]
    return None


async def compose_identity_context(
    *,
    store: Any,
    customer_id: str,
    operator_id: Optional[str],
    query_text: str,
    encoder: Any,
) -> tuple[Optional[str], Optional[str]]:
    """Resolve the operator and compose (identity_block, system_tail).

    Returns (None, None) whenever no operator resolves or anything
    fails — the run proceeds exactly as it would have before the
    Entities layer existed.
    """
    try:
        operator = await resolve_operator(
            store=store, customer_id=customer_id, operator_id=operator_id
        )
        if operator is None:
            return None, None

        lines = [
            "OPERATOR",
            "",
            (
                f"You are speaking with {operator.display_name} "
                f'({operator.role}). When they say "I", "me", or "my", '
                f"they mean {operator.display_name}."
            ),
        ]

        digest_lines: list[str] = []
        tail: Optional[str] = None
        entity = await store.get_entity_for_operator(operator.id)
        if entity is not None and entity.crystal_id:
            facts = await store.list_facts_for_crystal(entity.crystal_id)

            # Pinned digest: stated/verified only (Q4 read posture).
            trusted = [
                f for f in facts
                if f.source_kind == "operator_stated" or f.verified_by
            ]
            budget = _DIGEST_BUDGET_CHARS
            for f in trusted:
                line = f"- {f.claim_text.strip()}"
                if budget - len(line) < 0:
                    break
                digest_lines.append(line)
                budget -= len(line) + 1

            # Variance tail: query relevance over ALL the crystal's
            # facts (inferred included — surfaced only when relevant,
            # never pinned).
            if (
                query_text
                and encoder is not None
                and hasattr(encoder, "encode_native")
            ):
                from ..encoding.executor import encode_native_async

                qvec = list(await encode_native_async(encoder, query_text))
                shown = set(digest_lines)
                scored: list[tuple[float, str]] = []
                for f in facts:
                    if not f.vector:
                        continue
                    line = f"- {f.claim_text.strip()}"
                    if line in shown:
                        continue
                    scored.append((_cosine(qvec, list(f.vector)), line))
                scored.sort(key=lambda pair: pair[0], reverse=True)
                tail_lines: list[str] = []
                tbudget = _TAIL_BUDGET_CHARS
                for score, line in scored[:_TAIL_TOP_K]:
                    if score <= 0.0:
                        break
                    if tbudget - len(line) < 0:
                        break
                    tail_lines.append(line)
                    tbudget -= len(line) + 1
                if tail_lines:
                    tail = (
                        "OPERATOR CONTEXT (relevant to this message):\n"
                        + "\n".join(tail_lines)
                    )

        lines.append("")
        lines.append(
            "You have a dedicated memory crystal for this person. It holds "
            "what you know about them; what they STATE about themselves is "
            "trusted directly."
        )
        if digest_lines:
            lines.append("")
            lines.append("What you know about them:")
            lines.extend(digest_lines)
        return "\n".join(lines), tail
    except Exception:
        logger.warning("agent.identity_compose_failed", exc_info=True)
        return None, None
