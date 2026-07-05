"""Phase C — the reflection loop: the agent learns from fail→pass.

The BCB Hard experiments (crystal-cache-v1/docs/BCB_BENCHMARK_FINDINGS.md)
located the entire +17.6pp lift in ONE knowledge form: imperative rules
derived from failures the model subsequently got past. This module ports
that mechanism into CRYS background runs: when a run's verify FAILED at
least once and the final CLI verdict PASSED, a fast model distills one
lesson, and it lands as a private fact in the project bank under
`Reflections|Area|Slug`.

Two safety properties, both operator decisions:

FACT vs PRACTICE (the discriminator). A lesson earned from a verified
fail→fix may be imperative. But the reflector is FORBIDDEN from minting
rules out of observed codebase style — "this repo does X everywhere, so
do X" is an assumption becoming law, and if X is bad practice the loop
would reinforce it with the model's own authority. To enforce this, the
reflector sees the most-related GENERAL bank patterns (the curated,
cited floor) and must yield to them: a lesson contradicting an
established pattern is recorded as a narrow descriptive project fact,
or not at all. The general bank is the conscience of the learner.

PRIVACY. Reflections are derived from the customer's code and live in
the customer's bank. There is no promotion path to the general bank —
not automated, not assisted. Generalization is operator authorship
through the seeds pipeline.
"""
from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np

REFLECTION_MODEL_MAX_TOKENS = 400
DEDUP_SIMILARITY = 0.92
CONSCIENCE_TOP_K = 5
_TAIL_CHARS = 1200

REFLECTION_SYSTEM = """You distill ONE lesson from a coding agent's run that failed verification and was then fixed.

Reply with ONLY a JSON object, no prose, in one of two forms:

{"verdict": "lesson", "area": "<one word, e.g. Testing, Imports, Schema>", "slug": "<3-6 word slug>", "text": "<the rule>"}
{"verdict": "none"}

Rules for "text":
- ONE sentence, imperative, scoped to THIS project ("In this project, ..." or naming the concrete file/system) — never a universal claim.
- It must be a lesson EARNED from the failure→fix shown: state what failed and what prevents it.
- FORBIDDEN: codifying observed codebase style as a rule. "The code already does X" is not evidence X is right. If the only justification is existing practice, reply {"verdict": "none"}.
- ESTABLISHED PATTERNS below are curated best practice. If your candidate lesson contradicts one, do NOT state it as a rule — either record the narrow descriptive fact that forced the local workaround ("This project's X requires Y because Z"), or reply {"verdict": "none"}.
- A senior engineer should be willing to state it in code review. Typos, one-off mistakes, and task-specific trivia are {"verdict": "none"}.
Reply {"verdict": "none"} freely — most failures teach nothing durable."""


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


async def _conscience_patterns(
    store: Any, encoder: Any, customer_id: str, context_text: str
) -> list[str]:
    """Top-K general-bank patterns most related to the failure context.

    Uses the Phase A subscription-aware prefix scan (facts come back
    with vectors) and ranks in-process — the bank is a few hundred
    rows, so no FVS round trip is warranted.
    """
    from crystal_cache.encoding.executor import encode_native_async

    try:
        facts = await store.list_facts_by_key_prefix(
            customer_id, key_prefix="General|"
        )
    except Exception:  # noqa: BLE001 — conscience is best-effort, never blocking
        return []
    if not facts:
        return []
    qv = np.asarray(
        await encode_native_async(encoder, context_text[:600]), dtype=np.float32
    )
    scored = []
    for f in facts:
        vec = getattr(f, "vector", None) or []
        if not vec:
            continue
        scored.append((_cosine(qv, np.asarray(vec, dtype=np.float32)), f))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [
        f"- {getattr(f, 'claim_text', '')}".strip()
        for _, f in scored[:CONSCIENCE_TOP_K]
        if getattr(f, "claim_text", "")
    ]


async def _is_duplicate(
    store: Any, encoder: Any, customer_id: str, key: str
) -> bool:
    """True when an existing reflection is ≥ DEDUP_SIMILARITY to the
    candidate — the same mistake twice should strengthen nothing;
    duplicate facts only split retrieval mass."""
    from crystal_cache.encoding.executor import encode_native_async

    existing = await store.list_facts_by_key_prefix(
        customer_id, key_prefix="Reflections|"
    )
    if not existing:
        return False
    qv = np.asarray(await encode_native_async(encoder, key), dtype=np.float32)
    for f in existing:
        vec = getattr(f, "vector", None) or []
        if vec and _cosine(qv, np.asarray(vec, dtype=np.float32)) >= DEDUP_SIMILARITY:
            return True
    return False


def _sanitize_segment(text: str) -> str:
    """Key segments must not smuggle pipes into the sparse key."""
    return " ".join(text.replace("|", " ").split()).strip() or "Misc"


async def run_reflection(
    *,
    store: Any,
    encoder: Any,
    client: Any,
    model: str,
    customer_id: str,
    task: str,
    failing_tail: str,
    diffstat: str,
) -> Optional[dict[str, str]]:
    """One reflection attempt. Returns {key, claim, fact_id, crystal_id}
    when a lesson was stored, None otherwise (refused, duplicate, or
    model/store trouble — all logged, none raised: reflection is a
    bonus pass and must never fail a run that already succeeded).
    """
    import structlog

    logger = structlog.get_logger(__name__)
    context = f"TASK: {task}\nFIRST FAILING VERIFY OUTPUT:\n{failing_tail[-_TAIL_CHARS:]}"
    try:
        patterns = await _conscience_patterns(store, encoder, customer_id, context)
        user_msg = context + f"\n\nFINAL DIFFSTAT (the fix):\n{diffstat or '(none)'}"
        if patterns:
            user_msg += "\n\nESTABLISHED PATTERNS (yield to these):\n" + "\n".join(patterns)

        import asyncio

        # Through the provider-neutral seam on the session's own client;
        # temperature 1.0 preserves the historical API-default sampling.
        raw = await asyncio.to_thread(
            client.complete,
            system=REFLECTION_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=REFLECTION_MODEL_MAX_TOKENS,
            temperature=1.0,
            model=model,
        )
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        # Take the FIRST JSON object and ignore trailing chatter — fast
        # models append prose after the object despite instructions
        # (live: '{"verdict": "none"}\n<explanation>' crashed a plain
        # json.loads at char 20).
        start = raw.find("{")
        if start < 0:
            raise ValueError("reflection reply contained no JSON object")
        parsed, _ = json.JSONDecoder().raw_decode(raw[start:])
        if parsed.get("verdict") != "lesson":
            logger.info("reflection.declined", customer_id=customer_id)
            return None
        area = _sanitize_segment(str(parsed.get("area", "")).title())
        slug = _sanitize_segment(str(parsed.get("slug", "")))
        text = " ".join(str(parsed.get("text", "")).split()).strip()
        if not text or len(text) > 400:
            logger.info("reflection.rejected_shape", customer_id=customer_id)
            return None
        key = f"Reflections|{area}|{slug}"

        if await _is_duplicate(store, encoder, customer_id, key):
            logger.info("reflection.duplicate_skipped", customer_id=customer_id, key=key)
            return None

        written = await store.add_reflection_fact(
            customer_id, key=key, claim=text, encoder=encoder
        )
        return {"key": key, "claim": text, **written}
    except Exception as e:  # noqa: BLE001 — see docstring
        logger.warning("reflection.failed", error=f"{type(e).__name__}: {e}")
        return None
