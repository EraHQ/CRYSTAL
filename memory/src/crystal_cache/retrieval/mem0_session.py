"""Mem0 session memory — Phase 7 Wave 7F consolidation (D6 / BD-2).

Crystal Cache = document knowledge. Mem0 = conversation awareness.
Different layers, different jobs. This module wraps Mem0 to give the
chat proxy session-level continuity across turns of the same
conversation (anchored by sequence_id).

CONSOLIDATION NOTE (Wave 7F, 2026-05-25 — P0.9):

v1 shipped TWO Mem0 wrapper files:
  - `retrieval/v3_session_memory.py` (used by app.py, CC_MEM0_ENABLED
    env-gated, singleton via module globals)
  - `retrieval/v3_mem0.py` (NOT used by app.py — dead code at runtime,
    different API shape, different config defaults)

Per the Wave 7F inventory finding, v1's `app.py` only ever imports
`init_mem0` and `add_conversation_turn` from `v3_session_memory.py`.
`v3_mem0.py` is shipped but unreferenced. D6's framing of "two
near-identical wrappers" was wrong; the files have meaningfully
different behavior (env-gate vs always-init, singleton vs caller-held,
sequence_id-optional vs sequence_id-required, response truncation,
regex patterns, structlog event names). Per R1's clarified
discipline, this module ports the actually-used behavior from
`v3_session_memory.py`, not the dead code from `v3_mem0.py`. The
unused file's behavior is logged for Phase 11.5 review (whether to
delete or revive any of its alternate logic).

Configuration (all read from env, defaults preserved verbatim from v1):
  CC_MEM0_ENABLED         — gate ("true"/"1"/"yes" enables, default off)
  CC_MEM0_LLM_PROVIDER    — default "anthropic"
  CC_MEM0_LLM_MODEL       — default "claude-haiku-4-5-20251001"
  CC_MEM0_EMBEDDER_MODEL  — default "sentence-transformers/all-MiniLM-L6-v2"
  CC_MEM0_QDRANT_PATH     — default "./mem0_qdrant"
  ANTHROPIC_API_KEY       — required; passed through from settings

Public surface:
  init_mem0(*, anthropic_api_key=None) -> Optional[Memory]
  get_mem0() -> Optional[Memory]
  search_session_context(query_text, customer_id, sequence_id=None,
                         *, limit=5) -> dict[str, str]
  add_conversation_turn(query_text, response_text, customer_id,
                        sequence_id=None) -> None

Wire-format contracts preserved per R3:
  - Mem0 collection_name: "crystal_cache_sessions"
  - embedding_model_dims: 384 (MiniLM-L6 default)
  - structlog event names: mem0.disabled, mem0.no_api_key,
    mem0.initialized, mem0.context_found, mem0.search_failed,
    mem0.turn_added, mem0.add_failed, mem0.init_failed
  - Filter keys: user_id (customer_id), run_id (sequence_id)
  - Hint extraction regexes for downstream classifier hints:
    r'scene\s+(\d+)' → locator_prefix "Scene N"
    r'corporate\s+mistletoe', r'"([^"]+)"', r"'([^']+)'" → subject
"""
from __future__ import annotations

import os
import re
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)

# Module-level singleton state. Mirrors v1's v3_session_memory.py
# pattern: init_mem0() mutates these globals; get_mem0() reads them.
# The singleton makes init optional at first-use (callers don't have
# to hold a reference) and keeps the lifespan setup simple.
_mem0_instance: Optional[Any] = None
_mem0_enabled: bool = False


def init_mem0(*, anthropic_api_key: Optional[str] = None) -> Optional[Any]:
    """Initialize the Mem0 singleton.

    Gated on `CC_MEM0_ENABLED` env var ("true"/"1"/"yes" to enable;
    default disabled). When disabled or initialization fails, returns
    None and `get_mem0()` will continue returning None — endpoints
    check for None and degrade gracefully.

    Returns the Memory instance on success, None otherwise. Also
    mutates the module-level `_mem0_instance` and `_mem0_enabled`
    globals so subsequent `get_mem0()` calls see the same instance.
    """
    global _mem0_instance, _mem0_enabled

    if not os.environ.get("CC_MEM0_ENABLED", "").lower() in ("true", "1", "yes"):
        logger.info("mem0.disabled", note="Set CC_MEM0_ENABLED=true to enable")
        return None

    try:
        from mem0 import Memory

        api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            logger.warning("mem0.no_api_key")
            return None

        llm_provider = os.environ.get("CC_MEM0_LLM_PROVIDER", "anthropic")
        llm_model = os.environ.get("CC_MEM0_LLM_MODEL", "claude-haiku-4-5-20251001")
        embedder_model = os.environ.get("CC_MEM0_EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        qdrant_path = os.environ.get("CC_MEM0_QDRANT_PATH", "./mem0_qdrant")

        config = {
            "llm": {
                "provider": llm_provider,
                "config": {"model": llm_model, "api_key": api_key},
            },
            "embedder": {
                "provider": "huggingface",
                "config": {"model": embedder_model},
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "crystal_cache_sessions",
                    "embedding_model_dims": 384,
                    "path": qdrant_path,
                },
            },
        }

        _mem0_instance = Memory.from_config(config)
        _mem0_enabled = True
        logger.info("mem0.initialized", llm=llm_model, embedder=embedder_model)
        return _mem0_instance

    except Exception as e:
        logger.warning("mem0.init_failed", error=str(e))
        _mem0_enabled = False
        return None


def get_mem0() -> Optional[Any]:
    """Return the current Mem0 singleton (None if not initialized)."""
    return _mem0_instance


def search_session_context(
    query_text: str,
    customer_id: str,
    sequence_id: Optional[str] = None,
    *,
    limit: int = 5,
) -> dict[str, str]:
    """Search Mem0 for session context. Returns hints dict.

    Filters by both customer_id (Mem0's `user_id`) and sequence_id
    (Mem0's `run_id`, when provided) so different conversations from
    the same customer don't bleed context into each other.

    Empty dict on:
      - Mem0 not enabled
      - No matching memories
      - Search exception (logged, swallowed)
      - Memories present but no extractable hints

    Hint keys produced (when applicable):
      - "locator_prefix" (e.g. "Scene 5") — from r'scene\\s+(\\d+)'
      - "subject" (e.g. "Corporate Mistletoe") — from named-entity regex

    NOT CURRENTLY CALLED by chat_proxy as of Wave 7F. v1's
    `chat_completions` likewise never invoked this function in its
    runtime path — search-side Mem0 was a documented capability,
    never wired into the request flow. Preserved here verbatim per
    R1 so a future caller (Phase 7.5 agent, an SDK consumer, or a
    different endpoint) can pick it up without re-deriving the
    interface.
    """
    if not _mem0_enabled or _mem0_instance is None:
        return {}
    try:
        # Scope by both customer and sequence (conversation)
        # so different conversations don't bleed context.
        filters = {"user_id": customer_id}
        if sequence_id:
            filters["run_id"] = sequence_id
        results = _mem0_instance.search(
            query_text,
            filters=filters,
            top_k=limit,
        )
        if not results or not results.get("results"):
            return {}
        hints = _extract_hints_from_memories(results["results"])
        if hints:
            logger.info(
                "mem0.context_found",
                customer_id=customer_id,
                hints=hints,
                memories=len(results["results"]),
            )
        return hints
    except Exception as e:
        logger.warning("mem0.search_failed", error=str(e))
        return {}


def add_conversation_turn(
    query_text: str,
    response_text: str,
    customer_id: str,
    sequence_id: Optional[str] = None,
) -> None:
    """Feed a conversation turn to Mem0. Call fire-and-forget after the
    response is returned to the user.

    Per v1's behavior: response_text is truncated to 2000 chars before
    sending to Mem0's extractor (which is itself an LLM call). This
    keeps Mem0's fact-extraction prompt budget bounded; full responses
    can be hundreds of KB and would blow the extractor's context.

    Silent no-op if Mem0 isn't enabled. Exceptions logged and swallowed
    so a Mem0 failure can't break the user's request.
    """
    if not _mem0_enabled or _mem0_instance is None:
        return
    try:
        msgs = [
            {"role": "user", "content": query_text},
            {"role": "assistant", "content": response_text[:2000]},
        ]
        _mem0_instance.add(
            msgs,
            user_id=customer_id,
            run_id=sequence_id,
            metadata={"customer_id": customer_id},
        )
        logger.debug(
            "mem0.turn_added",
            customer_id=customer_id,
            query=query_text[:60],
        )
    except Exception as e:
        logger.warning("mem0.add_failed", error=str(e))


def _extract_hints_from_memories(memories: list[dict]) -> dict[str, str]:
    """Parse Mem0 memories into V3 classifier hints.

    Two patterns recognized today (verbatim from v1):
      - Scene numbers: r'scene\\s+(\\d+)' → "locator_prefix": "Scene N"
      - Quoted strings or known proper nouns → "subject": "..."

    Both are deliberately specific to GAIA-style benchmark traffic
    (the original validation domain). Production deployments with
    different conversational shapes can extend `_HINT_PATTERNS` or
    swap this function entirely — it's pure, no state.
    """
    hints: dict[str, str] = {}
    for mem in memories:
        text = mem.get("memory", "")
        if not text:
            continue
        tl = text.lower()
        m = re.search(r'scene\s+(\d+)', tl)
        if m and "locator_prefix" not in hints:
            hints["locator_prefix"] = f"Scene {m.group(1)}"
        for pat in [r'corporate\s+mistletoe', r'"([^"]+)"', r"'([^']+)'"]:
            m2 = re.search(pat, tl)
            if m2 and "subject" not in hints:
                hints["subject"] = (
                    m2.group(1).strip()
                    if m2.groups()
                    else m2.group(0).strip()
                )
                break
    return hints
