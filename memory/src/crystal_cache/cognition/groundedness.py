"""Groundedness assessment for cognition deliverables (C3).

The validator is barriered from step outputs by design — it sees only the
goal and the deliverable, never the retrieved facts — so it cannot tell
whether a claim is grounded in what was actually retrieved or invented by
a composition step. The idle-log incident (2026-06-09) showed the cost:
a deliverable asserting a file path and "verification status: all checks
passed" was approved at 0.85, even though the cognition toolset (crystal
search over the bank) has no filesystem access and cannot run any "check".

This module runs a deterministic, no-LLM check the ENGINE can apply at
commit time, where seeing step outputs is legitimate. It builds a corpus
from RETRIEVAL steps only (crystal_search / crystal_key_scan / web_search
— never analyze/synthesize/format, which are LLM-generated and would let
the deliverable "ground" itself in its own prose), then flags:

  - file/module paths asserted in the deliverable that do not appear in
    the retrieved corpus (likely reconstructed), and
  - self-certification phrases ("all checks passed", "signature
    confirmed", ...) that claim a verification process the toolset can't
    perform.

The result is advisory: the engine stamps it on the committed document so
the human reviewer sees it, and logs it for telemetry. It does NOT block
commit — cognition output already lands in the review queue, not live
knowledge, so the human remains the gate; this just makes the risk
visible instead of hidden behind a confident 0.85.
"""
from __future__ import annotations

import re
from typing import Any

# Retrieval step actions whose outputs count as "the retrieved corpus".
# Composition actions (analyze/synthesize/format) are LLM-generated and
# deliberately excluded — including them would let a deliverable appear
# grounded in its own reasoning.
_RETRIEVAL_ACTIONS = frozenset({"crystal_search", "crystal_key_scan", "web_search", "source_lookup"})

# File paths with a code/doc extension, e.g. crystal_cache/encoding/sparse_keys.py
_FILE_PATH_RE = re.compile(
    r"\b[\w./\\-]+\.(?:py|ts|tsx|js|jsx|md|json|ya?ml|toml|sql|txt|cfg|ini)\b",
    re.IGNORECASE,
)
# Dotted module paths of 3+ segments, e.g. crystal_cache.encoding.sparse_keys
_MODULE_PATH_RE = re.compile(r"\b[a-zA-Z_][\w]*(?:\.[a-zA-Z_][\w]*){2,}\b")

# Phrases that assert a verification process the cognition toolset cannot
# actually perform (it can only search the crystal bank). Matched
# case-insensitively as substrings.
_CERT_PHRASES = (
    "verification status",
    "all checks passed",
    "fully verified",
    "file exists",
    "signature confirmed",
    "implementation confirmed",
    "import path verified",
    "function name confirmed",
    "documentation complete",
)

_MAX_REPORTED = 10


def _retrieved_corpus(step_outputs: dict[Any, Any]) -> str:
    """Concatenate text from retrieval-step outputs only, lowercased."""
    parts: list[str] = []
    for out in (step_outputs or {}).values():
        action = getattr(out, "action", "") or ""
        if action not in _RETRIEVAL_ACTIONS:
            continue
        output = getattr(out, "output", None) or {}
        if not isinstance(output, dict):
            continue
        text = output.get("content_text") or ""
        if text:
            parts.append(str(text))
        for f in output.get("findings", []) or []:
            if isinstance(f, dict):
                parts.append(str(f.get("content", "")))
                parts.append(str(f.get("content_preview", "")))
                parts.append(str(f.get("key", "")))
        # source_lookup (read/list/search) carries grounding in op-specific
        # keys rather than content_text/findings: include the file content
        # + path (read), match paths + snippets (search), and entry names
        # (list) so a deliverable that cites what was actually read counts
        # as grounded against the real source.
        if output.get("content"):
            parts.append(str(output.get("content", "")))
        if output.get("path"):
            parts.append(str(output.get("path", "")))
        for m in output.get("matches", []) or []:
            if isinstance(m, dict):
                parts.append(str(m.get("path", "")))
                parts.append(str(m.get("text", "")))
        for e in output.get("entries", []) or []:
            if isinstance(e, dict):
                parts.append(str(e.get("name", "")))
    return "\n".join(parts).lower()


def assess_groundedness(deliverable: str, step_outputs: dict[Any, Any]) -> dict:
    """Return a deterministic groundedness verdict for a deliverable.

    Result dict:
      verdict:           "grounded" | "ungrounded"
      ungrounded_paths:  paths asserted but absent from the retrieved corpus
      cert_phrases:      self-certification phrases present in the deliverable
      had_retrieval:     whether any retrieval step contributed to the corpus
      corpus_chars:      size of the retrieved corpus (debug signal)

    "ungrounded" when there is at least one unsupported path OR any
    self-certification phrase. Heuristic and advisory by design.
    """
    text = deliverable or ""
    corpus = _retrieved_corpus(step_outputs)

    paths: set[str] = set()
    for m in _FILE_PATH_RE.findall(text):
        paths.add(m.lower())
    for m in _MODULE_PATH_RE.findall(text):
        paths.add(m.lower())

    ungrounded_paths = sorted(p for p in paths if p and p not in corpus)

    low = text.lower()
    cert_phrases = sorted({p for p in _CERT_PHRASES if p in low})

    grounded = not ungrounded_paths and not cert_phrases
    return {
        "verdict": "grounded" if grounded else "ungrounded",
        "ungrounded_paths": ungrounded_paths[:_MAX_REPORTED],
        "cert_phrases": cert_phrases,
        "had_retrieval": bool(corpus),
        "corpus_chars": len(corpus),
    }
