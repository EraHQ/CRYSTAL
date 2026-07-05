"""Ingestion-time prompt-injection screening (C2 mitigation, 2026-07-03).

The memory-poisoning risk (OWASP LLM04): a document ingested into a
crystal is recalled across future sessions, so instruction-shaped text in
that document ("ignore previous instructions and ...") becomes persistent
indirect prompt injection. C1 already fences retrieved content so the
model treats it as data; this module adds a SECOND layer at the ingestion
boundary: heuristically flag chunks whose text reads like an injection
attempt, so the pipeline can quarantine the resulting crystal. The
already-shipped tier signal then tells the model to distrust
quarantine-tier knowledge.

This is deliberately a HEURISTIC, not a guarantee. It errs toward
flagging (a false positive just lands a benign crystal in quarantine,
which the tier signal handles gracefully); it is not a content filter and
does not block ingestion. It catches the common, blatant shapes; a
determined adversary with novel phrasing can evade it, which is why it is
one layer of several (C1 fence + tier signal + this) rather than the sole
defense.
"""
from __future__ import annotations

import re

# Patterns that strongly suggest an attempt to override model behavior.
# Each is intentionally specific — matching instruction-override phrasing,
# not merely the presence of a word — to keep false positives low. Ordered
# roughly by how blatant the signal is.
_INJECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "ignore_prior",
        r"\b(ignore|disregard|forget)\b[^.\n]{0,40}\b"
        r"(previous|prior|earlier|above|all)\b[^.\n]{0,20}\b"
        r"(instruction|instructions|prompt|prompts|context|rules?)\b",
    ),
    (
        "override_instructions",
        r"\b(override|bypass|overrule)\b[^.\n]{0,30}\b"
        r"(instruction|instructions|system|prompt|guardrails?|safety)\b",
    ),
    (
        "role_reassignment",
        r"\byou\s+are\s+now\b[^.\n]{0,40}\b(a|an|the)\b",
    ),
    (
        "system_impersonation",
        r"(^|\n)\s*(system|assistant|developer)\s*[:\]]",
    ),
    (
        "new_instructions",
        r"\b(new|updated|revised)\s+(instruction|instructions|system\s+prompt|"
        r"directive|directives)\b",
    ),
    (
        "reveal_prompt",
        r"\b(reveal|print|repeat|output|show|disclose)\b[^.\n]{0,30}\b"
        r"(system\s+prompt|your\s+instructions|your\s+prompt|initial\s+prompt)\b",
    ),
    (
        "exfiltration",
        r"\b(exfiltrate|leak|send|forward|email|post)\b[^.\n]{0,40}\b"
        r"(secret|secrets|api\s*key|apikey|password|token|credential|"
        r"private\s+key)\b",
    ),
    (
        "instruction_delimiter",
        r"(\[INST\]|\[/INST\]|<\|im_start\|>|<\|system\|>|###\s*instruction)",
    ),
)

_COMPILED = tuple(
    (name, re.compile(pat, re.IGNORECASE)) for name, pat in _INJECTION_PATTERNS
)


def scan_for_injection(text: str) -> list[str]:
    """Return the names of injection patterns that matched `text`.

    Empty list = no injection signal detected. The caller decides what to
    do with a non-empty result (this module never blocks or mutates
    content). Case-insensitive.
    """
    if not text or not text.strip():
        return []
    hits: list[str] = []
    for name, rx in _COMPILED:
        if rx.search(text):
            hits.append(name)
    return hits


def looks_like_injection(text: str) -> bool:
    """True if `text` trips any injection heuristic."""
    return bool(scan_for_injection(text))
