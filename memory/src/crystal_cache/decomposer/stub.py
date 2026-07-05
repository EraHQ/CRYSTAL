"""StubDecomposer — rule-based decomposer for testing and development.

Returns hand-coded payloads based on lexical matching against a set of
StubRules. No LLM required. Unlocks end-to-end integration testing
without committing to an inference endpoint.

Usage:
    stub = StubDecomposer([
        StubRule(
            triggers=["algebra", "equation", "math problem"],
            payload={"intent": "solve", "topic": "algebra", "domain": "math"},
        ),
        StubRule(
            triggers=["python", "code", "debug"],
            payload={"intent": "debug", "domain": "programming"},
        ),
    ])

    result = await stub.decompose("help me with an algebra problem")
    # result.payload == {"intent": "solve", "topic": "algebra", "domain": "math"}

FALLBACK BEHAVIOR
-----------------
If no rule matches, returns a "general_chat" payload. This is the same
degraded-but-survivable mode a real LLM decomposer would fall back to
when it can't classify the query.

NOT FOR PRODUCTION ROUTING
--------------------------
Stub rules are just substring matching. Real queries need either an LLM
that can handle paraphrase + synonyms, or a trained classifier. Stub's
only job is to unblock the router and pipeline integration work so we
can wire up LLM-based decomposers later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from crystal_cache.decomposer.base import (
    DecompositionResult,
    Decomposer,
    DecomposerError,
)


@dataclass
class StubRule:
    """A single stub-decomposition rule.

    If any of `triggers` appears as a substring in the (lowercased)
    query text, this rule's `payload` is returned as the decomposition.

    Rules are evaluated in order; first match wins. This means more
    specific rules should come before more general ones.
    """

    triggers: list[str]
    payload: dict[str, Any]
    confidence: float = 1.0  # stubs are always "certain" in their matches


@dataclass
class CompoundStubRule:
    """Stub rule that emits a compound (multi-sub-query) result.

    Use this when a test wants to exercise the multi-query fan-out
    path (Shape A). When triggered, the decomposition emits a parent
    result with `sub_queries` populated by `sub_payloads`. The parent
    payload is metadata-only — callers consume sub_queries via
    `result.all_payloads()`.

    Example: a query like "compare q4 projections to my savings plan"
    has two retrieval targets, one in `org` data and one in `personal`
    data. A CompoundStubRule with two sub_payloads tests this fan-out.

    Like StubRule, this is for test/dev only. Real compound queries
    need a hosted-LLM decomposer that can actually parse the input.
    """

    triggers: list[str]
    sub_payloads: list[dict[str, Any]]
    parent_payload: dict[str, Any] = field(
        default_factory=lambda: {"intent": "compound"}
    )
    confidence: float = 1.0


DEFAULT_FALLBACK_PAYLOAD = {"intent": "general_chat"}


class StubDecomposer:
    """Rule-based decomposer. Implements the Decomposer protocol.

    Primarily used for:
      - Integration tests: set up rules that match the test queries,
        assert the router does the right thing downstream.
      - Local development: wire up the pipeline without a running LLM.
      - Fallback: if an LLM decomposer fails, a stub can ensure the
        pipeline still routes sanely for known query shapes.
    """

    def __init__(
        self,
        rules: Optional[list[StubRule]] = None,
        *,
        compound_rules: Optional[list[CompoundStubRule]] = None,
        fallback_payload: Optional[dict[str, Any]] = None,
        model_name: str = "stub:v1",
    ) -> None:
        self.rules = rules or []
        # Compound rules are evaluated BEFORE simple rules so a query
        # like "compare X to Y" doesn't get caught by a simple rule
        # matching "X" first. Order matters here — compound rules are
        # more specific by construction.
        self.compound_rules = compound_rules or []
        self.fallback_payload = fallback_payload or dict(DEFAULT_FALLBACK_PAYLOAD)
        self.model_name = model_name

    async def decompose(
        self,
        text: str,
        *,
        context: Optional[dict[str, Any]] = None,
    ) -> DecompositionResult:
        if not text or not text.strip():
            raise DecomposerError("empty text")

        haystack = text.lower()

        # Compound rules first — more specific by construction. A
        # match here returns a parent result with sub_queries
        # populated. Callers use result.all_payloads() to fan out.
        for rule in self.compound_rules:
            for trigger in rule.triggers:
                if trigger.lower() in haystack:
                    sub_results = [
                        DecompositionResult(
                            payload=dict(sub),
                            confidence=rule.confidence,
                            model_name=self.model_name,
                            raw_output=None,
                        )
                        for sub in rule.sub_payloads
                    ]
                    return DecompositionResult(
                        payload=dict(rule.parent_payload),
                        confidence=rule.confidence,
                        model_name=self.model_name,
                        raw_output=None,
                        sub_queries=sub_results,
                    )

        for rule in self.rules:
            for trigger in rule.triggers:
                if trigger.lower() in haystack:
                    return DecompositionResult(
                        payload=dict(rule.payload),  # copy so callers can't mutate our rules
                        confidence=rule.confidence,
                        model_name=self.model_name,
                        raw_output=None,
                    )

        # No rule matched — return fallback.
        return DecompositionResult(
            payload=dict(self.fallback_payload),
            confidence=0.1,  # low confidence on fallback
            model_name=self.model_name,
            raw_output=None,
        )

    def add_rule(self, rule: StubRule) -> None:
        """Append a rule. Convenience for test setup."""
        self.rules.append(rule)
