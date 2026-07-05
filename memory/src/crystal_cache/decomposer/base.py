"""Decomposer protocol and result type.

The `Decomposer` protocol is the interface that every concrete decomposer
implements. The `DecompositionResult` is what they return.

Design notes
------------

ASYNC BY DEFAULT.
  Decomposers are expected to make network/IO calls (local LLM inference,
  hosted API, Sawyer's service). Making the protocol async from the start
  avoids painful refactors later. Stub implementations can be async-trivial.

PAYLOAD SHAPE IS LOOSE.
  We don't prescribe a strict schema for the payload dict. Different
  implementations will emit different shapes:
    - A simple intent classifier: {"intent": "solve"}
    - A full entity extractor: {"intent": "solve", "topic": "algebra",
                                 "domain": "math"}
    - A compound decomposer: {"asks": [{...}, {...}]}
  `from_decomposer_output()` in crystal_cache.dsl handles all these.

CONFIDENCE IS OPTIONAL.
  Simple decomposers won't return a confidence score. LLM-based ones
  might (parsed from log-probs or emitted by the model). None of the
  routing logic depends on confidence today, but we keep the field so
  downstream can filter low-confidence decompositions out if needed.

MODEL-NAME IS OPTIONAL.
  For logging and debugging, it's useful to know which decomposer
  produced a result. "stub:v1", "llama-3.2-3b", "sawyer-classify-route".

CONTEXT IS FOR CONVERSATION HISTORY.
  "What about last Tuesday?" is ambiguous without the prior turn.
  Passing a small context dict (e.g. previous user message, tenant
  metadata) lets richer decomposers resolve references. Stub decomposers
  can ignore it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


class DecomposerError(Exception):
    """Raised when a decomposer fails to produce a valid result.

    Callers should catch this and fall through to the text-path-only
    branch rather than 500'ing the request. Decomposer failure is a
    degraded but survivable state.
    """


@dataclass
class DecompositionResult:
    """What a decomposer returns.

    Attributes:
        payload: The structured decomposition. Consumed by
            `crystal_cache.dsl.from_decomposer_output()`. Shape is
            implementation-dependent but must be JSON-serializable.
        confidence: Optional score in [0, 1]. None if the decomposer
            doesn't produce confidence estimates.
        model_name: Optional identifier for the decomposer that
            produced this result. Useful in logs.
        raw_output: Optional raw output from the decomposer (e.g. the
            full LLM response before JSON parsing). For debugging
            schema-compliance failures.
        sub_queries: Optional list of sub-decompositions when the
            input contains multiple queries. Empty list (default)
            means single-query semantics — callers can ignore this
            field. When non-empty, callers SHOULD fan out retrieval
            for each sub-query and inject the union of results.
            Shape A of the multi-query decomposer (2026-04-24): each
            sub_query is itself a DecompositionResult with its own
            payload. Independent retrieval per sub-query, no cross-
            dependencies. Shape B (graph-structured, where later
            queries depend on earlier ones) is backlog item B22 and
            will require a richer type than this list.
        is_compound: Convenience accessor. True when sub_queries is
            non-empty. Pipeline checks this flag rather than the
            list directly so the decomposer protocol can change its
            internal storage without breaking call sites.
    """

    payload: dict[str, Any]
    confidence: Optional[float] = None
    model_name: Optional[str] = None
    raw_output: Optional[str] = None
    sub_queries: list["DecompositionResult"] = field(default_factory=list)

    @property
    def is_compound(self) -> bool:
        """True when this decomposition has sub-queries to fan out."""
        return len(self.sub_queries) > 0

    def all_payloads(self) -> list[dict[str, Any]]:
        """Return the list of payloads to retrieve against.

        For a single-query result, returns [self.payload]. For a
        compound result, returns each sub-query's payload (and NOT
        self.payload — self.payload is the umbrella decomposition,
        the sub_queries are what we actually retrieve against).

        This is the call site for retrieval fan-out:

            result = await decomposer.decompose(text)
            for payload in result.all_payloads():
                hits = retrieve(payload)
                ...
        """
        if self.is_compound:
            return [sq.payload for sq in self.sub_queries]
        return [self.payload]


@runtime_checkable
class Decomposer(Protocol):
    """Interface every decomposer implements.

    Implementations:
        - StubDecomposer: hand-coded rules for testing
        - (future) LocalLLMDecomposer: Llama 3.2 3B via llama.cpp
        - (future) HostedLLMDecomposer: Groq / Anyscale / OpenAI-compat
        - (future) SawyerDecomposer: adapter for classify-and-route output
    """

    async def decompose(
        self,
        text: str,
        *,
        context: Optional[dict[str, Any]] = None,
    ) -> DecompositionResult:
        """Decompose free-text query into a structured payload.

        Args:
            text: The user's message. Must be non-empty.
            context: Optional conversation/tenant context the decomposer
                can use for disambiguation. Implementations may ignore.

        Returns:
            A DecompositionResult whose payload can be fed to
            `from_decomposer_output()`.

        Raises:
            DecomposerError: on empty input, malformed output, or
                infrastructure failure (network, model crash, etc.).
                Callers catch this and degrade gracefully.
        """
        ...
