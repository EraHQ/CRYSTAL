"""Decomposer - the free-text to concept-space bridge.

See docs/DECOMPOSER_PATH.md for architecture.

Public surface:

    from crystal_cache.decomposer import (
        Decomposer, DecompositionResult, DecomposerError,
        StubDecomposer, StubRule,
        HostedLLMDecomposer, LocalLLMDecomposer,
        TracingDecomposer, JsonlTraceWriter, TraceRecord,
        DslConfigStore,
    )

The Decomposer protocol is intentionally minimal - one async method
`decompose(text, context=None) -> DecompositionResult`. This lets us
plug in multiple implementations without touching the router:

  - StubDecomposer: rule-based, for testing and development
  - HostedLLMDecomposer: any OpenAI-compatible endpoint (Groq default)
  - LocalLLMDecomposer: preset of HostedLLMDecomposer pointed at localhost
  - TracingDecomposer: wraps any other decomposer and logs training data
  - (future) SawyerDecomposer: adapter for classify-and-route output

STATUS: experimental, shipping alongside DSL v0.2.
"""
from __future__ import annotations

from crystal_cache.decomposer.base import (
    DecompositionResult,
    Decomposer,
    DecomposerError,
)
from crystal_cache.decomposer.config_store import DslConfigStore
from crystal_cache.decomposer.hosted_llm import (
    HostedLLMDecomposer,
    LocalLLMDecomposer,
    SYSTEM_PROMPT,
)
from crystal_cache.decomposer.stub import CompoundStubRule, StubDecomposer, StubRule
from crystal_cache.decomposer.tracing import (
    JsonlTraceWriter,
    RoutingOutcome,
    RoutingTraceContext,
    TraceRecord,
    TracingDecomposer,
    build_trace_writer_from_settings,
)

__all__ = [
    # Protocol + result type
    "DecompositionResult",
    "Decomposer",
    "DecomposerError",
    # Implementations
    "StubDecomposer",
    "StubRule",
    "CompoundStubRule",
    "HostedLLMDecomposer",
    "LocalLLMDecomposer",
    "SYSTEM_PROMPT",
    # Tracing (for eventual classifier training)
    "TracingDecomposer",
    "JsonlTraceWriter",
    "TraceRecord",
    "RoutingOutcome",
    "RoutingTraceContext",
    "build_trace_writer_from_settings",
    # Config store
    "DslConfigStore",
]
