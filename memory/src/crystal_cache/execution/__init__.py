"""Execution layer — §7 of BUILD_PROPOSAL.md.

The real working surface:
  - text_injection: the `inject_text_context` free function + voicing
    headers (advisory / imperative / informational) + the
    sectioned-injection title — the production injection path the
    retrieval pipeline uses.
  - shadow_evaluator: opt-in sampling of MEDIUM matches to run a
    baseline alongside for ground-truth telemetry (research §4).
  - upstream_client: provider-agnostic OpenAI/Anthropic/SelfHosted
    upstream clients + streaming (the proxy's egress).

History note (launch-prep purge, 2026-07-02): v1's scaffolded
"HIGH/MEDIUM/LOW dispatch" placeholder classes (DirectAnswerResponder,
ContextInjector, HiddenStateInjectionPath, WProjection, ConfidenceGate,
FallThroughProxy) were NotImplementedError stubs and were removed. The
hidden-state / W-projection line is PARKED research, not dead — the
thesis was validated in v1 experiments and deliberately shelved to ship
the prompt-injection product first. See docs/RESEARCH_DIRECTIONS.md;
the code lives in git history and the v1 repo.
"""
from .text_injection import (
    TextInjectionPath,
    inject_text_context,
    Voicing,
    INJECTION_SYSTEM_ROLE_HEADER,
    INJECTION_SYSTEM_ROLE_HEADER_ADVISORY,
    INJECTION_SYSTEM_ROLE_HEADER_IMPERATIVE,
    INJECTION_SYSTEM_ROLE_HEADER_INFORMATIONAL,
    SECTIONED_INJECTION_TITLE,
)
from .shadow_evaluator import ShadowEvaluator
from .upstream_client import (
    UpstreamClient,
    UpstreamResponse,
    StreamChunk,
    OpenAIClient,
    AnthropicClient,
    SelfHostedClient,
    get_upstream_client,
)

__all__ = [
    # The text-injection surface (free function + voicing headers +
    # sectioned-injection title)
    "TextInjectionPath",
    "inject_text_context",
    "Voicing",
    "INJECTION_SYSTEM_ROLE_HEADER",
    "INJECTION_SYSTEM_ROLE_HEADER_ADVISORY",
    "INJECTION_SYSTEM_ROLE_HEADER_IMPERATIVE",
    "INJECTION_SYSTEM_ROLE_HEADER_INFORMATIONAL",
    "SECTIONED_INJECTION_TITLE",
    # Shadow eval (placeholder seat for the MCR shadow critic per Phase 9.5)
    "ShadowEvaluator",
    # Provider-agnostic upstream-client surface
    "UpstreamClient",
    "UpstreamResponse",
    "StreamChunk",
    "OpenAIClient",
    "AnthropicClient",
    "SelfHostedClient",
    "get_upstream_client",
]
