"""HostedLLMDecomposer - OpenAI-compatible API decomposer.

Calls any OpenAI-compatible chat-completions endpoint with JSON response
mode to produce structured decomposer payloads. Defaults to Groq's
Llama 3.1 8B Instant, but the same code works against any endpoint that
speaks the OpenAI chat-completions protocol:

  - Groq (api.groq.com/openai/v1) - default
  - Local llama.cpp server (http://localhost:8080/v1)
  - Together.ai, Fireworks, Anyscale
  - OpenAI itself (api.openai.com/v1)

Implementation notes
--------------------

WHY OPENAI-COMPATIBLE
  Most small-model hosts speak this protocol. Picking one narrows the
  abstraction we have to maintain; switching providers is a base-URL
  change.

WHY JSON MODE INSTEAD OF TOOL CALLING
  Tool calling adds a layer of negotiation (tool name + arguments) that
  we don't need - we want one JSON object back, no function semantics.
  Most providers implement response_format={"type": "json_object"}
  even when they don't support full tool calling.

RETRIES
  We retry on malformed JSON and on transient HTTP errors (5xx,
  timeouts). Beyond that we raise DecomposerError and let the router
  fall through to the text-only path.

PROMPT
  Adapted from docs/DECOMPOSER_PATH.md - the prompt template section.
  Kept inline here so the module is self-contained; if we iterate on
  the prompt, this is where to do it. Future work: per-tenant prompt
  overrides so customers can tune decomposition for their domain.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

import httpx
import structlog

from crystal_cache.config import settings as default_settings
from crystal_cache.decomposer.base import (
    DecompositionResult,
    DecomposerError,
)


logger = structlog.get_logger(__name__)


# The system prompt. Adapted from DECOMPOSER_PATH.md - the production
# prompt should be iterated against real queries; this is the v1 starter.
#
# v1 (May 2026, Phase 6.3 follow-up #2 S7 attempt #3):
#   - `topic` constrained to coarse category (the bonding-relevant
#     stability problem). v0 said "the subject"; v0 + Qwen 7B at
#     temperature=0 produced "saml_login", "okta_lumora_integration",
#     "authentication" for three paraphrases of one FAQ. v1 asks for
#     a category that covers the whole question, with examples
#     showing different paraphrases collapsing to the same topic.
#   - `domain` constrained to a closed enum (v0 said "high-level
#     category" with examples; the model treated examples as
#     suggestions and emitted novel domains like "scheduling" for
#     a SOC 2 compliance question).
#   - `tone` and `urgency` removed from the schema. They were
#     optional in v0; their absence/presence randomly toggled,
#     which destabilized payload_agreement (concept-HV cosine is
#     sensitive to which role bindings are present). Downstream
#     consumers that want them can re-add via prompt overrides.
#   - Explicit consistency rule: paraphrases of the same
#     conceptual question must produce identical payloads. Small
#     models at temperature=0 mostly follow this; the rule is
#     load-bearing for write-time bonding (Phase 6.3 follow-up #2,
#     three-axis bonder's axis 3).
#   - Topic-granularity examples deliberately use UNRELATED domains
#     (cooking, scheduling, math, weather) rather than examples
#     drawn from a specific test corpus. If we showed examples
#     from the FAQ corpus we test against, the model could pattern-
#     match the corpus rather than apply the rule. Future maintainers
#     reading the trace data would correctly flag that as
#     few-shot leakage. The whole point of S7's measurement is
#     "does the rule generalize," not "does the model reproduce
#     the examples."
SYSTEM_PROMPT = """You are an intent decomposer for a routing system.

Given a user message, extract the structured intent as a single JSON object.
The payload is used for routing similar questions to the same destination,
so STABILITY across paraphrases is critical.

Schema:
  {
    "intent":  string,    // EXACTLY one of:
                          //   solve_problem, retrieve_past_skill,
                          //   retrieve_template, list_skills,
                          //   external_tool, general_chat, debug,
                          //   draft, summarize
    "topic":   string,    // a COARSE category name covering the whole
                          //   question, lowercase snake_case. NOT a
                          //   specific subtype. Use the broadest
                          //   category that the question fits in.
    "domain":  string,    // EXACTLY one of:
                          //   math, code, writing, cooking, calendar,
                          //   teaching, memory, scheduling, security,
                          //   billing, account, integration, support,
                          //   data, general
  }

CRITICAL CONSISTENCY RULE:
  If two messages ask the same conceptual question in different words,
  they MUST produce IDENTICAL payloads (same intent, same topic, same
  domain). Do not vary the topic just because the wording varies.

Topic granularity examples (the topic should match the LEFT column,
not the right):

  Same topic: "recipe"
    - "how do I make spaghetti carbonara"
    - "carbonara recipe steps"
    - "what's the best way to cook carbonara"
  (NOT separate topics like spaghetti_carbonara, carbonara_steps,
   cooking_carbonara)

  Same topic: "meeting_scheduling"
    - "book a meeting with John for Tuesday"
    - "set up a call with John next Tuesday"
    - "add a Tuesday meeting with John to my calendar"
  (NOT separate topics like book_meeting, set_up_call,
   calendar_event_creation)

  Same topic: "derivatives"
    - "what's the derivative of x squared"
    - "differentiate x^2"
    - "how do I find d/dx of x squared"
  (NOT separate topics like polynomial_derivative, x_squared_diff,
   calculus_d_dx)

  Same topic: "weather_forecast"
    - "will it rain tomorrow"
    - "what's the weather forecast for tomorrow"
    - "is tomorrow going to be wet"
  (NOT separate topics like rain_tomorrow, tomorrow_weather,
   precipitation_forecast)

The principle behind every example: when paraphrases share a
conceptual question, the topic should name the CATEGORY of the
question, not any specific phrasing the user happened to use.

Rules:
  - Use lowercase snake_case for ALL values.
  - Return ONLY the JSON object - no explanation, no markdown fences.
  - For domain: pick the closest match from the enum. Default to
    "general" only when no enum value fits.
  - For topic: prefer broader categories. If you find yourself writing
    a topic with two underscores or naming a specific product feature,
    you've gone too narrow.
  - For multi-part requests, return {"asks": [obj1, obj2, ...]}.
"""


class HostedLLMDecomposer:
    """Calls an OpenAI-compatible endpoint to decompose queries.

    Typical usage:

        from crystal_cache.decomposer import HostedLLMDecomposer

        decomp = HostedLLMDecomposer()  # reads settings.groq_api_key
        result = await decomp.decompose("help me with algebra")
        # result.payload = {"intent": "solve_problem", "topic": "algebra",
        #                   "domain": "math"}

    For a local llama.cpp server:

        decomp = HostedLLMDecomposer(
            base_url="http://localhost:8080/v1",
            api_key="not-needed",
            model="llama-3.2-3b",
        )
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        system_prompt: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else default_settings.groq_api_key
        self._base_url = (base_url or default_settings.decomposer_base_url).rstrip("/")
        self._model = model or default_settings.decomposer_model
        self._timeout = timeout_seconds or default_settings.decomposer_timeout_seconds
        self._max_retries = (
            max_retries if max_retries is not None else default_settings.decomposer_max_retries
        )
        self._system_prompt = system_prompt or SYSTEM_PROMPT
        # If a client is injected, we don't own it (don't close it).
        # Otherwise we lazily construct one on first call.
        self._client = http_client
        self._owns_client = http_client is None

        if not self._api_key:
            raise DecomposerError(
                "HostedLLMDecomposer requires an API key. Set GROQ_API_KEY "
                "(or CC_GROQ_API_KEY) in the environment, or pass api_key=..."
            )

    @property
    def model_name(self) -> str:
        return self._model

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def decompose(
        self,
        text: str,
        *,
        context: Optional[dict[str, Any]] = None,
    ) -> DecompositionResult:
        if not text or not text.strip():
            raise DecomposerError("empty text")

        user_content = self._build_user_content(text, context)
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": 512,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._base_url}/chat/completions"

        last_error: Optional[Exception] = None
        raw_output: Optional[str] = None

        for attempt in range(self._max_retries + 1):
            try:
                client = await self._get_client()
                resp = await client.post(url, json=body, headers=headers)
            except httpx.HTTPError as e:
                last_error = e
                logger.warning(
                    "decomposer.http_error",
                    attempt=attempt,
                    error=str(e),
                )
                await asyncio.sleep(_backoff(attempt))
                continue

            if resp.status_code >= 500:
                last_error = DecomposerError(
                    f"upstream {resp.status_code}: {resp.text[:200]}"
                )
                logger.warning(
                    "decomposer.server_error",
                    attempt=attempt,
                    status=resp.status_code,
                )
                await asyncio.sleep(_backoff(attempt))
                continue

            if resp.status_code >= 400:
                # 4xx is a config/auth problem - don't retry.
                raise DecomposerError(
                    f"upstream {resp.status_code}: {resp.text[:200]}"
                )

            try:
                data = resp.json()
                raw_output = data["choices"][0]["message"]["content"]
                payload = json.loads(raw_output)
            except (KeyError, IndexError, json.JSONDecodeError, ValueError) as e:
                last_error = DecomposerError(f"malformed response: {e}")
                logger.warning(
                    "decomposer.parse_error",
                    attempt=attempt,
                    error=str(e),
                    raw=raw_output[:200] if raw_output else None,
                )
                await asyncio.sleep(_backoff(attempt))
                continue

            if not isinstance(payload, dict):
                last_error = DecomposerError(
                    f"expected JSON object, got {type(payload).__name__}"
                )
                await asyncio.sleep(_backoff(attempt))
                continue

            # Normalize: lowercase all string values recursively. The
            # prompt asks for lowercase but models drift; cleanup here
            # keeps concept-space lookups consistent.
            payload = _normalize_payload(payload)

            return DecompositionResult(
                payload=payload,
                confidence=None,  # most OpenAI-compat endpoints don't return logprobs
                model_name=self._model,
                raw_output=raw_output,
            )

        # Exhausted retries.
        assert last_error is not None
        raise DecomposerError(
            f"decomposer failed after {self._max_retries + 1} attempts: {last_error}"
        ) from last_error

    def _build_user_content(
        self, text: str, context: Optional[dict[str, Any]]
    ) -> str:
        if not context:
            return f"User message:\n{text}\n\nOutput:"
        prev = context.get("previous_turn")
        if prev:
            return (
                f"Previous turn:\n{prev}\n\n"
                f"User message:\n{text}\n\nOutput:"
            )
        return f"User message:\n{text}\n\nOutput:"


def _backoff(attempt: int) -> float:
    """Exponential backoff with cap. 0.2, 0.4, 0.8, 1.6, ... seconds."""
    return min(0.2 * (2 ** attempt), 5.0)


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Lowercase string values recursively, preserving structure."""
    out: dict[str, Any] = {}
    for k, v in payload.items():
        out[k] = _normalize_value(v)
    return out


def _normalize_value(v: Any) -> Any:
    if isinstance(v, str):
        return v.lower()
    if isinstance(v, dict):
        return _normalize_payload(v)
    if isinstance(v, list):
        return [_normalize_value(item) for item in v]
    return v


# ------------------------------------------------------------
# Local LLM preset
# ------------------------------------------------------------


def LocalLLMDecomposer(
    *,
    base_url: str = "http://localhost:8080/v1",
    api_key: str = "not-needed",
    model: str = "local-model",
    **kwargs: Any,
) -> HostedLLMDecomposer:
    """Convenience factory for a locally-hosted OpenAI-compatible server.

    Llama.cpp's server, LM Studio, Ollama (via openai-compat), and
    similar all speak the OpenAI protocol on localhost. This factory
    just sets sensible defaults.

    Usage:
        decomp = LocalLLMDecomposer()  # http://localhost:8080/v1
        decomp = LocalLLMDecomposer(base_url="http://localhost:11434/v1")  # Ollama
    """
    return HostedLLMDecomposer(
        base_url=base_url,
        api_key=api_key,
        model=model,
        **kwargs,
    )
