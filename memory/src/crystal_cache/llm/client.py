"""The provider-neutral LLM client. See this package's ``__init__`` for the why.

``complete()`` is SYNCHRONOUS — it covers the sync "SLM utility" call sites
(sparse keys today; document extraction, consolidation, reflection, inline
research / critique as they migrate). Async sites and the agent's tool-use
loop get their own seams in later slices; this one deliberately keeps to the
text-in / text-out shape those utility calls share.

Two backends sit behind one ``complete()``:
  * anthropic           — the ``anthropic`` SDK's Messages API (``system`` is
                          passed out-of-band).
  * openai              — any OpenAI-compatible ``/chat/completions`` endpoint
                          over httpx (``system`` folded in as a system message).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Tier -> default model, per provider. Call sites ask for a TIER; the client
# maps it to a provider-appropriate model so no provider-specific model string
# lives at the call site. The Anthropic tier defaults mirror the snapshots the
# codebase already used (Haiku / Sonnet / Opus in config.py). Non-Anthropic
# providers have NO universal default, so their per-tier models MUST be set via
# CC_LLM_MODEL_SMALL / _LARGE / _FRONTIER (a clear error fires otherwise).
_ANTHROPIC_TIER_DEFAULTS = {
    "small": "claude-haiku-4-5-20251001",
    "large": "claude-sonnet-4-6",
    "frontier": "claude-opus-4-8",
}

_VALID_TIERS = ("small", "large", "frontier")

# Anthropic's adaptive-thinking-only models reject sampling parameters:
# setting temperature / top_p / top_k to a non-default value returns a 400.
# They control reasoning depth via the effort parameter instead. The seam
# drops temperature for these so call sites can keep passing it uniformly.
_ANTHROPIC_NO_SAMPLING = frozenset({
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-fable-5",
    "claude-mythos-5",
})


@dataclass
class LLMResult:
    """Text plus normalized usage from one completion.

    ``model`` is the model actually used (after tier resolution). Token counts
    are normalized across providers -- Anthropic input/output/cache, OpenAI
    prompt/completion -- and are None when the provider did not report them.
    Cost sites feed these straight into record_model_call's explicit-token
    parameters, keeping metering provider-neutral.
    """

    text: str
    model: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_read_tokens: Optional[int] = None
    cache_creation_tokens: Optional[int] = None
    # Provider-normalized stop reason (2026-07-11): "max_tokens" when the
    # completion was cut by the output budget (OpenAI "length" is mapped
    # onto it), "end_turn" for a natural stop (OpenAI "stop" mapped), else
    # the provider's raw value. None when unreported (older fakes). The
    # cognition composition continuation loop keys off "max_tokens".
    stop_reason: Optional[str] = None


class LLMClient:
    """A thin provider-neutral wrapper exposing ``complete()``.

    Construct via :func:`get_llm_client` (reads settings); direct construction
    is for tests. Holds the chosen provider plus the resolved key / base URL /
    per-tier model overrides, and lazily builds the underlying SDK or HTTP
    client on first use.
    """

    def __init__(
        self,
        *,
        provider: str,
        api_key: Optional[str],
        base_url: Optional[str],
        model_small: Optional[str],
        model_large: Optional[str],
        model_frontier: Optional[str],
        vertex_project: Optional[str] = None,
        vertex_region: Optional[str] = None,
    ) -> None:
        self._provider = provider
        self._vertex_project = vertex_project
        self._vertex_region = vertex_region
        self._api_key = api_key
        self._base_url = base_url.rstrip("/") if base_url else None
        self._models = {
            "small": model_small,
            "large": model_large,
            "frontier": model_frontier,
        }
        self._anthropic_client = None  # lazy
        self._http_client = None       # lazy (openai-compatible)

    # -- model resolution ------------------------------------------------
    def _resolve_model(self, tier: str, explicit: Optional[str]) -> str:
        if explicit:
            return explicit
        if tier not in _VALID_TIERS:
            raise ValueError(
                f"unknown model tier {tier!r}; use one of {_VALID_TIERS}"
            )
        override = self._models.get(tier)
        if override:
            return override
        if self._provider == "anthropic":
            return _ANTHROPIC_TIER_DEFAULTS[tier]
        # vertex (Claude-on-Vertex) has NO universal defaults: Vertex model
        # ids are region/version-flavored (e.g. claude-sonnet-4-5@20250929),
        # so per-tier models must be configured explicitly, same as openai.
        raise ValueError(
            f"no model configured for tier {tier!r} with provider "
            f"{self._provider!r}; set CC_LLM_MODEL_{tier.upper()} "
            f"(there is no built-in default outside Anthropic)"
        )

    # -- public API ------------------------------------------------------
    def complete(
        self,
        *,
        system: Optional[str],
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float = 0.0,
        tier: str = "small",
        model: Optional[str] = None,
        json_schema: Optional[dict[str, Any]] = None,
    ) -> str:
        """Run one completion and return the assistant's text.

        ``messages`` are simple ``{role, content(str)}`` dicts (Anthropic and
        OpenAI both accept string content). ``system`` is passed out-of-band for
        Anthropic and folded in as a leading system message for an
        OpenAI-compatible provider. ``tier`` selects the model unless ``model``
        is given explicitly (in which case the caller owns provider-matching).
        ``json_schema`` requests structured output — Anthropic gets
        output_config json_schema, an OpenAI-compatible provider gets
        response_format json_schema — and the returned text is the JSON
        document.
        """
        return self.complete_detailed(
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tier=tier,
            model=model,
            json_schema=json_schema,
        ).text

    def complete_detailed(
        self,
        *,
        system: Optional[str],
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float = 0.0,
        tier: str = "small",
        model: Optional[str] = None,
        json_schema: Optional[dict[str, Any]] = None,
    ) -> "LLMResult":
        """Like :meth:`complete` but returns text plus normalized token usage.

        Cost-metering call sites use this and feed the token counts into
        record_model_call; every other site can keep using complete().
        """
        resolved = self._resolve_model(tier, model)
        if self._provider in ("anthropic", "vertex"):
            # vertex IS the Anthropic Messages API (served from Vertex AI);
            # only the client constructor differs (_get_anthropic).
            return self._complete_anthropic(
                model=resolved,
                system=system,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                json_schema=json_schema,
            )
        if self._provider == "openai":
            return self._complete_openai(
                model=resolved,
                system=system,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                json_schema=json_schema,
            )
        raise ValueError(
            f"unknown LLM provider {self._provider!r} "
            "(use 'anthropic', 'vertex', or 'openai')"
        )

    def complete_messages(
        self,
        *,
        system: Any,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        max_tokens: int,
        model: Optional[str] = None,
        tier: str = "large",
    ) -> Any:
        """Anthropic-shaped multi-turn completion for the agent loop.

        The internal representation is the Anthropic Messages shape: system
        as a string or block list, messages carrying tool_use / tool_result
        blocks, and Anthropic tool dicts (name/description/input_schema).

        - anthropic: pass-through to messages.create; the RAW SDK response
          is returned (.content blocks, .stop_reason, .usage). The caller
          owns Anthropic-only decoration (cache_control marks) and should
          apply it only when .provider is "anthropic".
        - openai: the agent adapters translate the request out (system must
          be a plain string on this path) and the completion back into an
          OpenAIChatShim that duck-types the SDK response, so the loop body
          is identical either way.

        No temperature is sent on either path (both providers default to
        1.0), matching the agent loop's historical behavior.
        """
        resolved = self._resolve_model(tier, model)
        if self._provider in ("anthropic", "vertex"):
            client = self._get_anthropic()
            kwargs: dict[str, Any] = {
                "model": resolved,
                "max_tokens": max_tokens,
                "messages": messages,
            }
            if system is not None:
                kwargs["system"] = system
            if tools:
                kwargs["tools"] = tools
            return client.messages.create(**kwargs)
        if self._provider == "openai":
            # Local import: agent.__init__ pulls the full agent surface
            # (which itself imports this package) — importing at module
            # level would cycle.
            from ..agent.adapters.openai import (
                messages_to_openai,
                parse_openai_response,
                tools_to_openai,
            )

            if not self._base_url:
                raise ValueError(
                    "openai-compatible provider needs a base URL; set "
                    "CC_LLM_BASE_URL (e.g. https://api.openai.com/v1 or "
                    "http://localhost:11434/v1)"
                )
            body: dict[str, Any] = {
                "model": resolved,
                "messages": messages_to_openai(system, messages),
                "max_tokens": max_tokens,
            }
            oai_tools = tools_to_openai(tools)
            if oai_tools:
                body["tools"] = oai_tools
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            resp = self._get_http().post(
                f"{self._base_url}/chat/completions",
                json=body,
                headers=headers,
            )
            resp.raise_for_status()
            return parse_openai_response(resp.json())
        raise ValueError(
            f"unknown LLM provider {self._provider!r} (use 'anthropic' or 'openai')"
        )

    @property
    def provider(self) -> str:
        """The configured provider name (for provider-conditional callers,
        e.g. the agent applies cache_control marks only under anthropic)."""
        return self._provider

    def is_ready(self) -> bool:
        """True when this client has what it needs to make a call.

        Anthropic needs a key; an OpenAI-compatible provider needs a key and
        a base URL. Optional-enrichment call sites use this to skip the LLM
        step cleanly when no provider is configured, rather than attempting a
        call that would fail.
        """
        if self._provider == "anthropic":
            return bool(self._api_key)
        if self._provider == "openai":
            return bool(self._api_key and self._base_url)
        return False

    # -- Anthropic backend ----------------------------------------------
    def _get_anthropic(self):
        if self._anthropic_client is None:
            import anthropic

            if self._provider == "vertex":
                # Claude-on-Vertex (2026-07-03, GCP consolidation): the SAME
                # Messages API served from Vertex AI, authenticated via GCP
                # Application Default Credentials instead of an API key —
                # needs the SDK's vertex extra (pip install anthropic[vertex]).
                if not self._vertex_project or not self._vertex_region:
                    raise ValueError(
                        "vertex provider needs CC_VERTEX_PROJECT and "
                        "CC_VERTEX_REGION (Claude regions e.g. us-east5)"
                    )
                self._anthropic_client = anthropic.AnthropicVertex(
                    project_id=self._vertex_project,
                    region=self._vertex_region,
                )
            elif self._api_key:
                self._anthropic_client = anthropic.Anthropic(api_key=self._api_key)
            else:
                # No explicit key: let the SDK read ANTHROPIC_API_KEY from the
                # environment (mirrors the prior per-call-site behavior).
                self._anthropic_client = anthropic.Anthropic()
        return self._anthropic_client

    def _complete_anthropic(
        self, *, model, system, messages, max_tokens, temperature,
        json_schema=None,
    ) -> "LLMResult":
        client = self._get_anthropic()
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        # Adaptive-thinking-only models 400 on a non-default temperature; omit
        # it for them (they use effort, not sampling). Other models get the
        # requested temperature.
        if model not in _ANTHROPIC_NO_SAMPLING:
            kwargs["temperature"] = temperature
        if system is not None:
            kwargs["system"] = system
        if json_schema is not None:
            kwargs["output_config"] = {
                "format": {"type": "json_schema", "schema": json_schema}
            }
        resp = client.messages.create(**kwargs)
        # Concatenate text blocks; tolerate non-text / empty content.
        parts = [
            getattr(b, "text", "")
            for b in (resp.content or [])
            if getattr(b, "type", None) == "text"
        ]
        # 2026-07-13 (run #8 ledger forensics): adaptive-thinking models
        # can spend the ENTIRE max_tokens on thinking blocks — content
        # is non-empty, the text join is "", output_tokens == the cap,
        # and we billed for reasoning while discarding the answer. Log
        # the block shape so this failure is diagnosable from logs; the
        # cognition seam escalates the budget on retry.
        if not any(p.strip() for p in parts) and resp.content:
            logger.warning(
                "anthropic.no_text_blocks model=%s block_types=%s stop_reason=%s",
                model,
                [getattr(b, "type", "?") for b in resp.content],
                getattr(resp, "stop_reason", None),
            )
        usage = getattr(resp, "usage", None)
        return LLMResult(
            text="".join(parts).strip(),
            model=model,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", None),
            cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", None),
            stop_reason=getattr(resp, "stop_reason", None),
        )

    # -- OpenAI-compatible backend --------------------------------------
    def _get_http(self):
        if self._http_client is None:
            import httpx

            self._http_client = httpx.Client(timeout=30.0)
        return self._http_client

    def _complete_openai(
        self, *, model, system, messages, max_tokens, temperature,
        json_schema=None,
    ) -> "LLMResult":
        if not self._base_url:
            raise ValueError(
                "openai-compatible provider needs a base URL; set "
                "CC_LLM_BASE_URL (e.g. https://api.openai.com/v1 or "
                "http://localhost:11434/v1)"
            )
        wire_messages: list[dict[str, Any]] = []
        if system is not None:
            wire_messages.append({"role": "system", "content": system})
        wire_messages.extend(messages)
        body = {
            "model": model,
            "messages": wire_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema},
            }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        client = self._get_http()
        resp = client.post(
            f"{self._base_url}/chat/completions", json=body, headers=headers
        )
        resp.raise_for_status()
        data = resp.json()
        text = (data["choices"][0]["message"]["content"] or "").strip()
        usage = data.get("usage") or {}
        finish = data["choices"][0].get("finish_reason")
        stop_reason = {"length": "max_tokens", "stop": "end_turn"}.get(
            finish, finish
        )
        return LLMResult(
            text=text,
            model=model,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            stop_reason=stop_reason,
        )


_client: Optional[LLMClient] = None


async def get_llm_client_for_customer(customer, store) -> LLMClient:
    """The per-tenant controlling-LLM seam (E4-Agent phase 2, 2026-07-06;
    ratified: the agent has everything the proxy has — this is the agent's
    get_upstream_client).

    managed  -> the process singleton (the PLATFORM's credentials), exactly
                as before.
    byok     -> a client built from the CUSTOMER's routing config: their
                provider, their Key B (decrypted), their base_url for
                self-hosted endpoints. Their key, their bill.

    Fails LOUD for a byok customer with no stored Key B — the very next
    model call would be unauthenticated; a clear 400 at the door beats an
    upstream auth error mid-run (the caller maps RuntimeError to a 400
    with a fix-it message).

    byok clients are built per-run (no cache): decrypt is cheap relative
    to a model call, and caching per-tenant credentials in process memory
    is a liability, not an optimization.
    """
    if getattr(customer, "inference_mode", "byok") == "managed":
        return get_llm_client()

    from ..config import settings
    from ..infrastructure.token_crypto import is_v2_encrypted

    cfg = customer.model_routing_config
    ref = (cfg.api_key_ref or "").strip()
    if not ref:
        raise RuntimeError(
            "This workspace uses its own provider key, but none is on "
            "file. Add your provider API key in Settings, or switch to "
            "managed inference."
        )
    # P4 (2026-07-10): enc:v2 only — tenant-scoped, AAD-bound. Anything
    # else at rest is refused (the v2 cutover nulled the single orphaned
    # v1 row; there is no legacy-decrypt path by design).
    if not is_v2_encrypted(ref):
        raise RuntimeError(
            "Stored provider key is not in the enc:v2 format — re-enter "
            "the key in Settings to store it under the tenant envelope."
        )
    api_key = await store.decrypt_tenant_secret(customer.id, "key_b", ref)

    provider = (cfg.provider or "anthropic").lower()
    if provider not in ("anthropic", "openai", "self_hosted"):
        raise RuntimeError(
            f"Unsupported provider for agent runs: {provider!r}"
        )
    return LLMClient(
        provider="openai" if provider == "self_hosted" else provider,
        api_key=api_key,
        base_url=cfg.base_url,
        # Tier models: the customer's configured model serves every tier —
        # byok tenants pick ONE model (Settings); tiered curation models
        # are a platform concern, not a tenant one.
        model_small=cfg.model_id or None,
        model_large=cfg.model_id or None,
        model_frontier=cfg.model_id or None,
        vertex_project=settings.vertex_project,
        vertex_region=settings.vertex_region,
    )


def get_llm_client() -> LLMClient:
    """Process-singleton provider-neutral LLM client, built from settings.

    Provider = ``CC_LLM_PROVIDER`` (default ``anthropic``). Key resolution: for
    Anthropic, ``CC_LLM_API_KEY`` or (back-compat) ``CC_ANTHROPIC_API_KEY`` /
    ``ANTHROPIC_API_KEY``; for an OpenAI-compatible provider, ``CC_LLM_API_KEY``
    plus ``CC_LLM_BASE_URL``. Per-tier models come from
    ``CC_LLM_MODEL_SMALL`` / ``_LARGE`` / ``_FRONTIER``.
    """
    global _client
    if _client is None:
        from ..config import settings

        provider = (settings.llm_provider or "anthropic").lower()
        if provider == "anthropic":
            api_key = settings.llm_api_key or settings.anthropic_api_key
        else:
            api_key = settings.llm_api_key
        _client = LLMClient(
            provider=provider,
            api_key=api_key,
            base_url=settings.llm_base_url,
            model_small=settings.llm_model_small,
            model_large=settings.llm_model_large,
            model_frontier=settings.llm_model_frontier,
            vertex_project=settings.vertex_project,
            vertex_region=settings.vertex_region,
        )
    return _client


def reset_llm_client() -> None:
    """Drop the cached client (tests, or after a settings change)."""
    global _client
    _client = None


def set_llm_client(client: object) -> None:
    """Install a specific client instance (tests). Pair with reset_llm_client."""
    global _client
    _client = client  # type: ignore[assignment]
