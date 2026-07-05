"""Provider-neutral LLM client — the swappable reasoning seam.

Crystal's reasoning calls (sparse-key generation, cognition, reflection,
consolidation, document extraction, inline research, critique) historically
each constructed an ``anthropic.Anthropic()`` directly, hard-coding the
provider. This package gives them ONE seam::

    from crystal_cache.llm import get_llm_client
    text = get_llm_client().complete(
        system=SYSTEM,
        messages=[{"role": "user", "content": q}],
        max_tokens=128,
        temperature=0.0,
        tier="small",
    )

The provider is chosen by config (``CC_LLM_PROVIDER``); models are selected by
TIER (small/large/frontier) so no provider-specific model string leaks into a
call site. Anthropic stays the default; any OpenAI-compatible chat-completions
endpoint (OpenAI, Groq, Together, Fireworks, a local llama.cpp / Ollama
server) is a config change, not a code change. ``CC_ANTHROPIC_API_KEY`` /
``ANTHROPIC_API_KEY`` keep working as the key when the provider is Anthropic.
"""
from __future__ import annotations

from .client import LLMClient, get_llm_client, reset_llm_client, set_llm_client

__all__ = ["LLMClient", "get_llm_client", "reset_llm_client", "set_llm_client"]
