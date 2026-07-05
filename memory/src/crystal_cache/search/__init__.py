"""search/ — external web search behind a provider seam.

Mirrors the LLM seam's shape (llm/client.py): one client, provider-routed,
config-selected, test-injectable. Providers:

  * searxng — a self-hosted meta-search instance (CC_WEB_SEARCH_URL).
    No API key, no vendor: the self-host / air-gap story. Level 2
    (2026-07-02) upgrades its snippet results to full page content via
    our own SSRF-guarded fetch + extraction (fetch.py) — content-grade
    with zero external keys.
  * tavily — hosted search with extracted page CONTENT per result
    (CC_WEB_SEARCH_API_KEY). Now genuinely optional — an alternative,
    not the only content-grade path.

DIRECTION (ratified 2026-07-02): paid services are a BRIDGE, not the
destination. **Level 2 delivered the destination**: SearXNG + our own
fetch/extraction replaces tavily for the zero-paid-services end state;
the only bill is our own servers.

Unconfigured is an EXPLICIT state, never an empty-success lie: the agent
tool returns an error result the model can react to.
"""
from __future__ import annotations

from .web import (
    WebSearchClient,
    build_web_search_client,
    get_web_search_client,
    reset_web_search_client,
    set_web_search_client,
)

__all__ = [
    "WebSearchClient",
    "build_web_search_client",
    "get_web_search_client",
    "reset_web_search_client",
    "set_web_search_client",
]
