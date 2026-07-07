"""Web-search seam — provider-routed external search (launch-prep, 2026-07-02).

One normalized surface over interchangeable backends, mirroring the LLM
seam: `get_web_search_client()` singleton, `set_/reset_web_search_client()`
test injection, sync httpx transport (call sites wrap in asyncio.to_thread).

Normalized result shape (what the tool returns and the crystallizer's
provenance extractor parses):

    {
        "query": str,
        "provider": "searxng" | "tavily",
        "results": [
            {"title": str, "url": str, "snippet": str, "content": str|None},
            ...
        ],
    }

`content` is extracted page text when the provider supplies it (tavily);
None for snippet-only providers (searxng). Content is capped per result so
one long page can't flood the agent's context.
"""
from __future__ import annotations

from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)

# Per-result extracted-content cap. One knob-free constant: enough for a
# reference page's substance, small enough that five results stay well
# under the loop's context budget.
_CONTENT_CAP_CHARS = 8000

_TAVILY_ENDPOINT = "https://api.tavily.com/search"


class WebSearchClient:
    """Provider-neutral web search over sync httpx."""

    def __init__(
        self,
        *,
        provider: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_results: int = 5,
    ) -> None:
        self._provider = (provider or "").lower()
        self._base_url = base_url.rstrip("/") if base_url else None
        self._api_key = api_key
        self._max_results = max_results
        self._http_client: Any = None  # lazy; tests inject directly

    @property
    def provider(self) -> str:
        return self._provider

    def is_configured(self) -> bool:
        """True when a provider is selected AND its required knob is set."""
        if self._provider == "searxng":
            return bool(self._base_url)
        if self._provider == "tavily":
            return bool(self._api_key)
        return False

    def _get_http(self):
        if self._http_client is None:
            import httpx

            self._http_client = httpx.Client(timeout=20.0)
        return self._http_client

    def search(self, query: str, *, max_results: Optional[int] = None) -> dict[str, Any]:
        """Run one search and return the normalized payload.

        Raises on transport/parse failure — callers own the fail-safe
        (the agent loop surfaces tool errors to the model, which replans).

        Level 2 (2026-07-02): when CC_WEB_SEARCH_FETCH_PAGES > 0, results
        whose provider left `content` None are upgraded by our own guarded
        fetch + extraction (search/fetch.py) — provider-orthogonal, so
        snippet-only providers become content-grade and content-carrying
        providers are untouched. Page failures are fail-safe per URL.
        """
        n = max_results or self._max_results
        if self._provider == "searxng":
            payload = self._search_searxng(query, n)
        elif self._provider == "tavily":
            payload = self._search_tavily(query, n)
        else:
            raise ValueError(
                "web search is not configured; set CC_WEB_SEARCH_PROVIDER to "
                "searxng (with CC_WEB_SEARCH_URL) or tavily (with "
                "CC_WEB_SEARCH_API_KEY)"
            )

        from ..config import settings

        if settings.web_search_fetch_pages > 0:
            from .fetch import fill_missing_content

            payload = fill_missing_content(
                payload,
                max_pages=settings.web_search_fetch_pages,
                content_cap=_CONTENT_CAP_CHARS,
            )
        return payload

    def _search_searxng(self, query: str, n: int) -> dict[str, Any]:
        resp = self._get_http().get(
            f"{self._base_url}/search",
            params={"q": query, "format": "json"},
            headers=_auth_headers(self._base_url),
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for r in (data.get("results") or [])[:n]:
            results.append({
                "title": str(r.get("title") or ""),
                "url": str(r.get("url") or ""),
                "snippet": str(r.get("content") or ""),
                "content": None,  # snippet-only provider
            })
        return {"query": query, "provider": "searxng", "results": results}

    def _search_tavily(self, query: str, n: int) -> dict[str, Any]:
        resp = self._get_http().post(
            _TAVILY_ENDPOINT,
            json={
                "api_key": self._api_key,
                "query": query,
                "max_results": n,
                "include_raw_content": True,
            },
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for r in (data.get("results") or [])[:n]:
            raw = r.get("raw_content")
            results.append({
                "title": str(r.get("title") or ""),
                "url": str(r.get("url") or ""),
                "snippet": str(r.get("content") or ""),
                "content": str(raw)[:_CONTENT_CAP_CHARS] if raw else None,
            })
        return {"query": query, "provider": "tavily", "results": results}


# --- Backend auth (2026-07-07) -------------------------------------------
# CC_WEB_SEARCH_AUTH=google_id_token attaches a Google ID token (audience
# = the backend base URL) to backend requests — the hosted posture:
# SearXNG behind Cloud Run IAM (--no-allow-unauthenticated) instead of
# network-reachability tricks. Token fetched from the metadata server
# (google-auth, already a dependency) and cached ~50 min. Self-host
# (empty setting) sends no header — byte-identical behavior.
_id_token_cache: dict[str, tuple[str, float]] = {}


def _fetch_google_id_token(audience: str) -> str:
    """SEAM (tests monkeypatch this): mint an ID token for `audience`."""
    import google.auth.transport.requests
    from google.oauth2 import id_token as google_id_token

    return google_id_token.fetch_id_token(
        google.auth.transport.requests.Request(), audience
    )


def _auth_headers(audience: str) -> dict[str, str]:
    import time as _time

    from ..config import get_settings

    if (get_settings().web_search_auth or "").strip() != "google_id_token":
        return {}
    cached = _id_token_cache.get(audience)
    if cached and cached[1] > _time.time():
        return {"Authorization": f"Bearer {cached[0]}"}
    token = _fetch_google_id_token(audience)
    _id_token_cache[audience] = (token, _time.time() + 50 * 60)
    return {"Authorization": f"Bearer {token}"}


def build_web_search_client() -> WebSearchClient:
    """Construct the client from settings (the factory the singleton uses)."""
    from ..config import settings

    return WebSearchClient(
        provider=settings.web_search_provider,
        base_url=settings.web_search_url,
        api_key=settings.web_search_api_key,
        max_results=settings.web_search_max_results,
    )


_client: Optional[WebSearchClient] = None


def get_web_search_client() -> WebSearchClient:
    global _client
    if _client is None:
        _client = build_web_search_client()
    return _client


def set_web_search_client(client: Any) -> None:
    """Test injection — mirror of llm.set_llm_client."""
    global _client
    _client = client


def reset_web_search_client() -> None:
    global _client
    _client = None
