"""Web-search seam (launch-prep sweep, 2026-07-02).

search/web.py routes searxng and tavily behind one normalized surface;
the web_search tool gates honestly on configuration, logs every search
to web_search_logs, and the crystallizer's provenance extractor parses
the new JSON shape (with the v1 repr fallback preserved).

R14 note: these assertions are verified by `pytest`; they describe
expected behavior and have not yet been run at authoring time.
"""
from __future__ import annotations

from typing import Any

from crystal_cache.learning.crystallizer import (
    ToolOutput,
    extract_web_search_provenance,
)
from crystal_cache.search import (
    WebSearchClient,
    reset_web_search_client,
    set_web_search_client,
)
from crystal_cache.search.web import _CONTENT_CAP_CHARS


class _FakeHttp:
    """httpx-shaped fake capturing the request and returning a payload."""

    def __init__(self, payload: dict):
        self._payload = payload
        self.last_get: Any = None
        self.last_post: Any = None

    def _resp(self):
        payload = self._payload

        class _Resp:
            @staticmethod
            def raise_for_status():
                return None

            @staticmethod
            def json():
                return payload

        return _Resp()

    def get(self, url, *, params):
        self.last_get = {"url": url, "params": params}
        return self._resp()

    def post(self, url, *, json, headers):
        self.last_post = {"url": url, "json": json}
        return self._resp()


def test_searxng_parses_snippets_and_content_is_none(monkeypatch):
    # Pin the Level-2 page-fetch OFF: this test covers provider parsing
    # only; the guarded fetch upgrade is tested in test_web_fetch.py, and
    # without the pin the default knob would attempt real DNS/network on
    # the fake URLs (fail-safe, but tests must never touch the network).
    from crystal_cache.config import settings as _settings
    monkeypatch.setattr(_settings, "web_search_fetch_pages", 0)
    client = WebSearchClient(provider="searxng", base_url="http://sx:8080/")
    fake = _FakeHttp({"results": [
        {"title": "Loop tax economics", "url": "https://a.example/x",
         "content": "the lever is call count"},
        {"title": "Second", "url": "https://b.example/y", "content": "snip"},
    ]})
    client._http_client = fake

    out = client.search("loop tax", max_results=2)

    assert client.is_configured()
    assert fake.last_get["url"] == "http://sx:8080/search"
    assert fake.last_get["params"]["q"] == "loop tax"
    assert out["provider"] == "searxng"
    assert out["results"][0] == {
        "title": "Loop tax economics",
        "url": "https://a.example/x",
        "snippet": "the lever is call count",
        "content": None,
    }


def test_tavily_parses_content_and_caps_it():
    client = WebSearchClient(provider="tavily", api_key="tv-key")
    fake = _FakeHttp({"results": [
        {"title": "T", "url": "https://t.example", "content": "snip",
         "raw_content": "x" * (_CONTENT_CAP_CHARS + 500)},
    ]})
    client._http_client = fake

    out = client.search("q")

    assert fake.last_post["json"]["api_key"] == "tv-key"
    assert fake.last_post["json"]["include_raw_content"] is True
    r = out["results"][0]
    assert r["snippet"] == "snip"
    assert len(r["content"]) == _CONTENT_CAP_CHARS


def test_unconfigured_states():
    assert WebSearchClient(provider="").is_configured() is False
    assert WebSearchClient(provider="searxng").is_configured() is False  # no url
    assert WebSearchClient(provider="tavily").is_configured() is False  # no key


async def test_tool_returns_explicit_error_when_unconfigured(
    customer, tool_state,
):
    from crystal_cache.agent.tools.external import web_search

    set_web_search_client(WebSearchClient(provider=""))
    try:
        out = await web_search(customer_id=customer.id, query="anything")
    finally:
        reset_web_search_client()

    assert out["results"] == []
    assert "not configured" in out["error"]
    assert "CC_WEB_SEARCH_PROVIDER" in out["error"]


async def test_tool_searches_and_logs_the_interaction(
    store, customer, tool_state,
):
    from crystal_cache.agent.tools.external import web_search
    from crystal_cache.agent.tools.retrievers import set_tool_state

    set_tool_state(tool_state)

    client = WebSearchClient(provider="tavily", api_key="k")
    client._http_client = _FakeHttp({"results": [
        {"title": "Answer", "url": "https://ans.example", "content": "snip",
         "raw_content": "full page text"},
    ]})
    set_web_search_client(client)
    try:
        out = await web_search(customer_id=customer.id, query="what is crys")
    finally:
        reset_web_search_client()

    assert out["provider"] == "tavily"
    assert out["results"][0]["content"] == "full page text"

    logs = await store.list_web_search_logs(customer.id)
    assert len(logs) == 1
    assert logs[0]["query"] == "what is crys"
    assert logs[0]["n_results"] == 1
    # Content never lands in the log — title/url/snippet only.
    assert logs[0]["results"][0] == {
        "title": "Answer", "url": "https://ans.example", "snippet": "snip",
    }


def test_provenance_parses_v2_json_shape():
    import json as _json

    output = _json.dumps({
        "query": "q", "provider": "tavily",
        "results": [
            {"title": "A", "url": "https://a", "snippet": "s", "content": None},
            {"title": "B", "url": "https://b", "snippet": "s2", "content": "c"},
        ],
    })
    pairs = extract_web_search_provenance(
        ToolOutput(name="web_search", input={}, output=output)
    )
    assert pairs == [("A", "https://a"), ("B", "https://b")]


def test_provenance_v1_repr_fallback_still_works():
    output = "[WebSearchResultBlock(title='Old', url='https://old.example')]"
    pairs = extract_web_search_provenance(
        ToolOutput(name="web_search", input={}, output=output)
    )
    assert pairs == [("Old", "https://old.example")]
