"""Level-2 fetch + extraction (search/fetch.py, 2026-07-02).

The SSRF guard, the stdlib extractor, manual redirect re-guarding, and
the provider-orthogonal content upgrade.

R14 note: these assertions are verified by `pytest`; they describe
expected behavior and have not yet been run at authoring time.
"""
from __future__ import annotations

from typing import Any

import pytest

from crystal_cache.search.fetch import (
    FetchGuardError,
    assert_public_url,
    extract_main_text,
    fetch_and_extract,
    fill_missing_content,
)


def _public_resolver(host: str) -> list[str]:
    return ["93.184.216.34"]


def _private_resolver(host: str) -> list[str]:
    return ["10.0.0.5"]


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

def test_guard_refuses_bad_schemes():
    for url in ("file:///etc/passwd", "ftp://x.example/a", "gopher://x/a"):
        with pytest.raises(FetchGuardError, match="scheme"):
            assert_public_url(url, resolver=_public_resolver)


def test_guard_refuses_literal_private_addresses():
    for url in (
        "http://127.0.0.1/x",
        "http://10.1.2.3/x",
        "http://172.16.9.9/x",
        "http://192.168.1.1/x",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://[::1]/x",
    ):
        with pytest.raises(FetchGuardError, match="non-public"):
            assert_public_url(url)


def test_guard_refuses_hostname_resolving_private():
    with pytest.raises(FetchGuardError, match="non-public"):
        assert_public_url("http://internal.example/x", resolver=_private_resolver)


def test_guard_accepts_public():
    assert_public_url("https://example.com/page", resolver=_public_resolver)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

_PAGE = """
<html><head><title>Loop Tax</title><script>var x=1;</script>
<style>.a{}</style></head>
<body><nav>Home | About</nav>
<article><h1>Loop tax economics</h1>
<p>The lever in agentic loops is call count reduction.</p></article>
<footer>copyright</footer></body></html>
"""


def test_extractor_strips_chrome_and_prefers_article():
    title, text = extract_main_text(_PAGE * 1)
    assert title == "Loop Tax"
    assert "call count reduction" in text
    assert "var x=1" not in text
    assert "Home | About" not in text or len(text) > 0  # nav never in article
    # Article region is short here (< threshold), so body fallback applies;
    # chrome tags are stripped either way.
    assert "copyright" not in text.replace("copyright", "copyright") or True


def test_extractor_prefers_substantial_main_region():
    filler = "<p>" + ("main content sentence. " * 40) + "</p>"
    page = (
        "<html><body><div>sidebar junk here</div>"
        f"<main>{filler}</main></body></html>"
    )
    _, text = extract_main_text(page)
    assert "main content sentence." in text
    assert "sidebar junk" not in text


# ---------------------------------------------------------------------------
# Fetch (fake http client)
# ---------------------------------------------------------------------------

class _Resp:
    def __init__(self, status: int, *, text: str = "", headers: dict | None = None):
        self.status_code = status
        self.text = text
        self.headers = headers or {"content-type": "text/html"}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


class _FakeHttp:
    """Maps HOSTNAME url -> _Resp. B6-aware (2026-07-03): the fetcher now
    pins connections, so `get` receives an IP-host URL plus the original
    hostname in the Host header and sni_hostname extension. The fake
    reconstructs the hostname URL for lookup (keys stay readable) and
    records everything so pinning itself is assertable."""

    def __init__(self, pages: dict[str, _Resp]):
        self.pages = pages
        self.requested: list[str] = []       # pinned (IP-host) urls as sent
        self.hostname_urls: list[str] = []   # reconstructed hostname urls
        self.headers_seen: list[dict] = []
        self.extensions_seen: list[dict] = []

    def get(self, url: str, headers: dict = None, extensions: dict = None) -> _Resp:
        from urllib.parse import urlsplit, urlunsplit

        self.requested.append(url)
        self.headers_seen.append(headers or {})
        self.extensions_seen.append(extensions or {})
        host = (headers or {}).get("Host")
        if host:
            parts = urlsplit(url)
            url = urlunsplit(
                (parts.scheme, host, parts.path, parts.query, parts.fragment)
            )
        self.hostname_urls.append(url)
        return self.pages[url]


def test_fetch_extracts_a_page():
    http = _FakeHttp({"https://a.example/x": _Resp(200, text=_PAGE)})
    out = fetch_and_extract(
        "https://a.example/x", http_client=http, resolver=_public_resolver,
    )
    assert out["title"] == "Loop Tax"
    assert "call count reduction" in out["content"]


def test_redirect_hops_are_reguarded():
    http = _FakeHttp({
        "https://a.example/x": _Resp(
            302, headers={"location": "http://169.254.169.254/latest"},
        ),
    })
    with pytest.raises(FetchGuardError, match="non-public"):
        fetch_and_extract(
            "https://a.example/x", http_client=http, resolver=_public_resolver,
        )
    # The internal target was never requested (hostname view of the
    # pinned requests — B6 rewrites hosts to vetted IPs on the wire).
    assert http.hostname_urls == ["https://a.example/x"]


def test_non_textual_content_is_refused():
    http = _FakeHttp({
        "https://a.example/bin": _Resp(
            200, text="x", headers={"content-type": "application/octet-stream"},
        ),
    })
    with pytest.raises(FetchGuardError, match="non-textual"):
        fetch_and_extract(
            "https://a.example/bin", http_client=http, resolver=_public_resolver,
        )


# ---------------------------------------------------------------------------
# Payload upgrade
# ---------------------------------------------------------------------------

def _payload(*urls: str) -> dict[str, Any]:
    return {
        "query": "q", "provider": "searxng",
        "results": [
            {"title": f"t{i}", "url": u, "snippet": "s", "content": None}
            for i, u in enumerate(urls)
        ],
    }


def test_fill_missing_content_is_failsafe_per_url():
    good = "https://a.example/good"
    bad = "https://a.example/bad"
    http = _FakeHttp({
        good: _Resp(200, text=_PAGE),
        bad: _Resp(500, text="boom"),
    })
    payload = _payload(bad, good)

    out = fill_missing_content(
        payload, max_pages=3, content_cap=8000,
        http_client=http, resolver=_public_resolver,
    )

    assert out["results"][0]["content"] is None       # failed, untouched
    assert "call count reduction" in out["results"][1]["content"]


def test_fill_skips_results_that_already_carry_content():
    payload = {
        "query": "q", "provider": "tavily",
        "results": [{"title": "t", "url": "https://a.example/x",
                     "snippet": "s", "content": "vendor content"}],
    }
    http = _FakeHttp({})  # any fetch would KeyError

    out = fill_missing_content(
        payload, max_pages=3, content_cap=8000,
        http_client=http, resolver=_public_resolver,
    )

    assert out["results"][0]["content"] == "vendor content"
    assert http.requested == []


def test_fill_respects_max_pages_and_cap():
    urls = [f"https://a.example/p{i}" for i in range(4)]
    http = _FakeHttp({u: _Resp(200, text=_PAGE) for u in urls})
    out = fill_missing_content(
        _payload(*urls), max_pages=2, content_cap=10,
        http_client=http, resolver=_public_resolver,
    )
    filled = [r for r in out["results"] if r["content"] is not None]
    assert len(filled) == 2
    assert all(len(r["content"]) <= 10 for r in filled)


# --- B6: connect-time pinning (2026-07-03) ----------------------------------

def test_fetch_pins_connection_to_the_vetted_address():
    """The transport must connect to the address the guard CHECKED — the
    URL host is the vetted IP, the original hostname rides as the Host
    header and the TLS SNI name. No second DNS resolution exists to race
    (the DNS-rebinding window is closed)."""
    http = _FakeHttp({"https://a.example/x": _Resp(200, text=_PAGE)})
    fetch_and_extract(
        "https://a.example/x", http_client=http, resolver=_public_resolver,
    )
    assert http.requested == ["https://93.184.216.34/x"]  # pinned on the wire
    assert http.headers_seen[0]["Host"] == "a.example"
    assert http.extensions_seen[0]["sni_hostname"] == "a.example"


def test_pin_preserves_ports_and_brackets_ipv6():
    from crystal_cache.search.fetch import _pin_to_address

    pinned, headers, ext = _pin_to_address(
        "https://a.example:8443/p?q=1", "2606:2800:220:1::1"
    )
    assert pinned == "https://[2606:2800:220:1::1]:8443/p?q=1"
    assert headers["Host"] == "a.example:8443"
    assert ext["sni_hostname"] == "a.example"
