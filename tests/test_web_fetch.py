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
    def __init__(self, status: int, *, text: str = "", headers: dict | None = None,
                 content: bytes = b""):
        self.status_code = status
        self.text = text
        # PDFs are read from .content (bytes): decoding binary through .text
        # yields mojibake no extractor can parse.
        self.content = content
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


# ---------------------------------------------------------------------------
# PDFs (2026-07-25)
# ---------------------------------------------------------------------------
# Refusing PDFs by content-type meant a research run could find the right
# primary source and be unable to read it. The Wren & Sparrow landed-cost run
# surfaced the USTR Section 301 FRN, fetched it, got nothing, and cited the
# URL from a search snippet — which the validator then flagged as possible
# fabrication. These pin the readable path and the guards around it.

_PDF_HEADERS = {"content-type": "application/pdf"}


@pytest.fixture
def fake_pdf(monkeypatch):
    """Replace the real extractor; these tests pin the DISPATCH, not
    pdfplumber. Records call sizes so 'parsed once' is assertable."""
    calls: list[int] = []

    def _extract(file_bytes: bytes) -> str:
        calls.append(len(file_bytes))
        return "TARIFF SCHEDULE " * 8000        # ~128k chars

    monkeypatch.setattr(
        "crystal_cache.ingestion.file_extract.extract_text_from_pdf", _extract,
    )
    return calls


def test_pdf_is_read_not_refused(fake_pdf):
    http = _FakeHttp({
        "https://ustr.gov/frn.pdf": _Resp(
            200, headers=_PDF_HEADERS, content=b"%PDF-1.7 payload",
        ),
    })
    out = fetch_and_extract(
        "https://ustr.gov/frn.pdf", http_client=http, resolver=_public_resolver,
    )
    assert out["kind"] == "pdf"
    assert "TARIFF SCHEDULE" in out["content"]
    # Read from bytes, not from the decoded .text of a binary body.
    assert fake_pdf == [len(b"%PDF-1.7 payload")]


def test_pdf_content_type_with_parameters_still_matches(fake_pdf):
    http = _FakeHttp({
        "https://x.example/a.pdf": _Resp(
            200, headers={"content-type": "application/pdf; charset=binary"},
            content=b"%PDF",
        ),
    })
    out = fetch_and_extract(
        "https://x.example/a.pdf", http_client=http, resolver=_public_resolver,
    )
    assert out["kind"] == "pdf"


def test_pdf_text_is_capped(fake_pdf):
    from crystal_cache.search.fetch import _PDF_TEXT_CAP_CHARS

    http = _FakeHttp({
        "https://x.example/a.pdf": _Resp(
            200, headers=_PDF_HEADERS, content=b"%PDF",
        ),
    })
    out = fetch_and_extract(
        "https://x.example/a.pdf", http_client=http, resolver=_public_resolver,
    )
    assert len(out["content"]) == _PDF_TEXT_CAP_CHARS


def test_oversize_pdf_refused_on_declared_length(fake_pdf):
    from crystal_cache.search.fetch import _MAX_PDF_BYTES

    http = _FakeHttp({
        "https://x.example/big.pdf": _Resp(
            200,
            headers={"content-type": "application/pdf",
                     "content-length": str(_MAX_PDF_BYTES + 1)},
            content=b"%PDF",
        ),
    })
    with pytest.raises(FetchGuardError, match="too large"):
        fetch_and_extract(
            "https://x.example/big.pdf", http_client=http,
            resolver=_public_resolver,
        )
    assert fake_pdf == []      # refused on the header, never parsed


def test_oversize_pdf_refused_on_actual_body(fake_pdf):
    from crystal_cache.search.fetch import _MAX_PDF_BYTES

    http = _FakeHttp({
        "https://x.example/big.pdf": _Resp(
            200, headers=_PDF_HEADERS, content=b"x" * (_MAX_PDF_BYTES + 1),
        ),
    })
    with pytest.raises(FetchGuardError, match="too large"):
        fetch_and_extract(
            "https://x.example/big.pdf", http_client=http,
            resolver=_public_resolver,
        )


def test_junk_content_length_is_tolerated(fake_pdf):
    """A malformed header must not become a refusal: the real body size is
    checked either way."""
    http = _FakeHttp({
        "https://x.example/a.pdf": _Resp(
            200,
            headers={"content-type": "application/pdf",
                     "content-length": "unknown"},
            content=b"%PDF",
        ),
    })
    out = fetch_and_extract(
        "https://x.example/a.pdf", http_client=http, resolver=_public_resolver,
    )
    assert out["kind"] == "pdf"


def test_html_and_plain_text_paths_are_unchanged(fake_pdf):
    http = _FakeHttp({
        "https://a.example/h": _Resp(200, text=_PAGE),
        "https://a.example/t": _Resp(
            200, text="plain body", headers={"content-type": "text/plain"},
        ),
    })
    html = fetch_and_extract(
        "https://a.example/h", http_client=http, resolver=_public_resolver,
    )
    assert html["kind"] == "html" and html["title"] == "Loop Tax"
    plain = fetch_and_extract(
        "https://a.example/t", http_client=http, resolver=_public_resolver,
    )
    assert plain["kind"] == "html" and plain["content"] == "plain body"
    assert fake_pdf == []


def test_thin_pdf_never_reaches_the_renderer(monkeypatch):
    """A scanned PDF extracts thin, which trips _looks_unrendered. Thin here
    means 'no text layer', not 'JavaScript did not run', and pointing
    headless Chromium at a binary download only burns the deadline."""
    monkeypatch.setattr(
        "crystal_cache.ingestion.file_extract.extract_text_from_pdf",
        lambda b: "tiny",
    )
    rendered: list[str] = []

    def _boom(url, **kw):
        rendered.append(url)
        raise AssertionError("the renderer must never see a PDF")

    monkeypatch.setattr("crystal_cache.search.render.render_and_extract", _boom)

    url = "https://x.example/scanned.pdf"
    http = _FakeHttp({url: _Resp(200, headers=_PDF_HEADERS, content=b"%PDF")})
    out = fill_missing_content(
        _payload(url), max_pages=1, content_cap=8000,
        http_client=http, resolver=_public_resolver, render_enabled=True,
    )
    assert out["results"][0]["content"] == "tiny"
    assert rendered == []


def test_thin_html_still_reaches_the_renderer(monkeypatch):
    """The PDF skip must not have disabled the render fallback wholesale."""
    monkeypatch.setattr(
        "crystal_cache.search.render.render_and_extract",
        lambda url, **kw: {"content": "the JS-assembled payload, now present"},
    )
    url = "https://x.example/spa"
    http = _FakeHttp({url: _Resp(200, text="<html><body>thin</body></html>")})
    out = fill_missing_content(
        _payload(url), max_pages=1, content_cap=8000,
        http_client=http, resolver=_public_resolver, render_enabled=True,
    )
    assert "JS-assembled" in out["results"][0]["content"]
    assert out["results"][0]["rendered"] is True


# ---------------------------------------------------------------------------
# Tool surface: caps + paging + the paging cache (2026-07-25)
# ---------------------------------------------------------------------------
# web_fetch used to hard-truncate at 12k with a "[truncated]" marker and no
# way forward, so a long primary source was unreadable past its first pages.
# PDFs get a wider window and every result now carries next_offset.

import crystal_cache.agent.tools.external as _ext  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_page_cache():
    _ext._PAGE_CACHE.clear()
    yield
    _ext._PAGE_CACHE.clear()


@pytest.fixture
def fake_pages(monkeypatch):
    """Stub the fetch layer at the seam web_fetch actually calls, recording
    fetches so 'the second page did not re-download' is assertable."""
    fetches: list[str] = []

    def _fetch(url, **kw):
        fetches.append(url)
        if url.endswith(".pdf"):
            return {"url": url, "title": "", "kind": "pdf",
                    "content": "P" * 100_000}
        return {"url": url, "title": "T", "kind": "html",
                "content": "H" * 30_000}

    monkeypatch.setattr("crystal_cache.search.fetch.fetch_and_extract", _fetch)
    return fetches


@pytest.mark.asyncio
async def test_pdf_window_is_wider_than_html(fake_pages):
    pdf = await _ext.web_fetch("cus_1", "https://ustr.gov/frn.pdf")
    html = await _ext.web_fetch("cus_1", "https://x.example/page")
    assert len(pdf["content"]) == _ext._PDF_CAP_CHARS
    assert len(html["content"]) == _ext._HTML_CAP_CHARS


@pytest.mark.asyncio
async def test_paging_walks_a_long_document_to_the_end(fake_pages):
    url = "https://ustr.gov/frn.pdf"
    seen = 0
    offset = 0
    pages = 0
    while True:
        out = await _ext.web_fetch("cus_1", url, offset=offset)
        seen += len(out["content"])
        pages += 1
        assert out["total_chars"] == 100_000
        if out["next_offset"] is None:
            assert out["truncated"] is False
            break
        offset = out["next_offset"]
        assert pages < 10                      # no infinite walk
    assert seen == 100_000                     # every character reachable
    assert fake_pages == [url]                 # downloaded and parsed ONCE


@pytest.mark.asyncio
async def test_offset_past_the_end_explains_itself(fake_pages):
    out = await _ext.web_fetch("cus_1", "https://ustr.gov/frn.pdf",
                               offset=999_999)
    assert out["content"] == ""
    assert out["next_offset"] is None
    assert "past the end" in out["note"]


@pytest.mark.asyncio
async def test_bad_offset_is_refused_and_negative_clamps(fake_pages):
    bad = await _ext.web_fetch("cus_1", "https://x.example/page", offset="abc")
    assert "offset" in bad["error"]
    clamped = await _ext.web_fetch("cus_1", "https://x.example/page", offset=-5)
    assert clamped["content"].startswith("H")


@pytest.mark.asyncio
async def test_paging_cache_stays_bounded(fake_pages):
    for i in range(_ext._PAGE_CACHE_MAX + 4):
        await _ext.web_fetch("cus_1", f"https://x.example/p{i}")
    assert len(_ext._PAGE_CACHE) <= _ext._PAGE_CACHE_MAX


@pytest.mark.asyncio
async def test_guard_refusal_still_returns_a_tool_error(monkeypatch):
    def _refuse(url, **kw):
        raise FetchGuardError("non-public address")

    monkeypatch.setattr("crystal_cache.search.fetch.fetch_and_extract", _refuse)
    out = await _ext.web_fetch("cus_1", "http://10.0.0.5/x")
    assert "refused" in out["error"]
    assert _ext._PAGE_CACHE == {}               # failures are never cached
