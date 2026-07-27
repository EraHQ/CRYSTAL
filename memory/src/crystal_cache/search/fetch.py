"""Own fetch + extraction — search Level 2 (BACKLOG §13, 2026-07-02).

DIRECTIVE (ratified 2026-07-02): the end state is ZERO paid external
services — the only bill is our own servers. This module is that end
state's second half: SearXNG discovers, we fetch and extract, and
snippet-only results become crystallization-grade content with no vendor
in the loop. It is provider-orthogonal — the seam applies it to any
result whose provider left `content` None, so tavily's own extracted
content is never re-fetched.

SSRF GUARD (load-bearing — the fetcher takes URLs from search results,
i.e. from the open web):
  * scheme allowlist: http/https only (file:, ftp:, gopher: refused);
  * the hostname's ENTIRE resolved address set (all A/AAAA records) must
    be public — private, loopback, link-local, multicast, reserved, and
    unspecified addresses are refused, which covers 10/8, 172.16/12,
    192.168/16, 127/8, 169.254.169.254 (cloud metadata), ::1, fc00::/7,
    fe80::/10; literal-IP hosts are checked directly;
  * redirects are followed MANUALLY and every hop is re-guarded (the
    classic bypass is a public URL 302-ing to an internal one), capped
    at 3 hops;
  * responses are streamed with a hard size cap and a request timeout,
    and only textual content-types are read.
DNS rebinding is CLOSED as of 2026-07-03 (B6): the transport connects
to the exact address the guard checked (host pinned into the URL, the
original hostname carried as Host header + TLS SNI), so there is no
second resolution to race. The threat model here is URLs surfaced
by search results rather than attacker-supplied requests.

Extraction is stdlib-only (html.parser): script/style/nav/header/footer/
aside/form dropped, <article>/<main> preferred when substantial, title
captured, whitespace collapsed. Deliberately no lxml/trafilatura — the
minimal-image hardening goal wins for v1; trafilatura is the named
drop-in upgrade if extraction quality bites (BACKLOG §13).
"""
from __future__ import annotations

import ipaddress
import socket
from html.parser import HTMLParser
from typing import Any, Callable, Optional
from urllib.parse import urljoin, urlsplit, urlunsplit

import structlog

logger = structlog.get_logger(__name__)

_ALLOWED_SCHEMES = ("http", "https")
_TEXTUAL_TYPES = ("text/html", "application/xhtml", "text/plain")
_MAX_BYTES = 2_000_000
_MAX_REDIRECTS = 3
_TIMEOUT_SECONDS = 15.0

# PDFs (2026-07-25). Government, trade and legal primary sources are
# overwhelmingly PDFs, and refusing them by content-type meant a research
# run could FIND the right primary source and not be able to read it. The
# Wren & Sparrow landed-cost run did exactly that: it surfaced the USTR
# Section 301 forced-labor FRN, fetched it, got nothing back, and cited the
# URL from a search snippet instead — which the validator then flagged as
# possible fabrication. The parser is the SAME one ingestion already trusts
# with uploaded files (ingestion/file_extract.py), not a second
# implementation. Bigger byte budget than HTML because a single HTS chapter
# exceeds 2 MB; the extracted TEXT is capped separately.
_PDF_TYPE = "application/pdf"
_MAX_PDF_BYTES = 20_000_000
# 2026-07-27 (post-OOM): raised 40k → 200k AND passed into the extractor
# as its early-stop budget. At 40k the fetch-layer cap equaled the tool's
# paging window, so paging could never walk past window one — and the
# extractor parsed EVERY page of a document whose text was then thrown
# away, which is what amplified a 20 MB tariff PDF into an OOM. Now the
# parse stops when the budget is covered; the tool windows 40k at a time
# over up to 200k.
_PDF_TEXT_CAP_CHARS = 200_000

# Tags whose subtree is page chrome / non-content.
_SKIP_TAGS = frozenset(
    ("script", "style", "noscript", "nav", "header", "footer",
     "aside", "form", "svg", "iframe", "template", "button")
)
_MAIN_TAGS = frozenset(("article", "main"))
# Prefer the <article>/<main> region only when it carries real substance.
_MAIN_MIN_CHARS = 400


class FetchGuardError(ValueError):
    """A URL was refused by the SSRF guard (never fetched)."""


def _split_url(url: str) -> tuple[str, str]:
    parts = urlsplit(url)
    scheme = (parts.scheme or "").lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise FetchGuardError(f"scheme {scheme!r} is not allowed (http/https only)")
    host = parts.hostname
    if not host:
        raise FetchGuardError("URL has no hostname")
    return scheme, host


def _resolve_all(host: str) -> list[str]:
    """All addresses the hostname resolves to (A + AAAA)."""
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    return sorted({info[4][0] for info in infos})


def assert_public_url(
    url: str,
    *,
    resolver: Optional[Callable[[str], list[str]]] = None,
) -> list[str]:
    """Raise FetchGuardError unless every address behind the URL is public.

    Returns the vetted address list so callers can CONNECT to a checked
    address directly (B6 pinning) instead of letting the HTTP client
    re-resolve — re-resolution is the DNS-rebinding window.
    `resolver` is injectable for tests; defaults to a real getaddrinfo.
    """
    _, host = _split_url(url)
    try:
        # Literal IP host: check it directly, no DNS.
        addresses = [str(ipaddress.ip_address(host))]
    except ValueError:
        try:
            addresses = (resolver or _resolve_all)(host)
        except socket.gaierror as e:
            raise FetchGuardError(f"hostname {host!r} did not resolve: {e}") from e
    if not addresses:
        raise FetchGuardError(f"hostname {host!r} resolved to no addresses")
    for raw in addresses:
        ip = ipaddress.ip_address(raw.split("%")[0])
        if (
            ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_multicast or ip.is_reserved or ip.is_unspecified
        ):
            raise FetchGuardError(
                f"refusing non-public address {ip} for host {host!r}"
            )
    return addresses


def _pin_to_address(url: str, address: str) -> tuple[str, dict, dict]:
    """Rewrite `url` to connect to the vetted `address` (B6, 2026-07-03).

    Returns (pinned_url, headers, extensions): the URL's host is replaced
    by the checked IP so the transport connects THERE (no second DNS
    resolution — the rebinding race is gone), while the original hostname
    rides along as the Host header and the TLS SNI/verification name via
    httpx's `sni_hostname` extension. Ports are preserved; IPv6 literals
    are bracketed.
    """
    parts = urlsplit(url)
    host = parts.hostname or ""
    ip = address.split("%")[0]
    ip_host = f"[{ip}]" if ":" in ip else ip
    netloc = ip_host if parts.port is None else f"{ip_host}:{parts.port}"
    pinned = urlunsplit(
        (parts.scheme, netloc, parts.path, parts.query, parts.fragment)
    )
    host_header = host if parts.port is None else f"{host}:{parts.port}"
    return pinned, {"Host": host_header}, {"sni_hostname": host}


class _TextExtractor(HTMLParser):
    """Stdlib main-text extraction (see module docstring)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._main_depth = 0
        self._in_title = False
        self.title = ""
        self._body: list[str] = []
        self._main: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _MAIN_TAGS:
            self._main_depth += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in _MAIN_TAGS and self._main_depth > 0:
            self._main_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
            return
        if self._skip_depth > 0:
            return
        text = data.strip()
        if not text:
            return
        self._body.append(text)
        if self._main_depth > 0:
            self._main.append(text)

    def text(self) -> str:
        main = " ".join(self._main)
        if len(main) >= _MAIN_MIN_CHARS:
            return main
        return " ".join(self._body)


def extract_main_text(html: str) -> tuple[str, str]:
    """(title, main_text) from an HTML document. Never raises on bad HTML —
    html.parser is tolerant; worst case is thin text."""
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 — extraction is best-effort by design
        pass
    return parser.title.strip(), parser.text()


def _get_http():
    import httpx

    return httpx.Client(timeout=_TIMEOUT_SECONDS, follow_redirects=False)


def fetch_and_extract(
    url: str,
    *,
    http_client: Any = None,
    resolver: Optional[Callable[[str], list[str]]] = None,
) -> dict[str, str]:
    """Fetch one guarded URL and extract its main text.

    Returns {"url": final_url, "title": ..., "content": ..., "kind": ...}
    where kind is "pdf" or "html". Raises FetchGuardError on guard refusal
    (including any redirect hop) and httpx errors on transport failure —
    callers own the fail-safe. Redirects are followed manually so EVERY
    hop is re-guarded.
    """
    client = http_client if http_client is not None else _get_http()
    current = url
    for _hop in range(_MAX_REDIRECTS + 1):
        addresses = assert_public_url(current, resolver=resolver)
        # B6: connect to the address we CHECKED. The guard's resolve and
        # the transport's connect used to be two separate resolutions —
        # a DNS-rebinding window. Pinning closes it.
        pinned, headers, extensions = _pin_to_address(current, addresses[0])
        resp = client.get(pinned, headers=headers, extensions=extensions)
        if resp.status_code in (301, 302, 303, 307, 308):
            location = resp.headers.get("location")
            if not location:
                raise FetchGuardError("redirect without a location header")
            current = urljoin(current, location)
            continue
        resp.raise_for_status()
        ctype = (resp.headers.get("content-type") or "").lower()

        if _PDF_TYPE in ctype:
            # Refuse on the DECLARED size first when the server offers one,
            # so an oversized document costs a header rather than 20 MB of
            # parse. A junk header is ignored, not fatal — the real body
            # size is checked either way.
            #
            # Parse inside the try, DECIDE outside it: FetchGuardError
            # subclasses ValueError, so raising within `except ValueError`
            # reach swallows the refusal one line after making it. Cost the
            # first time: a 20 MB document sailed through its own size gate.
            declared_size: Optional[int] = None
            declared = resp.headers.get("content-length")
            if declared:
                try:
                    declared_size = int(declared)
                except ValueError:
                    declared_size = None
            if declared_size is not None and declared_size > _MAX_PDF_BYTES:
                raise FetchGuardError(
                    f"pdf too large ({declared_size} bytes > "
                    f"{_MAX_PDF_BYTES})"
                )
            # .content, NOT .text — decoding binary into str produces
            # mojibake the extractor cannot read.
            raw = resp.content or b""
            if len(raw) > _MAX_PDF_BYTES:
                raise FetchGuardError(
                    f"pdf too large ({len(raw)} bytes > {_MAX_PDF_BYTES})"
                )
            from ..ingestion.file_extract import extract_text_from_pdf
            text = (
                extract_text_from_pdf(
                    raw, max_chars=_PDF_TEXT_CAP_CHARS,
                )
                or ""
            ).strip()
            logger.info(
                "web_fetch.pdf_extracted",
                url=current, bytes=len(raw), chars=len(text),
            )
            return {"url": current, "title": "", "kind": "pdf",
                    "content": text[:_PDF_TEXT_CAP_CHARS]}

        if not any(t in ctype for t in _TEXTUAL_TYPES):
            raise FetchGuardError(f"non-textual content-type {ctype!r}")
        body = resp.text[:_MAX_BYTES]
        if "text/plain" in ctype:
            return {"url": current, "title": "", "kind": "html",
                    "content": body}
        title, text = extract_main_text(body)
        return {"url": current, "title": title, "kind": "html",
                "content": text}
    raise FetchGuardError(f"too many redirects (> {_MAX_REDIRECTS}) from {url}")


# Static extractions shorter than this, or carrying an SPA marker, are
# treated as "the real page never rendered" and re-fetched through the
# headless renderer (search/render.py) when it's enabled + available.
_RENDER_MIN_STATIC_CHARS = 500
_SPA_MARKERS = (
    "error while loading",
    "please reload this page",
    "enable javascript",
    "javascript is required",
    "javascript is disabled",
)


def _looks_unrendered(content: str) -> bool:
    text = (content or "").strip()
    if len(text) < _RENDER_MIN_STATIC_CHARS:
        return True
    lowered = text.lower()
    return any(m in lowered for m in _SPA_MARKERS)


def fill_missing_content(
    payload: dict[str, Any],
    *,
    max_pages: int,
    content_cap: int,
    http_client: Any = None,
    resolver: Optional[Callable[[str], list[str]]] = None,
    deadline_seconds: float = 0.0,
    render_enabled: bool = False,
    render_timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    """Upgrade a seam search payload in place: fetch + extract for up to
    `max_pages` results whose provider left `content` None.

    Provider-orthogonal by construction (tavily results already carry
    content and are skipped). FAIL-SAFE PER URL — a guard refusal or
    transport error on one page logs and moves on; search results never
    get worse because a page was bad.

    Deadline (2026-07-11, ratified Q3A): `deadline_seconds` is a
    WALL-CLOCK budget for the whole enrichment pass, checked before each
    page. Before this there was no total bound — 15s per redirect HOP,
    sequential pages, and failures not consuming max_pages slots stacked
    to 200-250s per search (rematch #5 forensics). Pages that don't fit
    keep their snippet. 0 = unbounded (the old behavior).

    Render fallback (2026-07-11, ratified Q2A): when the static extract
    looks unrendered (thin, or SPA markers — GitHub-class pages assemble
    their payload with JS), the page is re-fetched through headless
    Chromium (search/render.py) within the same deadline. Render
    failure/unavailability keeps the static result.
    """
    if max_pages <= 0:
        return payload
    import time as _time
    started = _time.monotonic()

    def _remaining() -> float:
        if deadline_seconds <= 0:
            return float("inf")
        return deadline_seconds - (_time.monotonic() - started)

    fetched = 0
    client = http_client if http_client is not None else _get_http()
    for result in payload.get("results", []):
        if fetched >= max_pages:
            break
        if _remaining() <= 0:
            logger.info(
                "web_fetch.deadline_reached",
                deadline_seconds=deadline_seconds, fetched=fetched,
            )
            break
        if result.get("content") is not None:
            continue
        target = result.get("url") or ""
        if not target:
            continue
        content = ""
        page_kind = "html"
        try:
            page = fetch_and_extract(
                target, http_client=client, resolver=resolver,
            )
            content = (page.get("content") or "").strip()
            page_kind = page.get("kind") or "html"
        except FetchGuardError as e:
            logger.info("web_fetch.guard_refused", url=target, reason=str(e))
            continue
        except Exception as e:  # noqa: BLE001 — one bad page never kills a search
            logger.warning(
                "web_fetch.page_failed", url=target,
                error=(str(e).splitlines() or [repr(e)])[0][:300],
            )

        # A PDF NEVER goes to the renderer (2026-07-25): a scanned or
        # image-only PDF extracts thin, which trips _looks_unrendered, and
        # pointing headless Chromium at a binary download is a guaranteed
        # waste of the deadline. Thin means "no text layer" here, not
        # "JavaScript did not run".
        if (
            render_enabled
            and page_kind != "pdf"
            and _looks_unrendered(content)
            and _remaining() > 0
        ):
            try:
                from .render import render_and_extract
                rendered = render_and_extract(
                    target,
                    timeout_seconds=min(
                        render_timeout_seconds, max(_remaining(), 1.0),
                    ),
                    resolver=resolver,
                )
                if rendered and len(
                    (rendered.get("content") or "").strip()
                ) > len(content):
                    content = rendered["content"].strip()
                    result["rendered"] = True
                    if rendered.get("salvaged"):
                        result["salvaged"] = True
            except Exception as e:  # noqa: BLE001
                logger.warning("web_render.fallback_failed",
                               url=target, error=str(e))

        if content:
            result["content"] = content[:content_cap]
            fetched += 1
    return payload
