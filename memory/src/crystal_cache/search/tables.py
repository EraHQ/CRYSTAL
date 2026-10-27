"""HTML tables as rows (Q124 A, 2026-10-25).

web_fetch extracts a page's prose; a page that is mostly tables (a
schedule, a standings page, a price list, a directory) came back a few
records at a time. This reads every <table> on a page, including tables a
site ships inside HTML comments, into headers and rows, keeping each cell's
link when it has one (the link often carries the record's id). Standard
library only; any site, any table.
"""
from __future__ import annotations

import re
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from typing import Any, Optional

MAX_TABLES = 40
MAX_ROWS = 5000

_COMMENT_TABLE = re.compile(r"<!--(.*?)-->", re.S)


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[dict[str, Any]] = []
        self._stack: list[dict[str, Any]] = []   # nested tables
        self._row: Optional[list[dict[str, Any]]] = None
        self._row_in_head = False
        self._cell: Optional[dict[str, Any]] = None
        self._caption: Optional[list[str]] = None
        self._section = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        a = dict(attrs)
        if tag == "table":
            self._stack.append({"id": a.get("id") or "", "caption": "", "head": [], "rows": []})
        elif not self._stack:
            return
        elif tag == "caption":
            self._caption = []
        elif tag in ("thead", "tbody", "tfoot"):
            self._section = tag
        elif tag == "tr":
            self._row = []
            self._row_in_head = self._section == "thead"
        elif tag in ("td", "th") and self._row is not None:
            try:
                span = max(1, min(int(a.get("colspan") or 1), 50))
            except ValueError:
                span = 1
            self._cell = {"text": [], "href": None, "th": tag == "th", "span": span, "stat": a.get("data-stat")}
        elif tag == "a" and self._cell is not None and self._cell["href"] is None:
            self._cell["href"] = a.get("href")
        elif tag == "br" and self._cell is not None:
            self._cell["text"].append(" ")

    def handle_endtag(self, tag: str) -> None:
        if not self._stack:
            return
        t = self._stack[-1]
        if tag == "caption" and self._caption is not None:
            t["caption"] = " ".join("".join(self._caption).split())
            self._caption = None
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            c = self._cell
            text = " ".join("".join(c["text"]).split())
            for i in range(c["span"]):
                self._row.append({"text": text if i == 0 else "", "href": c["href"] if i == 0 else None,
                                  "th": c["th"], "stat": c["stat"]})
            self._cell = None
        elif tag == "tr" and self._row is not None:
            row = self._row
            self._row = None
            if not row:
                return
            all_th = all(c["th"] for c in row)
            if self._row_in_head or (all_th and not t["rows"]):
                t["head"].append(row)
            elif all_th:
                return  # a header row repeated inside the body (sites do this every 20 rows)
            elif len(t["rows"]) < MAX_ROWS:
                t["rows"].append(row)
        elif tag in ("thead", "tbody", "tfoot"):
            self._section = ""
        elif tag == "table":
            done = self._stack.pop()
            if done["rows"] and len(self.tables) < MAX_TABLES:
                self.tables.append(done)

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell["text"].append(data)
        elif self._caption is not None:
            self._caption.append(data)


def _headers(t: dict[str, Any]) -> list[str]:
    """One name per column: the last header row's text (data-stat when a
    cell has no text), else data-stat names from the first row, else
    col1, col2..."""
    width = max(len(r) for r in t["rows"])
    names: list[str] = []
    if t["head"]:
        last = t["head"][-1]
        names = [c["text"] or (c["stat"] or "") for c in last]
    elif any(c["stat"] for c in t["rows"][0]):
        names = [c["stat"] or "" for c in t["rows"][0]]
    names = (names + [""] * width)[:width]
    out: list[str] = []
    for i, n in enumerate(names):
        n = n.strip() or f"col{i + 1}"
        base, k = n, 2
        while n in out:  # repeated headers ("Pts" twice) stay addressable
            n = f"{base} ({k})"
            k += 1
        out.append(n)
    return out


def parse_tables(html: str) -> list[dict[str, Any]]:
    """Every table with data rows: {index, id, caption, headers, rows:
    [[{text, href}]], row_count}. Rows that repeat the header (sites do
    that every 20 rows) are dropped."""
    html = html or ""
    extra = "".join(m for m in _COMMENT_TABLE.findall(html) if "<table" in m)
    p = _TableParser()
    p.feed(html + extra)
    p.close()
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for t in p.tables:
        headers = _headers(t)
        rows = []
        for r in t["rows"]:
            cells = [{"text": unescape(c["text"]), "href": c["href"]} for c in r][:len(headers)]
            cells += [{"text": "", "href": None}] * (len(headers) - len(cells))
            if [c["text"] for c in cells] == headers:
                continue
            if not any(c["text"] for c in cells):
                continue
            rows.append(cells)
        sig = f"{t['id']}|{'|'.join(headers)}|{len(rows)}"
        if not rows or sig in seen:  # a table repeated in a comment counts once
            continue
        seen.add(sig)
        out.append({"index": len(out), "id": t["id"], "caption": t["caption"], "headers": headers,
                    "rows": rows, "row_count": len(rows)})
    return out


def table_preview(t: dict[str, Any], *, rows: int = 5) -> dict[str, Any]:
    return {"table": t["index"], "id": t["id"], "caption": t["caption"], "headers": t["headers"],
            "row_count": t["row_count"],
            "sample": [[c["text"] for c in r] for r in t["rows"][:rows]],
            "links": [h for h, c in zip(t["headers"], t["rows"][0]) if c["href"]],
            # The link a pattern will run on, so a pattern is written against a real one.
            "link_samples": {h: c["href"] for h, c in zip(t["headers"], t["rows"][0]) if c["href"]}}


_DATE_FORMATS = ("%Y-%m-%d", "%a, %b %d, %Y", "%a, %B %d, %Y", "%b %d, %Y", "%B %d, %Y", "%m/%d/%Y",
                 "%m/%d/%y", "%d %b %Y", "%d %B %Y", "%a %b %d %Y", "%Y%m%d", "%Y/%m/%d")


def to_iso_date(text: str) -> Optional[str]:
    s = " ".join((text or "").replace(".", "").split())
    if not s:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def pick(cell: dict[str, Any], *, link: bool = False, pattern: Optional[str] = None) -> Optional[str]:
    """One cell's value: its text, or its link; with pattern, the regex's
    first group (or whole match) found in that value."""
    value = cell.get("href") if link else cell.get("text")
    if value is None:
        return None
    if pattern:
        m = re.search(pattern, value)
        if not m:
            return None
        value = m.group(1) if m.groups() else m.group(0)
    return value


# ------------------------------------------------------------------ loading

_CACHE: dict[str, tuple[float, str, list[dict[str, Any]]]] = {}
_CACHE_SECONDS = 600.0
_CACHE_MAX = 32


class TablesError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


async def load_tables(url: str, *, fetch: Any = None) -> tuple[str, list[dict[str, Any]]]:
    """(final url, tables) for one public page, through the same guarded
    fetcher as web_fetch (SSRF guard, size cap). Cached ten minutes so a
    look with web_tables and an import cost one download."""
    import asyncio
    import time

    from .fetch import FetchGuardError, fetch_and_extract

    target = (url or "").strip()
    if not target:
        raise TablesError("url_required", "Give the page's URL.")
    if not target.lower().startswith(("http://", "https://")):
        target = f"https://{target}"
    hit = _CACHE.get(target)
    now = time.monotonic()
    if hit and now - hit[0] < _CACHE_SECONDS:
        return hit[1], hit[2]
    try:
        page = await asyncio.to_thread(fetch or (lambda u: fetch_and_extract(u, raw_html=True)), target)
    except FetchGuardError as e:
        raise TablesError("refused", f"refused: {e}")  # detail-ok: our own URL policy text
    except Exception as e:  # noqa: BLE001
        from ..hygiene import safe_error
        _ref, message = safe_error("web_tables.failed", e, user_message="The page could not be fetched.", url=target)
        raise TablesError("fetch_failed", message)
    if page.get("kind") != "html" or not page.get("html"):
        raise TablesError("no_tables", "That page has no HTML tables (a PDF or plain text); use web_fetch.")
    final = page.get("url") or target
    tables = absolute_links(parse_tables(page["html"]), final)
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)))
        if len(_RELATED) >= _CACHE_MAX:
            _RELATED.pop(next(iter(_RELATED)))
    _RELATED[final] = sibling_pages(page["html"], final)
    _CACHE[target] = (now, final, tables)
    return final, tables


# 2026-10-27: a season's schedule is split into month pages; the first
# import read October only (82 of about 1,230 games) and called the step
# done. web_tables lists the page's siblings so the split is visible.
_RELATED: dict[str, list[str]] = {}
_HREF = re.compile(r"<a\s[^>]*?href\s*=\s*[\"']([^\"'#]+)[\"']", re.I)
MAX_RELATED = 24


def sibling_pages(html: str, base: str) -> list[str]:
    """Links on the page to pages shaped like it: the same site and folder,
    the same extension, a name that differs only in its tail (months,
    weeks, page numbers, letters) or only in one query value."""
    import posixpath as os_path  # URL paths, on any OS
    from urllib.parse import parse_qsl, urljoin, urlsplit

    me = urlsplit(base)
    folder, name = os_path.split(me.path)
    stem, ext = os_path.splitext(name)
    out: list[str] = []
    for href in _HREF.findall(html or ""):
        if href.lower().startswith(("javascript:", "mailto:")):
            continue
        u = urlsplit(urljoin(base, href.strip()))
        if u.netloc != me.netloc or (u.path == me.path and u.query == me.query):
            continue
        f, n = os_path.split(u.path)
        st, ex = os_path.splitext(n)
        same = False
        if u.path == me.path:
            a, b = dict(parse_qsl(u.query)), dict(parse_qsl(me.query))
            same = bool(a) and a.keys() == b.keys() and sum(1 for k in a if a[k] != b.get(k)) == 1
        elif f == folder and ex == ext and st and stem:
            common = len(os_path.commonprefix([st, stem]))
            same = common >= max(3, int(0.5 * min(len(st), len(stem))))
        url = u.geturl()
        if same and url not in out:
            out.append(url)
        if len(out) >= MAX_RELATED:
            break
    return out


def related_pages(url: str) -> list[str]:
    return list(_RELATED.get(url) or [])


def absolute_links(tables: list[dict[str, Any]], base: str) -> list[dict[str, Any]]:
    """Series 372 (2026-10-26): sites link relatively (/boxscores/x.html);
    a url column refused every row of the first live import. Each link
    becomes absolute against the page; in-page anchors and scripts are
    not links to anything."""
    from urllib.parse import urljoin

    for t in tables:
        for r in t["rows"]:
            for c in r:
                h = (c.get("href") or "").strip()
                if not h or h.startswith("#") or h.lower().startswith("javascript:"):
                    c["href"] = None
                else:
                    c["href"] = urljoin(base, h)
    return tables


def find_table(tables: list[dict[str, Any]], table: Any) -> dict[str, Any]:
    """By index (0, 1, ...) or by the table's id attribute."""
    if isinstance(table, int) or (isinstance(table, str) and table.isdigit()):
        i = int(table)
        if 0 <= i < len(tables):
            return tables[i]
    else:
        for t in tables:
            if t["id"] and t["id"] == str(table):
                return t
    raise TablesError("no_such_table", f"The page has tables 0 to {len(tables) - 1}" + (
        f" (ids: {', '.join(t['id'] for t in tables if t['id'])[:300]})" if any(t["id"] for t in tables) else "") + ".")


def reset_for_tests() -> None:
    _CACHE.clear()
