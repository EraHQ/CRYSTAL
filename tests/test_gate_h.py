"""Gate H (2026-07-23): the text-adapter batch — pptx, rtf, odt, epub,
ipynb — plus the C3 MIME-map fallback.

Design record: H-Q1=A (no fragment carve; pptx slides are in-text
`=== SLIDE N ===` locator markers — decks have no set-in-stone format,
the general chunker builds the shape). H-Q2=A (python-pptx + striprtf
in core deps with stdlib fallbacks coded; odt/epub/ipynb stdlib by
design). Fallback paths are tested DIRECTLY so a stripped install's
behavior is pinned even when the libraries are present.
"""

from __future__ import annotations

import io
import json
import zipfile

import pytest

from crystal_cache.ingestion.file_extract import (
    _extract_pptx_stdlib,
    _extract_rtf_stdlib,
    extract_text_from_epub,
    extract_text_from_file,
    extract_text_from_ipynb,
    extract_text_from_odt,
    extract_text_from_pptx,
    extract_text_from_rtf,
)


def _make_pptx() -> bytes:
    pptx = pytest.importorskip("pptx")
    prs = pptx.Presentation()
    s1 = prs.slides.add_slide(prs.slide_layouts[1])
    s1.shapes.title.text = "Q3 Review"
    s1.placeholders[1].text = "Revenue up 12%"
    s1.notes_slide.notes_text_frame.text = "mention churn"
    s2 = prs.slides.add_slide(prs.slide_layouts[1])
    s2.shapes.title.text = "Roadmap"
    s2.placeholders[1].text = "Ship Gate H"
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def test_pptx_primary_slides_notes_and_markers():
    out = extract_text_from_pptx(_make_pptx())
    assert "=== SLIDE 1 ===" in out and "=== SLIDE 2 ===" in out
    assert "Q3 Review" in out and "Revenue up 12%" in out
    assert "Notes: mention churn" in out
    assert out.index("Q3 Review") < out.index("Roadmap")


def test_pptx_stdlib_fallback_on_real_deck():
    out = _extract_pptx_stdlib(_make_pptx())
    assert "=== SLIDE 1 ===" in out
    assert "Q3 Review" in out and "Ship Gate H" in out


RTF = rb"{\rtf1\ansi{\fonttbl{\f0 Arial;}}\f0 Hello team\par Second line\par}"


def test_rtf_primary_and_fallback():
    p = extract_text_from_rtf(RTF)
    assert "Hello team" in p and "Second line" in p
    f = _extract_rtf_stdlib(RTF)
    assert "Hello team" in f and "Second line" in f
    assert "Arial" not in f, "font table must not leak into text"


def _make_odt() -> bytes:
    content = (
        b'<?xml version="1.0"?>'
        b'<office:document-content'
        b' xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"'
        b' xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">'
        b"<office:body><office:text>"
        b"<text:h>Policy Title</text:h>"
        b"<text:p>First <text:span>paragraph</text:span> here.</text:p>"
        b"<text:p>Second paragraph.</text:p>"
        b"</office:text></office:body></office:document-content>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("content.xml", content)
    return buf.getvalue()


def test_odt_paragraphs_headings_nested_spans():
    out = extract_text_from_odt(_make_odt())
    assert out == "Policy Title\n\nFirst paragraph here.\n\nSecond paragraph."


def _make_epub() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?>'
            '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles><rootfile full-path="OEBPS/content.opf"'
            ' media-type="application/oebps-package+xml"/></rootfiles>'
            "</container>",
        )
        # Chapter two written FIRST in the zip: spine order must win.
        zf.writestr(
            "OEBPS/ch2.xhtml",
            "<html><body><p>Chapter two body text with several plain"
            " sentences of readable content for the extractor to keep"
            " and return in order.</p></body></html>",
        )
        zf.writestr(
            "OEBPS/ch1.xhtml",
            "<html><body><p>Chapter one body text with several plain"
            " sentences of readable content for the extractor to keep"
            " and return in order.</p></body></html>",
        )
        zf.writestr(
            "OEBPS/content.opf",
            '<?xml version="1.0"?>'
            '<package xmlns="http://www.idpf.org/2007/opf">'
            '<manifest><item id="c1" href="ch1.xhtml"'
            ' media-type="application/xhtml+xml"/>'
            '<item id="c2" href="ch2.xhtml"'
            ' media-type="application/xhtml+xml"/></manifest>'
            '<spine><itemref idref="c1"/><itemref idref="c2"/></spine>'
            "</package>",
        )
    return buf.getvalue()


def test_epub_spine_order_wins_over_zip_order():
    out = extract_text_from_epub(_make_epub())
    assert "Chapter one body" in out and "Chapter two body" in out
    assert out.index("Chapter one body") < out.index("Chapter two body")


def test_ipynb_markdown_and_fenced_code():
    nb = {
        "metadata": {"kernelspec": {"language": "python"}},
        "cells": [
            {"cell_type": "markdown",
             "source": ["# Analysis\n", "Findings below."]},
            {"cell_type": "code",
             "source": ["import pandas as pd\n", "df.head()"]},
            {"cell_type": "code", "source": []},
        ],
    }
    out = extract_text_from_ipynb(json.dumps(nb).encode())
    assert "# Analysis" in out and "```python" in out and "df.head()" in out
    assert out.count("```") == 2, "empty cell must not emit a fence"


def test_dispatch_routes_all_five_extensions():
    assert "Policy Title" in extract_text_from_file(_make_odt(), "doc.odt")
    assert "Hello team" in extract_text_from_file(RTF, "memo.rtf")
    assert "Chapter one body" in extract_text_from_file(
        _make_epub(), "book.epub",
    )
    nb = json.dumps({"cells": [
        {"cell_type": "markdown", "source": ["hello notebook"]},
    ]}).encode()
    assert "hello notebook" in extract_text_from_file(nb, "nb.ipynb")
    assert "Q3 Review" in extract_text_from_file(_make_pptx(), "deck.pptx")


def test_mime_fallback_for_extensionless_sources():
    # No usable extension; the declared MIME routes it (C3, Gate H).
    out = extract_text_from_file(
        _make_odt(), "attachment",
        mime="application/vnd.oasis.opendocument.text",
    )
    assert "Policy Title" in out
    out2 = extract_text_from_file(
        RTF, "clip", mime="text/rtf; charset=latin-1",
    )
    assert "Hello team" in out2, "MIME parameters must be stripped"
    # Unknown MIME + undecodable-as-anything still falls to plain text.
    plain = extract_text_from_file(b"just words", "x", mime="application/x-whatever")
    assert plain == "just words"
