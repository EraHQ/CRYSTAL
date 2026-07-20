"""File-based document upload and text extraction.

Accepts PDF, DOCX, and TXT files via multipart upload.
Extracts text content and stores in document_uploads table.
"""
from __future__ import annotations

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)


import re


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber (preferred) or pypdf."""
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n\n".join(pages)
    except ImportError:
        pass

    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    except ImportError:
        pass

    raise ImportError("No PDF library available. Install pdfplumber or pypdf.")


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX bytes."""
    try:
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except ImportError:
        # Fallback: docx files are ZIP archives with XML
        import zipfile
        import xml.etree.ElementTree as ET

        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            with zf.open("word/document.xml") as f:
                tree = ET.parse(f)

        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs = []
        for p in tree.iter(f"{{{ns['w']}}}p"):
            texts = []
            for t in p.iter(f"{{{ns['w']}}}t"):
                if t.text:
                    texts.append(t.text)
            if texts:
                paragraphs.append("".join(texts))
        return "\n\n".join(paragraphs)


def extract_text_from_html(file_bytes: bytes) -> str:
    """Gate A (2026-07-16): .html/.htm through the same main-text
    extractor the web lane ships (chrome-stripped, title recovered as
    the first line so detection and chunk labels see it)."""
    from ..search.fetch import extract_main_text

    title, body = extract_main_text(
        file_bytes.decode("utf-8", errors="replace")
    )
    return (f"{title.strip()}\n\n{body}" if (title or "").strip()
            else body)


def extract_transcript_from_subtitles(file_bytes: bytes) -> str:
    """Gate A (2026-07-16): .vtt/.srt -> speaker-attributed transcript
    text. Zoom/Meet exports carry 'Name: text' in cues (or <v Name>
    voice tags); WEBVTT headers, NOTE/STYLE blocks, cue ids, and
    timestamp lines are dropped. The result lands on the transcript
    detected_type — the dynamics profile for free."""
    text = file_bytes.decode("utf-8", errors="replace")
    lines: list[str] = []
    for rawline in text.splitlines():
        s = rawline.strip().lstrip("\ufeff")
        if not s:
            continue
        upper = s.upper()
        if upper.startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue
        if "-->" in s:
            continue
        if s.isdigit():
            continue
        m = re.match(r"<v\s+([^>]+)>(.*)", s)
        if m:
            s = f"{m.group(1).strip()}: {m.group(2)}"
        s = re.sub(r"<[^>]+>", "", s).strip()
        if s:
            lines.append(s)
    return "\n".join(lines)


def extract_text_from_file(
    file_bytes: bytes,
    filename: str,
) -> str:
    """Extract text from a file based on its extension."""
    lower = filename.lower()

    if lower.endswith(".xlsx"):
        return extract_tabular_from_xlsx(file_bytes)
    elif lower.endswith(".csv"):
        return extract_tabular_from_delimited(file_bytes, ",")
    elif lower.endswith(".tsv"):
        return extract_tabular_from_delimited(file_bytes, "\t")
    elif lower.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif lower.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    elif lower.endswith(".html") or lower.endswith(".htm"):
        return extract_text_from_html(file_bytes)
    elif lower.endswith(".vtt") or lower.endswith(".srt"):
        return extract_transcript_from_subtitles(file_bytes)
    elif lower.endswith(".txt") or lower.endswith(".md"):
        return file_bytes.decode("utf-8", errors="replace")
    else:
        # Try as plain text
        try:
            return file_bytes.decode("utf-8", errors="replace")
        except Exception:
            raise ValueError(f"Unsupported file type: {filename}")


# --- Gate E (2026-07-20): tabular extraction -------------------------------
# One CANONICAL text form for every tabular source, so the chunker and
# the mechanical row extractor parse exactly one format: optional
# "=== SHEET: <name> ===" markers (xlsx only), first line = headers,
# tab-separated rows. Zero LLM anywhere in this lane (F4).

TABULAR_SHEET_MARKER = "=== SHEET: "


def extract_tabular_from_delimited(file_bytes: bytes, delimiter: str) -> str:
    """csv/tsv -> canonical TSV text (quotes and embedded delimiters
    resolved by the csv parser, tabs/newlines inside cells flattened)."""
    import csv
    import io
    text = file_bytes.decode("utf-8", errors="replace")
    out_lines = []
    for row in csv.reader(io.StringIO(text), delimiter=delimiter):
        out_lines.append("\t".join(
            (cell or "").replace("\t", " ").replace("\n", " ").strip()
            for cell in row
        ))
    return "\n".join(out_lines)


def extract_tabular_from_xlsx(file_bytes: bytes) -> str:
    """xlsx -> canonical text with sheet markers. read_only mode keeps
    memory flat on big workbooks; formulas arrive as computed values
    when the file carries them."""
    import io
    from openpyxl import load_workbook
    wb = load_workbook(
        io.BytesIO(file_bytes), read_only=True, data_only=True,
    )
    sections = []
    for ws in wb.worksheets:
        lines = [f"{TABULAR_SHEET_MARKER}{ws.title} ==="]
        for row in ws.iter_rows(values_only=True):
            cells = [
                str(c).replace("\t", " ").replace("\n", " ").strip()
                if c is not None else ""
                for c in row
            ]
            if any(cells):
                lines.append("\t".join(cells))
        if len(lines) > 1:
            sections.append("\n".join(lines))
    wb.close()
    return "\n\n".join(sections)
