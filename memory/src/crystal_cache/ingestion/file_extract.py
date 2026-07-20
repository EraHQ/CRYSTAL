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

    if lower.endswith(".eml"):
        return extract_chat_from_eml(file_bytes)
    elif lower.endswith(".mbox"):
        return extract_chat_from_mbox(file_bytes)
    elif lower.endswith(".xlsx"):
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


# --- Gate F (2026-07-20): conversational extraction ------------------------
# Canonical chat text: optional "=== WINDOW: YYYY-MM ===" markers
# (archives only, F-Q1=C), one unit (email / thread) per
# "--- <unit-ref> ---" section, speaker-attributed lines. Message-IDs
# and reply refs are PRESERVED in unit headers — slice 2's chain
# material.

CHAT_WINDOW_MARKER = "=== WINDOW: "
CHAT_UNIT_MARKER = "--- "


def _email_month(msg) -> str:
    from email.utils import parsedate_to_datetime
    try:
        dt = parsedate_to_datetime(msg.get("Date", ""))
        return f"{dt.year:04d}-{dt.month:02d}"
    except Exception:  # noqa: BLE001
        return "undated"


def _email_unit_text(msg) -> str:
    """One email -> one canonical unit."""
    def _body(m) -> str:
        if m.is_multipart():
            for part in m.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        return payload.decode(
                            part.get_content_charset() or "utf-8",
                            errors="replace",
                        )
            return ""
        payload = m.get_payload(decode=True)
        if payload:
            return payload.decode(
                m.get_content_charset() or "utf-8", errors="replace",
            )
        return str(m.get_payload() or "")

    refs = msg.get("In-Reply-To") or msg.get("References") or ""
    header = (
        f"{CHAT_UNIT_MARKER}message-id={msg.get('Message-ID', '').strip()}"
        f" in-reply-to={refs.strip().split()[-1] if refs.strip() else ''} ---"
    )
    lines = [
        header,
        f"From: {msg.get('From', '')} | Date: {msg.get('Date', '')}"
        f" | Subject: {msg.get('Subject', '')}",
    ]
    body = _body(msg).strip()
    sender = (msg.get("From") or "unknown").split("<")[0].strip() or "unknown"
    for ln in body.splitlines():
        if ln.strip():
            lines.append(f"{sender}: {ln.strip()}")
    return "\n".join(lines)


def extract_chat_from_eml(file_bytes: bytes) -> str:
    """Single email: whole-file, no window markers (F-Q1=C)."""
    import email
    from email import policy
    msg = email.message_from_bytes(file_bytes, policy=policy.default)
    return _email_unit_text(msg)


def extract_chat_from_mbox(file_bytes: bytes) -> str:
    """Archive: monthly windows (F-Q1=C — C4's worked example)."""
    import email
    import mailbox
    import os
    import tempfile
    from email import policy

    with tempfile.NamedTemporaryFile(
        suffix=".mbox", delete=False,
    ) as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    try:
        box = mailbox.mbox(path)
        by_month: dict[str, list[str]] = {}
        for raw in box:
            msg = email.message_from_bytes(
                raw.as_bytes(), policy=policy.default,
            )
            by_month.setdefault(_email_month(msg), []).append(
                _email_unit_text(msg)
            )
        box.close()
    finally:
        os.unlink(path)

    sections = []
    for month in sorted(by_month):
        units = "\n\n".join(by_month[month])
        sections.append(f"{CHAT_WINDOW_MARKER}{month} ===\n{units}")
    return "\n\n".join(sections)


def looks_like_slack_export(text: str) -> bool:
    """Mechanical shape check: a JSON array of message objects with
    ts + (text|user). Generic JSON stays untouched — Gate G's
    schema-inference owns it."""
    import json
    head = text.lstrip()
    if not head.startswith("["):
        return False
    try:
        data = json.loads(text)
    except Exception:  # noqa: BLE001
        return False
    if not isinstance(data, list) or not data:
        return False
    sample = [d for d in data[:5] if isinstance(d, dict)]
    return bool(sample) and all(
        "ts" in d and ("text" in d or "user" in d) for d in sample
    )


def extract_chat_from_slack_json(text: str) -> str:
    """Slack channel export -> threads as units, monthly windows when
    the export spans months (F-Q1=C)."""
    import json
    from datetime import datetime, timezone

    data = json.loads(text)
    threads: dict[str, list[dict]] = {}
    order: list[str] = []
    for m in data:
        if not isinstance(m, dict):
            continue
        root = str(m.get("thread_ts") or m.get("ts") or "")
        if root not in threads:
            threads[root] = []
            order.append(root)
        threads[root].append(m)

    def _month(ts: str) -> str:
        try:
            dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
            return f"{dt.year:04d}-{dt.month:02d}"
        except Exception:  # noqa: BLE001
            return "undated"

    by_month: dict[str, list[str]] = {}
    for root in order:
        msgs = sorted(threads[root], key=lambda m: float(m.get("ts") or 0))
        lines = [f"{CHAT_UNIT_MARKER}thread={root} ---"]
        for m in msgs:
            who = m.get("user") or m.get("username") or "unknown"
            txt = (m.get("text") or "").replace("\n", " ").strip()
            if txt:
                lines.append(f"{who}: {txt}")
        if len(lines) > 1:
            by_month.setdefault(_month(root), []).append("\n".join(lines))

    if len(by_month) <= 1:
        # Single month / small export: whole-file (F-Q1=C).
        return "\n\n".join(
            u for units in by_month.values() for u in units
        )
    sections = []
    for month in sorted(by_month):
        units = "\n\n".join(by_month[month])
        sections.append(f"{CHAT_WINDOW_MARKER}{month} ===\n{units}")
    return "\n\n".join(sections)
