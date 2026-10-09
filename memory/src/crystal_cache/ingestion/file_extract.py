"""File-based document upload and text extraction.

Accepts PDF, DOCX, and TXT files via multipart upload.
Extracts text content and stores in document_uploads table.
"""
from __future__ import annotations

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)


import hashlib
import re
import time
import unicodedata
import zipfile


class _PublicRefusal(ValueError):
    """A refusal whose text is written for the user (it names the limit
    and what to do, never library internals or file contents). Routes
    send `public_message`, not str(e), so the S2-S7 sweep can tell this
    apart from exception text that must stay in the logs."""

    def __init__(self, public_message: str):
        super().__init__(public_message)
        self.public_message = public_message


class UnsupportedFileType(_PublicRefusal):
    """The bytes are not a file type ingestion accepts (HTTP 415)."""


class DocumentTooLarge(_PublicRefusal):
    """The file is past a hard reading limit (HTTP 413)."""


# --- Lockdown PR-1 (2026-10-08, AUDIT_LAUNCH_VERIFY B2-6, B2-7) ----------
# Every zip-based format (docx, pptx, xlsx, odt, epub) used to be opened
# with no check on member count, declared size or compression ratio, so a
# 5 MB archive whose XML inflates to 2 GB was parsed in full inside the
# API process. The pre-flight below runs before any library touches the
# archive. Headers can lie (zipfile reads by COMPRESSED size and only
# notices a wrong declared size at the CRC check), so after the header
# checks every member is streamed once through a bounded read; a member
# that is larger than it declares is refused there. Nested archives and
# embedded objects (word/embeddings/*, ppt/media/*, OLE) are never
# opened: the extractors read named XML members only, and the libraries
# that load every part (python-pptx, openpyxl) now see an archive whose
# every member has already been measured.
ZIP_MAX_MEMBERS = 2_000
ZIP_MAX_TOTAL_BYTES = 100 * 2**20
ZIP_MAX_MEMBER_BYTES = 50 * 2**20
ZIP_MAX_RATIO = 100
_ZIP_PROBE_CHUNK = 1 << 20

# PDF: at most this many pages, and this much wall clock per file.
# pdfplumber's `pdf.pages` builds a Page object for EVERY page before
# returning, so the page tree is walked here by hand and stops at the cap.
PDF_MAX_PAGES = 500
PDF_TIME_BUDGET_SECONDS = 60.0

# The upload allowlist (Q45=A, 2026-10-08): exactly the console's
# accept= list (frontend/src/pages/KnowledgeManager.tsx). Anything else,
# images, standalone archives, legacy Office binaries and executables
# included, is refused with 415 instead of being UTF-8-decoded and paid
# for as text. Extensions in ACCEPTED_TEXT_EXTENSIONS decode as UTF-8
# text; the rest have a dedicated extractor in extract_text_from_file.
ACCEPTED_TEXT_EXTENSIONS = frozenset({
    ".txt", ".md", ".json", ".jsonl", ".ndjson",
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java",
    ".rb", ".c", ".h", ".cpp", ".cs", ".php", ".swift", ".kt", ".sh",
})
ACCEPTED_EXTENSIONS = ACCEPTED_TEXT_EXTENSIONS | frozenset({
    ".pdf", ".docx", ".pptx", ".xlsx", ".odt", ".epub", ".rtf",
    ".html", ".htm", ".csv", ".tsv", ".eml", ".mbox", ".vtt", ".srt",
    ".ipynb",
})

# --- Lockdown PR-5 (Q45=A, 2026-10-09): the upload lockdown table ---------
# Every rule below drives the real upload route; tests/test_lockdown_pr5.py
# has one pin per rule.

# Content sniff (Q45): the bytes must be what the name says. Families:
# pdf starts with %PDF-; docx/pptx/xlsx are zips carrying
# [Content_Types].xml; odt/epub are zips whose FIRST member is `mimetype`
# with the format's value; everything else is text and must strict-UTF-8
# decode in its first 64 KiB with no NUL in its first 8 KiB. A mismatch is
# 415, never a decode-and-pay.
_OOXML_EXTENSIONS = frozenset({".docx", ".pptx", ".xlsx"})
_ODF_MIMETYPES = {
    ".odt": b"application/vnd.oasis.opendocument.text",
    ".epub": b"application/epub+zip",
}
SNIFF_UTF8_BYTES = 64 * 1024
SNIFF_NUL_BYTES = 8 * 1024

# xlsx caps (Q45): a workbook past these is refused (413), never read in
# part; a row wider than XLSX_MAX_COLUMNS is cut with a visible marker.
XLSX_MAX_SHEETS = 50
XLSX_MAX_ROWS = 100_000
XLSX_MAX_COLUMNS = 200
XLSX_COLUMNS_TRUNCATED_MARKER = "[columns truncated]"

# mbox caps (Q45): more messages than this is refused (413); one
# message's text past MBOX_MAX_MESSAGE_TEXT_BYTES is cut with a marker.
MBOX_MAX_MESSAGES = 5_000
MBOX_MAX_MESSAGE_TEXT_BYTES = 1 * 2**20
MBOX_MESSAGE_TRUNCATED_MARKER = "[message text truncated at 1 MiB]"

# Label rules (Q45, B2-9). The label is a model prompt input, a dedup
# identity and the console title, so it is normalised on every lane
# (multipart, JSON, MCP, source sync) by the one function below.
LABEL_MAX_CHARS = 200
_LABEL_STRIP = re.compile(
    "[\u202a-\u202e\u2066-\u2069\u200b-\u200d\ufeff]"
)
_LABEL_WS = re.compile(r"\s+")

# crystal_type on upload (Q45): a customer bucket, lowercase, bounded.
# general:*, assumption and reflection have semantics of their own and
# are never an upload's bucket.
UPLOAD_CRYSTAL_TYPE_RE = re.compile(r"^customer:[a-z0-9_.-]{1,64}$")


def sanitize_label(raw: "str | None", *, text: str = "") -> str:
    """The one label normaliser (Q45): Unicode NFC; every C0/C1 control
    removed (newlines and tabs included); bidi controls U+202A..U+202E
    and U+2066..U+2069 and the zero-width U+200B..U+200D, U+FEFF removed;
    whitespace runs collapsed to one space; stripped; at most
    LABEL_MAX_CHARS. An empty result (or the literal "Untitled", the
    pre-existing MCP rule) becomes `Untitled <sha256(text)[:12]>` so two
    unlabelled uploads of different texts are different sources."""
    s = unicodedata.normalize("NFC", raw or "")
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Cc")
    s = _LABEL_STRIP.sub("", s)
    s = _LABEL_WS.sub(" ", s).strip()
    if len(s) > LABEL_MAX_CHARS:
        s = s[:LABEL_MAX_CHARS].rstrip()
    if not s or s.lower() == "untitled":
        digest = hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()[:12]
        return f"Untitled {digest}"
    return s


def label_from_filename(filename: "str | None") -> str:
    """Q45: a filename is used only for dispatch and as the default label,
    never as a path. The basename after splitting on both separators."""
    name = filename or ""
    for sep in ("/", "\\"):
        name = name.rsplit(sep, 1)[-1]
    return name


def valid_upload_crystal_type(value: "str | None") -> bool:
    return bool(value) and UPLOAD_CRYSTAL_TYPE_RE.match(value) is not None


def _extension_for(filename: str, mime: Optional[str]) -> str:
    """Which accepted extension a file dispatches on (Q45): the name's
    own extension when it is an accepted one; else the declared MIME
    through the table (plus text/plain); else refused. image/*,
    application/zip and application/octet-stream are not in the table,
    so a nameless file declared as any of those is 415."""
    lower = (filename or "").lower()
    for ext in sorted(ACCEPTED_EXTENSIONS, key=len, reverse=True):
        if lower.endswith(ext):
            return ext
    declared = (mime or "").split(";")[0].strip().lower()
    ext = _MIME_EXTENSIONS.get(declared)
    if ext is None and declared == "text/plain":
        ext = ".txt"
    if ext is None:
        raise UnsupportedFileType(
            "Unsupported file type. Upload a PDF, Word, PowerPoint, Excel, "
            "OpenDocument, EPUB, RTF, text, Markdown, HTML, CSV, JSON, "
            "email, subtitle, notebook or source-code file."
        )
    return ext


def _zip_names(file_bytes: bytes) -> list[str]:
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            return zf.namelist()
    except zipfile.BadZipFile:
        return []


def sniff(file_bytes: bytes, ext: str) -> None:
    """Q45 content sniff: raise UnsupportedFileType (415) when the bytes
    are not the family `ext` names. See the family table above."""
    if ext == ".pdf":
        if not file_bytes.startswith(b"%PDF-"):
            raise UnsupportedFileType("The file is not a PDF.")
        return
    if ext in _OOXML_EXTENSIONS:
        if "[Content_Types].xml" not in _zip_names(file_bytes):
            raise UnsupportedFileType(
                f"The file is not a {ext[1:]} document."
            )
        return
    if ext in _ODF_MIMETYPES:
        names = _zip_names(file_bytes)
        ok = False
        if names and names[0] == "mimetype":
            try:
                with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                    with zf.open("mimetype") as fh:
                        ok = fh.read(256).strip() == _ODF_MIMETYPES[ext]
            except (zipfile.BadZipFile, KeyError, OSError):
                ok = False
        if not ok:
            raise UnsupportedFileType(
                f"The file is not an {ext[1:]} document."
            )
        return
    # Every other accepted type is text.
    if b"\0" in file_bytes[:SNIFF_NUL_BYTES]:
        raise UnsupportedFileType("The file is not a text file.")
    sample = file_bytes[:SNIFF_UTF8_BYTES]
    # A multi-byte character may be cut at the sample boundary: drop up
    # to three trailing continuation/lead bytes before deciding.
    for cut in range(0, 4):
        try:
            sample[: len(sample) - cut].decode("utf-8", errors="strict")
            return
        except UnicodeDecodeError:
            continue
    raise UnsupportedFileType("The file is not UTF-8 text.")


def _check_zip_members(zf: "zipfile.ZipFile") -> None:
    infos = zf.infolist()
    if len(infos) > ZIP_MAX_MEMBERS:
        raise DocumentTooLarge(
            f"The archive has {len(infos):,} parts; the limit is "
            f"{ZIP_MAX_MEMBERS:,}. Split the document."
        )
    declared_total = 0
    for zi in infos:
        if zi.is_dir():
            continue
        if zi.file_size > ZIP_MAX_MEMBER_BYTES:
            raise DocumentTooLarge(
                "One part of the document is larger than 50 MB. "
                "Split the document."
            )
        if (zi.file_size > _ZIP_PROBE_CHUNK
                and zi.file_size > ZIP_MAX_RATIO * max(zi.compress_size, 1)):
            raise DocumentTooLarge(
                "The document is compressed far beyond what a real "
                "document compresses to, so it can't be read safely."
            )
        declared_total += zi.file_size
        if declared_total > ZIP_MAX_TOTAL_BYTES:
            raise DocumentTooLarge(
                "The document unpacks to more than 100 MB. Split the document."
            )
    actual_total = 0
    for zi in infos:
        if zi.is_dir():
            continue
        seen = 0
        with zf.open(zi) as fh:
            while True:
                chunk = fh.read(_ZIP_PROBE_CHUNK)
                if not chunk:
                    break
                seen += len(chunk)
                actual_total += len(chunk)
                if seen > ZIP_MAX_MEMBER_BYTES or actual_total > ZIP_MAX_TOTAL_BYTES:
                    raise DocumentTooLarge(
                        "One part of the document is larger than it "
                        "declares, so it can't be read safely."
                    )


def looks_binary(file_bytes: bytes) -> bool:
    """A NUL byte in the first 8 KiB means the bytes are not text (the
    check every code indexer uses). Source sync skips such files instead
    of decoding them as garbage and paying to extract it."""
    return b"\0" in file_bytes[:8192]


def safe_zip_bytes(file_bytes: bytes) -> bytes:
    """Pre-flight a zip-based document and hand the same bytes back.
    Raises DocumentTooLarge (413) past a limit and UnsupportedFileType
    (415) when the bytes are not a zip at all."""
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            _check_zip_members(zf)
    except zipfile.BadZipFile as e:
        raise UnsupportedFileType("The file is not a valid document archive") from e
    return file_bytes


def _bounded_zip_member(zf: "zipfile.ZipFile", name: str) -> io.BytesIO:
    """Read one member through a bounded stream (the pre-flight has
    measured it; this keeps the bound local to the read as well)."""
    buf = bytearray()
    with zf.open(name) as fh:
        while True:
            chunk = fh.read(_ZIP_PROBE_CHUNK)
            if not chunk:
                break
            buf += chunk
            if len(buf) > ZIP_MAX_MEMBER_BYTES:
                raise DocumentTooLarge(
                    "One part of the document is larger than 50 MB. "
                    "Split the document."
                )
    return io.BytesIO(bytes(buf))


def extract_text_from_pdf(
    file_bytes: bytes,
    max_chars: "int | None" = None,
    *,
    max_pages: int = PDF_MAX_PAGES,
    time_budget_seconds: float = PDF_TIME_BUDGET_SECONDS,
) -> str:
    """Extract text from PDF bytes using pdfplumber (preferred) or pypdf.

    max_chars (2026-07-27): stop parsing once the accumulated text
    reaches the budget, and release each page's parse objects as we
    go. Added after a live OOM: pdfplumber builds per-page layout
    objects, so a 20 MB government PDF can amplify to gigabytes of
    heap — a worker at 4 Gi died at 4112 MiB parsing a tariff
    document whose text was then truncated anyway. Parsing pages the
    caller will throw away is pure memory burn. None = unbounded.

    Lockdown PR-1 (B2-7): at most `max_pages` pages (a PDF past that is
    refused with DocumentTooLarge, never silently cut) and at most
    `time_budget_seconds` of wall clock per file. The page tree is
    walked page by page (pdfplumber's own `pages` property builds every
    Page object up front, so a 1M-page file would loop unbounded).
    """
    started = time.monotonic()

    def _over_time() -> bool:
        return (time.monotonic() - started) > time_budget_seconds

    try:
        import pdfplumber
        from pdfminer.pdfpage import PDFPage
        from pdfplumber.page import Page

        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = []
            total = 0
            doctop = 0
            for i, page_obj in enumerate(PDFPage.create_pages(pdf.doc)):
                page_number = i + 1
                if page_number > max_pages:
                    raise DocumentTooLarge(
                        f"The PDF has more than {max_pages} pages. Split the file."
                    )
                if _over_time():
                    raise DocumentTooLarge(
                        "The PDF took too long to read. Split the file."
                    )
                page = Page(pdf, page_obj, page_number=page_number, initial_doctop=doctop)
                text = page.extract_text()
                doctop += page.height
                # Release this page's layout objects before moving on —
                # the cache, not the raw bytes, is what amplifies memory.
                try:
                    page.flush_cache()
                except Exception:  # noqa: BLE001 — older pdfplumber
                    pass
                if text:
                    pages.append(text)
                    total += len(text)
                    if max_chars is not None and total >= max_chars:
                        break
            return "\n\n".join(pages)
    except ImportError:
        pass

    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        if len(reader.pages) > max_pages:
            raise DocumentTooLarge(
                f"The PDF has more than {max_pages} pages. Split the file."
            )
        pages = []
        total = 0
        for page in reader.pages:
            if _over_time():
                raise DocumentTooLarge(
                    "The PDF took too long to read. Split the file."
                )
            text = page.extract_text()
            if text:
                pages.append(text)
                total += len(text)
                if max_chars is not None and total >= max_chars:
                    break
        return "\n\n".join(pages)
    except ImportError:
        pass

    raise ImportError("No PDF library available. Install pdfplumber or pypdf.")


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX bytes."""
    file_bytes = safe_zip_bytes(file_bytes)
    try:
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except ImportError:
        # Fallback: docx files are ZIP archives with XML
        import xml.etree.ElementTree as ET

        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            try:
                tree = ET.parse(_bounded_zip_member(zf, "word/document.xml"))
            except KeyError as e:
                raise UnsupportedFileType("The file is not a Word document") from e

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
    mime: Optional[str] = None,
    *,
    max_chars: "int | None" = None,
) -> str:
    """Extract text from a file: extension dispatch first, declared
    MIME as the fallback for extensionless sources (C3, wired by
    Gate H).

    Lockdown PR-1 (B2-8): a file whose type is not recognised is refused
    with UnsupportedFileType (415). It used to be UTF-8 decoded and paid
    for as text, images and archives included. `max_chars` lets the PDF
    path stop parsing early; callers that enforce a character cap pass
    cap + 1 so they can tell "over" from "exactly at".
    """
    # Lockdown PR-5 (Q45): ONE resolution of the type (the name's accepted
    # extension, else the declared MIME through the table), then the
    # bytes are sniffed for that family before any extractor runs.
    ext = _extension_for(filename, mime)
    sniff(file_bytes, ext)

    if ext == ".eml":
        return extract_chat_from_eml(file_bytes)
    if ext == ".mbox":
        return extract_chat_from_mbox(file_bytes)
    if ext == ".xlsx":
        return extract_tabular_from_xlsx(file_bytes)
    if ext == ".csv":
        return extract_tabular_from_delimited(file_bytes, ",")
    if ext == ".tsv":
        return extract_tabular_from_delimited(file_bytes, "\t")
    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes, max_chars=max_chars)
    if ext == ".docx":
        return extract_text_from_docx(file_bytes)
    if ext in (".html", ".htm"):
        return extract_text_from_html(file_bytes)
    if ext in (".vtt", ".srt"):
        return extract_transcript_from_subtitles(file_bytes)
    if ext in ACCEPTED_TEXT_EXTENSIONS:
        return file_bytes.decode("utf-8", errors="replace")
    if ext == ".pptx":
        return extract_text_from_pptx(file_bytes)
    if ext == ".rtf":
        return extract_text_from_rtf(file_bytes)
    if ext == ".odt":
        return extract_text_from_odt(file_bytes)
    if ext == ".epub":
        return extract_text_from_epub(file_bytes)
    if ext == ".ipynb":
        return extract_text_from_ipynb(file_bytes)
    raise UnsupportedFileType("Unsupported file type.")  # unreachable by construction


# --- Gate H (2026-07-23): text-adapter batch --------------------------------
# H-Q1=A: no fragment carve for any of these — prose-class documents;
# pptx slides ride as in-text `=== SLIDE N ===` locator markers (decks
# have no set-in-stone format; the general chunker builds the shape).
# H-Q2=A: python-pptx + striprtf in core deps, stdlib fallbacks coded
# (the docx precedent). odt/epub/ipynb are stdlib by design.

PPTX_SLIDE_MARKER = "=== SLIDE "

_MIME_EXTENSIONS = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.oasis.opendocument.text": ".odt",
    "application/rtf": ".rtf",
    "text/rtf": ".rtf",
    "application/epub+zip": ".epub",
    "application/x-ipynb+json": ".ipynb",
    "application/json": ".json",
    "text/html": ".html",
    "text/csv": ".csv",
    "text/tab-separated-values": ".tsv",
    "text/markdown": ".md",
    "message/rfc822": ".eml",
}


def _extract_pptx_stdlib(file_bytes: bytes) -> str:
    """Fallback: pptx is a zip; slide text lives in <a:t> runs inside
    ppt/slides/slideN.xml. Loses tables-as-structure and notes; keeps
    every visible text run in slide order."""
    import zipfile
    import xml.etree.ElementTree as ET

    a_ns = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    parts: list[str] = []
    with zipfile.ZipFile(io.BytesIO(safe_zip_bytes(file_bytes))) as zf:
        slide_names = sorted(
            (n for n in zf.namelist()
             if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
            key=lambda n: int(re.search(r"(\d+)", n).group(1)),
        )
        for name in slide_names:
            num = int(re.search(r"(\d+)", name).group(1))
            tree = ET.parse(_bounded_zip_member(zf, name))
            runs = [t.text for t in tree.iter(f"{a_ns}t") if t.text]
            body = "\n".join(r for r in runs if r.strip())
            parts.append(f"{PPTX_SLIDE_MARKER}{num} ===\n{body}".rstrip())
    return "\n\n".join(parts)


def extract_text_from_pptx(file_bytes: bytes) -> str:
    """Slides in order with `=== SLIDE N ===` markers. python-pptx
    reads shapes, tables, and speaker notes; the stdlib fallback keeps
    text runs only."""
    file_bytes = safe_zip_bytes(file_bytes)  # PR-1: python-pptx loads every part
    try:
        from pptx import Presentation
    except ImportError:
        return _extract_pptx_stdlib(file_bytes)
    prs = Presentation(io.BytesIO(file_bytes))
    parts: list[str] = []
    for i, slide in enumerate(prs.slides, start=1):
        lines: list[str] = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    text = "".join(run.text for run in para.runs).strip()
                    if text:
                        lines.append(text)
            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    cells = [c.text.strip() for c in row.cells]
                    if any(cells):
                        lines.append("\t".join(cells))
        notes = ""
        if slide.has_notes_slide:
            notes = (slide.notes_slide.notes_text_frame.text or "").strip()
        body = "\n".join(lines)
        if notes:
            body = f"{body}\nNotes: {notes}" if body else f"Notes: {notes}"
        parts.append(f"{PPTX_SLIDE_MARKER}{i} ===\n{body}".rstrip())
    return "\n\n".join(parts)


_RTF_CONTROL = re.compile(
    r"\\\'[0-9a-fA-F]{2}|\\[a-zA-Z]+-?\d*[ ]?|[{}]|\\[^a-zA-Z]"
)


def _extract_rtf_stdlib(file_bytes: bytes) -> str:
    """Fallback: strip control words/groups. Approximate by design —
    plain documents come through clean; embedded objects degrade."""
    text = file_bytes.decode("latin-1", errors="replace")
    for group in ("fonttbl", "colortbl", "stylesheet", "pict", "info"):
        text = re.sub(
            r"\{\\" + group + r".*?\}", "", text, flags=re.DOTALL,
        )
    text = text.replace("\\par", "\n").replace("\\line", "\n")
    text = _RTF_CONTROL.sub("", text)
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def extract_text_from_rtf(file_bytes: bytes) -> str:
    try:
        from striprtf.striprtf import rtf_to_text
    except ImportError:
        return _extract_rtf_stdlib(file_bytes)
    return rtf_to_text(
        file_bytes.decode("latin-1", errors="replace"),
    ).strip()


def extract_text_from_odt(file_bytes: bytes) -> str:
    """Stdlib only by design: odt is a zip whose content.xml carries
    every paragraph/heading in <text:p>/<text:h> — a library adds
    nothing for text extraction."""
    import zipfile
    import xml.etree.ElementTree as ET

    t_ns = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
    with zipfile.ZipFile(io.BytesIO(safe_zip_bytes(file_bytes))) as zf:
        try:
            tree = ET.parse(_bounded_zip_member(zf, "content.xml"))
        except KeyError as e:
            raise UnsupportedFileType("The file is not an OpenDocument text file") from e
    paragraphs: list[str] = []
    for el in tree.iter():
        if el.tag in (f"{t_ns}p", f"{t_ns}h"):
            text = "".join(el.itertext()).strip()
            if text:
                paragraphs.append(text)
    return "\n\n".join(paragraphs)


def extract_text_from_epub(file_bytes: bytes) -> str:
    """Stdlib zip walk: container.xml -> OPF -> spine order -> each
    xhtml chapter through the SAME html extractor the web lane ships
    (chrome-stripped, title recovered). Chapters join in reading
    order."""
    import zipfile
    import xml.etree.ElementTree as ET

    c_ns = "{urn:oasis:names:tc:opendocument:xmlns:container}"
    o_ns = "{http://www.idpf.org/2007/opf}"
    with zipfile.ZipFile(io.BytesIO(safe_zip_bytes(file_bytes))) as zf:
        try:
            container = ET.parse(_bounded_zip_member(zf, "META-INF/container.xml"))
            rootfile = container.find(f".//{c_ns}rootfile")
            opf_path = rootfile.get("full-path") if rootfile is not None else None
            if not opf_path:
                raise KeyError("rootfile")
            opf = ET.parse(_bounded_zip_member(zf, opf_path))
        except KeyError as e:
            raise UnsupportedFileType("The file is not an EPUB book") from e
        base = opf_path.rsplit("/", 1)[0] + "/" if "/" in opf_path else ""
        items = {
            it.get("id"): it.get("href")
            for it in opf.iter(f"{o_ns}item")
        }
        chapters: list[str] = []
        for ref in opf.iter(f"{o_ns}itemref"):
            href = items.get(ref.get("idref"))
            if not href or not href.lower().endswith(
                (".xhtml", ".html", ".htm")
            ):
                continue
            try:
                raw = _bounded_zip_member(zf, f"{base}{href}").getvalue()
            except KeyError:
                continue
            text = extract_text_from_html(raw).strip()
            if text:
                chapters.append(text)
    return "\n\n".join(chapters)


def extract_text_from_ipynb(file_bytes: bytes) -> str:
    """Stdlib json: markdown cells verbatim, code cells fenced with the
    notebook's language. A notebook is a document with code in it —
    general chunking treats it honestly."""
    import json

    nb = json.loads(file_bytes.decode("utf-8", errors="replace"))
    lang = (
        (nb.get("metadata") or {})
        .get("kernelspec", {})
        .get("language", "")
        or (nb.get("metadata") or {})
        .get("language_info", {})
        .get("name", "")
    )
    parts: list[str] = []
    for cell in nb.get("cells", []):
        source = cell.get("source") or []
        body = "".join(source) if isinstance(source, list) else str(source)
        body = body.rstrip()
        if not body.strip():
            continue
        kind = cell.get("cell_type")
        if kind == "markdown":
            parts.append(body)
        elif kind == "code":
            parts.append(f"```{lang}\n{body}\n```")
    return "\n\n".join(parts)


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
        io.BytesIO(safe_zip_bytes(file_bytes)), read_only=True, data_only=True,
    )
    sections = []
    # Lockdown PR-5 (Q45): sheet, row and column caps. Sheets and rows
    # past the cap refuse the workbook (413, "split it"); a row wider
    # than the column cap is cut with a visible marker.
    if len(wb.worksheets) > XLSX_MAX_SHEETS:
        wb.close()
        raise DocumentTooLarge(
            f"The workbook has more than {XLSX_MAX_SHEETS} sheets; split it."
        )
    rows_seen = 0
    for ws in wb.worksheets:
        lines = [f"{TABULAR_SHEET_MARKER}{ws.title} ==="]
        for row in ws.iter_rows(values_only=True):
            cells = [
                str(c).replace("\t", " ").replace("\n", " ").strip()
                if c is not None else ""
                for c in row
            ]
            if any(cells):
                rows_seen += 1
                if rows_seen > XLSX_MAX_ROWS:
                    wb.close()
                    raise DocumentTooLarge(
                        f"The workbook has more than {XLSX_MAX_ROWS:,} "
                        "non-empty rows; split it."
                    )
                if len(cells) > XLSX_MAX_COLUMNS:
                    cells = cells[:XLSX_MAX_COLUMNS] + [XLSX_COLUMNS_TRUNCATED_MARKER]
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
    # Lockdown PR-5 (Q45): one message's text is bounded.
    if len(body.encode("utf-8", errors="replace")) > MBOX_MAX_MESSAGE_TEXT_BYTES:
        body = body.encode("utf-8", errors="replace")[:MBOX_MAX_MESSAGE_TEXT_BYTES].decode(
            "utf-8", errors="ignore",
        ) + f"\n{MBOX_MESSAGE_TRUNCATED_MARKER}"
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
        seen = 0
        for raw in box:
            seen += 1
            if seen > MBOX_MAX_MESSAGES:
                # Q45: refused, never read in part.
                box.close()
                raise DocumentTooLarge(
                    f"The mailbox has more than {MBOX_MAX_MESSAGES:,} messages; split it."
                )
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
