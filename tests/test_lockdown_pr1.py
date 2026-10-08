"""Lockdown PR-1 (2026-10-08, AUDIT_LAUNCH_VERIFY): nothing a request can
do takes the instance down.

- B2-5 / B5-1: a file upload is bounded in bytes before extraction and in
  characters after it (413, never silent truncation); the JSON route had
  the character cap all along, the file route had neither.
- B2-6: zip-based documents (docx, pptx, xlsx, odt, epub) pass a
  pre-flight (member count, declared sizes, compression ratio, and a
  bounded read that headers cannot lie past) before any library opens
  them.
- B2-7: a PDF past 500 pages, or past the time budget, is refused.
- B2-8: a file type ingestion does not accept is 415 instead of being
  UTF-8 decoded and paid for as text.
- B1-2 = B3-3 = B5-2: the topology restore gunzips through a hard ceiling.
- B5-8: one topology export in flight per tenant; a second is 409.
- B5-10: the transcript speaker regex is linear.
- B1-44: the MCP ingest cap is never "unbounded".
- Source sync: an oversize or binary file is skipped for good, not
  retried three times.

Every route pin drives the real app over the real store.
"""
from __future__ import annotations

import gzip
import io
import time
import zipfile

import pytest

from crystal_cache.ingestion import file_extract as fx
from crystal_cache.infrastructure.schema import DocumentUploadRow


def _app(store, encoder, vector_store, fact_vector_store):
    try:
        from tests.test_endpoint_smoke import _build_app
    except ModuleNotFoundError:
        from test_endpoint_smoke import _build_app
    return _build_app(store, encoder, vector_store, fact_vector_store)


def _client(app):
    from httpx import ASGITransport, AsyncClient
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t")


def _zip(members: dict[str, bytes]) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return b.getvalue()


def _minimal_pdf(n_pages: int) -> bytes:
    """A valid PDF with n empty pages and a correct xref table."""
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{3 + i} 0 R" for i in range(n_pages))
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    for _ in range(n_pages):
        objs.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(out)


def _msg(r) -> str:
    """The test app mounts routers without app.py's envelope handlers, so
    read both shapes (the production contract is error.message)."""
    body = r.json()
    if isinstance(body.get("error"), dict):
        return body["error"].get("message", "")
    return str(body.get("detail", ""))


async def _doc_count(store) -> int:
    from sqlalchemy import func, select
    async with store.session() as s:
        return int((await s.execute(select(func.count(DocumentUploadRow.id)))).scalar() or 0)


# ---------------------------------------------------------------------------
# Upload route: byte cap, character cap, unknown type, archive bombs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upload_past_the_byte_cap_is_413_before_extraction(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store, monkeypatch,
):
    from crystal_cache.config import settings
    from crystal_cache.endpoints import documents as docs_mod
    monkeypatch.setattr(settings, "upload_max_bytes", 1 << 20)
    calls = []
    monkeypatch.setattr(docs_mod, "extract_text_from_file",
                        lambda *a, **k: calls.append(1) or "text")
    app = _app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    before = await _doc_count(store)
    async with _client(app) as c:
        r = await c.post(
            "/v1/documents/upload",
            headers={"Authorization": f"Bearer {customer.api_key}"},
            files={"file": ("big.txt", b"a" * ((1 << 20) + 1), "text/plain")},
        )
    assert r.status_code == 413, r.text
    assert "larger than 1 MB" in _msg(r)
    assert calls == [], "extraction ran on a file past the byte cap"
    assert await _doc_count(store) == before


@pytest.mark.asyncio
async def test_upload_past_the_character_cap_is_413_and_never_truncated(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store, monkeypatch,
):
    from crystal_cache.config import settings
    monkeypatch.setattr(settings, "document_max_chars", 1_000)
    app = _app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    before = await _doc_count(store)
    headers = {"Authorization": f"Bearer {customer.api_key}"}
    async with _client(app) as c:
        r = await c.post("/v1/documents/upload", headers=headers,
                         files={"file": ("long.txt", b"a" * 1_001, "text/plain")})
        assert r.status_code == 413, r.text
        assert "Extracted text exceeds 1,000 characters" in _msg(r)
        assert await _doc_count(store) == before
        # Exactly at the cap is accepted whole.
        r = await c.post("/v1/documents/upload", headers=headers,
                         files={"file": ("ok.txt", b"a" * 1_000, "text/plain")})
        assert r.status_code == 200, r.text
        assert r.json()["char_count"] == 1_000


@pytest.mark.asyncio
async def test_unknown_file_types_are_415(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    app = _app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    before = await _doc_count(store)
    headers = {"Authorization": f"Bearer {customer.api_key}"}
    async with _client(app) as c:
        for name, mime in (
            ("photo.png", "image/png"),
            ("bundle.zip", "application/zip"),
            ("tool.exe", "application/octet-stream"),
            ("legacy.doc", "application/msword"),
            ("blob", "image/jpeg"),
        ):
            r = await c.post("/v1/documents/upload", headers=headers,
                             files={"file": (name, b"\x89PNG\r\n\x1a\n....", mime)})
            assert r.status_code == 415, (name, r.status_code, r.text[:200])
            assert "Unsupported file type" in _msg(r)
    assert await _doc_count(store) == before


@pytest.mark.asyncio
async def test_code_and_text_types_on_the_allowlist_still_upload(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    app = _app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    headers = {"Authorization": f"Bearer {customer.api_key}"}
    async with _client(app) as c:
        for name in ("notes.md", "main.py", "data.jsonl", "run.sh", "app.tsx"):
            r = await c.post("/v1/documents/upload", headers=headers,
                             files={"file": (name, b"hello from " + name.encode(), "application/octet-stream")})
            assert r.status_code == 200, (name, r.text[:200])


@pytest.mark.asyncio
async def test_docx_bomb_is_refused_at_the_route(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    # 2 MiB of zeros compresses ~1000:1 — a real document never does.
    bomb = _zip({"[Content_Types].xml": b"<x/>", "word/document.xml": b"\0" * (2 << 20)})
    app = _app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    before = await _doc_count(store)
    async with _client(app) as c:
        r = await c.post(
            "/v1/documents/upload",
            headers={"Authorization": f"Bearer {customer.api_key}"},
            files={"file": ("bomb.docx", bomb, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        )
    assert r.status_code == 413, r.text
    assert "compressed far beyond" in _msg(r)
    assert await _doc_count(store) == before


# ---------------------------------------------------------------------------
# Extraction module: every zip family, every limit, the PDF caps
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["a.docx", "a.pptx", "a.xlsx", "a.odt", "a.epub"])
def test_every_zip_family_runs_the_preflight(name):
    members = {
        "[Content_Types].xml": b"<x/>",
        "word/document.xml": b"\0" * (2 << 20),
        "ppt/slides/slide1.xml": b"\0" * (2 << 20),
        "content.xml": b"\0" * (2 << 20),
        "META-INF/container.xml": b"\0" * (2 << 20),
    }
    with pytest.raises(fx.DocumentTooLarge):
        fx.extract_text_from_file(_zip(members), name)


def test_preflight_limits_members_declared_size_and_lying_headers(monkeypatch):
    with pytest.raises(fx.DocumentTooLarge, match="2,001 parts"):
        fx.safe_zip_bytes(_zip({f"m{i}.xml": b"<a/>" for i in range(2_001)}))

    monkeypatch.setattr(fx, "ZIP_MAX_MEMBER_BYTES", 1 << 20)
    with pytest.raises(fx.DocumentTooLarge, match="larger than"):
        fx.safe_zip_bytes(_zip({"a.xml": bytes(range(256)) * (8 << 12)}))  # 2 MiB, ratio ~1

    # A header that declares a tiny member hiding a big stream: zipfile
    # reads by compressed size, so only the bounded read (or the CRC
    # check it triggers) can catch it. Either way it is refused.
    real = _zip({"a.xml": bytes(range(256)) * (8 << 12)})

    class _Liar(zipfile.ZipFile):
        def infolist(self):
            infos = super().infolist()
            for zi in infos:
                zi.file_size = 10
            return infos

    monkeypatch.setattr(fx.zipfile, "ZipFile", _Liar)
    with pytest.raises((fx.DocumentTooLarge, fx.UnsupportedFileType)):
        fx.safe_zip_bytes(real)


def test_not_a_zip_and_missing_members_are_unsupported():
    with pytest.raises(fx.UnsupportedFileType):
        fx.extract_text_from_file(b"definitely not a zip", "a.docx")
    for name in ("a.docx", "a.odt", "a.epub"):
        with pytest.raises(fx.UnsupportedFileType):
            fx.extract_text_from_file(_zip({"x.xml": b"<a/>"}), name)


def test_pdf_page_cap_and_time_budget():
    pytest.importorskip("pdfplumber")
    with pytest.raises(fx.DocumentTooLarge, match="more than 500 pages"):
        fx.extract_text_from_pdf(_minimal_pdf(501))
    assert fx.extract_text_from_pdf(_minimal_pdf(3)) == ""  # empty pages, no error
    with pytest.raises(fx.DocumentTooLarge, match="took too long"):
        fx.extract_text_from_pdf(_minimal_pdf(2), time_budget_seconds=0)


def test_allowlist_is_exactly_the_console_accept_list():
    """The console's accept= list is the server allowlist (Q45=A)."""
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "frontend" / "src" / "pages" / "KnowledgeManager.tsx"
    text = src.read_text(encoding="utf-8")
    import re
    accepts = set()
    for m in re.finditer(r'accept="([^"]+)"', text):
        accepts.update(x.strip() for x in m.group(1).split(","))
    assert accepts, "no accept= list found in KnowledgeManager.tsx"
    assert accepts == set(fx.ACCEPTED_EXTENSIONS)


# ---------------------------------------------------------------------------
# Topology restore: bounded gunzip
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_topology_gzip_bomb_is_413_and_valid_gzip_still_imports(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store, monkeypatch,
):
    from crystal_cache.config import settings
    monkeypatch.setattr(settings, "topology_import_max_bytes", 1 << 20)
    app = _app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    headers = {"Authorization": f"Bearer {customer.api_key}", "Content-Encoding": "gzip",
               "Content-Type": "application/json"}
    async with _client(app) as c:
        bomb = gzip.compress(b"\0" * (4 << 20))
        r = await c.post("/v1/import/topology", headers=headers, content=bomb)
        assert r.status_code == 413, r.text
        assert "unpacks to more than 1 MB" in _msg(r)

        r = await c.post("/v1/import/topology", headers=headers, content=bomb[:500])
        assert r.status_code == 400, r.text
        assert "not valid gzip" in _msg(r)

        ok = gzip.compress(b'{"crystals": [], "facts": []}')
        r = await c.post("/v1/import/topology", headers=headers, content=ok)
        assert r.status_code == 200, r.text


def test_gunzip_bounded_edges():
    from crystal_cache.endpoints.sdk import gunzip_bounded
    small = gzip.compress(b'{"a":1}')
    assert gunzip_bounded(small, 1_000) == b'{"a":1}'
    assert gunzip_bounded(small + b"trailing garbage", 1_000) == b'{"a":1}'
    with pytest.raises(ValueError, match="too large"):
        gunzip_bounded(gzip.compress(b"\0" * (8 << 20)), 1 << 20)
    with pytest.raises(ValueError, match="truncated"):
        gunzip_bounded(gzip.compress(b"\0" * (8 << 20))[:200], 1 << 30)


# ---------------------------------------------------------------------------
# Export: one in flight per tenant
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_second_concurrent_export_is_409_and_the_lock_is_released(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    import asyncio
    from crystal_cache.endpoints import sdk as sdk_mod
    app = _app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    headers = {"Authorization": f"Bearer {customer.api_key}"}
    lock = sdk_mod._export_locks.setdefault(customer.id, asyncio.Lock())
    await lock.acquire()
    try:
        async with _client(app) as c:
            r = await c.post("/v1/export/topology", headers=headers)
            assert r.status_code == 409, r.text
            assert "already running" in _msg(r)
    finally:
        lock.release()
    async with _client(app) as c:
        r = await c.post("/v1/export/topology", headers=headers)
        assert r.status_code == 200, r.text
        assert r.headers.get("content-encoding") == "gzip"
    assert not sdk_mod._export_locks[customer.id].locked()


# ---------------------------------------------------------------------------
# Regex, MCP cap, source sync
# ---------------------------------------------------------------------------

def test_transcript_speaker_regex_is_linear():
    from crystal_cache.ingestion.document_chunker import _chunk_transcript
    line = "A" + " " * 100_000
    started = time.monotonic()
    _chunk_transcript(line + "\nAlice: hi\nBob Smith - hello\n")
    assert time.monotonic() - started < 1.0
    # Speaker detection is unchanged for the shapes transcripts carry.
    import re
    from crystal_cache.ingestion import document_chunker as dc
    src = __import__("inspect").getsource(dc._chunk_transcript)
    pat = re.compile(re.search(r"re\.compile\(r'([^']+)'", src).group(1))
    for ln, who in (("Alice: hi", "Alice"), ("Bob Smith - hello", "Bob Smith"),
                    ("Mary Jane Watson: x", "Mary Jane Watson"), ("alice: no", None)):
        m = pat.match(ln)
        assert (m.group(1) if m else None) == who, ln


@pytest.mark.asyncio
async def test_mcp_ingest_cap_zero_means_the_document_cap_not_unbounded(monkeypatch):
    from crystal_cache.agent import mcp_server
    from crystal_cache.config import settings
    monkeypatch.setattr(settings, "mcp_ingest_max_chars", 0)
    monkeypatch.setattr(settings, "document_max_chars", 1_000)
    async def _open(**kw):
        return None

    monkeypatch.setattr(mcp_server, "_write_admission_block", _open)
    fn = getattr(mcp_server.memory_ingest, "fn", mcp_server.memory_ingest)
    token = mcp_server._current_customer_id.set("cus_test")
    try:
        out = await fn(text="a" * 1_001, label="x")
    finally:
        mcp_server._current_customer_id.reset(token)
    assert out.get("code") == "ingest_too_large", out
    assert out.get("max_chars") == 1_000


@pytest.mark.asyncio
async def test_source_sync_skips_an_oversize_or_binary_file_for_good(store, customer, monkeypatch):
    """Through the real sync loop: a file past a hard limit is attempted
    once, not MAX_FILE_ATTEMPTS times, and the cycle advances."""
    from crystal_cache.ingestion.source_handlers import ChangeSet, SourceEnvelope, register_handler
    from crystal_cache.workers import source_sync

    class _Handler:
        scheme = "testbig"

        async def check(self, watch, token):
            if (watch.last_state or {}).get("head") == "h1":
                return None
            return ChangeSet(new_state={"head": "h1"}, changed=["big.pdf", "logo.png", "good.md"])

        async def fetch(self, watch, path, token):
            payload = b"\0" * 64 if path == "logo.png" else b"x"
            return SourceEnvelope(payload_bytes=payload, mime_type="application/octet-stream",
                                  source_uri=f"repo://r/{path}", label=path)

    register_handler(_Handler())
    calls: dict[str, int] = {}
    real_ingest = source_sync._ingest_envelope

    async def _ingest(store_, encoder, vs, fvs, llm, watch, envelope):
        calls[envelope.label] = calls.get(envelope.label, 0) + 1
        if envelope.label == "big.pdf":
            raise fx.DocumentTooLarge("The file is larger than 25 MB. Split it.")
        if envelope.label == "logo.png":
            # the real function's binary check, driven for real
            await real_ingest(store_, encoder, vs, fvs, llm, watch, envelope)
        # good.md: nothing to do

    monkeypatch.setattr(source_sync, "_ingest_envelope", _ingest)
    w = await store.create_source_watch(
        customer.id, scheme="testbig", source_name="r", config={},
        cadence_minutes=15, review_mode="auto", encrypted_token=None,
    )
    for _ in range(4):
        w = await store.get_source_watch(w.id, customer.id)
        await source_sync.sync_one_watch(store, None, None, None, None, w)
    assert calls == {"big.pdf": 1, "logo.png": 1, "good.md": 1}, calls
    w = await store.get_source_watch(w.id, customer.id)
    assert (w.last_state or {}).get("head") == "h1", "the cycle did not advance"
    failures = (w.last_state or {}).get(source_sync.FILE_FAILURES_KEY) or {}
    assert failures["big.pdf"]["attempts"] == source_sync.MAX_FILE_ATTEMPTS
    assert failures["logo.png"]["attempts"] == source_sync.MAX_FILE_ATTEMPTS
