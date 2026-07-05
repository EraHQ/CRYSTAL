"""Tests for the create_document artifact tool (P4d).

create_document is a pure packaging tool — no state, no external calls — so
these assert the artifact contract the frontend renders against, plus the
format guard and filename normalization. Registration into the agent tool
registry is verified out-of-band in a fresh process (the registry is a
session-global singleton, so resetting it in-suite poisons later tests).
"""
from crystal_cache.agent.tools.artifacts import create_document


async def test_create_document_md_shape():
    r = await create_document(
        "cus_x", content="# Title\n\nbody", filename="notes", format="md", title="My Notes"
    )
    assert r["type"] == "document"
    assert r["filename"] == "notes.md"
    assert r["format"] == "md"
    assert r["mime"] == "text/markdown"
    assert r["title"] == "My Notes"
    assert r["content"] == "# Title\n\nbody"
    assert r["bytes"] == len("# Title\n\nbody".encode("utf-8"))


async def test_create_document_defaults_to_md_and_adds_extension():
    r = await create_document("cus_x", content="hello")
    assert r["format"] == "md"
    assert r["filename"].endswith(".md")
    # title falls back to the filename when omitted
    assert r["title"] == r["filename"]


async def test_create_document_txt_and_html_mime():
    rt = await create_document("cus_x", content="plain", filename="a.txt", format="txt")
    assert rt["mime"] == "text/plain"
    assert rt["filename"] == "a.txt"

    rh = await create_document("cus_x", content="<p>hi</p>", filename="page", format="html")
    assert rh["mime"] == "text/html"
    assert rh["filename"] == "page.html"


async def test_create_document_rejects_unknown_format():
    r = await create_document("cus_x", content="x", format="pdf")
    assert "error" in r
    assert "pdf" in r["error"]


async def test_create_document_filename_has_no_path_separators():
    r = await create_document(
        "cus_x", content="x", filename="../../etc/passwd", format="txt"
    )
    assert "/" not in r["filename"]
    assert "\\" not in r["filename"]
    assert r["filename"].endswith(".txt")
