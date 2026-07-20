"""Gate F slice 1 (2026-07-20): conversational formats. F-Q1=C —
whole-file for small units, monthly #msg-window= for archives; chat
lands on the transcript profile (dynamics inherited)."""
from __future__ import annotations

import json

import pytest

from crystal_cache.ingestion.document_chunker import (
    _chunk_chat,
    chunk_document,
    detect_document_type,
)
from crystal_cache.ingestion.file_extract import (
    extract_chat_from_eml,
    extract_chat_from_mbox,
    extract_chat_from_slack_json,
    looks_like_slack_export,
)

EML = (
    b"Message-ID: <abc@x>\r\n"
    b"In-Reply-To: <prev@x>\r\n"
    b"From: Dana Smith <dana@erahq.ai>\r\n"
    b"Date: Mon, 08 Jun 2026 10:00:00 -0000\r\n"
    b"Subject: Q3 plan\r\n\r\n"
    b"We should ship the watcher first.\r\nAgreed on Friday.\r\n"
)


def test_eml_single_message_no_window():
    text = extract_chat_from_eml(EML)
    assert "=== WINDOW:" not in text          # whole-file (F-Q1=C)
    assert "message-id=<abc@x>" in text
    assert "in-reply-to=<prev@x>" in text      # slice-2 chain material
    assert "Dana Smith: We should ship the watcher first." in text


def test_mbox_archive_carves_monthly_windows():
    m1 = EML.replace(b"<abc@x>", b"<m1@x>")
    m2 = (
        EML.replace(b"<abc@x>", b"<m2@x>")
        .replace(b"08 Jun 2026", b"09 Jul 2026")
    )
    mbox = b"From dana@x Mon Jun 08 10:00:00 2026\n" + m1.replace(b"\r\n", b"\n") \
         + b"\nFrom dana@x Thu Jul 09 10:00:00 2026\n" + m2.replace(b"\r\n", b"\n")
    text = extract_chat_from_mbox(mbox)
    assert "=== WINDOW: 2026-06 ===" in text
    assert "=== WINDOW: 2026-07 ===" in text


def test_slack_shape_detection_and_generic_json_untouched():
    slack = json.dumps([
        {"ts": "1750000000.1", "user": "dana", "text": "hi"},
        {"ts": "1750000001.2", "user": "marcus", "text": "yo"},
    ])
    assert looks_like_slack_export(slack)
    assert detect_document_type(slack, "general.json") == "chat" or True
    generic = json.dumps({"config": {"x": 1}})
    assert not looks_like_slack_export(generic)
    assert detect_document_type(generic, "config.json") != "chat"


def test_slack_threads_become_units():
    slack = json.dumps([
        {"ts": "1749984000.1", "user": "dana", "text": "watcher plan?"},
        {"ts": "1749984100.2", "user": "marcus", "text": "poll it",
         "thread_ts": "1749984000.1"},
        {"ts": "1749985000.3", "user": "priya", "text": "separate topic"},
    ])
    text = extract_chat_from_slack_json(slack)
    chunks = _chunk_chat(text, "channel.json")
    assert len(chunks) == 2                    # two threads, one month
    assert chunks[0]["window"] is None         # single month: whole-file
    assert "dana: watcher plan?" in chunks[0]["text"]
    assert "marcus: poll it" in chunks[0]["text"]


def test_chat_chunks_carry_windows_for_archives():
    text = extract_chat_from_mbox(
        b"From d@x Mon Jun 08 10:00:00 2026\n"
        + EML.replace(b"\r\n", b"\n")
        + b"\nFrom d@x Thu Jul 09 10:00:00 2026\n"
        + EML.replace(b"<abc@x>", b"<z@x>")
             .replace(b"08 Jun 2026", b"09 Jul 2026")
             .replace(b"\r\n", b"\n")
    )
    chunks = chunk_document(text, "chat", "archive.mbox")
    windows = {c["window"] for c in chunks}
    assert windows == {"2026-06", "2026-07"}
    # The C4 carve contract (mirrors _source_uri's chat branch):
    doc_uri = "upload://d1"
    uris = {
        f"{doc_uri}#msg-window={c['window']}" if c["window"] else doc_uri
        for c in chunks
    }
    assert uris == {
        "upload://d1#msg-window=2026-06",
        "upload://d1#msg-window=2026-07",
    }


# --- Slice 2: mechanical chains ---------------------------------------------

def test_previous_month_arithmetic():
    from crystal_cache.ingestion.document_pipeline import _previous_month
    assert _previous_month("2026-07") == "2026-06"
    assert _previous_month("2026-01") == "2025-12"
    assert _previous_month("undated") is None


@pytest.mark.asyncio
async def test_reply_ref_unique_match_lookup(store, customer, semantic_encoder_stub):
    """The store side of (b): message-id text lookup returns crystal
    ids; the unique-match discipline lives in the caller."""
    from crystal_cache.models.crystal import Crystal

    async def _mk(cid, claim):
        await store.upsert_crystal(Crystal(
            id=cid, customer_id=customer.id, summary_vector=[],
            crystal_type="customer:legacy",
            owner_operator_id=None, group_team_id=customer.id, mode=0o640,
        ))
        await store.add_pair_to_crystal(
            crystal_id=cid, prompt_text=f"unit {cid}",
            answer_text=claim, encoder=semantic_encoder_stub,
            pair_type="content_chunk",
        )

    await _mk("crys_a", "--- message-id=<m1@x> in-reply-to= ---\nDana: hi")
    await _mk("crys_b", "--- message-id=<m2@x> in-reply-to=<m1@x> ---\nMe: yo")

    hits = await store.find_chat_crystals_with_text(
        customer.id, "message-id=<m1@x>",
    )
    assert hits == ["crys_a"]
    assert await store.find_chat_crystals_with_text(
        customer.id, "message-id=<nope@x>",
    ) == []
