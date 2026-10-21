"""Q93 A (2026-10-20): document extraction asks for its reply shape with
a strict JSON schema, and reads the schema's object as well as the bare
array older fakes return. Found on the demo tenant's first seed on the
Nemotron lane: two of thirteen documents came back with zero items
because the small model's reply held no parseable array.

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

import json

import pytest

from crystal_cache.ingestion.document_pipeline import (
    EXTRACTION_SCHEMA,
    EXTRACTION_SYSTEM,
    DocumentPipeline,
)


class _Reply:
    model = "m"; input_tokens = 1; output_tokens = 1
    cache_creation_tokens = 0; cache_read_tokens = 0

    def __init__(self, text: str):
        self.text = text


class _Fake:
    def __init__(self, text: str):
        self.text = text
        self.kwargs: list[dict] = []

    def complete_detailed(self, **kwargs):
        self.kwargs.append(kwargs)
        return _Reply(self.text)


def _pipeline(fake) -> DocumentPipeline:
    return DocumentPipeline(store=None, encoder=None, vector_store=None, client=fake)


@pytest.mark.asyncio
async def test_extraction_sends_the_schema_and_reads_its_object():
    fake = _Fake(json.dumps({"items": [
        {"key": "Pro price", "segments": ["Kerning", "Pricing", "Pro"], "value": "Pro is $12 a month.", "citation": "", "type": "fact"},
        {"key": "Lena Marsh", "segments": ["Kerning", "People", "Lena"], "value": "Lena Marsh advises Dana.", "citation": "", "type": "entity"},
    ]}))
    items = await _pipeline(fake).extract_items(text="Pro is $12 a month. Lena Marsh advises Dana.", label="Brief", chunk_size=500)
    assert [i.value for i in items] == ["Pro is $12 a month.", "Lena Marsh advises Dana."]
    assert items[0].sparse_key.startswith("Kerning|Pricing")
    sent = fake.kwargs[0]
    assert sent["json_schema"] is EXTRACTION_SCHEMA and sent["tier"] == "small"
    assert EXTRACTION_SCHEMA["required"] == ["items"]
    assert "type" in EXTRACTION_SCHEMA["properties"]["items"]["items"]["required"]
    assert '{"items": [' in EXTRACTION_SYSTEM


@pytest.mark.asyncio
async def test_extraction_still_reads_a_bare_array_and_a_fenced_one():
    bare = json.dumps([{"key": "k", "segments": ["A", "B"], "value": "v", "type": "fact"}])
    assert len(await _pipeline(_Fake(bare)).extract_items(text="text", label="L", chunk_size=500)) == 1
    fenced = "```json\n" + json.dumps({"items": [{"key": "k", "segments": ["A"], "value": "v", "type": "fact"}]}) + "\n```"
    assert len(await _pipeline(_Fake(fenced)).extract_items(text="text", label="L", chunk_size=500)) == 1


@pytest.mark.asyncio
async def test_unparseable_reply_yields_nothing_and_is_logged(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="crystal_cache.ingestion.document_pipeline"):
        items = await _pipeline(_Fake("Here are the facts I found: none worth noting.")).extract_items(text="text", label="L", chunk_size=500)
    assert items == []
    assert any("extraction_unparsed" in r.getMessage() for r in caplog.records)
