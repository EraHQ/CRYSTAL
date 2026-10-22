"""Hackathon item 4 (Q97 A, 2026-10-22): an image is a document whose text
a vision model reads off it. Detection by extension then MIME (a named
text document wins over a lying image MIME), bytes sniffed by magic
number, the describer picks Super-VL on the Nemotron lane and the large
tier on Anthropic, ledgers the call, and the upload route stores the
description as the document text. No network: the model is a fake."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from crystal_cache.ingestion import file_extract as fx
from crystal_cache.ingestion import image_describe as img
from crystal_cache.llm.client import LLMClient, LLMResult, reset_llm_client, set_llm_client

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 64
VISION = "nvidia/Nemotron-3.5-Super-VL"


def test_image_detection_and_sniff():
    assert fx.image_mime_for("Whiteboard.JPG", None) == "image/jpeg"
    assert fx.image_mime_for("shot.png", "application/octet-stream") == "image/png"
    assert fx.image_mime_for("blob", "image/webp") == "image/webp"
    assert fx.image_mime_for("blob", "image/gif") is None
    assert fx.image_mime_for("notes.pdf", "image/png") is None  # the named document wins
    assert fx.image_mime_for("notes.md", None) is None
    fx.sniff_image(PNG, "image/png"); fx.sniff_image(JPEG, "image/jpeg"); fx.sniff_image(WEBP, "image/webp")
    with pytest.raises(fx.UnsupportedFileType, match="not a PNG image"):
        fx.sniff_image(JPEG, "image/png")
    with pytest.raises(fx.UnsupportedFileType, match="not a WebP image"):
        fx.sniff_image(PNG, "image/webp")
    # Images never joined the text allow-list: the extractors refuse them as before.
    assert ".png" not in fx.ACCEPTED_EXTENSIONS
    with pytest.raises(fx.UnsupportedFileType):
        fx.extract_text_from_file(PNG, "photo.png", mime="image/png")


def test_image_message_takes_each_provider_shape():
    a = LLMClient(provider="anthropic", api_key="k", base_url=None, model_small="s", model_large="l", model_frontier="f")
    m = a.image_message(image_bytes=PNG, mime="image/png", prompt="Read it")
    assert m["role"] == "user" and m["content"][0]["type"] == "image"
    assert m["content"][0]["source"]["media_type"] == "image/png" and m["content"][0]["source"]["data"].startswith("iVBOR")
    assert m["content"][1] == {"type": "text", "text": "Read it"}
    o = LLMClient(provider="openai", api_key="k", base_url="https://tf.test/v1", model_small="s", model_large="l", model_frontier="f", family="nemotron")
    m = o.image_message(image_bytes=PNG, mime="image/png", prompt="Read it")
    assert m["content"][0] == {"type": "text", "text": "Read it"}
    assert m["content"][1]["type"] == "image_url" and m["content"][1]["image_url"]["url"].startswith("data:image/png;base64,iVBOR")


class _FakeVision:
    """A client with the two seam methods the describer uses."""

    def __init__(self, family, text="## Text\nPro plan $15\n\n## Worth remembering\n- Pro costs $15 from October", ready=True):
        self.family = family
        self.calls = []
        self._text = text
        self._ready = ready
        self._provider = "openai" if family == "nemotron" else "anthropic"

    def is_ready(self):
        return self._ready

    def image_message(self, *, image_bytes, mime, prompt):
        return LLMClient.image_message(self, image_bytes=image_bytes, mime=mime, prompt=prompt)

    def complete_detailed(self, *, system, messages, max_tokens, temperature=0.0, tier="small", model=None, json_schema=None):
        self.calls.append({"system": system, "messages": messages, "max_tokens": max_tokens, "tier": tier, "model": model})
        return LLMResult(text=self._text, model=model or f"{self.family}-large", input_tokens=900, output_tokens=120)


@pytest.fixture
def recorded(monkeypatch):
    rows = []

    async def fake_record(**kw):
        rows.append(kw); return {"id": "llm_x"}
    monkeypatch.setattr(img, "record_model_call", fake_record)
    return rows


@pytest.mark.asyncio
async def test_describe_image_picks_the_lane_model_and_ledgers_the_call(monkeypatch, recorded):
    customer = SimpleNamespace(id="cus_img", inference_mode="managed",
                               model_routing_config=SimpleNamespace(model_id="claude-sonnet-5-5", background_model_id=None, model_caps={}, api_key_ref=""))
    # Nemotron lane: Super-VL by explicit id, small-tier policy (reasoning off), the house prompt.
    nem = _FakeVision("nemotron")
    set_llm_client(nem)
    try:
        text = await img.describe_image(PNG, "image/png", customer=customer, store=None, label="whiteboard.png")
    finally:
        reset_llm_client()
    assert text.startswith("Image: whiteboard.png\n\n## Text")
    call = nem.calls[0]
    assert call["model"] == VISION and call["tier"] == "small" and call["max_tokens"] == 3000
    assert "## Worth remembering" in call["system"] and "never invent" in call["system"]
    assert call["messages"][0]["content"][0]["text"] == "Read this image. Its file name is: whiteboard.png"
    assert call["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert recorded[-1]["origin"] == "image_describe" and recorded[-1]["model"] == VISION
    assert recorded[-1]["input_tokens"] == 900 and recorded[-1]["customer_id"] == "cus_img"
    # Anthropic lane: no explicit model, the large tier reads the image natively.
    cla = _FakeVision("claude")
    set_llm_client(cla)
    try:
        await img.describe_image(JPEG, "image/jpeg", customer=customer, store=None, label="receipt.jpg")
    finally:
        reset_llm_client()
    assert cla.calls[0]["model"] is None and cla.calls[0]["tier"] == "large"
    assert cla.calls[0]["messages"][0]["content"][0]["type"] == "image"
    # No model: refused at the door; empty read: refused; too big: refused before any call.
    set_llm_client(_FakeVision("claude", ready=False))
    try:
        with pytest.raises(img.VisionUnavailable):
            await img.describe_image(PNG, "image/png", customer=customer, store=None, label="x.png")
    finally:
        reset_llm_client()
    set_llm_client(_FakeVision("nemotron", text="   "))
    try:
        with pytest.raises(img.VisionEmpty):
            await img.describe_image(PNG, "image/png", customer=customer, store=None, label="x.png")
    finally:
        reset_llm_client()
    from crystal_cache import config
    monkeypatch.setattr(config.settings, "image_max_bytes", 32)
    big = _FakeVision("nemotron")
    set_llm_client(big)
    try:
        with pytest.raises(fx.DocumentTooLarge):
            await img.describe_image(PNG, "image/png", customer=customer, store=None, label="x.png")
    finally:
        reset_llm_client()
    assert big.calls == []


@pytest.mark.asyncio
async def test_upload_route_stores_the_read_image_as_a_document(
    monkeypatch, store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    try:
        from tests.test_lockdown_pr5 import _app, _client, _upload
    except ModuleNotFoundError:
        from test_lockdown_pr5 import _app, _client, _upload
    from crystal_cache.infrastructure.schema import DocumentUploadRow
    from sqlalchemy import select

    seen = []

    async def fake_describe(image_bytes, mime, *, customer, store, label):
        seen.append((mime, label, len(image_bytes), customer.id))
        return f"Image: {label}\n\n## Text\nLaunch day November 12"
    monkeypatch.setattr(img, "describe_image", fake_describe)
    app = _app(store, semantic_encoder_stub, vector_store, fact_vector_store)
    async with _client(app) as c:
        r = await _upload(c, customer.api_key, "Whiteboard photo.jpg", JPEG, "image/jpeg")
        assert r.status_code == 200, r.text
        doc_id = r.json()["id"]
        # A PNG by name whose bytes are a JPEG never reaches the model.
        r = await _upload(c, customer.api_key, "fake.png", JPEG, "image/png")
        assert r.status_code == 415 and "not a PNG image" in r.text
    assert seen == [("image/jpeg", "Whiteboard photo.jpg", len(JPEG), customer.id)]
    async with store.session() as session:
        row = await session.get(DocumentUploadRow, doc_id)
    assert row.label == "Whiteboard photo.jpg" and row.text.startswith("Image: Whiteboard photo.jpg")
    assert "Launch day November 12" in row.text
