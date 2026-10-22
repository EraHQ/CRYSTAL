"""Images as documents (hackathon item 4, Q97 A, 2026-10-22).

A photo of a whiteboard, a screenshot, a chart: the knowledge in it is
whatever a vision model can read off it. This module turns the image into
the TEXT the rest of ingestion already knows how to handle (chunker,
extractor, review), so facts born from an image cite the image file like
facts from a PDF cite the PDF, and nothing downstream learns a new type.

The model is the tenant's background lane: on the Nemotron lane (a
managed tenant's pick, or a tenant's own Token Factory key) the call goes
to Super-VL, the one NVIDIA vision model Token Factory lists
(CC_NEMOTRON_MODEL_VISION); on the Anthropic lane the large tier reads
images natively; a byok tenant on another provider gets their own model,
which must take images. No model at all is a 415 at the door
(VisionUnavailable), never a silent text document.

One call per image, reasoning off on Nemotron (a transcription is not a
judgment), ledgered under origin image_describe like every model call.
"""
from __future__ import annotations

import asyncio

import structlog

from ..config import get_settings
from ..cost.emit import record_model_call
from ..llm.client import get_llm_client
from ..llm.lanes import tenant_background
from .file_extract import DocumentTooLarge

logger = structlog.get_logger(__name__)

ORIGIN = "image_describe"

IMAGE_SYSTEM = """You read one image for a personal knowledge base and write down everything in it that a person might want to remember or ask about later.

Write Markdown with these sections, in this order, leaving a section out when it has nothing:

## Text
Every piece of legible text, verbatim, in reading order. Keep numbers, names, dates, prices and units exactly as written. Write [illegible] for what cannot be read; never guess a word.

## What it shows
What the image is (whiteboard, screenshot, photo, chart, diagram, table, receipt, slide) and what it depicts, in plain sentences. For a chart, state the axes and the values you can read. For a table, write it as a Markdown table. For a diagram, list the parts and the arrows between them. For a screenshot, say what application or page it is and the state it is in.

## Worth remembering
The facts a person would want recalled: decisions, numbers, names, dates, relationships between things, anything marked, circled or highlighted. One per line. Only what the image supports; no interpretation beyond it.

Rules: describe, never invent; say so when something is unclear; no preamble and no closing remarks."""

IMAGE_PROMPT = "Read this image. Its file name is: {label}"


class VisionUnavailable(Exception):
    """No model that reads images is configured for this tenant's lane."""


class VisionEmpty(Exception):
    """The model returned nothing legible for the image."""


async def describe_image(
    image_bytes: bytes,
    mime: str,
    *,
    customer,
    store,
    label: str,
) -> str:
    """The image as document text: an 'Image: <label>' header line, then
    the model's Markdown. Raises DocumentTooLarge, VisionUnavailable or
    VisionEmpty; the callers map them to the upload codes."""
    settings = get_settings()
    max_bytes = int(settings.image_max_bytes)
    if len(image_bytes) > max_bytes:
        raise DocumentTooLarge(
            f"The image is larger than {max_bytes // 2**20} MB. Resize it."
        )
    async with tenant_background(customer, store):
        client = get_llm_client()
        if not client.is_ready():
            raise VisionUnavailable("No model that reads images is configured.")
        family = getattr(client, "family", None)
        model = settings.nemotron_model_vision if family == "nemotron" else None
        # Nemotron: the explicit vision id, small-tier policy (reasoning
        # off); Anthropic: the large tier reads images; another byok
        # provider: its own model, large tier.
        tier = "small" if family == "nemotron" else "large"
        message = client.image_message(
            image_bytes=image_bytes, mime=mime,
            prompt=IMAGE_PROMPT.format(label=label or "image"),
        )
        result = await asyncio.to_thread(
            client.complete_detailed,
            system=IMAGE_SYSTEM,
            messages=[message],
            max_tokens=int(settings.image_describe_max_tokens),
            temperature=0.0,
            tier=tier,
            model=model,
        )
        await record_model_call(
            customer_id=getattr(customer, "id", ""),
            model=result.model,
            origin=ORIGIN,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cache_creation_tokens=result.cache_creation_tokens,
            cache_read_tokens=result.cache_read_tokens,
            store=store,
        )
    text = (result.text or "").strip()
    logger.info("image.described", customer_id=getattr(customer, "id", None),
                label=label, model=result.model, bytes=len(image_bytes),
                chars=len(text), input_tokens=result.input_tokens,
                output_tokens=result.output_tokens)
    if not text:
        raise VisionEmpty("The model read nothing from the image.")
    return f"Image: {label}\n\n{text}"
