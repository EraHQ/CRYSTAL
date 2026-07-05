"""Artifact tools — produce user-facing artifacts on CRYS's web surface.

On the local (coding) surface CRYS writes files to disk via its hands; on the
web surface there is no shared disk, so a "produced file" is delivered to the
caller as an artifact payload that the Inspector chat renders inline (download
+ preview). `create_document` is the web-surface equivalent of writing a file:
CRYS authors the content, this tool wraps it into a typed artifact the
frontend renders.

Document generation is NOT a provider feature — the controlling model already
produces the content; this tool only packages it (filename + mime + content),
so it needs no external API, no key, and no shared filesystem.

CONTEXT (D-A10): agent-only. Cognition workers compose/analyze; producing a
user-facing document is an agent action on its surface, not a cognition
composition primitive.

FORMATS: md / txt / html (zero-dependency text formats). Binary formats
(docx / pdf) are a later addition — they need rendering libraries and a
bytes/base64 transport rather than the inline text content used here.
"""
from __future__ import annotations

from typing import Any

import structlog

from ..tool_registry import register_tool

logger = structlog.get_logger(__name__)


# Supported document formats -> MIME type. The frontend keys its preview
# (rendered markdown vs. plain text vs. html source) off `format`.
_FORMAT_MIME: dict[str, str] = {
    "md": "text/markdown",
    "txt": "text/plain",
    "html": "text/html",
}

_DEFAULT_FORMAT = "md"


def _normalize_filename(filename: str, fmt: str) -> str:
    """Make a safe download label from the model-supplied name.

    This is a download label, never a filesystem path, so path separators
    are flattened and leading dots stripped. Falls back to a sane default
    when the model omits a name, and ensures the format extension is present.
    """
    name = (filename or "").strip() or f"document.{fmt}"
    name = name.replace("/", "_").replace("\\", "_").lstrip(".")
    if not name:
        name = f"document.{fmt}"
    if not name.lower().endswith(f".{fmt}"):
        name = f"{name}.{fmt}"
    return name


@register_tool(
    name="create_document",
    description=(
        "Produce a downloadable document from content you have written, "
        "delivered to the user inline (with download + preview). Use this "
        "when the user asks for a document, file, report, write-up, notes, or "
        "anything they will want to keep, download, or share rather than just "
        "read in chat. You author the full content yourself; this tool only "
        "packages it. Supported formats: md (markdown), txt (plain text), "
        "html. Returns the artifact descriptor; the surface renders it."
    ),
    contexts={"agent"},
    parameters_schema={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": (
                    "The full document content you have written, in the chosen "
                    "format (e.g. markdown source when format='md')."
                ),
            },
            "filename": {
                "type": "string",
                "description": (
                    "Suggested file name for download (e.g. 'summary.md'). The "
                    "format extension is appended if missing."
                ),
            },
            "format": {
                "type": "string",
                "description": "Document format: 'md', 'txt', or 'html'. Default 'md'.",
                "enum": ["md", "txt", "html"],
                "default": _DEFAULT_FORMAT,
            },
            "title": {
                "type": "string",
                "description": "Optional human title for the document, shown on the card.",
            },
        },
        "required": ["content"],
    },
    returns_description=(
        "{'type': 'document', 'filename': str, 'format': str, 'mime': str, "
        "'title': str, 'content': str, 'bytes': int}  OR  {'error': str}"
    ),
)
async def create_document(
    customer_id: str,
    content: str,
    filename: str = "",
    format: str = _DEFAULT_FORMAT,
    title: str = "",
) -> dict[str, Any]:
    fmt = (format or _DEFAULT_FORMAT).strip().lower()
    if fmt not in _FORMAT_MIME:
        return {
            "error": (
                f"unsupported format {format!r}; supported: "
                f"{', '.join(sorted(_FORMAT_MIME))}"
            ),
        }

    body = content if isinstance(content, str) else str(content)
    name = _normalize_filename(filename, fmt)
    byte_len = len(body.encode("utf-8"))

    logger.info(
        "create_document",
        customer_id=customer_id,
        format=fmt,
        filename=name,
        bytes=byte_len,
    )

    return {
        "type": "document",
        "filename": name,
        "format": fmt,
        "mime": _FORMAT_MIME[fmt],
        "title": (title or "").strip() or name,
        "content": body,
        "bytes": byte_len,
    }
