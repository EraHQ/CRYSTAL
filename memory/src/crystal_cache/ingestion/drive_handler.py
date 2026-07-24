"""Google Drive source handler — DRIVE-Q1=B (2026-07-24): the
"unified drive" tenant the source-handler registry's docstring
anticipated. A watched Drive folder is a normal source_watch
(scheme=gdrive); files normalize into the C6 envelope and ride the
one ingestion spine — schema gates, tabular lanes, typed events,
supersede/retire — exactly like every other source.

Watch shape:
    scheme      = "gdrive"
    source_name = the label authority (e.g. "drive:Wren Ops")
    config      = {"connection_id": ..., "folder_id": ..., "folder_name": ...}

Credentials: NOT the per-watch encrypted_token (that's git-PAT
machinery). Drive tokens live on drive_connections; the handler
resolves the connection from config and refreshes per poll via the
legacy connector's proven dance.

Google-native files export as their office formats — Docs -> .docx,
Sheets -> .xlsx, Slides -> .pptx — landing on the extractors Gates
E and H shipped. Regular files download raw and dispatch by name/mime
through the Gate H MIME map.
"""

from __future__ import annotations

from typing import Optional

import structlog

from .source_handlers import ChangeSet, SourceEnvelope

logger = structlog.get_logger(__name__)

# Google-native mime -> (export mime, extension)
_NATIVE_EXPORTS: dict[str, tuple[str, str]] = {
    "application/vnd.google-apps.document": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".docx",
    ),
    "application/vnd.google-apps.spreadsheet": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xlsx",
    ),
    "application/vnd.google-apps.presentation": (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".pptx",
    ),
}

_FOLDER_MIME = "application/vnd.google-apps.folder"


def _ingestible(name: str, mime: str) -> bool:
    """Everything the ingestion spine can eat: Google-native exports,
    anything in the Gate H MIME map, text/*, or a filename whose
    extension the extractor dispatch knows. Unknown binaries skip —
    a Drive folder full of PSDs should not become error rows."""
    from .file_extract import _MIME_EXTENSIONS

    if mime in _NATIVE_EXPORTS:
        return True
    if mime == _FOLDER_MIME:
        return False
    base = (mime or "").split(";")[0].strip().lower()
    if base in _MIME_EXTENSIONS or base.startswith("text/"):
        return True
    lower = name.lower()
    known = (
        ".pdf", ".docx", ".pptx", ".rtf", ".odt", ".epub", ".txt", ".md",
        ".html", ".htm", ".xlsx", ".csv", ".tsv", ".json", ".jsonl",
        ".ndjson", ".eml", ".mbox", ".vtt", ".srt", ".ipynb",
        ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs",
        ".java", ".rb", ".c", ".h", ".cpp", ".cs", ".php", ".swift",
        ".kt", ".sh",
    )
    return lower.endswith(known)


class DriveSourceHandler:
    scheme = "gdrive"

    def __init__(self, store) -> None:
        # The handler needs the store for connection/token resolution —
        # unlike git, whose credential rides the watch row itself.
        self._store = store

    async def _access_token(self, watch) -> str:
        from ..infrastructure.drive_connector import refresh_access_token

        config = watch.config or {}
        connection_id = config.get("connection_id") or ""
        conn = await self._store.get_drive_connection(
            connection_id, watch.customer_id,
        )
        if conn is None:
            raise ValueError(
                f"Drive connection {connection_id!r} not found for watch",
            )
        return await refresh_access_token(
            self._store, conn.customer_id,
            conn.encrypted_refresh_token, conn.token_nonce,
        )

    async def check(self, watch, token: Optional[str]) -> Optional[ChangeSet]:
        """One poll: full folder listing vs last_state. Full (not
        modified_after-windowed) because removal detection requires
        seeing what is NOT there anymore — deletions delete (M design
        statement), Drive included."""
        from ..infrastructure.drive_connector import list_folder_files

        config = watch.config or {}
        folder_id = config.get("folder_id") or ""
        access = await self._access_token(watch)
        files = await list_folder_files(
            access, folder_id, supported_only=False,
        )

        new_state: dict[str, str] = {}
        names: dict[str, str] = {}
        for f in files:
            if not _ingestible(f.get("name") or "", f.get("mimeType") or ""):
                continue
            fid = f["id"]
            new_state[fid] = f.get("modifiedTime") or ""
            names[fid] = f.get("name") or fid

        old_state: dict[str, str] = dict(
            (watch.last_state or {}).get("files") or {},
        )

        changed = [
            fid for fid, mtime in new_state.items()
            if old_state.get(fid) != mtime
        ]
        removed = [fid for fid in old_state if fid not in new_state]

        if not changed and not removed:
            return None
        return ChangeSet(
            new_state={"files": new_state, "names": names},
            changed=changed,
            removed=removed,
        )

    async def fetch(
        self, watch, path: str, token: Optional[str],
    ) -> SourceEnvelope:
        """path = the Drive file id. Metadata -> download/export ->
        envelope. Native files export as office formats; everything
        else downloads raw and dispatches by name + declared MIME."""
        from ..infrastructure.drive_connector import (
            download_file_bytes,
            get_file_metadata,
        )

        config = watch.config or {}
        access = await self._access_token(watch)
        meta = await get_file_metadata(access, path)
        name = (meta.get("name") or path) if meta else path
        mime = (meta.get("mimeType") or "") if meta else ""

        payload, effective_mime, effective_name = await download_file_bytes(
            access, path, mime, name,
        )
        return SourceEnvelope(
            payload_bytes=payload,
            mime_type=effective_mime,
            source_uri=f"gdrive://{config.get('folder_id')}/{path}",
            label=f"{watch.source_name}/{effective_name}",
            source_modified_at=None,
            connection_id=config.get("connection_id"),
            extra={"drive_file_id": path},
        )
