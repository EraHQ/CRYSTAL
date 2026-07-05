"""Google Drive connector — OAuth flow + folder monitoring.

Handles:
  - OAuth 2.0 authorization code exchange
  - Token refresh
  - File listing within watched folders
  - File content reading (Google Docs exported as text, PDFs/DOCX downloaded)
  - PHI access audit logging

The OAuth flow:
  1. Frontend redirects user to Google's consent URL (constructed client-side)
  2. Google redirects back with an authorization code
  3. Backend exchanges code for access + refresh tokens
  4. Refresh token is AES-256-GCM encrypted and stored in drive_connections
  5. Access tokens are obtained on-demand from refresh tokens (never stored)

Environment variables:
  CC_GOOGLE_CLIENT_ID     — OAuth 2.0 client ID
  CC_GOOGLE_CLIENT_SECRET — OAuth 2.0 client secret
  CC_TOKEN_ENCRYPTION_KEY — 32-byte hex key for token encryption
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from .token_crypto import encrypt_token, decrypt_token

logger = logging.getLogger(__name__)

GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"
GOOGLE_DRIVE_FILES_URL = "https://www.googleapis.com/drive/v3/files"
GOOGLE_DRIVE_EXPORT_URL = "https://www.googleapis.com/drive/v3/files/{file_id}/export"
GOOGLE_DRIVE_DOWNLOAD_URL = "https://www.googleapis.com/drive/v3/files/{file_id}"

SCOPES = "https://www.googleapis.com/auth/drive.readonly"

# MIME types we can extract text from
SUPPORTED_MIME_TYPES = {
    "application/vnd.google-apps.document",      # Google Docs
    "application/vnd.google-apps.spreadsheet",    # Google Sheets
    "application/vnd.google-apps.presentation",   # Google Slides
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # DOCX
    "text/plain",
    "text/markdown",
    "text/csv",
}

# Google Workspace native types that need export (not download)
GOOGLE_NATIVE_TYPES = {
    "application/vnd.google-apps.document",
    "application/vnd.google-apps.spreadsheet",
    "application/vnd.google-apps.presentation",
}


def get_client_id() -> str:
    from ..config import settings
    return settings.google_client_id or ""


def get_client_secret() -> str:
    from ..config import settings
    return settings.google_client_secret or ""


def build_auth_url(redirect_uri: str, state: str = "") -> str:
    """Build the Google OAuth consent URL for the frontend to redirect to."""
    client_id = get_client_id()
    if not client_id:
        raise ValueError("CC_GOOGLE_CLIENT_ID not set")

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": SCOPES,
        "access_type": "offline",  # Gets us a refresh token
        "prompt": "consent",       # Always show consent to get refresh token
        "include_granted_scopes": "true",
    }
    if state:
        params["state"] = state

    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"https://accounts.google.com/o/oauth2/v2/auth?{query}"


async def exchange_code(code: str, redirect_uri: str) -> dict[str, Any]:
    """Exchange authorization code for tokens.

    Returns dict with: access_token, refresh_token, expires_in, scope, token_type
    """
    client_id = get_client_id()
    client_secret = get_client_secret()
    if not client_id or not client_secret:
        raise ValueError("CC_GOOGLE_CLIENT_ID and CC_GOOGLE_CLIENT_SECRET must be set")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        resp.raise_for_status()
        return resp.json()


async def refresh_access_token(encrypted_refresh: str, nonce: str) -> str:
    """Get a fresh access token from an encrypted refresh token.

    Returns the access_token string. Raises on failure.
    """
    refresh_token = decrypt_token(encrypted_refresh, nonce)
    client_id = get_client_id()
    client_secret = get_client_secret()

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "refresh_token": refresh_token,
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "refresh_token",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["access_token"]


async def get_user_email(access_token: str) -> Optional[str]:
    """Get the email of the authenticated Google user."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if resp.status_code == 200:
            return resp.json().get("email")
    return None


async def list_folder_files(
    access_token: str,
    folder_id: str,
    modified_after: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    """List files in a Drive folder.

    Returns list of dicts with: id, name, mimeType, modifiedTime, size
    """
    query_parts = [f"'{folder_id}' in parents", "trashed = false"]

    if modified_after:
        ts = modified_after.strftime("%Y-%m-%dT%H:%M:%S")
        query_parts.append(f"modifiedTime > '{ts}'")

    query = " and ".join(query_parts)

    all_files: list[dict[str, Any]] = []
    page_token: Optional[str] = None

    async with httpx.AsyncClient() as client:
        while True:
            params: dict[str, Any] = {
                "q": query,
                "fields": "nextPageToken, files(id, name, mimeType, modifiedTime, size)",
                "pageSize": 100,
                "orderBy": "modifiedTime desc",
            }
            if page_token:
                params["pageToken"] = page_token

            resp = await client.get(
                GOOGLE_DRIVE_FILES_URL,
                headers={"Authorization": f"Bearer {access_token}"},
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()

            for f in data.get("files", []):
                if f.get("mimeType") in SUPPORTED_MIME_TYPES:
                    all_files.append(f)

            page_token = data.get("nextPageToken")
            if not page_token:
                break

    return all_files


async def list_folders(access_token: str, parent_id: str = "root") -> list[dict[str, Any]]:
    """List folders in a Drive folder (for folder picker UI)."""
    query = f"'{parent_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            GOOGLE_DRIVE_FILES_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            params={
                "q": query,
                "fields": "files(id, name, mimeType)",
                "pageSize": 100,
                "orderBy": "name",
            },
        )
        resp.raise_for_status()
        return resp.json().get("files", [])


async def read_file_text(access_token: str, file_id: str, mime_type: str) -> str:
    """Read a file's text content from Drive.

    Google Workspace files (Docs, Sheets) are exported as plain text.
    Other files (PDF, DOCX) are downloaded as bytes and need local extraction.
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        if mime_type in GOOGLE_NATIVE_TYPES:
            # Export Google Workspace files as plain text
            url = GOOGLE_DRIVE_EXPORT_URL.format(file_id=file_id)
            resp = await client.get(
                url,
                headers={"Authorization": f"Bearer {access_token}"},
                params={"mimeType": "text/plain"},
            )
            resp.raise_for_status()
            return resp.text

        elif mime_type == "text/plain" or mime_type == "text/markdown" or mime_type == "text/csv":
            # Download text files directly
            url = GOOGLE_DRIVE_DOWNLOAD_URL.format(file_id=file_id)
            resp = await client.get(
                url,
                headers={"Authorization": f"Bearer {access_token}"},
                params={"alt": "media"},
            )
            resp.raise_for_status()
            return resp.text

        elif mime_type == "application/pdf":
            # Download PDF bytes and extract text
            url = GOOGLE_DRIVE_DOWNLOAD_URL.format(file_id=file_id)
            resp = await client.get(
                url,
                headers={"Authorization": f"Bearer {access_token}"},
                params={"alt": "media"},
            )
            resp.raise_for_status()
            from ..ingestion.file_extract import extract_text_from_file
            return extract_text_from_file(resp.content, "document.pdf")

        elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # Download DOCX bytes and extract text
            url = GOOGLE_DRIVE_DOWNLOAD_URL.format(file_id=file_id)
            resp = await client.get(
                url,
                headers={"Authorization": f"Bearer {access_token}"},
                params={"alt": "media"},
            )
            resp.raise_for_status()
            from ..ingestion.file_extract import extract_text_from_file
            return extract_text_from_file(resp.content, "document.docx")

        else:
            raise ValueError(f"Unsupported MIME type: {mime_type}")


def store_connection(
    refresh_token: str,
    customer_id: str,
    email: Optional[str] = None,
    scopes: str = SCOPES,
) -> tuple[str, str, str, str]:
    """Encrypt a refresh token and return fields for DB storage.

    Returns: (connection_id, encrypted_token, nonce, scopes)
    """
    connection_id = f"drv_{uuid.uuid4().hex[:16]}"
    encrypted, nonce = encrypt_token(refresh_token)
    return connection_id, encrypted, nonce, scopes


async def get_file_metadata(
    access_token: str,
    file_id: str,
) -> dict[str, Any]:
    """Get metadata for a single Drive file (name, mimeType, modifiedTime)."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{GOOGLE_DRIVE_FILES_URL}/{file_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"fields": "id, name, mimeType, modifiedTime, size"},
        )
        resp.raise_for_status()
        return resp.json()
