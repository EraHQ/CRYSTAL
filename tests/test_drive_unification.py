"""DRIVE-Q1=B (2026-07-24): Google Drive unified into the source-watch
framework. The legacy drive_sync worker + watched_folders/watched_files
tables retired; a watched Drive folder is a normal source_watch
(scheme=gdrive) synced by DriveSourceHandler under the one sync loop.

These pin the new machinery: the handler's full-listing diff (three-
poll semantics), the _ingestible filter, store-gated registration, the
keyless admin routes' tenancy posture (404-not-an-oracle), disconnect
removing the watches riding the connection, and the native-export vs
raw-download split in the connector primitive.
"""
from __future__ import annotations

import json

import pytest
from fastapi import HTTPException


# --- _ingestible: what the ingestion spine can eat -------------------------

def test_ingestible_table():
    from crystal_cache.ingestion.drive_handler import _ingestible

    # Google-native types ingest (they export as office formats).
    assert _ingestible("Plan", "application/vnd.google-apps.document")
    assert _ingestible("Numbers", "application/vnd.google-apps.spreadsheet")
    assert _ingestible("Deck", "application/vnd.google-apps.presentation")
    # Folders never ingest.
    assert not _ingestible("Sub", "application/vnd.google-apps.folder")
    # Known extensions / text mimes ingest even with a blank mime.
    assert _ingestible("notes.md", "")
    assert _ingestible("report.pdf", "application/pdf")
    assert _ingestible("data.csv", "text/csv; charset=utf-8")
    assert _ingestible("readme", "text/plain")
    # Unknown binaries skip — a folder of PSDs must not become error rows.
    assert not _ingestible("art.psd", "image/vnd.adobe.photoshop")
    assert not _ingestible("movie.mp4", "video/mp4")


# --- registration: gdrive is store-gated -----------------------------------

def test_register_builtin_handlers_gates_gdrive_on_store(monkeypatch):
    from crystal_cache.ingestion import source_handlers as sh
    from crystal_cache.workers.source_sync import register_builtin_handlers

    monkeypatch.setattr(sh, "_HANDLERS", {})
    register_builtin_handlers()  # no store -> git only
    assert "git" in sh.registered_schemes()
    assert "gdrive" not in sh.registered_schemes()

    sentinel = object()
    register_builtin_handlers(store=sentinel)  # store -> gdrive joins
    assert "gdrive" in sh.registered_schemes()
    assert sh.get_handler("gdrive")._store is sentinel


# --- the handler: full-listing diff (three-poll semantics) -----------------

class _Conn:
    customer_id = "cus_x"
    status = "active"
    encrypted_refresh_token = "enc:v2:fake"
    token_nonce = "v2"


class _FakeStore:
    def __init__(self, conn):
        self._conn = conn

    async def get_drive_connection(self, connection_id, customer_id):
        return self._conn


class _W:
    """gdrive watch shape (config carries the connection + folder)."""
    def __init__(self, config, last_state=None, source_name="drive-ops"):
        self.config = config
        self.last_state = last_state
        self.source_name = source_name
        self.customer_id = "cus_x"
        self.encrypted_token = None


@pytest.mark.asyncio
async def test_drive_three_poll_diff(monkeypatch):
    from crystal_cache.infrastructure import drive_connector as dc
    from crystal_cache.ingestion.drive_handler import DriveSourceHandler

    listing = [
        {"id": "f1", "name": "notes.md", "mimeType": "text/markdown",
         "modifiedTime": "t1"},
        {"id": "f2", "name": "plan",
         "mimeType": "application/vnd.google-apps.document",
         "modifiedTime": "t1"},
        {"id": "skip", "name": "art.psd",
         "mimeType": "image/vnd.adobe.photoshop", "modifiedTime": "t1"},
    ]

    async def fake_refresh(store, cid, tok, nonce):
        return "access"

    async def fake_list(access, folder_id, supported_only=True):
        # Removal detection requires the FULL listing (deletions delete).
        assert supported_only is False
        assert folder_id == "fold_1"
        return list(listing)

    monkeypatch.setattr(dc, "refresh_access_token", fake_refresh)
    monkeypatch.setattr(dc, "list_folder_files", fake_list)

    h = DriveSourceHandler(_FakeStore(_Conn()))
    w = _W({"connection_id": "drv_1", "folder_id": "fold_1"})

    # Poll 1 — first sync: every ingestible file changed; PSD ignored.
    cs = await h.check(w, None)
    assert sorted(cs.changed) == ["f1", "f2"]
    assert cs.removed == []
    assert set(cs.new_state["files"]) == {"f1", "f2"}
    assert cs.new_state["names"]["f2"] == "plan"

    # Poll 2 — unchanged listing vs stored state: no work at all.
    w.last_state = cs.new_state
    assert await h.check(w, None) is None

    # Poll 3 — f1 modified, f2 gone, f3 new.
    listing[:] = [
        {"id": "f1", "name": "notes.md", "mimeType": "text/markdown",
         "modifiedTime": "t2"},
        {"id": "f3", "name": "new.txt", "mimeType": "text/plain",
         "modifiedTime": "t1"},
    ]
    cs3 = await h.check(w, None)
    assert sorted(cs3.changed) == ["f1", "f3"]
    assert cs3.removed == ["f2"]


@pytest.mark.asyncio
async def test_drive_fetch_builds_envelope(monkeypatch):
    from crystal_cache.infrastructure import drive_connector as dc
    from crystal_cache.ingestion.drive_handler import DriveSourceHandler

    async def fake_refresh(store, cid, tok, nonce):
        return "access"

    async def fake_meta(access, fid):
        return {"name": "plan",
                "mimeType": "application/vnd.google-apps.document"}

    async def fake_download(access, fid, mime, name):
        assert mime == "application/vnd.google-apps.document"
        return (
            b"PK docx bytes",
            "application/vnd.openxmlformats-officedocument"
            ".wordprocessingml.document",
            "plan.docx",
        )

    monkeypatch.setattr(dc, "refresh_access_token", fake_refresh)
    monkeypatch.setattr(dc, "get_file_metadata", fake_meta)
    monkeypatch.setattr(dc, "download_file_bytes", fake_download)

    h = DriveSourceHandler(_FakeStore(_Conn()))
    w = _W({"connection_id": "drv_1", "folder_id": "fold_1"})
    env = await h.fetch(w, "f2", None)
    assert env.payload_bytes == b"PK docx bytes"
    assert env.source_uri == "gdrive://fold_1/f2"
    assert env.label == "drive-ops/plan.docx"
    assert env.connection_id == "drv_1"
    assert env.extra == {"drive_file_id": "f2"}


# --- the connector primitive: native export vs raw download ----------------

class _FakeResp:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        pass


class _FakeClient:
    calls: list = []

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, headers=None, params=None):
        _FakeClient.calls.append((url, dict(params or {})))
        return _FakeResp(b"bytes")


@pytest.mark.asyncio
async def test_download_native_exports_raw_passes_through(monkeypatch):
    from crystal_cache.infrastructure import drive_connector as dc

    monkeypatch.setattr(dc.httpx, "AsyncClient", _FakeClient)
    _FakeClient.calls = []

    # Google-native -> export URL, office mime, extension appended.
    payload, mime, name = await dc.download_file_bytes(
        "tok", "f1", "application/vnd.google-apps.document", "plan",
    )
    assert payload == b"bytes"
    assert name == "plan.docx"
    assert mime.endswith("wordprocessingml.document")
    url, params = _FakeClient.calls[-1]
    assert "/export" in url
    assert params["mimeType"] == mime

    # Already-suffixed native name is not double-suffixed.
    _, _, name2 = await dc.download_file_bytes(
        "tok", "f1b", "application/vnd.google-apps.spreadsheet",
        "numbers.xlsx",
    )
    assert name2 == "numbers.xlsx"

    # Regular file -> raw download (alt=media), mime passes through.
    _, mime3, name3 = await dc.download_file_bytes(
        "tok", "f2", "text/plain", "a.txt",
    )
    assert (mime3, name3) == ("text/plain", "a.txt")
    url, params = _FakeClient.calls[-1]
    assert "/export" not in url
    assert params == {"alt": "media"}

    # Blank mime -> octet-stream default.
    _, mime4, _ = await dc.download_file_bytes("tok", "f3", "", "blob.bin")
    assert mime4 == "application/octet-stream"


# --- keyless admin routes: tenancy posture ---------------------------------

def test_gdrive_admin_routes_are_tenant_pathed():
    """No allowlist edits were needed: the three console routes live
    under /admin/api/customers/{cid}/* and the tenant-path rule admits
    own-id, 404s foreign — pin that they actually match it."""
    from crystal_cache.ingress.auth import _TENANT_PATH_RE

    for p in (
        "/admin/api/customers/cus_a/gdrive/auth-url",
        "/admin/api/customers/cus_a/gdrive/connections",
        "/admin/api/customers/cus_a/gdrive/connections/drv_1",
    ):
        m = _TENANT_PATH_RE.match(p)
        assert m is not None and m.group(1) == "cus_a"


@pytest.mark.asyncio
async def test_gdrive_auth_url_unknown_customer_is_404(store):
    from crystal_cache.endpoints.drive import admin_gdrive_auth_url

    with pytest.raises(HTTPException) as exc:
        await admin_gdrive_auth_url(object(), "cus_missing", store)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_gdrive_disconnect_foreign_customer_is_404(store, customer):
    from crystal_cache.endpoints.drive import admin_gdrive_disconnect

    await store.create_drive_connection(
        customer.id, connection_id="drv_t1", email="a@b.c",
        encrypted_refresh_token="enc:v2:fake", token_nonce="v2",
        scopes=None,
    )
    with pytest.raises(HTTPException) as exc:
        await admin_gdrive_disconnect("cus_other", "drv_t1", store)
    assert exc.value.status_code == 404
    # The store-level delete is equally tenancy-checked: a foreign
    # attempt is a no-op, the connection survives.
    await store.delete_drive_connection("drv_t1", "cus_other")
    assert await store.get_drive_connection("drv_t1", customer.id) is not None


@pytest.mark.asyncio
async def test_gdrive_disconnect_removes_riding_watches(store, customer):
    """Disconnect removes the connection AND the gdrive watches riding
    it — through the standard watch delete, one deletion path per
    object. Unrelated watches (other schemes, other connections) stay."""
    from crystal_cache.endpoints.drive import admin_gdrive_disconnect

    await store.create_drive_connection(
        customer.id, connection_id="drv_t2", email="a@b.c",
        encrypted_refresh_token="enc:v2:fake", token_nonce="v2",
        scopes=None,
    )
    riding = await store.create_source_watch(
        customer.id, scheme="gdrive", source_name="drive-ops",
        config={"connection_id": "drv_t2", "folder_id": "fold_1",
                "folder_name": "Ops"},
        cadence_minutes=15,
    )
    unrelated = await store.create_source_watch(
        customer.id, scheme="git", source_name="somerepo", config={},
    )

    resp = await admin_gdrive_disconnect(customer.id, "drv_t2", store)
    body = json.loads(resp.body)
    assert body["deleted"] is True
    assert body["removed_watches"] == 1
    assert await store.get_source_watch(riding.id, customer.id) is None
    assert await store.get_source_watch(unrelated.id, customer.id) is not None
    assert await store.get_drive_connection("drv_t2", customer.id) is None
