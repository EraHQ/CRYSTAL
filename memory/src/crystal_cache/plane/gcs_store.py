"""GCS-backed TaskStore (Phase 3 §9 item 5, ratified G4/G5).

Per-task layout under one plane-owned bucket:

    gs://{bucket}/tasks/{tenant_id}/{task_id}/seed.tar.gz
    gs://{bucket}/tasks/{tenant_id}/{task_id}/harvest.tar.gz

The refs handed to the job are per-task V4 SIGNED URLS — a read-only GET
for the seed and a write-only PUT for the harvest — so the box carries
capabilities, never credentials (G5: isolation by absence; the runner
service account has no roles and never touches GCS as itself).

Signing on Cloud Run: the plane's service account has no private key
file, so V4 signing goes through the IAM signBlob path — the storage
library accepts (service_account_email, access_token) and we fetch both
from Application Default Credentials. Requires the plane SA to hold
roles/iam.serviceAccountTokenCreator ON ITSELF (documented in the infra
runbook). Local development with a key file signs directly; the same
code path handles both because the kwargs are only added when no key is
present.

Blocking google-cloud-storage calls are wrapped in asyncio.to_thread —
the store's callers are async (run_remote_task's monitor loop).
"""
from __future__ import annotations

import asyncio
import datetime as _dt
from typing import Optional

import structlog

logger = structlog.get_logger()

_SEED = "seed.tar.gz"
_HARVEST = "harvest.tar.gz"


class GcsTaskStore:
    """TaskStore protocol implementation over one GCS bucket."""

    def __init__(
        self,
        bucket_name: str,
        *,
        tenant_id: str,
        prefix: str = "tasks",
        seed_url_ttl_seconds: int = 3600,
        harvest_url_ttl_seconds: int = 86_400,
        client=None,
    ) -> None:
        self._bucket_name = bucket_name
        self._tenant = tenant_id
        self._prefix = prefix.strip("/")
        self._seed_ttl = int(seed_url_ttl_seconds)
        self._harvest_ttl = int(harvest_url_ttl_seconds)
        self._client = client  # injectable for tests; lazy real client

    # -- plumbing ---------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            from google.cloud import storage

            self._client = storage.Client()
        return self._client

    def _blob(self, task_id: str, leaf: str):
        bucket = self._get_client().bucket(self._bucket_name)
        return bucket.blob(f"{self._prefix}/{self._tenant}/{task_id}/{leaf}")

    def _signing_kwargs(self) -> dict:
        """Key-file creds sign natively; keyless (Cloud Run) creds sign via
        IAM signBlob, which needs the SA email + a live access token."""
        creds = getattr(self._get_client(), "_credentials", None)
        if creds is None or getattr(creds, "sign_bytes", None) is not None:
            return {}
        import google.auth
        import google.auth.transport.requests

        if not getattr(creds, "valid", False):
            creds.refresh(google.auth.transport.requests.Request())
        return {
            "service_account_email": getattr(creds, "service_account_email", None),
            "access_token": creds.token,
        }

    def _sign(self, blob, *, method: str, ttl: int, content_type: Optional[str] = None) -> str:
        kwargs = dict(
            version="v4",
            expiration=_dt.timedelta(seconds=ttl),
            method=method,
            **self._signing_kwargs(),
        )
        if content_type:
            kwargs["content_type"] = content_type
        return blob.generate_signed_url(**kwargs)

    # -- TaskStore protocol -------------------------------------------------

    async def put_seed(self, task_id: str, data: bytes) -> str:
        """Upload the seed and return a READ-ONLY signed GET url."""
        def _put() -> str:
            blob = self._blob(task_id, _SEED)
            blob.upload_from_string(data, content_type="application/gzip")
            return self._sign(blob, method="GET", ttl=self._seed_ttl)

        url = await asyncio.to_thread(_put)
        logger.info("gcs_task_store.seed_put", task_id=task_id, size=len(data))
        return url

    async def harvest_ref(self, task_id: str) -> str:
        """A WRITE-ONLY signed PUT url the job uploads its harvest to.
        Content-type is pinned into the signature — the runner entrypoint
        sends application/gzip, anything else is rejected by GCS itself."""
        def _sign_put() -> str:
            blob = self._blob(task_id, _HARVEST)
            return self._sign(
                blob, method="PUT", ttl=self._harvest_ttl,
                content_type="application/gzip",
            )

        return await asyncio.to_thread(_sign_put)

    async def get_harvest(self, task_id: str) -> Optional[bytes]:
        def _get() -> Optional[bytes]:
            blob = self._blob(task_id, _HARVEST)
            if not blob.exists():
                return None
            return blob.download_as_bytes()

        return await asyncio.to_thread(_get)

    async def delete_task_prefix(self, task_id: str) -> None:
        """Teardown: everything under the task's prefix goes. Idempotent —
        missing blobs are fine; a bucket lifecycle rule (age > 7d) backstops
        anything a crashed plane leaves behind (ratified G4)."""
        def _delete() -> int:
            client = self._get_client()
            prefix = f"{self._prefix}/{self._tenant}/{task_id}/"
            blobs = list(client.list_blobs(self._bucket_name, prefix=prefix))
            for b in blobs:
                try:
                    b.delete()
                except Exception:  # noqa: BLE001 — teardown never raises
                    pass
            return len(blobs)

        n = await asyncio.to_thread(_delete)
        logger.info("gcs_task_store.prefix_deleted", task_id=task_id, blobs=n)
