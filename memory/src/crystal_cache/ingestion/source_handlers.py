"""Source handlers — the C6 envelope as running code (Gate M slice 2).

Ratified 2026-07-16 (C6) and 2026-07-18 (M-Q1): every watched source —
git now; watch-folder, unified Drive, and whatever comes after —
normalizes into ONE envelope shape, and the sync loop dispatches on
scheme through this registry. A new source is one handler class and
one register call: no table churn, no loop churn, no envelope churn
(scheme-specific metadata rides `extra`).

The handler contract is deliberately two methods:

  check(watch, token)  -> ChangeSet | None
      One poll. Compare the remote's current state against
      watch.last_state; None means unchanged (the cheap path — for
      git, a single head-SHA comparison). A ChangeSet names exactly
      what moved and carries the new state to persist AFTER the
      changes are ingested (crash between = re-poll re-finds them;
      idempotent by replace semantics).

  fetch(watch, path, token) -> SourceEnvelope
      Retrieve one changed item, normalized. Everything below the
      envelope is the existing ingestion spine — identity (C1/D6),
      comprehension (D2), review or auto per M-Q3.

Credentials (M-Q5): per-watch tokens are enc:v2 under the tenant DEK
(the Drive-credential pattern); `resolve_watch_token` decrypts when
present and returns None otherwise — scheme-level fallbacks (e.g.
git's CC_GITHUB_TOKEN env) are handler policy, not registry policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Protocol, runtime_checkable

_TOKEN_FAMILY = "source_watch_token"


@dataclass
class SourceEnvelope:
    """C6 verbatim: the one shape every source normalizes into."""
    payload_bytes: bytes
    mime_type: str
    source_uri: str
    label: Optional[str] = None
    source_modified_at: Optional[datetime] = None
    connection_id: Optional[str] = None
    extra: dict = field(default_factory=dict)


@dataclass
class ChangeSet:
    """What one poll found. Paths are scheme-shaped (repo-relative for
    git). `removed` feeds deletion — the repo is the source of truth
    (M design statement: deletions delete, via the D2 cascade)."""
    new_state: dict
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.changed and not self.removed


@runtime_checkable
class SourceHandler(Protocol):
    scheme: str

    async def check(
        self, watch, token: Optional[str],
    ) -> Optional[ChangeSet]: ...

    async def fetch(
        self, watch, path: str, token: Optional[str],
    ) -> SourceEnvelope: ...


_HANDLERS: dict[str, SourceHandler] = {}


def register_handler(handler: SourceHandler) -> None:
    """Idempotent by scheme — re-registration replaces (supports
    reloads and test doubles)."""
    _HANDLERS[handler.scheme] = handler


def get_handler(scheme: str) -> Optional[SourceHandler]:
    return _HANDLERS.get(scheme)


def registered_schemes() -> list[str]:
    return sorted(_HANDLERS)


async def encrypt_watch_token(store, customer_id: str, plaintext: str) -> str:
    """enc:v2 under the tenant DEK — the value that lands in
    source_watches.encrypted_token."""
    return await store.encrypt_tenant_secret(
        customer_id, _TOKEN_FAMILY, plaintext,
    )


async def resolve_watch_token(store, watch) -> Optional[str]:
    """The per-watch credential, or None. Never raises into the sync
    loop: a corrupt/foreign token logs as a watch error upstream via
    the ValueError; callers treat token failure as this watch failing
    this cycle, not the loop dying."""
    enc = getattr(watch, "encrypted_token", None)
    if not enc:
        return None
    return await store.decrypt_tenant_secret(
        watch.customer_id, _TOKEN_FAMILY, enc,
    )
