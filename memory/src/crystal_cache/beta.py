"""The Beta list (2026-10-24): private invites to features.

Anthony: "Make a beta list for private invites to features like this.
Everything but keep Gmail private." A feature is open (everyone),
invite (only the invited) or closed (no one, not even the invited). An
invite names a tenant or an email; an email invite binds to the tenant
when that person signs in. Platform admins manage the list from the
console (Admin > Beta) or the admin API.
"""
from __future__ import annotations

from typing import Any, Iterable

DEFAULT_FEATURES: tuple[dict[str, str], ...] = (
    {"key": "connections", "title": "Connected apps",
     "description": "Connect Google Drive, Google Calendar, Notion and Slack.", "status": "open"},
    {"key": "goals", "title": "Goals",
     "description": "Standing directives Crystal plans and works on.", "status": "open"},
    {"key": "gmail", "title": "Gmail",
     "description": "Watch, remember and reply to email. Private beta: Google's review of mail access is pending.",
     "status": "invite"},
)
STATUSES = ("open", "invite", "closed")

_ensured = False


async def ensure_defaults(store: Any) -> None:
    """Insert the default features once per process (never overwrites a
    platform admin's status choice)."""
    global _ensured
    if _ensured:
        return
    await store.ensure_beta_features(DEFAULT_FEATURES)
    _ensured = True


def reset_for_tests() -> None:
    global _ensured
    _ensured = False


async def features_for(store: Any, customer_id: str, emails: Iterable[str] = ()) -> dict[str, bool]:
    """{feature key: may this tenant use it}."""
    await ensure_defaults(store)
    # A default missing from the table (a fresh database in a process that
    # already ensured once) reads as its default, never as "no features".
    status = {f["key"]: f["status"] for f in DEFAULT_FEATURES}
    status.update({f["key"]: f["status"] for f in await store.list_beta_features()})
    invited = await store.beta_keys_for(customer_id, emails)
    out: dict[str, bool] = {}
    for key, st in status.items():
        out[key] = True if st == "open" else (key in invited if st == "invite" else False)
    return out


async def has_feature(store: Any, customer_id: str, key: str, emails: Iterable[str] = ()) -> bool:
    return (await features_for(store, customer_id, emails)).get(key, False)


async def tenant_emails(store: Any, customer_id: str) -> list[str]:
    """Every email of the tenant's people (an email invite reaches them)."""
    return await store.list_user_emails_for_customer(customer_id)
