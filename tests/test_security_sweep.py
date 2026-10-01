"""Security sweep pins (docs/AUDIT_SECURITY.md, 2026-09-30, Q8=A).

B1: a BYOK key with a stray byte is refused at the door and stripped at
    the seams, so httpx can never build a malformed header from it.
B2: platform-admin bootstrap needs a verified email from a federated
    provider; a password account on the allowlisted address gets nothing.
B4: the on-demand conflicts scan obeys the tenant pin and clamps its
    spend knobs.
S1: every secret-bearing Settings field is stripped at load and boot
    refuses one with whitespace inside.
Hygiene: redact() scrubs every key shape we handle.
"""
import pytest
from fastapi import HTTPException

from crystal_cache.config import Settings
from crystal_cache.hygiene import SecretFormatError, clean_secret, redact, safe_error


# ----------------------------------------------------------------------------
# hygiene
# ----------------------------------------------------------------------------

def test_clean_secret_strips_edges_and_refuses_interior_whitespace():
    assert clean_secret("  sk-ant-abc\n") == "sk-ant-abc"
    assert clean_secret("sk-ant-abc\r\n") == "sk-ant-abc"
    assert clean_secret(None) == ""
    assert clean_secret("") == ""
    for bad in ("sk-ant a", "sk-ant\tb", "sk-ant\x00c", "sk\x7fd"):
        with pytest.raises(SecretFormatError):
            clean_secret(bad)


def test_redact_scrubs_every_key_shape():
    text = (
        "sk-ant-api03-ABCdef_123 sk-proj-zzz sk_live_51X whsec_9 gsk_1 tvly-2 "
        "ghp_3 github_pat_4 ya29.a0Af cc_sk_5 AIzaSyCg8JtF1Doy2uufxaY3yXn-TvvpZzVT4rI "
        "Authorization: Bearer eyJhbGci x-api-key=abc123 api_key='def456'"
    )
    out = redact(text)
    for secret in (
        "ABCdef_123", "sk-proj-zzz", "sk_live_51X", "whsec_9", "gsk_1", "tvly-2",
        "ghp_3", "github_pat_4", "ya29.a0Af", "cc_sk_5", "AIzaSyCg8JtF1Doy2uufxaY3yXn-TvvpZzVT4rI",
        "eyJhbGci", "abc123", "def456",
    ):
        assert secret not in out, secret
    assert "[redacted]" in out


def test_safe_error_never_returns_exception_text():
    ref, msg = safe_error("test.event", RuntimeError("header 'Bearer sk_live_LEAK'"))
    assert len(ref) == 10
    assert "sk_live_LEAK" not in msg
    assert "Bearer" not in msg
    assert ref in msg
    assert "contact support" in msg


def test_redact_scrubs_url_passwords_but_keeps_the_host():
    out = redact("postgresql+asyncpg://crystal:s3cr3t@10.0.0.1:5432/db")
    assert "s3cr3t" not in out
    assert "crystal:[redacted]@10.0.0.1:5432/db" in out


def test_log_processor_redacts_every_string_in_the_event():
    from crystal_cache.hygiene import redact_event_dict

    event = redact_event_dict(None, "error", {
        "event": "x",
        "error": "InvalidHeader: 'Bearer sk_live_ABC'",
        "nested": {"url": "https://u:pw@h/x", "n": 1},
        "items": ["whsec_zz", 2],
        "exception": "Traceback ...\nValueError: key sk-ant-api03-LEAK\n",
    })
    flat = str(event)
    for secret in ("sk_live_ABC", "u:pw@", "whsec_zz", "sk-ant-api03-LEAK"):
        assert secret not in flat, secret
    assert event["nested"]["n"] == 1
    assert event["items"][1] == 2


def test_structlog_is_configured_with_the_redaction_processor():
    import structlog

    import crystal_cache  # noqa: F401  (configures on import)
    from crystal_cache.hygiene import redact_event_dict

    assert structlog.is_configured()
    assert redact_event_dict in structlog.get_config()["processors"]


def test_no_route_echoes_exception_text_into_a_response():
    """S2 to S5, S7, pinned as an invariant over the source tree: no
    HTTPException detail, tool-result error/reason field, or persisted
    error message is built from exception text. A line that genuinely
    must (a domain error whose message is ours) carries a `detail-ok:`
    comment on one of the three lines above it."""
    import re
    from pathlib import Path

    import crystal_cache

    root = Path(crystal_cache.__file__).parent
    # Exception-text shapes: str(e)/repr(e), or {e}/{exc}/{err} inside an
    # f-string. `{error}` and `{env_id}` are ordinary variables, not these.
    exc = r"(?:str\(e\w*\)|repr\(e\w*\)|f[\"'][^\"']*\{(?:e|exc|err)\}|f[\"'][^\"']*\{str\((?:e|exc|err)\)\})"
    pats = (
        re.compile(r"detail\s*=\s*" + exc),
        re.compile(r"[\"'](?:error|reason|message|error_message)[\"']\s*:\s*" + exc),
        re.compile(r"\[[\"'](?:error|reason|message|error_message)[\"']\]\s*=\s*" + exc),
        re.compile(r"mark_document_error\([^)]*" + exc),
    )
    offenders = []
    for path in root.rglob("*.py"):
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            window = lines[max(0, i - 3): i + 1]
            if not any(p.search(line) for p in pats):
                continue
            if any("detail-ok" in w for w in window):
                continue
            # stdlib `logger.x(..., extra={...})` dicts are log fields,
            # not client output; the structlog chain covers the rest.
            if any("extra=" in w or "logger." in w for w in window):
                continue
            offenders.append(f"{path.relative_to(root)}:{i + 1}")
    assert offenders == [], offenders


# ----------------------------------------------------------------------------
# S1: Settings
# ----------------------------------------------------------------------------

_ENV_SECRET_VARS = (
    "ANTHROPIC_API_KEY", "CC_ANTHROPIC_API_KEY", "CC_LLM_API_KEY",
    "GITHUB_TOKEN", "CC_SOURCE_GITHUB_TOKEN", "CC_ADMIN_API_KEY",
    "CC_API_KEY_PEPPER", "CC_STRIPE_SECRET_KEY",
)


def _isolated_settings(monkeypatch, **values) -> Settings:
    """Build Settings from `values` only: no .env file, no ambient env.
    Without this the test reads the developer's real keys, and a failing
    assertion would print one (it did, once)."""
    for var in _ENV_SECRET_VARS:
        monkeypatch.delenv(var, raising=False)
    return Settings(_env_file=None, **values)


def test_settings_strip_every_secret_field_at_load(monkeypatch):
    s = _isolated_settings(
        monkeypatch,
        CC_ANTHROPIC_API_KEY="sk-ant-x\n",       # aliased field: kwarg by alias
        llm_api_key=" sk-y ",
        CC_SOURCE_GITHUB_TOKEN="ghp_z\r\n",      # aliased field
        admin_api_key="adm\n",
        api_key_pepper="pep\n",
        stripe_secret_key="sk_test_q\n",
    )
    assert s.anthropic_api_key == "sk-ant-x"
    assert s.llm_api_key == "sk-y"
    assert s.source_github_token == "ghp_z"
    assert s.admin_api_key == "adm"
    assert s.api_key_pepper == "pep"
    assert s.stripe_secret_key == "sk_test_q"


def test_settings_refuse_interior_whitespace_in_a_secret(monkeypatch):
    with pytest.raises(Exception) as e:
        _isolated_settings(monkeypatch, admin_api_key="adm in")
    assert "CC_ADMIN_API_KEY" in str(e.value)


def test_the_suite_never_sees_a_real_key():
    """conftest sets CC_ENV_FILE="" and purges secret env vars before the
    first import, so no test can read the developer's .env, and no
    failing assertion can ever print a live key again."""
    import os

    assert os.environ.get("CC_ENV_FILE") == ""
    assert Settings.model_config.get("env_file") is None
    s = Settings()
    assert s.anthropic_api_key is None
    assert s.llm_api_key is None
    assert s.source_github_token is None
    assert s.stripe_secret_key == ""


# ----------------------------------------------------------------------------
# B1: Key B
# ----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_key_b_is_stripped_at_write_and_at_read(store, customer):
    assert await store.update_customer_upstream_key(customer.id, "sk-test-abc\n")
    c = await store.get_customer_by_id(customer.id)
    ct = c.model_routing_config.api_key_ref
    assert ct.startswith("enc:")
    assert await store.decrypt_tenant_secret(customer.id, "key_b", ct) == "sk-test-abc"


@pytest.mark.asyncio
async def test_key_b_with_interior_whitespace_is_refused(store, customer):
    with pytest.raises(SecretFormatError):
        await store.update_customer_upstream_key(customer.id, "sk-test abc")


@pytest.mark.asyncio
async def test_update_upstream_key_route_400s_on_bad_key(store, customer, monkeypatch):
    from crystal_cache.endpoints import customers as customers_mod

    async def _allow(customer_id, request, store):
        return None

    monkeypatch.setattr(customers_mod, "require_customer_self_or_admin", _allow)

    class _Req:
        async def json(self):
            return {"api_key_ref": "sk-test abc"}

    with pytest.raises(HTTPException) as e:
        await customers_mod.update_upstream_key(customer.id, _Req(), store)
    assert e.value.status_code == 400
    assert "whitespace" in e.value.detail


# ----------------------------------------------------------------------------
# B2: admin bootstrap
# ----------------------------------------------------------------------------

def _bootstrap_env(monkeypatch):
    from crystal_cache.ingress import auth as auth_mod

    monkeypatch.setattr(
        auth_mod, "get_settings",
        lambda: Settings(
            firebase_project_id="test-proj",
            platform_admin_emails="admin@test.dev",
        ),
    )
    return auth_mod


@pytest.mark.asyncio
async def test_bootstrap_refuses_unverified_or_password_claims(store, monkeypatch):
    auth_mod = _bootstrap_env(monkeypatch)
    cases = [
        {"sub": "uid_pw", "email": "admin@test.dev"},
        {"sub": "uid_pw", "email": "admin@test.dev", "email_verified": False,
         "firebase": {"sign_in_provider": "google.com"}},
        {"sub": "uid_pw", "email": "admin@test.dev", "email_verified": True,
         "firebase": {"sign_in_provider": "password"}},
        {"sub": "uid_pw", "email": "admin@test.dev", "email_verified": "true",
         "firebase": {"sign_in_provider": "google.com"}},
    ]
    for claims in cases:
        monkeypatch.setattr(auth_mod, "_verify_firebase_jwt", lambda tok, proj, c=claims: c)
        assert await auth_mod.resolve_firebase_user(store, "eyJx.eyJy.sig") is None, claims
    assert await store.get_user_by_id("uid_pw") is None


@pytest.mark.asyncio
async def test_bootstrap_mints_admin_for_verified_federated_claims(store, monkeypatch):
    auth_mod = _bootstrap_env(monkeypatch)
    monkeypatch.setattr(
        auth_mod, "_verify_firebase_jwt",
        lambda tok, proj: {
            "sub": "uid_ok", "email": "admin@test.dev", "email_verified": True,
            "firebase": {"sign_in_provider": "google.com"},
        },
    )
    user = await auth_mod.resolve_firebase_user(store, "eyJx.eyJy.sig")
    assert user is not None
    assert user.role == "platform_admin"
    assert user.customer_id is None


@pytest.mark.asyncio
async def test_signup_on_admin_email_with_password_account_is_403(store, monkeypatch):
    """The exact B2 attack: register the allowlisted address with a
    password. Signup refuses with a clear 403, mints nothing."""
    from crystal_cache.endpoints import me as me_mod

    auth_mod = _bootstrap_env(monkeypatch)
    monkeypatch.setattr(me_mod, "get_settings", auth_mod.get_settings)
    monkeypatch.setattr(
        auth_mod, "_verify_firebase_jwt",
        lambda tok, proj: {
            "sub": "uid_attacker", "email": "admin@test.dev", "email_verified": True,
            "firebase": {"sign_in_provider": "password"},
        },
    )

    class _Req:
        headers = {"authorization": "Bearer eyJx.eyJy.sig"}

        async def json(self):
            return {}

    with pytest.raises(HTTPException) as e:
        await me_mod.signup(_Req(), store)
    assert e.value.status_code == 403
    assert await store.get_user_by_id("uid_attacker") is None


# ----------------------------------------------------------------------------
# B4: conflicts scan
# ----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_conflicts_scan_obeys_tenant_pin_and_clamps(store, monkeypatch):
    import dataclasses

    from crystal_cache.endpoints import admin as admin_mod

    seen: dict = {}

    @dataclasses.dataclass
    class _Result:
        scanned: int = 0

    async def _scan(*, store, customer_id, max_candidate_pairs, max_discriminator_calls):
        seen.update(
            customer_id=customer_id, pairs=max_candidate_pairs,
            calls=max_discriminator_calls,
        )
        return _Result()

    class _Client:
        def is_ready(self):
            return True

    monkeypatch.setattr(admin_mod, "scan_for_contradictions", _scan)
    monkeypatch.setattr(admin_mod, "get_llm_client", lambda: _Client())

    class _State:
        tenant_pin = "cus_victim_owner"

    class _Req:
        state = _State()

    await admin_mod.admin_scan_conflicts(
        _Req(), store, customer_id="cus_someone_else",
        max_calls=10_000_000, max_pairs=10_000_000,
    )
    assert seen["customer_id"] == "cus_victim_owner"
    assert seen["pairs"] == admin_mod.settings.convergence_max_pairs_per_scan
    assert seen["calls"] == admin_mod.settings.convergence_max_calls_per_cycle

    # The platform admin (no pin) may still target a customer explicitly.
    class _NoPin:
        state = type("S", (), {})()

    await admin_mod.admin_scan_conflicts(
        _NoPin(), store, customer_id="cus_chosen", max_calls=1, max_pairs=2,
    )
    assert seen["customer_id"] == "cus_chosen"
    assert seen["pairs"] == 2
    assert seen["calls"] == 1
