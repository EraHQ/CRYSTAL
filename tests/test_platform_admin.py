"""WS D / D.1 — platform-admin gate.

The entire gate policy lives in ingress.auth as pure functions, so these
tests need neither the full app nor its lifespan. Covers:
  - the gate-active rule (off in dev, on when a key is set or in production),
  - the constant-time key check,
  - the Bearer-header parser,
  - the protected-surface matcher,
  - platform_admin_error — the integrated allow/deny decision the app
    middleware is thin glue over.
"""
from __future__ import annotations

import pytest

from crystal_cache.config import Settings
from crystal_cache.ingress import auth as auth_mod


ADMIN_KEY = "cc_sk_admin_test_0000"


def _settings(**kw) -> Settings:
    base = dict(environment="development", admin_api_key="", api_key_pepper="",
                host="127.0.0.1")
    base.update(kw)
    return Settings(**base)


def _use(monkeypatch, **kw) -> None:
    """Point ingress.auth at a Settings built from kw."""
    monkeypatch.setattr(auth_mod, "get_settings", lambda: _settings(**kw))


# --- gate-active rule -------------------------------------------------------

def test_gate_inactive_in_dev_without_key(monkeypatch):
    # Dev + no key + LOOPBACK bind + no serverless markers → off (the
    # preserved zero-config path). Markers cleared so the test is stable even
    # when the suite itself runs on a hosted CI box.
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.delenv("K_REVISION", raising=False)
    _use(monkeypatch)
    assert auth_mod.platform_admin_gate_active() is False


# --- B2 hardening: fail-closed on a networked bind --------------------------

def test_gate_active_on_nonloopback_bind_without_key(monkeypatch):
    """The B2 fix: a networked (0.0.0.0) dev server with no key ENFORCES the
    gate — the admin surface is not silently open off-machine."""
    _use(monkeypatch, host="0.0.0.0")
    assert auth_mod.platform_admin_gate_active() is True


def test_gate_active_on_lan_ip_bind_without_key(monkeypatch):
    _use(monkeypatch, host="192.168.1.50")
    assert auth_mod.platform_admin_gate_active() is True


def test_gate_inactive_on_ipv6_loopback(monkeypatch):
    _no_hosted_env(monkeypatch)
    _use(monkeypatch, host="::1")
    assert auth_mod.platform_admin_gate_active() is False


def test_gate_inactive_on_localhost_literal(monkeypatch):
    _no_hosted_env(monkeypatch)
    _use(monkeypatch, host="localhost")
    assert auth_mod.platform_admin_gate_active() is False


# --- Cloud Run fix: hosted platform enforces despite a stale loopback host --
# Regression for the 2026-07-06 fail-open incident. On Cloud Run the container
# CMD binds 0.0.0.0 but Settings.host keeps its 127.0.0.1 default, so the old
# gate read a stale loopback value and left /admin/api OPEN on the public
# internet. K_SERVICE (always set by Cloud Run) is now the trusted signal.

def _hosted_env(monkeypatch) -> None:
    """Simulate a Cloud Run container environment."""
    monkeypatch.setenv("K_SERVICE", "crystal-api")
    monkeypatch.setenv("K_REVISION", "crystal-api-00042-abc")


def _no_hosted_env(monkeypatch) -> None:
    """Simulate a self-host / local box (no serverless markers)."""
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.delenv("K_REVISION", raising=False)


def test_gate_active_on_cloud_run_with_stale_loopback_host(monkeypatch):
    """THE incident: dev-defaulted settings (host=127.0.0.1, no key, not
    production) but running on Cloud Run. The gate MUST enforce — the stale
    loopback host is not trustworthy when a platform marker is present."""
    _hosted_env(monkeypatch)
    _use(monkeypatch)  # host defaults to 127.0.0.1
    assert auth_mod.platform_admin_gate_active() is True


def test_gate_active_on_cloud_run_via_k_revision_only(monkeypatch):
    """Either marker suffices; K_REVISION alone still means hosted."""
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.setenv("K_REVISION", "crystal-api-00042-abc")
    _use(monkeypatch)
    assert auth_mod.platform_admin_gate_active() is True


def test_gate_disable_cannot_open_hosted_surface(monkeypatch):
    """A CC_ADMIN_GATE_DISABLE left on from local dev must NOT open a real
    Cloud Run admin surface — hosted enforcement sits above the hatch."""
    _hosted_env(monkeypatch)
    _use(monkeypatch, admin_gate_disable=True)  # stale hatch from dev
    assert auth_mod.platform_admin_gate_active() is True


def test_gate_inactive_on_self_host_loopback_no_markers(monkeypatch):
    """The fix must not disturb self-host / local dev: loopback bind, no key,
    no serverless markers → gate stays OFF (zero-config ergonomics)."""
    _no_hosted_env(monkeypatch)
    _use(monkeypatch, host="127.0.0.1")
    assert auth_mod.platform_admin_gate_active() is False


def test_gate_disable_escape_hatch_forces_off(monkeypatch):
    """CC_ADMIN_GATE_DISABLE is the conscious opt-out: even a networked bind
    goes off (non-hosted). Strongly discouraged, but explicit."""
    _no_hosted_env(monkeypatch)
    _use(monkeypatch, host="0.0.0.0", admin_gate_disable=True)
    assert auth_mod.platform_admin_gate_active() is False


def test_gate_disable_does_not_override_production(monkeypatch):
    """Production ALWAYS enforces — the escape hatch can never open a
    production admin surface, even if a dev left CC_ADMIN_GATE_DISABLE on.
    The hatch applies only to the non-production, no-key case."""
    _use(monkeypatch, environment="production", admin_gate_disable=True)
    assert auth_mod.platform_admin_gate_active() is True


def test_gate_disable_does_not_override_explicit_key(monkeypatch):
    """An explicit admin key also wins over the hatch — if you set a key you
    want the gate on."""
    _use(monkeypatch, host="0.0.0.0", admin_api_key=ADMIN_KEY,
         admin_gate_disable=True)
    assert auth_mod.platform_admin_gate_active() is True


def test_gate_active_when_key_set(monkeypatch):
    _use(monkeypatch, admin_api_key=ADMIN_KEY)
    assert auth_mod.platform_admin_gate_active() is True


def test_gate_active_in_production_even_without_key(monkeypatch):
    _use(monkeypatch, environment="production")
    assert auth_mod.platform_admin_gate_active() is True


# --- constant-time key check ------------------------------------------------

def test_is_platform_admin_token_matches_only_configured(monkeypatch):
    _use(monkeypatch, admin_api_key=ADMIN_KEY)
    assert auth_mod.is_platform_admin_token(ADMIN_KEY) is True
    assert auth_mod.is_platform_admin_token(f"  {ADMIN_KEY}  ") is True  # trimmed
    assert auth_mod.is_platform_admin_token("wrong") is False
    assert auth_mod.is_platform_admin_token(None) is False
    assert auth_mod.is_platform_admin_token("") is False


def test_token_never_matches_when_unconfigured(monkeypatch):
    _use(monkeypatch, admin_api_key="")
    assert auth_mod.is_platform_admin_token("anything") is False


# --- Bearer header parser ---------------------------------------------------

@pytest.mark.parametrize("header,expected", [
    ("Bearer abc", "abc"),
    ("bearer abc", "abc"),
    ("Bearer   abc  ", "abc"),
    ("Basic abc", None),
    ("abc", None),
    ("", None),
    (None, None),
    ("Bearer ", None),
])
def test_bearer_token_from_header(header, expected):
    assert auth_mod._bearer_token_from_header(header) == expected


# --- protected-surface matcher ---------------------------------------------

@pytest.mark.parametrize("method,path,expected", [
    ("GET", "/admin/api/customers", True),
    ("POST", "/admin/api/push-queue/x/approve", True),
    ("GET", "/admin/api/cognition/environments", True),
    ("GET", "/admin/api/metacognition/state", True),
    ("POST", "/v1/customers", True),
    ("POST", "/v1/customers/", True),
    ("GET", "/admin", False),               # SPA shell
    ("GET", "/admin/assets/app.js", False),  # SPA assets
    ("GET", "/admin/cognition", False),      # SPA client route (not /admin/api)
    ("GET", "/v1/customers/abc", False),     # team-scoped read, not platform
    ("PATCH", "/v1/customers/abc/upstream_key", False),
    ("POST", "/v1/chat/completions", False),
    ("POST", "/mcp/", False),
])
def test_path_needs_platform_admin(method, path, expected):
    assert auth_mod.path_needs_platform_admin(method, path) is expected


# --- integrated decision (what the middleware renders) ----------------------

def test_error_none_when_gate_inactive(monkeypatch):
    _use(monkeypatch)  # dev, no key
    assert auth_mod.platform_admin_error("GET", "/admin/api/customers", None) is None
    assert auth_mod.platform_admin_error("POST", "/v1/customers", None) is None


def test_error_allows_public_paths_even_when_gated(monkeypatch):
    _use(monkeypatch, admin_api_key=ADMIN_KEY)
    assert auth_mod.platform_admin_error("GET", "/admin", None) is None
    assert auth_mod.platform_admin_error("GET", "/admin/assets/app.js", None) is None
    assert auth_mod.platform_admin_error("POST", "/v1/chat/completions", None) is None
    assert auth_mod.platform_admin_error("GET", "/v1/customers/abc", None) is None


def test_error_401_on_admin_surface_without_or_wrong_key(monkeypatch):
    _use(monkeypatch, admin_api_key=ADMIN_KEY)
    assert auth_mod.platform_admin_error("GET", "/admin/api/customers", None) == (
        401, "platform admin credential required",
    )
    assert auth_mod.platform_admin_error(
        "GET", "/admin/api/customers", "Bearer wrong"
    ) == (401, "platform admin credential required")


def test_error_allows_admin_surface_with_correct_key(monkeypatch):
    _use(monkeypatch, admin_api_key=ADMIN_KEY)
    assert auth_mod.platform_admin_error(
        "GET", "/admin/api/customers", f"Bearer {ADMIN_KEY}"
    ) is None


def test_error_gates_customer_minting(monkeypatch):
    _use(monkeypatch, admin_api_key=ADMIN_KEY)
    assert auth_mod.platform_admin_error("POST", "/v1/customers", None) == (
        401, "platform admin credential required",
    )
    assert auth_mod.platform_admin_error(
        "POST", "/v1/customers", f"Bearer {ADMIN_KEY}"
    ) is None


def test_error_gates_in_production_without_key(monkeypatch):
    # Production turns the gate on; with no key configured nothing can satisfy
    # it, so the admin surface is 401 to everyone (the boot guard separately
    # refuses to start, but the policy itself fails closed regardless).
    _use(monkeypatch, environment="production", admin_api_key="")
    assert auth_mod.platform_admin_error(
        "GET", "/admin/api/customers", "Bearer anything"
    ) == (401, "platform admin credential required")
