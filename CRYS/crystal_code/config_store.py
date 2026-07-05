"""The coding agent's own configuration — its home, separate from any project.

Stores who you are, which provider/model, and the API key in
~/.crystal-code/config.json. The agent owning its own config (rather than
depending on a project's .env) is what makes the first run identical
locally and in production: a fresh install asks three things, saves them,
and never asks again.

Credential precedence (highest first):
  1. A real exported environment variable (how a server/production box
     injects the key — no file, no prompt).
  2. This saved config (~/.crystal-code/config.json).
  3. The project's .env at the repo root (a convenience for this repo).
  4. Nothing -> the caller runs first-time setup.

The .env is parsed directly (NOT loaded into the environment) so it can
never masquerade as a real exported variable and outrank the saved
config.
"""
from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


CONFIG_DIR = Path.home() / ".crystal-code"
CONFIG_PATH = CONFIG_DIR / "config.json"
TOKEN_KEY_PATH = CONFIG_DIR / "token.key"


def ensure_token_encryption_key() -> str:
    """Zero-config secrets-at-rest for LOCAL flows (2026-07-02 follow-up to
    the launch-prep security pass).

    The library encrypts Key B unconditionally and needs
    CC_TOKEN_ENCRYPTION_KEY. Precedence mirrors credential precedence: a
    real exported env var wins (the server/production pattern — see
    SELF_HOSTING.md); otherwise load, or generate ONCE, a per-user key at
    ~/.crystal-code/token.key (0600) and export it into this process
    before the library builds its cached settings. Fail-safe: on any file
    error return "" and let the library raise its clear
    set-the-key message at first store use rather than dying here.
    """
    existing = os.environ.get("CC_TOKEN_ENCRYPTION_KEY")
    if existing:
        return existing
    try:
        if TOKEN_KEY_PATH.exists():
            key = TOKEN_KEY_PATH.read_text(encoding="utf-8").strip()
        else:
            import secrets
            key = secrets.token_hex(32)
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            TOKEN_KEY_PATH.write_text(key + "\n", encoding="utf-8")
            try:
                os.chmod(TOKEN_KEY_PATH, stat.S_IRUSR | stat.S_IWUSR)
            except OSError:
                pass
        if key:
            os.environ["CC_TOKEN_ENCRYPTION_KEY"] = key
        return key
    except OSError:
        return ""

# Provider -> its env-var name and default model. The table is the single
# place a provider gets added; runtime.py reads the recorded provider and
# builds the provider-neutral LLM client from it (build_llm_client), so
# both entries route through the same seam.
PROVIDERS: dict[str, dict[str, str]] = {
    "anthropic": {
        "env_var": "ANTHROPIC_API_KEY",
        "default_model": "claude-sonnet-4-5-20250929",
    },
    # Any OpenAI-compatible chat-completions endpoint (OpenAI, vLLM,
    # Ollama, Groq, ...). Requires base_url (config "base_url" or
    # CC_LLM_BASE_URL) and a provider-native model name — there is no
    # sensible cross-server default model, so setup must record one.
    "openai": {
        "env_var": "CC_LLM_API_KEY",
        "default_model": "",
    },
}

DEFAULT_PROVIDER = "anthropic"


@dataclass
class Credentials:
    """Resolved credentials plus where they came from (for a startup note)."""
    name: str
    provider: str
    model: str
    api_key: str
    source: str  # "env" | "config" | "dotenv" | "setup"
    # OpenAI-compatible endpoints only; None under anthropic.
    base_url: Optional[str] = None


def provider_meta(provider: str) -> dict[str, str]:
    """Return a provider's metadata, falling back to the default provider."""
    return PROVIDERS.get(provider, PROVIDERS[DEFAULT_PROVIDER])


def load_config() -> Optional[dict]:
    """Read the saved config, or None if there isn't one (or it's unreadable)."""
    if not CONFIG_PATH.exists():
        return None
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save_config(
    *, name: str, provider: str, model: str, api_key: str,
    base_url: Optional[str] = None,
) -> Path:
    """Write the config to the agent's own home with owner-only permissions.

    Merges over the existing config rather than replacing it, so fields
    owned by other flows (the /login state: cc_db, cc_customer_id)
    survive a /setup rerun.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = load_config() or {}
    payload.update({
        "name": name,
        "provider": provider,
        "model": model,
        "api_key": api_key,
    })
    if base_url is not None:
        payload["base_url"] = base_url
    CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Best-effort: restrict to owner read/write. A no-op on platforms that
    # don't honor POSIX permission bits, but harmless there.
    try:
        os.chmod(CONFIG_PATH, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    return CONFIG_PATH


# ---------------------------------------------------------------------------
# Crystal Cache login (/login) — which knowledge store + customer the agent
# uses. Persisted alongside the LLM credentials in the same config file.
#
# Only the DB location and the RESOLVED customer id are stored. The
# customer's API key (Key A) is used once during /login to resolve the
# customer and is never written to disk — the local agent loads the
# customer record directly by id, so the key isn't needed again.
# ---------------------------------------------------------------------------

def load_login() -> tuple[Optional[str], Optional[str]]:
    """Return the saved (cc_db, cc_customer_id), either may be None."""
    cfg = load_config() or {}
    return cfg.get("cc_db") or None, cfg.get("cc_customer_id") or None


def save_login(db: Optional[str], customer_id: str) -> Path:
    """Persist the Crystal Cache login (merge-preserving)."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = load_config() or {}
    payload["cc_db"] = db or ""
    payload["cc_customer_id"] = customer_id
    CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        os.chmod(CONFIG_PATH, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    return CONFIG_PATH


def clear_login() -> None:
    """Remove the saved Crystal Cache login (LLM credentials untouched)."""
    cfg = load_config()
    if not cfg:
        return
    cfg.pop("cc_db", None)
    cfg.pop("cc_customer_id", None)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _read_key_from_dotenv(env_var: str) -> Optional[str]:
    """Parse the repo-root .env directly for a provider's key.

    Lowest-priority source. Parsed by hand rather than loaded into the
    environment, so it never looks like a real exported variable and
    can't outrank the saved config.
    """
    # config_store.py -> crystal_code -> CRYS -> repo root
    root_env = Path(__file__).resolve().parents[2] / ".env"
    if not root_env.exists():
        return None
    wanted = {env_var, f"CC_{env_var}"}
    for raw in root_env.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() in wanted and value.strip():
            return value.strip()
    return None


def resolve_credentials() -> Optional[Credentials]:
    """Resolve credentials by precedence. None means 'run first-time setup'."""
    cfg = load_config()
    provider = (cfg or {}).get("provider", DEFAULT_PROVIDER)
    meta = provider_meta(provider)
    model = (cfg or {}).get("model") or meta["default_model"]
    name = (cfg or {}).get("name", "")
    env_var = meta["env_var"]
    # OpenAI-compatible endpoints need a base URL; env wins over config.
    base_url = os.environ.get("CC_LLM_BASE_URL") or (cfg or {}).get("base_url") or None

    # 1. Real exported environment variable (the production path).
    env_key = os.environ.get(env_var) or os.environ.get(f"CC_{env_var}")
    if env_key:
        return Credentials(name, provider, model, env_key, "env", base_url)

    # 2. The agent's own saved config.
    if cfg and cfg.get("api_key"):
        return Credentials(name, provider, model, cfg["api_key"], "config", base_url)

    # 3. The project .env (parsed directly, lowest priority).
    dotenv_key = _read_key_from_dotenv(env_var)
    if dotenv_key:
        return Credentials(name, provider, model, dotenv_key, "dotenv", base_url)

    # 4. Nothing found — caller runs setup.
    return None
