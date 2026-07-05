"""First-time setup for the coding agent.

Asks three things — your name, the provider, and the API key — and saves
them to the agent's own config (config_store). The same walk runs on a
fresh install anywhere, which is why it doubles as the production
onboarding flow. Triggered when no key is found by any source, or on
demand via the `/setup` command.
"""
from __future__ import annotations

import getpass

from . import config_store


def run_setup() -> config_store.Credentials:
    """Run the interactive setup and return the resulting credentials."""
    print("\nFirst-time setup for the Crystal Cache coding agent.")
    print("(Run /setup anytime to change these.)\n")

    name = input("Your name: ").strip()

    # Provider. Only Anthropic is wired today; the choice is still
    # recorded so the flow matches production and other providers can be
    # added later without changing these prompts.
    wired = ", ".join(sorted(config_store.PROVIDERS))
    print(f"\nProvider (wired today: {wired}):")
    provider = input(f"  provider [{config_store.DEFAULT_PROVIDER}]: ").strip().lower()
    provider = provider or config_store.DEFAULT_PROVIDER
    if provider not in config_store.PROVIDERS:
        print(f"  '{provider}' isn't wired yet — using "
              f"'{config_store.DEFAULT_PROVIDER}' for now.")
        provider = config_store.DEFAULT_PROVIDER
    meta = config_store.provider_meta(provider)

    # API key — read hidden so it isn't echoed to the screen or scrollback.
    print(f"\nPaste your {meta['env_var']} (input hidden):")
    api_key = ""
    while not api_key:
        api_key = getpass.getpass("  key: ").strip()
        if not api_key:
            print("  (a key is required)")

    model = meta["default_model"]
    path = config_store.save_config(
        name=name, provider=provider, model=model, api_key=api_key,
    )
    print(f"\nSaved to {path}\n")

    return config_store.Credentials(
        name=name, provider=provider, model=model, api_key=api_key, source="setup",
    )
