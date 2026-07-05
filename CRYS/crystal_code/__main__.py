"""Entry point: `python -m crystal_code`.

Log quieting happens HERE, before cli (and through it the library) is
imported — tool-registration debug lines fire at import time, so
configuring inside main() would be too late. A raw argv scan stands in
for argparse, which hasn't run yet; cli's --verbose flag is the
documented face of the same switch.
"""
import sys

from .style import quiet_library_logs

quiet_library_logs(verbose=("-v" in sys.argv or "--verbose" in sys.argv))

# Showcase needs these capability flags ON before the library builds its cached
# settings (which happens on the first crystal_cache import, via cli below).
# Same raw-argv-before-import reason as the log quieting above; showcase.py
# mirrors these as a backstop. Gated on the flag so normal runs are untouched.
# Matches --showcase AND --showcase-acts (the latter implies the showcase in
# cli.main), including the --showcase-acts=... form.
if any(a.startswith("--showcase") for a in sys.argv):
    import os
    for _k, _v in {
        "CC_TEXT_ENCODER": "semantic",
        "CC_ENABLE_CITATIONS": "1",
        "CC_ENABLE_MARKETPLACE_METERING": "1",
        "CC_ENABLE_COST_ACCOUNTING": "1",
        "CC_AGENT_RECALL": "1",
    }.items():
        os.environ.setdefault(_k, _v)

# Zero-config secrets-at-rest for local flows (2026-07-02): the library
# encrypts Key B unconditionally and needs CC_TOKEN_ENCRYPTION_KEY. Load or
# generate the per-user key BEFORE the first crystal_cache import builds the
# cached settings; a real exported env var still wins (production pattern).
from .config_store import ensure_token_encryption_key  # noqa: E402

ensure_token_encryption_key()

from .cli import main  # noqa: E402 — must come after log config

main()
