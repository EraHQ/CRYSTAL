"""Entry point: `python -m crys`.

CRYS is the brand; `crystal_code` is the package. This shim makes the
name also the command, with zero duplication — it configures logging
exactly like crystal_code/__main__.py (before imports, because tool
registrations log at import time) and hands off to the same main().
"""
import sys

from crystal_code.style import quiet_library_logs

quiet_library_logs(verbose=("-v" in sys.argv or "--verbose" in sys.argv))

from crystal_code.cli import main  # noqa: E402 — must come after log config

main()
