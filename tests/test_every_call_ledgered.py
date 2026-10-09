"""RC-11 (2026-10-06): no model call escapes the ledger.

Twelve sites did `getattr(client, "complete_detailed", None)` with a
fallback to the unmetered complete(); the fallback was reachable only by
a fake without the method, and such a fake hid two unmetered spenders
for months. All twelve now bind the seam's method directly. The proxy's
tool-loop second call, a real second paid call per research turn, is
ledgered under origin tool_loop. memory_import keeps an exported key
verbatim (key_is_path) instead of paying a Haiku call per record.
"""
import importlib
import inspect
import pkgutil
import re

import crystal_cache


def test_no_complete_detailed_fallback_anywhere_in_the_package():
    offenders = []
    for mod in pkgutil.walk_packages(crystal_cache.__path__, "crystal_cache."):
        if ".benchmarks" in mod.name or ".scripts" in mod.name:
            continue
        try:
            m = importlib.import_module(mod.name)
            src = inspect.getsource(m)
        except Exception:  # noqa: BLE001  (optional deps, builtins)
            continue
        for line_no, line in enumerate(src.splitlines(), 1):
            if re.search(r'getattr\([^,]+,\s*"(complete_detailed|complete_with_usage)"', line):
                offenders.append(f"{mod.name}:{line_no}: {line.strip()}")
            # Lockdown PR-4 (RC-11 residue, class 3): the equivalent dead
            # shape — bind the bound method, test it for None, fall back
            # to complete(). A bound method is never None, so the branch
            # was dead and the comment next to it ("fakes exposing only
            # complete() run unmetered") was false.
            if re.search(r'\b\w*detailed\w*\s+is\s+(not\s+)?None\b', line):
                offenders.append(f"{mod.name}:{line_no}: {line.strip()}")
            if re.search(r'\bif\s+\w+\s+is\s+not\s+None\s+else\s+\w+\.complete\b', line):
                offenders.append(f"{mod.name}:{line_no}: {line.strip()}")
    assert offenders == [], offenders


def test_proxy_tool_loop_second_call_is_ledgered():
    from crystal_cache.endpoints import chat_proxy

    src = inspect.getsource(chat_proxy)
    loop = src.split("loop_upstream = await client.complete(", 1)[1]
    head = loop.split("push_pull.tool_loop_complete", 1)[0]
    assert "record_model_call(" in head
    assert 'origin="tool_loop"' in head


# The key_is_path pin moved to a behavioural test (PR-2, 2026-10-08):
# tests/test_fact_export_roundtrip.py::test_mcp_export_and_import_speak_the_same_record
# imports a key_is_path record with no model client wired, so a wrong
# branch would try to derive the key and fail.
