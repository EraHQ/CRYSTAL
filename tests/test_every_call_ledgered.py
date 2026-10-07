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
    assert offenders == [], offenders


def test_proxy_tool_loop_second_call_is_ledgered():
    from crystal_cache.endpoints import chat_proxy

    src = inspect.getsource(chat_proxy)
    loop = src.split("loop_upstream = await client.complete(", 1)[1]
    head = loop.split("push_pull.tool_loop_complete", 1)[0]
    assert "record_model_call(" in head
    assert 'origin="tool_loop"' in head


def test_mcp_export_marks_keys_as_paths_and_import_honours_it():
    from crystal_cache.agent import mcp_server

    src = inspect.getsource(mcp_server)
    export_body = src.split("async def memory_export(", 1)[1].split("async def ", 1)[0]
    assert '"key_is_path": True' in export_body
    import_body = src.split("async def memory_import(", 1)[1].split("async def ", 1)[0]
    assert 'key if rec.get("key_is_path") else' in import_body
