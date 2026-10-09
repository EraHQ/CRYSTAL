"""RC-14 (2026-10-06): nothing slow runs on the event loop, and the SDK's
limits are the ones the non-streaming guard needs.

- AST walk: inside any `async def` in the package, a direct call to a
  sync model seam (complete / complete_detailed), file extraction, or a
  KMS wrap/unwrap must be inside asyncio.to_thread. A sync call there
  stalls every other request for its duration.
- The Anthropic client is constructed with the SDK's own DEFAULT_TIMEOUT
  object and max_retries=2. A plain float does not compare equal to the
  default and silently disables the SDK's "streaming required above
  21,333 tokens" guard (verified in a sandbox against anthropic 1.2.0).
- Detached agent runs are drained with a bounded wait on shutdown.
"""
import ast
import importlib
import inspect
import pkgutil

import crystal_cache

_SYNC_SEAM_ATTRS = {"complete", "complete_detailed", "unwrap", "wrap"}
_SYNC_FUNCS = {"extract_text_from_file", "_call_describe", "_synopsize",
               "_consolidate_llm", "_run_meta_reflection_llm"}

# Sites the walk flags that are NOT sync calls, each with its reason.
_NOT_SYNC = {
    # The proxy's upstream client is ASYNC: client.complete(...) returns a
    # coroutine that is awaited later through asyncio.gather alongside the
    # shadow call (verified: chat_proxy.run_chat_completion, primary_coro).
    ("crystal_cache.endpoints.chat_proxy", "run_chat_completion", "complete"),
}


def _call_name(node: ast.Call):
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return None


def _is_inside_to_thread(path):
    """True if any enclosing Call in `path` is asyncio.to_thread(...)."""
    for anc in path:
        if isinstance(anc, ast.Call):
            n = _call_name(anc)
            if n == "to_thread":
                return True
    return False


def _walk_with_path(node, path=()):
    yield node, path
    for child in ast.iter_child_nodes(node):
        yield from _walk_with_path(child, path + (node,))


def test_no_sync_seam_call_runs_directly_on_the_loop():
    offenders = []
    for mod in pkgutil.walk_packages(crystal_cache.__path__, "crystal_cache."):
        if ".benchmarks" in mod.name or ".scripts" in mod.name:
            continue
        try:
            m = importlib.import_module(mod.name)
            src = inspect.getsource(m)
        except Exception:  # noqa: BLE001
            continue
        tree = ast.parse(src)
        for node, path in _walk_with_path(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name not in _SYNC_SEAM_ATTRS and name not in _SYNC_FUNCS:
                continue
            # Only calls whose nearest enclosing function is async matter.
            enclosing = next((a for a in reversed(path)
                              if isinstance(a, (ast.AsyncFunctionDef, ast.FunctionDef, ast.Lambda))), None)
            if not isinstance(enclosing, ast.AsyncFunctionDef):
                continue
            # The seam's own async wrappers call the sync method via to_thread
            # with the method passed as an argument (functools.partial or a
            # bare attribute), which is not a Call node and never trips this.
            if _is_inside_to_thread(path + (node,)):
                continue
            # Awaited coroutines named complete() (e.g. an async client) are fine.
            if path and isinstance(path[-1], ast.Await):
                continue
            if (mod.name, enclosing.name, name) in _NOT_SYNC:
                continue
            offenders.append(f"{mod.name}:{node.lineno}: {name}(...) inside async def {enclosing.name}")
    assert offenders == [], "\n".join(offenders)


def test_anthropic_client_uses_the_sdk_default_timeout_object():
    import anthropic

    from crystal_cache.llm.client import _sdk_limits

    limits = _sdk_limits()
    assert limits["timeout"] is anthropic.DEFAULT_TIMEOUT
    assert limits["max_retries"] == 2
    # And the fact the guard depends on, pinned so a future float edit fails here:
    assert 600.0 != anthropic.DEFAULT_TIMEOUT


def test_lifespan_drains_detached_runs():
    from crystal_cache import app as app_mod

    src = inspect.getsource(app_mod)
    # RC-14 (PR-4): one budget for the whole shutdown, inside Cloud Run's
    # 10 s SIGTERM window; the drain takes what the workers leave.
    assert app_mod.SHUTDOWN_BUDGET_SECONDS <= 8.0
    assert app_mod.SHUTDOWN_WORKER_SECONDS < app_mod.SHUTDOWN_BUDGET_SECONDS
    assert "_DETACHED_RUNS" in src and "asyncio.wait(pending, timeout=_remaining())" in src
    assert "asyncio.wait_for(task, timeout=10)" not in src
