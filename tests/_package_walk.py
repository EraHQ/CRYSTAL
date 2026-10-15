"""The one package walk every structural pin uses (Lockdown PR-8, Q57
class 7, 2026-10-13).

The four walks that scanned `crystal_cache` by importing every module
each swallowed every import error (`except Exception: continue`) and
asserted `offenders == []` with no floor on what they inspected: a
broken import made the walk empty and green. This helper imports every
module, FAILS on an import error, and asserts a floor on the modules
it returns, so a walk that sees nothing cannot pass.

Modules under `.benchmarks` and `.scripts` are skipped as before (they
are tools, not the service). A module that cannot be imported because
an OPTIONAL dependency is absent on this machine is listed in
OPTIONAL_DEPENDENCY_MODULES with the dependency named; anything else
that fails to import is a finding.
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil

import crystal_cache

# Expected count of importable, source-bearing modules (267 .py files on
# 2026-10-13). A floor well under the real number so an added or removed
# module never trips it, but an empty or half-empty walk does.
MIN_MODULES = 200

# module name -> the optional dependency whose absence is tolerated.
OPTIONAL_DEPENDENCY_MODULES: dict[str, str] = {}


def walk_modules():
    """Yield (name, module, source) for every module in the package.
    Asserts that nothing failed to import (beyond the declared optional
    set) and that at least MIN_MODULES were seen."""
    found: list[tuple[str, object, str]] = []
    failures: list[str] = []
    for mod in pkgutil.walk_packages(crystal_cache.__path__, "crystal_cache."):
        if ".benchmarks" in mod.name or ".scripts" in mod.name:
            continue
        try:
            m = importlib.import_module(mod.name)
        except Exception as e:  # noqa: BLE001 — recorded, then asserted below
            if mod.name in OPTIONAL_DEPENDENCY_MODULES:
                continue
            failures.append(f"{mod.name}: {type(e).__name__}: {e}")
            continue
        try:
            src = inspect.getsource(m)
        except (OSError, TypeError):
            # A namespace package or an extension module has no source to scan.
            continue
        found.append((mod.name, m, src))
    assert not failures, "modules that failed to import:\n" + "\n".join(failures)
    assert len(found) >= MIN_MODULES, f"only {len(found)} modules walked (floor {MIN_MODULES})"
    return found


def package_sources() -> list[tuple[str, str]]:
    """(module name, source) for every module, with the walk's guarantees."""
    return [(name, src) for name, _m, src in walk_modules()]
