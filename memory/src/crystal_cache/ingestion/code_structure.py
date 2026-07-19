"""Code structure extraction — the MECHANICAL half of Gate D2 (code
comprehension at ingest, ratified 2026-07-17).

Doctrine split: mechanism in code, judgment in models. This module is
the mechanism: imports and cross-file references derived from source
text with zero model spend, deterministic and reproducible. The
judgment half (what a symbol DOES) is the code describer's output,
promoted to facts by the pipeline alongside these.

v1 scope: Python import syntax (the AST chunker is Python-only today;
other languages ride the same seam when their chunkers land). Regex
over reconstructed chunk text rather than ast.parse — the pipeline
holds chunks, not the pristine file, and a reconstruction that fails
to parse must not cost us the imports we can plainly see.
"""
from __future__ import annotations

import re
from typing import Any, Optional

# `import a.b.c` / `import a.b.c as x`  and  `from a.b.c import y` /
# `from ..a import y` (relative levels preserved as leading dots).
_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([.\w]+)\s+import|import\s+([.\w]+))",
    re.MULTILINE,
)


def extract_imports(text: str) -> list[str]:
    """All imported module paths in source order, deduped, dots kept
    (relative imports keep their leading dots — resolution handles
    them). Comments/strings can false-positive in pathological cases;
    acceptable for v1 — a phantom import fact is inert."""
    seen: dict[str, None] = {}
    for m in _IMPORT_RE.finditer(text or ""):
        mod = (m.group(1) or m.group(2) or "").strip()
        if mod and mod != ".":
            seen.setdefault(mod, None)
    return list(seen)


def _module_suffixes(module: str, importer_path: str) -> list[str]:
    """Candidate path suffixes a module could live at inside the bank.

    'crystal_cache.cost.emit'  -> ['crystal_cache/cost/emit.py', ...]
    '..cost.emit' (relative)   -> ['cost/emit.py']
    Package imports also try '<path>/__init__.py'.
    Suffixes shorter than 2 path segments are dropped (single-name
    matching is ambiguity bait); the resolver additionally requires a
    UNIQUE match.
    """
    dotted = module.lstrip(".")
    if not dotted:
        return []
    parts = dotted.split(".")
    out: list[str] = []
    # Longest suffix first (most specific), down to 2 segments.
    for i in range(len(parts)):
        segs = parts[i:]
        if len(segs) < 2 and len(parts) >= 2:
            break
        path = "/".join(segs)
        out.append(f"{path}.py")
        out.append(f"{path}/__init__.py")
        if len(segs) < 2:
            break
    return out


def _authority(path: str) -> Optional[str]:
    """The largest parent — segment one of a multi-segment path.

    Gate D6 (ratified 2026-07-18, amends C1): repo identity is
    repo://<authority>/<path>. The picked root (or named source) IS
    the repo's scope; imports resolve only within their own authority,
    because a Python import cannot reach outside the tree it was
    uploaded with. Single-segment paths (bare single-file uploads)
    have no authority and resolve unscoped — legacy behavior.
    """
    if "/" in (path or ""):
        return path.split("/", 1)[0]
    return None


def resolve_import_target(
    module: str, importer_path: str, candidates: list[Any],
) -> Optional[Any]:
    """The crystal a module import points at, or None.

    Conservative by design: a candidate matches when its source_path or
    source_uri path ends with one of the module's path suffixes; the
    resolution is accepted ONLY when exactly one crystal matches (an
    ambiguous import becomes a fact without a chain — never a wrong
    edge). External packages match nothing and stay facts-only.

    Authority scoping (Gate D6): when the importer has an authority,
    only same-authority candidates are considered — a same-shaped
    subtree in a DIFFERENT repo can neither steal the edge nor
    suppress it as ambiguity.
    """
    auth = _authority(importer_path)
    if auth is not None:
        candidates = [
            c for c in candidates
            if _authority(str(getattr(c, "source_path", "") or "")) == auth
            or _authority(
                (getattr(c, "source_uri", "") or "")[len("repo://"):]
                if (getattr(c, "source_uri", "") or "").startswith("repo://")
                else ""
            ) == auth
        ]
    for suffix in _module_suffixes(module, importer_path):
        matched = []
        for c in candidates:
            paths = []
            if getattr(c, "source_path", None):
                paths.append(str(c.source_path))
            uri = getattr(c, "source_uri", None) or ""
            if uri.startswith("repo://"):
                paths.append(uri[len("repo://"):])
            if any(p == suffix or p.endswith("/" + suffix) for p in paths):
                matched.append(c)
        uniq = {c.id: c for c in matched}
        if len(uniq) == 1:
            return next(iter(uniq.values()))
    return None
