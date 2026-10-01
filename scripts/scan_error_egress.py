"""List every site where exception text flows into a client-visible or
persisted error (the invariant pinned by
tests/test_security_sweep.py::test_no_route_echoes_exception_text_into_a_response).

Run from the repo root:  python scripts/scan_error_egress.py

A line that must keep exception text (our own domain message) carries a
`detail-ok:` comment on one of the three lines above it.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "memory" / "src" / "crystal_cache"
EXC = r"(?:str\(e\w*\)|repr\(e\w*\)|f[\"'][^\"']*\{(?:e|exc|err)\}|f[\"'][^\"']*\{str\((?:e|exc|err)\)\})"
PATTERNS = (
    re.compile(r"detail\s*=\s*" + EXC),
    re.compile(r"[\"'](?:error|reason|message|error_message)[\"']\s*:\s*" + EXC),
    re.compile(r"\[[\"'](?:error|reason|message|error_message)[\"']\]\s*=\s*" + EXC),
    re.compile(r"mark_document_error\([^)]*" + EXC),
)


def offenders():
    for path in sorted(ROOT.rglob("*.py")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            window = lines[max(0, i - 3): i + 1]
            if not any(p.search(line) for p in PATTERNS):
                continue
            if any("detail-ok" in w for w in window):
                continue
            if any("extra=" in w or "logger." in w for w in window):
                continue
            yield f"{path.relative_to(ROOT)}:{i + 1}: {line.strip()}"


if __name__ == "__main__":
    found = list(offenders())
    print("\n".join(found))
    print(f"\n{len(found)} site(s)")
