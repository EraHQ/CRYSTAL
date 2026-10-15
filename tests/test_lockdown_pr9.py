"""Lockdown PR-9 (B8, 2026-10-15): dependencies are pinned, floored and hashed.

Structural pins over the files the image is built from. Each one fails
the moment someone undoes the control it names:

- every `FROM` in every Dockerfile is pinned by digest (B8-3);
- the advisory floors are declared in pyproject.toml (B8-4);
- the image lock exists, is hash-complete, carries the CPU torch build
  and no CUDA wheels, and covers every package pyproject declares (B8-1);
- the old dev-venv freeze is gone (B8-2).
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "requirements.linux.lock"

DOCKERFILES = [
    ROOT / "Dockerfile",
    ROOT / "deploy" / "inspector" / "Dockerfile",
    ROOT / "deploy" / "searxng" / "Dockerfile",
]

FLOORS = {
    "starlette": "0.47.2",
    "h11": "0.16.0",
    "pdfminer.six": "20251107",
    "torch": "2.6",
}


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def test_every_base_image_is_pinned_by_digest():
    for df in DOCKERFILES:
        froms = [l for l in df.read_text(encoding="utf-8").splitlines() if l.startswith("FROM ")]
        assert froms, df
        for line in froms:
            ref = line.split()[1]
            assert re.search(r"@sha256:[0-9a-f]{64}$", ref), f"{df.name}: {line}"
            assert ":latest@" not in ref and not ref.startswith("searxng/searxng:latest"), line


def test_pyproject_declares_the_advisory_floors():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for name, floor in FLOORS.items():
        assert f'"{name}>={floor}"' in text, f"{name}>={floor} missing from pyproject.toml"
    # The decoder extra's old 2.2 floor is gone, not merely joined by a new one.
    assert '"torch>=2.2"' not in text


def test_the_image_lock_is_hash_complete_and_cpu_only():
    assert LOCK.exists(), "requirements.linux.lock missing (recipe: deploy/requirements-image.in)"
    text = LOCK.read_text(encoding="utf-8")
    assert "--extra-index-url https://download.pytorch.org/whl/cpu" in text
    pins = re.findall(r"^([A-Za-z0-9_.\-]+)==([^ \\\n]+)", text, flags=re.M)
    assert len(pins) >= 100, len(pins)
    names = {_norm(n) for n, _ in pins}
    versions = dict((_norm(n), v) for n, v in pins)
    assert versions["torch"].endswith("+cpu"), versions["torch"]
    assert not [n for n in names if n.startswith("nvidia-")]
    # Every pinned line is followed by at least one hash before the next pin.
    blocks = re.split(r"^(?=[A-Za-z0-9_.\-]+==)", text, flags=re.M)[1:]
    unhashed = [b.splitlines()[0] for b in blocks if "--hash=sha256:" not in b]
    assert unhashed == [], unhashed
    for name, floor in FLOORS.items():
        assert _norm(name) in names, f"{name} not in the lock"


def test_the_lock_covers_every_declared_dependency():
    tomllib = pytest.importorskip("tomllib")
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = list(data["project"]["dependencies"])
    for extra in ("embeddings", "sqlite-vec", "qdrant", "gcp"):
        declared += data["project"]["optional-dependencies"][extra]
    declared += [
        l.strip() for l in (ROOT / "deploy" / "requirements-image.in").read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.startswith("#")
    ]
    wanted = {_norm(re.split(r"[<>=!~\[; ]", d.strip(), 1)[0]) for d in declared}
    locked = {_norm(n) for n, _ in re.findall(r"^([A-Za-z0-9_.\-]+)==([^ \\\n]+)", LOCK.read_text(encoding="utf-8"), flags=re.M)}
    missing = sorted(wanted - locked)
    assert missing == [], missing


def test_the_dev_venv_freeze_is_gone():
    assert not (ROOT / "requirements.lock").exists()
    guide = (ROOT / "docs" / "DEPLOYMENT_GUIDE.md").read_text(encoding="utf-8")
    assert "pip freeze > requirements.lock" not in guide
    assert "requirements.linux.lock" in guide


def test_the_dockerfile_installs_from_the_lock_in_hash_mode():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "--require-hashes" in text and "--only-binary :all:" in text
    assert "pip install --no-cache-dir --no-deps ." in text
    assert "pip check" in text
    # Nothing in the image is resolved fresh any more.
    assert 'pip install --no-cache-dir "sentence-transformers' not in text
    assert 'pip install --no-cache-dir "playwright' not in text
    assert "pip install --no-cache-dir torch" not in text
