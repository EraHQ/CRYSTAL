"""Phase 3 slice 4: the runner-image entrypoint (2026-07-03).

Drives CRYS/runner/entrypoint.py as a real subprocess — exactly
how the job container runs it — against file:// refs standing in for the
production signed URLs (same code path, different transport). Proves the
whole in-box lifecycle with no GCS and no container: seed fetch+unpack
(fail-safe to empty), task exec in the workspace with env passthrough,
harvest upload EVEN ON TASK FAILURE, exit-code propagation.

The interop test packs the seed with the PLANE's pack_seed and unpacks
the harvest with the PLANE's unpack_harvest — the two sides of the wire
proven against each other.

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "CRYS"))

from crystal_code.disposable import SeedMode
from crystal_code.remote_env import pack_seed, unpack_harvest

ENTRYPOINT = (
    Path(__file__).parent.parent / "CRYS" / "runner" / "entrypoint.py"
)


def _run(cmd: list[str], *, workspace: Path, seed: Path | None,
         harvest: Path | None) -> subprocess.CompletedProcess:
    env = {
        "CRYS_WORKSPACE": str(workspace),
        "CRYS_TASK_KEY": "ck_task_test",
        "PATH": "/usr/bin:/bin",
        "SYSTEMROOT": r"C:\Windows",   # subprocess on Windows needs it
    }
    if seed is not None:
        env["CRYS_SEED_REF"] = seed.as_uri()
    if harvest is not None:
        env["CRYS_HARVEST_REF"] = harvest.as_uri()
    return subprocess.run(
        [sys.executable, str(ENTRYPOINT), *cmd],
        env=env, capture_output=True, text=True, timeout=60,
    )


def test_full_lifecycle_plane_interop(tmp_path):
    """Plane packs the seed → box unpacks it, runs the task, uploads the
    harvest → plane unpacks the harvest. Both directions of the wire."""
    src = tmp_path / "proj"
    src.mkdir()
    (src / "input.txt").write_text("42")
    seed_file = tmp_path / "seed.tar.gz"
    seed_file.write_bytes(pack_seed(SeedMode.COPY, working_dir=src))

    ws = tmp_path / "ws"
    harvest_file = tmp_path / "store" / "harvest.tar.gz"

    code = (
        "import pathlib;"
        "v = pathlib.Path('input.txt').read_text();"
        "pathlib.Path('harvest/answer.txt').write_text(v + '!')"
    )
    proc = _run([sys.executable, "-c", code],
                workspace=ws, seed=seed_file, harvest=harvest_file)
    assert proc.returncode == 0, proc.stderr
    assert harvest_file.exists()

    out = tmp_path / "unpacked"
    unpack_harvest(harvest_file.read_bytes(), out)
    assert (out / "answer.txt").read_text() == "42!"


def test_task_failure_propagates_but_harvest_still_uploads(tmp_path):
    ws = tmp_path / "ws"
    harvest_file = tmp_path / "h.tar.gz"
    code = (
        "import pathlib, sys;"
        "pathlib.Path('harvest/partial.log').write_text('got this far');"
        "sys.exit(3)"
    )
    proc = _run([sys.executable, "-c", code],
                workspace=ws, seed=None, harvest=harvest_file)
    assert proc.returncode == 3          # the task's verdict, verbatim
    assert harvest_file.exists()         # partial output beat silence

    out = tmp_path / "u"
    unpack_harvest(harvest_file.read_bytes(), out)
    assert (out / "partial.log").read_text() == "got this far"


def test_missing_seed_fails_safe_to_empty_workspace(tmp_path):
    ws = tmp_path / "ws"
    missing = tmp_path / "nope.tar.gz"    # never created
    code = (
        "import pathlib;"
        "n = len([p for p in pathlib.Path('.').iterdir()"
        " if p.name != 'harvest']);"
        "pathlib.Path('harvest/count.txt').write_text(str(n))"
    )
    proc = _run([sys.executable, "-c", code],
                workspace=ws, seed=missing, harvest=tmp_path / "h.tar.gz")
    assert proc.returncode == 0, proc.stderr
    assert "seed fetch failed" in proc.stderr
    out = tmp_path / "u"
    unpack_harvest((tmp_path / "h.tar.gz").read_bytes(), out)
    assert (out / "count.txt").read_text() == "0"   # box was empty


def test_no_harvest_produced_means_no_upload(tmp_path):
    ws = tmp_path / "ws"
    harvest_file = tmp_path / "h.tar.gz"
    proc = _run([sys.executable, "-c", "pass"],
                workspace=ws, seed=None, harvest=harvest_file)
    assert proc.returncode == 0
    assert not harvest_file.exists()
    assert "no harvest produced" in proc.stderr


def test_task_key_rides_the_environment(tmp_path):
    ws = tmp_path / "ws"
    harvest_file = tmp_path / "h.tar.gz"
    code = (
        "import os, pathlib;"
        "pathlib.Path('harvest/key.txt')"
        ".write_text(os.environ.get('CRYS_TASK_KEY', 'MISSING'))"
    )
    proc = _run([sys.executable, "-c", code],
                workspace=ws, seed=None, harvest=harvest_file)
    assert proc.returncode == 0, proc.stderr
    out = tmp_path / "u"
    unpack_harvest(harvest_file.read_bytes(), out)
    assert (out / "key.txt").read_text() == "ck_task_test"
