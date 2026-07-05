#!/usr/bin/env python3
"""The runner-image entrypoint (Phase 3 slice 4, 2026-07-03).

This script IS the job container's PID-1 wrapper: it runs inside the
disposable box, so it is deliberately stdlib-only with zero repo imports
— the runner image needs python3 and nothing else. Lifecycle:

    1. fetch the seed tarball from CRYS_SEED_REF and unpack it into the
       workspace (missing/empty ref = empty workspace, fail-safe — the
       same semantics seed_workspace gives local boxes);
    2. run the task command IN the workspace, with the box's environment
       passed through (CRYS_TASK_KEY rides here — the task's only
       credential, per ratified G3);
    3. pack the harvest dir (workspace/harvest by convention) and upload
       it to CRYS_HARVEST_REF — even when the task FAILED, because a
       partial deliverable plus logs is worth more than silence;
    4. exit with the task command's own exit code, so the job's
       succeeded/failed state reflects the task, not the plumbing.

Refs are file:// or https://. file:// exists so the WHOLE lifecycle is
provable locally (pytest drives this script as a subprocess against a
temp-dir "store") and with the Phase 2 container backend; https:// is
the production shape (per-task signed URLs, GET for seed, PUT for
harvest). Same code path either way — only the transport differs.

Env contract:
    CRYS_SEED_REF      where the seed tarball lives (optional)
    CRYS_HARVEST_REF   where to PUT the harvest tarball (optional)
    CRYS_WORKSPACE     working dir (default /workspace)
    CRYS_HARVEST_DIR   deliverable dir (default $CRYS_WORKSPACE/harvest)
    CRYS_TASK_KEY      passed through untouched — the task's credential

Usage: entrypoint.py <command> [args...]
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname


def _file_ref_path(ref: str) -> Path:
    """file:// ref -> local Path, cross-platform. url2pathname handles the
    Windows drive-letter form (file:///C:/x -> C:\\x) that naive prefix
    stripping mangles into an invalid leading-slash path."""
    return Path(url2pathname(urlsplit(ref).path))


def _fetch(ref: str) -> bytes | None:
    """GET the bytes behind a ref; None on any failure (fail-safe seed)."""
    try:
        if ref.startswith("file://"):
            return _file_ref_path(ref).read_bytes()
        with urllib.request.urlopen(ref, timeout=60) as resp:  # noqa: S310
            return resp.read()
    except Exception as e:  # noqa: BLE001 — empty box beats a dead box
        print(f"[crys-runner] seed fetch failed ({e}); starting empty",
              file=sys.stderr)
        return None


def _put(ref: str, data: bytes) -> bool:
    """PUT bytes to a ref. file:// writes; https:// is an HTTP PUT (the
    shape GCS signed upload URLs expect)."""
    try:
        if ref.startswith("file://"):
            dest = _file_ref_path(ref)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            return True
        req = urllib.request.Request(  # noqa: S310
            ref, data=data, method="PUT",
            headers={"Content-Type": "application/gzip"},
        )
        with urllib.request.urlopen(req, timeout=120):  # noqa: S310
            return True
    except Exception as e:  # noqa: BLE001
        print(f"[crys-runner] harvest upload failed: {e}", file=sys.stderr)
        return False


def _unpack_seed(data: bytes, workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
        # data filter: the plane packed this, but rigor costs nothing.
        tar.extractall(workspace, filter="data")


def _pack_dir(root: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for item in sorted(root.rglob("*")):
            tar.add(item, arcname=str(item.relative_to(root)))
    return buf.getvalue()


def main(argv: list[str]) -> int:
    if not argv:
        print("[crys-runner] usage: entrypoint.py <command> [args...]",
              file=sys.stderr)
        return 2

    workspace = Path(os.environ.get("CRYS_WORKSPACE", "/workspace"))
    harvest_dir = Path(
        os.environ.get("CRYS_HARVEST_DIR", str(workspace / "harvest"))
    )
    seed_ref = os.environ.get("CRYS_SEED_REF", "")
    harvest_ref = os.environ.get("CRYS_HARVEST_REF", "")

    # 1. Seed.
    workspace.mkdir(parents=True, exist_ok=True)
    if seed_ref:
        data = _fetch(seed_ref)
        if data:
            _unpack_seed(data, workspace)
    harvest_dir.mkdir(parents=True, exist_ok=True)

    # 2. The task. Environment passes through whole — CRYS_TASK_KEY is in
    # it, and the task talks to Era's public proxy with it (G3).
    proc = subprocess.run(argv, cwd=workspace)
    code = proc.returncode

    # 3. Harvest — even on failure (partial output + logs beat silence).
    if harvest_ref:
        produced = any(harvest_dir.rglob("*"))
        if produced:
            ok = _put(harvest_ref, _pack_dir(harvest_dir))
            print(f"[crys-runner] harvest upload: {'ok' if ok else 'FAILED'}",
                  file=sys.stderr)
        else:
            print("[crys-runner] no harvest produced", file=sys.stderr)

    # 4. The task's exit code IS the job's verdict.
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
