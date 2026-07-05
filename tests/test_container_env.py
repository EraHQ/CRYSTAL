"""Container disposable workspace — Phase 2 (2026-07-03).

Runtime-agnostic (ratified 2026-07-03): the backend drives the CLI subset
podman and docker share, prefers podman, falls back to docker, and honors
CC_CONTAINER_RUNTIME. Pure tests (detection precedence, run-argv flag
surface) run everywhere; live tests run the real lifecycle against a local
busybox image and SKIP where no runtime invocation works (mirrors the
bubblewrap-test pattern: isolation asserts run where the backend exists).

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import pytest

_CA = Path(__file__).resolve().parents[1] / "CRYS"
if str(_CA) not in sys.path:
    sys.path.insert(0, str(_CA))

from crystal_code.container_env import (  # noqa: E402
    ContainerWorkspaceEnv,
    detect_container_runtime,
    _run_argv,
)
from crystal_code.disposable import (  # noqa: E402
    DisposableLimits,
    SeedMode,
    TaskOutcome,
    run_disposable_task,
)


# --- pure: runtime detection precedence -------------------------------------

def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("CC_CONTAINER_RUNTIME", "docker")
    monkeypatch.setattr(
        "crystal_code.container_env.shutil.which",
        lambda name: f"/usr/bin/{name}" if name == "docker" else None,
    )
    assert detect_container_runtime() == "/usr/bin/docker"


def test_forced_but_missing_runtime_is_none(monkeypatch):
    monkeypatch.setenv("CC_CONTAINER_RUNTIME", "podman")
    monkeypatch.setattr(
        "crystal_code.container_env.shutil.which", lambda name: None,
    )
    assert detect_container_runtime() is None  # explicit choice, fail loud


def test_podman_preferred_over_docker(monkeypatch):
    monkeypatch.delenv("CC_CONTAINER_RUNTIME", raising=False)
    monkeypatch.setattr(
        "crystal_code.container_env.shutil.which",
        lambda name: f"/usr/bin/{name}",  # both present
    )
    assert detect_container_runtime() == "/usr/bin/podman"


def test_docker_fallback(monkeypatch):
    monkeypatch.delenv("CC_CONTAINER_RUNTIME", raising=False)
    monkeypatch.setattr(
        "crystal_code.container_env.shutil.which",
        lambda name: "/usr/bin/docker" if name == "docker" else None,
    )
    assert detect_container_runtime() == "/usr/bin/docker"


def test_no_runtime_is_none(monkeypatch):
    monkeypatch.delenv("CC_CONTAINER_RUNTIME", raising=False)
    monkeypatch.setattr(
        "crystal_code.container_env.shutil.which", lambda name: None,
    )
    assert detect_container_runtime() is None


# --- pure: run argv flag surface ---------------------------------------------

def test_run_argv_mounts_workspace_and_defaults():
    ws = Path("/tmp/ws")
    argv = _run_argv("podman", "box1", ws, "img:latest")
    assert argv[:3] == ["podman", "run", "-d"]
    # The mount arg is the HOST path in the platform's native rendering
    # (Windows Docker Desktop wants backslashed host paths) + the fixed
    # container-side /workspace — so assert the contract, not one
    # platform's spelling of it.
    assert "-v" in argv and f"{ws}:/workspace" in argv
    assert ["-w", "/workspace"] == argv[argv.index("-w"):argv.index("-w") + 2]
    # default network = allow: no --network flag at all
    assert "--network" not in argv
    assert "--memory" not in argv and "--cpus" not in argv
    assert argv[-4] == "img:latest"  # then /bin/sh -c <loop>


def test_run_argv_network_none_and_limits():
    argv = _run_argv(
        "docker", "box2", Path("/tmp/ws"), "img",
        network="none", memory_mb=512, cpus=1.5,
    )
    i = argv.index("--network")
    assert argv[i + 1] == "none"
    assert argv[argv.index("--memory") + 1] == "512m"
    assert argv[argv.index("--cpus") + 1] == "1.5"


def test_run_argv_extra_args_precede_image():
    argv = _run_argv(
        "podman", "box3", Path("/tmp/ws"), "img",
        extra_run_args=["--runtime", "runc"],
    )
    assert argv.index("--runtime") < argv.index("img")


# --- live: the real lifecycle (skips where no runtime invocation works) ------

_IMAGE = "crys-test-base:local"


def _build_local_image(runtime: str) -> bool:
    """Import a minimal busybox rootfs as _IMAGE so live tests need no
    registry access. Returns False when the pieces aren't available."""
    busybox = Path("/bin/busybox")
    if not busybox.exists():
        return False
    have = subprocess.run(
        [runtime, "image", "exists", _IMAGE],
        capture_output=True,
    )
    if have.returncode == 0:
        return True
    with tempfile.TemporaryDirectory() as td:
        rootfs = Path(td) / "rootfs"
        (rootfs / "bin").mkdir(parents=True)
        import shutil as _sh
        _sh.copy2(busybox, rootfs / "bin" / "busybox")
        for cmd in ("sh", "echo", "cat", "ls", "touch", "sleep"):
            (rootfs / "bin" / cmd).symlink_to("busybox")
        tar = Path(td) / "rootfs.tar"
        with tarfile.open(tar, "w") as tf:
            for p in rootfs.rglob("*"):
                tf.add(p, arcname=str(p.relative_to(rootfs)))
        imp = subprocess.run(
            [runtime, "import", str(tar), _IMAGE], capture_output=True,
        )
        return imp.returncode == 0


def _usable_extra_args(runtime: str) -> "list[str] | None":
    """Find a `run` invocation that works on THIS host: default first, then
    --runtime runc (nested-container environments where crun/cgroups
    misbehave). None when nothing works -> live tests skip."""
    for extra in ([], ["--runtime", "runc"]):
        probe = subprocess.run(
            [runtime, "run", "--rm", *extra, _IMAGE, "/bin/echo", "ok"],
            capture_output=True, text=True, timeout=120,
        )
        if probe.returncode == 0 and "ok" in probe.stdout:
            return extra
    return None


def _live_setup() -> "tuple[str, list[str]] | None":
    runtime = detect_container_runtime()
    if not runtime:
        return None
    if not _build_local_image(runtime):
        return None
    extra = _usable_extra_args(runtime)
    if extra is None:
        return None
    return runtime, extra


_LIVE = _live_setup()

live = pytest.mark.skipif(
    _LIVE is None,
    reason="no usable container runtime on this host (podman/docker + a "
    "working run invocation); live lifecycle is covered where one exists",
)


def _env(tmp_path, **kw):
    runtime, extra = _LIVE
    return ContainerWorkspaceEnv(
        tmp_path, image=_IMAGE, runtime=runtime,
        extra_run_args=extra, **kw,
    )


@live
async def test_seed_visible_inside_and_writes_visible_outside(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "seeded.txt").write_text("from-host")

    async def worker(handle, stop):
        env = worker.env
        r = env.exec_in_workspace(handle, "cat /workspace/seeded.txt", 60)
        wrote = env.exec_in_workspace(
            handle, "echo from-container > /workspace/out.txt", 60,
        )
        return {
            "read": r.output, "read_ok": not r.is_error,
            "write_ok": not wrote.is_error,
            "host_sees": (handle.root / "out.txt").read_text().strip(),
        }

    env = _env(proj)
    worker.env = env
    result = await run_disposable_task(env, worker)
    assert result.outcome == TaskOutcome.COMPLETED
    assert result.harvest["read_ok"] and result.harvest["read"] == "from-host"
    assert result.harvest["write_ok"]
    assert result.harvest["host_sees"] == "from-container"


@live
async def test_teardown_removes_container_and_dir(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    captured = {}

    async def worker(handle, stop):
        captured["root"] = handle.root
        captured["name"] = handle.meta["container"]
        return None

    env = _env(proj)
    await run_disposable_task(env, worker)

    runtime, _ = _LIVE
    exists = subprocess.run(
        [runtime, "container", "exists", captured["name"]],
        capture_output=True,
    )
    assert exists.returncode != 0  # container gone
    assert not captured["root"].exists()  # host dir gone


@live
async def test_teardown_runs_even_on_worker_crash(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    captured = {}

    async def worker(handle, stop):
        captured["name"] = handle.meta["container"]
        raise RuntimeError("boom")

    env = _env(proj)
    result = await run_disposable_task(env, worker)
    assert result.outcome == TaskOutcome.FAILED

    runtime, _ = _LIVE
    exists = subprocess.run(
        [runtime, "container", "exists", captured["name"]],
        capture_output=True,
    )
    assert exists.returncode != 0  # torn down despite the crash


@live
async def test_exec_reports_exit_code_and_timeout_shape(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()

    async def worker(handle, stop):
        env = worker.env
        bad = env.exec_in_workspace(handle, "ls /does-not-exist", 60)
        return {"bad_is_error": bad.is_error, "backend": bad.backend}

    env = _env(proj)
    worker.env = env
    result = await run_disposable_task(env, worker)
    assert result.harvest["bad_is_error"] is True
    assert result.harvest["backend"] == "container"


@live
async def test_deadline_kills_and_tears_down_container(tmp_path):
    import asyncio

    proj = tmp_path / "proj"
    proj.mkdir()

    async def slow_worker(handle, stop):
        # Loops until the deadline trips (never finishes on its own). The
        # deadline is generous enough to cover container provision — the
        # whole-task wall clock INCLUDES provision by design, which is why
        # a tiny deadline would trip before the worker even starts.
        while True:
            if stop() is not None:
                return None
            await asyncio.sleep(0.02)

    env = _env(proj)
    result = await run_disposable_task(
        env, slow_worker, limits=DisposableLimits(deadline_seconds=8),
    )
    assert result.outcome == TaskOutcome.DEADLINE_EXCEEDED

    # Teardown holds regardless of whether the trip happened pre-worker or
    # mid-worker: NO disposable container may remain.
    runtime, _ = _LIVE
    ps = subprocess.run(
        [runtime, "ps", "-a", "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    assert "crys-disposable-" not in (ps.stdout or "")


# --- GPU tier flag surface (2026-07-03) --------------------------------------

def test_gpu_flag_docker_emits_gpus_all():
    argv = _run_argv("docker", "g1", Path("/tmp/ws"), "img", gpu=True)
    i = argv.index("--gpus")
    assert argv[i + 1] == "all"
    assert "--device" not in argv


def test_gpu_flag_podman_emits_cdi_device():
    """podman has no --gpus; the CDI device route is its supported path.
    The runtime may be a full path — matched on basename."""
    argv = _run_argv("/usr/bin/podman", "g2", Path("/tmp/ws"), "img", gpu=True)
    i = argv.index("--device")
    assert argv[i + 1] == "nvidia.com/gpu=all"
    assert "--gpus" not in argv


def test_no_gpu_flag_by_default():
    argv = _run_argv("docker", "g3", Path("/tmp/ws"), "img")
    assert "--gpus" not in argv and "--device" not in argv


def test_gpu_env_defaults_profile_to_gpu(tmp_path):
    env = ContainerWorkspaceEnv(tmp_path, image="img", gpu=True)
    assert env._profile == "gpu"
    cpu_env = ContainerWorkspaceEnv(tmp_path, image="img")
    assert cpu_env._profile == "cpu_untrusted"
