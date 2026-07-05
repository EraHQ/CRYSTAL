"""Container-backed disposable workspace — Phase 2 (2026-07-03).

Implements the DisposableEnvironment protocol (disposable.py) on an OCI
container runtime. RUNTIME-AGNOSTIC by design (ratified 2026-07-03: Docker
works everywhere the product ships — Windows/Mac via Docker Desktop + WSL — but
Docker alone is lock-in): the backend drives the CLI-compatible subset both
runtimes share, auto-detecting `podman` first (daemonless, smaller privilege
surface) then `docker`, with CC_CONTAINER_RUNTIME to force either (or a
full path to a compatible binary).

Shape: provision seeds a host temp dir (the SAME seed_workspace as the
local backend, so semantics never drift) and starts a long-lived container
with that dir bind-mounted at /workspace. The worker keeps operating on
handle.root from the HOST (harvest stays host-side); commands that must run
INSIDE the box go through exec_in_workspace(). Teardown force-removes the
container and deletes the host dir — unconditional and idempotent, like
every disposable teardown.

Network: the disposable box is default-ALLOW (design §4 — disposability is
the safety argument; output is reviewed on the way out). network="none"
tightens to no network for operators who want it. Hosted-plane segmentation
(box cannot reach the API plane / other tenants) is a Phase 3 provision-time
property, not a per-command check here.

Resource limits: memory_mb / cpus map to --memory / --cpus when set —
kernel-enforced caps that complement (not replace) the runner's wall-clock
deadline + cost budget. extra_run_args is the escape hatch for host quirks
(e.g. --runtime runc where crun/cgroups misbehave) and is where the GPU
tier's --gpus/--device flags will land.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from .disposable import SeedMode, WorkspaceHandle, seed_workspace
from .sandbox import SandboxResult


def detect_container_runtime() -> Optional[str]:
    """The runtime binary to drive, or None when no runtime exists.

    Precedence: CC_CONTAINER_RUNTIME (a name or full path — the operator's
    explicit choice always wins) > podman (preferred: daemonless, rootless
    by default) > docker. Returns something `subprocess` can exec.
    """
    forced = os.environ.get("CC_CONTAINER_RUNTIME", "").strip()
    if forced:
        found = shutil.which(forced)
        return found  # None when the forced runtime isn't actually present
    for candidate in ("podman", "docker"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def _gpu_args(runtime: str) -> list[str]:
    """Per-runtime GPU exposure flags (GPU tier, 2026-07-03).

    docker: `--gpus all` (NVIDIA Container Toolkit's native flag).
    podman: `--device nvidia.com/gpu=all` (CDI — podman's supported route;
    podman does not implement docker's --gpus). The runtime string may be
    a full path, so match on the basename.
    """
    base = Path(runtime).name.lower()
    if "podman" in base:
        return ["--device", "nvidia.com/gpu=all"]
    return ["--gpus", "all"]


def _run_argv(
    runtime: str,
    name: str,
    workspace: Path,
    image: str,
    *,
    network: str = "default",
    memory_mb: Optional[int] = None,
    cpus: Optional[float] = None,
    gpu: bool = False,
    extra_run_args: Optional[list[str]] = None,
) -> list[str]:
    """The `run` argv for the long-lived workspace container. Pure so the
    flag surface is testable without a runtime present."""
    argv = [
        runtime, "run", "-d",
        "--name", name,
        "-v", f"{workspace}:/workspace",
        "-w", "/workspace",
    ]
    if network == "none":
        argv += ["--network", "none"]
    if memory_mb is not None:
        argv += ["--memory", f"{memory_mb}m"]
    if cpus is not None:
        argv += ["--cpus", str(cpus)]
    if gpu:
        argv += _gpu_args(runtime)
    if extra_run_args:
        argv += list(extra_run_args)
    # Keep the box alive until teardown. A shell loop rather than
    # `sleep infinity` so minimal images (busybox) work too.
    argv += [image, "/bin/sh", "-c", "while :; do sleep 3600; done"]
    return argv


class ContainerWorkspaceEnv:
    """DisposableEnvironment backed by an OCI container (podman or docker).

    Same lifecycle + seeding as LocalWorkspaceEnv; the container adds a
    kernel-isolated place to EXECUTE (exec_in_workspace) with its own
    filesystem view (the workspace bind mount), network posture, and
    optional memory/cpu caps. run_disposable_task() drives it unchanged.
    """

    def __init__(
        self,
        working_dir: Path,
        *,
        image: str,
        seed_mode: SeedMode = SeedMode.COPY,
        seed_path: Optional[Path] = None,
        runtime: Optional[str] = None,
        network: str = "default",
        memory_mb: Optional[int] = None,
        cpus: Optional[float] = None,
        gpu: bool = False,
        extra_run_args: Optional[list[str]] = None,
        profile: Optional[str] = None,
    ) -> None:
        self._working_dir = Path(working_dir)
        self._image = image
        self._seed_mode = seed_mode
        self._seed_path = Path(seed_path) if seed_path else None
        self._runtime = runtime or detect_container_runtime()
        self._network = network
        self._memory_mb = memory_mb
        self._cpus = cpus
        self._gpu = bool(gpu)
        self._extra_run_args = list(extra_run_args or [])
        # Profile names the box's capability class; defaults follow the
        # design doc taxonomy (cpu_untrusted / gpu).
        self._profile = profile or ("gpu" if self._gpu else "cpu_untrusted")

    @property
    def runtime(self) -> Optional[str]:
        return self._runtime

    async def provision(self) -> WorkspaceHandle:
        if not self._runtime:
            raise RuntimeError(
                "no container runtime found: install podman or docker, or "
                "set CC_CONTAINER_RUNTIME"
            )
        root = Path(tempfile.mkdtemp(prefix="crys-disposable-"))
        seed_workspace(root, self._seed_mode, self._working_dir, self._seed_path)
        name = f"crys-disposable-{uuid.uuid4().hex[:12]}"
        argv = _run_argv(
            self._runtime, name, root, self._image,
            network=self._network,
            memory_mb=self._memory_mb,
            cpus=self._cpus,
            gpu=self._gpu,
            extra_run_args=self._extra_run_args,
        )
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            # Fail-safe: never leave the host dir behind on a failed start.
            shutil.rmtree(root, ignore_errors=True)
            raise RuntimeError(
                f"container start failed ({proc.returncode}): "
                f"{(proc.stderr or proc.stdout).strip()[-500:]}"
            )
        return WorkspaceHandle(
            root=root, seed_mode=self._seed_mode, profile=self._profile,
            meta={"container": name, "runtime": self._runtime},
        )

    def exec_in_workspace(
        self, handle: WorkspaceHandle, command: str, timeout: int,
    ) -> SandboxResult:
        """Run one shell command INSIDE the box (cwd /workspace). Returns
        the same SandboxResult shape as the E1c chokepoint so callers treat
        both execution paths identically."""
        name = handle.meta.get("container", "")
        runtime = handle.meta.get("runtime") or self._runtime
        if not name or not runtime:
            return SandboxResult(
                exit_code=1, output="no live container on this handle",
                backend="container",
            )
        try:
            proc = subprocess.run(
                [runtime, "exec", name, "/bin/sh", "-c", command],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                exit_code=124, output=f"command timed out after {timeout}s",
                timed_out=True, backend="container",
            )
        combined = ((proc.stdout or "") + (proc.stderr or "")).strip()
        return SandboxResult(
            exit_code=proc.returncode, output=combined, backend="container",
        )

    async def teardown(self, handle: WorkspaceHandle) -> None:
        # Unconditional + idempotent: force-remove the container (rm -f is a
        # no-op when it's already gone), then delete the host dir.
        name = handle.meta.get("container", "")
        runtime = handle.meta.get("runtime") or self._runtime
        if name and runtime:
            subprocess.run(
                [runtime, "rm", "-f", name],
                capture_output=True, text=True, timeout=60,
            )
        shutil.rmtree(handle.root, ignore_errors=True)
