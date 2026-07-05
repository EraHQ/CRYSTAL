"""Execution sandbox — the single chokepoint for model-influenced commands.

E1c (security review): the agent runs shell/verify/hook commands with the
user's full OS permissions and network, uncontained. The fix is a sandbox,
but a sandbox is only trustworthy if EVERY model-influenced execution
routes through ONE seam — otherwise a command smuggled through the verify
path bypasses it. This module is that seam.

Phased build (see docs/E1C_SANDBOX_DESIGN.md):
  * Phase 0 (this file, first cut): the chokepoint + the passthrough
    backend. No isolation yet — behavior is byte-identical to the prior
    direct subprocess calls. The point is that shell.py, verify.py, and the
    guard hooks now all call sandbox_run(), so Phase 1 can insert the
    bubblewrap backend in ONE place.
  * Phase 1: the bubblewrap `cpu` backend (filesystem scope, net deny,
    cgroup limits) — closes E1c for local/self-host.
  * Phase 2: the OCI container backend (`gpu` / `cpu-untrusted` profiles)
    for 3D rendering and hosted tiers.

The execution-model split from E1b is preserved: interactive
human-approved commands may use shell features; auto-approved commands run
as an argv vector with no shell. That decision stays in the CALLER (it owns
the approval context); the sandbox honors it via allow_shell_features.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


# Secret-shaped env var name fragments scrubbed from every child process.
# (Kept here so ALL execution paths share one definition, not just shell.)
_SENSITIVE_ENV = (
    "API_KEY", "APIKEY", "SECRET", "TOKEN", "PASSWORD", "PASSWD",
    "CREDENTIAL", "PRIVATE_KEY", "AUTH",
)

_MAX_OUTPUT_CHARS = 4000

logger = logging.getLogger(__name__)

_WARNED_NO_SANDBOX = False


def scrubbed_env() -> dict[str, str]:
    """The parent environment minus anything secret-shaped."""
    return {
        k: v for k, v in os.environ.items()
        if not any(s in k.upper() for s in _SENSITIVE_ENV)
    }


class SandboxProfile(str, Enum):
    """Capability tier for an execution. Phase 0 treats them all as the
    passthrough backend; later phases map them to real backends:
      CPU            -> bubblewrap (Phase 1) — the default.
      CPU_UNTRUSTED  -> container/microVM (Phase 2) — hosted multi-tenant.
      GPU            -> container + GPU device (Phase 2) — 3D/render/ML.
    """
    CPU = "cpu"
    CPU_UNTRUSTED = "cpu_untrusted"
    GPU = "gpu"


@dataclass
class SandboxResult:
    """The outcome of one sandboxed execution."""
    exit_code: int
    output: str          # combined stdout+stderr, tail-truncated
    timed_out: bool = False
    backend: str = "passthrough"

    @property
    def is_error(self) -> bool:
        return self.timed_out or self.exit_code != 0


def _truncate(text: str) -> str:
    text = (text or "").strip()
    if len(text) > _MAX_OUTPUT_CHARS:
        return (
            f"... (output truncated to the last {_MAX_OUTPUT_CHARS} chars)\n"
            + text[-_MAX_OUTPUT_CHARS:]
        )
    return text


# ---------------------------------------------------------------------------
# Backend selection. Phase 0 has ONE backend (passthrough). The selector is
# here so Phase 1/2 add branches without touching callers. The env override
# CC_SANDBOX lets an operator force a backend (or 'off' for the explicit,
# loud, never-default escape hatch documented in the design).
# ---------------------------------------------------------------------------

def _selected_backend() -> str:
    """Which backend to use, from CC_SANDBOX (default 'auto').

    auto        - Phase 1 default: use bubblewrap when it's available and
                  usable; on a platform without it (macOS/Windows dev) fall
                  back to passthrough WITH A WARNING rather than blocking dev.
    bubblewrap  - force the bwrap backend; error out if unavailable.
    off / passthrough - the explicit, loud, never-default escape hatch:
                  run uncontained. For local debugging only.
    """
    return (os.environ.get("CC_SANDBOX", "") or "auto").strip().lower()


_BWRAP_CHECKED: Optional[bool] = None


def _bwrap_usable() -> bool:
    """True if bubblewrap is installed AND can actually create namespaces
    here (containers sometimes ship bwrap but block unprivileged user
    namespaces). Probed once and cached."""
    global _BWRAP_CHECKED
    if _BWRAP_CHECKED is not None:
        return _BWRAP_CHECKED
    exe = shutil.which("bwrap")
    if not exe:
        _BWRAP_CHECKED = False
        return False
    try:
        probe = subprocess.run(
            [exe, "--ro-bind", "/", "/", "--unshare-all", "true"],
            capture_output=True, timeout=10,
        )
        _BWRAP_CHECKED = probe.returncode == 0
    except Exception:  # noqa: BLE001
        _BWRAP_CHECKED = False
    return _BWRAP_CHECKED


def _bwrap_argv(
    project_dir: Path, *, net_allowed: bool,
) -> list[str]:
    """Build the bubblewrap wrapper argv for the `cpu` profile.

    Scope: system dirs read-only (so tools/interpreters work), the project
    dir read-WRITE (the agent edits real files), a private /tmp and /dev,
    and all namespaces unshared. Network is unshared (denied) UNLESS
    net_allowed — which the caller sets True only for interactive
    human-approved commands (E1b/option-b trust split), False for
    auto-approved ones.
    """
    exe = shutil.which("bwrap") or "bwrap"
    argv = [exe]
    # Read-only system dirs (only bind those that exist on this host).
    for d in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc"):
        if Path(d).exists():
            argv += ["--ro-bind", d, d]
    # Private /tmp, a minimal /dev, and a private /proc — declared BEFORE the
    # project bind so that if the project happens to live under /tmp (e.g. a
    # scratch checkout), the project bind below overlays the tmpfs rather
    # than being masked by it. bwrap applies mounts in argv order.
    argv += ["--tmpfs", "/tmp", "--dev", "/dev", "--proc", "/proc"]
    # The project dir is read-write — the agent's real work happens here.
    # Bound LAST so it wins over any earlier mount covering the same path.
    argv += ["--bind", str(project_dir), str(project_dir)]
    # Working dir = project.
    argv += ["--chdir", str(project_dir)]
    # Namespace isolation. --unshare-all covers pid/ipc/uts/cgroup/user and
    # network; we then RE-share the network when it's allowed.
    argv += ["--unshare-all"]
    if net_allowed:
        argv += ["--share-net"]
    # Die with the parent so a killed agent doesn't leak sandboxed procs.
    argv += ["--die-with-parent"]
    return argv


def _run_bubblewrap(
    command: str,
    project_dir: Path,
    timeout: int,
    *,
    allow_shell_features: bool,
    env: Optional[dict[str, str]],
) -> SandboxResult:
    """Phase 1 `cpu` backend: run the command inside a bubblewrap sandbox.

    Filesystem is scoped to the project dir (read-write) plus read-only
    system dirs; everything else on the host is invisible. Network follows
    the option-b trust split: allowed for interactive human-approved
    commands, denied for auto-approved. The E1b shell/no-shell split is
    preserved INSIDE the sandbox.
    """
    # option (b): interactive (shell-features) => net allowed; auto => denied.
    net_allowed = allow_shell_features
    wrapper = _bwrap_argv(project_dir, net_allowed=net_allowed)

    inner_spec, use_shell = _build_argv_or_command(
        command, allow_shell_features=allow_shell_features,
    )
    if not use_shell and not inner_spec:
        return SandboxResult(exit_code=1, output="empty command",
                             backend="bubblewrap")
    # Compose the full argv. For shell mode we invoke a shell INSIDE the
    # sandbox; for no-shell mode we hand bwrap the argv vector directly.
    if use_shell:
        full = wrapper + ["/bin/sh", "-c", command]
    else:
        full = wrapper + list(inner_spec)
    try:
        proc = subprocess.run(
            full,
            cwd=str(project_dir),
            env=env if env is not None else scrubbed_env(),
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(
            exit_code=-1,
            output=f"command timed out after {timeout}s",
            timed_out=True, backend="bubblewrap",
        )
    combined = (proc.stdout or "") + (
        ("\n" + proc.stderr) if proc.stderr else ""
    )
    return SandboxResult(
        exit_code=proc.returncode,
        output=_truncate(combined) or "(no output)",
        backend="bubblewrap",
    )


def _build_argv_or_command(
    command: str, *, allow_shell_features: bool,
) -> tuple[object, bool]:
    """Return (spec, use_shell). spec is the raw string (shell) or an argv
    list (no-shell). Mirrors the E1b execution-model split."""
    if allow_shell_features:
        return command, True
    argv = shlex.split(command)
    return argv, False


def _run_passthrough(
    command: str,
    project_dir: Path,
    timeout: int,
    *,
    allow_shell_features: bool,
    env: Optional[dict[str, str]],
) -> SandboxResult:
    """Phase 0 backend: run directly (no isolation). Byte-for-byte the prior
    behavior of the shell/verify/hook exec paths, just centralized."""
    spec, use_shell = _build_argv_or_command(
        command, allow_shell_features=allow_shell_features,
    )
    if not use_shell and not spec:
        return SandboxResult(exit_code=1, output="empty command")
    try:
        proc = subprocess.run(
            spec,
            shell=use_shell,
            cwd=str(project_dir),
            env=env if env is not None else scrubbed_env(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(
            exit_code=-1,
            output=f"command timed out after {timeout}s",
            timed_out=True,
        )
    combined = (proc.stdout or "") + (
        ("\n" + proc.stderr) if proc.stderr else ""
    )
    return SandboxResult(
        exit_code=proc.returncode,
        output=_truncate(combined) or "(no output)",
    )


def sandbox_run(
    command: str,
    project_dir: Path,
    timeout: int,
    *,
    allow_shell_features: bool,
    profile: SandboxProfile = SandboxProfile.CPU,
    env: Optional[dict[str, str]] = None,
) -> SandboxResult:
    """THE execution chokepoint. Every model-influenced command runs here.

    command              the command string the model/config proposed.
    project_dir          the working directory (project scope).
    timeout              hard wall-clock cap in seconds.
    allow_shell_features True for interactive human-approved (shell=True),
                         False for auto-approved (argv, shell=False). The
                         caller owns this decision (it holds the approval
                         context); callers using False MUST have screened
                         the command for shell operators first.
    profile              capability tier (Phase 0: all -> passthrough).
    env                  optional explicit environment; defaults to the
                         scrubbed parent environment.

    Backend selection (CC_SANDBOX, default 'auto'):
      auto        -> bubblewrap if usable here, else passthrough + a one-time
                     warning (keeps macOS/Windows dev working).
      bubblewrap  -> force bubblewrap; if it is not usable, FAIL CLOSED
                     (refuse to run) rather than silently running uncontained.
      off/passthrough -> explicit escape hatch: run uncontained (local debug).

    Network follows option (b): the bubblewrap backend allows network for
    interactive human-approved commands (allow_shell_features=True) and
    denies it for auto-approved ones. Filesystem is always scoped to the
    project dir. The agent still reads/edits/writes the real project files
    and runs real commands — the sandbox scopes WHERE they reach, it does
    not remove the ability to act.
    """
    global _WARNED_NO_SANDBOX
    backend = _selected_backend()

    if backend in ("off", "passthrough"):
        return _run_passthrough(
            command, project_dir, timeout,
            allow_shell_features=allow_shell_features, env=env,
        )

    if backend == "bubblewrap":
        if not _bwrap_usable():
            # Explicitly requested but unavailable -> fail closed.
            return SandboxResult(
                exit_code=-1,
                output=(
                    "sandbox refused: CC_SANDBOX=bubblewrap was requested but "
                    "bubblewrap is not usable on this host (not installed, or "
                    "user namespaces are blocked). Install bubblewrap, or set "
                    "CC_SANDBOX=off to run uncontained (not recommended)."
                ),
                backend="none",
            )
        return _run_bubblewrap(
            command, project_dir, timeout,
            allow_shell_features=allow_shell_features, env=env,
        )

    # backend == 'auto' (the default) or anything unrecognized.
    if _bwrap_usable():
        return _run_bubblewrap(
            command, project_dir, timeout,
            allow_shell_features=allow_shell_features, env=env,
        )
    # No sandbox available on this platform: warn ONCE, then passthrough so
    # local dev on macOS/Windows keeps working. Operators who require
    # containment set CC_SANDBOX=bubblewrap to turn this into a hard refusal.
    if not _WARNED_NO_SANDBOX:
        logger.warning(
            "sandbox.unavailable_passthrough",
            extra={"detail": (
                "bubblewrap is not usable on this host; commands run "
                "UNCONTAINED. Install bubblewrap for isolation, or set "
                "CC_SANDBOX=bubblewrap to require it (fail closed)."
            )},
        )
        _WARNED_NO_SANDBOX = True
    return _run_passthrough(
        command, project_dir, timeout,
        allow_shell_features=allow_shell_features, env=env,
    )


async def sandbox_run_async(
    command: str,
    project_dir: Path,
    timeout: int,
    *,
    allow_shell_features: bool,
    profile: SandboxProfile = SandboxProfile.CPU,
    env: Optional[dict[str, str]] = None,
) -> SandboxResult:
    """Async wrapper — runs the (blocking) sandbox in a worker thread so the
    agent's event loop is never blocked by a child process."""
    return await asyncio.to_thread(
        sandbox_run,
        command, project_dir, timeout,
        allow_shell_features=allow_shell_features,
        profile=profile, env=env,
    )
