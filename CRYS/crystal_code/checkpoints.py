"""F2 — Checkpoints / rewind (undo for the agent).

A snapshot of the project's files is taken automatically before the
agent's FIRST approved write of each turn (lazily — read-only turns
take none). `/checkpoints` lists them; `/rewind [n]` restores the
files to a snapshot.

HOW SNAPSHOTS WORK (and why they're safe):
Everything goes through git plumbing with a TEMPORARY index file
(GIT_INDEX_FILE), never the user's real index:

    git add -A .          (into the temp index; honors .gitignore)
    git write-tree        (tree object from the temp index)
    git commit-tree       (parentless commit wrapping the tree)
    git update-ref refs/crystal-code/checkpoints/cp-<ts>

The user's own staging area, stash, branches, and HEAD are untouched —
the only trace is private refs under refs/crystal-code/, pruned to the
most recent MAX_CHECKPOINTS.

REWIND restores every file recorded in the snapshot to its recorded
content (`git restore --source=<commit> --worktree -- .`). Files the
agent CREATED after the snapshot are deliberately left in place and
reported — deleting files automatically is how an undo feature causes
the damage it exists to prevent.

Non-git projects: `available` is False, snapshots no-op, and the
commands explain themselves. Everything else works normally.

Subprocess calls are synchronous inside the CLI's async loop — same
deliberate choice as the guard's input(): single-user local REPL,
nothing else should be running while a snapshot lands.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

REF_PREFIX = "refs/crystal-code/checkpoints"
MAX_CHECKPOINTS = 20


def _run(args: list[str], cwd: Path, env: Optional[dict] = None) -> subprocess.CompletedProcess:
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=full_env,
        capture_output=True,
        text=True,
        timeout=60,
    )


class CheckpointManager:
    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir
        self.available = (
            _run(["rev-parse", "--git-dir"], project_dir).returncode == 0
        )
        self._turn_label = ""
        self._snapshotted_this_turn = False

    # -- turn lifecycle (called by the CLI) ----------------------------

    def begin_turn(self, label: str) -> None:
        """Mark a new user turn; the next approved write snapshots first."""
        self._turn_label = " ".join(label.split())[:60]
        self._snapshotted_this_turn = False

    def on_write(self, _path: str) -> None:
        """Guard's notify_write target: snapshot once per turn, lazily."""
        if not self.available or self._snapshotted_this_turn:
            return
        if self.take(self._turn_label or "agent edit") is not None:
            self._snapshotted_this_turn = True

    # -- snapshots ------------------------------------------------------

    def take(self, label: str) -> Optional[str]:
        """Snapshot the working tree. Returns the ref name, or None."""
        if not self.available:
            return None
        tree = self._worktree_tree()
        if tree is None:
            return None
        msg = f"crystal-code checkpoint: {label}" if label else "crystal-code checkpoint"
        commit = _run(
            ["commit-tree", tree, "-m", msg],
            self.project_dir,
            {
                "GIT_AUTHOR_NAME": "crystal-code",
                "GIT_AUTHOR_EMAIL": "agent@crystal-code.local",
                "GIT_COMMITTER_NAME": "crystal-code",
                "GIT_COMMITTER_EMAIL": "agent@crystal-code.local",
            },
        )
        if commit.returncode != 0:
            return None
        ref = f"{REF_PREFIX}/cp-{int(time.time() * 1000)}"
        if _run(["update-ref", ref, commit.stdout.strip()], self.project_dir).returncode != 0:
            return None
        self._prune()
        return ref

    def _worktree_tree(self) -> Optional[str]:
        """Tree hash of the CURRENT worktree (incl. untracked, minus
        gitignored), via a temporary index — the user's index is never
        touched. Used both to take snapshots and to diff against them."""
        fd, index_path = tempfile.mkstemp(prefix="crystal-code-index-")
        os.close(fd)
        os.unlink(index_path)  # git wants to create it itself
        env = {"GIT_INDEX_FILE": index_path}
        try:
            if _run(["add", "-A", "."], self.project_dir, env).returncode != 0:
                return None
            tree = _run(["write-tree"], self.project_dir, env)
            if tree.returncode != 0:
                return None
            return tree.stdout.strip()
        finally:
            if os.path.exists(index_path):
                try:
                    os.unlink(index_path)
                except OSError:
                    pass

    def list(self) -> list[dict]:
        """Newest first: [{ref, when, label}]."""
        if not self.available:
            return []
        out = _run(
            ["for-each-ref", "--sort=-refname",
             "--format=%(refname)%09%(creatordate:relative)%09%(subject)",
             REF_PREFIX],
            self.project_dir,
        )
        items = []
        for line in out.stdout.splitlines():
            parts = line.split("\t", 2)
            if len(parts) == 3:
                label = parts[2].removeprefix("crystal-code checkpoint: ")
                items.append({"ref": parts[0], "when": parts[1], "label": label})
        return items

    def rewind(self, index: int = 0) -> str:
        """Restore files from checkpoint `index` (0 = most recent).

        Returns a human-readable report. Files created after the
        snapshot are left in place and named in the report.
        """
        if not self.available:
            return "this project isn't a git repository, so checkpoints are off."
        items = self.list()
        if not items:
            return "no checkpoints yet — one is taken before the agent's first edit each turn."
        if not (0 <= index < len(items)):
            return f"no checkpoint #{index}; there are {len(items)} (0 = most recent)."
        ref = items[index]["ref"]

        restore = _run(
            ["restore", "--source", ref, "--worktree", "--", "."],
            self.project_dir,
        )
        if restore.returncode != 0:
            return f"rewind failed: {restore.stderr.strip()}"

        # Files present now but absent from the snapshot (the agent
        # created them after it) — left alone, but named. Plain
        # `git diff <commit>` would MISS them: they're untracked in the
        # user's index and diff ignores untracked files. So compare
        # tree-to-tree, deriving the current worktree's tree through the
        # same temp-index plumbing the snapshots use.
        leftovers: list[str] = []
        now_tree = self._worktree_tree()
        if now_tree:
            diff = _run(
                ["diff", "--name-status", ref, now_tree], self.project_dir
            )
            for line in diff.stdout.splitlines():
                if line.startswith("A\t"):
                    leftovers.append(line.split("\t", 1)[1])

        report = f"restored files to checkpoint: {items[index]['label'] or ref} ({items[index]['when']})"
        if leftovers:
            shown = ", ".join(leftovers[:8]) + (" ..." if len(leftovers) > 8 else "")
            report += (
                f"\n  note: {len(leftovers)} file(s) created after that snapshot were "
                f"left in place (nothing is auto-deleted): {shown}"
            )
        return report

    # -- internals ------------------------------------------------------

    def _prune(self) -> None:
        items = self.list()
        for item in items[MAX_CHECKPOINTS:]:
            _run(["update-ref", "-d", item["ref"]], self.project_dir)
