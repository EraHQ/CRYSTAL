"""Project instructions — the standing-rules file, tracked live.

One file at the project root, first found wins:

    CRYSTAL.md  >  AGENTS.md  >  CLAUDE.md

Ours wins when present; the other two give ecosystem compatibility for
free (teams already carrying an AGENTS.md or CLAUDE.md get it picked up
with zero setup). The file's contents ride in EVERY turn's system
prompt as a PROJECT INSTRUCTIONS section — standing rules are an
identity contract, not retrieval content, so they're injected verbatim
rather than fetched by resemblance. Reference knowledge about the
codebase belongs in the bank (via /ingest), which is why this file can
stay small where other agents need a 2,000-line CLAUDE.md.

Two live behaviors:

  TRACK — `refresh()` is called once per turn (a stat call). If the
  file changed on disk, was created, removed, or a higher-precedence
  file appeared, the next prompt carries the new state and the caller
  gets a one-line notice to print.

  PERSIST — the injected section instructs the agent to record NEW
  standing rules the user establishes ("always...", "from now on...")
  by editing this file with its normal file tools. That write goes
  through the guard like any other edit: diff shown, user approves.
  With no file present, a one-line policy tells the agent to create
  CRYSTAL.md when the first standing rule arrives.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

INSTRUCTION_FILES = ("CRYSTAL.md", "AGENTS.md", "CLAUDE.md")

# Past this size the file is eating real token budget on every turn —
# warn once and point at /ingest for the reference material.
SIZE_WARN_CHARS = 10_000

_SECTION = """

PROJECT INSTRUCTIONS — standing rules from {name} at the project root. \
Follow them; where they conflict with your default policies, they win:

{text}

When the user establishes a NEW standing rule in conversation \
("always...", "never...", "from now on..."), persist it: edit {name} \
with your file tools, adding the rule under a fitting heading. Standing \
rules live in that file, not in the knowledge bank."""

_NO_FILE_POLICY = (
    "\n\nThis project has no instructions file (CRYSTAL.md / AGENTS.md / "
    "CLAUDE.md). If the user establishes a standing rule (\"always...\", "
    "\"never...\", \"from now on...\"), create CRYSTAL.md at the project "
    "root with your file tools and record the rule there — standing rules "
    "live in that file, not in the knowledge bank."
)


class ProjectInstructions:
    """Resolve, cache, and live-track the project's instructions file."""

    def __init__(self, project_dir: Path) -> None:
        self._dir = project_dir
        self.path: Optional[Path] = None
        self._mtime: Optional[float] = None
        self._text = ""
        self._size_warned = False
        # The notice from the initial resolve, for the caller's startup
        # banner. Subsequent changes surface through refresh()'s return.
        self.startup_notice: Optional[str] = self.refresh()

    def _resolve(self) -> Optional[Path]:
        for name in INSTRUCTION_FILES:
            p = self._dir / name
            if p.is_file():
                return p
        return None

    def refresh(self) -> Optional[str]:
        """Re-resolve precedence and re-read on change.

        Cheap when nothing changed (one stat). Returns a printable
        one-line notice when the active file was found, updated,
        switched (a higher-precedence file appeared), or removed —
        None when nothing changed.
        """
        p = self._resolve()
        if p is None:
            if self.path is None:
                return None
            self.path, self._mtime, self._text = None, None, ""
            return "project instructions file removed — section dropped from the prompt"
        try:
            mtime = p.stat().st_mtime
        except OSError:
            return None  # raced a delete; next turn's refresh settles it
        if p == self.path and mtime == self._mtime:
            return None
        first = self.path is None
        switched = self.path is not None and p != self.path
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return f"could not read {p.name}: {e}"
        self.path, self._mtime, self._text = p, mtime, text
        verb = "found" if first else ("switched to" if switched else "updated")
        notice = f"project instructions {verb}: {p.name} ({len(text):,} chars)"
        if len(text) > SIZE_WARN_CHARS and not self._size_warned:
            self._size_warned = True
            notice += (
                f" — over {SIZE_WARN_CHARS:,} chars. This rides in every "
                "prompt; consider moving reference material into the bank "
                "via /ingest and keeping only the rules here"
            )
        return notice

    def addendum(self) -> str:
        """The system-prompt section. Always non-empty: with a file, the
        full rules section; without one (or an empty one), the one-line
        create-CRYSTAL.md policy so persistence works from zero."""
        if self.path is None or not self._text.strip():
            return _NO_FILE_POLICY
        return _SECTION.format(name=self.path.name, text=self._text.strip())
