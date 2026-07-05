"""Per-project configuration: `.crystal-code.json` at the project root.

One small file the user owns, read at startup. Shared by three
features (see docs/CODING_AGENT_FEATURES_PLAN.md):

    {
      "verify_command": "pytest -q",                       (F3)
      "models": {"main": "...", "fast": "..."},            (F6)
      "shell": {                                            (shell v1)
        "allow": ["git", "pytest", "python"],
        "deny": ["push --force"],
        "timeout_seconds": 120
      },
      "hooks": {
        "block_paths": ["migrations/**", ".env"],          (F5)
        "on_file_edited": ["python -m black {file}"]
      }
    }

Missing file or missing keys are fine — every feature degrades to off.
A malformed file never crashes the CLI; it's reported and treated as
absent, because a broken config should cost the user a warning, not
their session.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

CONFIG_NAME = ".crystal-code.json"


@dataclass
class ProjectConfig:
    verify_command: Optional[str] = None
    models: dict = field(default_factory=dict)
    hooks: dict = field(default_factory=dict)
    shell: dict = field(default_factory=dict)
    error: Optional[str] = None  # set when the file existed but was unreadable


def load_project_config(project_dir: Path) -> ProjectConfig:
    path = project_dir / CONFIG_NAME
    if not path.exists():
        return ProjectConfig()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return ProjectConfig(error=f"{CONFIG_NAME} could not be read: {e}")
    if not isinstance(raw, dict):
        return ProjectConfig(error=f"{CONFIG_NAME} must contain a JSON object")

    verify = raw.get("verify_command")
    models = raw.get("models")
    hooks = raw.get("hooks")
    shell = raw.get("shell")
    return ProjectConfig(
        verify_command=str(verify).strip() if isinstance(verify, str) and str(verify).strip() else None,
        models=models if isinstance(models, dict) else {},
        hooks=hooks if isinstance(hooks, dict) else {},
        shell=shell if isinstance(shell, dict) else {},
    )
