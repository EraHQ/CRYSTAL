# CRYS — a coding agent with a real memory

CRYS is a terminal coding agent built on CRYSTAL's memory. It reads,
searches, and edits code in one project folder — and it has a
persistent knowledge bank: ingest a codebase once and the agent can
answer questions about it later, even from a different folder, even
after a restart.

What makes it different from the agent CLIs you've used:

- **Every change is gated.** File writes show you a colored diff and
  ask before touching anything. Shell commands are shown verbatim and
  individually approved — there is no "approve all" for shell, and
  `/auto` never applies to it. The browser asks a one-time y/n the
  first moment the agent needs it.
- **Verification is ground truth, not vibes.** Configure your test
  command once and the agent runs it after edits — and in background
  mode, the CLI re-runs it ITSELF and reports the real exit code. The
  agent's "all tests pass ✓" is never taken at its word.
- **It remembers.** `/ingest` crystallizes a codebase into a local
  knowledge bank (code is keyed per-symbol, e.g.
  `Code|greetings.py|greet`). Re-running is an idempotent sync —
  unchanged files skip, changed files replace their old knowledge.
- **Standing rules persist.** Drop an `AGENTS.md` (or `CLAUDE.md`, or
  `CRYSTAL.md`) in the project root and its rules ride in every prompt.
  Say "from now on, always X" in conversation and the agent writes the
  rule into that file — through the same diff-and-approve gate as any
  edit.
- **It works unattended — with your sign-off.** Hand it a whole task
  headlessly (`--task`), or queue work for the CRYS daemon to complete
  later (`--queue`, or just ask the agent to queue it). Background runs
  happen on a fresh git branch with shell and browser disabled, your
  tests run by the CLI as ground truth, and your branch restored no
  matter what.
- **It can browse, when you say so.** The first time the agent reaches
  for the web you get one y/n; after that, reading flows freely and
  page interactions are individually approved.

Full documentation — architecture and safety model, commands, config,
standing rules, the knowledge bank, and background runs — lives on the
website.

## Prerequisites

- Python 3.11+
- Node.js (the file/search tools are MCP servers fetched via `npx`)
- `rg` (ripgrep) on your PATH — for code-content search
- An Anthropic API key

## Install

```bash
git clone https://github.com/EraHQ/CRYSTAL.git
cd CRYSTAL
python -m venv .venv
source .venv/Scripts/activate        # Git Bash on Windows
# source .venv/bin/activate          # macOS / Linux
pip install -e ".[embeddings,agent]"
```

## Your API key

CRYS looks for a key in this order and tells you at startup which one
it found:

1. an exported `ANTHROPIC_API_KEY` environment variable;
2. your saved CRYS config (`~/.crystal-code/config.json`);
3. a `.env` file at the repo root (read directly, never exported);
4. none of the above → first launch asks three questions (name,
   provider, key), saves them, and never asks again. `/setup` re-runs
   this any time.

## Run

```bash
cd CRYS
python -m crys /path/to/some/project
```

(`python -m crystal_code` is the same program. Run from
`CRYS/` — the package lives there. On Windows Git Bash, use
forward slashes in the path.)

First launch downloads the embedding model (~250MB, one time). File
access is limited to the project folder you point it at — it cannot
touch anything else.

## Where your knowledge lives

By default the bank is a local SQLite file (`crystal_cache.db`) in the
folder you launched from — zero setup, survives restarts, yours alone.
Options when you want more: `--db path-or-url` points at a specific
store (a file, or Postgres for a shared deployment); `/login` connects
to a shared store as a real customer with that customer's full memory;
`/logout` returns to local. The background daemon shares the queue
with whoever launched from the same folder (or the same `--db`).

## A ten-minute test drive

1. Point it at a small project of yours and just talk to it — ask what
   a file does, ask it to fix something. Watch the dim activity lines
   narrate what it's doing; approve or reject the diffs it proposes.
2. Drop a one-rule `AGENTS.md` in the project root (e.g. "Always write
   docstrings in the imperative mood"). New session — the rule now
   shapes its edits. Edit the file mid-session; the next turn picks it
   up.
3. Tell it: *"From now on, always include a usage example in
   docstrings."* It will propose an edit to AGENTS.md — your rule, now
   permanent.
4. `/ingest` the project (it asks before any cost is incurred; code
   ingests without LLM calls). Then launch against a DIFFERENT folder
   and ask about the first project — the answer comes from the bank.
5. Create `.crystal-code.json` with `{"verify_command": "python -m
   pytest -q"}`, then hand it a whole task headlessly:

   ```bash
   python -m crys /path/to/project \
       --task "add input validation to the parse function" \
       --branch agent/validate
   ```

   It plans read-only, executes on a fresh git branch (your worktree
   must be clean — it refuses otherwise), runs your tests itself, and
   leaves you a reviewable branch. Your branch is restored no matter
   what happens — crashes included.

6. Try the daemon: in one terminal `python -m crys --daemon`, in
   another ask the agent to *"queue a background task to …"* —
   approve the queue prompt, watch the daemon claim and run it, then
   `/tasks` shows it done.

## Commands

`/info` where everything stands · `/ingest` load a codebase into the
bank · `/plan` + `/go` propose-then-approve mode · `/auto` lift
approval prompts for the session (writes only — never shell) ·
`/tasks` the background queue · `/checkpoints` + `/rewind [n]` undo
the agent's file changes · `/login` connect to a shared knowledge
store · `/model` see model routing · `/reset` clear the conversation ·
`-v` / `--verbose` full structured logs if you want to see the
machinery. Full reference on the website.

## Honest limits (told upfront)

- Shell commands run with YOUR permissions once you approve them — the
  prompt shows the exact command, and reading it is the contract. An
  OS sandbox (which would let shell run without per-command prompts)
  is on the roadmap. Browser pages are untrusted input; interactions
  (clicks, typing, forms) are individually approved.
- The knowledge bank reflects what was ingested — edit a file and the
  bank is stale until the next `/ingest` (a file watcher is on the
  backlog). Re-syncs are cheap: unchanged files skip by content hash.
- Single-user, local-first. The bank is a SQLite file in the launch
  folder unless you `/login` to a shared one. If a schema-mismatch
  error ever appears after pulling updates, delete
  `CRYS/crystal_cache.db` and re-ingest — local stores don't
  migrate yet.
- The verify command and approved shell commands run with your shell:
  only approve what you'd run yourself.

## Feedback wanted

The interesting questions: Does the approval flow feel like safety or
friction? Does the bank answer accurately after `/ingest`? Do standing
rules in AGENTS.md actually steer the edits you see? Where does it
reach for the wrong tool?
