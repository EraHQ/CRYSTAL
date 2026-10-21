"""Seed the demo tenant (Q84 A, 2026-10-20): the solo founder's memory.

Reads docs/demo/solo-founder/: facts.json (operator-stated facts, one
POST /v1/store each) and the numbered *.md documents (one POST
/v1/documents each, crystallized immediately). The tenant's API key
comes from the DEMO_KEY environment variable and is never printed.

    read -rsp "demo key: " DEMO_KEY; export DEMO_KEY
    python scripts/demo_seed.py --base https://api.erahq.ai
    unset DEMO_KEY

Re-seeding after Erase all memories is the same command. Standard
library only; runs on Windows Git Bash.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "docs" / "demo" / "solo-founder"


def _post(base: str, path: str, key: str, body: dict, *, timeout: float) -> tuple[int, dict | str]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        base.rstrip("/") + path, data=data, method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "User-Agent": "crystal-demo-seed/1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        return e.code, raw[:400]


def _label(path: Path, text: str) -> str:
    """The document's own H1 when it has one, else the file name."""
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()[:256]
    stem = re.sub(r"^\d+-", "", path.stem)
    return stem.replace("-", " ").capitalize()


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed the solo-founder demo tenant.")
    ap.add_argument("--base", default="https://api.erahq.ai")
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--dry-run", action="store_true", help="print what would be sent, send nothing")
    ap.add_argument("--skip-facts", action="store_true")
    ap.add_argument("--skip-docs", action="store_true")
    args = ap.parse_args()

    key = os.environ.get("DEMO_KEY", "").strip()
    if not key and not args.dry_run:
        print("DEMO_KEY is not set (read -rsp 'demo key: ' DEMO_KEY; export DEMO_KEY)", file=sys.stderr)
        return 2
    folder = Path(args.dir)
    facts_file = folder / "facts.json"
    docs = sorted(p for p in folder.glob("*.md"))
    print(f"seed folder: {folder}")
    print(f"key length: {len(key)}  base: {args.base}")

    failures = 0
    if not args.skip_facts and facts_file.exists():
        facts = json.loads(facts_file.read_text(encoding="utf-8"))
        print(f"\nfacts: {len(facts)}")
        for f in facts:
            if args.dry_run:
                print(f"  would store: {f['key'][:60]}")
                continue
            status, out = _post(args.base, "/v1/store", key, {"key": f["key"], "value": f["value"]}, timeout=60)
            ok = status == 200
            failures += 0 if ok else 1
            print(f"  {'ok ' if ok else 'ERR'} {status} {f['key'][:50]}" + ("" if ok else f" -> {out}"))

    if not args.skip_docs:
        print(f"\ndocuments: {len(docs)}")
        for p in docs:
            text = p.read_text(encoding="utf-8")
            label = _label(p, text)
            if args.dry_run:
                print(f"  would ingest: {label} ({len(text)} chars)")
                continue
            t0 = time.monotonic()
            status, out = _post(args.base, "/v1/documents", key,
                                {"text": text, "label": label, "auto_crystallize": True}, timeout=600)
            took = f"{time.monotonic() - t0:.0f}s"
            if status == 200 and isinstance(out, dict):
                print(f"  ok  {label}: {out.get('status')} · {out.get('crystals_written')} crystals · "
                      f"{out.get('items_extracted')} facts · {took}")
            else:
                failures += 1
                print(f"  ERR {status} {label} ({took}) -> {out}")

    print(f"\ndone · {failures} failure{'s' if failures != 1 else ''}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
