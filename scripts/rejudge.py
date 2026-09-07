#!/usr/bin/env python3
"""Re-judge stored answers for specific question ids (judge audit, C=2026-09-06).

Re-runs judge() on the model_answer ALREADY in the results file under the
CURRENT judge prompt (JUDGE_PROMPT_VERSION) and appends corrected result rows
(last-wins) plus a rejudge manifest line. Never re-asks the model; never
touches a server. Spend: one small judge call per id.

Usage (desktop, ANTHROPIC_API_KEY exported):
  python scripts/rejudge.py --results results/lme_s_headline_v1.jsonl \
      --ids id1,id2,... [--judge-model claude-sonnet-4-6] [--show-prompts]

--show-prompts prints each rendered judge prompt — for anomaly investigation
(the 8464fc84 / Roscioli case: a bare 'no' on an answer that contains the
gold verbatim; if it repeats deterministically, the rendered prompt is the
evidence to read).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import longmemeval_harness as lme  # noqa: E402


def rejudge_row(row: dict, correct: bool, raw: str) -> dict:
    """Corrected result row: same evidence, new verdict, honest provenance.
    Pure so it can be pinned."""
    out = dict(row)
    out["correct"] = bool(correct)
    out["judge_verdict_raw"] = raw
    out["rejudged"] = True
    out["judge_prompt_version"] = lme.JUDGE_PROMPT_VERSION
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--ids", required=True,
                    help="comma-separated question ids to re-judge")
    ap.add_argument("--judge-model", default=None,
                    help="default: the results file's last manifest")
    ap.add_argument("--show-prompts", action="store_true")
    args = ap.parse_args(argv)

    import anthropic
    client = anthropic.Anthropic()

    path = Path(args.results)
    rows, manifest = lme.load_results_file(path)
    by_id = {str(r.get("question_id")): r for r in rows}
    judge_model = args.judge_model or manifest.get("judge_model")
    if not judge_model:
        print("no judge model (none in manifest; pass --judge-model)")
        return 2

    ids = [i.strip() for i in args.ids.split(",") if i.strip()]
    print(f"re-judging {len(ids)} row(s) with {judge_model} "
          f"under judge prompt {lme.JUDGE_PROMPT_VERSION}")

    flipped, unchanged, missing = [], [], []
    appended: list[dict] = []
    for qid in ids:
        row = by_id.get(qid)
        if row is None or "model_answer" not in row:
            print(f"  {qid}: NOT FOUND or has no stored answer — skipped")
            missing.append(qid)
            continue
        q = {
            "question_id": qid,
            "question_type": row.get("question_type", ""),
            "question": row.get("question", ""),
            "answer": row.get("expected_answer", ""),
        }
        if args.show_prompts:
            print("---- prompt for", qid)
            print(lme._judge_prompt(q, str(row.get("model_answer"))))
            print("----")
        correct, raw = lme.judge(
            client, judge_model, q, str(row.get("model_answer"))
        )
        was = bool(row.get("correct"))
        tag = "FLIPPED" if correct != was else "unchanged"
        (flipped if correct != was else unchanged).append(qid)
        print(f"  {qid}: {was} -> {correct}  [{tag}]  raw={raw!r}")
        appended.append(rejudge_row(row, correct, raw))
        time.sleep(0.2)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "record": "rejudge_manifest",
            "ts": datetime.now(timezone.utc).isoformat(),
            "judge_model": judge_model,
            "judge_prompt_version": lme.JUDGE_PROMPT_VERSION,
            "ids": ids,
            "flipped": flipped,
            "unchanged": unchanged,
            "missing": missing,
        }) + "\n")
        for out in appended:
            f.write(json.dumps(out) + "\n")

    print(f"\nappended {len(appended)} corrected row(s) + manifest to {path}")
    print(f"flipped: {len(flipped)}  unchanged: {len(unchanged)}  "
          f"missing: {len(missing)}")
    print("run the harness --report against the file for the updated table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
