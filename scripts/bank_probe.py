#!/usr/bin/env python3
"""Bank probe (Q1-Q4=A, 2026-09-04) — joins the headline results file to the
live banks on a restored snapshot VM and classifies every miss by ROOT CAUSE:

  BLIND            expected answer present in the crystals the agent SURFACED
  SHORTFALL        present in the tenant's bank, absent from what surfaced
  UNSEARCHED-AVAIL never searched (A-class), but the bank had it
  EXTRACTION-LOSS  absent from the entire bank — ingest never captured it
  REVIEW           presence ambiguous (token overlap in the middle band)

Access is via the shipped keyless admin routes over localhost (Q1=A, R9-clean:
no SQL here). Presence is normalized token-overlap with an adjudication tier
(Q2=A): counts are honest RANGES until the REVIEW cards are human-read.
Scope: all wrong rows + a seeded correct control (Q3=A). Output: a single
self-contained HTML dossier + probe_summary.json (Q4=A). Outputs embed
dataset and bank text — commit PRIVATE-ONLY, never mirror.

Run on the restored VM (api container up, rate limiter off via bench .env):
  python scripts/bank_probe.py --results results/lme_s_headline_v1.jsonl \
      --out probe/
"""
from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import httpx

try:
    import longmemeval_harness as lme
except ImportError:  # imported via importlib (tests) — sibling path missing
    sys.path.append(str(Path(__file__).resolve().parent))
    import longmemeval_harness as lme

BASE = os.environ.get("CC_BASE_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = httpx.Timeout(60.0)

# Give-up phrasing (matches the 2026-09-04 analysis; heuristic by design —
# the dossier exists so a human adjudicates edge cards).
GIVEUP_PHRASES = [
    "don't have any information", "no information", "don't have information",
    "doesn't contain", "don't have enough information", "not have any record",
    "no record", "don't see any", "couldn't find", "could not find",
    "don't have specific", "don't have details", "didn't return",
    "didn't turn up", "didn't surface", "did not return",
    "nothing specific about", "doesn't have a record",
    "no specific information", "wasn't able to find", "unable to find",
    "don't have access to", "bank doesn't have", "doesn't appear in",
]

_STOP = {"the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for",
         "with", "was", "were", "is", "are", "that", "this", "it", "my",
         "your", "their", "his", "her"}


def gave_up(answer: str) -> bool:
    a = (answer or "").lower()
    return any(p in a for p in GIVEUP_PHRASES)


def classify_behavior(row: dict) -> str:
    """The 2026-09-04 seven-bucket behavior taxonomy. Correct rows -> OK."""
    if row.get("correct"):
        return "OK"
    tools = row.get("tool_calls") or []
    answer = str(row.get("model_answer") or "")
    if not tools:
        return "A1" if gave_up(answer) else "A2"
    if gave_up(answer):
        return "B1"
    qtype = row.get("question_type", "")
    expected = str(row.get("expected_answer") or "")
    if qtype == "knowledge-update":
        return "C3"
    if re.search(r"\bat least\b", answer) or (
        "how many" in str(row.get("question", "")).lower()
        and any(c.isdigit() for c in expected)
    ):
        return "C1"
    if qtype == "temporal-reasoning":
        return "C2"
    return "C4"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", str(s or "").lower())


def content_tokens(s: str) -> list[str]:
    return [t for t in _norm(s).split() if len(t) > 3 and t not in _STOP]


def presence(expected: str, text: str) -> tuple[float, str]:
    """(score, PRESENT | ABSENT | REVIEW). Q2=A thresholds: >=0.75 PRESENT,
    <0.15 ABSENT, middle REVIEW. Short/numeric expected answers (no content
    tokens) fall back to exact normalized substring: hit=PRESENT else REVIEW —
    never ABSENT on the weak test alone."""
    ntext = _norm(text)
    toks = content_tokens(expected)
    if not toks:
        needle = _norm(expected).strip()
        if needle and f" {needle} " in f" {ntext} ":
            return 1.0, "PRESENT"
        return 0.0, "REVIEW"
    score = sum(1 for t in toks if t in ntext) / len(toks)
    if score >= 0.75:
        return score, "PRESENT"
    if score < 0.15:
        return score, "ABSENT"
    return score, "REVIEW"


def classify_root_cause(behavior: str, surf_verdict: str,
                        bank_verdict: str) -> str:
    if behavior in ("A1", "A2"):
        if bank_verdict == "PRESENT":
            return "UNSEARCHED-AVAILABLE"
        return "EXTRACTION-LOSS" if bank_verdict == "ABSENT" else "REVIEW"
    if surf_verdict == "PRESENT":
        return "BLIND"
    if bank_verdict == "PRESENT":
        return "SHORTFALL"
    if bank_verdict == "ABSENT":
        return "EXTRACTION-LOSS"
    return "REVIEW"


# ---- admin-route fetchers (Q1=A) -------------------------------------------

def bank_crystal_ids(client: httpx.Client, customer_id: str,
                     cap: int = 800) -> list[str]:
    ids: list[str] = []
    offset = 0
    while len(ids) < cap:
        r = client.get(
            f"{BASE}/admin/api/customers/{customer_id}/crystals",
            params={"limit": 50, "offset": offset},
        )
        r.raise_for_status()
        body = r.json()
        page = [c["id"] for c in body.get("crystals") or []]
        ids.extend(page)
        offset += 50
        if offset >= int(body.get("total") or 0) or not page:
            break
    return ids[:cap]


_TEXT_CACHE: dict[str, str] = {}


def crystal_text(client: httpx.Client, crystal_id: str) -> str:
    if crystal_id in _TEXT_CACHE:
        return _TEXT_CACHE[crystal_id]
    r = client.get(f"{BASE}/admin/api/crystals/{crystal_id}")
    if r.status_code == 404:
        _TEXT_CACHE[crystal_id] = ""
        return ""
    r.raise_for_status()
    body = r.json()
    parts = [str((body.get("crystal") or {}).get("summary_text") or "")]
    for f in body.get("facts") or []:
        parts.extend(str(f.get(k) or "") for k in
                     ("claim_text", "prompt_text", "answer_value"))
    text = "\n".join(p for p in parts if p)
    _TEXT_CACHE[crystal_id] = text
    return text


def best_presence(expected: str, texts: dict[str, str]) -> dict:
    best = {"score": 0.0, "verdict": "ABSENT", "crystal_id": None}
    for cid, text in texts.items():
        score, verdict = presence(expected, text)
        rank = {"PRESENT": 2, "REVIEW": 1, "ABSENT": 0}
        if (rank[verdict], score) > (rank[best["verdict"]], best["score"]):
            best = {"score": round(score, 2), "verdict": verdict,
                    "crystal_id": cid}
    return best


# ---- main -------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    rows, _ = lme.load_results_file(Path(args.results))
    rows = [r for r in rows if r.get("customer_id")]
    wrong = [r for r in rows if not r.get("correct")]
    correct = [r for r in rows if r.get("correct")]
    random.seed(args.seed)
    control = random.sample(correct, min(args.control, len(correct)))
    targets = [(r, False) for r in wrong] + [(r, True) for r in control]
    print(f"probing {len(wrong)} wrong + {len(control)} control rows "
          f"against {BASE}")

    cards, counts = [], Counter()
    with httpx.Client(timeout=TIMEOUT) as client:
        for i, (row, is_control) in enumerate(targets, 1):
            qid = row.get("question_id")
            cid = row["customer_id"]
            expected = str(row.get("expected_answer") or "")
            behavior = classify_behavior(row)
            surfaced_ids = row.get("surfaced_crystal_ids") or []
            try:
                all_ids = bank_crystal_ids(client, cid)
                bank_texts = {c: crystal_text(client, c) for c in all_ids}
                surf_texts = {c: bank_texts.get(c) or
                              crystal_text(client, c) for c in surfaced_ids}
            except Exception as e:
                print(f"[{i}] {qid}: fetch error {type(e).__name__}: {e}")
                continue
            surf_best = best_presence(expected, surf_texts) if surf_texts \
                else {"score": 0.0, "verdict": "ABSENT", "crystal_id": None}
            bank_best = best_presence(expected, bank_texts)
            root = ("CONTROL" if is_control else
                    classify_root_cause(behavior, surf_best["verdict"],
                                        bank_best["verdict"]))
            stale = None
            if behavior == "C3":
                wrong_val = str(row.get("model_answer") or "")[:300]
                stale = best_presence(wrong_val, bank_texts)
            counts[root] += 1
            cards.append({
                "question_id": qid, "customer_id": cid,
                "question_type": row.get("question_type"),
                "behavior": behavior, "root_cause": root,
                "control": is_control,
                "question": row.get("question"),
                "expected_answer": expected,
                "model_answer": str(row.get("model_answer") or ""),
                "queries": [t.get("input") for t in
                            (row.get("tool_calls_detail") or [])],
                "surfaced": [{"id": c, "text": surf_texts.get(c, "")[:4000]}
                             for c in surfaced_ids],
                "surfaced_best": surf_best, "bank_best": bank_best,
                "bank_size": len(bank_texts),
                "bank_hit_text": (bank_texts.get(bank_best["crystal_id"], "")
                                  [:4000] if bank_best["crystal_id"] else ""),
                "stale_check": stale,
                "control_note": ("surfaced presence expected PRESENT"
                                 if is_control else None),
            })
            if i % 25 == 0:
                print(f"  [{i}/{len(targets)}] cached crystals: "
                      f"{len(_TEXT_CACHE)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ctrl = [c for c in cards if c["control"]]
    ctrl_present = sum(1 for c in ctrl
                       if c["surfaced_best"]["verdict"] == "PRESENT")
    summary = {
        "results_file": args.results,
        "rows_probed": len(cards),
        "root_cause_counts": dict(counts),
        "root_cause_by_behavior": {
            b: dict(Counter(c["root_cause"] for c in cards
                            if c["behavior"] == b and not c["control"]))
            for b in sorted({c["behavior"] for c in cards
                             if not c["control"]})
        },
        "control_calibration": {
            "n": len(ctrl), "surfaced_PRESENT": ctrl_present,
            "note": ("correct rows should score PRESENT on surfaced text; "
                     "a low rate means the overlap thresholds under-detect "
                     "and EXTRACTION-LOSS/SHORTFALL are overcounted"),
        },
        "caveat": ("token-overlap heuristic (Q2=A); REVIEW cards need human "
                   "adjudication — treat counts as ranges"),
    }
    (out / "probe_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    (out / "dossier.html").write_text(render_html(cards, summary),
                                      encoding="utf-8")
    print(json.dumps(summary["root_cause_counts"], indent=2))
    print(f"control: {ctrl_present}/{len(ctrl)} PRESENT-on-surfaced")
    print(f"wrote {out/'dossier.html'} and {out/'probe_summary.json'}")
    return 0


def render_html(cards: list[dict], summary: dict) -> str:
    data = json.dumps(cards).replace("</", "<\\/")
    summ = html.escape(json.dumps(summary["root_cause_counts"]))
    return """<!doctype html><meta charset="utf-8">
<title>Bank probe dossier</title>
<style>
body{font-family:system-ui,sans-serif;margin:20px;background:#fafafa}
.card{background:#fff;border:1px solid #ddd;border-radius:8px;
      padding:14px;margin:12px 0}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;color:#fff;
       font-size:12px;margin-right:6px}
.BLIND{background:#c0392b}.SHORTFALL{background:#d68910}
.EXTRACTION-LOSS{background:#6c3483}.REVIEW{background:#7f8c8d}
.UNSEARCHED-AVAILABLE{background:#2471a3}.CONTROL{background:#1e8449}
pre{white-space:pre-wrap;background:#f4f4f4;padding:8px;border-radius:6px;
    max-height:260px;overflow:auto;font-size:12px}
mark{background:#fff176}
button{margin:2px;padding:4px 10px;border-radius:6px;border:1px solid #bbb;
       background:#fff;cursor:pointer}
button.on{background:#2c3e50;color:#fff}
.meta{color:#555;font-size:13px}
</style>
<h1>Bank probe dossier</h1>
<p class="meta">Root-cause counts: <code>""" + summ + """</code>
&nbsp;|&nbsp; REVIEW cards need your eye — counts are ranges until then.</p>
<div id="filters"></div><div id="cards"></div>
<script>
const CARDS = """ + data + """;
const CAUSES=[...new Set(CARDS.map(c=>c.root_cause))].sort();
const TYPES=[...new Set(CARDS.map(c=>c.question_type))].sort();
let fc=new Set(CAUSES), ft=new Set(TYPES);
function esc(s){const d=document.createElement('div');
  d.textContent=s==null?'':String(s);return d.innerHTML;}
function hl(text,expected){
  let out=esc(text);
  const toks=(expected||'').toLowerCase().replace(/[^a-z0-9 ]/g,' ')
    .split(' ').filter(t=>t.length>3);
  for(const t of new Set(toks)){
    out=out.replace(new RegExp('('+t+')','gi'),'<mark>$1</mark>');}
  return out;}
function render(){
  document.getElementById('cards').innerHTML=CARDS
   .filter(c=>fc.has(c.root_cause)&&ft.has(c.question_type))
   .map(c=>`<div class="card">
    <span class="badge ${c.root_cause}">${c.root_cause}</span>
    <b>${esc(c.behavior)}</b> · ${esc(c.question_type)} ·
    <code>${esc(c.question_id)}</code> ·
    tenant <code>${esc(c.customer_id)}</code> ·
    bank ${c.bank_size} crystals
    <p><b>Q:</b> ${esc(c.question)}<br>
    <b>expected:</b> ${esc(c.expected_answer)}<br>
    <b>model:</b> ${esc(c.model_answer)}</p>
    <p class="meta"><b>queries:</b> ${esc(JSON.stringify(c.queries))}</p>
    <p class="meta">surfaced best: ${c.surfaced_best.verdict}
      (${c.surfaced_best.score}) · bank best: ${c.bank_best.verdict}
      (${c.bank_best.score})
      ${c.stale_check?` · wrong-value-in-bank: ${c.stale_check.verdict}`:''}
    </p>
    ${c.surfaced.map(s=>`<details><summary>surfaced ${esc(s.id)}</summary>
      <pre>${hl(s.text,c.expected_answer)}</pre></details>`).join('')}
    ${c.bank_hit_text&&c.surfaced_best.verdict!=='PRESENT'
      ?`<details open><summary>bank hit ${esc(c.bank_best.crystal_id)}
        (NOT surfaced)</summary>
        <pre>${hl(c.bank_hit_text,c.expected_answer)}</pre></details>`:''}
   </div>`).join('');}
function mkbtn(label,set,key){
  const b=document.createElement('button');b.textContent=label;
  b.className='on';
  b.onclick=()=>{set.has(key)?set.delete(key):set.add(key);
    b.classList.toggle('on');render();};return b;}
const f=document.getElementById('filters');
f.append('cause: ');CAUSES.forEach(c=>f.append(mkbtn(c,fc,c)));
f.append(document.createElement('br'));
f.append('type: ');TYPES.forEach(t=>f.append(mkbtn(t,ft,t)));
render();
</script>"""


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default="probe")
    ap.add_argument("--control", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
