"""Completion criteria: a step is done when its evidence says so (Q135 A,
Q136 A, 2026-10-27).

Anthony: "We need a system more like define a completion criteria and if
that step submission falls short it is rejected... what about when the
row count isnt the criteria but more content coverage". 2026-10-26 a
check-in marked "import every game of three seasons" done after reading
the October pages: 82 games of about 3,900. mark_step took the model's
word. Now the planner writes each step's criteria (targets from the
survey's measured numbers, labelled measured or estimated), the design
review checks them, the person can edit them, and a check-in submits a
step: the server measures every criterion and accepts the step only when
all of them pass. A rejection names each shortfall with the measured
value, the target and examples of what is missing.

Kinds (each with label, basis measured | estimated, basis_detail):
  rows        lattice, filters, min                 at least min rows
  coverage    lattice, columns, filters, min_fraction
                                                    rows with every named column filled
  distinct    lattice, column, filters, min         distinct values (teams, dates, seasons)
  reference   lattice, column, of_lattice, of_column, filters, min_per, min_fraction
                                                    each parent (a game) has min_per child rows (its players)
  sources     lattice, column, min                  distinct source pages behind a column
  saturation  lattice, runs                         the last `runs` check-ins that worked the
                                                    lattice found nothing new (open-ended collection)
  groups      lattice, column, filters, min, max, values, exclusive
                                                    every value of column (season, month, team) has
                                                    between min and max rows; values: the ones that
                                                    must be there; exclusive: no others (Series 384)
rows and distinct also take max (0 = no ceiling): too many fails like too
few (2026-10-27: 93 games from the wrong season helped a row count pass).
  rubric      text                                  an independent large-tier verifier reads the evidence
  person      text                                  waits for the person to approve the step
A step with no criteria gets one implied rubric: the step's own words.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)

KINDS = ("rows", "coverage", "distinct", "reference", "sources", "saturation", "groups", "rubric", "person")
MEASURED = ("rows", "coverage", "distinct", "reference", "sources", "saturation", "groups")
MAX_VALUES = 200
MAX_CRITERIA = 8
VERIFY_TIER = "large"
VERIFY_ORIGIN = "goal"
DEFAULT_FRACTION = 0.98

CRITERION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": list(KINDS)},
        "label": {"type": "string"},
        "lattice": {"type": "string"}, "column": {"type": "string"},
        "columns": {"type": "array", "items": {"type": "string"}},
        "filters": {"type": "array", "items": {
            "type": "object",
            "properties": {"column": {"type": "string"}, "op": {"type": "string", "enum": ["eq", "contains", "gte", "lte"]},
                           "value": {"type": "string"}},
            "required": ["column", "op", "value"], "additionalProperties": False}},
        "min": {"type": "number"}, "min_fraction": {"type": "number"},
        "of_lattice": {"type": "string"}, "of_column": {"type": "string"},
        "min_per": {"type": "integer"}, "runs": {"type": "integer"},
        "max": {"type": "number"}, "values": {"type": "array", "items": {"type": "string"}},
        "exclusive": {"type": "boolean"},
        "text": {"type": "string"},
        "basis": {"type": "string", "enum": ["measured", "estimated"]},
        "basis_detail": {"type": "string"},
    },
    "required": ["kind", "label", "lattice", "column", "columns", "filters", "min", "min_fraction", "of_lattice",
                 "of_column", "min_per", "runs", "max", "values", "exclusive", "text", "basis", "basis_detail"],
    "additionalProperties": False,
}

CRITERIA_RULES = """Every step that is not a repeating "each check-in" step gets criteria: how the server tells the step is complete. A check-in submits the step; the server measures each criterion and rejects the step when any falls short, so criteria must be checkable and must be able to fail. Kinds:
- rows: lattice (+ filters) holds at least min rows.
- coverage: of the lattice's rows (+ filters), at least min_fraction (0.98 by default) have every one of columns filled.
- distinct: column takes at least min distinct values (30 teams, 165 game dates, 3 seasons).
- reference: every value of of_column in of_lattice (+ filters on of_lattice) has at least min_per rows in lattice whose column holds it (each game has its 18+ player rows); min_fraction of parents must pass.
- sources: the column's values came from at least min distinct source pages.
- saturation: the last runs check-ins that worked the lattice found nothing new (only for open-ended collection with no known total).
- groups: every value of column (a season, a month, a team) has between min and max rows (+ filters); values lists the ones that must be there, exclusive true when no others belong (the three seasons asked for and only those).
rows and distinct take max too. Give a ceiling whenever the survey knows the size: too many rows (another season's games, duplicates) fails like too few. Prefer groups with min and max per season or per item over one total count: a total passes with the wrong rows and hides which part is missing.
- rubric: text an independent reviewer checks against the evidence (for what counts cannot show: a summary's quality, drafts that follow the instructions).
- person: text the person checks and approves.
Targets come from the survey's measured numbers whenever it has one (basis measured, basis_detail naming the source and number); a number you had to estimate is basis estimated with the reasoning in basis_detail. Prefer coverage and reference over raw row counts: a row count passes with the wrong rows. Each thing the person named (each league, each season, each value) is covered by some criterion. Fields a kind does not use are "" or 0 or []."""


def _s(v: Any, n: int = 200) -> str:
    return str(v or "").strip()[:n]


def _num(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if f == f else default


def _lat_cols(lattices: Optional[list[dict[str, Any]]]) -> Optional[dict[str, set[str]]]:
    if lattices is None:
        return None
    return {str(x.get("name") or "").strip().lower(): {str(c.get("key")) for c in x.get("columns") or []}
            for x in lattices}


def normalize_criteria(raw: Any, lattices: Optional[list[dict[str, Any]]] = None) -> tuple[list[dict[str, Any]], list[str]]:
    """The model's (or the person's) criteria, cleaned. lattices: the plan's
    lattice specs; when given, a criterion naming a lattice or column that
    isn't there is dropped. Returns (criteria, why each dropped one went)."""
    cols = _lat_cols(lattices)
    out: list[dict[str, Any]] = []
    dropped: list[str] = []

    def lattice_ok(name: str, *columns: str) -> bool:
        if cols is None:
            return bool(name)
        have = cols.get(name.lower())
        if have is None:
            dropped.append(f"criterion on '{name}': no such lattice in the plan")
            return False
        bad = [c for c in columns if c and c not in have]
        if bad:
            dropped.append(f"criterion on '{name}': no column {bad[0]}")
            return False
        return True

    for it in raw or []:
        if len(out) >= MAX_CRITERIA or not isinstance(it, dict):
            continue
        kind = _s(it.get("kind"), 20).lower()
        if kind not in KINDS:
            continue
        c: dict[str, Any] = {"kind": kind, "label": _s(it.get("label"), 200),
                             "basis": "measured" if _s(it.get("basis"), 20).lower() == "measured" else "estimated",
                             "basis_detail": _s(it.get("basis_detail"), 300)}
        filters = [{"column": _s(f.get("column"), 40).lower(), "op": _s(f.get("op"), 10).lower() or "eq",
                    "value": _s(f.get("value"), 200)}
                   for f in (it.get("filters") or []) if isinstance(f, dict) and _s(f.get("column"))][:5]
        lattice = _s(it.get("lattice"), 120)
        if kind in ("rubric", "person"):
            text = _s(it.get("text"), 1000)
            if len(text) < 12:
                dropped.append(f"{kind} criterion without a checkable text")
                continue
            c["text"] = text
        elif kind == "groups":
            column = _s(it.get("column"), 40).lower()
            if not column:
                dropped.append(f"groups criterion on '{lattice}' names no column")
                continue
            lo, hi = int(_num(it.get("min"))), int(_num(it.get("max")))
            values = [x for x in dict.fromkeys(_s(v, 120) for v in (it.get("values") or []) if _s(v)) if x][:MAX_VALUES]
            if lo < 1 and not hi and not values:
                dropped.append(f"groups criterion on '{lattice}' cannot fail (no min, max or values)")
                continue
            if hi and hi < lo:
                dropped.append(f"groups criterion on '{lattice}': max {hi} is below min {lo}")
                continue
            if not lattice_ok(lattice, column, *[f["column"] for f in filters]):
                continue
            c.update(lattice=lattice, column=column, filters=filters, min=max(0, lo), max=max(0, hi), values=values,
                     exclusive=bool(it.get("exclusive")) and bool(values))
        elif kind in ("rows", "distinct", "sources"):
            column = _s(it.get("column"), 40).lower()
            need = int(_num(it.get("min")))
            if need < (2 if kind == "sources" else 1):
                dropped.append(f"{kind} criterion on '{lattice}' cannot fail (min {need})")
                continue
            if kind != "rows" and not column:
                dropped.append(f"{kind} criterion on '{lattice}' names no column")
                continue
            if not lattice_ok(lattice, column, *[f["column"] for f in filters]):
                continue
            c.update(lattice=lattice, min=need, filters=filters, **({"column": column} if kind != "rows" else {}))
            hi = int(_num(it.get("max")))
            if kind != "sources" and hi:
                if hi < need:
                    dropped.append(f"{kind} criterion on '{lattice}': max {hi} is below min {need}")
                    continue
                c["max"] = hi
        elif kind == "coverage":
            columns = [x for x in dict.fromkeys(_s(x, 40).lower() for x in it.get("columns") or []) if x][:12]
            if not columns and _s(it.get("column")):
                columns = [_s(it.get("column"), 40).lower()]
            frac = _num(it.get("min_fraction"), DEFAULT_FRACTION) or DEFAULT_FRACTION
            if not columns:
                dropped.append(f"coverage criterion on '{lattice}' names no columns")
                continue
            if not lattice_ok(lattice, *columns, *[f["column"] for f in filters]):
                continue
            c.update(lattice=lattice, columns=columns, filters=filters, min_fraction=max(0.01, min(frac, 1.0)))
        elif kind == "reference":
            column, of_lattice, of_column = (_s(it.get("column"), 40).lower(), _s(it.get("of_lattice"), 120),
                                             _s(it.get("of_column"), 40).lower() or _s(it.get("column"), 40).lower())
            if not (column and of_lattice):
                dropped.append(f"reference criterion on '{lattice}' needs column and of_lattice")
                continue
            if not (lattice_ok(lattice, column) and lattice_ok(of_lattice, of_column, *[f["column"] for f in filters])):
                continue
            frac = _num(it.get("min_fraction"), DEFAULT_FRACTION) or DEFAULT_FRACTION
            c.update(lattice=lattice, column=column, of_lattice=of_lattice, of_column=of_column, filters=filters,
                     min_per=max(1, int(_num(it.get("min_per"), 1))), min_fraction=max(0.01, min(frac, 1.0)))
        elif kind == "saturation":
            if not lattice_ok(lattice):
                continue
            c.update(lattice=lattice, runs=max(2, min(int(_num(it.get("runs"), 2)), 5)))
        if not c["label"]:
            c["label"] = describe(c)
        out.append(c)
    return out, dropped


def describe(c: dict[str, Any]) -> str:
    k = c["kind"]
    f = "".join(f", {x['column']} {x['op']} {x['value']}" for x in c.get("filters") or [])
    top = f" and at most {c['max']:,}" if c.get("max") else ""
    if k == "rows":
        return f"{c['lattice']}{f}: at least {c['min']:,}{top} rows"
    if k == "groups":
        bound = (f"{c['min']:,} to {c['max']:,}" if c.get("max") else f"at least {c['min']:,}") + " rows"
        vals = f" ({', '.join(c['values'][:6])}{', ...' if len(c['values']) > 6 else ''}{' only' if c.get('exclusive') else ''})" \
            if c.get("values") else ""
        return f"{c['lattice']}{f}: each {c['column']}{vals} has {bound}"
    if k == "coverage":
        return f"{c['lattice']}{f}: {c['min_fraction']:.0%} of rows have {', '.join(c['columns'])}"
    if k == "distinct":
        return f"{c['lattice']}{f}: at least {c['min']:,}{top} distinct {c['column']}"
    if k == "reference":
        return f"each {c['of_column']} in {c['of_lattice']}{f} has {c['min_per']}+ rows in {c['lattice']}"
    if k == "sources":
        return f"{c['lattice']}.{c['column']} from at least {c['min']} sources"
    if k == "saturation":
        return f"{c['lattice']}: the last {c['runs']} check-ins found nothing new"
    return c.get("text", "")[:200]


def implied(step: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"kind": "rubric", "label": "The step's own words", "implied": True, "basis": "estimated", "basis_detail": "",
             "text": f"{step.get('title', '')}: {step.get('detail', '')}"[:1000]}]


def lint_criteria(plan: dict[str, Any]) -> list[str]:
    """What the review must fix: a step that moves rows with nothing that
    measures them; a measured criterion whose target was guessed while the
    survey measured the same thing is the planner's job to notice."""
    issues = []
    names = [str(x.get("name") or "") for x in plan.get("lattices") or []]
    for s in plan.get("steps") or []:
        if s.get("repeats"):
            continue
        crit = s.get("criteria") or []
        text = f"{s.get('title', '')} {s.get('detail', '')}".lower()
        named = [n for n in names if n and n.lower() in text]
        if named and not any(c["kind"] in MEASURED for c in crit):
            issues.append(f"step {s['n']} ({s['title'][:60]}) works {', '.join(named)} but no criterion measures it; "
                          "add coverage, reference or distinct criteria with targets from the survey.")
        elif not crit:
            issues.append(f"step {s['n']} ({s['title'][:60]}) has no criteria.")
    return issues


# ------------------------------------------------------------------ measure

def _rid(c: dict[str, Any]) -> dict[str, Any]:
    return {"kind": c["kind"], "label": c.get("label") or describe(c), "basis": c.get("basis"),
            "basis_detail": c.get("basis_detail", "")}


async def _lattice(store: Any, customer_id: str, name: str) -> Optional[dict[str, Any]]:
    from ..lattices import service as lat
    try:
        return await lat._lattice(store, customer_id, name)
    except lat.LatticeRefused:
        return None


def _fail(c: dict[str, Any], gap: str, **kw: Any) -> dict[str, Any]:
    return {**_rid(c), "passed": False, "gap": gap, **kw}


async def measure(store: Any, customer_id: str, goal: dict[str, Any], c: dict[str, Any], *,
                  current_actions: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
    """One measured criterion's result: passed, measured value, target, the
    gap in words and examples of what is missing. Never raises."""
    from ..lattices import service as lat
    kind = c["kind"]
    if kind not in MEASURED:
        return {**_rid(c), "passed": None, "gap": ""}
    lt = await _lattice(store, customer_id, c.get("lattice", ""))
    if lt is None:
        return _fail(c, f"There is no lattice '{c.get('lattice')}'.", invalid=True)
    try:
        if kind == "saturation":
            return await _saturation(store, customer_id, goal, c, lt, current_actions or [])
        if kind == "reference":
            parent = await _lattice(store, customer_id, c["of_lattice"])
            if parent is None:
                return _fail(c, f"There is no lattice '{c['of_lattice']}'.", invalid=True)
            where = lat._filters(parent, c.get("filters"))
            m = await store.lattice_reference_coverage(customer_id, parent["id"], c["of_column"], lt["id"], c["column"],
                                                       min_per=int(c["min_per"]), parent_where=where)
            frac = (m["covered"] / m["parents"]) if m["parents"] else 0.0
            ok = m["parents"] > 0 and frac >= float(c["min_fraction"])
            gap = "" if ok else (
                f"{c['of_lattice']} has no rows to check yet." if not m["parents"] else
                f"{m['parents'] - m['covered']:,} of {m['parents']:,} {c['of_column']} values have fewer than "
                f"{c['min_per']} rows in {c['lattice']} (fewest {m['fewest']}, most {m['most']}).")
            return {**_rid(c), "passed": ok, "value": round(frac, 4), "target": c["min_fraction"],
                    "measured": f"{m['covered']:,} of {m['parents']:,} covered", "gap": gap,
                    "missing_sample": m["missing_sample"]}
        where = lat._filters(lt, c.get("filters"))
        if kind == "groups":
            return await _groups(store, customer_id, c, lt, where)
        if kind == "coverage":
            m = await store.lattice_measure(customer_id, lt["id"], where=where, filled=c["columns"])
            frac = (m["filled"] / m["rows"]) if m["rows"] else 0.0
            ok = m["rows"] > 0 and frac >= float(c["min_fraction"])
            gap = "" if ok else ("No matching rows yet." if not m["rows"] else
                                 f"{m['rows'] - m['filled']:,} of {m['rows']:,} rows lack {', '.join(c['columns'])}.")
            return {**_rid(c), "passed": ok, "value": round(frac, 4), "target": c["min_fraction"],
                    "measured": f"{m['filled']:,} of {m['rows']:,} filled", "gap": gap,
                    "missing_sample": m.get("missing_sample", [])}
        if kind == "rows":
            m = await store.lattice_measure(customer_id, lt["id"], where=where)
            v = m["rows"]
        elif kind == "distinct":
            m = await store.lattice_measure(customer_id, lt["id"], where=where, distinct=c["column"])
            v = m["distinct"]
        else:  # sources
            m = await store.lattice_measure(customer_id, lt["id"], where=where, source_of=c["column"])
            v = m["sources"]
        need = int(c["min"])
        top = int(c.get("max") or 0)
        unit = {"rows": "rows", "distinct": f"distinct {c.get('column')}", "sources": "sources"}[kind]
        if top and v > top:
            return {**_rid(c), "passed": False, "value": v, "target": top, "measured": f"{v:,} {unit}",
                    "gap": f"{v - top:,} {unit} over the most expected ({top:,}): rows that don't belong, or duplicates."}
        return {**_rid(c), "passed": v >= need, "value": v, "target": need, "measured": f"{v:,} {unit}",
                "gap": "" if v >= need else f"{need - v:,} {unit} short of {need:,}."}
    except lat.LatticeRefused as e:
        return _fail(c, f"This criterion can't be measured: {e.message}", invalid=True)
    except Exception as e:  # noqa: BLE001 — a broken measure rejects, never crashes the check-in
        logger.warning("goals.criterion_failed", kind=kind, error_type=type(e).__name__, error=str(e)[:200])
        return _fail(c, "This criterion couldn't be measured just now.")


async def _groups(store: Any, customer_id: str, c: dict[str, Any], lt: dict[str, Any],
                  where: Any) -> dict[str, Any]:
    """Each value of the column with its row count, against min, max and
    the values that must be there (Series 384)."""
    counts = await store.lattice_groups(customer_id, lt["id"], c["column"], where=where)
    seen = {str(k).lower(): (str(k), n) for k, n in counts.items() if k is not None}
    lo, hi = int(c.get("min") or 0), int(c.get("max") or 0)
    missing = [v for v in c.get("values") or [] if v.lower() not in seen]
    extra = [k for k, _ in seen.values() if c.get("exclusive") and k.lower() not in {v.lower() for v in c["values"]}]
    low = [(k, n) for k, n in seen.values() if lo and n < lo and (not c.get("exclusive") or k not in extra)]
    high = [(k, n) for k, n in seen.values() if hi and n > hi and (not c.get("exclusive") or k not in extra)]
    parts = []
    if missing:
        parts.append(f"missing {', '.join(missing[:8])}{' and more' if len(missing) > 8 else ''}")
    if extra:
        parts.append(f"{c['column']} values that don't belong: {', '.join(extra[:8])}")
    if low:
        parts.append("under " + f"{lo:,}: " + ", ".join(f"{k} ({n:,})" for k, n in sorted(low, key=lambda x: x[1])[:6]))
    if high:
        parts.append("over " + f"{hi:,}: " + ", ".join(f"{k} ({n:,})" for k, n in sorted(high, key=lambda x: -x[1])[:6]))
    total = len(seen) + len(missing)
    ok = not parts and bool(seen)
    in_range = len(seen) - len({k for k, _ in low} | {k for k, _ in high}) - len(extra)
    return {**_rid(c), "passed": ok, "value": in_range, "target": total,
            "measured": f"{in_range:,} of {total:,} {c['column']} in range" if seen else "no rows yet",
            "gap": "" if ok else ("No matching rows yet." if not seen else "; ".join(parts).capitalize() + "."),
            "missing_sample": (missing + extra + [k for k, _ in low] + [k for k, _ in high])[:6]}


async def _saturation(store: Any, customer_id: str, goal: dict[str, Any], c: dict[str, Any], lt: dict[str, Any],
                      current: list[dict[str, Any]]) -> dict[str, Any]:
    names = {lt["name"], lt["id"]}

    def worked(actions: list[dict[str, Any]]) -> Optional[int]:
        hits = [a for a in actions if a.get("action") in ("lattice_write", "lattice_import") and a.get("lattice") in names]
        return sum(int(a.get("inserted") or 0) for a in hits) if hits else None
    passes: list[int] = []
    now = worked(current)
    if now is not None:
        passes.append(now)
    for r in await store.list_goal_runs(customer_id, goal["id"], phase="run", limit=30):
        if len(passes) >= int(c["runs"]):
            break
        if r["status"] != "complete":
            continue
        n = worked(list((r.get("result") or {}).get("actions") or []))
        if n is not None:
            passes.append(n)
    found = sum(passes)
    ok = len(passes) >= int(c["runs"]) and found == 0
    gap = "" if ok else (f"Only {len(passes)} check-in(s) have worked {lt['name']}; {c['runs']} must find nothing new."
                         if len(passes) < int(c["runs"]) else f"The last {len(passes)} passes still found {found:,} new rows.")
    return {**_rid(c), "passed": ok, "value": found, "target": 0, "measured": f"{found:,} new in the last {len(passes)}",
            "gap": gap}


# ------------------------------------------------------------------- verify

VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"passed": {"type": "boolean"}, "why": {"type": "string"},
                   "missing": {"type": "array", "items": {"type": "string"}}},
    "required": ["passed", "why", "missing"],
    "additionalProperties": False,
}

VERIFIER_SYSTEM = """You verify one step of a person's goal, independently. The worker who did the step says it is done; you decide from the evidence alone. Pass only when the evidence shows the criterion is met. The worker's note is a claim, not evidence: rows, counts and items are evidence. When the evidence is too thin to tell, do not pass. Reply as JSON: passed, why (one or two sentences naming the evidence), missing (concrete things still missing, empty when passed)."""


async def _evidence(store: Any, customer_id: str, goal: dict[str, Any], step: dict[str, Any], note: str,
                    others: list[dict[str, Any]]) -> str:
    from ..lattices import service as lat
    plan = goal.get("plan") or {}
    parts = [f"Goal: {goal['title']}\nIn the person's words: {(goal.get('directive') or '')[:1500]}",
             f"Step {step.get('n')}: {step.get('title')}\nDetail: {step.get('detail', '')}",
             f"The worker's note: {note or '(none)'}"]
    if others:
        parts.append("Measured by the server:\n" + "\n".join(
            f"- {r['label']}: {r.get('measured', '')} ({'met' if r['passed'] else 'NOT met: ' + r.get('gap', '')})"
            for r in others if r.get("passed") is not None))
    for spec in (plan.get("lattices") or [])[:6]:
        try:
            q = await lat.query(store, customer_id, spec["name"], limit=5)
        except lat.LatticeRefused:
            continue
        rows = [r.get("values") for r in q.get("rows") or []]
        parts.append(f"Lattice {spec['name']}: {q.get('total', 0):,} rows; columns "
                     f"{', '.join(c['key'] for c in spec.get('columns') or [])}; first rows: {json.dumps(rows, default=str)[:1500]}")
    items = await store.list_goal_items(customer_id, goal["id"], limit=10)
    if items:
        parts.append("Recent items: " + "; ".join(f"[{i['kind']}/{i['status']}] {i['title'][:100]}" for i in items))
    return "\n\n".join(parts)[:14000]


async def judge(store: Any, customer_id: str, goal: dict[str, Any], step: dict[str, Any], c: dict[str, Any], *,
                note: str, others: list[dict[str, Any]], llm: Any) -> dict[str, Any]:
    from ..scan._seam import metered_call
    user = (await _evidence(store, customer_id, goal, step, note, others)
            + f"\n\nThe criterion to check: {c['text']}")
    try:
        raw = await metered_call(llm, customer_id=customer_id, origin=VERIFY_ORIGIN, system=VERIFIER_SYSTEM, user=user,
                                 max_tokens=800, tier=VERIFY_TIER, store=store, json_schema=VERDICT_SCHEMA)
        out = json.loads((raw or "").strip())
        ok = out.get("passed") is True
        missing = [str(x)[:200] for x in (out.get("missing") or []) if str(x).strip()][:6]
        return {**_rid(c), "passed": ok, "measured": str(out.get("why") or "")[:400],
                "gap": "" if ok else ("; ".join(missing) or str(out.get("why") or "")[:400]),
                **({"implied": True} if c.get("implied") else {})}
    except Exception as e:  # noqa: BLE001 — an unreadable verdict is not a pass
        logger.warning("goals.verify_failed", goal_id=goal["id"], error_type=type(e).__name__, error=str(e)[:200])
        return {**_rid(c), "passed": False, "gap": "The reviewer couldn't reach a verdict; submit again later."}


async def evaluate(store: Any, customer_id: str, goal: dict[str, Any], step: dict[str, Any], *, note: str = "",
                   llm: Any = None, current_actions: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
    """Every criterion of one step. Measured ones first; the reviewer is
    asked only when every measured criterion passed (no spend on a step
    that is short anyway). verdict: accepted | rejected | awaiting_review."""
    crit = list(step.get("criteria") or []) or implied(step)
    results: list[dict[str, Any]] = []
    for c in crit:
        if c["kind"] in MEASURED:
            results.append(await measure(store, customer_id, goal, c, current_actions=current_actions))
    measured_ok = all(r["passed"] for r in results)
    for c in crit:
        if c["kind"] == "rubric":
            if measured_ok and llm is not None:
                results.append(await judge(store, customer_id, goal, step, c, note=note, others=results, llm=llm))
            else:
                results.append({**_rid(c), "passed": None, "gap": "Checked once the measured criteria pass."})
        elif c["kind"] == "person":
            results.append({**_rid(c), "passed": None, "gap": "Waits for the person.", "text": c["text"]})
    failed = [r for r in results if r["passed"] is False or (r["passed"] is None and r["kind"] == "rubric")]
    waiting = any(r["kind"] == "person" for r in results)
    verdict = "rejected" if failed else "awaiting_review" if waiting else "accepted"
    return {"verdict": verdict, "results": results, "at": datetime.now(timezone.utc).isoformat()}


async def progress(store: Any, customer_id: str, goal: dict[str, Any]) -> list[dict[str, Any]]:
    """Every step with its measured criteria right now (no reviewer calls):
    the goal page's live view."""
    out = []
    for s in (goal.get("plan") or {}).get("steps", []):
        results = []
        if not s.get("done") and not s.get("repeats"):
            for c in s.get("criteria") or []:
                results.append(await measure(store, customer_id, goal, c) if c["kind"] in MEASURED
                               else {**_rid(c), "passed": None, "gap": ""})
        out.append({"n": s.get("n"), "title": s.get("title"), "done": bool(s.get("done")), "state": s.get("state"),
                    "repeats": bool(s.get("repeats")), "results": results, "last_check": s.get("last_check")})
    return out
