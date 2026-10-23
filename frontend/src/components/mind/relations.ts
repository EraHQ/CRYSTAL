// The Mind walk (2026-10-23). Anthony: "I should be able to walk a path.
// Right now it seems like the bonds don't relate. I can't easily see where
// an assumption comes from, what birthed it... even searching the mind
// from the starting place and navigating the mind manually."
//
// Everything here is derived from the one /mind payload: no second read.
// relationsOf() says what a picked thing came from, sits alongside, and
// led to; the scene lights exactly those and draws the paths; the
// inspector lists them as the next steps of the walk. searchMind() is the
// starting place.
import type { MindScene } from "@/lib/types";
import type { MindSelection } from "./scene";

export interface Relation { sel: MindSelection; label: string; note?: string }
export interface Relations { from: Relation[]; alongside: Relation[]; to: Relation[] }

export const keyOf = (s: MindSelection): string => (s.kind === "overview" ? "overview" : `${s.kind}:${s.id}`);

const short = (s: string, n = 72): string => (s.length > n ? s.slice(0, n - 1) + "…" : s);
const LIMIT = 8;

export function relationsOf(data: MindScene, sel: MindSelection): Relations {
  const out: Relations = { from: [], alongside: [], to: [] };
  const regionName = (r: number | null | undefined) => data.regions.find((x) => x.id === r)?.name ?? "a region";
  const factRel = (f: MindScene["facts"][number], note?: string): Relation => ({ sel: { kind: "fact", id: f.id }, label: short(f.claim), note: note ?? (f.tier === "trusted" ? "trusted" : f.tier === "reviewed" ? "reviewed" : "new") });
  const sourceRel = (id: string | null | undefined): Relation | null => {
    const s = id ? data.sources.find((x) => x.id === id) : null;
    return s ? { sel: { kind: "source", id: s.id }, label: s.label, note: `${s.facts} facts` } : null;
  };
  const push = (list: Relation[], r: Relation | null) => { if (r && !list.some((x) => keyOf(x.sel) === keyOf(r.sel)) && list.length < LIMIT * 3) list.push(r); };
  const factsOfCluster = (crystalId: string) => data.facts.filter((f) => f.crystal_id === crystalId);

  switch (sel.kind) {
    case "overview": return out;
    case "region": {
      data.sources.filter((s) => s.region === sel.id).slice(0, LIMIT).forEach((s) => push(out.from, sourceRel(s.id)));
      data.facts.filter((f) => f.region === sel.id).sort((a, b) => b.hits - a.hits).slice(0, LIMIT).forEach((f) => push(out.to, factRel(f)));
      data.gaps.filter((g) => g.region === sel.id).slice(0, 4).forEach((g) => push(out.to, { sel: { kind: "gap", id: g.id }, label: short(g.missing), note: "gap" }));
      return out;
    }
    case "fact": {
      const f = data.facts.find((x) => x.id === sel.id); if (!f) return out;
      push(out.from, sourceRel(f.doc_id));
      data.runs.filter((r) => r.result_crystal_id === f.crystal_id).forEach((r) => push(out.from, { sel: { kind: "run", id: r.id }, label: short(r.title), note: "produced it" }));
      factsOfCluster(f.crystal_id).filter((x) => x.id !== f.id).slice(0, LIMIT).forEach((x) => push(out.alongside, factRel(x, "same cluster")));
      data.conflicts.filter((c) => c.a.fact_id === f.id || c.b.fact_id === f.id).forEach((c) => push(out.to, { sel: { kind: "conflict", id: c.id }, label: short(c.subject ?? (c.a.fact_id === f.id ? c.b.claim : c.a.claim)), note: c.kind === "duplicate" ? "duplicate" : c.detector === "implication_scan" ? "found by reasoning" : "disagreement" }));
      data.assumptions.filter((a) => a.parents.some((p) => p.crystal_id === f.crystal_id)).forEach((a) => push(out.to, { sel: { kind: "assumption", id: a.id }, label: short(a.statement), note: a.state === "waiting" ? "assumption, waiting" : `assumption, ${a.state}` }));
      // Q99 C: what this fact implies, and the consequences it was checked against.
      (data.consequences ?? []).filter((q) => q.parent_fact_id === f.id).forEach((q) => push(out.to, { sel: { kind: "consequence", id: q.id }, label: short(q.statement), note: "it implies" }));
      (data.consequences ?? []).filter((q) => q.checks.some((c) => c.fact_id === f.id)).forEach((q) => {
        const v = q.checks.find((c) => c.fact_id === f.id)!.verdict;
        push(out.alongside, { sel: { kind: "consequence", id: q.id }, label: short(q.statement), note: v === "CONSISTENT" ? "agrees with this" : v === "CONTRADICTS" ? "contradicts this" : "checked, unrelated" });
      });
      data.recalls.filter((q) => q.touched.includes(f.id)).slice(0, LIMIT).forEach((q) => push(out.to, { sel: { kind: "recall", id: q.id }, label: short(q.question), note: "recalled it" }));
      return out;
    }
    case "source": {
      data.facts.filter((f) => f.doc_id === sel.id).slice(0, LIMIT * 2).forEach((f) => push(out.to, factRel(f)));
      const s = data.sources.find((x) => x.id === sel.id);
      if (s) push(out.alongside, { sel: { kind: "region", id: s.region }, label: regionName(s.region), note: "region" });
      return out;
    }
    case "bond": {
      const [a, b] = sel.id.split("|");
      factsOfCluster(a).slice(0, LIMIT).forEach((f) => push(out.from, factRel(f, regionName(f.region))));
      factsOfCluster(b).slice(0, LIMIT).forEach((f) => push(out.to, factRel(f, regionName(f.region))));
      return out;
    }
    case "conflict": {
      const c = data.conflicts.find((x) => x.id === sel.id); if (!c) return out;
      for (const side of [c.a, c.b]) {
        const f = data.facts.find((x) => x.id === side.fact_id);
        if (f) { push(out.from, factRel(f, side === c.a ? "A" : "B")); push(out.from, sourceRel(f.doc_id)); }
        else push(out.from, { sel: { kind: "fact", id: side.fact_id }, label: short(side.claim), note: side === c.a ? "A" : "B" });
      }
      return out;
    }
    case "consequence": {
      const q = (data.consequences ?? []).find((x) => x.id === sel.id); if (!q) return out;
      const pf = q.parent_fact_id ? data.facts.find((f) => f.id === q.parent_fact_id) : null;
      if (pf) { push(out.from, factRel(pf, "derived from")); push(out.from, sourceRel(pf.doc_id)); }
      q.checks.forEach((c) => {
        const f = data.facts.find((x) => x.id === c.fact_id);
        if (f) push(out.to, factRel(f, c.verdict === "CONSISTENT" ? "agrees" : c.verdict === "CONTRADICTS" ? "contradicts" : "unrelated"));
        if (c.verdict === "CONTRADICTS") {
          data.conflicts.filter((k) => (k.a.fact_id === c.fact_id && k.b.fact_id === q.parent_fact_id) || (k.b.fact_id === c.fact_id && k.a.fact_id === q.parent_fact_id))
            .forEach((k) => push(out.to, { sel: { kind: "conflict", id: k.id }, label: short(k.subject ?? k.a.claim), note: "found by reasoning" }));
        }
      });
      return out;
    }
    case "assumption": {
      const a = data.assumptions.find((x) => x.id === sel.id); if (!a) return out;
      a.parents.forEach((p) => {
        const facts = factsOfCluster(p.crystal_id);
        facts.slice(0, 4).forEach((f) => push(out.from, factRel(f, `from ${p.label}`)));
        facts.forEach((f) => push(out.from, sourceRel(f.doc_id)));
        if (!facts.length) push(out.from, { sel: { kind: "region", id: p.region ?? -1 }, label: p.label, note: "parent" });
      });
      return out;
    }
    case "gap": {
      const g = data.gaps.find((x) => x.id === sel.id); if (!g) return out;
      if (g.region != null) {
        push(out.alongside, { sel: { kind: "region", id: g.region }, label: regionName(g.region), note: "region" });
        data.facts.filter((f) => f.region === g.region).sort((x, y) => y.hits - x.hits).slice(0, 4).forEach((f) => push(out.alongside, factRel(f)));
      }
      data.runs.filter((r) => r.title && g.missing && r.title.toLowerCase().includes(g.missing.toLowerCase().slice(0, 24))).forEach((r) => push(out.to, { sel: { kind: "run", id: r.id }, label: short(r.title), note: "researched it" }));
      return out;
    }
    case "run": {
      const r = data.runs.find((x) => x.id === sel.id); if (!r) return out;
      if (r.result_crystal_id) factsOfCluster(r.result_crystal_id).slice(0, LIMIT).forEach((f) => push(out.to, factRel(f, "landed")));
      else if (r.region != null) push(out.to, { sel: { kind: "region", id: r.region }, label: regionName(r.region), note: "landed in" });
      return out;
    }
    case "recall": {
      const q = data.recalls.find((x) => x.id === sel.id); if (!q) return out;
      q.touched.forEach((id) => { const f = data.facts.find((x) => x.id === id); if (f) push(out.to, factRel(f, "answered with")); });
      return out;
    }
  }
}

/** The keys the scene lights for a selection: the selection, every relative, and the facts among them. */
export function focusOf(data: MindScene, sel: MindSelection): { keys: string[]; facts: string[] } {
  const rels = relationsOf(data, sel);
  const all = [...rels.from, ...rels.alongside, ...rels.to].map((r) => r.sel);
  const keys = new Set<string>([keyOf(sel), ...all.map(keyOf)]);
  const facts = new Set<string>(all.filter((s) => s.kind === "fact").map((s) => (s as { id: string }).id));
  if (sel.kind === "fact") facts.add(sel.id);
  if (sel.kind === "region") data.facts.filter((f) => f.region === sel.id).forEach((f) => facts.add(f.id));
  return { keys: [...keys], facts: [...facts] };
}

/** The starting place: a text search over everything the mind holds. */
export function searchMind(data: MindScene, query: string, limit = 12): Relation[] {
  const q = query.trim().toLowerCase();
  if (q.length < 2) return [];
  const hit = (s: string | null | undefined) => !!s && s.toLowerCase().includes(q);
  const out: Relation[] = [];
  const add = (r: Relation) => { if (out.length < limit) out.push(r); };
  data.regions.filter((r) => hit(r.name)).forEach((r) => add({ sel: { kind: "region", id: r.id }, label: r.name, note: `region · ${r.facts} facts` }));
  data.facts.filter((f) => hit(f.claim) || hit(f.key)).slice(0, limit).forEach((f) => add({ sel: { kind: "fact", id: f.id }, label: short(f.claim), note: `fact · ${f.key ? short(f.key, 40) : f.tier}` }));
  data.assumptions.filter((a) => hit(a.statement)).forEach((a) => add({ sel: { kind: "assumption", id: a.id }, label: short(a.statement), note: "assumption" }));
  (data.consequences ?? []).filter((q) => hit(q.statement)).forEach((q) => add({ sel: { kind: "consequence", id: q.id }, label: short(q.statement), note: "consequence" }));
  data.conflicts.filter((c) => hit(c.subject) || hit(c.a.claim) || hit(c.b.claim)).forEach((c) => add({ sel: { kind: "conflict", id: c.id }, label: short(c.subject ?? c.a.claim), note: c.kind === "duplicate" ? "duplicate" : "disagreement" }));
  data.gaps.filter((g) => hit(g.missing) || hit(g.subject)).forEach((g) => add({ sel: { kind: "gap", id: g.id }, label: short(g.missing), note: "gap" }));
  data.sources.filter((s) => hit(s.label)).forEach((s) => add({ sel: { kind: "source", id: s.id }, label: s.label, note: "source" }));
  data.runs.filter((r) => hit(r.title) || hit(r.summary)).forEach((r) => add({ sel: { kind: "run", id: r.id }, label: short(r.title), note: "run" }));
  data.recalls.filter((r) => hit(r.question)).forEach((r) => add({ sel: { kind: "recall", id: r.id }, label: short(r.question), note: "recall" }));
  return out;
}
