// Automation rules in words (Q101, 2026-10-23). The server's
// system_rules/curation.py::rule_text says the same sentence; both are
// pinned so the console never shows a rule the engine reads differently.
import type { SystemRule } from "@/lib/types";

export type RuleKind = "fact" | "assumption" | "consequence" | "any";
export type RuleSource = "document" | "chat" | "research" | "reasoning";
export type RuleTier = "quarantine" | "neutral" | "whitelist";
export type RuleDecision = "allow" | "deny";

export const DEFAULT_LINE = "Otherwise, a person decides.";

export const KIND_OPTIONS: Array<{ value: RuleKind; label: string }> = [
  { value: "any", label: "Anything" }, { value: "fact", label: "Facts" },
  { value: "assumption", label: "Assumptions" }, { value: "consequence", label: "Consequences" },
];
export const SOURCE_OPTIONS: Array<{ value: RuleSource | ""; label: string }> = [
  { value: "", label: "Any source" }, { value: "document", label: "A document" }, { value: "chat", label: "Chat" },
  { value: "research", label: "Research" }, { value: "reasoning", label: "Found by reasoning" },
];
export const TIER_OPTIONS: Array<{ value: RuleTier | ""; label: string }> = [
  { value: "", label: "Any tier" }, { value: "quarantine", label: "Quarantine" }, { value: "neutral", label: "Neutral" }, { value: "whitelist", label: "Whitelist" },
];

/** The builder's state: one of each part, in the words the form shows. */
export interface RuleDraft {
  kind: RuleKind;
  keyPrefix: string;
  source: RuleSource | "";
  tier: RuleTier | "";
  agree: boolean; agreeN: number;
  noContradictions: boolean;
  cited: boolean; citedN: number;
  parentTrusted: boolean;
  confident: boolean; confidencePct: number;
  waited: boolean; waitedDays: number;
  scanPassed: boolean;
  decision: RuleDecision;
  trust: boolean;
}

export function emptyDraft(decision: RuleDecision = "allow"): RuleDraft {
  return {
    kind: "any", keyPrefix: "", source: "", tier: "",
    agree: false, agreeN: 2, noContradictions: decision === "allow", cited: false, citedN: 1, parentTrusted: false,
    confident: false, confidencePct: 80, waited: false, waitedDays: 7, scanPassed: false,
    decision, trust: false,
  };
}

/** The rule JSON the server validates (strict: only the parts in use are sent). */
export function draftToRule(d: RuleDraft): { selector: Record<string, unknown>; conditions: Record<string, unknown>; action: Record<string, unknown> } {
  const selector: Record<string, unknown> = { kind: d.kind };
  if (d.keyPrefix.trim()) selector.key_prefix = d.keyPrefix.trim();
  if (d.source) selector.source = d.source;
  if (d.tier) selector.tier = d.tier;
  const conditions: Record<string, unknown> = {};
  if (d.agree) conditions.min_agreeing_facts = Math.max(1, Math.round(d.agreeN));
  if (d.noContradictions) conditions.no_contradictions = true;
  if (d.cited) conditions.min_citations = Math.max(1, Math.round(d.citedN));
  if (d.parentTrusted) conditions.parent_trusted = true;
  if (d.confident) conditions.min_confidence = Math.max(0, Math.min(100, d.confidencePct)) / 100;
  if (d.waited) conditions.waited_days = Math.max(0, Math.round(d.waitedDays));
  if (d.scanPassed) conditions.scan_passed = true;
  const action: Record<string, unknown> = { decision: d.decision };
  if (d.decision === "allow" && d.trust) action.tier = "neutral";
  return { selector, conditions, action };
}

/** A saved rule back into the builder (edit in place). */
export function ruleToDraft(r: SystemRule): RuleDraft {
  const f = foldLegacy(r);
  const sel = f.selector, cond = f.conditions, act = f.action;
  const d = emptyDraft(act.decision === "deny" ? "deny" : "allow");
  d.kind = (["fact", "assumption", "consequence", "any"].includes(String(sel.kind)) ? sel.kind : "any") as RuleKind;
  d.keyPrefix = typeof sel.key_prefix === "string" ? sel.key_prefix : "";
  d.source = (["document", "chat", "research", "reasoning"].includes(String(sel.source)) ? sel.source : "") as RuleSource | "";
  d.tier = (["quarantine", "neutral", "whitelist"].includes(String(sel.tier)) ? sel.tier : "") as RuleTier | "";
  if (typeof cond.min_agreeing_facts === "number") { d.agree = true; d.agreeN = cond.min_agreeing_facts; }
  d.noContradictions = cond.no_contradictions === true;
  if (typeof cond.min_citations === "number") { d.cited = true; d.citedN = cond.min_citations; }
  d.parentTrusted = cond.parent_trusted === true;
  if (typeof cond.min_confidence === "number") { d.confident = true; d.confidencePct = Math.round(cond.min_confidence * 100); }
  if (typeof cond.waited_days === "number") { d.waited = true; d.waitedDays = cond.waited_days; }
  d.scanPassed = cond.scan_passed === true;
  d.trust = act.tier === "neutral";
  return d;
}

/** The older promotion / approval shapes as a curation rule (the server folds them on read; this covers a cached list). */
export function foldLegacy(r: SystemRule): SystemRule {
  if (r.rule_type === "promotion") {
    const selector: Record<string, unknown> = { kind: "fact" };
    if (r.selector.origin === "background_worker") selector.source = "research";
    const conditions: Record<string, unknown> = {};
    if (r.conditions.outbound_scan_passed === true) conditions.scan_passed = true;
    if (r.conditions.no_open_conflicts === true) conditions.no_contradictions = true;
    if (typeof r.conditions.min_grounded_citations === "number" && r.conditions.min_grounded_citations > 0) conditions.min_citations = r.conditions.min_grounded_citations;
    const action: Record<string, unknown> = { decision: "allow" };
    if (r.action.set_tier === "neutral") action.tier = "neutral";
    return { ...r, rule_type: "curation", selector, conditions, action };
  }
  if (r.rule_type === "approval") {
    const conditions: Record<string, unknown> = { min_agreeing_facts: Number(r.conditions.min_consistent_edges ?? 1) };
    if (r.conditions.no_contradicted_edges !== false) conditions.no_contradictions = true;
    return { ...r, rule_type: "curation", selector: { kind: r.selector.kind ?? "consequence" }, conditions, action: { decision: "allow" } };
  }
  return r;
}

const SUBJECT: Record<string, string> = { fact: "a fact", assumption: "an assumption", consequence: "a consequence", any: "anything" };
const SOURCE_WORDS: Record<string, string> = { document: "from a document", chat: "from chat", research: "from research", reasoning: "found by reasoning" };
const plural = (n: number, one: string, many: string) => (n === 1 ? one : many);

/** One rule in words. */
export function ruleText(input: SystemRule): string {
  const r = foldLegacy(input);
  if (r.rule_type !== "curation") return `${r.rule_type} rule.`;
  const sel = r.selector, cond = r.conditions, act = r.action;
  const kind = String(sel.kind ?? "any");
  const deny = act.decision === "deny";
  const verb = deny ? "Hold" : kind === "fact" ? "Let in" : "Approve";
  const parts = [verb, SUBJECT[kind] ?? "anything"];
  if (typeof sel.key_prefix === "string" && sel.key_prefix) parts.push(`whose key starts with "${sel.key_prefix}"`);
  if (typeof sel.source === "string" && SOURCE_WORDS[sel.source]) parts.push(SOURCE_WORDS[sel.source]);
  if (typeof sel.tier === "string" && sel.tier) parts.push(`in the ${sel.tier} tier`);
  if (deny) parts.push("for a person");
  const when: string[] = [];
  if (typeof cond.min_agreeing_facts === "number") { const n = cond.min_agreeing_facts; when.push(`${n} stated ${plural(n, "fact agrees", "facts agree")} with it`); }
  if (cond.no_contradictions === true) when.push("nothing contradicts it");
  if (typeof cond.min_citations === "number") { const n = cond.min_citations; when.push(`it has been cited at least ${n} ${plural(n, "time", "times")}`); }
  if (cond.parent_trusted === true) when.push("its parent is trusted");
  if (typeof cond.min_confidence === "number") when.push(`its confidence is at least ${Math.round(cond.min_confidence * 100)}%`);
  if (typeof cond.waited_days === "number") { const n = cond.waited_days; when.push(`it has waited at least ${n} ${plural(n, "day", "days")}`); }
  if (cond.scan_passed === true) when.push("the review scan passed");
  let text = parts.join(" ");
  if (when.length) text += " when " + (when.length === 1 ? when[0] : `${when.slice(0, -1).join(", ")} and ${when[when.length - 1]}`);
  if (!deny && act.tier === "neutral") text += ", and trust it";
  return text + ".";
}

export const draftText = (d: RuleDraft): string =>
  ruleText({ id: "", rule_type: "curation", name: "", enabled: true, priority: 0, fire_count: 0, last_fired_at: null, created_at: null, ...draftToRule(d) });
