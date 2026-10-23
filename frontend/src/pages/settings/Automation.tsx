// Settings › Curation › Automation (Q101, 2026-10-23). Anthony: "It should
// be easy like criteria met then approve fact. Or I could exclude on x
// facts from automatically whitelisting. Think of a networking allow and
// deny rules." An ORDERED list: rules run top to bottom over everything
// waiting for a decision; the first match decides; Allow approves, Deny
// holds for a person; the implicit last line is the default. Each rule
// is one sentence, built from four parts (Applies to, Matching, When,
// Then) and shown as that sentence.
import { useState } from "react";
import type { ReactNode } from "react";
import { ArrowDown, ArrowUp } from "lucide-react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, errorMessage } from "@/lib/api";
import {
  DEFAULT_LINE, KIND_OPTIONS, SOURCE_OPTIONS, TIER_OPTIONS, draftText, draftToRule, emptyDraft, ruleText, ruleToDraft,
} from "@/lib/rules";
import type { RuleDecision, RuleDraft, RuleKind, RuleSource, RuleTier } from "@/lib/rules";
import type { SystemRule } from "@/lib/types";
import { ConfirmInline, ErrorBanner, IconButton, Segmented, StatusPill, Switch, useToast } from "@/components/ui";
import { cn, fmtRelative } from "@/lib/utils";
import { BTN_PRIMARY, BTN_SECONDARY, BTN_QUIET, INPUT, Section } from "./shared";

const NUM = INPUT + " w-16 px-2 py-1 text-[12.5px]";

function Check({ checked, onChange, children, disabled }: { checked: boolean; onChange: (v: boolean) => void; children: ReactNode; disabled?: boolean }) {
  return (
    <label className={cn("flex items-center gap-2 text-[13px] text-fg", disabled && "opacity-60")}>
      <input type="checkbox" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)}
        className="h-4 w-4 shrink-0 rounded border-line-strong accent-accent" />
      <span className="flex flex-wrap items-center gap-1.5">{children}</span>
    </label>
  );
}

function Num({ value, onChange, label, min = 1, max = 99 }: { value: number; onChange: (n: number) => void; label: string; min?: number; max?: number }) {
  return (
    <input type="number" min={min} max={max} value={value} aria-label={label} className={NUM}
      onChange={(e) => onChange(Math.max(min, Math.min(max, Number(e.target.value) || min)))} />
  );
}

function Part({ title, hint, children }: { title: string; hint?: string; children: ReactNode }) {
  return (
    <fieldset className="min-w-0">
      <legend className="text-[12px] font-semibold uppercase tracking-wide text-fg-3">{title}</legend>
      {hint && <p className="mt-0.5 text-[12px] text-fg-3">{hint}</p>}
      <div className="mt-2">{children}</div>
    </fieldset>
  );
}

/** The four-part builder; the sentence it will save is always in view. */
export function RuleForm({ initial, label, busy, onSave, onCancel }: {
  initial: RuleDraft; label: string; busy: boolean;
  onSave: (draft: RuleDraft) => void; onCancel: () => void;
}) {
  const [d, setD] = useState<RuleDraft>(initial);
  const set = (patch: Partial<RuleDraft>) => setD((prev) => ({ ...prev, ...patch }));
  const allow = d.decision === "allow";
  return (
    <form aria-label={label} className="mt-3 space-y-4 rounded-lg border border-line bg-surface px-3 py-3"
      onSubmit={(e) => { e.preventDefault(); onSave(d); }}>
      <Part title="Applies to">
        <Segmented size="sm" label="Applies to" value={d.kind} onChange={(v) => set({ kind: v as RuleKind })} options={KIND_OPTIONS} />
      </Part>
      <Part title="Matching" hint="Optional. Narrow it to some of them.">
        <div className="grid gap-2 min-[560px]:grid-cols-3">
          <label className="block text-[12.5px] text-fg-2">Key starts with
            <input value={d.keyPrefix} onChange={(e) => set({ keyPrefix: e.target.value })} placeholder="Finance|" maxLength={120} className={INPUT + " mt-1"} />
          </label>
          <label className="block text-[12.5px] text-fg-2">Source
            <select value={d.source} onChange={(e) => set({ source: e.target.value as RuleSource | "" })} className={INPUT + " mt-1"}>
              {SOURCE_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
          </label>
          <label className="block text-[12.5px] text-fg-2">Current tier
            <select value={d.tier} onChange={(e) => set({ tier: e.target.value as RuleTier | "" })} className={INPUT + " mt-1"}>
              {TIER_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
          </label>
        </div>
      </Part>
      <Part title="When" hint="All of these must hold. None means the rule always matches what it applies to.">
        <div className="space-y-2">
          <Check checked={d.agree} onChange={(v) => set({ agree: v })}>
            <Num value={d.agreeN} onChange={(n) => set({ agreeN: n })} label="How many facts must agree" /> stated {d.agreeN === 1 ? "fact agrees" : "facts agree"} with it
          </Check>
          <Check checked={d.noContradictions} onChange={(v) => set({ noContradictions: v })}>nothing contradicts it</Check>
          <Check checked={d.cited} onChange={(v) => set({ cited: v })}>
            it has been cited at least <Num value={d.citedN} onChange={(n) => set({ citedN: n })} label="How many citations" /> {d.citedN === 1 ? "time" : "times"}
          </Check>
          <Check checked={d.parentTrusted} onChange={(v) => set({ parentTrusted: v })}>its parent is trusted</Check>
          <Check checked={d.confident} onChange={(v) => set({ confident: v })}>
            its confidence is at least <Num value={d.confidencePct} onChange={(n) => set({ confidencePct: n })} label="Confidence percent" min={0} max={100} />%
          </Check>
          <Check checked={d.waited} onChange={(v) => set({ waited: v })}>
            it has waited at least <Num value={d.waitedDays} onChange={(n) => set({ waitedDays: n })} label="How many days" min={0} max={365} /> {d.waitedDays === 1 ? "day" : "days"}
          </Check>
          <Check checked={d.scanPassed} onChange={(v) => set({ scanPassed: v })}>the review scan passed</Check>
        </div>
      </Part>
      <Part title="Then">
        <div className="flex flex-wrap items-start gap-x-4 gap-y-2">
          <Segmented size="sm" label="Decision" value={d.decision} onChange={(v) => set({ decision: v as RuleDecision, noContradictions: v === "allow" })}
            options={[{ value: "allow", label: "Allow" }, { value: "deny", label: "Deny" }]} />
          {allow ? (
            <Switch checked={d.trust} onChange={(v) => set({ trust: v })} label="And trust it"
              hint="A fact is let in at the neutral tier. An approved assumption takes its parent's standing." />
          ) : (
            <p className="text-[12.5px] text-fg-3">Held for a person. Nothing below this line is consulted.</p>
          )}
        </div>
      </Part>
      <p className="rounded-md bg-inset px-3 py-2 text-[13px] text-fg" aria-live="polite">{draftText(d)}</p>
      <div className="flex items-center gap-2">
        <button type="submit" disabled={busy} className={BTN_PRIMARY}>Save rule</button>
        <button type="button" disabled={busy} onClick={onCancel} className={BTN_SECONDARY}>Cancel</button>
      </div>
    </form>
  );
}

function fireText(r: SystemRule): string {
  if (!r.enabled) return "Off: this line is skipped.";
  if (r.action.decision === "deny") return "Holds for you.";
  if (!r.fire_count) return "Never fired.";
  return `Fired ${r.fire_count} time${r.fire_count === 1 ? "" : "s"}${r.last_fired_at ? `, last ${fmtRelative(r.last_fired_at)}` : ""}.`;
}

export function Automation({ customerId }: { customerId: string }) {
  const qc = useQueryClient();
  const toast = useToast();
  const [error, setError] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const [editing, setEditing] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const rules = useQuery({ queryKey: ["rules", customerId], queryFn: () => api.listRules(customerId) });
  const refresh = () => qc.invalidateQueries({ queryKey: ["rules", customerId] });
  const fail = (what: string) => (e: unknown) => setError(errorMessage(e, what));
  const create = useMutation({
    mutationFn: (draft: RuleDraft) => api.createRule(customerId, { rule_type: "curation", name: draftText(draft).slice(0, 120), ...draftToRule(draft) }),
    onSuccess: () => { setError(null); setAdding(false); toast("Rule added at the bottom. It runs in the background from the next idle cycle."); refresh(); },
    onError: fail("Couldn't save the rule."),
  });
  const update = useMutation({
    mutationFn: ({ id, draft }: { id: string; draft: RuleDraft }) => api.updateRule(customerId, id, { name: draftText(draft).slice(0, 120), ...draftToRule(draft) }),
    onSuccess: () => { setError(null); setEditing(null); toast("Rule changed."); refresh(); },
    onError: fail("Couldn't change the rule."),
  });
  const toggle = useMutation({
    mutationFn: ({ id, enabled }: { id: string; enabled: boolean }) => api.updateRule(customerId, id, { enabled }),
    onSuccess: (_o, v) => { setError(null); toast(v.enabled ? "Rule on." : "Rule off. That line is skipped until it is on again."); refresh(); },
    onError: fail("Couldn't change the rule."),
  });
  const move = useMutation({
    mutationFn: ({ id, direction }: { id: string; direction: "up" | "down" }) => api.moveRule(customerId, id, direction),
    onSuccess: (data) => { setError(null); qc.setQueryData(["rules", customerId], data); },
    onError: fail("Couldn't move the rule."),
  });
  const remove = useMutation({
    mutationFn: (id: string) => api.deleteRule(customerId, id),
    onSuccess: () => { setError(null); setConfirmDelete(null); toast("Rule deleted."); refresh(); },
    onError: fail("Couldn't delete the rule."),
  });
  const list = rules.data?.rules ?? [];
  const busy = create.isPending || update.isPending || toggle.isPending || move.isPending || remove.isPending;
  return (
    <Section title="Automation" help={<>Rules run top to bottom on everything waiting for a decision: memory held out of recall, assumptions, consequences.
        The first rule that matches decides. <span className="font-medium text-fg">Allow</span> approves it; <span className="font-medium text-fg">Deny</span> holds it for you.</>}>
      <div>
        {error && <ErrorBanner title="That didn't go through" message={error} className="mb-3" />}
        {rules.isError && <ErrorBanner title="Couldn't load rules" error={rules.error} onRetry={() => rules.refetch()} />}
        <ol className="space-y-2" aria-label="Rules">
          {list.map((r, i) => {
            const deny = r.action.decision === "deny";
            return (
              <li key={r.id} className={cn("rounded-lg border border-line bg-inset px-3 py-2.5", !r.enabled && "opacity-70")}>
                <div className="flex items-start gap-2.5">
                  <span aria-hidden className="mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-surface text-[11px] font-semibold text-fg-2 ring-1 ring-inset ring-line">{i + 1}</span>
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <StatusPill variant={deny ? "warn" : "ok"} raw={deny ? "deny" : "allow"}>{deny ? "Deny" : "Allow"}</StatusPill>
                      <span className="min-w-0 text-[13.5px] text-fg">{r.text ?? ruleText(r)}</span>
                    </div>
                    <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1.5">
                      <span className="text-[12px] text-fg-3">{fireText(r)}</span>
                      <span className="flex-1" />
                      <span className="inline-flex items-center gap-0.5">
                        <IconButton size="sm" label="Move up" icon={ArrowUp} disabled={busy || i === 0} onClick={() => move.mutate({ id: r.id, direction: "up" })} />
                        <IconButton size="sm" label="Move down" icon={ArrowDown} disabled={busy || i === list.length - 1} onClick={() => move.mutate({ id: r.id, direction: "down" })} />
                      </span>
                      <Switch checked={r.enabled} disabled={busy} onChange={(next) => toggle.mutate({ id: r.id, enabled: next })} label={r.enabled ? "On" : "Off"} />
                      <button type="button" disabled={busy} onClick={() => { setAdding(false); setEditing(editing === r.id ? null : r.id); }} className={BTN_QUIET}>Edit</button>
                      <button type="button" disabled={busy} onClick={() => setConfirmDelete(r.id)} className="text-[12.5px] text-fg-3 hover:text-bad disabled:opacity-50">Delete</button>
                    </div>
                  </div>
                </div>
                <ConfirmInline open={confirmDelete === r.id} message="Delete this rule? The lines below it move up." confirmLabel="Delete" busy={remove.isPending}
                  onConfirm={() => remove.mutate(r.id)} onCancel={() => setConfirmDelete(null)} />
                {editing === r.id && (
                  <RuleForm key={r.id} initial={ruleToDraft(r)} label="Edit rule" busy={busy}
                    onSave={(draft) => update.mutate({ id: r.id, draft })} onCancel={() => setEditing(null)} />
                )}
              </li>
            );
          })}
          <li className="flex items-start gap-2.5 rounded-lg border border-dashed border-line px-3 py-2.5" aria-label="Default">
            <span aria-hidden className="mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[11px] font-semibold text-fg-3 ring-1 ring-inset ring-line">{list.length + 1}</span>
            <span className="text-[13.5px] text-fg-2">{rules.data?.default ?? DEFAULT_LINE}{list.length === 0 && !rules.isLoading ? " No rules yet, so you decide everything." : ""}</span>
          </li>
        </ol>
        {adding ? (
          <RuleForm initial={emptyDraft()} label="New rule" busy={busy} onSave={(draft) => create.mutate(draft)} onCancel={() => setAdding(false)} />
        ) : (
          <div className="mt-3">
            <button type="button" disabled={busy || rules.isLoading} onClick={() => { setEditing(null); setAdding(true); }} className={BTN_SECONDARY}>Add a rule</button>
          </div>
        )}
      </div>
    </Section>
  );
}
