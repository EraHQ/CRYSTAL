// Settings › Curation (Refine PR 9, 5.8 item 5). The one per-workspace
// curation knob the server has: explore mode for assumptions
// (tri-state; null = the platform default, which is on). The spec's
// "Scan for conflicts nightly" and "Propose assumptions" switches have
// no per-tenant setting behind them (those are platform flags), so they
// are not shown. Shadow critic is admin-only and lives under Model &
// inference.
import { useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, errorMessage } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";
import type { SystemRule } from "@/lib/types";
import { ConfirmInline, ErrorBanner, Switch, useToast } from "@/components/ui";
import { fmtRelative } from "@/lib/utils";
import { BTN_PRIMARY, BTN_SECONDARY, INPUT, Section } from "./shared";

/** One rule in words (Q99 C, 2026-10-23). */
export function ruleText(r: SystemRule): string {
  if (r.rule_type === "approval") {
    const n = Number(r.conditions.min_consistent_edges ?? 1);
    const kind = String(r.selector.kind ?? "consequence");
    const what = kind === "any" ? "an assumption or consequence" : kind === "assumption" ? "an assumption" : "a consequence";
    const strict = r.conditions.no_contradicted_edges === false ? "" : " and nothing contradicts it";
    return `Approve ${what} once ${n} stated fact${n === 1 ? "" : "s"} agree${n === 1 ? "s" : ""} with it${strict}.`;
  }
  if (r.rule_type === "promotion") return `Let ${String(r.selector.origin ?? "background")} memory into recall when its conditions hold.`;
  return `${r.rule_type} rule.`;
}

// Automation (Q99 C + the first approval rule). Anthony: "control how
// automated you want the system to be and create approval rules." One
// rule type today: approve a consequence once enough stated facts agree
// with it. No rules means a person approves everything, as before.
function Automation({ customerId }: { customerId: string }) {
  const qc = useQueryClient();
  const toast = useToast();
  const [error, setError] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const [n, setN] = useState(2);
  const [strict, setStrict] = useState(true);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const rules = useQuery({ queryKey: ["rules", customerId], queryFn: () => api.listRules(customerId) });
  const refresh = () => qc.invalidateQueries({ queryKey: ["rules", customerId] });
  const create = useMutation({
    mutationFn: () => api.createRule(customerId, {
      rule_type: "approval", name: `Approve after ${n} agree`,
      selector: { kind: "consequence" }, conditions: { min_consistent_edges: n, no_contradicted_edges: strict }, action: { approve: true },
    }),
    onSuccess: () => { setError(null); setAdding(false); toast("Rule saved. It runs in the background from the next idle cycle."); refresh(); },
    onError: (e) => setError(errorMessage(e, "Couldn't save the rule.")),
  });
  const toggle = useMutation({
    mutationFn: ({ id, enabled }: { id: string; enabled: boolean }) => api.updateRule(customerId, id, { enabled }),
    onSuccess: (_o, v) => { setError(null); toast(v.enabled ? "Rule on." : "Rule off. A person approves until it is on again."); refresh(); },
    onError: (e) => setError(errorMessage(e, "Couldn't change the rule.")),
  });
  const remove = useMutation({
    mutationFn: (id: string) => api.deleteRule(customerId, id),
    onSuccess: () => { setError(null); setConfirmDelete(null); toast("Rule deleted."); refresh(); },
    onError: (e) => setError(errorMessage(e, "Couldn't delete the rule.")),
  });
  const list = rules.data?.rules ?? [];
  const busy = create.isPending || toggle.isPending || remove.isPending;
  return (
    <Section title="Automation" help={<>Rules that approve on your behalf. Without one, nothing is approved by the system: you approve in <Link to="/assumptions" className="text-accent hover:underline">Assumptions</Link>.</>}>
      {error && <ErrorBanner title="That didn't go through" message={error} className="mb-3" />}
      {rules.isError && <ErrorBanner title="Couldn't load rules" error={rules.error} onRetry={() => rules.refetch()} />}
      {!rules.isLoading && list.length === 0 && !adding && <p className="text-[13px] text-fg-3">No rules yet. You approve everything.</p>}
      <ul className="space-y-2" aria-label="Rules">
        {list.map((r) => (
          <li key={r.id} className="rounded-lg border border-line bg-inset px-3 py-2.5">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[13.5px] font-medium text-fg">{r.name}</span>
              <span className="text-[12px] text-fg-3">{r.fire_count ? `fired ${r.fire_count} time${r.fire_count === 1 ? "" : "s"}${r.last_fired_at ? `, last ${fmtRelative(r.last_fired_at)}` : ""}` : "never fired"}</span>
              <span className="flex-1" />
              <Switch checked={r.enabled} disabled={busy} onChange={(next) => toggle.mutate({ id: r.id, enabled: next })} label={r.enabled ? "On" : "Off"} />
              <button type="button" disabled={busy} onClick={() => setConfirmDelete(r.id)} className="text-[12.5px] text-fg-3 hover:text-bad disabled:opacity-50">Delete</button>
            </div>
            <p className="mt-1 text-[12.5px] text-fg-2">{ruleText(r)}</p>
            <ConfirmInline open={confirmDelete === r.id} message="Delete this rule?" confirmLabel="Delete" busy={remove.isPending}
              onConfirm={() => remove.mutate(r.id)} onCancel={() => setConfirmDelete(null)} />
          </li>
        ))}
      </ul>
      {adding ? (
        <form className="mt-3 rounded-lg border border-line px-3 py-3" onSubmit={(e) => { e.preventDefault(); create.mutate(); }} aria-label="New rule">
          <p className="text-[13px] text-fg">Approve a consequence once this many stated facts agree with it:</p>
          <label className="mt-2 block text-[12.5px] text-fg-2">
            <span className="sr-only">How many facts must agree</span>
            <input type="number" min={1} max={20} value={n} onChange={(e) => setN(Math.max(1, Math.min(20, Number(e.target.value) || 1)))} className={INPUT + " max-w-[120px]"} aria-label="How many facts must agree" />
          </label>
          <div className="mt-2">
            <Switch checked={strict} onChange={setStrict} label="Only if nothing contradicts it" hint="A consequence with a contradicting fact, or an open conflict, waits for you." />
          </div>
          <div className="mt-3 flex items-center gap-2">
            <button type="submit" disabled={busy} className={BTN_PRIMARY}>Save rule</button>
            <button type="button" disabled={busy} onClick={() => setAdding(false)} className={BTN_SECONDARY}>Cancel</button>
          </div>
        </form>
      ) : (
        <div className="mt-3">
          <button type="button" disabled={busy || rules.isLoading} onClick={() => setAdding(true)} className={BTN_SECONDARY}>Add a rule</button>
        </div>
      )}
    </Section>
  );
}

export function Curation() {
  const { selectedCustomerId } = useSelectedCustomer();
  const qc = useQueryClient();
  const toast = useToast();
  const [error, setError] = useState<string | null>(null);
  const record = useQuery({ queryKey: ["own-customer", selectedCustomerId], queryFn: () => api.getCustomer(selectedCustomerId!), enabled: !!selectedCustomerId });
  const explore = useMutation({
    mutationFn: (next: boolean | null) => api.setAssumptionsExplore(selectedCustomerId!, next),
    onSuccess: (_o, next) => {
      setError(null);
      toast(next === null ? "Back to the default." : next ? "Exploring beyond what you ask." : "Only pairs your questions connect are tested.");
      qc.invalidateQueries({ queryKey: ["own-customer", selectedCustomerId] });
    },
    onError: (e) => setError(errorMessage(e, "Couldn't save that.")),
  });
  const stored: boolean | null | undefined = record.data?.assumptions_explore;
  const on = stored ?? true;

  return (
    <div className="space-y-4">
      {error && <ErrorBanner title="That didn't go through" message={error} />}
      <Section title="Assumptions" help={<>Crystal proposes assumptions that bridge two things you have saved. They stay out of recall until you approve them in <Link to="/assumptions" className="text-accent hover:underline">Assumptions</Link>.</>}>
        <Switch checked={on} disabled={explore.isPending || record.isLoading} onChange={(next) => explore.mutate(next)} label="Explore beyond what you ask"
          hint="Also tests pairs found by structure alone (shared sources, similar topics), not only pairs your questions connected. Uses more capacity." />
        <div className="mt-3 flex items-center gap-3">
          <span className="text-[12px] text-fg-3">{stored == null ? "Using the default (on)." : "Set for this workspace."}</span>
          {stored != null && (
            <button type="button" disabled={explore.isPending} onClick={() => explore.mutate(null)} className={BTN_SECONDARY}>Reset to default</button>
          )}
        </div>
      </Section>
      {selectedCustomerId && <StewardSwitch customerId={selectedCustomerId} />}
      {selectedCustomerId && <Automation customerId={selectedCustomerId} />}
    </div>
  );
}

// The Steward (Q100 A, 2026-10-23): "Let Crystal run itself". One standing
// session per idle cycle works the queue with the curation toolset until
// the day's capacity is used; every action lands on Activity › Steward
// with a 24-hour undo; at the wall it asks for a refill.
function StewardSwitch({ customerId }: { customerId: string }) {
  const qc = useQueryClient();
  const toast = useToast();
  const [error, setError] = useState<string | null>(null);
  const status = useQuery({ queryKey: ["steward", customerId], queryFn: () => api.stewardStatus(customerId) });
  const flip = useMutation({
    mutationFn: (enabled: boolean) => api.setSteward(customerId, enabled),
    onSuccess: (_o, enabled) => {
      setError(null);
      toast(enabled ? "Crystal runs itself from the next idle cycle. Watch Activity › Steward." : "Crystal no longer runs itself. A person curates.");
      qc.invalidateQueries({ queryKey: ["steward", customerId] });
    },
    onError: (e) => setError(errorMessage(e, "Couldn't change that.")),
  });
  const on = status.data?.enabled ?? false;
  const paused = status.data?.paused ?? null;
  return (
    <Section title="Let Crystal run itself" help={<>The Steward works this bank on its own: settles conflicts the facts decide, approves what the bank agrees with, researches gaps it can answer, leaves the rest for you. Every action is on <Link to="/activity?view=steward" className="text-accent hover:underline">Activity › Steward</Link> with a 24-hour undo. It runs until the day's AI capacity is used, then asks for more.</>}>
      {error && <ErrorBanner title="That didn't go through" message={error} className="mb-3" />}
      <Switch checked={on} disabled={flip.isPending || status.isLoading} onChange={(next) => flip.mutate(next)} label="Let Crystal run itself"
        hint={on ? "On. One session each idle cycle, within your plan's capacity." : "Off. You approve, settle and research by hand."} />
      {on && paused && (
        <p className="mt-3 rounded-lg border border-warn/40 bg-warn-soft px-3 py-2 text-[12.5px] text-fg">
          <span className="font-medium text-warn">Paused today.</span> {paused.code === "session_cap" ? "Today's sessions are used; it resumes tomorrow." : "Today's AI capacity is used. Upgrade your plan, raise the cap, or wait for midnight UTC."}
        </p>
      )}
    </Section>
  );
}
