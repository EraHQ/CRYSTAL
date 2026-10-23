// Settings › Curation (Refine PR 9, 5.8 item 5). The one per-workspace
// curation knob the server has: explore mode for assumptions
// (tri-state; null = the platform default, which is on). The spec's
// "Scan for conflicts nightly" and "Propose assumptions" switches have
// no per-tenant setting behind them (those are platform flags), so they
// are not shown. Shadow critic is admin-only and lives under Model &
// inference. Automation (Q101, 2026-10-23): the ordered allow/deny rule
// list lives in ./Automation; the sentence for a rule in @/lib/rules.
import { useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, errorMessage } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";
import { ErrorBanner, Switch, useToast } from "@/components/ui";
import { BTN_SECONDARY, Section } from "./shared";
import { Automation } from "./Automation";

export { ruleText } from "@/lib/rules";

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
