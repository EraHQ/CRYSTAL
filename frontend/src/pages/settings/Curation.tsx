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
import { ErrorBanner, Switch, useToast } from "@/components/ui";
import { BTN_SECONDARY, Section } from "./shared";

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
    </div>
  );
}
