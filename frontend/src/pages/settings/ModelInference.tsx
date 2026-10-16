// Settings › Model & inference (Refine PR 9, 5.8 item 4). Inference mode
// with its consequence in plain words, today's AI capacity as a percent
// (dollars are admin-only, Q4=A), the provider key, the model as radio
// cards with the plan lock from /v1/me allowed_models (Q62=A: locked
// with Upgrade, never hidden), and the Autonomous research switch (the
// same auto_research budget row the Cognition page flips, Q61=A).
import { useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, errorMessage } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useSelectedCustomer } from "@/lib/selected-customer";
import { ErrorBanner, Meter, Switch, useToast } from "@/components/ui";
import { BACKGROUND_RESEARCH_ON_CAP } from "@/pages/Cognition";
import { BTN_PRIMARY, BTN_SECONDARY, ChoiceCard, INPUT, Section } from "./shared";

// Labels only; which of these a tenant may pick comes from /v1/me
// (allowed_models, per tier).
export const MANAGED_MODELS = [
  { value: "claude-haiku-4-5", label: "Haiku", body: "Fastest and lightest. Good for recall and quick answers." },
  { value: "claude-sonnet-5", label: "Sonnet", body: "Balanced speed and depth. The default." },
  { value: "claude-opus-4-8", label: "Opus", body: "Deepest reasoning for hard questions." },
];

const usd = (micro: number) => `$${(micro / 1_000_000).toFixed(2)}`;

export function ModelInference() {
  const { selectedCustomerId } = useSelectedCustomer();
  const { me } = useAuth();
  const qc = useQueryClient();
  const toast = useToast();
  const isPlatformAdmin = me?.role === "platform_admin" || me?.kind === "platform_admin_key";
  const [keyDraft, setKeyDraft] = useState("");
  const [modelDraft, setModelDraft] = useState("");
  const [capDraft, setCapDraft] = useState("");
  const [error, setError] = useState<string | null>(null);

  const spend = useQuery({ queryKey: ["customer-spend", selectedCustomerId], queryFn: () => api.customerSpend(selectedCustomerId!), enabled: !!selectedCustomerId });
  const record = useQuery({ queryKey: ["own-customer", selectedCustomerId], queryFn: () => api.getCustomer(selectedCustomerId!), enabled: !!selectedCustomerId });
  const budgets = useQuery({ queryKey: ["budgets", selectedCustomerId], queryFn: () => api.listBudgets(selectedCustomerId!), enabled: !!selectedCustomerId });
  const origins = useQuery({ queryKey: ["spend-origins", selectedCustomerId], queryFn: () => api.spendOrigins(selectedCustomerId!, 30), enabled: !!selectedCustomerId && isPlatformAdmin });

  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["customer-spend", selectedCustomerId] });
    qc.invalidateQueries({ queryKey: ["own-customer", selectedCustomerId] });
    qc.invalidateQueries({ queryKey: ["budgets", selectedCustomerId] });
  };
  const fail = (fallback: string) => (e: unknown) => setError(errorMessage(e, fallback));

  const setMode = useMutation({
    mutationFn: (m: "managed" | "byok") => api.setInferenceMode(selectedCustomerId!, m),
    onSuccess: (_o, m) => { setError(null); toast(m === "managed" ? "Inference now runs on Crystal's managed models." : "Inference now runs on your own provider key."); refresh(); },
    onError: fail("Couldn't change the inference mode."),
  });
  const saveKey = useMutation({
    mutationFn: () => api.updateUpstreamKey(selectedCustomerId!, keyDraft.trim()),
    onSuccess: () => { setError(null); setKeyDraft(""); toast("Provider key saved."); refresh(); },
    onError: fail("Couldn't save the provider key."),
  });
  const setModel = useMutation({
    mutationFn: (id: string) => api.setModel(selectedCustomerId!, id),
    onSuccess: () => { setError(null); setModelDraft(""); toast("Model updated."); refresh(); },
    onError: fail("Couldn't change the model."),
  });
  const research = useMutation({
    mutationFn: (on: boolean) => api.upsertBudget(selectedCustomerId!, "auto_research", on ? BACKGROUND_RESEARCH_ON_CAP : 0),
    onSuccess: (_o, on) => { setError(null); toast(on ? "Autonomous research is on." : "Autonomous research is off."); refresh(); },
    onError: fail("Couldn't change autonomous research."),
  });
  const saveCap = useMutation({
    mutationFn: () => {
      const dollars = parseFloat(capDraft);
      if (!isFinite(dollars) || dollars < 0) throw new Error("Enter a valid amount.");
      return api.upsertBudget(selectedCustomerId!, "auto_research", Math.round(dollars * 1_000_000));
    },
    onSuccess: () => { setError(null); setCapDraft(""); toast("Monthly cap saved."); refresh(); },
    onError: fail("Couldn't save the cap."),
  });

  const mode = spend.data?.inference_mode ?? "managed";
  const hasKey = (spend.data as { has_upstream_key?: boolean } | undefined)?.has_upstream_key === true;
  const currentModel: string = record.data?.model_id ?? record.data?.model_routing_config?.model_id ?? "";
  const allowed = me?.allowed_models;
  const pct = me?.usage?.ai_capacity_pct;
  const mtd = spend.data?.managed_month_to_date_micro_usd ?? 0;
  const cap = spend.data?.managed_monthly_cap_micro_usd ?? 0;
  const researchRow = budgets.data?.budgets?.find((b) => b.function === "auto_research" && !b.operator_id);
  const researchOn = (researchRow?.cap_micro_usd ?? 0) > 0;
  const shadow = budgets.data?.shadow_critic;
  const busy = setMode.isPending || saveKey.isPending || setModel.isPending || research.isPending || saveCap.isPending;

  return (
    <div className="space-y-4">
      {error && <ErrorBanner title="That didn't go through" message={error} />}

      <Section title="Inference" help="Where the model work runs.">
        <div role="radiogroup" aria-label="Inference" className="grid gap-2 min-[720px]:grid-cols-2">
          <ChoiceCard checked={mode === "managed"} onPick={() => mode !== "managed" && setMode.mutate("managed")} disabled={busy}
            title="Managed" body="Crystal pays for the model. Work counts toward your daily AI capacity." />
          <ChoiceCard checked={mode === "byok"} onPick={() => mode !== "byok" && setMode.mutate("byok")} disabled={busy || (!hasKey && mode !== "byok")}
            title="My own provider key" body="Work on your key does not count toward daily capacity."
            raw={!hasKey && mode !== "byok" ? "Save a provider key below first" : undefined} />
        </div>
        {typeof pct === "number" && (
          <Meter className="mt-4" value={pct} label="AI capacity used today" valueLabel={`${Math.round(pct)}%`} />
        )}
        {isPlatformAdmin && cap > 0 && (
          <p className="mt-2 text-[12.5px] text-fg-3">Admin view: managed spend this month {usd(mtd)} of {usd(cap)}.</p>
        )}
      </Section>

      <Section title="Provider API key" help="Stored encrypted and never shown again. Needed before switching to your own key.">
        <form className="flex flex-wrap gap-2" onSubmit={(e) => { e.preventDefault(); if (keyDraft.trim()) saveKey.mutate(); }}>
          <input type="password" aria-label="Provider API key" value={keyDraft} onChange={(e) => setKeyDraft(e.target.value)} placeholder="sk-ant-… or sk-…" className={INPUT + " min-w-0 flex-1"} />
          <button type="submit" disabled={!keyDraft.trim() || busy} className={BTN_PRIMARY}>{saveKey.isPending ? "Saving…" : "Save key"}</button>
        </form>
        <p className="mt-2 text-[12.5px] text-fg-2">{hasKey ? "A provider key is stored. Saving again replaces it." : "No provider key stored yet."}</p>
      </Section>

      <Section title="Model" help={mode === "managed" ? "Which of Crystal's managed models answers." : "Any model your provider key can serve."}>
        {mode === "managed" ? (
          <div role="radiogroup" aria-label="Model" className="grid gap-2 min-[720px]:grid-cols-3">
            {MANAGED_MODELS.map((m) => {
              const locked = Array.isArray(allowed) && !allowed.includes(m.value);
              return (
                <ChoiceCard key={m.value} checked={currentModel === m.value} disabled={busy || locked}
                  onPick={() => currentModel !== m.value && setModel.mutate(m.value)}
                  title={m.label} body={m.body} raw={m.value}
                  aside={locked ? <Link to="/billing" className="text-[12px] font-medium text-accent hover:underline" onClick={(e) => e.stopPropagation()}>Upgrade to use</Link> : undefined} />
              );
            })}
          </div>
        ) : (
          <form className="flex flex-wrap gap-2" onSubmit={(e) => { e.preventDefault(); if (modelDraft.trim()) setModel.mutate(modelDraft.trim()); }}>
            <input aria-label="Model id" value={modelDraft || currentModel} onChange={(e) => setModelDraft(e.target.value)} placeholder="model id" className={INPUT + " min-w-0 flex-1 font-mono"} />
            <button type="submit" disabled={busy || !modelDraft.trim() || modelDraft === currentModel} className={BTN_SECONDARY}>Save model</button>
          </form>
        )}
      </Section>

      <Section title="Autonomous research">
        <Switch checked={researchOn} disabled={busy || budgets.isLoading} onChange={(on) => research.mutate(on)} label="Autonomous research"
          hint="Crystal researches knowledge gaps on its own in the background. It stops when today's capacity is used up; off means gaps wait for you to press Research." />
        {isPlatformAdmin && (
          <form className="mt-3 flex flex-wrap items-center gap-2" onSubmit={(e) => { e.preventDefault(); saveCap.mutate(); }}>
            <label className="text-[12.5px] text-fg-3" htmlFor="research-cap">Admin: monthly cap</label>
            <span className="text-[12.5px] text-fg-3">$</span>
            <input id="research-cap" value={capDraft} onChange={(e) => setCapDraft(e.target.value)} placeholder={researchOn ? ((researchRow?.cap_micro_usd ?? 0) / 1_000_000).toFixed(2) : "5.00"} className={INPUT + " w-24"} />
            <button type="submit" disabled={busy || !capDraft} className={BTN_SECONDARY}>Save cap</button>
          </form>
        )}
      </Section>

      {isPlatformAdmin && (
        <Section title="Admin: spend" help="Platform view only; tenants never see dollars.">
          {!(origins.data?.origins?.length) ? (
            <p className="text-[12.5px] text-fg-3">No ledgered calls in the last 30 days.</p>
          ) : (
            <table className="w-full text-[12.5px]">
              <thead>
                <tr className="text-left text-fg-3">
                  <th className="py-1 font-medium">Origin</th>
                  <th className="py-1 text-right font-medium">Calls</th>
                  <th className="py-1 text-right font-medium">Cache reads</th>
                  <th className="py-1 text-right font-medium">Cost</th>
                </tr>
              </thead>
              <tbody>
                {origins.data.origins.map((o: { origin: string; call_count: number; cache_read_tokens?: number; cost_micro_usd: number }) => (
                  <tr key={o.origin} className="border-t border-line text-fg">
                    <td className="py-1.5 font-mono">{o.origin}</td>
                    <td className="py-1.5 text-right">{o.call_count.toLocaleString()}</td>
                    <td className="py-1.5 text-right">{(o.cache_read_tokens ?? 0).toLocaleString()}</td>
                    <td className="py-1.5 text-right">${(o.cost_micro_usd / 1_000_000).toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {shadow && (
            <p className="mt-3 text-[12.5px] text-fg-2">
              Shadow critic: cap/day {shadow.effective_max_per_day ?? "—"}{shadow.max_per_day == null ? " (platform default)" : " (custom)"} · runs (24h) {shadow.runs_last_24h} · spend (24h) ${((shadow.cost_micro_usd_last_24h ?? 0) / 1_000_000).toFixed(4)}
            </p>
          )}
        </Section>
      )}
    </div>
  );
}
