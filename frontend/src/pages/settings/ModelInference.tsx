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
import type { CatalogModel, UpstreamProvider } from "@/lib/types";
import { useAuth } from "@/lib/auth";
import { useSelectedCustomer } from "@/lib/selected-customer";
import { ErrorBanner, Meter, Switch, useToast } from "@/components/ui";
import { BACKGROUND_RESEARCH_ON_CAP } from "@/pages/Cognition";
import { BTN_PRIMARY, BTN_SECONDARY, ChoiceCard, INPUT, Section } from "./shared";

// Labels only; which of these a tenant may pick comes from /v1/me
// (allowed_models, per tier).
// Q88=A (2026-10-17): the 5.5 line, Fable on top.
export const MANAGED_MODELS = [
  { value: "claude-haiku-5-5", label: "Haiku", body: "Fastest and lightest. Good for recall and quick answers." },
  { value: "claude-sonnet-5-5", label: "Sonnet", body: "Balanced speed and depth. The default." },
  { value: "claude-opus-5-5", label: "Opus", body: "Deep reasoning for hard questions." },
  { value: "claude-fable-5-1", label: "Fable", body: "Anthropic's most capable model. For the hardest questions." },
];

// A tenant who picked before the move to the 5.5 line still stores the
// old id until the server rewrites it; the card it maps to stays selected.
export const LEGACY_MODEL_ALIASES: Record<string, string> = {
  "claude-haiku-4-5": "claude-haiku-5-5",
  "claude-haiku-4-5-20251001": "claude-haiku-5-5",
  "claude-sonnet-5": "claude-sonnet-5-5",
  "claude-sonnet-4-6": "claude-sonnet-5-5",
  "claude-sonnet-4-5-20250929": "claude-sonnet-5-5",
  "claude-opus-5": "claude-opus-5-5",
  "claude-opus-4-8": "claude-opus-5-5",
  "claude-opus-4-7": "claude-opus-5-5",
  "claude-fable-5": "claude-fable-5-1",
};
export const canonicalModel = (id: string): string => LEGACY_MODEL_ALIASES[id] ?? id;

// M3 (2026-10-19): NVIDIA Nemotron on Nebius Token Factory. The exact ids
// are verified server-side against Token Factory's /v1/models and reach
// the console only through /v1/me allowed_models, so the cards are
// derived from the ids the platform offers, never hard-coded here.
export const NEMOTRON_FAMILY_LABEL = "NVIDIA Nemotron on Nebius Token Factory";
export const familyOf = (id: string): string | null =>
  id.startsWith("claude-") ? "claude" : id.toLowerCase().startsWith("nvidia/") ? "nemotron" : null;

export function nemotronLabel(id: string): string {
  const s = id.toLowerCase();
  if (s.includes("lightning")) return "Nemotron Lightning";
  if (s.includes("nano")) return "Nemotron Nano";
  if (s.includes("super")) return "Nemotron Super";
  if (s.includes("ultra")) return "Nemotron Ultra";
  return id.replace(/^nvidia\//i, "");
}
function nemotronBody(id: string): string {
  const s = id.toLowerCase();
  if (s.includes("lightning") || s.includes("nano")) return `Fast and cheap open model. ${NEMOTRON_FAMILY_LABEL}.`;
  if (s.includes("ultra")) return `NVIDIA's largest open model. ${NEMOTRON_FAMILY_LABEL}.`;
  return `Capable open model, low cost. ${NEMOTRON_FAMILY_LABEL}.`;
}
/** The Nemotron cards the platform offers this tenant (none when the lane is off). */
export function nemotronCards(allowed: string[] | undefined | null): { value: string; label: string; body: string }[] {
  return (allowed ?? []).filter((id) => familyOf(id) === "nemotron").map((id) => ({ value: id, label: nemotronLabel(id), body: nemotronBody(id) }));
}
/** The light model a family runs quick background steps on (mirrors llm/lanes.py family_small). */
export function familySmall(fam: string | null, allowed: string[] | undefined | null): string | null {
  if (fam === "claude") return "claude-haiku-5-5";
  if (fam === "nemotron") {
    const ids = (allowed ?? []).filter((id) => familyOf(id) === "nemotron");
    return ids.find((id) => /lightning|nano/i.test(id)) ?? ids[0] ?? null;
  }
  return null;
}
// Kept for the Anthropic family (older callers and tests).
export const FAMILY_SMALL: Record<string, string> = { claude: "claude-haiku-5-5" };
/** A model id in the user's words — "Sonnet", "Nemotron Super" — here, in
 * the chat trace and in the Query log; unknown ids show as recorded, none as "". */
export const modelLabel = (id: string | null | undefined): string =>
  !id ? "" : MANAGED_MODELS.find((m) => m.value === canonicalModel(id))?.label ?? (familyOf(id) === "nemotron" ? nemotronLabel(id) : id);

// What a background pick runs where, in words (the "What runs where" row).
export function backgroundSummary(backgroundId: string | null | undefined, allowed?: string[] | null): string {
  if (!backgroundId) return "Crystal's tier map (Haiku for small jobs, Sonnet for the rest)";
  const fam = familyOf(backgroundId);
  const small = familySmall(fam, allowed);
  const label = modelLabel(backgroundId);
  return small && canonicalModel(backgroundId) !== small ? `${label}, quick steps on ${modelLabel(small)}` : label;
}

const usd = (micro: number) => `$${(micro / 1_000_000).toFixed(2)}`;

// Item 2 (2026-10-22): the providers a tenant's own key can belong to.
export const PROVIDERS: { value: UpstreamProvider; label: string; placeholder: string }[] = [
  { value: "anthropic", label: "Anthropic", placeholder: "sk-ant-…" },
  { value: "openai", label: "OpenAI", placeholder: "sk-…" },
  { value: "nebius", label: "Nebius Token Factory", placeholder: "Token Factory API key" },
  { value: "self_hosted", label: "Self-hosted (OpenAI-compatible)", placeholder: "API key" },
];
export const providerLabel = (p: string | null | undefined): string => PROVIDERS.find((x) => x.value === p)?.label ?? (p || "");

// Hackathon item 1 (2026-10-22): any text model Token Factory lists is a
// pick. The catalog row carries a context length and a cost band (never
// dollars) and, once picked, what the probe found (JSON, Tools).
export const ctxLabel = (n: number): string => (n >= 1_000_000 ? `${Math.round(n / 1_000_000)}M` : n >= 1000 ? `${Math.round(n / 1024)}k` : `${n}`);
export const BAND_WORDS: Record<CatalogModel["band"], string> = { light: "light cost", standard: "standard cost", heavy: "heavy cost" };
/** "DeepSeek-V4-Pro" for "deepseek-ai/DeepSeek-V4-Pro"; Nemotron ids keep their short names. */
export const catalogLabel = (m: CatalogModel): string =>
  familyOf(m.id) === "nemotron" && m.nvidia ? nemotronLabel(m.id) + (m.fast ? " (fast)" : "") : (m.name || m.id.split("/").pop() || m.id);
export const catalogOption = (m: CatalogModel): string => `${catalogLabel(m)} · ${ctxLabel(m.context_length)} context · ${BAND_WORDS[m.band]}`;

function CapsChips({ model }: { model: CatalogModel | undefined }) {
  if (!model?.caps) return null;
  const chip = (ok: boolean, name: string) => (
    <span className={"rounded px-1.5 py-0.5 text-[11px] font-medium " + (ok ? "bg-ok-soft text-ok" : "bg-warn-soft text-warn")}>
      {name} {ok ? "yes" : "no"}
    </span>
  );
  return (
    <span className="inline-flex flex-wrap items-center gap-1" role="group" aria-label={`What ${catalogLabel(model)} can do`}>
      {chip(model.caps.json_schema, "JSON")}
      {chip(model.caps.tools, "Tools")}
    </span>
  );
}

function CatalogPicker({ label, models, current, busy, onPick }: {
  label: string; models: CatalogModel[]; current: string | null; busy: boolean; onPick: (id: string) => void;
}) {
  const [draft, setDraft] = useState("");
  if (!models.length) return null;
  const chosen = draft || (models.some((m) => m.id === current) ? current! : "");
  return (
    <form className="mt-2 flex flex-wrap items-center gap-2" onSubmit={(e) => { e.preventDefault(); if (chosen && chosen !== current) onPick(chosen); }}>
      <label className="text-[12.5px] text-fg-3" htmlFor={`catalog-${label}`}>Or any model on Token Factory</label>
      <select id={`catalog-${label}`} aria-label={`${label}: any Token Factory model`} value={chosen} onChange={(e) => setDraft(e.target.value)} disabled={busy}
        className={INPUT + " w-auto min-w-[220px] flex-1"}>
        <option value="">Choose a model</option>
        {models.map((m) => <option key={m.id} value={m.id}>{catalogOption(m)}</option>)}
      </select>
      <button type="submit" disabled={busy || !chosen || chosen === current} className={BTN_SECONDARY}>Use</button>
    </form>
  );
}

export function ModelInference() {
  const { selectedCustomerId } = useSelectedCustomer();
  const { me } = useAuth();
  const qc = useQueryClient();
  const toast = useToast();
  const isPlatformAdmin = me?.role === "platform_admin" || me?.kind === "platform_admin_key";
  const [keyDraft, setKeyDraft] = useState("");
  const [providerDraft, setProviderDraft] = useState<UpstreamProvider | "">("");
  const [baseUrlDraft, setBaseUrlDraft] = useState("");
  const [modelDraft, setModelDraft] = useState("");
  const [capDraft, setCapDraft] = useState("");
  const [error, setError] = useState<string | null>(null);

  const spend = useQuery({ queryKey: ["customer-spend", selectedCustomerId], queryFn: () => api.customerSpend(selectedCustomerId!), enabled: !!selectedCustomerId });
  const record = useQuery({ queryKey: ["own-customer", selectedCustomerId], queryFn: () => api.getCustomer(selectedCustomerId!), enabled: !!selectedCustomerId });
  const budgets = useQuery({ queryKey: ["budgets", selectedCustomerId], queryFn: () => api.listBudgets(selectedCustomerId!), enabled: !!selectedCustomerId });
  const catalog = useQuery({ queryKey: ["model-catalog", selectedCustomerId], queryFn: () => api.modelCatalog(selectedCustomerId!), enabled: !!selectedCustomerId });
  const origins = useQuery({ queryKey: ["spend-origins", selectedCustomerId], queryFn: () => api.spendOrigins(selectedCustomerId!, 30), enabled: !!selectedCustomerId && isPlatformAdmin });

  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["customer-spend", selectedCustomerId] });
    qc.invalidateQueries({ queryKey: ["own-customer", selectedCustomerId] });
    qc.invalidateQueries({ queryKey: ["budgets", selectedCustomerId] });
    qc.invalidateQueries({ queryKey: ["model-catalog", selectedCustomerId] });
  };
  const fail = (fallback: string) => (e: unknown) => setError(errorMessage(e, fallback));

  const setMode = useMutation({
    mutationFn: (m: "managed" | "byok") => api.setInferenceMode(selectedCustomerId!, m),
    onSuccess: (_o, m) => { setError(null); toast(m === "managed" ? "Inference now runs on Crystal's managed models." : "Inference now runs on your own provider key."); refresh(); },
    onError: fail("Couldn't change the inference mode."),
  });
  const saveKey = useMutation({
    mutationFn: () => api.updateUpstreamKey(selectedCustomerId!, keyDraft.trim(), providerDraft || undefined,
      providerDraft === "self_hosted" ? baseUrlDraft.trim() || undefined : undefined),
    onSuccess: (r) => {
      setError(null); setKeyDraft(""); setBaseUrlDraft("");
      toast(r?.provider === "nebius"
        ? `Token Factory key saved${typeof r.models_seen === "number" ? ` (${r.models_seen} models available)` : ""}.`
        : "Provider key saved.");
      refresh();
    },
    onError: fail("Couldn't save the provider key."),
  });
  const setModel = useMutation({
    mutationFn: (id: string) => api.setModel(selectedCustomerId!, id),
    onSuccess: () => { setError(null); setModelDraft(""); toast("Answers model updated."); refresh(); },
    onError: fail("Couldn't change the model."),
  });
  const setBackground = useMutation({
    mutationFn: (id: string) => api.setBackgroundModel(selectedCustomerId!, id),
    onSuccess: () => { setError(null); toast("Background curation model updated."); refresh(); },
    onError: fail("Couldn't change the background model."),
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
  const storedProvider: string = record.data?.provider ?? "anthropic";
  const provider: UpstreamProvider | "" = providerDraft || (PROVIDERS.some((p) => p.value === storedProvider) ? (storedProvider as UpstreamProvider) : "");
  // Item 2: a tenant on their own Token Factory key picks from the same
  // cards and catalog as a managed tenant, with no plan locks.
  const ownTokenFactory = mode === "byok" && storedProvider === "nebius";
  const storedModel: string = record.data?.model_id ?? record.data?.model_routing_config?.model_id ?? "";
  const currentModel = canonicalModel(storedModel);
  const storedBackground: string | null = record.data?.background_model_id ?? null;
  const currentBackground = storedBackground ? canonicalModel(storedBackground) : null;
  const allowed = me?.allowed_models;
  const pct = me?.usage?.ai_capacity_pct;
  const mtd = spend.data?.managed_month_to_date_micro_usd ?? 0;
  const cap = spend.data?.managed_monthly_cap_micro_usd ?? 0;
  const researchRow = budgets.data?.budgets?.find((b) => b.function === "auto_research" && !b.operator_id);
  const researchOn = (researchRow?.cap_micro_usd ?? 0) > 0;
  const shadow = budgets.data?.shadow_critic;
  const busy = setMode.isPending || saveKey.isPending || setModel.isPending || setBackground.isPending || research.isPending || saveCap.isPending;

  // Q87: one card set, drawn twice (Answers, Background curation). M3:
  // the Nemotron cards the platform offers follow the Anthropic four.
  const catalogNvidiaIds = (catalog.data?.models ?? []).filter((m) => m.nvidia).map((m) => m.id);
  const cardSet = ownTokenFactory ? nemotronCards(catalogNvidiaIds) : [...MANAGED_MODELS, ...nemotronCards(allowed)];
  // On the tenant's own key nothing is plan-locked, and the light sibling comes from the catalog.
  const lockList = ownTokenFactory ? undefined : allowed;
  const familyIds = ownTokenFactory ? catalogNvidiaIds : allowed;
  // Item 1: the rest of Token Factory's catalog sits in a picker under
  // each card set (the models already drawn as cards are left out).
  const cardIds = new Set(cardSet.map((m) => m.value.toLowerCase()));
  const catalogModels = (catalog.data?.enabled ? catalog.data.models : []).filter((m) => !cardIds.has(m.id.toLowerCase()));
  const catalogRow = (id: string | null) => (catalog.data?.models ?? []).find((m) => id && m.id.toLowerCase() === id.toLowerCase());
  const cards = (label: string, current: string | null, pick: (id: string) => void) => (
    <div role="radiogroup" aria-label={label} className="grid gap-2 min-[520px]:grid-cols-2 min-[900px]:grid-cols-4">
      {cardSet.map((m) => {
        const locked = Array.isArray(lockList) && !lockList.includes(m.value);
        return (
          <ChoiceCard key={m.value} checked={current === m.value} disabled={busy || locked}
            onPick={() => current !== m.value && pick(m.value)}
            title={m.label} body={m.body} raw={m.value}
            aside={locked ? <Link to="/billing" className="text-[12px] font-medium text-accent hover:underline" onClick={(e) => e.stopPropagation()}>Upgrade to use</Link> : undefined} />
        );
      })}
    </div>
  );

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

      <Section title="Provider API key" help="Stored encrypted and never shown again. Needed before switching to your own key. A Token Factory key is checked against Nebius before it is stored.">
        <form className="flex flex-wrap gap-2" onSubmit={(e) => { e.preventDefault(); if (keyDraft.trim()) saveKey.mutate(); }}>
          <select aria-label="Key provider" value={provider} onChange={(e) => setProviderDraft(e.target.value as UpstreamProvider)} disabled={busy} className={INPUT + " w-auto"}>
            {PROVIDERS.map((p) => <option key={p.value} value={p.value}>{p.label}</option>)}
          </select>
          <input type="password" aria-label="Provider API key" value={keyDraft} onChange={(e) => setKeyDraft(e.target.value)}
            placeholder={PROVIDERS.find((p) => p.value === provider)?.placeholder ?? "API key"} className={INPUT + " min-w-0 flex-1"} />
          {provider === "self_hosted" && (
            <input aria-label="Endpoint URL" value={baseUrlDraft} onChange={(e) => setBaseUrlDraft(e.target.value)} placeholder="https://your-host/v1" className={INPUT + " min-w-0 flex-1"} />
          )}
          <button type="submit" disabled={!keyDraft.trim() || busy || (provider === "self_hosted" && !baseUrlDraft.trim())} className={BTN_PRIMARY}>{saveKey.isPending ? "Saving…" : "Save key"}</button>
        </form>
        <p className="mt-2 text-[12.5px] text-fg-2">
          {hasKey ? `A ${providerLabel(storedProvider)} key is stored. Saving again replaces it.` : "No provider key stored yet."}
          {provider === "nebius" && " On a Token Factory key, answers and background curation both run on your key and never count toward daily capacity."}
        </p>
      </Section>

      <Section title="Models" help={mode === "managed"
        ? "Two choices: the model that answers you, and the model that works on your memory in the background. Infrastructure jobs (sparse keys, the quality critics, structural scans) always run on Crystal's own models."
        : ownTokenFactory
          ? "Two choices on your Token Factory key: the model that answers you, and the model that works on your memory in the background. Infrastructure jobs (sparse keys, the quality critics, structural scans) run on Crystal's own models."
          : "Any model your provider key can serve answers, and background curation runs on it too."}>
        {mode === "managed" || ownTokenFactory ? (
          <div className="space-y-5">
            <div>
              <div className="mb-2 flex flex-wrap items-baseline gap-2">
                <h3 className="text-[13.5px] font-semibold text-fg">Answers</h3>
                <span className="text-[12.5px] text-fg-3">Chat, every connected tool, scheduled briefs</span>
              </div>
              {cards("Answers model", currentModel, (id) => setModel.mutate(id))}
              <CatalogPicker label="Answers" models={catalogModels} current={currentModel || null} busy={busy} onPick={(id) => setModel.mutate(id)} />
            </div>
            <div>
              <div className="mb-2 flex flex-wrap items-baseline gap-2">
                <h3 className="text-[13.5px] font-semibold text-fg">Background curation</h3>
                <span className="text-[12.5px] text-fg-3">Research, synthesis, extraction, contradiction and gap scans, assumptions</span>
              </div>
              {cards("Background curation model", currentBackground, (id) => setBackground.mutate(id))}
              <CatalogPicker label="Background" models={catalogModels} current={currentBackground} busy={busy} onPick={(id) => setBackground.mutate(id)} />
              <p className="mt-2 text-[12px] text-fg-3">
                Quick, repetitive steps (classifying one fact, naming a subject) use the lightest model in the same family, so a Sonnet choice runs those on Haiku.
                {currentBackground && <> <button type="button" onClick={() => setBackground.mutate("")} disabled={busy} className="text-accent hover:underline">Use Crystal's default</button></>}
              </p>
            </div>
            <dl aria-label="What runs where" className="grid gap-2 rounded-lg border border-line bg-inset p-3 text-[12.5px] min-[720px]:grid-cols-3">
              <div><dt className="font-medium text-fg">Answers</dt><dd className="text-fg-2">{catalogRow(currentModel) ? catalogLabel(catalogRow(currentModel)!) : modelLabel(currentModel || "claude-sonnet-5-5")} <CapsChips model={catalogRow(currentModel)} /></dd></div>
              <div><dt className="font-medium text-fg">Background curation</dt><dd className="text-fg-2">{catalogRow(currentBackground) && !cardIds.has(currentBackground!.toLowerCase()) ? `${catalogLabel(catalogRow(currentBackground)!)}, quick steps on ${modelLabel(familySmall("nemotron", familyIds) ?? "")}` : backgroundSummary(currentBackground, familyIds)} <CapsChips model={catalogRow(currentBackground)} /></dd></div>
              <div><dt className="font-medium text-fg">Infrastructure</dt><dd className="text-fg-2">Crystal's models. Not affected by your choices.</dd></div>
            </dl>
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
