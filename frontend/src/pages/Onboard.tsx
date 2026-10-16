// New customer — Console Refine PR 13 (docs/CONSOLE_UX_AUDIT.md 5.12).
// Platform-admin only. Creates a workspace with its own memory and shows
// its Crystal key once. The spec's Workspace name, Owner email, Plan and
// "Send a welcome email" have no server side (POST /v1/customers takes
// routing config only; a customer has no name), so they are not shown —
// the form is the defaults line plus the Advanced expander.
import { useState, type ReactNode } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { Check, ChevronDown, ChevronRight, Copy, Users } from "lucide-react";
import { api, ApiError, errorMessage } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";
import { EmptyState, ErrorBanner, PageHeader } from "@/components/ui";
import { BTN_PRIMARY, BTN_SECONDARY, INPUT } from "@/pages/settings/shared";
import type { CreateCustomerRequest, CreateCustomerResponse } from "@/lib/types";
import { cn } from "@/lib/utils";

type Provider = "anthropic" | "openai" | "self_hosted";
type Injection = "text" | "hidden_state" | "none";

const MODEL_PLACEHOLDER: Record<Provider, string> = {
  anthropic: "claude-sonnet-4-5-20250929",
  openai: "gpt-4o",
  self_hosted: "Qwen/Qwen3-0.6B",
};
const PROVIDER_LABEL: Record<Provider, string> = { anthropic: "Anthropic", openai: "OpenAI", self_hosted: "Self-hosted" };
const INJECTION_LABEL: Record<Injection, string> = { text: "text injection", hidden_state: "hidden-state injection", none: "no injection" };

// Inline hint for a base URL as it is typed: a near-miss scheme gets the
// correction, anything without a scheme gets the rule, a good URL gets null.
export function baseUrlHint(value: string): string | null {
  const v = value.trim();
  if (!v) return null;
  if (/^https?:\/\//i.test(v)) return null;
  const m = v.match(/^([a-z]+):\/?\/?/i);
  if (m) return `That URL starts with "${m[1]}". Did you mean https://?`;
  return "Enter a full URL, starting with https://";
}

// The one-line summary on the collapsed Advanced expander.
export function advancedSummary(o: { provider: Provider; key: string; injection: Injection; shadow: string }): string {
  const access = o.key.trim() ? "own key" : "managed";
  const pct = Math.round((Number(o.shadow) || 0) * 100);
  return `${access} ${PROVIDER_LABEL[o.provider]} · ${INJECTION_LABEL[o.injection]} · shadow ${pct}%`;
}

// Which field a server message belongs to, so it renders beside that field.
export function fieldFor(message: string): "base_url" | "api_key_ref" | "model_id" | null {
  const m = message.toLowerCase();
  if (m.includes("base_url") || m.includes("base url")) return "base_url";
  if (m.includes("api_key_ref") || m.includes("api key") || m.includes("upstream key")) return "api_key_ref";
  if (m.includes("model")) return "model_id";
  return null;
}

export function Onboard() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const { setSelectedCustomerId } = useSelectedCustomer();

  const [provider, setProvider] = useState<Provider>("anthropic");
  const [modelId, setModelId] = useState("");
  const [upstreamKey, setUpstreamKey] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [injection, setInjection] = useState<Injection>("text");
  const [shadowRate, setShadowRate] = useState("0.05");
  const [advanced, setAdvanced] = useState(false);
  const [result, setResult] = useState<CreateCustomerResponse | null>(null);
  const [copied, setCopied] = useState(false);
  const [saved, setSaved] = useState(false);
  const [managedNote, setManagedNote] = useState<string | null>(null);

  const customers = useQuery({ queryKey: ["customers"], queryFn: () => api.listCustomers() });

  const create = useMutation({
    mutationFn: async (body: CreateCustomerRequest) => {
      const res = await api.createCustomer(body);
      // No key pasted → the platform's managed key serves this workspace.
      if (!body.api_key_ref) {
        try {
          await api.setInferenceMode(res.id, "managed");
        } catch (e) {
          setManagedNote(errorMessage(e, "Couldn't switch it to the managed key; set that in Settings › Model."));
        }
      }
      return res;
    },
    onSuccess: (res) => {
      setResult(res);
      setSaved(false);
      setCopied(false);
      void queryClient.invalidateQueries({ queryKey: ["customers"] });
    },
  });

  const urlHint = provider === "self_hosted" ? baseUrlHint(baseUrl) : null;
  const serverMsg = create.error ? errorMessage(create.error, "Couldn't create the customer.") : null;
  const serverField = serverMsg && create.error instanceof ApiError && create.error.status < 500 ? fieldFor(serverMsg) : null;
  const fieldError = (f: "base_url" | "api_key_ref" | "model_id") => (serverField === f ? serverMsg : null);

  const submit = () => {
    if (urlHint) { setAdvanced(true); return; }
    const body: CreateCustomerRequest = {
      provider,
      model_id: modelId.trim() || MODEL_PLACEHOLDER[provider],
      api_key_ref: upstreamKey.trim(),
      injection_preference: injection,
      shadow_sample_rate: Math.min(1, Math.max(0, Number(shadowRate) || 0)),
    };
    if (provider === "self_hosted" && baseUrl.trim()) body.base_url = baseUrl.trim();
    setManagedNote(null);
    create.mutate(body);
  };

  const copyKey = async () => {
    if (!result) return;
    try {
      await navigator.clipboard.writeText(result.api_key);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard can be unavailable; the key is visible to select by hand.
    }
  };

  const reset = () => {
    setResult(null);
    setModelId("");
    setUpstreamKey("");
    setBaseUrl("");
    setSaved(false);
    setManagedNote(null);
    create.reset();
  };

  const n = customers.data?.items.length;
  const meta = `Creates a workspace with its own memory.${n != null ? ` ${n} customer${n === 1 ? "" : "s"} so far.` : ""}`;

  if (result) {
    return (
      <div className="mx-auto max-w-2xl">
        <PageHeader title="New customer" meta={meta} />
        <section aria-label="Customer created" className="rounded-xl border border-ok/40 bg-surface p-4 min-[720px]:p-5">
          <h2 className="text-[16px] font-semibold text-fg">Workspace {result.id} is ready</h2>
          <p className="mt-1 text-[13px] text-fg-2">{result.provider} · {result.model_id}</p>
          {managedNote && <ErrorBanner title="Managed key not set" message={managedNote} />}
          <dl className="mt-4 space-y-3">
            <div>
              <dt className="text-[11px] font-medium uppercase tracking-[0.08em] text-fg-3">Customer id</dt>
              <dd className="mt-0.5 font-mono text-[13px] text-fg">{result.id}</dd>
            </div>
            <div>
              <dt className="text-[11px] font-medium uppercase tracking-[0.08em] text-fg-3">API key · shown once</dt>
              <dd className="mt-1 flex flex-wrap items-center gap-2">
                <code className="min-w-0 flex-1 break-all rounded-md border border-line bg-inset px-2.5 py-1.5 font-mono text-[12px] text-fg">{result.api_key}</code>
                <button type="button" onClick={() => void copyKey()} className={BTN_SECONDARY} aria-label="Copy API key">
                  {copied ? <><Check aria-hidden className="h-3.5 w-3.5" /> Copied</> : <><Copy aria-hidden className="h-3.5 w-3.5" /> Copy</>}
                </button>
              </dd>
            </div>
          </dl>
          <p className="mt-2 text-[12.5px] text-fg-2">This is the workspace's Crystal key: paste it into a tool's connection, or into the coding agent's /login. It is never shown again.</p>
          <label className="mt-4 flex items-center gap-2 text-[13px] text-fg">
            <input type="checkbox" checked={saved} onChange={(e) => setSaved(e.target.checked)} className="h-4 w-4 accent-accent" />
            I copied the key somewhere safe
          </label>
          <div className="mt-4 flex flex-wrap gap-2">
            <button type="button" onClick={() => { setSelectedCustomerId(result.id); navigate("/"); }} disabled={!saved} className={BTN_PRIMARY}>Use this customer</button>
            <button type="button" onClick={reset} className={BTN_SECONDARY}>Create another</button>
          </div>
        </section>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-2xl">
      <PageHeader title="New customer" meta={meta} />
      {customers.isError && <ErrorBanner title="Couldn't count customers" error={customers.error} onRetry={() => customers.refetch()} />}
      <form onSubmit={(e) => { e.preventDefault(); submit(); }} className="rounded-xl border border-line bg-surface p-4 min-[720px]:p-5" aria-label="New customer">
        <p className="text-[13px] text-fg-2">
          The defaults suit most workspaces: Crystal's managed Anthropic key, text injection, and the critic double-checking 5% of answers. Open Advanced to change them.
        </p>

        <div className="mt-4 rounded-lg border border-line">
          <button type="button" onClick={() => setAdvanced((v) => !v)} aria-expanded={advanced}
            className="flex w-full items-center gap-2 px-3 py-2.5 text-left text-[13.5px] hover:bg-inset">
            {advanced ? <ChevronDown aria-hidden className="h-4 w-4 shrink-0 text-fg-3" /> : <ChevronRight aria-hidden className="h-4 w-4 shrink-0 text-fg-3" />}
            <span className="font-medium text-fg">Advanced</span>
            <span className="min-w-0 flex-1 truncate text-[12.5px] text-fg-3">{advancedSummary({ provider, key: upstreamKey, injection, shadow: shadowRate })}</span>
          </button>
          {advanced && (
            <div className="space-y-4 border-t border-line px-3 py-3">
              <Field id="nc-provider" label="Provider">
                <select id="nc-provider" value={provider} onChange={(e) => setProvider(e.target.value as Provider)} className={cn(INPUT, "cursor-pointer")}>
                  <option value="anthropic">Anthropic</option>
                  <option value="openai">OpenAI</option>
                  <option value="self_hosted">Self-hosted</option>
                </select>
              </Field>
              <Field id="nc-model" label="Model" hint={`Blank uses ${MODEL_PLACEHOLDER[provider]}.`} error={fieldError("model_id")}>
                <input id="nc-model" value={modelId} onChange={(e) => setModelId(e.target.value)} placeholder={MODEL_PLACEHOLDER[provider]} className={INPUT} />
              </Field>
              <Field id="nc-key" label="Upstream API key" hint="Leave blank to use Crystal's managed key." error={fieldError("api_key_ref")}>
                <input id="nc-key" type="password" autoComplete="off" value={upstreamKey} onChange={(e) => setUpstreamKey(e.target.value)}
                  placeholder={provider === "anthropic" ? "sk-ant-…" : provider === "openai" ? "sk-…" : "provider key"} className={cn(INPUT, "font-mono")} />
              </Field>
              {provider === "self_hosted" && (
                <Field id="nc-url" label="Base URL" hint="Where the self-hosted model listens." error={urlHint ?? fieldError("base_url")}>
                  <input id="nc-url" value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} placeholder="https://models.example.com/v1" className={INPUT}
                    aria-invalid={urlHint ? true : undefined} />
                </Field>
              )}
              <div className="grid gap-4 min-[520px]:grid-cols-2">
                <Field id="nc-injection" label="Injection">
                  <select id="nc-injection" value={injection} onChange={(e) => setInjection(e.target.value as Injection)} className={cn(INPUT, "cursor-pointer")}>
                    <option value="text">Text</option>
                    <option value="hidden_state">Hidden state</option>
                    <option value="none">None</option>
                  </select>
                </Field>
                <Field id="nc-shadow" label="Shadow sample rate" hint="Share of answers the critic double-checks.">
                  <input id="nc-shadow" type="number" min="0" max="1" step="0.01" value={shadowRate} onChange={(e) => setShadowRate(e.target.value)} className={INPUT} />
                </Field>
              </div>
            </div>
          )}
        </div>

        {serverMsg && !serverField && <ErrorBanner title="Couldn't create the customer" message={serverMsg} />}

        <div className="mt-4">
          <button type="submit" disabled={create.isPending} className={BTN_PRIMARY}>
            <Users aria-hidden className="h-4 w-4" />
            {create.isPending ? "Creating…" : "Create customer"}
          </button>
        </div>
      </form>
      {customers.data && customers.data.items.length === 0 && (
        <EmptyState icon={Users} title="No customers yet" help="The first workspace you create becomes the selected customer." className="mt-4" />
      )}
    </div>
  );
}

function Field({ id, label, hint, error, children }: { id: string; label: string; hint?: string; error?: string | null; children: ReactNode }) {
  return (
    <div>
      <label htmlFor={id} className="block text-[12.5px] font-medium text-fg">{label}</label>
      <div className="mt-1">{children}</div>
      {error ? <p className="mt-1 text-[12px] text-bad" role="alert">{error}</p> : hint ? <p className="mt-1 text-[12px] text-fg-3">{hint}</p> : null}
    </div>
  );
}
