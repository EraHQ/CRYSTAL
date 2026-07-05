import { useState, type ReactNode } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { UserPlus, Copy, Check, KeyRound } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";
import { CrystalButton, ErrorBanner } from "@/components/ui";
import type { CreateCustomerRequest, CreateCustomerResponse } from "@/lib/types";

type Provider = "anthropic" | "openai" | "self_hosted";

const MODEL_PLACEHOLDER: Record<Provider, string> = {
  anthropic: "claude-sonnet-4-5-20250929",
  openai: "gpt-4o",
  self_hosted: "Qwen/Qwen3-0.6B",
};

const inputCls =
  "w-full rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm text-gray-800 placeholder:text-gray-400 focus:outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-500/20";
const selectCls = `${inputCls} cursor-pointer`;

export function Onboard() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const { setSelectedCustomerId } = useSelectedCustomer();

  const [provider, setProvider] = useState<Provider>("anthropic");
  const [modelId, setModelId] = useState("");
  const [upstreamKey, setUpstreamKey] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [injection, setInjection] = useState<"text" | "hidden_state" | "none">("text");
  const [shadowRate, setShadowRate] = useState("0.05");
  const [result, setResult] = useState<CreateCustomerResponse | null>(null);
  const [copied, setCopied] = useState(false);

  const create = useMutation({
    mutationFn: (body: CreateCustomerRequest) => api.createCustomer(body),
    onSuccess: (res) => {
      setResult(res);
      queryClient.invalidateQueries({ queryKey: ["customers"] });
    },
  });

  const submit = () => {
    const body: CreateCustomerRequest = {
      provider,
      model_id: modelId.trim() || MODEL_PLACEHOLDER[provider],
      api_key_ref: upstreamKey.trim(),
      injection_preference: injection,
      shadow_sample_rate: Number(shadowRate) || 0,
    };
    if (provider === "self_hosted" && baseUrl.trim()) body.base_url = baseUrl.trim();
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

  const useCustomer = () => {
    if (!result) return;
    setSelectedCustomerId(result.id);
    navigate("/");
  };

  const reset = () => {
    setResult(null);
    setModelId("");
    setUpstreamKey("");
    setBaseUrl("");
  };

  const errMsg =
    create.error instanceof ApiError
      ? `${create.error.status}: ${JSON.stringify(create.error.body)}`
      : create.error
      ? String(create.error)
      : undefined;

  // ---- Success panel: show the key once ----
  if (result) {
    return (
      <div className="mx-auto max-w-2xl space-y-5">
        <div>
          <h1 className="text-lg font-semibold text-gray-900">Customer created</h1>
          <p className="text-sm text-gray-500">Save the key below — it's shown only once.</p>
        </div>

        <div className="space-y-4 rounded-xl border border-emerald-200 bg-emerald-50/50 p-5">
          <div>
            <p className="text-[11px] uppercase tracking-wider text-gray-500">Customer ID</p>
            <code className="mt-0.5 block font-mono text-sm text-gray-900">{result.id}</code>
          </div>
          <div>
            <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-wider text-gray-500">
              <KeyRound className="h-3.5 w-3.5" /> Crystal Cache key (coding-agent login)
            </div>
            <div className="mt-1 flex items-center gap-2">
              <code className="flex-1 break-all rounded-lg border border-gray-200 bg-white px-3 py-2 font-mono text-xs text-gray-900">
                {result.api_key}
              </code>
              <CrystalButton size="sm" variant="secondary" onClick={copyKey}>
                {copied ? (
                  <>
                    <Check className="h-3.5 w-3.5" /> Copied
                  </>
                ) : (
                  <>
                    <Copy className="h-3.5 w-3.5" /> Copy
                  </>
                )}
              </CrystalButton>
            </div>
            <p className="mt-1.5 text-xs text-amber-700">
              Shown once. Paste it into the coding agent's <code className="font-mono">/login</code> to point it at this bank.
            </p>
          </div>
          <div className="flex gap-2 pt-1">
            <CrystalButton onClick={useCustomer}>Use this customer</CrystalButton>
            <CrystalButton variant="secondary" onClick={reset}>
              Create another
            </CrystalButton>
          </div>
        </div>
      </div>
    );
  }

  // ---- Form ----
  return (
    <div className="mx-auto max-w-2xl space-y-5">
      <div>
        <h1 className="text-lg font-semibold text-gray-900">Onboard a customer</h1>
        <p className="text-sm text-gray-500">
          Create an account that owns a knowledge bank. The key it returns is the coding agent's login.
        </p>
      </div>

      <div className="space-y-4 rounded-xl border border-gray-200 bg-white p-5">
        <Field label="Upstream provider">
          <select
            value={provider}
            onChange={(e) => setProvider(e.target.value as Provider)}
            className={selectCls}
          >
            <option value="anthropic">Anthropic</option>
            <option value="openai">OpenAI</option>
            <option value="self_hosted">Self-hosted</option>
          </select>
        </Field>

        <Field label="Model">
          <input
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            placeholder={MODEL_PLACEHOLDER[provider]}
            className={inputCls}
          />
        </Field>

        <Field label="Upstream API key" hint="The provider key Crystal Cache uses to call the model.">
          <input
            type="password"
            value={upstreamKey}
            onChange={(e) => setUpstreamKey(e.target.value)}
            placeholder={
              provider === "anthropic" ? "sk-ant-..." : provider === "openai" ? "sk-..." : "provider key"
            }
            className={`${inputCls} font-mono`}
          />
        </Field>

        {provider === "self_hosted" && (
          <Field label="Base URL" hint="Required for self-hosted providers.">
            <input
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="http://localhost:8001/v1"
              className={inputCls}
            />
          </Field>
        )}

        <div className="grid grid-cols-2 gap-4">
          <Field label="Injection">
            <select
              value={injection}
              onChange={(e) => setInjection(e.target.value as "text" | "hidden_state" | "none")}
              className={selectCls}
            >
              <option value="text">text</option>
              <option value="hidden_state">hidden_state</option>
              <option value="none">none</option>
            </select>
          </Field>
          <Field label="Shadow sample rate">
            <input
              type="number"
              min="0"
              max="1"
              step="0.01"
              value={shadowRate}
              onChange={(e) => setShadowRate(e.target.value)}
              className={inputCls}
            />
          </Field>
        </div>

        {errMsg && <ErrorBanner title="Couldn't create customer" message={errMsg} />}

        <div className="pt-1">
          <CrystalButton onClick={submit} disabled={create.isPending || !upstreamKey.trim()}>
            <UserPlus className="h-4 w-4" />
            {create.isPending ? "Creating…" : "Create customer"}
          </CrystalButton>
        </div>
      </div>
    </div>
  );
}

function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="block">
      <span className="text-sm font-medium text-gray-700">{label}</span>
      {hint && <span className="mt-0.5 block text-xs text-gray-400">{hint}</span>}
      <div className="mt-1.5">{children}</div>
    </label>
  );
}
