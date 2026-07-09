// Settings → API (Accounts Phase C). The tenant's control surface:
// inference mode (managed vs your own key), BYOK key entry, and the
// month-to-date managed spend against the tier cap. Platform admins see
// the same page scoped to the picked customer.
import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Copy, KeyRound, Loader2 } from "lucide-react";
import { api } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";
import { useAuth } from "@/lib/auth";

const MANAGED_MODELS = [
  { value: "claude-haiku-4-5", label: "Haiku — fastest, most economical" },
  { value: "claude-sonnet-5", label: "Sonnet — balanced (default)" },
  { value: "claude-opus-4-8", label: "Opus — deepest reasoning" },
];

function usd(micro: number): string {
  return `$${(micro / 1_000_000).toFixed(2)}`;
}

export function SettingsApi() {
  const { selectedCustomerId } = useSelectedCustomer();
  const qc = useQueryClient();
  const [keyDraft, setKeyDraft] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const spend = useQuery({
    queryKey: ["customer-spend", selectedCustomerId],
    queryFn: () => api.customerSpend(selectedCustomerId!),
    enabled: Boolean(selectedCustomerId),
  });
  const record = useQuery({
    queryKey: ["own-customer", selectedCustomerId],
    queryFn: () => api.getCustomer(selectedCustomerId!),
    enabled: Boolean(selectedCustomerId),
  });
  const [modelDraft, setModelDraft] = useState("");
  const [confirmRotate, setConfirmRotate] = useState(false);
  const [rotatedKey, setRotatedKey] = useState<string | null>(null);
  const [keyCopied, setKeyCopied] = useState(false);
  const [reauthPassword, setReauthPassword] = useState("");
  // S4 Budgets: v1 exposes the auto-research allowance (USD input,
  // stored as micro-USD). 0 / no row = auto-research OFF.
  const budgets = useQuery({
    queryKey: ["budgets", selectedCustomerId],
    queryFn: () => api.listBudgets(selectedCustomerId!),
    enabled: !!selectedCustomerId,
  });
  // S12: where the money goes (30-day window) + the shadow critic's
  // real, count-based cap surfaced read-only.
  const origins = useQuery({
    queryKey: ["spend-origins", selectedCustomerId],
    queryFn: () => api.spendOrigins(selectedCustomerId!, 30),
    enabled: !!selectedCustomerId,
  });
  const shadowInfo = (budgets.data as any)?.shadow_critic;
  const autoResearchRow = budgets.data?.budgets?.find(
    (b: any) => b.function === "auto_research" && !b.operator_id
  );
  const autoResearchOn = (autoResearchRow?.cap_micro_usd ?? 0) > 0;
  const [budgetDraft, setBudgetDraft] = useState("");
  const { status: authStatus, reauthProvider, reauthenticate } = useAuth();
  const needsPassword =
    authStatus === "signedIn" && reauthProvider() === "password";
  const currentModel: string =
    record.data?.model_id ?? record.data?.model_routing_config?.model_id ?? "";

  useEffect(() => {
    setNote(null);
    setError(null);
  }, [selectedCustomerId]);

  if (!selectedCustomerId) {
    return (
      <div className="p-8 text-[13px] text-gray-500">
        Select a customer to manage its API settings.
      </div>
    );
  }

  const mode = spend.data?.inference_mode ?? "…";
  const mtd = spend.data?.managed_month_to_date_micro_usd ?? 0;
  const cap = spend.data?.managed_monthly_cap_micro_usd ?? 0;
  const pct = cap > 0 ? Math.min(100, Math.round((mtd / cap) * 100)) : 0;

  const act = async (name: string, fn: () => Promise<unknown>, ok: string) => {
    setBusy(name);
    setError(null);
    setNote(null);
    try {
      await fn();
      setNote(ok);
      await qc.invalidateQueries({ queryKey: ["customer-spend"] });
      await qc.invalidateQueries({ queryKey: ["own-customer"] });
    } catch (e) {
      const detail = (e as { body?: { detail?: string } })?.body?.detail;
      setError(detail ?? "That didn't work — please try again.");
    } finally {
      setBusy(null);
    }
  };

  const saveKey = () =>
    act("key", async () => {
      await api.updateUpstreamKey(selectedCustomerId, keyDraft.trim());
      setKeyDraft("");
    }, "Provider key saved.");

  const setMode = (m: "managed" | "byok") =>
    act("mode", () => api.setInferenceMode(selectedCustomerId, m),
        m === "managed"
          ? "Inference now runs on CRYSTAL's managed models."
          : "Inference now runs on your own provider key.");

  return (
    <div className="mx-auto max-w-2xl space-y-6 p-8">
      <div>
        <h1 className="text-[17px] font-semibold text-gray-900">API & Inference</h1>
        <p className="mt-1 text-[13px] text-gray-500">
          Customer <code className="text-[12px]">{selectedCustomerId}</code>
          {spend.data?.subscription_tier && (
            <> · plan <span className="font-medium">{spend.data.subscription_tier}</span></>
          )}
        </p>
      </div>

      {/* Inference mode */}
      <section className="rounded-xl border border-gray-200 bg-white p-5">
        <h2 className="mb-1 text-[14px] font-semibold text-gray-900">Inference</h2>
        <p className="mb-4 text-[12.5px] text-gray-500">
          Managed runs on CRYSTAL's models with a monthly plan budget. Your
          own key routes every call to your provider account instead.
        </p>
        <div className="flex gap-2">
          <button
            disabled={busy !== null || mode === "managed"}
            onClick={() => void setMode("managed")}
            className={
              mode === "managed"
                ? "rounded-lg bg-[#6f72f7] px-4 py-2 text-[12.5px] font-semibold text-white"
                : "rounded-lg border border-gray-300 px-4 py-2 text-[12.5px] font-medium text-gray-700 hover:bg-gray-50"
            }
          >
            Managed (default)
          </button>
          <button
            disabled={busy !== null || mode === "byok"}
            onClick={() => void setMode("byok")}
            className={
              mode === "byok"
                ? "rounded-lg bg-[#6f72f7] px-4 py-2 text-[12.5px] font-semibold text-white"
                : "rounded-lg border border-gray-300 px-4 py-2 text-[12.5px] font-medium text-gray-700 hover:bg-gray-50"
            }
          >
            My own key
          </button>
          {busy === "mode" && <Loader2 className="h-5 w-5 animate-spin self-center text-gray-400" />}
        </div>

        {mode === "managed" && cap > 0 && (
          <div className="mt-5">
            <div className="mb-1.5 flex items-baseline justify-between text-[12px]">
              <span className="text-gray-500">Managed usage this month</span>
              <span className="font-medium text-gray-800">
                {usd(mtd)} of {usd(cap)}
              </span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-gray-100">
              <div
                className={pct >= 90 ? "h-full bg-red-500" : "h-full bg-[#6f72f7]"}
                style={{ width: `${pct}%` }}
              />
            </div>
            <p className="mt-1.5 text-[11.5px] text-gray-400">
              Resets on the 1st (UTC). At the cap, chat pauses until reset,
              an upgrade, or a switch to your own key.
            </p>
          </div>
        )}
      </section>

      {/* Model (hosted parity: same knob self-host has) */}
      <section className="rounded-xl border border-gray-200 bg-white p-5">
        <h2 className="mb-1 text-[14px] font-semibold text-gray-900">Model</h2>
        <p className="mb-3 text-[12.5px] text-gray-500">
          {mode === "managed"
            ? "Pick from CRYSTAL's managed models."
            : "Any model your provider key can serve."}
          {currentModel && (
            <> Currently <code className="text-[12px]">{currentModel}</code>.</>
          )}
        </p>
        <div className="flex gap-2">
          {mode === "managed" ? (
            <select
              value={modelDraft || currentModel}
              onChange={(e) => setModelDraft(e.target.value)}
              className="min-w-0 flex-1 rounded-lg border border-gray-300 px-3 py-2 text-[13px] outline-none focus:border-[#6f72f7]"
            >
              {MANAGED_MODELS.map((m) => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </select>
          ) : (
            <input
              value={modelDraft || currentModel}
              onChange={(e) => setModelDraft(e.target.value)}
              placeholder="model id"
              className="min-w-0 flex-1 rounded-lg border border-gray-300 px-3 py-2 text-[13px] outline-none focus:border-[#6f72f7]"
            />
          )}
          <button
            disabled={busy !== null || !(modelDraft || "").trim()
              || modelDraft === currentModel}
            onClick={() =>
              void act("model",
                () => api.setModel(selectedCustomerId, modelDraft.trim()),
                "Model updated.")}
            className="shrink-0 rounded-lg bg-gray-900 px-4 py-2 text-[12.5px] font-semibold text-white transition hover:bg-gray-700 disabled:opacity-40"
          >
            {busy === "model" ? "Saving…" : "Save model"}
          </button>
        </div>
      </section>

      {/* BYOK */}
      <section className="rounded-xl border border-gray-200 bg-white p-5">
        <h2 className="mb-1 text-[14px] font-semibold text-gray-900">
          Provider API key
        </h2>
        <p className="mb-3 text-[12.5px] text-gray-500">
          Stored encrypted, never displayed again. Required before switching
          to your own key.
        </p>
        <div className="flex gap-2">
          <input
            type="password"
            value={keyDraft}
            onChange={(e) => setKeyDraft(e.target.value)}
            placeholder="sk-ant-… or sk-…"
            className="min-w-0 flex-1 rounded-lg border border-gray-300 px-3 py-2 text-[13px] outline-none focus:border-[#6f72f7]"
          />
          <button
            disabled={!keyDraft.trim() || busy !== null}
            onClick={() => void saveKey()}
            className="shrink-0 rounded-lg bg-gray-900 px-4 py-2 text-[12.5px] font-semibold text-white transition hover:bg-gray-700 disabled:opacity-40"
          >
            {busy === "key" ? "Saving…" : "Save key"}
          </button>
        </div>
      </section>

      {/* S4: Budgets — the auto-research allowance */}
      <section className="rounded-xl border border-gray-200 bg-white p-5">
        <h2 className="mb-1 text-[14px] font-semibold text-gray-900">Budgets</h2>
        <p className="mb-4 text-[12.5px] leading-relaxed text-gray-500">
          Let CRYS research knowledge gaps on its own, up to a monthly
          allowance. Off means gaps wait for you to press Research —
          nothing is spent automatically.
        </p>
        <div className="flex items-center gap-2">
          <span className="min-w-0 flex-1 text-[13px] text-gray-700">
            Autonomous research
            <span className={"ml-2 rounded-full px-2 py-0.5 text-[11px] font-medium " +
              (autoResearchOn ? "bg-emerald-50 text-emerald-700" : "bg-gray-100 text-gray-500")}>
              {autoResearchOn
                ? `on — $${((autoResearchRow?.cap_micro_usd ?? 0) / 1_000_000).toFixed(2)}/mo`
                : "off"}
            </span>
          </span>
          <div className="flex shrink-0 items-center gap-2">
            <span className="text-[12.5px] text-gray-400">$</span>
            <input
              value={budgetDraft}
              onChange={(e) => setBudgetDraft(e.target.value)}
              placeholder={autoResearchOn
                ? ((autoResearchRow?.cap_micro_usd ?? 0) / 1_000_000).toFixed(2)
                : "5.00"}
              className="w-20 rounded-lg border border-gray-300 px-3 py-2 text-[12.5px] outline-none focus:border-[#6f72f7]"
            />
            <span className="text-[12.5px] text-gray-400">/mo</span>
            <button
              disabled={busy !== null || !budgetDraft}
              onClick={() =>
                void act("budget", async () => {
                  const usd = parseFloat(budgetDraft);
                  if (!isFinite(usd) || usd < 0) throw new Error("Enter a valid amount");
                  await api.upsertBudget(
                    selectedCustomerId, "auto_research",
                    Math.round(usd * 1_000_000));
                  setBudgetDraft("");
                  await budgets.refetch();
                }, "Budget saved.")}
              className="rounded-lg bg-gray-900 px-3.5 py-2 text-[12.5px] font-semibold text-white transition hover:bg-gray-700 disabled:opacity-40"
            >
              {busy === "budget" ? "Saving…" : "Save"}
            </button>
            {autoResearchOn && (
              <button
                disabled={busy !== null}
                onClick={() =>
                  void act("budget_off", async () => {
                    await api.upsertBudget(selectedCustomerId, "auto_research", 0);
                    await budgets.refetch();
                  }, "Autonomous research turned off.")}
                className="rounded-lg border border-gray-300 px-3.5 py-2 text-[12.5px] font-medium text-gray-600 hover:bg-gray-50"
              >
                Turn off
              </button>
            )}
          </div>
        </div>

        {/* S12: shadow-critic card — read-only; the enforced cap is
            count-based (per-day), not an S4 spend row, so we show the
            truth rather than an editable number nothing enforces. */}
        {shadowInfo && (
          <div className="mt-4 border-t border-gray-100 pt-4">
            <div className="text-[13px] font-medium text-gray-800 mb-1">Shadow critic</div>
            <p className="text-[12.5px] text-gray-500 mb-2">
              Metacognition R&D spend — the shadow reviews a sample of the
              agent's turns. Capped per day
              {shadowInfo.max_per_day == null ? " (platform default)" : " (custom override)"}.
            </p>
            <div className="flex gap-6 text-[12.5px] text-gray-700">
              <div><span className="text-gray-400">Cap/day:</span>{" "}
                {shadowInfo.effective_max_per_day ?? "—"}</div>
              <div><span className="text-gray-400">Runs (24h):</span>{" "}
                {shadowInfo.runs_last_24h}</div>
              <div><span className="text-gray-400">Spend (24h):</span>{" "}
                ${((shadowInfo.cost_micro_usd_last_24h ?? 0) / 1_000_000).toFixed(4)}</div>
            </div>
          </div>
        )}
      </section>

      {/* S12: spend by origin — where the money goes */}
      <section className="rounded-xl border border-gray-200 bg-white p-5">
        <h2 className="mb-1 text-[14px] font-semibold text-gray-900">Spend by origin</h2>
        <p className="mb-3 text-[12.5px] leading-relaxed text-gray-500">
          Last 30 days, from the call ledger. Every model invocation is
          attributed to the workload that made it.
        </p>
        {!(origins.data?.origins?.length) ? (
          <div className="text-[12.5px] text-gray-400">No ledgered calls in the window.</div>
        ) : (
          <table className="w-full text-[12.5px]">
            <thead>
              <tr className="text-left text-gray-400">
                <th className="py-1 font-medium">Origin</th>
                <th className="py-1 font-medium text-right">Calls</th>
                <th className="py-1 font-medium text-right">Cache reads</th>
                <th className="py-1 font-medium text-right">Cost</th>
              </tr>
            </thead>
            <tbody>
              {origins.data.origins.map((o: any) => (
                <tr key={o.origin} className="border-t border-gray-100 text-gray-700">
                  <td className="py-1.5 font-mono">{o.origin}</td>
                  <td className="py-1.5 text-right">{o.call_count.toLocaleString()}</td>
                  <td className="py-1.5 text-right">{(o.cache_read_tokens ?? 0).toLocaleString()}</td>
                  <td className="py-1.5 text-right">${(o.cost_micro_usd / 1_000_000).toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      {/* Key A: one-time reveal at signup + regeneration */}
      <section className="rounded-xl border border-gray-200 bg-white p-5">
        <h2 className="mb-1 flex items-center gap-2 text-[14px] font-semibold text-gray-900">
          <KeyRound className="h-4 w-4 text-gray-400" />
          Your CRYSTAL API key
        </h2>
        <p className="mb-3 text-[12.5px] leading-relaxed text-gray-500">
          Shown once at signup and stored as a hash — it cannot be
          retrieved. It authenticates SDK and API access; the console never
          needs it. Lost it? Regenerate below.
        </p>

        {rotatedKey ? (
          <div>
            <p className="mb-2 text-[12.5px] font-medium text-gray-800">
              Your new key — shown only this once:
            </p>
            <div className="flex items-center gap-2 rounded-lg border border-gray-200 bg-gray-50 px-3 py-2.5">
              <code className="min-w-0 flex-1 truncate text-[12px] text-emerald-700">
                {rotatedKey}
              </code>
              <button
                onClick={() => {
                  void navigator.clipboard.writeText(rotatedKey);
                  setKeyCopied(true);
                  setTimeout(() => setKeyCopied(false), 1600);
                }}
                className="shrink-0 rounded-md p-1.5 text-gray-500 transition hover:bg-gray-200"
                title="Copy"
              >
                {keyCopied ? <Check className="h-4 w-4 text-emerald-600" /> : <Copy className="h-4 w-4" />}
              </button>
            </div>
          </div>
        ) : confirmRotate ? (
          <div className="flex items-center gap-2">
            <p className="min-w-0 flex-1 text-[12.5px] text-red-600">
              Your current key stops working immediately.
              {authStatus === "signedIn" &&
                " You'll verify your sign-in first."}
            </p>
            {needsPassword && (
              <input
                type="password"
                value={reauthPassword}
                onChange={(e) => setReauthPassword(e.target.value)}
                placeholder="Your password"
                className="w-40 shrink-0 rounded-lg border border-gray-300 px-3 py-2 text-[12.5px] outline-none focus:border-[#6f72f7]"
              />
            )}
            <button
              disabled={busy !== null || (needsPassword && !reauthPassword)}
              onClick={() =>
                void act("rotate", async () => {
                  // Step-up: re-prove the credential (refreshes auth_time)
                  // — the backend refuses stale sessions regardless.
                  if (authStatus === "signedIn") {
                    await reauthenticate(
                      needsPassword ? reauthPassword : undefined);
                  }
                  const out = await api.rotateApiKey(selectedCustomerId);
                  setRotatedKey(out.api_key);
                  setConfirmRotate(false);
                  setReauthPassword("");
                }, "New API key generated.")}
              className="shrink-0 rounded-lg bg-red-600 px-3.5 py-2 text-[12.5px] font-semibold text-white transition hover:bg-red-500"
            >
              {busy === "rotate" ? "Generating…" : "Yes, regenerate"}
            </button>
            <button
              onClick={() => setConfirmRotate(false)}
              className="shrink-0 rounded-lg border border-gray-300 px-3.5 py-2 text-[12.5px] font-medium text-gray-600 hover:bg-gray-50"
            >
              Cancel
            </button>
          </div>
        ) : (
          <button
            onClick={() => setConfirmRotate(true)}
            className="rounded-lg border border-gray-300 px-3.5 py-2 text-[12.5px] font-medium text-gray-700 transition hover:bg-gray-50"
          >
            Regenerate key
          </button>
        )}
      </section>

      {(note || error) && (
        <p
          className={
            error
              ? "rounded-lg bg-red-50 px-3 py-2 text-[12.5px] text-red-600"
              : "rounded-lg bg-emerald-50 px-3 py-2 text-[12.5px] text-emerald-700"
          }
        >
          {error ?? note}
        </p>
      )}
    </div>
  );
}
