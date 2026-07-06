// Settings → API (Accounts Phase C). The tenant's control surface:
// inference mode (managed vs your own key), BYOK key entry, and the
// month-to-date managed spend against the tier cap. Platform admins see
// the same page scoped to the picked customer.
import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { KeyRound, Loader2 } from "lucide-react";
import { api } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";

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

      {/* Key A note */}
      <section className="rounded-xl border border-gray-200 bg-white p-5">
        <h2 className="mb-1 flex items-center gap-2 text-[14px] font-semibold text-gray-900">
          <KeyRound className="h-4 w-4 text-gray-400" />
          Your CRYSTAL API key
        </h2>
        <p className="text-[12.5px] leading-relaxed text-gray-500">
          Shown once at signup and stored as a hash — it cannot be
          retrieved. It authenticates SDK and API access; the console never
          needs it.
        </p>
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
