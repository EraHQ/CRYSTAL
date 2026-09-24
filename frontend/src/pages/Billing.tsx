// Billing (T1c, 2026-09-25 — supersedes the S4=B trial-era page).
// Three-column pricing truth + the two capacity meters (Q4=A: customers
// see capacity, never dollars). States: free (upgrade CTA), paid
// (manage via Stripe portal), legacy trial (countdown/expired banner
// until those accounts age out). Payment stays on Stripe's hosted
// pages — card data never touches this console.
import { useState } from "react";
import { Check, Clock, CreditCard, Mail, ShieldCheck } from "lucide-react";
import { api } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { CrystalButton, ErrorBanner } from "@/components/ui";

function daysLeft(iso: string | null | undefined): number | null {
  if (!iso) return null;
  const ms = new Date(iso).getTime() - Date.now();
  return Math.max(0, Math.ceil(ms / 86_400_000));
}

function Meter({ label, note, pct, state, detail }: {
  label: string; note: string; pct: number; state: "ok" | "warning" | "blocked";
  detail: string;
}) {
  const bar =
    state === "blocked" ? "bg-red-500"
    : state === "warning" ? "bg-amber-500"
    : "bg-indigo-500";
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <span className="text-sm font-medium text-gray-900">{label}</span>
        <span className="text-xs text-gray-500">{detail}</span>
      </div>
      <div className="mt-1 h-2 w-full rounded-full bg-gray-100">
        <div
          className={`h-2 rounded-full ${bar}`}
          style={{ width: `${Math.min(100, pct)}%` }}
        />
      </div>
      <p className="mt-0.5 text-xs text-gray-400">{note}</p>
    </div>
  );
}

const PLANS = [
  {
    name: "Free", price: "$0", tierMatch: (t: string) => t === "free",
    features: [
      "500 crystals of memory",
      "Full self-curation: conflicts, duplicates, gaps, quality tiers",
      "Daily AI capacity for everyday use",
      "The four-tool chat surface everywhere",
    ],
  },
  {
    name: "Starter", price: "$29/mo",
    tierMatch: (t: string) => t.startsWith("starter") || t.startsWith("trial"),
    features: [
      "25,000 crystals of memory",
      "Everything in Free",
      "10x the daily AI capacity",
      "Priority background curation",
    ],
  },
  {
    name: "Scale", price: "$49/seat/mo",
    tierMatch: () => false,
    features: [
      "Unlimited memory",
      "Everything in Starter",
      "Seats for your whole team (min 2)",
      "Shared team banks when seats ship",
    ],
  },
];

export function Billing() {
  const { me } = useAuth();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const tier = me?.subscription_tier ?? null;
  const isTrial = !!tier && tier.startsWith("trial");
  const left = daysLeft(me?.trial_expires_at);
  const expired = isTrial && left !== null && left <= 0;
  const paid = !!tier && !isTrial && tier !== "free";
  const usage = me?.usage;
  const origin = window.location.origin;

  const upgrade = async () => {
    setBusy(true);
    setError(null);
    try {
      const out = await api.billingCheckout(
        `${origin}/billing?upgraded=1`, `${origin}/billing`,
      );
      window.location.assign(out.checkout_url);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not start checkout");
      setBusy(false);
    }
  };

  const managePlan = async () => {
    setBusy(true);
    setError(null);
    try {
      const out = await api.billingPortal(`${origin}/billing`);
      window.location.assign(out.portal_url);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not open the portal");
      setBusy(false);
    }
  };

  const crystalPct =
    usage?.crystals_used != null && usage?.crystal_cap
      ? (usage.crystals_used / usage.crystal_cap) * 100
      : null;

  return (
    <div className="mx-auto max-w-4xl space-y-5 p-6">
      <h1 className="text-lg font-semibold text-gray-900">Plan and usage</h1>
      {error && <ErrorBanner title="Billing" message={error} />}

      {expired && (
        <div className="rounded-xl border border-amber-300 bg-amber-50 p-4">
          <div className="flex items-center gap-2 font-medium text-amber-800">
            <Clock className="h-4 w-4" /> Trial expired — writing is paused
          </div>
          <p className="mt-1 text-sm text-amber-700">
            Your memories are safe and recall stays fully available.
            Upgrading resumes remembering immediately.
          </p>
        </div>
      )}

      {usage && usage.crystals_used != null && (
        <div className="space-y-4 rounded-xl border border-gray-200 bg-white p-5">
          {crystalPct != null ? (
            <Meter
              label="Memory"
              detail={`${usage.crystals_used.toLocaleString()} of ${usage.crystal_cap!.toLocaleString()} crystals`}
              pct={crystalPct}
              state={usage.crystal_state}
              note={
                usage.crystal_state === "blocked"
                  ? "Memory is full — everything stays recallable and exportable; upgrade to keep remembering."
                  : usage.crystal_state === "warning"
                    ? "Getting full — writes keep working a little past the limit, then pause."
                    : "Every conversation worth keeping becomes crystals here."
              }
            />
          ) : (
            <Meter
              label="Memory"
              detail={`${usage.crystals_used.toLocaleString()} crystals · unlimited`}
              pct={0}
              state="ok"
              note="This plan has no memory cap."
            />
          )}
          {usage.ai_capacity_pct != null && (
            <Meter
              label="Daily AI capacity"
              detail={`${usage.ai_capacity_pct}% used today`}
              pct={usage.ai_capacity_pct}
              state={usage.ai_capacity_pct >= 100 ? "blocked"
                : usage.ai_capacity_pct >= 90 ? "warning" : "ok"}
              note="Powers curation, gap filling, and agent runs. Resets at midnight UTC."
            />
          )}
        </div>
      )}

      <div className="grid gap-4 md:grid-cols-3">
        {PLANS.map((p) => {
          const current = !!tier && p.tierMatch(tier);
          return (
            <div
              key={p.name}
              className={`flex flex-col rounded-xl border bg-white p-5 ${
                current ? "border-indigo-400 ring-1 ring-indigo-200" : "border-gray-200"
              }`}
            >
              <div className="flex items-baseline justify-between">
                <span className="text-sm font-semibold text-gray-900">{p.name}</span>
                <span className="text-sm text-gray-600">{p.price}</span>
              </div>
              <ul className="mt-3 flex-1 space-y-1.5">
                {p.features.map((f) => (
                  <li key={f} className="flex gap-2 text-xs text-gray-600">
                    <Check className="mt-0.5 h-3 w-3 shrink-0 text-indigo-500" />
                    {f}
                  </li>
                ))}
              </ul>
              <div className="mt-4">
                {p.name === "Scale" ? (
                  <a
                    className="flex items-center justify-center gap-1.5 rounded-lg border border-gray-300 px-3 py-1.5 text-xs font-medium text-gray-700 hover:bg-gray-50"
                    href="mailto:hello@erahq.ai?subject=Crystal%20Cache%20Scale%20early%20access"
                  >
                    <Mail className="h-3.5 w-3.5" /> Contact for early access
                  </a>
                ) : current && !expired ? (
                  paid ? (
                    <CrystalButton onClick={managePlan} disabled={busy}>
                      <CreditCard className="h-4 w-4" /> Manage billing
                    </CrystalButton>
                  ) : (
                    <div className="rounded-lg bg-gray-50 px-3 py-1.5 text-center text-xs text-gray-500">
                      Current plan
                    </div>
                  )
                ) : p.name === "Starter" ? (
                  <CrystalButton onClick={upgrade} disabled={busy}>
                    <ShieldCheck className="h-4 w-4" /> Upgrade — $29/mo
                  </CrystalButton>
                ) : (
                  <div className="rounded-lg bg-gray-50 px-3 py-1.5 text-center text-xs text-gray-500">
                    {current ? "Current plan" : "—"}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      <p className="text-xs text-gray-400">
        A memory product never holds memories hostage: whatever your plan
        state, everything already stored stays recallable and exportable.
      </p>
    </div>
  );
}
