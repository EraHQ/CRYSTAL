// Billing (T1c, 2026-09-25; supersedes the S4=B trial-era page).
// Three-column pricing truth + the two capacity meters (Q4=A: customers
// see capacity, never dollars). States: free (upgrade CTAs), paid
// (plan changes and billing via the Stripe portal, Q4=A 2026-09-30),
// legacy trial (countdown/expired banner until those accounts age out).
// Payment stays on Stripe's hosted pages; card data never touches this
// console. Memory is metered in crystal facts (unit switch 2026-09-30).
import { useState } from "react";
import { ArrowRightLeft, Check, Clock, CreditCard, Mail, ShieldCheck } from "lucide-react";
import { api } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { CrystalButton, ErrorBanner } from "@/components/ui";

type PlanId = "free" | "starter" | "scale";

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

const PLANS: {
  id: PlanId; name: string; price: string; cta: string;
  tierMatch: (t: string) => boolean; features: string[];
}[] = [
  {
    id: "free", name: "Free", price: "$0", cta: "",
    tierMatch: (t) => t === "free",
    features: [
      "2,500 crystal facts of memory",
      "Full self-curation: conflicts, duplicates, gaps, quality tiers",
      "Daily AI capacity for everyday use",
      "Reads never stop, at any tier",
    ],
  },
  {
    id: "starter", name: "Starter", price: "$29/mo", cta: "Upgrade: $29/mo",
    tierMatch: (t) => t.startsWith("starter") || t.startsWith("trial"),
    features: [
      "50,000 crystal facts of memory",
      "Everything in Free",
      "10x the daily AI capacity",
      "Priority background curation",
    ],
  },
  {
    id: "scale", name: "Scale", price: "$49/mo", cta: "Upgrade: $49/mo",
    tierMatch: (t) => t.startsWith("scale"),
    features: [
      "Unlimited crystal facts",
      "Everything in Starter",
      "5x Starter's daily AI capacity",
      "Built for heavy daily use",
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
  // Launch gate (2026-09-27): checkout only against LIVE Stripe. Until
  // then the paid cards offer the waitlist, honestly.
  const billingLive = me?.billing_live === true;
  // Stripe returns the browser to a full URL, so it must include the
  // SPA's base path (/admin/ in prod). A bare /billing 404s at nginx
  // (2026-09-30: every return from Stripe landed on a 404).
  const billingUrl = `${window.location.origin}${import.meta.env.BASE_URL}billing`;

  const upgrade = async (plan: "starter" | "scale") => {
    setBusy(true);
    setError(null);
    try {
      const out = await api.billingCheckout(
        `${billingUrl}?upgraded=1`, billingUrl, plan,
      );
      window.location.assign(out.checkout_url);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not start checkout");
      setBusy(false);
    }
  };

  // Q4=A: every change to an existing paid plan (switch, cancel,
  // invoices) happens in Stripe's portal; checkout is for free tenants.
  const managePlan = async () => {
    setBusy(true);
    setError(null);
    try {
      const out = await api.billingPortal(billingUrl);
      window.location.assign(out.portal_url);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not open the portal");
      setBusy(false);
    }
  };

  const factPct =
    usage?.facts_used != null && usage?.fact_cap
      ? (usage.facts_used / usage.fact_cap) * 100
      : null;

  const quietNote = (text: string) => (
    <div className="rounded-lg bg-gray-50 px-3 py-1.5 text-center text-xs text-gray-500">
      {text}
    </div>
  );

  const action = (p: (typeof PLANS)[number]) => {
    const current = !!tier && p.tierMatch(tier);
    if (current && !expired) {
      return paid ? (
        <CrystalButton onClick={managePlan} disabled={busy}>
          <CreditCard className="h-4 w-4" /> Manage billing
        </CrystalButton>
      ) : quietNote("Current plan");
    }
    if (p.id === "free") return quietNote(paid ? "Always available" : "Current plan");
    if (paid) {
      return (
        <CrystalButton onClick={managePlan} disabled={busy}>
          <ArrowRightLeft className="h-4 w-4" /> Switch to {p.name}
        </CrystalButton>
      );
    }
    if (!billingLive) {
      return (
        <a
          className="flex items-center justify-center gap-1.5 rounded-lg border border-gray-300 px-3 py-1.5 text-xs font-medium text-gray-700 hover:bg-gray-50"
          href={`mailto:hello@erahq.ai?subject=Crystal%20${p.name}%20waitlist`}
        >
          <Mail className="h-3.5 w-3.5" /> Arriving shortly: join the list
        </a>
      );
    }
    const plan = p.id === "scale" ? "scale" : "starter";
    return (
      <CrystalButton onClick={() => upgrade(plan)} disabled={busy}>
        <ShieldCheck className="h-4 w-4" /> {p.cta}
      </CrystalButton>
    );
  };

  return (
    <div className="mx-auto max-w-4xl space-y-5 p-6">
      <h1 className="text-lg font-semibold text-gray-900">Plan and usage</h1>
      {error && <ErrorBanner title="Billing" message={error} />}

      {expired && (
        <div className="rounded-xl border border-amber-300 bg-amber-50 p-4">
          <div className="flex items-center gap-2 font-medium text-amber-800">
            <Clock className="h-4 w-4" /> Trial expired: writing is paused
          </div>
          <p className="mt-1 text-sm text-amber-700">
            Your memories are safe and recall stays fully available.
            Upgrading resumes remembering immediately.
          </p>
        </div>
      )}

      {usage && usage.facts_used != null && (
        <div className="space-y-4 rounded-xl border border-gray-200 bg-white p-5">
          {factPct != null ? (
            <Meter
              label="Memory"
              detail={`${usage.facts_used.toLocaleString()} of ${usage.fact_cap!.toLocaleString()} crystal facts`}
              pct={factPct}
              state={usage.fact_state}
              note={
                usage.fact_state === "blocked"
                  ? "Memory is full. Everything stays recallable and exportable; upgrade to keep remembering."
                  : usage.fact_state === "warning"
                    ? "Getting full. Writes keep working a little past the limit, then pause."
                    : "Every fact worth keeping becomes a crystal fact here. Reads never count."
              }
            />
          ) : (
            <Meter
              label="Memory"
              detail={`${usage.facts_used.toLocaleString()} crystal facts · unlimited`}
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
              note="Powers ingesting documents, curation, gap filling and agent runs. Remembering and recall never count. Resets at midnight UTC."
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
              <div className="mt-4">{action(p)}</div>
            </div>
          );
        })}
      </div>

      <p className="text-xs text-gray-400">
        A memory product never holds memories hostage: whatever your plan
        state, everything already stored stays recallable and exportable.
        Only facts you store count toward your plan; facts Crystal derives
        while curating are free.
      </p>
    </div>
  );
}
