// Billing (L2-S4=B, 2026-09-08). One page, three states:
//  - active trial  → countdown + "Upgrade — $29/mo" (Stripe hosted Checkout)
//  - expired trial → writes-paused notice (memories safe, recall alive) + upgrade
//  - paid          → "Manage billing" (Stripe hosted customer portal)
// Card data never touches this app — both buttons redirect to Stripe pages.
import { useState } from "react";
import { CreditCard, ShieldCheck, Clock } from "lucide-react";
import { api } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { CrystalButton, ErrorBanner } from "@/components/ui";

function daysLeft(iso: string | null | undefined): number | null {
  if (!iso) return null;
  const ms = new Date(iso).getTime() - Date.now();
  return Math.max(0, Math.ceil(ms / 86_400_000));
}

export function Billing() {
  const { me } = useAuth();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const tier = me?.subscription_tier ?? null;
  const isTrial = !!tier && tier.startsWith("trial");
  const left = daysLeft(me?.trial_expires_at);
  const expired = isTrial && left !== null && left <= 0;
  const paid = !!tier && !isTrial;
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

  return (
    <div className="mx-auto max-w-2xl space-y-4 p-6">
      <h1 className="text-lg font-semibold text-gray-900">Billing</h1>
      {error && <ErrorBanner title="Billing" message={error} />}

      {expired && (
        <div className="rounded-xl border border-amber-300 bg-amber-50 p-4">
          <div className="flex items-center gap-2 font-medium text-amber-900">
            <Clock className="h-4 w-4" /> Trial expired — writing is paused
          </div>
          <p className="mt-1 text-sm text-amber-800">
            Your memories are safe and recall stays fully available. Upgrading
            resumes remembering, ingestion, and agent runs immediately.
          </p>
        </div>
      )}

      <div className="rounded-xl border border-gray-200 bg-white p-5">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-medium text-gray-900">
              {paid && "Starter — $29/mo"}
              {isTrial && !expired &&
                `Free trial — ${left} day${left === 1 ? "" : "s"} left`}
              {isTrial && expired && "Free trial — ended"}
              {!tier && "No subscription on this account"}
            </div>
            <p className="mt-0.5 text-xs text-gray-500">
              {paid
                ? "Invoices, payment method, and cancellation are managed on Stripe's secure portal."
                : "The Starter plan is $29/month. Payment happens on Stripe's hosted page — card details never touch this console."}
            </p>
          </div>
          {paid ? (
            <CrystalButton onClick={managePlan} disabled={busy}>
              <CreditCard className="h-4 w-4" /> Manage billing
            </CrystalButton>
          ) : (
            <CrystalButton onClick={upgrade} disabled={busy}>
              <ShieldCheck className="h-4 w-4" /> Upgrade — $29/mo
            </CrystalButton>
          )}
        </div>
      </div>

      <p className="text-xs text-gray-400">
        A memory product never holds memories hostage: whatever your plan
        state, everything already stored stays recallable and exportable.
      </p>
    </div>
  );
}
