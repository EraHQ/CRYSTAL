// Plan & usage — Console Refine PR 10 (docs/CONSOLE_UX_AUDIT.md 5.9).
// The two meters with a note per state, the promise next to them, three
// plan cards from lib/plans.ts (Current / Recommended badges, one
// primary), the post-checkout confirmation when Stripe sends the
// browser back, and the self-host state (bug 4). Payment stays on
// Stripe's hosted pages; card data never touches this console. Tenants
// see capacity as a percent, never dollars (Q4=A).
import { useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { ArrowRightLeft, Check, Clock, CreditCard, Mail, ShieldCheck } from "lucide-react";
import { api, errorMessage } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { EmptyState, ErrorBanner, Meter, PageHeader, Segmented, StatusPill } from "@/components/ui";
import { PLANS, PROMISE, RECOMMENDED, capacityLabel, factsLabel, hasAnnual, planFor, priceLabel, type Plan } from "@/lib/plans";
import { cn } from "@/lib/utils";

function daysLeft(iso: string | null | undefined): number | null {
  if (!iso) return null;
  const ms = new Date(iso).getTime() - Date.now();
  return Math.max(0, Math.ceil(ms / 86_400_000));
}

export function memoryNote(state: "ok" | "warning" | "blocked", cap: number | null): string {
  if (cap == null) return "No memory limit on this plan.";
  if (state === "blocked") return "Full. New memories are paused. Recall, edits and export still work.";
  if (state === "warning") return `Almost full. At ${cap.toLocaleString()}, new memories pause; everything already stored stays recallable.`;
  return "Plenty of room.";
}

const BTN_PRIMARY = "inline-flex w-full items-center justify-center gap-1.5 rounded-md bg-accent px-3 py-2 text-[13px] font-medium text-on-accent hover:bg-accent-hover disabled:opacity-50";
const BTN_SECONDARY = "inline-flex w-full items-center justify-center gap-1.5 rounded-md border border-line px-3 py-2 text-[13px] font-medium text-fg-2 hover:bg-inset hover:text-fg disabled:opacity-50";

export function Billing() {
  const { me, refreshMe } = useAuth();
  const [params, setParams] = useSearchParams();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [period, setPeriod] = useState<"monthly" | "annual">("monthly");
  const polled = useRef(0);

  const tier = me?.subscription_tier ?? null;
  const isUser = me?.kind === "user";
  const plan = planFor(tier);
  const isTrial = !!tier && tier.startsWith("trial");
  const left = daysLeft(me?.trial_expires_at);
  const expired = isTrial && left !== null && left <= 0;
  const paid = !!tier && !isTrial && tier !== "free";
  const billingLive = me?.billing_live === true;
  const justUpgraded = params.get("upgraded") === "1";
  // Stripe returns the browser to a full URL, so it must include the
  // SPA's base path (/admin/ in prod). A bare /billing 404s at nginx.
  const billingUrl = `${window.location.origin}${import.meta.env.BASE_URL}billing`;

  // Back from checkout: the webhook stamps the tier a moment after
  // Stripe redirects, so re-read /v1/me a few times until it shows.
  useEffect(() => {
    if (!justUpgraded || paid || polled.current >= 6) return;
    polled.current += 1;
    const t = setTimeout(() => void refreshMe(), polled.current === 1 ? 0 : 2000);
    return () => clearTimeout(t);
  }, [justUpgraded, paid, refreshMe, me]);

  const upgrade = async (target: Plan) => {
    setBusy(true);
    setError(null);
    try {
      const out = await api.billingCheckout(`${billingUrl}?upgraded=1`, billingUrl, target.id === "scale" ? "scale" : "starter");
      window.location.assign(out.checkout_url);
    } catch (e) {
      setError(errorMessage(e, "Couldn't start checkout."));
      setBusy(false);
    }
  };
  // Every change to an existing paid plan happens in Stripe's portal.
  const managePlan = async () => {
    setBusy(true);
    setError(null);
    try {
      const out = await api.billingPortal(billingUrl);
      window.location.assign(out.portal_url);
    } catch (e) {
      setError(errorMessage(e, "Couldn't open the billing portal."));
      setBusy(false);
    }
  };

  const usage = me?.usage;
  if (!isUser || !usage) {
    return (
      <div>
        <PageHeader title="Plan & usage" />
        <EmptyState icon={ShieldCheck} title="Self-hosted: no plan limits" help="Plans, meters and billing apply to hosted workspaces. This deployment sets its own limits in its environment." />
      </div>
    );
  }

  const factPct = usage?.facts_used != null && usage.fact_cap ? Math.min(100, (usage.facts_used / usage.fact_cap) * 100) : 0;
  const capPct = usage?.ai_capacity_pct ?? null;

  const action = (p: Plan) => {
    const current = p.id === plan.id;
    if (current && !expired) {
      return paid
        ? <button type="button" onClick={() => void managePlan()} disabled={busy} className={BTN_SECONDARY}><CreditCard aria-hidden className="h-4 w-4" /> Manage billing</button>
        : <p className="text-center text-[12.5px] text-fg-3">Your plan</p>;
    }
    if (p.id === "free") return <p className="text-center text-[12.5px] text-fg-3">Always available</p>;
    if (paid) {
      return <button type="button" onClick={() => void managePlan()} disabled={busy} className={BTN_SECONDARY}><ArrowRightLeft aria-hidden className="h-4 w-4" /> Switch to {p.name}</button>;
    }
    if (!billingLive) {
      return (
        <a className={BTN_SECONDARY} href={`mailto:hello@erahq.ai?subject=Crystal%20${p.name}%20waitlist`}>
          <Mail aria-hidden className="h-3.5 w-3.5" /> Arriving shortly: join the list
        </a>
      );
    }
    return p.id === RECOMMENDED
      ? <button type="button" onClick={() => void upgrade(p)} disabled={busy} className={BTN_PRIMARY}><ShieldCheck aria-hidden className="h-4 w-4" /> Upgrade to {p.name}</button>
      : <button type="button" onClick={() => void upgrade(p)} disabled={busy} className={BTN_SECONDARY}>Choose {p.name}</button>;
  };

  return (
    <div>
      <PageHeader title="Plan & usage" meta={`${plan.name} plan · capacity resets at midnight UTC`} />
      {error && <ErrorBanner title="Billing" message={error} className="mb-4" />}

      {justUpgraded && (
        <section aria-label="Upgrade confirmation" className="mb-4 rounded-xl border border-ok/40 bg-ok-soft p-4">
          {paid ? (
            <>
              <h2 className="text-[15px] font-semibold text-fg">You're on {plan.name}</h2>
              <p className="mt-1 text-[13px] text-fg-2">Payment received by Stripe; the receipt is in your email and under Manage billing. New memories are saving again.</p>
              <table className="mt-3 w-full text-[13px]">
                <caption className="sr-only">What changed</caption>
                <thead><tr className="text-left text-fg-3"><th className="py-1 font-medium">What changed</th><th className="py-1 font-medium">Free</th><th className="py-1 font-medium">{plan.name}</th></tr></thead>
                <tbody className="text-fg">
                  <tr className="border-t border-line"><td className="py-1">Memory limit</td><td>{factsLabel(PLANS[0])} facts</td><td>{factsLabel(plan)} facts</td></tr>
                  <tr className="border-t border-line"><td className="py-1">Daily AI capacity</td><td>{capacityLabel(PLANS[0])}</td><td>{capacityLabel(plan)}</td></tr>
                  <tr className="border-t border-line"><td className="py-1">Opus</td><td>{PLANS[0].opus ? "Included" : "Not included"}</td><td>{plan.opus ? "Included" : "Not included"}</td></tr>
                </tbody>
              </table>
              <div className="mt-3 flex flex-wrap gap-2">
                <button type="button" onClick={() => void managePlan()} disabled={busy} className="inline-flex items-center gap-1.5 rounded-md border border-line px-3 py-1.5 text-[13px] font-medium text-fg-2 hover:bg-inset hover:text-fg"><CreditCard aria-hidden className="h-4 w-4" /> Manage billing</button>
                <button type="button" onClick={() => setParams({}, { replace: true })} className="rounded-md px-3 py-1.5 text-[13px] text-fg-2 hover:text-fg">Back to plans</button>
              </div>
            </>
          ) : (
            <p className="text-[13px] text-fg">Confirming your payment with Stripe… this takes a few seconds.</p>
          )}
        </section>
      )}

      {expired && (
        <div role="status" className="mb-4 rounded-xl border border-warn/40 bg-warn-soft p-4">
          <p className="flex items-center gap-2 text-[13.5px] font-medium text-fg"><Clock aria-hidden className="h-4 w-4" /> Trial ended: new memories are paused</p>
          <p className="mt-1 text-[13px] text-fg-2">Your memories are safe and recall stays available. Upgrading resumes remembering right away.</p>
        </div>
      )}

      <section aria-label="Usage" className="mb-4 rounded-xl border border-line bg-surface p-4 min-[720px]:p-5">
        {usage.facts_used != null && (
          <div>
            <Meter value={factPct} label="Memory"
              valueLabel={usage.fact_cap ? `${usage.facts_used.toLocaleString()} of ${usage.fact_cap.toLocaleString()} facts · ${Math.round(factPct)}%` : `${usage.facts_used.toLocaleString()} facts · unlimited`}
              tone={usage.fact_state === "blocked" ? "bad" : usage.fact_state === "warning" ? "warn" : "accent"} />
            <p className="mt-1 text-[12.5px] text-fg-3">{memoryNote(usage.fact_state, usage.fact_cap)}</p>
          </div>
        )}
        {capPct != null && (
          <div className="mt-4">
            <Meter value={capPct} label="Daily AI capacity" valueLabel={`${Math.round(capPct)}% used today`} />
            <p className="mt-1 text-[12.5px] text-fg-3">Powers ingesting documents, curation, gap filling and agent runs. Remembering and recall never count. Resets at midnight UTC; a large ingest can carry into the next day.</p>
          </div>
        )}
        <p className="mt-4 border-t border-line pt-3 text-[12.5px] text-fg-2">{PROMISE}</p>
      </section>

      {hasAnnual() && (
        <div className="mb-3 flex items-center gap-3">
          <Segmented label="Billing period" size="sm" value={period} onChange={setPeriod}
            options={[{ value: "monthly", label: "Monthly" }, { value: "annual", label: "Annual" }]} />
          <span className="text-[12.5px] text-fg-3">Annual: 2 months free</span>
        </div>
      )}

      <div className="grid gap-3 min-[720px]:grid-cols-3">
        {PLANS.map((p) => {
          const current = p.id === plan.id;
          const recommended = p.id === RECOMMENDED && !paid;
          return (
            <article key={p.id} aria-label={`${p.name} plan`}
              className={cn("flex flex-col rounded-xl border bg-surface p-4", current ? "border-accent" : recommended ? "border-line-strong" : "border-line")}>
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-[15px] font-semibold text-fg">{p.name}</span>
                {current && <StatusPill variant="info">Current</StatusPill>}
                {recommended && !current && <StatusPill variant="ok">Recommended</StatusPill>}
                <span className="flex-1" />
                <span className="text-[13px] text-fg-2">{priceLabel(p, period)}</span>
              </div>
              <ul className="mt-3 flex-1 space-y-1.5">
                {p.features.map((f) => (
                  <li key={f} className="flex gap-2 text-[12.5px] text-fg-2"><Check aria-hidden className="mt-0.5 h-3.5 w-3.5 shrink-0 text-accent" /> {f}</li>
                ))}
              </ul>
              <div className="mt-4">{action(p)}</div>
            </article>
          );
        })}
      </div>
      <p className="mt-4 text-[12px] text-fg-3">Changes to a paid plan, invoices and cancellation happen in the Stripe billing portal. Your API key and settings live in <Link to="/settings" className="text-accent hover:underline">Settings</Link>.</p>
    </div>
  );
}
