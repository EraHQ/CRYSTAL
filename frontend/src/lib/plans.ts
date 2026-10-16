// The ONE limits table (Console Refine PR 10, docs/CONSOLE_UX_AUDIT.md
// 4.4 and 5.9): Billing and PlanWallModal both read from here, so the
// multipliers cannot disagree again. The numbers mirror the server's
// control/admission.py TIER_TABLE (ratified 2026-09-23 / Q3=A
// 2026-09-30): Free 2,500 facts and $0.50 a day of managed AI; Starter
// 50,000 facts (20x) and 10x the daily capacity; Scale unlimited facts
// and 50x (5x Starter). Dollars are the platform's language, never a
// tenant's: capacity is shown as a multiple or a percent.
//
// Annual prices: none exist in Stripe or in the server's settings
// (config.py has stripe_price_starter / stripe_price_scale only), so
// `annualUsd` is null everywhere and Billing hides the Monthly / Annual
// switch (Q64). Adding annual = a Stripe price, two settings, a period
// on CheckoutRequest, and the numbers here.

export type PlanId = "free" | "starter" | "scale";

export interface Plan {
  id: PlanId;
  name: string;
  monthlyUsd: number;
  annualUsd: number | null;
  /** Memory limit in facts; null = unlimited. */
  facts: number | null;
  /** Daily AI capacity as a multiple of Free's. */
  capacityX: number;
  opus: boolean;
  features: string[];
  /** Stamped tier strings that mean this plan (control/admission.py). */
  tierMatch: (tier: string) => boolean;
}

export const PLANS: Plan[] = [
  {
    id: "free", name: "Free", monthlyUsd: 0, annualUsd: null, facts: 2_500, capacityX: 1, opus: false,
    features: ["2,500 facts of memory", "Full self-curation: conflicts, duplicates, gaps, quality tiers", "Daily AI capacity for everyday use", "Haiku and Sonnet", "Recall never stops, on any plan"],
    tierMatch: (t) => t === "free",
  },
  {
    id: "starter", name: "Starter", monthlyUsd: 29, annualUsd: null, facts: 50_000, capacityX: 10, opus: true,
    features: ["50,000 facts of memory, 20x Free", "10x Free's daily AI capacity", "Opus alongside Haiku and Sonnet", "Everything in Free"],
    tierMatch: (t) => t.startsWith("starter") || t.startsWith("trial") || t === "pro",
  },
  {
    id: "scale", name: "Scale", monthlyUsd: 49, annualUsd: null, facts: null, capacityX: 50, opus: true,
    features: ["Unlimited facts", "50x Free's daily AI capacity, 5x Starter", "Built for heavy daily use", "Everything in Starter"],
    tierMatch: (t) => t.startsWith("scale"),
  },
];

export const RECOMMENDED: PlanId = "starter";

export const PROMISE =
  "Whatever your plan, everything you stored stays recallable and exportable. Only facts you store count; facts Crystal derives while curating are free.";

export function planFor(tier: string | null | undefined): Plan {
  if (!tier) return PLANS[0];
  return PLANS.find((p) => p.tierMatch(tier)) ?? PLANS[0];
}

/** The plan to offer from this tier: Starter from Free, Scale from Starter, none from Scale. */
export function nextPlan(tier: string | null | undefined): Plan | null {
  const cur = planFor(tier);
  const idx = PLANS.findIndex((p) => p.id === cur.id);
  return PLANS[idx + 1] ?? null;
}

export function hasAnnual(): boolean {
  return PLANS.some((p) => p.annualUsd != null);
}

export function priceLabel(p: Plan, period: "monthly" | "annual" = "monthly"): string {
  if (p.monthlyUsd === 0) return "$0";
  if (period === "annual" && p.annualUsd != null) return `$${Math.round(p.annualUsd / 12)}/mo, $${p.annualUsd}/yr`;
  return `$${p.monthlyUsd}/mo`;
}

export function factsLabel(p: Plan): string {
  return p.facts == null ? "Unlimited" : p.facts.toLocaleString();
}

export function capacityLabel(p: Plan): string {
  return p.capacityX === 1 ? "1x" : `${p.capacityX}x Free`;
}
