// PlanWallModal — Console Refine PR 10 (docs/CONSOLE_UX_AUDIT.md 5.16).
// The ONE upgrade prompt for every plan wall the server raises: the API
// client announces a wall through the `crystal:plan-wall` event the
// moment any response carries X-Plan-Wall (jsonFetch, authedFetch and
// the upload XHR all dispatch it since PR 2 / PR 5). Built on Modal
// (focus trap, Escape, backdrop close). The body is specific from what
// the console already knows (/v1/me usage: facts used and cap, today's
// capacity percent; midnight UTC; the first of next month) with the
// server's message as the fallback; the server does not yet send
// `needed` (Part 6), so "would add n facts" is not stated. Benefits and
// prices come from lib/plans.ts so they match Billing. Every wall keeps
// a path that is not an upgrade.
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowUpRight, Clock, Cpu, Database } from "lucide-react";
import { useAuth } from "@/lib/auth";
import type { Me } from "@/lib/auth";
import { Meter, Modal } from "@/components/ui";
import { PLANS, nextPlan, planFor, type Plan } from "@/lib/plans";

export type Wall = { code: string; message: string; status: number };

export function untilMidnightUtc(now: Date = new Date()): string {
  const next = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate() + 1);
  const mins = Math.max(0, Math.round((next - now.getTime()) / 60_000));
  const h = Math.floor(mins / 60);
  const m = mins % 60;
  return h > 0 ? `${h} h ${m} min` : `${m} min`;
}

export function nextMonthStart(now: Date = new Date()): string {
  const d = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth() + 1, 1));
  return d.toLocaleDateString(undefined, { month: "long", day: "numeric", timeZone: "UTC" });
}

export interface WallView {
  title: string;
  icon: typeof Database;
  body: string;
  meter?: { value: number; label: string; valueLabel: string };
  benefits: string[];
  /** The non-upgrade path: label and where it goes (null = just close). */
  other: { label: string; to: string | null };
}

// What each wall says, from the console's own numbers (5.16 items 2 to 5).
export function wallView(wall: Wall, me: Me | null, now: Date = new Date()): WallView {
  const cur = planFor(me?.subscription_tier);
  const next = nextPlan(me?.subscription_tier);
  const u = me?.usage;
  const starter = PLANS[1];
  switch (wall.code) {
    case "memory_full": {
      const used = u?.facts_used ?? null;
      const cap = u?.fact_cap ?? cur.facts;
      const room = used != null && cap != null ? Math.max(0, cap - used) : null;
      return {
        title: "Your memory is full",
        icon: Database,
        body: room != null
          ? (room > 0
            ? `${cur.name} has room for ${room.toLocaleString()} more (${used!.toLocaleString()} of ${cap!.toLocaleString()} used).`
            : `${cur.name} holds ${cap!.toLocaleString()} facts and they are all used (${used!.toLocaleString()} stored).`)
          : wall.message,
        meter: used != null && cap ? { value: Math.min(100, (used / cap) * 100), label: "Memory", valueLabel: `${used.toLocaleString()} of ${cap.toLocaleString()}` } : undefined,
        benefits: [
          `${starter.name} holds ${starter.facts!.toLocaleString()} facts, 20x Free`,
          "Scale is unlimited",
          "Everything already stored stays recallable and exportable",
        ],
        other: { label: "Free up space", to: "/" },
      };
    }
    case "daily_capacity": {
      const pct = u?.ai_capacity_pct ?? null;
      const leftPct = pct != null ? Math.max(0, 100 - pct) : null;
      return {
        title: "Daily AI capacity used up",
        icon: Clock,
        body: leftPct != null
          ? `You have ${Math.round(leftPct)}% of today's capacity left. It resets at midnight UTC, in ${untilMidnightUtc(now)}. Remembering and recall keep working.`
          : wall.message,
        meter: pct != null ? { value: pct, label: "Daily AI capacity", valueLabel: `${Math.round(pct)}% used today` } : undefined,
        benefits: [
          `${starter.name} gives ${starter.capacityX}x the daily capacity for ingest, curation and agent runs`,
          `Scale gives ${PLANS[2].capacityX}x`,
          "Or use your own provider key: that work never counts",
        ],
        other: { label: "Use my own key", to: "/settings/model" },
      };
    }
    case "monthly_budget":
      return {
        title: "This month's AI capacity is used up",
        icon: Clock,
        body: `It renews on ${nextMonthStart(now)}. Saving and recall keep working until then.`,
        benefits: [
          `${starter.name}'s monthly capacity is ${starter.capacityX}x Free's`,
          `Scale's is ${PLANS[2].capacityX}x`,
          "Or use your own provider key to continue right away",
        ],
        other: { label: "Use my own key", to: "/settings/model" },
      };
    case "model_not_in_plan":
      return {
        title: "That model needs a bigger plan",
        icon: Cpu,
        body: wall.message || `${cur.name} includes Haiku and Sonnet.`,
        benefits: [
          "Starter and Scale include Opus alongside Haiku and Sonnet",
          `${starter.capacityX}x the daily AI capacity to run it`,
          "Or use your own provider key for any model",
        ],
        other: { label: "Use Sonnet", to: "/settings/model" },
      };
    case "trial_expired":
      return {
        title: "Your trial has ended",
        icon: Clock,
        body: wall.message || "New memories are paused. Everything stored stays recallable and exportable.",
        benefits: ["Upgrade to resume remembering right away", "Your memories are safe and recall stays available"],
        other: { label: "Not now", to: null },
      };
    default:
      return {
        title: "This needs a bigger plan",
        icon: ArrowUpRight,
        body: wall.message || "See what each plan includes on Plan & usage.",
        benefits: next ? [`${next.name} is $${next.monthlyUsd}/mo`] : [],
        other: { label: "Not now", to: null },
      };
  }
}

export function PlanWallModal() {
  const [wall, setWall] = useState<Wall | null>(null);
  const navigate = useNavigate();
  const { me } = useAuth();

  useEffect(() => {
    const onWall = (e: Event) => {
      const detail = (e as CustomEvent<Wall>).detail;
      if (detail?.code) setWall(detail);
    };
    window.addEventListener("crystal:plan-wall", onWall);
    return () => window.removeEventListener("crystal:plan-wall", onWall);
  }, []);

  if (!wall) return null;
  const view = wallView(wall, me);
  const Icon = view.icon;
  const next: Plan | null = nextPlan(me?.subscription_tier);
  const paid = !!me?.subscription_tier && me.subscription_tier !== "free" && !me.subscription_tier.startsWith("trial");
  const close = () => setWall(null);
  const go = (to: string) => { close(); navigate(to); };

  return (
    <Modal open onClose={close} title={<span className="flex items-center gap-2"><Icon aria-hidden className="h-4 w-4 text-accent" /> {view.title}</span>}
      footer={<>
        <span className="mr-auto text-[11.5px] text-fg-3">Esc closes</span>
        <button type="button" onClick={close} className="rounded-md px-2.5 py-1.5 text-[13px] text-fg-2 hover:text-fg">Not now</button>
        {view.other.to && (
          <button type="button" onClick={() => go(view.other.to!)} className="rounded-md border border-line px-2.5 py-1.5 text-[13px] font-medium text-fg-2 hover:bg-inset hover:text-fg">{view.other.label}</button>
        )}
        <button type="button" onClick={() => go("/billing")} className="rounded-md bg-accent px-3 py-1.5 text-[13px] font-medium text-on-accent hover:bg-accent-hover">
          {paid || !next ? "Compare all plans" : `Upgrade to ${next.name} · $${next.monthlyUsd}/mo`}
        </button>
      </>}>
      <p className="text-[13.5px] text-fg">{view.body}</p>
      {view.meter && <Meter className="mt-3" value={view.meter.value} label={view.meter.label} valueLabel={view.meter.valueLabel} />}
      <ul className="mt-4 space-y-1.5">
        {view.benefits.map((b) => (
          <li key={b} className="flex gap-2 text-[13px] text-fg-2"><ArrowUpRight aria-hidden className="mt-0.5 h-4 w-4 shrink-0 text-accent" /> {b}</li>
        ))}
      </ul>
      {!paid && next && (
        <button type="button" onClick={() => go("/billing")} className="mt-3 text-[12.5px] text-accent hover:underline">Compare all plans</button>
      )}
    </Modal>
  );
}
