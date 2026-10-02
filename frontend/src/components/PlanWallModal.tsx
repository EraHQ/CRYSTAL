// PlanWallModal (v110, Q28=A, 2026-10-01): the ONE upgrade prompt for
// every plan wall the server raises. The API client announces a wall
// through the `crystal:plan-wall` event the moment any response carries
// X-Plan-Wall, so no page has to remember to show this. The copy is
// wall-specific: what hit, what the next plan gives for exactly that,
// and one button to Billing. Dismissing leaves the user where they were.
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowUpRight, Clock, Cpu, Database, X } from "lucide-react";
import { useAuth } from "@/lib/auth";

type Wall = {
  code: string;
  message: string;
  status: number;
};

const COPY: Record<
  string,
  { title: string; icon: typeof Database; benefits: string[]; cta: string }
> = {
  memory_full: {
    title: "Your memory is full",
    icon: Database,
    benefits: [
      "Starter holds 50,000 crystal facts, twenty times the free bank",
      "Scale is unlimited",
      "Everything already stored stays recallable and exportable",
    ],
    cta: "See plans",
  },
  daily_capacity: {
    title: "Daily AI capacity used up",
    icon: Clock,
    benefits: [
      "Starter gives 10x the daily capacity for ingest, curation and agent runs",
      "Scale gives 50x",
      "Remembering and recall keep working on every plan",
    ],
    cta: "See plans",
  },
  monthly_budget: {
    title: "Monthly AI budget reached",
    icon: Clock,
    benefits: [
      "Starter's monthly budget is 10x the free plan's",
      "Scale's is 50x",
      "Or use your own API key in Settings to continue immediately",
    ],
    cta: "See plans",
  },
  model_not_in_plan: {
    title: "That model needs a bigger plan",
    icon: Cpu,
    benefits: [
      "Starter and Scale include Opus alongside Haiku and Sonnet",
      "10x the daily AI capacity to run it",
      "Or use your own API key for any model",
    ],
    cta: "See plans",
  },
  trial_expired: {
    title: "Your trial has ended",
    icon: Clock,
    benefits: [
      "Upgrade to resume remembering immediately",
      "Your memories are safe and recall stays available",
    ],
    cta: "See plans",
  },
};

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
  const copy = COPY[wall.code] ?? {
    title: "This needs a bigger plan",
    icon: ArrowUpRight,
    benefits: ["See what each plan includes on the Billing page"],
    cta: "See plans",
  };
  const Icon = copy.icon;
  const paid = !!me?.subscription_tier && me.subscription_tier !== "free";

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="plan-wall-title"
      onClick={() => setWall(null)}
    >
      <div
        className="w-full max-w-md rounded-2xl border border-gray-200 bg-white p-6 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-indigo-50">
            <Icon className="h-5 w-5 text-indigo-600" />
          </div>
          <button
            onClick={() => setWall(null)}
            className="rounded-md p-1 text-gray-400 hover:bg-gray-100 hover:text-gray-600"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <h2 id="plan-wall-title" className="mt-4 text-base font-semibold text-gray-900">
          {copy.title}
        </h2>
        {wall.message && (
          <p className="mt-1 text-sm text-gray-600">{wall.message}</p>
        )}
        <ul className="mt-4 space-y-1.5">
          {copy.benefits.map((b) => (
            <li key={b} className="flex gap-2 text-sm text-gray-700">
              <ArrowUpRight className="mt-0.5 h-4 w-4 shrink-0 text-indigo-500" />
              {b}
            </li>
          ))}
        </ul>
        <div className="mt-5 flex gap-2">
          <button
            onClick={() => {
              setWall(null);
              navigate("/billing");
            }}
            className="flex-1 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-700"
          >
            {paid ? "Change plan" : copy.cta}
          </button>
          <button
            onClick={() => setWall(null)}
            className="rounded-lg border border-gray-300 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50"
          >
            Not now
          </button>
        </div>
      </div>
    </div>
  );
}
