// Cognition tracker widgets (2026-07-14, ratified Q1C + Q2 B+C):
// EventsFeed (the machinery narrating itself, live slide-in ticker),
// AmendmentsPanel (the contract-amendment audit trail), badges
// (retry routes, provenance, infeasibility), LiveTimer + CountUp
// (motion between polls), ScoreSparkline + StepDurationBars
// (recharts over real run data).
import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import {
  FileEdit, GitBranch, Layers, RefreshCcw, Repeat, ShieldAlert,
  Sparkles, TimerReset, Wrench,
} from "lucide-react";
import {
  Bar, BarChart, Line, LineChart, ResponsiveContainer, Tooltip,
  XAxis, YAxis,
} from "recharts";

// ----- Events feed -----------------------------------------------------------

export interface CogEvent {
  ts: string;
  kind: string;
  step_id?: number;
  [k: string]: unknown;
}

const EVENT_META: Record<string, { Icon: typeof Sparkles; tint: string; label: (e: CogEvent) => string }> = {
  agentic_step: {
    Icon: Wrench, tint: "text-indigo-600 bg-indigo-50",
    label: (e) => `agentic worker finished step ${e.step_id} — ${e.tool_calls} tool call${e.tool_calls === 1 ? "" : "s"}, ${e.iterations} iterations`,
  },
  agentic_fallback: {
    Icon: ShieldAlert, tint: "text-amber-700 bg-amber-50",
    label: (e) => `agentic session failed on step ${e.step_id} — fell back to classic composition`,
  },
  composition_continued: {
    Icon: Repeat, tint: "text-blue-600 bg-blue-50",
    label: (e) => `step ${e.step_id} output continued ${e.continuations}× (${Number(e.chars).toLocaleString()} chars)`,
  },
  composition_empty_retry: {
    Icon: RefreshCcw, tint: "text-amber-700 bg-amber-50",
    label: (e) => `step ${e.step_id} returned empty — retrying (attempt ${e.attempt})`,
  },
  validator_envelopes: {
    Icon: Layers, tint: "text-purple-600 bg-purple-50",
    label: (e) => `deliverable split into ${e.parts} envelopes for validation (${Number(e.chars).toLocaleString()} chars)`,
  },
  contract_amended: {
    Icon: FileEdit, tint: "text-teal-700 bg-teal-50",
    label: (e) => `contract criterion ${Number(e.criterion_index) + 1} amended on evidence`,
  },
  amendment_rejected: {
    Icon: ShieldAlert, tint: "text-red-600 bg-red-50",
    label: (e) => `amendment for criterion ${Number(e.criterion_index) + 1} rejected — not flagged by the validator`,
  },
  research_degraded: {
    Icon: GitBranch, tint: "text-amber-700 bg-amber-50",
    label: (e) => `research step ${e.step_id} degraded to web search (agentic workers off)`,
  },
};

function relTime(ts: string): string {
  const s = Math.max(0, (Date.now() - new Date(ts).getTime()) / 1000);
  if (s < 60) return `${Math.floor(s)}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  return `${Math.floor(s / 3600)}h ago`;
}

export function EventsFeed({ events, live }: { events: CogEvent[]; live: boolean }) {
  const reduce = useReducedMotion();
  if (!events?.length) return null;
  const tail = [...events].slice(-8).reverse();
  return (
    <div>
      <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1.5 flex items-center gap-1.5">
        Activity
        {live && <span className="inline-block w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />}
      </h4>
      <div className="space-y-1">
        <AnimatePresence initial={false}>
          {tail.map((e) => {
            const meta = EVENT_META[e.kind] ?? {
              Icon: Sparkles, tint: "text-gray-500 bg-gray-50",
              label: () => e.kind.replace(/_/g, " "),
            };
            return (
              <motion.div
                key={`${e.ts}-${e.kind}`}
                initial={reduce ? false : { opacity: 0, y: -8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ type: "spring", stiffness: 400, damping: 28 }}
                className="flex items-center gap-2 text-[11px]"
              >
                <span className={`p-1 rounded ${meta.tint}`}>
                  <meta.Icon className="h-3 w-3" />
                </span>
                <span className="text-gray-600 flex-1 truncate">{meta.label(e)}</span>
                <span className="text-gray-300 flex-shrink-0">{relTime(e.ts)}</span>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </div>
  );
}

// ----- Amendments panel -------------------------------------------------------

export interface Amendment {
  attempt: number;
  index: number;
  original: string;
  amended: string;
  evidence: string;
}

export function AmendmentsPanel({ amendments }: { amendments: Amendment[] }) {
  if (!amendments?.length) return null;
  return (
    <div className="mt-2 border border-teal-200 bg-teal-50 rounded p-2 space-y-1.5">
      <div className="flex items-center gap-1.5 text-[11px] font-semibold text-teal-800">
        <FileEdit className="h-3 w-3" />
        Contract amended on evidence
      </div>
      {amendments.map((a, i) => (
        <div key={i} className="text-[11px] space-y-0.5">
          <div className="text-gray-500 line-through">{a.original}</div>
          <div className="text-teal-900 font-medium">→ {a.amended}</div>
          {a.evidence && (
            <div className="text-gray-500 italic">evidence: {a.evidence}</div>
          )}
          <div className="text-gray-400 text-[10px]">applied on attempt {a.attempt}</div>
        </div>
      ))}
    </div>
  );
}

// ----- Badges -----------------------------------------------------------------

const ROUTE_STYLES: Record<string, string> = {
  compose_only: "bg-blue-50 text-blue-700",
  gap_fill: "bg-amber-50 text-amber-700",
  replan: "bg-purple-50 text-purple-700",
  give_up: "bg-gray-100 text-gray-600",
  amend_contract: "bg-teal-50 text-teal-700",
};

export function RouteBadge({ route }: { route?: string }) {
  if (!route) return null;
  return (
    <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${ROUTE_STYLES[route] ?? "bg-gray-50 text-gray-500"}`}>
      {route.replace(/_/g, " ")}
    </span>
  );
}

export function ProvenanceBadges({ output }: { output: any }) {
  const findings: any[] = Array.isArray(output?.findings) ? output.findings : [];
  const flags = new Set<string>();
  if (findings.some((f) => f?.source === "github_api")) flags.add("github api");
  if (findings.some((f) => f?.rendered)) flags.add("rendered");
  if (findings.some((f) => f?.salvaged)) flags.add("salvaged");
  if (output?.retried_query || findings.some((f) => f?.retried_query)) flags.add("retried query");
  if (output?.degraded) flags.add("degraded");
  if (!flags.size) return null;
  return (
    <span className="inline-flex gap-1 ml-1">
      {[...flags].map((f) => (
        <span key={f} className="px-1 py-0.5 rounded text-[9px] bg-slate-100 text-slate-600">{f}</span>
      ))}
    </span>
  );
}

export function InfeasibleFlag() {
  return (
    <span className="ml-1 px-1.5 py-0.5 rounded text-[9px] font-medium bg-orange-50 text-orange-700 border border-orange-200">
      possibly infeasible
    </span>
  );
}

// ----- Live timer + count-up ---------------------------------------------------

/** Ticks every second against a start timestamp — smooth motion between
 * the 2s detail polls. */
export function LiveTimer({ since }: { since: string }) {
  const [, force] = useState(0);
  useEffect(() => {
    const id = setInterval(() => force((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, []);
  const s = Math.max(0, Math.floor((Date.now() - new Date(since).getTime()) / 1000));
  const mm = Math.floor(s / 60);
  const ss = s % 60;
  return (
    <span className="inline-flex items-center gap-0.5 font-mono text-[11px] text-blue-600">
      <TimerReset className="h-3 w-3" />
      {mm}:{String(ss).padStart(2, "0")}
    </span>
  );
}

/** Animates numeric changes between polls (token/cost counters). */
export function CountUp({ value, decimals = 0, prefix = "" }: {
  value: number; decimals?: number; prefix?: string;
}) {
  const reduce = useReducedMotion();
  const [shown, setShown] = useState(value);
  const prev = useRef(value);
  useEffect(() => {
    if (reduce || prev.current === value) { setShown(value); prev.current = value; return; }
    const from = prev.current;
    prev.current = value;
    const t0 = performance.now();
    const dur = 600;
    let raf = 0;
    const tick = (t: number) => {
      const p = Math.min(1, (t - t0) / dur);
      const eased = 1 - Math.pow(1 - p, 3);
      setShown(from + (value - from) * eased);
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value, reduce]);
  return <span>{prefix}{shown.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}</span>;
}

// ----- Charts (Q2 C) ------------------------------------------------------------

export function ScoreSparkline({ attempts, current }: {
  attempts: Array<{ attempt: number; validation: { score: number } | null }>;
  current: { score: number } | null;
}) {
  const data = attempts.map((a) => ({
    attempt: `#${a.attempt}`,
    score: Math.round(((a.validation?.score ?? 0) as number) * 100),
  }));
  if (current) data.push({ attempt: `#${data.length + 1}`, score: Math.round(current.score * 100) });
  if (data.length < 2) return null;
  return (
    <div className="flex-1 min-w-0">
      <div className="text-[10px] font-semibold text-gray-400 uppercase tracking-wide mb-1">Score by attempt</div>
      <ResponsiveContainer width="100%" height={64}>
        <LineChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 4 }}>
          <XAxis dataKey="attempt" tick={{ fontSize: 9 }} axisLine={false} tickLine={false} />
          <YAxis hide domain={[0, 100]} />
          <Tooltip formatter={(v) => [`${v}%`, "score"]} contentStyle={{ fontSize: 11 }} />
          <Line type="monotone" dataKey="score" stroke="#6366f1" strokeWidth={2}
                dot={{ r: 2.5, fill: "#6366f1" }} isAnimationActive />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function StepDurationBars({ steps }: {
  steps: Record<string, { action: string; duration_ms: number }>;
}) {
  const data = Object.values(steps)
    .filter((s) => s.duration_ms > 0)
    .map((s, i) => ({ name: `${i + 1} ${s.action}`, sec: +(s.duration_ms / 1000).toFixed(1) }));
  if (data.length < 2) return null;
  return (
    <div className="flex-1 min-w-0">
      <div className="text-[10px] font-semibold text-gray-400 uppercase tracking-wide mb-1">Step duration (s)</div>
      <ResponsiveContainer width="100%" height={64}>
        <BarChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 4 }}>
          <XAxis dataKey="name" tick={{ fontSize: 8 }} axisLine={false} tickLine={false} interval={0} />
          <YAxis hide />
          <Tooltip formatter={(v) => [`${v}s`, "duration"]} contentStyle={{ fontSize: 11 }} />
          <Bar dataKey="sec" fill="#93c5fd" radius={[3, 3, 0, 0]} isAnimationActive />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
