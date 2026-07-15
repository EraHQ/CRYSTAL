// Cognition — the Evidence Bench (ratified 2026-07-15, merged design).
//
// Three panes: run rail (search + status filters + critique badges) →
// anatomy tree (the run's parts, chronological, live) → reading pane
// (ONE node at full fidelity — the zero-truncation rule; the pane is
// dedicated, so nothing is ever clipped). Every node carries
// [ Detail | Critiques ]; critiques pin to target paths and feed the
// orchestrator on the next attempt / next run of the trigger (Q2B).
// The live machinery (pipeline strip, events, count-ups) lives in the
// selected run's header.
import { useMemo, useState } from "react";
import { authedFetch } from "@/lib/api";
import { useQuery } from "@tanstack/react-query";
import { useSelectedCustomer } from "@/lib/selected-customer";
import {
  Activity, Brain, CheckCircle2, ChevronDown, ChevronRight,
  DollarSign, FileText, Maximize2, MessageSquare, Minimize2,
  Scale, ScrollText, Search, Wrench, XCircle, Zap,
} from "lucide-react";
import { PipelineStrip } from "@/components/cognition/PipelineStrip";
import {
  AmendmentsPanel, CogEvent, CountUp, EventsFeed, InfeasibleFlag,
  LiveTimer, ProvenanceBadges, RouteBadge, ScoreSparkline,
  StepDurationBars,
} from "@/components/cognition/widgets";
import { CritiquePanel } from "@/components/cognition/CritiquePanel";
import {
  Critique, fetchCritiques, openCount,
} from "@/components/cognition/critiques";

// ---------------------------------------------------------------- types

interface EnvironmentSummary {
  id: string;
  customer_id: string;
  status: string;
  trigger_type: string;
  goal_title: string;
  attempts: number;
  max_attempts: number;
  step_count: number;
  steps_complete: number;
  cost_usd: number;
  tokens_used: number;
  created_at: string;
  open_critiques?: number;
}

interface CriterionEval {
  criterion: string;
  status: string;
  evidence: string;
  possibly_infeasible?: boolean;
}

interface EnvironmentDetail {
  id: string;
  customer_id: string;
  status: string;
  trigger_type: string;
  trigger_id?: string;
  attempts: number;
  max_attempts: number;
  cost_usd: number;
  tokens_used: number;
  created_at: string;
  goal: {
    title: string;
    description: string;
    acceptance_criteria: string[];
    amendments?: Array<{
      attempt: number; index: number; original: string;
      amended: string; evidence: string;
    }>;
  } | null;
  plan: {
    reasoning: string;
    steps: Array<{
      id: number;
      action: string;
      description: string;
      parallel_group: string | null;
      model: string;
    }>;
    suggested_key: string;
    retry_route?: string;
  } | null;
  steps: Record<string, {
    step_id: number;
    action: string;
    status: string;
    model_used: string;
    duration_ms: number;
    output: any;
    error: string | null;
  }>;
  deliverables: Record<string, string>;
  validation: {
    approved: boolean;
    score: number;
    reasoning: string;
    criteria_evaluation: CriterionEval[];
    issues: string[];
  } | null;
  rejection_log: Array<{ attempt: number; reasoning: string; score: number }>;
  events?: CogEvent[];
  attempt_history?: Array<{
    attempt: number;
    plan: { reasoning?: string; retry_route?: string; steps?: Array<{ id: number; action: string; description?: string }> } | null;
    steps: Record<string, { step_id?: number; action: string; status: string; duration_ms?: number; model_used?: string; output?: any; error?: string | null }>;
    deliverable: string;
    validation: { approved: boolean; score: number; reasoning?: string; issues?: string[] };
  }>;
}

// ------------------------------------------------------------------ api

async function fetchEnvironments(customerId: string): Promise<{ total: number; environments: EnvironmentSummary[] }> {
  const res = await authedFetch(`/admin/api/cognition/environments?customer_id=${encodeURIComponent(customerId)}`);
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}

async function fetchEnvironmentDetail(envId: string): Promise<EnvironmentDetail> {
  const res = await authedFetch(`/admin/api/cognition/environments/${encodeURIComponent(envId)}`);
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}

const ACTIVE = ["orchestrating", "working", "validating", "rejected"];
const isActiveStatus = (s: string) => ACTIVE.includes(s);

// ------------------------------------------------------------- run rail

const STATUS_DOT: Record<string, string> = {
  orchestrating: "bg-blue-400 animate-pulse",
  working: "bg-amber-400 animate-pulse",
  validating: "bg-purple-400 animate-pulse",
  rejected: "bg-red-400 animate-pulse",
  complete: "bg-green-500",
  failed: "bg-red-600",
  needs_human_review: "bg-orange-500",
};

const FILTERS: Array<{ key: string; label: string; match: (s: string) => boolean }> = [
  { key: "live", label: "live", match: isActiveStatus },
  { key: "complete", label: "done", match: (s) => s === "complete" },
  { key: "failed", label: "failed", match: (s) => s === "failed" },
  { key: "review", label: "review", match: (s) => s === "needs_human_review" },
];

function RunRail({ envs, selectedId, onSelect }: {
  envs: EnvironmentSummary[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const [q, setQ] = useState("");
  const [filter, setFilter] = useState<string | null>(null);

  const filtered = useMemo(() => {
    let out = envs;
    if (filter) {
      const f = FILTERS.find((x) => x.key === filter);
      if (f) out = out.filter((e) => f.match(e.status));
    }
    const needle = q.trim().toLowerCase();
    if (needle) {
      out = out.filter((e) =>
        (e.goal_title || "").toLowerCase().includes(needle) ||
        (e.trigger_type || "").toLowerCase().includes(needle) ||
        e.id.toLowerCase().includes(needle));
    }
    // Live runs pinned on top, then newest first (list arrives newest-first).
    return [...out].sort((a, b) =>
      Number(isActiveStatus(b.status)) - Number(isActiveStatus(a.status)));
  }, [envs, q, filter]);

  return (
    <div className="flex flex-col min-h-0">
      <div className="relative mb-2">
        <Search className="h-3.5 w-3.5 text-gray-300 absolute left-2 top-1/2 -translate-y-1/2" />
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search runs"
          className="w-full text-xs border border-gray-200 rounded pl-7 pr-2 py-1.5 outline-none focus:border-indigo-300"
        />
      </div>
      <div className="flex gap-1 mb-2 flex-wrap">
        {FILTERS.map((f) => (
          <button key={f.key}
            onClick={() => setFilter(filter === f.key ? null : f.key)}
            className={`text-[10px] px-1.5 py-0.5 rounded border ${
              filter === f.key
                ? "bg-indigo-50 border-indigo-300 text-indigo-700"
                : "border-gray-200 text-gray-500 hover:border-gray-300"}`}>
            {f.label}
          </button>
        ))}
      </div>
      <div className="flex-1 overflow-y-auto space-y-1 pr-1 min-h-0">
        {filtered.map((e) => (
          <button key={e.id}
            onClick={() => onSelect(e.id)}
            className={`w-full text-left rounded border px-2.5 py-2 transition-colors ${
              selectedId === e.id
                ? "border-indigo-300 bg-indigo-50"
                : "border-gray-200 hover:bg-gray-50"}`}>
            <div className="flex items-center gap-1.5">
              <span className={`inline-block w-1.5 h-1.5 rounded-full flex-shrink-0 ${STATUS_DOT[e.status] ?? "bg-gray-300"}`} />
              <span className="text-xs font-medium text-gray-800 truncate flex-1">
                {e.goal_title || e.trigger_type || e.id.slice(0, 12)}
              </span>
              {(e.open_critiques ?? 0) > 0 && (
                <span className="inline-flex items-center gap-0.5 text-[9px] text-amber-700 bg-amber-50 rounded px-1">
                  <MessageSquare className="h-2.5 w-2.5" />{e.open_critiques}
                </span>
              )}
            </div>
            <div className="flex items-center gap-2 mt-0.5 text-[10px] text-gray-400">
              <span>{e.status.replace(/_/g, " ")}</span>
              <span>{e.steps_complete}/{e.step_count}</span>
              <span className="flex items-center gap-0.5">
                <DollarSign className="h-2.5 w-2.5" />{e.cost_usd.toFixed(3)}
              </span>
            </div>
          </button>
        ))}
        {!filtered.length && (
          <p className="text-xs text-gray-400 px-1 py-3">No runs match.</p>
        )}
      </div>
    </div>
  );
}

// -------------------------------------------------------- anatomy tree

type NodePath = string; // "activity" | "contract" | "criterion:N" | "execution" | "step:N" | "verdict" | "attempt:N" | "deliverable" | "critiques"

interface TreeRowProps {
  path: NodePath;
  depth: number;
  label: string;
  icon?: React.ReactNode;
  meta?: React.ReactNode;
  selected: NodePath;
  onSelect: (p: NodePath) => void;
  critiques: Critique[];
}

function TreeRow({ path, depth, label, icon, meta, selected, onSelect, critiques }: TreeRowProps) {
  const n = openCount(critiques, path);
  return (
    <button
      onClick={() => onSelect(path)}
      className={`w-full flex items-center gap-1.5 text-left rounded px-2 py-1 text-xs ${
        selected === path
          ? "bg-indigo-50 text-indigo-800"
          : "text-gray-700 hover:bg-gray-50"}`}
      style={{ paddingLeft: `${8 + depth * 14}px` }}>
      {icon}
      <span className="truncate flex-1">{label}</span>
      {n > 0 && (
        <span className="inline-flex items-center gap-0.5 text-[9px] text-amber-700 bg-amber-50 rounded px-1">
          <MessageSquare className="h-2.5 w-2.5" />{n}
        </span>
      )}
      {meta}
    </button>
  );
}

function stepStateIcon(status: string) {
  if (status === "complete") return <CheckCircle2 className="h-3 w-3 text-green-500 flex-shrink-0" />;
  if (status === "failed") return <XCircle className="h-3 w-3 text-red-400 flex-shrink-0" />;
  if (status === "running") return <span className="inline-block w-2 h-2 rounded-full bg-blue-400 animate-pulse flex-shrink-0" />;
  return <span className="inline-block w-2 h-2 rounded-full bg-gray-200 flex-shrink-0" />;
}

function AnatomyTree({ detail, selected, onSelect, critiques }: {
  detail: EnvironmentDetail;
  selected: NodePath;
  onSelect: (p: NodePath) => void;
  critiques: Critique[];
}) {
  const [showAttempts, setShowAttempts] = useState(true);
  const steps = detail.plan?.steps ?? [];
  const common = { selected, onSelect, critiques };

  return (
    <div className="space-y-0.5 overflow-y-auto min-h-0 pr-1">
      {(detail.events?.length ?? 0) > 0 && (
        <TreeRow {...common} path="activity" depth={0} label="Activity"
          icon={<Activity className="h-3 w-3 text-blue-400" />}
          meta={isActiveStatus(detail.status)
            ? <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" /> : undefined} />
      )}

      <TreeRow {...common} path="contract" depth={0} label="Contract"
        icon={<ScrollText className="h-3 w-3 text-gray-400" />} />
      {(detail.goal?.acceptance_criteria ?? []).map((c, i) => (
        <TreeRow key={i} {...common} path={`criterion:${i}`} depth={1}
          label={`${i + 1}. ${c}`}
          meta={detail.validation?.criteria_evaluation?.[i]?.possibly_infeasible
            ? <span className="text-[9px] text-orange-600">⚠</span> : undefined} />
      ))}

      <TreeRow {...common} path="execution" depth={0} label="Execution"
        icon={<Wrench className="h-3 w-3 text-gray-400" />} />
      {steps.map((ps) => {
        const sr = detail.steps[String(ps.id)];
        return (
          <TreeRow key={ps.id} {...common} path={`step:${ps.id}`} depth={1}
            label={`${ps.id}. ${ps.action}`}
            icon={stepStateIcon(sr?.status ?? "pending")}
            meta={sr?.duration_ms
              ? <span className="text-[9px] text-gray-300">{(sr.duration_ms / 1000).toFixed(1)}s</span>
              : undefined} />
        );
      })}

      <TreeRow {...common} path="verdict" depth={0} label="Verdict"
        icon={<Scale className="h-3 w-3 text-gray-400" />}
        meta={detail.validation
          ? <span className={`text-[9px] px-1 rounded ${detail.validation.approved ? "bg-green-50 text-green-700" : "bg-red-50 text-red-700"}`}>
              {(detail.validation.score * 100).toFixed(0)}%
            </span>
          : undefined} />
      {(detail.attempt_history?.length ?? 0) > 0 && (
        <>
          <button onClick={() => setShowAttempts(!showAttempts)}
            className="w-full flex items-center gap-1 text-left rounded px-2 py-1 text-[10px] text-gray-400 hover:bg-gray-50"
            style={{ paddingLeft: "22px" }}>
            {showAttempts ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
            {detail.attempt_history!.length} archived attempt{detail.attempt_history!.length === 1 ? "" : "s"}
          </button>
          {showAttempts && detail.attempt_history!.map((a) => (
            <TreeRow key={a.attempt} {...common} path={`attempt:${a.attempt}`} depth={2}
              label={`Attempt ${a.attempt}`}
              meta={<span className={`text-[9px] px-1 rounded ${a.validation?.approved ? "bg-green-50 text-green-700" : "bg-red-50 text-red-700"}`}>
                {((a.validation?.score ?? 0) * 100).toFixed(0)}%
              </span>} />
          ))}
        </>
      )}

      {Object.keys(detail.deliverables).length > 0 && (
        <TreeRow {...common} path="deliverable" depth={0} label="Deliverable"
          icon={<FileText className="h-3 w-3 text-gray-400" />} />
      )}

      {critiques.length > 0 && (
        <TreeRow {...common} path="critiques" depth={0}
          label={`All critiques (${critiques.length})`}
          icon={<MessageSquare className="h-3 w-3 text-amber-500" />} />
      )}
    </div>
  );
}

// ------------------------------------------------------- node renderers

function SectionLabel({ children }: { children: React.ReactNode }) {
  return <div className="text-[10px] font-semibold text-gray-400 uppercase tracking-wide mb-1">{children}</div>;
}

function ContractNode({ detail }: { detail: EnvironmentDetail }) {
  const g = detail.goal;
  if (!g) return <p className="text-xs text-gray-400">No goal contract yet.</p>;
  return (
    <div className="space-y-4">
      <div>
        <SectionLabel>Goal</SectionLabel>
        <p className="text-sm text-gray-800">{g.description}</p>
      </div>
      <div>
        <SectionLabel>Acceptance criteria</SectionLabel>
        <ol className="space-y-1.5">
          {g.acceptance_criteria.map((c, i) => (
            <li key={i} className="text-xs text-gray-700 flex gap-2">
              <span className="text-gray-400">{i + 1}.</span>
              <span>{c}
                {detail.validation?.criteria_evaluation?.[i]?.possibly_infeasible && <InfeasibleFlag />}
              </span>
            </li>
          ))}
        </ol>
      </div>
      {(g.amendments?.length ?? 0) > 0 && <AmendmentsPanel amendments={g.amendments!} />}
      {detail.plan?.reasoning && (
        <div>
          <SectionLabel>Plan</SectionLabel>
          <div className="flex items-center gap-2 mb-1">
            <RouteBadge route={detail.plan.retry_route} />
            {detail.plan.suggested_key && (
              <code className="text-[10px] bg-gray-100 px-1 py-0.5 rounded text-gray-600">{detail.plan.suggested_key}</code>
            )}
          </div>
          <p className="text-xs text-gray-600">{detail.plan.reasoning}</p>
        </div>
      )}
    </div>
  );
}

function CriterionNode({ detail, index }: { detail: EnvironmentDetail; index: number }) {
  const text = detail.goal?.acceptance_criteria?.[index];
  const ev = detail.validation?.criteria_evaluation?.[index];
  const amendments = (detail.goal?.amendments ?? []).filter((a) => a.index === index);
  return (
    <div className="space-y-4">
      <div>
        <SectionLabel>Criterion {index + 1}</SectionLabel>
        <p className="text-sm text-gray-800">{text}
          {ev?.possibly_infeasible && <InfeasibleFlag />}
        </p>
      </div>
      {ev && (
        <div>
          <SectionLabel>Latest evaluation</SectionLabel>
          <span className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-medium ${
            ev.status === "MET" ? "bg-green-100 text-green-700" :
            ev.status === "PARTIALLY_MET" ? "bg-yellow-100 text-yellow-700" :
            "bg-red-100 text-red-700"}`}>{ev.status}</span>
          {ev.evidence && <p className="text-xs text-gray-600 mt-1.5 whitespace-pre-wrap">{ev.evidence}</p>}
        </div>
      )}
      {amendments.length > 0 && <AmendmentsPanel amendments={amendments} />}
    </div>
  );
}

function ExecutionNode({ detail }: { detail: EnvironmentDetail }) {
  return (
    <div className="space-y-4">
      <div className="flex gap-4">
        <ScoreSparkline attempts={detail.attempt_history ?? []} current={detail.validation} />
        <StepDurationBars steps={detail.steps as any} />
      </div>
      <div>
        <SectionLabel>Steps</SectionLabel>
        <div className="space-y-1">
          {(detail.plan?.steps ?? []).map((ps) => {
            const sr = detail.steps[String(ps.id)];
            return (
              <div key={ps.id} className="flex items-center gap-2 text-xs">
                {stepStateIcon(sr?.status ?? "pending")}
                <span className="font-mono text-gray-700">{ps.action}</span>
                <span className="text-gray-400 truncate flex-1">{ps.description}</span>
                {sr?.duration_ms ? <span className="text-gray-300">{(sr.duration_ms / 1000).toFixed(1)}s</span> : null}
              </div>
            );
          })}
        </div>
        <p className="text-[10px] text-gray-400 mt-2">Select a step in the tree for its full trace.</p>
      </div>
    </div>
  );
}

function StepTraceBody({ sr }: { sr: { output?: any; error?: string | null } | undefined }) {
  const [openOutput, setOpenOutput] = useState(false);
  const out = sr?.output ?? {};
  const toolCalls: any[] = Array.isArray(out.tool_calls) ? out.tool_calls : [];
  const findings: any[] = Array.isArray(out.findings) ? out.findings : [];
  return (
    <div className="space-y-4">
      {sr?.error && (
        <div className="text-xs text-red-700 bg-red-50 border border-red-100 rounded p-2 whitespace-pre-wrap">{sr.error}</div>
      )}
      {toolCalls.length > 0 && (
        <div>
          <SectionLabel>Tool calls ({toolCalls.length})</SectionLabel>
          <div className="space-y-2">
            {toolCalls.map((c, i) => (
              <div key={i} className="border border-indigo-100 rounded p-2.5">
                <div className="flex items-center gap-2 text-xs mb-1">
                  <span className="text-[9px] font-semibold text-indigo-400 bg-indigo-50 rounded px-1">{c.iteration ?? i + 1}</span>
                  <span className="font-mono text-gray-800">{c.tool}</span>
                </div>
                {c.input && (
                  <pre className="text-[10px] text-gray-600 bg-gray-50 rounded p-1.5 whitespace-pre-wrap break-words mb-1">{JSON.stringify(c.input, null, 1)}</pre>
                )}
                {c.output_head && (
                  <pre className="text-[10px] text-gray-700 bg-white border border-gray-100 rounded p-1.5 whitespace-pre-wrap break-words max-h-64 overflow-y-auto">{c.output_head}</pre>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {findings.length > 0 && (
        <div>
          <SectionLabel>Findings ({findings.length})</SectionLabel>
          <div className="space-y-1.5">
            {findings.map((f, i) => (
              <FindingBlock key={i} f={f} />
            ))}
          </div>
        </div>
      )}

      {(out.content || out.content_text) && (
        <div>
          <button onClick={() => setOpenOutput(!openOutput)}
            className="flex items-center gap-1 text-[11px] text-gray-500 hover:text-gray-700 mb-1">
            {openOutput ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
            Raw output
          </button>
          {openOutput && (
            <pre className="text-[11px] text-gray-700 bg-gray-50 border border-gray-100 rounded p-3 whitespace-pre-wrap break-words">{String(out.content ?? out.content_text)}</pre>
          )}
        </div>
      )}
      {out.results_count !== undefined && (
        <p className="text-xs text-gray-500">Found {out.results_count} results</p>
      )}
    </div>
  );
}

function StepNode({ detail, stepId }: { detail: EnvironmentDetail; stepId: number }) {
  const ps = (detail.plan?.steps ?? []).find((s) => s.id === stepId);
  const sr = detail.steps[String(stepId)];
  const out = sr?.output ?? {};
  const stepEvents = (detail.events ?? []).filter((e) => e.step_id === stepId);
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 flex-wrap">
        {stepStateIcon(sr?.status ?? "pending")}
        <span className="text-sm font-medium text-gray-800">{ps?.action ?? "step"} · step {stepId}</span>
        {out.agentic && (
          <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-indigo-50 text-indigo-700">
            agentic · {out.iterations ?? "?"} iterations
          </span>
        )}
        {sr?.model_used && sr.model_used !== "none" && (
          <span className="px-1.5 py-0.5 rounded text-[10px] bg-purple-50 text-purple-600">{sr.model_used}</span>
        )}
        {sr?.duration_ms ? <span className="text-[10px] text-gray-400">{(sr.duration_ms / 1000).toFixed(1)}s</span> : null}
        <ProvenanceBadges output={out} />
      </div>
      {ps?.description && <p className="text-xs text-gray-500">{ps.description}</p>}
      {stepEvents.length > 0 && (
        <div>
          <SectionLabel>Step events</SectionLabel>
          <EventsFeed events={stepEvents} live={false} />
        </div>
      )}
      <StepTraceBody sr={sr} />
    </div>
  );
}

function ArchivedStep({ s: rec }: { s: any }) {
  const [open, setOpen] = useState(false);
  const out = rec?.output ?? {};
  const nTools = Array.isArray(out.tool_calls) ? out.tool_calls.length : 0;
  const nFindings = Array.isArray(out.findings) ? out.findings.length : 0;
  const summary = rec?.error
    ? String(rec.error).split("\n")[0]
    : [
        nTools ? `${nTools} tool call${nTools === 1 ? "" : "s"}` : null,
        nFindings ? `${nFindings} finding${nFindings === 1 ? "" : "s"}` : null,
        out.results_count !== undefined ? `${out.results_count} results` : null,
        (out.content || out.content_text) ? "composed output" : null,
      ].filter(Boolean).join(" · ") || "no recorded output";
  return (
    <div className="border border-gray-100 rounded-lg">
      <button onClick={() => setOpen(!open)} className="w-full flex items-center gap-2 px-2.5 py-2 text-left">
        {open ? <ChevronDown className="h-3 w-3 text-gray-400 flex-shrink-0" /> : <ChevronRight className="h-3 w-3 text-gray-400 flex-shrink-0" />}
        {rec.status === "complete"
          ? <CheckCircle2 className="h-3 w-3 text-green-500 flex-shrink-0" />
          : <XCircle className="h-3 w-3 text-red-400 flex-shrink-0" />}
        <span className="text-xs font-mono text-gray-700">{rec.action}</span>
        {rec.model_used && rec.model_used !== "none" && (
          <span className="px-1 rounded text-[9px] bg-purple-50 text-purple-600">{rec.model_used}</span>
        )}
        <span className={cnSummary(rec)}>{summary}</span>
        <span className="flex-1" />
        {rec.duration_ms != null && <span className="text-[10px] text-gray-300">{(rec.duration_ms / 1000).toFixed(1)}s</span>}
      </button>
      {open && (
        <div className="px-3 pb-3 pt-1 border-t border-gray-50">
          <StepTraceBody sr={rec} />
        </div>
      )}
    </div>
  );
}

function cnSummary(rec: any): string {
  return rec?.error
    ? "text-[11px] text-red-600 truncate"
    : "text-[11px] text-gray-400 truncate";
}

function FindingBlock({ f }: { f: any }) {
  const [open, setOpen] = useState(false);
  const title = f?.title || f?.url || "finding";
  return (
    <div className="border border-gray-100 rounded p-2">
      <button onClick={() => setOpen(!open)} className="w-full flex items-center gap-1.5 text-left">
        {open ? <ChevronDown className="h-3 w-3 text-gray-400" /> : <ChevronRight className="h-3 w-3 text-gray-400" />}
        <span className="text-xs text-gray-700 truncate flex-1">{title}</span>
        <ProvenanceBadges output={{ findings: [f] }} />
      </button>
      {open && (
        <div className="mt-1.5 pl-4">
          {f?.url && <p className="text-[10px] text-blue-600 break-all mb-1">{f.url}</p>}
          <pre className="text-[10px] text-gray-600 whitespace-pre-wrap break-words max-h-72 overflow-y-auto">{f?.content || f?.snippet || ""}</pre>
        </div>
      )}
    </div>
  );
}

function VerdictNode({ detail }: { detail: EnvironmentDetail }) {
  const v = detail.validation;
  return (
    <div className="space-y-4">
      {v ? (
        <>
          <div className={`rounded border p-3 ${v.approved ? "border-green-200 bg-green-50" : "border-red-200 bg-red-50"}`}>
            <div className="flex items-center gap-2 mb-1">
              {v.approved ? <CheckCircle2 className="h-4 w-4 text-green-600" /> : <XCircle className="h-4 w-4 text-red-600" />}
              <span className={`text-sm font-medium ${v.approved ? "text-green-800" : "text-red-800"}`}>
                {v.approved ? "Approved" : "Rejected"}
              </span>
              <span className={`text-xs px-1.5 py-0.5 rounded ${v.approved ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"}`}>
                {(v.score * 100).toFixed(0)}%
              </span>
            </div>
            <p className="text-xs text-gray-700 whitespace-pre-wrap">{v.reasoning}</p>
          </div>
          <div>
            <SectionLabel>Criteria</SectionLabel>
            <div className="space-y-2">
              {v.criteria_evaluation.map((c, i) => (
                <div key={i} className="flex items-start gap-2 text-xs">
                  <span className={`inline-flex px-1.5 py-0.5 rounded text-[10px] font-medium min-w-[86px] justify-center ${
                    c.status === "MET" ? "bg-green-100 text-green-700" :
                    c.status === "PARTIALLY_MET" ? "bg-yellow-100 text-yellow-700" :
                    "bg-red-100 text-red-700"}`}>{c.status}</span>
                  <div>
                    <span className="text-gray-700">{c.criterion}</span>
                    {c.possibly_infeasible && <InfeasibleFlag />}
                    {c.evidence && <p className="text-gray-500 mt-0.5 whitespace-pre-wrap">{c.evidence}</p>}
                  </div>
                </div>
              ))}
            </div>
          </div>
          {v.issues.length > 0 && (
            <div>
              <SectionLabel>Issues</SectionLabel>
              <ul className="space-y-0.5">
                {v.issues.map((iss, i) => (
                  <li key={i} className="text-xs text-red-600">• {iss}</li>
                ))}
              </ul>
            </div>
          )}
        </>
      ) : (
        <p className="text-xs text-gray-400">No validation yet.</p>
      )}
      <div className="flex gap-4">
        <ScoreSparkline attempts={detail.attempt_history ?? []} current={v} />
      </div>
    </div>
  );
}

function AttemptNode({ detail, attempt }: { detail: EnvironmentDetail; attempt: number }) {
  const a = (detail.attempt_history ?? []).find((x) => x.attempt === attempt);
  if (!a) return <p className="text-xs text-gray-400">Attempt not found.</p>;
  const steps = Object.values(a.steps ?? {});
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <span className="text-sm font-medium text-gray-800">Attempt {a.attempt}</span>
        <RouteBadge route={a.plan?.retry_route} />
        <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${a.validation?.approved ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"}`}>
          score {((a.validation?.score ?? 0) * 100).toFixed(0)}%
        </span>
      </div>
      {a.plan?.reasoning && <p className="text-xs text-gray-600">{a.plan.reasoning}</p>}
      <div>
        <SectionLabel>Execution — full trace per step</SectionLabel>
        <div className="space-y-1.5">
          {steps.map((s, i) => (
            <ArchivedStep key={i} s={s} />
          ))}
        </div>
      </div>
      {a.deliverable && (
        <div>
          <SectionLabel>Deliverable</SectionLabel>
          <pre className="text-[11px] text-gray-700 bg-gray-50 border border-gray-100 rounded p-3 whitespace-pre-wrap break-words max-h-96 overflow-y-auto">{a.deliverable}</pre>
        </div>
      )}
      {a.validation?.reasoning && (
        <div>
          <SectionLabel>Verdict</SectionLabel>
          <p className="text-xs text-gray-600 whitespace-pre-wrap">{a.validation.reasoning}</p>
          {(a.validation.issues ?? []).map((iss, i) => (
            <p key={i} className="text-xs text-red-600 mt-0.5">• {iss}</p>
          ))}
        </div>
      )}
    </div>
  );
}

function DeliverableNode({ detail, fullscreen, setFullscreen }: {
  detail: EnvironmentDetail;
  fullscreen: boolean;
  setFullscreen: (v: boolean) => void;
}) {
  const entries = Object.entries(detail.deliverables);
  if (!entries.length) return <p className="text-xs text-gray-400">No deliverable yet.</p>;
  return (
    <div className="space-y-6">
      {entries.map(([name, content]) => (
        <div key={name}>
          <div className="flex items-center justify-between mb-2">
            <SectionLabel>{name}</SectionLabel>
            <div className="flex items-center gap-3">
              <button
                onClick={() => navigator.clipboard?.writeText(content)}
                className="text-[10px] text-gray-400 hover:text-gray-600">copy</button>
              <button onClick={() => setFullscreen(!fullscreen)}
                className="text-gray-400 hover:text-gray-600">
                {fullscreen ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
              </button>
            </div>
          </div>
          {/* The document reader: full contents, reading width, no box,
              no truncation — the deliverable is the product of the run
              and reads like one. */}
          <div className="max-w-[70ch] text-[13px] leading-relaxed text-gray-800 whitespace-pre-wrap break-words">
            {content}
          </div>
        </div>
      ))}
    </div>
  );
}

function AllCritiquesNode({ critiques, onSelect }: {
  critiques: Critique[];
  onSelect: (p: NodePath) => void;
}) {
  if (!critiques.length) return <p className="text-xs text-gray-400">No critiques on this run.</p>;
  return (
    <div className="space-y-2">
      {critiques.map((c) => (
        <button key={c.id}
          onClick={() => onSelect(pathToNode(c.target_path))}
          className={`w-full text-left rounded border p-2.5 text-xs ${
            c.status === "open" ? "border-amber-200 bg-amber-50" : "border-gray-100 bg-gray-50 opacity-70"}`}>
          <div className="flex items-center gap-2 mb-0.5">
            <code className="text-[10px] bg-white/70 border border-gray-200 rounded px-1 text-gray-600">{c.target_path}</code>
            <span className="text-gray-400">{c.author}</span>
            <span className="flex-1" />
            <span className={`text-[9px] px-1 rounded ${c.status === "open" ? "bg-amber-100 text-amber-700" : "bg-gray-100 text-gray-500"}`}>{c.status}</span>
          </div>
          <p className="text-gray-700">{c.text}</p>
        </button>
      ))}
    </div>
  );
}

/** A critique's target_path → the tree node that owns it. */
function pathToNode(target: string): NodePath {
  const root = target.split("/")[0];
  if (root.startsWith("step:") || root.startsWith("criterion:") || root.startsWith("attempt:")) return root;
  if (["run", "contract", "execution", "verdict", "deliverable", "activity"].includes(root)) {
    return root === "run" ? "contract" : root;
  }
  return "contract";
}

// --------------------------------------------------------- reading pane

function ReadingPane({ detail, node, critiques, onSelect }: {
  detail: EnvironmentDetail;
  node: NodePath;
  critiques: Critique[];
  onSelect: (p: NodePath) => void;
}) {
  const [tab, setTab] = useState<"detail" | "critiques">("detail");
  const [fullscreen, setFullscreen] = useState(false);
  const active = isActiveStatus(detail.status);

  // The critique target for this node: the node path itself ("contract"
  // critiques pin to "run" — the whole-run target).
  const targetPath = node === "contract" ? "run" : node;
  const n = openCount(critiques, node === "contract" ? "run" : node);

  const body = (() => {
    if (tab === "critiques") {
      return <CritiquePanel envId={detail.id} targetPath={targetPath} critiques={critiques} />;
    }
    if (node === "activity") return <EventsFeed events={detail.events ?? []} live={active} />;
    if (node === "contract") return <ContractNode detail={detail} />;
    if (node.startsWith("criterion:")) return <CriterionNode detail={detail} index={Number(node.split(":")[1])} />;
    if (node === "execution") return <ExecutionNode detail={detail} />;
    if (node.startsWith("step:")) return <StepNode detail={detail} stepId={Number(node.split(":")[1])} />;
    if (node === "verdict") return <VerdictNode detail={detail} />;
    if (node.startsWith("attempt:")) return <AttemptNode detail={detail} attempt={Number(node.split(":")[1])} />;
    if (node === "deliverable") return <DeliverableNode detail={detail} fullscreen={fullscreen} setFullscreen={setFullscreen} />;
    if (node === "critiques") return <AllCritiquesNode critiques={critiques} onSelect={onSelect} />;
    return <ContractNode detail={detail} />;
  })();

  const pane = (
    <div className={fullscreen
      ? "fixed inset-0 z-50 bg-white p-8 overflow-y-auto"
      : "flex-1 min-w-0 overflow-y-auto"}>
      {node !== "critiques" && (
        <div className="flex items-center gap-3 border-b border-gray-100 mb-4">
          {(["detail", "critiques"] as const).map((t) => (
            <button key={t} onClick={() => setTab(t)}
              className={`text-xs pb-2 -mb-px border-b-2 ${
                tab === t ? "border-indigo-500 text-indigo-700 font-medium" : "border-transparent text-gray-400 hover:text-gray-600"}`}>
              {t === "detail" ? "Detail" : `Critiques${n ? ` (${n})` : ""}`}
            </button>
          ))}
          {fullscreen && (
            <button onClick={() => setFullscreen(false)} className="ml-auto text-gray-400 hover:text-gray-600">
              <Minimize2 className="h-4 w-4" />
            </button>
          )}
        </div>
      )}
      {body}
    </div>
  );
  return pane;
}

// --------------------------------------------------------------- page

export function CognitionTracker() {
  const { selectedCustomerId } = useSelectedCustomer();
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [node, setNode] = useState<NodePath>("contract");

  const envQuery = useQuery({
    queryKey: ["cognition-environments", selectedCustomerId],
    queryFn: () => fetchEnvironments(selectedCustomerId!),
    enabled: !!selectedCustomerId,
    refetchInterval: 3000,
  });
  const envs = envQuery.data?.environments ?? [];
  const runId = selectedRun ?? envs[0]?.id ?? null;
  const summary = envs.find((e) => e.id === runId);
  const active = summary ? isActiveStatus(summary.status) : false;

  const detailQuery = useQuery({
    queryKey: ["cognition-env-detail", runId],
    queryFn: () => fetchEnvironmentDetail(runId!),
    enabled: !!runId,
    refetchInterval: active ? 2000 : false,
  });

  const critiquesQuery = useQuery({
    queryKey: ["run-critiques", runId],
    queryFn: () => fetchCritiques(runId!),
    enabled: !!runId,
  });
  const critiques = critiquesQuery.data ?? [];

  const activeCount = envs.filter((e) => isActiveStatus(e.status)).length;

  if (!envs.length) {
    return (
      <div className="bg-white border border-gray-200 rounded-lg p-5">
        <div className="flex items-center gap-2 mb-2">
          <Activity className="h-4 w-4 text-gray-400" />
          <h3 className="text-sm font-semibold text-gray-900">Cognition</h3>
        </div>
        <div className="flex flex-col items-center justify-center py-8 text-center">
          <Brain className="h-8 w-8 text-gray-300 mb-2" />
          <p className="text-sm font-medium text-gray-500">No cognition runs yet</p>
          <p className="text-xs text-gray-400 mt-1">
            Runs appear here when research tasks or knowledge gaps are processed — and completed runs stay visible.
          </p>
        </div>
      </div>
    );
  }

  const detail = detailQuery.data;

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-5">
      <div className="flex items-center gap-2 mb-4">
        <Activity className="h-4 w-4 text-gray-500" />
        <h3 className="text-sm font-semibold text-gray-900">Cognition</h3>
        <span className="text-xs text-gray-400">({envs.length})</span>
        {activeCount > 0 && (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-blue-50 text-blue-700">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
            {activeCount} active
          </span>
        )}
      </div>

      <div className="flex gap-4" style={{ height: "calc(100vh - 220px)", minHeight: 420 }}>
        <div className="w-56 flex-shrink-0 flex flex-col min-h-0">
          <RunRail envs={envs} selectedId={runId}
            onSelect={(id) => { setSelectedRun(id); setNode("contract"); }} />
        </div>

        <div className="flex-1 min-w-0 flex flex-col min-h-0">
          {detail ? (
            <>
              {/* Run header: the live machinery survives the redesign. */}
              <div className="border border-gray-200 rounded-lg px-4 py-2.5 mb-3">
                <div className="flex items-center gap-3 text-xs text-gray-500 mb-1">
                  <span className="text-sm font-medium text-gray-900 truncate">
                    {detail.goal?.title || summary?.goal_title || detail.trigger_type}
                  </span>
                  <span>attempt {detail.attempts}/{detail.max_attempts}</span>
                  <span className="flex items-center gap-0.5">
                    <DollarSign className="h-3 w-3" />
                    <CountUp value={summary?.cost_usd ?? detail.cost_usd ?? 0} decimals={4} />
                  </span>
                  <span><CountUp value={summary?.tokens_used ?? detail.tokens_used ?? 0} /> tokens</span>
                  {active && summary && <LiveTimer since={summary.created_at} />}
                </div>
                <PipelineStrip status={detail.status} steps={detail.steps} />
              </div>

              <div className="flex gap-4 flex-1 min-h-0">
                <div className="w-60 flex-shrink-0 border-r border-gray-100 pr-2 overflow-y-auto min-h-0">
                  <AnatomyTree detail={detail} selected={node} onSelect={setNode} critiques={critiques} />
                </div>
                <ReadingPane detail={detail} node={node} critiques={critiques} onSelect={setNode} />
              </div>
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <Zap className="h-5 w-5 text-gray-200 animate-pulse" />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
