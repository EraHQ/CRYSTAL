import { useState } from "react";
import { authedFetch } from "@/lib/api";
import { useQuery } from "@tanstack/react-query";
import { useSelectedCustomer } from "@/lib/selected-customer";
import {
  Brain, Play, CheckCircle2, XCircle,
  Search, FileText, Zap, ChevronDown, ChevronUp,
  RotateCcw, DollarSign, Activity,
} from "lucide-react";

// Types

interface StepInfo {
  action: string;
  status: string;
  duration_ms: number;
}

interface EnvironmentSummary {
  id: string;
  customer_id: string;
  status: string;
  trigger_type: string;
  goal_title: string;
  output_type: string;
  attempts: number;
  max_attempts: number;
  step_count: number;
  steps_complete: number;
  steps: Record<string, StepInfo>;
  validation: { approved: boolean; score: number } | null;
  tokens_used: number;
  cost_usd: number;
  created_at: string;
}

interface CriterionEval {
  criterion: string;
  status: string;
  evidence: string;
}

interface EnvironmentDetail {
  id: string;
  status: string;
  customer_id: string;
  created_at: string;
  trigger_type: string;
  output_type: string;
  attempts: number;
  max_attempts: number;
  tokens_used: number;
  total_cost_usd: number;
  goal: {
    title: string;
    description: string;
    acceptance_criteria: string[];
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
  } | null;
  steps: Record<string, {
    step_id: number;
    action: string;
    status: string;
    output: any;
    error: string | null;
    tokens_in: number;
    tokens_out: number;
    model_used: string;
    duration_ms: number;
  }>;
  deliverables: Record<string, string>;
  validation: {
    approved: boolean;
    score: number;
    reasoning: string;
    criteria_evaluation: CriterionEval[];
    issues: string[];
    suggestions: string[];
  } | null;
  rejection_log: Array<{
    attempt: number;
    reasoning: string;
    issues: string[];
    suggestions: string[];
    score: number;
  }>;
  // 2026-07-09: full per-attempt archive (plan + steps + deliverable +
  // verdict), captured before the engine's retry hygiene clears state.
  attempt_history?: Array<{
    attempt: number;
    plan: { reasoning?: string; steps?: Array<{ id: number; action: string; description?: string }> } | null;
    steps: Record<string, { action: string; status: string; duration_ms?: number; error?: string | null }>;
    deliverable: string;
    validation: { approved: boolean; score: number; reasoning?: string; issues?: string[] };
  }>;
}

// 2026-07-09 — one rejected attempt's FULL flow: what was planned, what
// each step did, what deliverable came out, and why the validator said
// no. The engine used to wipe all of this on rejection; the archive
// exists precisely so this component can render it.
function AttemptFlow({ a }: {
  a: NonNullable<EnvironmentDetail["attempt_history"]>[number];
}) {
  const [open, setOpen] = useState(false);
  const steps = Object.values(a.steps ?? {});
  return (
    <div className="rounded-lg border border-gray-200 bg-white">
      <button
        className="w-full flex items-center justify-between px-3 py-2 text-left"
        onClick={() => setOpen(!open)}
      >
        <div className="flex items-center gap-2 text-xs">
          <RotateCcw className="h-3.5 w-3.5 text-gray-400" />
          <span className="font-medium text-gray-800">Attempt {a.attempt}</span>
          <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${a.validation?.approved ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"}`}>
            score {((a.validation?.score ?? 0) * 100).toFixed(0)}%
          </span>
          <span className="text-gray-400">{steps.length} steps</span>
        </div>
        {open ? <ChevronUp className="h-4 w-4 text-gray-400" /> : <ChevronDown className="h-4 w-4 text-gray-400" />}
      </button>
      {open && (
        <div className="border-t border-gray-100 px-3 py-2 space-y-3">
          {a.plan?.reasoning && (
            <div>
              <div className="text-[10px] font-semibold text-gray-400 uppercase tracking-wide mb-0.5">Plan</div>
              <p className="text-xs text-gray-600">{a.plan.reasoning}</p>
            </div>
          )}
          <div>
            <div className="text-[10px] font-semibold text-gray-400 uppercase tracking-wide mb-1">Execution</div>
            <div className="space-y-1">
              {steps.map((s, i) => (
                <div key={i} className="flex items-center gap-2 text-xs">
                  {s.status === "complete"
                    ? <CheckCircle2 className="h-3 w-3 text-green-500" />
                    : <XCircle className="h-3 w-3 text-red-400" />}
                  <span className="font-mono text-gray-700">{s.action}</span>
                  {s.duration_ms != null && <span className="text-gray-400">{s.duration_ms}ms</span>}
                  {s.error && <span className="text-red-500 truncate">{s.error}</span>}
                </div>
              ))}
            </div>
          </div>
          {a.deliverable && (
            <div>
              <div className="text-[10px] font-semibold text-gray-400 uppercase tracking-wide mb-0.5">Deliverable</div>
              <div className="bg-gray-50 border border-gray-100 rounded p-2 text-[11px] text-gray-700 max-h-32 overflow-y-auto whitespace-pre-wrap font-mono">
                {a.deliverable}
              </div>
            </div>
          )}
          {a.validation?.reasoning && (
            <div>
              <div className="text-[10px] font-semibold text-gray-400 uppercase tracking-wide mb-0.5">Verdict</div>
              <p className="text-xs text-gray-600">{a.validation.reasoning}</p>
              {(a.validation.issues ?? []).map((iss, i) => (
                <p key={i} className="text-xs text-red-600 mt-0.5">&bull; {iss}</p>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// API

async function fetchEnvironments(customerId: string): Promise<{ total: number; environments: EnvironmentSummary[] }> {
  // 2026-07-09: bare fetch carried no Authorization header — the accounts
  // guard 401'd this pane silently since Phase A while its error state
  // rendered as "no runs yet". Third dead-pane of the audit, same species.
  const res = await authedFetch(`/admin/api/cognition/environments?customer_id=${encodeURIComponent(customerId)}`);
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}

async function fetchEnvironmentDetail(envId: string): Promise<EnvironmentDetail> {
  const res = await authedFetch(`/admin/api/cognition/environments/${encodeURIComponent(envId)}`);
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}

// Sub-components

function StatusDot({ status }: { status: string }) {
  const colors: Record<string, string> = {
    created: "bg-gray-300",
    orchestrating: "bg-blue-400 animate-pulse",
    working: "bg-amber-400 animate-pulse",
    validating: "bg-purple-400 animate-pulse",
    complete: "bg-green-500",
    rejected: "bg-red-400 animate-pulse",
    failed: "bg-red-600",
    needs_human_review: "bg-orange-500",
    destroyed: "bg-gray-400",
    pending: "bg-gray-300",
    running: "bg-blue-400 animate-pulse",
  };
  return <span className={`inline-block w-2 h-2 rounded-full ${colors[status] ?? "bg-gray-300"}`} />;
}

function StatusLabel({ status }: { status: string }) {
  const styles: Record<string, string> = {
    created: "text-gray-500 bg-gray-50",
    orchestrating: "text-blue-700 bg-blue-50",
    working: "text-amber-700 bg-amber-50",
    validating: "text-purple-700 bg-purple-50",
    complete: "text-green-700 bg-green-50",
    rejected: "text-red-700 bg-red-50",
    failed: "text-red-800 bg-red-100",
    needs_human_review: "text-orange-700 bg-orange-50",
    pending: "text-gray-500 bg-gray-50",
    running: "text-blue-700 bg-blue-50",
  };
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium ${styles[status] ?? "text-gray-500 bg-gray-50"}`}>
      <StatusDot status={status} />
      {status.replace(/_/g, " ")}
    </span>
  );
}

function ActionIcon({ action }: { action: string }) {
  const icons: Record<string, typeof Search> = {
    crystal_search: Search,
    web_search: Search,
    analyze: Brain,
    synthesize: Zap,
    format: FileText,
  };
  const Icon = icons[action] ?? Activity;
  return <Icon className="h-3.5 w-3.5" />;
}

function StepTimeline({ plan, steps }: { plan: EnvironmentDetail["plan"]; steps: EnvironmentDetail["steps"] }) {
  if (!plan) return null;

  return (
    <div className="space-y-1">
      {plan.steps.map((planStep, idx) => {
        const stepResult = steps[String(planStep.id)];
        const status = stepResult?.status ?? "pending";
        const isLast = idx === plan.steps.length - 1;

        return (
          <div key={planStep.id} className="flex items-start gap-3">
            <div className="flex flex-col items-center">
              <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium border
                ${status === "complete" ? "bg-green-50 border-green-300 text-green-700" :
                  status === "running" ? "bg-blue-50 border-blue-300 text-blue-700 animate-pulse" :
                  status === "failed" ? "bg-red-50 border-red-300 text-red-700" :
                  "bg-gray-50 border-gray-200 text-gray-400"}`}
              >
                {status === "complete" ? <CheckCircle2 className="h-3.5 w-3.5" /> :
                 status === "running" ? <Play className="h-3 w-3" /> :
                 status === "failed" ? <XCircle className="h-3.5 w-3.5" /> :
                 planStep.id}
              </div>
              {!isLast && (
                <div className={`w-px h-6 ${status === "complete" ? "bg-green-200" : "bg-gray-200"}`} />
              )}
            </div>

            <div className="flex-1 min-w-0 pb-2">
              <div className="flex items-center gap-2">
                <ActionIcon action={planStep.action} />
                <span className="text-xs font-medium text-gray-700">{planStep.action}</span>
                {planStep.parallel_group && (
                  <span className="text-[10px] text-gray-400 bg-gray-50 px-1 rounded">parallel: {planStep.parallel_group}</span>
                )}
                {stepResult?.model_used && stepResult.model_used !== "none" && stepResult.model_used !== "none (tool call)" && (
                  <span className="text-[10px] text-purple-500 bg-purple-50 px-1 rounded">{stepResult.model_used}</span>
                )}
                {stepResult?.duration_ms ? (
                  <span className="text-[10px] text-gray-400">{stepResult.duration_ms}ms</span>
                ) : null}
              </div>
              <p className="text-xs text-gray-500 mt-0.5 truncate">{planStep.description}</p>

              {stepResult?.status === "complete" && stepResult.output?.content && (
                <div className="mt-1 text-xs text-gray-600 bg-gray-50 rounded p-1.5 max-h-16 overflow-hidden">
                  {stepResult.output.content.slice(0, 150)}...
                </div>
              )}
              {stepResult?.status === "complete" && stepResult.output?.results_count !== undefined && (
                <div className="mt-1 text-xs text-gray-500">
                  Found {stepResult.output.results_count} results
                </div>
              )}
              {stepResult?.error && (
                <div className="mt-1 text-xs text-red-600 bg-red-50 rounded p-1.5">
                  {stepResult.error}
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function ValidationPanel({ validation, rejectionLog }: {
  validation: EnvironmentDetail["validation"];
  rejectionLog: EnvironmentDetail["rejection_log"];
}) {
  const [expanded, setExpanded] = useState(false);

  if (!validation && !rejectionLog.length) return null;

  return (
    <div className="space-y-2">
      {validation && (
        <div className={`rounded-lg border p-3 ${validation.approved ? "border-green-200 bg-green-50" : "border-red-200 bg-red-50"}`}>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              {validation.approved ? (
                <CheckCircle2 className="h-4 w-4 text-green-600" />
              ) : (
                <XCircle className="h-4 w-4 text-red-600" />
              )}
              <span className={`text-sm font-medium ${validation.approved ? "text-green-800" : "text-red-800"}`}>
                {validation.approved ? "Approved" : "Rejected"}
              </span>
              <span className={`text-xs px-1.5 py-0.5 rounded ${validation.approved ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"}`}>
                {(validation.score * 100).toFixed(0)}%
              </span>
            </div>
            <button onClick={() => setExpanded(!expanded)} className="text-gray-400 hover:text-gray-600">
              {expanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </button>
          </div>
          <p className="text-xs text-gray-700">{validation.reasoning}</p>

          {expanded && (
            <div className="mt-3 space-y-2">
              {validation.criteria_evaluation.map((crit, i) => (
                <div key={i} className="flex items-start gap-2 text-xs">
                  <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium min-w-[80px] justify-center
                    ${crit.status === "MET" ? "bg-green-100 text-green-700" :
                      crit.status === "PARTIALLY_MET" ? "bg-yellow-100 text-yellow-700" :
                      "bg-red-100 text-red-700"}`}>
                    {crit.status}
                  </span>
                  <div>
                    <span className="text-gray-700">{crit.criterion}</span>
                    {crit.evidence && <p className="text-gray-500 mt-0.5">{crit.evidence}</p>}
                  </div>
                </div>
              ))}

              {validation.issues.length > 0 && (
                <div className="mt-2">
                  <span className="text-xs font-medium text-red-700">Issues:</span>
                  <ul className="mt-1 space-y-0.5">
                    {validation.issues.map((issue, i) => (
                      <li key={i} className="text-xs text-red-600 flex items-start gap-1">
                        <span className="mt-0.5">&bull;</span> {issue}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {rejectionLog.length > 0 && (
        <div className="space-y-1">
          {rejectionLog.map((entry, i) => (
            <div key={i} className="text-xs text-gray-500 bg-gray-50 rounded p-2 border border-gray-100">
              <div className="flex items-center gap-1.5">
                <RotateCcw className="h-3 w-3 text-gray-400" />
                <span className="font-medium">Attempt {entry.attempt}</span>
                <span className="text-gray-400">score: {(entry.score * 100).toFixed(0)}%</span>
              </div>
              <p className="mt-0.5 text-gray-600">{entry.reasoning.slice(0, 120)}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// Environment Card

function EnvironmentCard({ env: summary }: { env: EnvironmentSummary }) {
  const [expanded, setExpanded] = useState(false);
  const isActive = ["orchestrating", "working", "validating", "rejected"].includes(summary.status);

  const detail = useQuery({
    queryKey: ["cognition-env-detail", summary.id],
    queryFn: () => fetchEnvironmentDetail(summary.id),
    enabled: expanded,
    refetchInterval: isActive ? 2000 : false,
  });

  const progress = summary.step_count > 0
    ? Math.round((summary.steps_complete / summary.step_count) * 100)
    : 0;

  return (
    <div className={`border rounded-lg overflow-hidden transition-all
      ${isActive ? "border-blue-200 shadow-sm" : "border-gray-200"}`}>

      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full text-left px-4 py-3 flex items-center gap-3 hover:bg-gray-50 transition-colors"
      >
        <StatusDot status={summary.status} />

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-gray-900 truncate">
              {summary.goal_title || summary.trigger_type}
            </span>
            <StatusLabel status={summary.status} />
          </div>
          <div className="flex items-center gap-3 mt-0.5 text-xs text-gray-400">
            <span>{summary.id.slice(0, 16)}</span>
            <span>attempt {summary.attempts}/{summary.max_attempts}</span>
            <span className="flex items-center gap-0.5">
              <DollarSign className="h-3 w-3" />
              {summary.cost_usd.toFixed(4)}
            </span>
            <span>{summary.tokens_used.toLocaleString()} tokens</span>
          </div>
        </div>

        <div className="flex items-center gap-2 flex-shrink-0">
          <div className="w-24 h-1.5 bg-gray-100 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500
                ${summary.status === "complete" ? "bg-green-500" :
                  summary.status === "failed" ? "bg-red-500" :
                  "bg-blue-500"}`}
              style={{ width: `${summary.status === "complete" ? 100 : progress}%` }}
            />
          </div>
          <span className="text-xs text-gray-400 min-w-[40px] text-right">
            {summary.steps_complete}/{summary.step_count}
          </span>
          {expanded ? <ChevronUp className="h-4 w-4 text-gray-400" /> : <ChevronDown className="h-4 w-4 text-gray-400" />}
        </div>
      </button>

      {expanded && detail.data && (
        <div className="px-4 pb-4 pt-1 border-t border-gray-100 space-y-4">

          {detail.data.goal && (
            <div>
              <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1.5">Goal contract</h4>
              <div className="bg-gray-50 rounded p-3 text-xs space-y-1.5">
                <p className="text-gray-700">{detail.data.goal.description}</p>
                <div className="mt-2">
                  <span className="text-gray-500 font-medium">Acceptance criteria:</span>
                  <ul className="mt-1 space-y-0.5">
                    {detail.data.goal.acceptance_criteria.map((c: string, i: number) => (
                      <li key={i} className="text-gray-600 flex items-start gap-1">
                        <span className="text-gray-400 mt-0.5">{i + 1}.</span> {c}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )}

          {detail.data.plan?.reasoning && (
            <div>
              <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1.5">Plan</h4>
              <p className="text-xs text-gray-600 mb-2">{detail.data.plan.reasoning}</p>
              {detail.data.plan.suggested_key && (
                <div className="text-xs text-gray-400 mb-2">
                  Target key: <code className="bg-gray-100 px-1 py-0.5 rounded">{detail.data.plan.suggested_key}</code>
                </div>
              )}
            </div>
          )}

          <div>
            <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Execution</h4>
            <StepTimeline plan={detail.data.plan} steps={detail.data.steps} />
          </div>

          {Object.keys(detail.data.deliverables).length > 0 && (
            <div>
              <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1.5">Deliverable</h4>
              {Object.entries(detail.data.deliverables).map(([name, content]) => (
                <div key={name} className="bg-gray-50 border border-gray-100 rounded p-3 text-xs text-gray-700 max-h-40 overflow-y-auto whitespace-pre-wrap font-mono">
                  {content}
                </div>
              ))}
            </div>
          )}

          {(detail.data.attempt_history?.length ?? 0) > 0 && (
            <div>
              <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1.5">Attempts</h4>
              <div className="space-y-2">
                {detail.data.attempt_history!.map((a) => (
                  <AttemptFlow key={a.attempt} a={a} />
                ))}
              </div>
            </div>
          )}

          <div>
            <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1.5">Validation</h4>
            <ValidationPanel
              validation={detail.data.validation}
              rejectionLog={(detail.data.attempt_history?.length ?? 0) > 0 ? [] : detail.data.rejection_log}
            />
          </div>
        </div>
      )}
    </div>
  );
}

// Main Component

export function CognitionTracker() {
  const { selectedCustomerId } = useSelectedCustomer();

  const envQuery = useQuery({
    queryKey: ["cognition-environments", selectedCustomerId],
    queryFn: () => fetchEnvironments(selectedCustomerId!),
    enabled: !!selectedCustomerId,
    refetchInterval: 3000,
  });

  const envs = envQuery.data?.environments ?? [];
  const activeCount = envs.filter(e =>
    ["orchestrating", "working", "validating", "rejected"].includes(e.status)
  ).length;

  if (!envs.length) {
    return (
      <div className="bg-white border border-gray-200 rounded-lg p-5">
        <div className="flex items-center gap-2 mb-2">
          <Activity className="h-4 w-4 text-gray-400" />
          <h3 className="text-sm font-semibold text-gray-900">Cognition Environments</h3>
        </div>
        <div className="flex flex-col items-center justify-center py-8 text-center">
          <Brain className="h-8 w-8 text-gray-300 mb-2" />
          <p className="text-sm font-medium text-gray-500">No cognition runs yet</p>
          <p className="text-xs text-gray-400 mt-1">
            Runs appear here when research tasks or knowledge gaps are processed — and recent completed runs stay visible.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-5">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-gray-500" />
          <h3 className="text-sm font-semibold text-gray-900">Cognition Environments</h3>
          <span className="text-xs text-gray-400">({envs.length})</span>
          {activeCount > 0 && (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-blue-50 text-blue-700">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse" />
              {activeCount} active
            </span>
          )}
        </div>
      </div>

      <div className="space-y-3">
        {envs.map(env => (
          <EnvironmentCard key={env.id} env={env} />
        ))}
      </div>
    </div>
  );
}
