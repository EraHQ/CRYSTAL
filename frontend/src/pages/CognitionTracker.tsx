// Cognition run drawer — Console Refine PR 6 (docs/CONSOLE_UX_AUDIT.md
// 5.4 item 4). The Evidence Bench (ratified 2026-07-15) survives as the
// content: one run at full fidelity (the zero-truncation rule), its
// goal and checklist, every step's trace, the verdicts, and the notes
// operators pin for the next attempt. It now opens as a Drawer at
// /cognition/run/:runId instead of three panes inside a fixed-height
// card, and tenants see a share of today's capacity, never dollars.
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, ChevronDown, ChevronRight, CircleDashed, Loader2, RotateCcw, Square, XCircle } from "lucide-react";
import { api, errorMessage } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { Drawer, ErrorBanner, Skeleton, StatusPill, useToast } from "@/components/ui";
import type { StatusVariant } from "@/components/ui";
import type { CognitionRunDetail } from "@/lib/types";
import { cn } from "@/lib/utils";
import { PipelineStrip } from "@/components/cognition/PipelineStrip";
import {
  AmendmentsPanel, type CogEvent, CountUp, EventsFeed, InfeasibleFlag, LiveTimer, ProvenanceBadges, RouteBadge,
  ScoreSparkline, StepDurationBars,
} from "@/components/cognition/widgets";
import { CritiquePanel } from "@/components/cognition/CritiquePanel";
import { type Critique, fetchCritiques } from "@/components/cognition/critiques";

// ---------------------------------------------------------------- labels

export const ACTIVE_STATUSES = ["orchestrating", "working", "validating", "rejected"];
export const isActiveStatus = (s: string) => ACTIVE_STATUSES.includes(s);

// Run status (cognition engine) in plain words; the enum rides in `title`.
export const RUN_STATUS: Record<string, { label: string; variant: StatusVariant }> = {
  orchestrating: { label: "Planning", variant: "info" },
  working: { label: "Working", variant: "info" },
  validating: { label: "Checking", variant: "info" },
  rejected: { label: "Retrying", variant: "warn" },
  complete: { label: "Done", variant: "ok" },
  failed: { label: "Gave up", variant: "bad" },
  needs_human_review: { label: "Needs you", variant: "warn" },
  // Operator stopped it: neutral, never red ("I stopped this" and "this broke" never share a color).
  cancelled: { label: "Stopped", variant: "neutral" },
};
export function runStatus(s: string) {
  return RUN_STATUS[s] ?? { label: s, variant: "neutral" as StatusVariant };
}

const CRITERION: Record<string, { label: string; variant: StatusVariant }> = {
  MET: { label: "Met", variant: "ok" },
  PARTIALLY_MET: { label: "Partly", variant: "warn" },
  NOT_MET: { label: "Not yet", variant: "bad" },
};
function criterion(status: string | undefined) {
  if (!status) return { label: "Not yet", variant: "neutral" as StatusVariant };
  return CRITERION[status] ?? { label: "Not yet", variant: "bad" as StatusVariant };
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return <div className="mb-1 text-[11px] font-medium uppercase tracking-[0.08em] text-fg-3">{children}</div>;
}

function stepStateIcon(status: string) {
  if (status === "complete") return <CheckCircle2 aria-hidden className="h-3.5 w-3.5 shrink-0 text-ok" />;
  if (status === "running") return <Loader2 aria-hidden className="h-3.5 w-3.5 shrink-0 animate-spin text-accent" />;
  if (status === "failed") return <XCircle aria-hidden className="h-3.5 w-3.5 shrink-0 text-bad" />;
  return <CircleDashed aria-hidden className="h-3.5 w-3.5 shrink-0 text-fg-3" />;
}

const STEP_STATE: Record<string, string> = { complete: "done", running: "running", failed: "failed", pending: "waiting" };

// ------------------------------------------------------------ tab: goal

function GoalTab({ detail }: { detail: CognitionRunDetail }) {
  const g = detail.goal;
  if (!g) return <p className="text-[12.5px] text-fg-3">No goal yet. The planner writes it first.</p>;
  const evals = detail.validation?.criteria_evaluation ?? [];
  return (
    <div className="space-y-4">
      <div>
        <SectionLabel>Goal</SectionLabel>
        <p className="whitespace-pre-wrap break-words text-[13.5px] text-fg">{g.description}</p>
      </div>
      <div>
        <SectionLabel>Checklist</SectionLabel>
        <ol className="space-y-2">
          {g.acceptance_criteria.map((c, i) => {
            const ev = evals[i];
            const st = criterion(ev?.status);
            return (
              <li key={i} className="rounded-lg border border-line bg-surface px-3 py-2 text-[13px]">
                <div className="flex items-start gap-2">
                  <StatusPill variant={st.variant} raw={ev?.status} className="shrink-0">{st.label}</StatusPill>
                  <span className="min-w-0 flex-1 break-words text-fg">{c}{ev?.possibly_infeasible && <InfeasibleFlag />}</span>
                </div>
                {ev?.evidence && <p className="mt-1.5 whitespace-pre-wrap break-words text-[12.5px] text-fg-2">{ev.evidence}</p>}
              </li>
            );
          })}
        </ol>
      </div>
      {(g.amendments?.length ?? 0) > 0 && (
        <div>
          <SectionLabel>Checklist changed on evidence</SectionLabel>
          <AmendmentsPanel amendments={g.amendments!} />
        </div>
      )}
      {detail.plan?.reasoning && (
        <div>
          <SectionLabel>Plan</SectionLabel>
          <div className="mb-1 flex items-center gap-2">
            <RouteBadge route={detail.plan.retry_route} />
            {detail.plan.suggested_key && <code className="rounded bg-inset px-1 py-0.5 text-[11px] text-fg-2">{detail.plan.suggested_key}</code>}
          </div>
          <p className="whitespace-pre-wrap break-words text-[12.5px] text-fg-2">{detail.plan.reasoning}</p>
        </div>
      )}
    </div>
  );
}

// ----------------------------------------------------------- tab: steps

function StepTraceBody({ sr }: { sr: { output?: any; error?: string | null } | undefined }) {
  const [openOutput, setOpenOutput] = useState(false);
  const out = sr?.output ?? {};
  const toolCalls: any[] = Array.isArray(out.tool_calls) ? out.tool_calls : [];
  const findings: any[] = Array.isArray(out.findings) ? out.findings : [];
  return (
    <div className="space-y-3">
      {sr?.error && <ErrorBanner title="This step failed" message={sr.error} />}
      {toolCalls.length > 0 && (
        <div>
          <SectionLabel>Tool calls ({toolCalls.length})</SectionLabel>
          <div className="space-y-2">
            {toolCalls.map((c, i) => (
              <div key={i} className="rounded-lg border border-accent/30 p-2.5">
                <div className="mb-1 flex items-center gap-2 text-[12px]">
                  <span className="rounded bg-accent-soft px-1 text-[10px] font-semibold text-accent">{c.iteration ?? i + 1}</span>
                  <span className="font-mono text-fg">{c.tool}</span>
                </div>
                {c.input && <pre className="mb-1 whitespace-pre-wrap break-words rounded bg-inset p-1.5 text-[11px] text-fg-2">{JSON.stringify(c.input, null, 1)}</pre>}
                {c.output_head && <pre className="max-h-64 overflow-y-auto whitespace-pre-wrap break-words rounded border border-line bg-surface p-1.5 text-[11px] text-fg-2">{c.output_head}</pre>}
              </div>
            ))}
          </div>
        </div>
      )}
      {findings.length > 0 && (
        <div>
          <SectionLabel>Findings ({findings.length})</SectionLabel>
          <div className="space-y-1.5">{findings.map((f, i) => <FindingBlock key={i} f={f} />)}</div>
        </div>
      )}
      {(out.content || out.content_text) && (
        <div>
          <button type="button" onClick={() => setOpenOutput(!openOutput)} aria-expanded={openOutput}
            className="mb-1 flex items-center gap-1 text-[12px] text-fg-2 hover:text-fg">
            {openOutput ? <ChevronDown aria-hidden className="h-3 w-3" /> : <ChevronRight aria-hidden className="h-3 w-3" />}
            Raw output
          </button>
          {openOutput && <pre className="whitespace-pre-wrap break-words rounded-lg border border-line bg-inset p-3 text-[12px] text-fg-2">{String(out.content ?? out.content_text)}</pre>}
        </div>
      )}
      {out.results_count !== undefined && <p className="text-[12.5px] text-fg-2">Found {out.results_count} results</p>}
    </div>
  );
}

function FindingBlock({ f }: { f: any }) {
  const [open, setOpen] = useState(false);
  const title = f?.title || f?.url || "finding";
  return (
    <div className="rounded-lg border border-line p-2">
      <button type="button" onClick={() => setOpen(!open)} aria-expanded={open} className="flex w-full items-center gap-1.5 text-left">
        {open ? <ChevronDown aria-hidden className="h-3 w-3 text-fg-3" /> : <ChevronRight aria-hidden className="h-3 w-3 text-fg-3" />}
        <span className="min-w-0 flex-1 truncate text-[12.5px] text-fg">{title}</span>
        <ProvenanceBadges output={{ findings: [f] }} />
      </button>
      {open && (
        <div className="mt-1.5 pl-4">
          {f?.url && <p className="mb-1 break-all text-[11px] text-info">{f.url}</p>}
          <pre className="max-h-72 overflow-y-auto whitespace-pre-wrap break-words text-[11.5px] text-fg-2">{f?.content || f?.snippet || ""}</pre>
        </div>
      )}
    </div>
  );
}

function StepCard({ detail, stepId }: { detail: CognitionRunDetail; stepId: number }) {
  const [open, setOpen] = useState(false);
  const ps = (detail.plan?.steps ?? []).find((s) => s.id === stepId);
  const sr = detail.steps[String(stepId)];
  const out = sr?.output ?? {};
  const stepEvents = ((detail.events ?? []) as CogEvent[]).filter((e) => e.step_id === stepId);
  const state = sr?.status ?? "pending";
  return (
    <div className="rounded-lg border border-line bg-surface">
      <button type="button" onClick={() => setOpen(!open)} aria-expanded={open} className="flex w-full flex-wrap items-center gap-2 px-3 py-2 text-left">
        {stepStateIcon(state)}
        <span className="font-mono text-[12.5px] text-fg">{ps?.action ?? "step"}</span>
        <span className="min-w-0 flex-1 truncate text-[12.5px] text-fg-3">{ps?.description}</span>
        <span className="text-[11.5px] text-fg-3" title={state}>{STEP_STATE[state] ?? state}</span>
        {sr?.duration_ms ? <span className="text-[11.5px] text-fg-3">{(sr.duration_ms / 1000).toFixed(1)}s</span> : null}
      </button>
      {open && (
        <div className="space-y-3 border-t border-line px-3 py-2.5">
          <div className="flex flex-wrap items-center gap-2 text-[11.5px]">
            {out.agentic && <StatusPill variant="info">agentic · {out.iterations ?? "?"} iterations</StatusPill>}
            {sr?.model_used && sr.model_used !== "none" && <StatusPill variant="vio" raw="model_used">{sr.model_used}</StatusPill>}
            <ProvenanceBadges output={out} />
          </div>
          {stepEvents.length > 0 && (
            <div>
              <SectionLabel>Step events</SectionLabel>
              <EventsFeed events={stepEvents} live={false} />
            </div>
          )}
          <StepTraceBody sr={sr} />
        </div>
      )}
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
    <div className="rounded-lg border border-line">
      <button type="button" onClick={() => setOpen(!open)} aria-expanded={open} className="flex w-full items-center gap-2 px-2.5 py-2 text-left">
        {open ? <ChevronDown aria-hidden className="h-3 w-3 shrink-0 text-fg-3" /> : <ChevronRight aria-hidden className="h-3 w-3 shrink-0 text-fg-3" />}
        {stepStateIcon(rec.status)}
        <span className="font-mono text-[12px] text-fg">{rec.action}</span>
        <span className={cn("min-w-0 flex-1 truncate text-[11.5px]", rec?.error ? "text-bad" : "text-fg-3")}>{summary}</span>
        {rec.duration_ms != null && <span className="text-[11px] text-fg-3">{(rec.duration_ms / 1000).toFixed(1)}s</span>}
      </button>
      {open && <div className="border-t border-line px-3 pb-3 pt-2"><StepTraceBody sr={rec} /></div>}
    </div>
  );
}

function StepsTab({ detail }: { detail: CognitionRunDetail }) {
  const steps = detail.plan?.steps ?? [];
  const history = detail.attempt_history ?? [];
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-4">
        <ScoreSparkline attempts={history} current={detail.validation} />
        <StepDurationBars steps={detail.steps as any} />
      </div>
      <div>
        <SectionLabel>This attempt</SectionLabel>
        {steps.length === 0 && <p className="text-[12.5px] text-fg-3">No steps planned yet.</p>}
        <div className="space-y-1.5">
          {steps.map((ps) => <StepCard key={ps.id} detail={detail} stepId={ps.id} />)}
        </div>
      </div>
      {history.map((a) => (
        <div key={a.attempt}>
          <SectionLabel>Attempt {a.attempt} · score {((a.validation?.score ?? 0) * 100).toFixed(0)}% <RouteBadge route={a.plan?.retry_route} /></SectionLabel>
          {a.plan?.reasoning && <p className="mb-1.5 whitespace-pre-wrap text-[12.5px] text-fg-2">{a.plan.reasoning}</p>}
          <div className="space-y-1.5">
            {Object.values(a.steps ?? {}).map((s, i) => <ArchivedStep key={i} s={s} />)}
          </div>
        </div>
      ))}
    </div>
  );
}

// -------------------------------------------------------- tab: verdicts

function VerdictsTab({ detail }: { detail: CognitionRunDetail }) {
  const v = detail.validation;
  const entries = Object.entries(detail.deliverables ?? {});
  const history = detail.attempt_history ?? [];
  return (
    <div className="space-y-4">
      {v ? (
        <div className={cn("rounded-lg border p-3", v.approved ? "border-ok/40 bg-ok-soft" : "border-bad/40 bg-bad-soft")}>
          <div className="mb-1 flex items-center gap-2">
            {v.approved ? <CheckCircle2 aria-hidden className="h-4 w-4 text-ok" /> : <XCircle aria-hidden className="h-4 w-4 text-bad" />}
            <span className={cn("text-[13.5px] font-medium", v.approved ? "text-ok" : "text-bad")}>{v.approved ? "Approved" : "Rejected"}</span>
            <StatusPill variant={v.approved ? "ok" : "bad"}>{(v.score * 100).toFixed(0)}%</StatusPill>
          </div>
          <p className="whitespace-pre-wrap break-words text-[12.5px] text-fg">{v.reasoning}</p>
          {v.issues.length > 0 && (
            <ul className="mt-2 space-y-0.5">
              {v.issues.map((iss, i) => <li key={i} className="text-[12.5px] text-bad">• {iss}</li>)}
            </ul>
          )}
        </div>
      ) : (
        <p className="text-[12.5px] text-fg-3">No verdict yet. The checker writes one after the steps finish.</p>
      )}
      {detail.rejection_log?.length > 0 && (
        <div>
          <SectionLabel>Earlier verdicts</SectionLabel>
          <div className="space-y-1.5">
            {detail.rejection_log.map((r, i) => (
              <div key={i} className="rounded-lg border border-line px-3 py-2 text-[12.5px]">
                <div className="mb-0.5 flex items-center gap-2 text-fg-3">
                  <span>Attempt {r.attempt}</span>
                  <StatusPill variant="bad">{((r.score ?? 0) * 100).toFixed(0)}%</StatusPill>
                </div>
                <p className="whitespace-pre-wrap break-words text-fg-2">{r.reasoning}</p>
              </div>
            ))}
          </div>
        </div>
      )}
      {history.filter((a) => a.deliverable).map((a) => (
        <div key={a.attempt}>
          <SectionLabel>Attempt {a.attempt} deliverable</SectionLabel>
          <pre className="max-h-96 overflow-y-auto whitespace-pre-wrap break-words rounded-lg border border-line bg-inset p-3 text-[12px] text-fg-2">{a.deliverable}</pre>
        </div>
      ))}
      {entries.length > 0 && (
        <div className="space-y-5">
          {entries.map(([name, content]) => (
            <div key={name}>
              <div className="mb-1 flex items-center justify-between">
                <SectionLabel>{name}</SectionLabel>
                <button type="button" onClick={() => navigator.clipboard?.writeText(content)} className="text-[12px] text-fg-3 hover:text-fg">Copy</button>
              </div>
              {/* The run's product, read at reading width, never truncated. */}
              <div className="max-w-[70ch] whitespace-pre-wrap break-words text-[13px] leading-relaxed text-fg">{content}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ----------------------------------------------------------- tab: notes

function NotesTab({ detail, critiques }: { detail: CognitionRunDetail; critiques: Critique[] }) {
  return (
    <div className="space-y-4">
      <p className="text-[12.5px] text-fg-2">
        A note pins to this run. Open notes are read by the planner on the next attempt and the next run of the same task.
      </p>
      <CritiquePanel envId={detail.id} targetPath="run" critiques={critiques} />
      {critiques.some((c) => c.target_path !== "run") && (
        <div>
          <SectionLabel>Notes on parts of this run</SectionLabel>
          <div className="space-y-2">
            {critiques.filter((c) => c.target_path !== "run").map((c) => (
              <div key={c.id} className={cn("rounded-lg border p-2.5 text-[12.5px]", c.status === "open" ? "border-warn/40 bg-warn-soft" : "border-line bg-inset opacity-70")}>
                <div className="mb-0.5 flex items-center gap-2">
                  <code className="rounded border border-line bg-surface px-1 text-[11px] text-fg-2">{c.target_path}</code>
                  <span className="text-fg-3">{c.author}</span>
                  <span className="flex-1" />
                  <StatusPill variant={c.status === "open" ? "warn" : "neutral"}>{c.status === "open" ? "Open" : "Resolved"}</StatusPill>
                </div>
                <p className="whitespace-pre-wrap break-words text-fg">{c.text}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ----------------------------------------------------------------- drawer

export function RunDrawer({ runId, onClose }: { runId: string; onClose: () => void }) {
  const { me } = useAuth();
  const isPlatformAdmin = me?.role === "platform_admin" || me?.kind === "platform_admin_key";
  const qc = useQueryClient();
  const toast = useToast();
  const [tab, setTab] = useState<"goal" | "steps" | "verdicts" | "notes">("goal");
  const [actionError, setActionError] = useState<string | null>(null);

  const detailQuery = useQuery({
    queryKey: ["cognition-env-detail", runId],
    queryFn: () => api.getCognitionRun(runId),
    refetchInterval: (q) => (q.state.data && isActiveStatus(q.state.data.status) ? 2000 : false),
  });
  const critiquesQuery = useQuery({ queryKey: ["run-critiques", runId], queryFn: () => fetchCritiques(runId) });
  const critiques = critiquesQuery.data ?? [];
  const d = detailQuery.data;
  const active = d ? isActiveStatus(d.status) : false;
  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["cognition-environments"] });
    qc.invalidateQueries({ queryKey: ["cognition-env-detail", runId] });
  };
  const stop = useMutation({
    mutationFn: () => api.cancelCognitionTask(d!.trigger_id!),
    onSuccess: (out) => { setActionError(null); toast(out.cancelled === "requested" ? "Stopping at the next step." : "Run stopped."); refresh(); },
    onError: (e) => setActionError(errorMessage(e, "Couldn't stop this run.")),
  });
  const rerun = useMutation({
    mutationFn: () => api.requeueCognitionTask(d!.trigger_id!),
    onSuccess: () => { setActionError(null); toast("Queued. A new cycle starts on the next poll."); refresh(); },
    onError: (e) => setActionError(errorMessage(e, "Couldn't queue a new cycle.")),
  });
  const taskRun = !!d?.trigger_id && d.trigger_type !== "fill_gap";
  const st = d ? runStatus(d.status) : null;
  const openNotes = critiques.filter((c) => c.status === "open").length;

  const tabs = [
    ["goal", "Goal and checklist"], ["steps", `Steps ${d ? Object.keys(d.steps ?? {}).length : ""}`.trim()],
    ["verdicts", "Verdicts"], ["notes", `Notes${openNotes ? ` ${openNotes}` : ""}`],
  ] as const;

  return (
    <Drawer open onClose={onClose} title={d?.goal?.title ?? (detailQuery.isLoading ? "Loading…" : "Run")} className="min-[720px]:w-[640px]">
      <nav className="mb-3 text-[12.5px]">
        <button type="button" onClick={onClose} className="text-fg-2 hover:text-fg">‹ All runs</button>
      </nav>
      {detailQuery.isLoading && <Skeleton rows={5} />}
      {detailQuery.isError && <ErrorBanner title="Couldn't load this run" error={detailQuery.error} onRetry={() => detailQuery.refetch()} />}
      {actionError && <ErrorBanner title="That didn't go through" message={actionError} className="mb-3" />}
      {d && st && (
        <>
          <div className="flex flex-wrap items-center gap-2">
            <StatusPill variant={st.variant} raw={d.status} dot={active}>{st.label}</StatusPill>
            <span className="flex-1" />
            {taskRun && active && (
              <button type="button" onClick={() => stop.mutate()} disabled={stop.isPending}
                className="inline-flex items-center gap-1 rounded-md border border-line px-2.5 py-1 text-[12.5px] font-medium text-fg-2 hover:bg-bad-soft hover:text-bad disabled:opacity-50"
                title="A live run stops at its next step, never mid-call">
                <Square aria-hidden className="h-3 w-3" /> {stop.isPending ? "Stopping…" : "Stop"}
              </button>
            )}
            {taskRun && (
              <button type="button" onClick={() => rerun.mutate()} disabled={rerun.isPending}
                className="inline-flex items-center gap-1 rounded-md border border-line px-2.5 py-1 text-[12.5px] font-medium text-fg-2 hover:bg-inset hover:text-fg disabled:opacity-50"
                title="Queue this task again; the next cycle sees this run's verdict and open notes">
                <RotateCcw aria-hidden className={cn("h-3 w-3", rerun.isPending && "animate-spin")} /> Re-run
              </button>
            )}
          </div>
          <p className="mt-2 text-[12.5px] text-fg-2">
            attempt {d.attempts} of {d.max_attempts}
            {(d.cycle ?? 1) > 1 ? ` · cycle ${d.cycle} of ${d.cycle_cap ?? 3}` : ""}
            {d.capacity_share_pct != null ? ` · ${d.capacity_share_pct}% of today's capacity` : ""}
            {isPlatformAdmin ? <> · $<CountUp value={d.cost_usd ?? 0} decimals={4} /></> : null}
            {" · "}<CountUp value={d.tokens_used ?? 0} /> tokens
            {active ? <> · <LiveTimer since={d.created_at} /></> : null}
          </p>
          <div className="mt-3"><PipelineStrip status={d.status} steps={d.steps ?? {}} /></div>

          <div role="tablist" aria-label="Run sections" className="mt-4 mb-3 flex flex-wrap gap-4 border-b border-line text-[12.5px]">
            {tabs.map(([id, label]) => (
              <button key={id} type="button" role="tab" aria-selected={tab === id} onClick={() => setTab(id)}
                className={cn("-mb-px border-b-2 pb-2", tab === id ? "border-accent font-medium text-accent" : "border-transparent text-fg-3 hover:text-fg-2")}>
                {label}
              </button>
            ))}
          </div>
          {tab === "goal" && <GoalTab detail={d} />}
          {tab === "steps" && <StepsTab detail={d} />}
          {tab === "verdicts" && <VerdictsTab detail={d} />}
          {tab === "notes" && <NotesTab detail={d} critiques={critiques} />}
        </>
      )}
    </Drawer>
  );
}
