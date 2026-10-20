// Activity → Runs (Q90=A, 2026-10-19): every cognition run for this
// memory, newest first: what asked for it (a schedule, a knowledge gap,
// an assumption check, a connected tool), what it was asked, how it
// ended, what it produced and where it went. Rows come from the task
// list (the unit a person or a schedule triggers) joined to the engine's
// runs (attempts, steps) and to the schedules (delivery). "Pipeline
// details" opens the same drawer Cognition uses. Not a replacement for
// Cognition or Scheduled runs: those are where work is done; this is
// where it is found and audited.
import { Fragment, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Brain, ChevronDown, ChevronRight, Search } from "lucide-react";
import { api } from "@/lib/api";
import type { CognitionRun, CognitionTask, ScheduledRun } from "@/lib/types";
import { EmptyState, ErrorBanner, FilterChips, Skeleton, StatusPill, Toolbar } from "@/components/ui";
import type { StatusVariant } from "@/components/ui";
import { BTN_SECONDARY } from "@/pages/settings/shared";
import { Markdown } from "@/components/Markdown";
import { bankHref } from "@/components/cognition/ScheduleResults";
import { cn, fmtDateTime, fmtRelative } from "@/lib/utils";

export type RunSource = "scheduled" | "gap" | "assumption" | "tool" | "research";
export type RunState = "queued" | "running" | "completed" | "unfinished" | "failed" | "cancelled";

export interface RunEntry {
  id: string;
  run_id: string | null;
  when: string;
  finished: string | null;
  took: string | null;
  source: RunSource;
  source_label: string;
  task: string;
  state: RunState;
  /** The failure or the engine's reason when there is no write-up. */
  reason: string;
  /** The write-up (Markdown) when the run produced one. */
  text: string;
  crystal_id: string | null;
  attempts: string | null;
  tokens: number | null;
  delivery: string | null;
}

export const SOURCE_TEXT: Record<RunSource, string> = {
  scheduled: "Scheduled", gap: "Knowledge gap", assumption: "Assumption check", tool: "Asked by a tool", research: "Research",
};

export const STATE: Record<RunState, { label: string; variant: StatusVariant }> = {
  queued: { label: "Queued", variant: "info" },
  running: { label: "Running", variant: "info" },
  completed: { label: "Completed", variant: "ok" },
  unfinished: { label: "Couldn't finish", variant: "warn" },
  failed: { label: "Failed", variant: "bad" },
  cancelled: { label: "Cancelled", variant: "neutral" },
};

export function fmtDuration(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} min${s % 60 ? ` ${s % 60}s` : ""}`;
  const h = Math.floor(m / 60);
  return `${h} h ${m % 60} min`;
}

function hostOf(url: string): string {
  try { return new URL(url).hostname; } catch { return url; }
}

function sourceOf(t: CognitionTask): RunSource {
  const p = t.payload ?? {};
  if (p.schedule_id) return "scheduled";
  if (p.gap_id) return "gap";
  if (p.assumption_crystal_id || t.task_type.includes("verification")) return "assumption";
  if (t.task_type === "agent_research") return "tool";
  return "research";
}

function stateOf(t: CognitionTask): RunState {
  if (t.status === "pending") return "queued";
  if (t.status === "running") return "running";
  if (t.status === "failed") return "failed";
  if (t.status === "cancelled") return "cancelled";
  if (t.status === "complete") return t.result?.action === "no_actionable_findings" ? "unfinished" : "completed";
  return "queued";
}

/** Join tasks, runs and schedules into one audit list, newest first. */
export function runEntries(tasks: CognitionTask[], runs: CognitionRun[], schedules: ScheduledRun[]): RunEntry[] {
  const latestRun = new Map<string, CognitionRun>();
  for (const r of runs) {
    if (!r.trigger_id) continue;
    const prev = latestRun.get(r.trigger_id);
    if (!prev || (r.created_at ?? "") > (prev.created_at ?? "")) latestRun.set(r.trigger_id, r);
  }
  const byId = new Map(schedules.map((s) => [s.id, s]));
  return tasks
    .map((t): RunEntry => {
      const p = t.payload ?? {};
      const r = t.result ?? {};
      const run = latestRun.get(t.id) ?? null;
      const source = sourceOf(t);
      const sched = source === "scheduled" ? byId.get(String(p.schedule_id)) : undefined;
      const title = String(p.schedule_title || sched?.title || "");
      const state = stateOf(t);
      const findings = typeof r.findings === "string" ? r.findings.trim() : "";
      const reason = t.status === "failed" ? (t.error_message || "The run failed.") : (typeof r.reason === "string" ? r.reason.trim() : "");
      const start = t.started_at ?? t.created_at;
      const took = t.completed_at && start ? fmtDuration(new Date(t.completed_at).getTime() - new Date(start).getTime()) : null;
      let delivery: string | null = null;
      if (sched?.deliver_url && sched.last_task_id === t.id && sched.last_delivery) {
        delivery = sched.last_delivery === "sent" ? `Sent to ${hostOf(sched.deliver_url)}` : `Delivery ${sched.last_delivery}`;
      }
      return {
        id: t.id,
        run_id: run?.id ?? null,
        when: t.created_at,
        finished: t.completed_at ?? null,
        took,
        source,
        source_label: source === "scheduled" && title ? `${SOURCE_TEXT.scheduled} · ${title}` : SOURCE_TEXT[source],
        task: String(p.topic || p.question || p.statement || run?.goal_title || title || "Research"),
        state,
        reason,
        text: state === "completed" ? findings : "",
        crystal_id: t.result_crystal_id || (typeof r.crystal_id === "string" && r.crystal_id) || null,
        attempts: run ? `attempt ${run.attempts}/${run.max_attempts}${(run.cycle ?? 1) > 1 ? ` · cycle ${run.cycle}/${run.cycle_cap ?? 3}` : ""}` : null,
        tokens: run?.tokens_used ?? null,
        delivery,
      };
    })
    .sort((a, b) => b.when.localeCompare(a.when));
}

function matches(e: RunEntry, term: string): boolean {
  const t = term.trim().toLowerCase();
  if (!t) return true;
  return [e.task, e.source_label, e.reason, e.text].some((s) => s.toLowerCase().includes(t));
}

const STATE_FILTERS: { value: string; label: string; match: (s: RunState) => boolean }[] = [
  { value: "all", label: "All", match: () => true },
  { value: "working", label: "Working", match: (s) => s === "queued" || s === "running" },
  { value: "completed", label: "Completed", match: (s) => s === "completed" },
  { value: "unfinished", label: "Couldn't finish", match: (s) => s === "unfinished" || s === "failed" || s === "cancelled" },
];

function Expanded({ e, onPipeline }: { e: RunEntry; onPipeline: (runId: string) => void }) {
  return (
    <div className="grid gap-4 min-[720px]:grid-cols-[minmax(0,1fr)_260px]">
      <div className="min-w-0">
        <p className="text-[11px] font-medium uppercase tracking-[0.08em] text-fg-3">Asked</p>
        <p className="mt-1 whitespace-pre-wrap break-words text-[13px] text-fg">{e.task}</p>
        {e.reason && (
          <>
            <p className="mt-3 text-[11px] font-medium uppercase tracking-[0.08em] text-fg-3">{e.state === "completed" ? "Note" : "What happened"}</p>
            <p className="mt-1 whitespace-pre-wrap break-words text-[13px] text-fg-2">{e.reason}</p>
          </>
        )}
        {e.text && (
          <>
            <p className="mt-3 text-[11px] font-medium uppercase tracking-[0.08em] text-fg-3">Produced</p>
            <div className="mt-1 max-h-72 overflow-y-auto rounded-lg border border-line bg-surface px-3 py-2 text-[13px] leading-relaxed text-fg-2"><Markdown text={e.text} /></div>
          </>
        )}
        <div className="mt-3 flex flex-wrap gap-2">
          {e.run_id && <button type="button" onClick={() => onPipeline(e.run_id!)} className={BTN_SECONDARY}>Pipeline details</button>}
          {e.crystal_id && <a href={bankHref(e.crystal_id, e.task)} className={BTN_SECONDARY}>Open in bank</a>}
        </div>
      </div>
      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-[12.5px]">
        <dt className="text-fg-3">Source</dt><dd className="text-fg">{e.source_label}</dd>
        <dt className="text-fg-3">Queued</dt><dd className="text-fg">{fmtDateTime(e.when)}</dd>
        {e.finished && <><dt className="text-fg-3">Finished</dt><dd className="text-fg">{fmtDateTime(e.finished)}</dd></>}
        {e.took && <><dt className="text-fg-3">Took</dt><dd className="text-fg">{e.took}</dd></>}
        {e.attempts && <><dt className="text-fg-3">Attempts</dt><dd className="text-fg">{e.attempts}</dd></>}
        {e.tokens != null && <><dt className="text-fg-3">Tokens</dt><dd className="text-fg">{e.tokens.toLocaleString()}</dd></>}
        <dt className="text-fg-3">Saved</dt><dd className="text-fg">{e.crystal_id ? "In your bank" : "Nothing saved"}</dd>
        {e.delivery && <><dt className="text-fg-3">Delivered</dt><dd className="text-fg">{e.delivery}</dd></>}
      </dl>
    </div>
  );
}

export function RunsLog({ customerId, onPipeline }: { customerId: string; onPipeline: (runId: string) => void }) {
  const [term, setTerm] = useState("");
  const [state, setState] = useState("all");
  const [source, setSource] = useState<string | null>(null);
  const [openId, setOpenId] = useState<string | null>(null);
  const tasks = useQuery({ queryKey: ["activity-tasks", customerId], queryFn: () => api.listCognitionTasks(customerId, { limit: 200 }), refetchInterval: 15_000 });
  const runs = useQuery({ queryKey: ["cognition-environments", customerId], queryFn: () => api.listCognitionRuns(customerId), refetchInterval: 15_000 });
  const schedules = useQuery({ queryKey: ["schedules", customerId], queryFn: () => api.listSchedules(customerId) });
  const entries = useMemo(
    () => runEntries(tasks.data?.items ?? [], runs.data ?? [], schedules.data?.schedules ?? []),
    [tasks.data, runs.data, schedules.data],
  );
  const searched = entries.filter((e) => matches(e, term));
  const stateMatch = STATE_FILTERS.find((f) => f.value === state)!.match;
  const rows = searched.filter((e) => stateMatch(e.state) && (!source || e.source === source));
  const countState = (f: (s: RunState) => boolean) => searched.filter((e) => f(e.state) && (!source || e.source === source)).length;
  const sources = (Object.keys(SOURCE_TEXT) as RunSource[]).filter((s) => searched.some((e) => e.source === s));
  const countSource = (s: RunSource | null) => searched.filter((e) => stateMatch(e.state) && (!s || e.source === s)).length;
  const loading = tasks.isLoading || runs.isLoading;

  return (
    <div>
      <p className="mb-2 text-[13px] text-fg-2">{loading ? "Loading…" : `${entries.length.toLocaleString()} run${entries.length === 1 ? "" : "s"} · the latest 200`}</p>
      <div className="mb-4">
        <Toolbar search={
          <label className="relative block">
            <Search aria-hidden className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-fg-3" />
            <input value={term} onChange={(e) => setTerm(e.target.value)} placeholder="Search runs" aria-label="Search runs"
              className="w-full rounded-md border border-line bg-inset py-1.5 pl-8 pr-2.5 text-[13px] text-fg outline-none placeholder:text-fg-3 focus:border-accent" />
          </label>
        }>
          <FilterChips label="Run state" value={state} onChange={(v) => setState(v ?? "all")}
            options={STATE_FILTERS.map((f) => ({ value: f.value, label: f.label, count: countState(f.match) }))} />
          {sources.length > 1 && (
            <FilterChips label="Source" value={source ?? "any"} onChange={(v) => setSource(v === "any" || v === null ? null : v)}
              options={[{ value: "any", label: "Any", count: countSource(null) }, ...sources.map((s) => ({ value: s, label: SOURCE_TEXT[s], count: countSource(s) }))]} />
          )}
        </Toolbar>
      </div>
      {tasks.isError && <ErrorBanner title="Couldn't load runs" error={tasks.error} onRetry={() => tasks.refetch()} />}
      {runs.isError && <ErrorBanner title="Couldn't load run details" error={runs.error} onRetry={() => runs.refetch()} />}
      {loading && <Skeleton rows={5} />}
      {!loading && !tasks.isError && entries.length === 0 && (
        <EmptyState icon={Brain} title="No runs yet" help="A run starts when a schedule fires, a gap is researched, an assumption is checked, or a connected tool asks for research." />
      )}
      {!loading && entries.length > 0 && rows.length === 0 && <EmptyState icon={Search} title="Nothing matches" help="Try another phrase or filter." />}
      {rows.length > 0 && (
        <div className="overflow-x-auto rounded-xl border border-line bg-surface">
          <table className="w-full min-w-[560px] text-[13px]">
            <thead>
              <tr className="text-left text-[11px] uppercase tracking-[0.08em] text-fg-3">
                <th className="px-3 py-2 font-medium">Time</th>
                <th className="hidden px-3 py-2 font-medium min-[720px]:table-cell">Source</th>
                <th className="px-3 py-2 font-medium">Asked</th>
                <th className="px-3 py-2 font-medium">Result</th>
                <th className="hidden px-3 py-2 font-medium min-[720px]:table-cell">Where it went</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((e) => {
                const open = openId === e.id;
                const st = STATE[e.state];
                return (
                  <Fragment key={e.id}>
                    <tr className={cn("cursor-pointer border-t border-line hover:bg-inset", open && "bg-inset")} onClick={() => setOpenId(open ? null : e.id)} aria-expanded={open}>
                      <td className="whitespace-nowrap px-3 py-2 text-[12px] text-fg-3" title={fmtDateTime(e.when)}>
                        <span className="inline-flex items-center gap-1">{open ? <ChevronDown aria-hidden className="h-3.5 w-3.5" /> : <ChevronRight aria-hidden className="h-3.5 w-3.5" />}{fmtRelative(e.when)}</span>
                      </td>
                      <td className="hidden max-w-[220px] truncate px-3 py-2 text-fg-2 min-[720px]:table-cell" title={e.source_label}>{e.source_label}</td>
                      <td className="max-w-[320px] truncate px-3 py-2 text-fg" title={e.task}>{e.task}</td>
                      <td className="px-3 py-2"><StatusPill variant={st.variant} raw={e.state} dot={e.state === "running"}>{st.label}</StatusPill></td>
                      <td className="hidden max-w-[220px] truncate px-3 py-2 text-fg-2 min-[720px]:table-cell">
                        {[e.crystal_id ? "Your bank" : null, e.delivery].filter(Boolean).join(" · ") || <span className="text-fg-3">—</span>}
                      </td>
                    </tr>
                    {open && (
                      <tr className="border-t border-line bg-inset/60">
                        <td colSpan={5} className="px-3 py-3"><Expanded e={e} onPipeline={onPipeline} /></td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
