// Scheduled runs: the results, under the schedules (Anthony, 2026-10-19
// 10:52: "scheduler results should just show underneath the schedule
// section. You shouldn't have to know how to set up a webhook just to
// have a good view of the results"). Every run of a schedule is a
// cognition task carrying payload.schedule_id, and the task list already
// returns the result text, so this is a view, not a new endpoint.
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { FileText } from "lucide-react";
import { api } from "@/lib/api";
import type { CognitionTask } from "@/lib/types";
import { EmptyState, ErrorBanner, Skeleton, StatusPill } from "@/components/ui";
import type { StatusVariant } from "@/components/ui";
import { BTN_SECONDARY } from "@/pages/settings/shared";
import { fmtDateTime, fmtRelative } from "@/lib/utils";
import { Markdown } from "@/pages/ChatPlayground";

export const PREVIEW_CHARS = 700;

/** Where "open" goes: a crystal opens in the reader; a document (the
 *  engine's write-up) lists the crystals born from it. */
export function bankHref(id: string, title: string): string {
  if (id.startsWith("doc_")) return `/?doc=${encodeURIComponent(id)}&from=${encodeURIComponent(title)}`;
  return `/bank/${encodeURIComponent(id)}`;
}

export interface ScheduleResult {
  task_id: string;
  schedule_id: string;
  title: string;
  status: "complete" | "failed" | "pending" | "running" | "cancelled" | string;
  when: string | null;
  text: string;
  crystal_id: string | null;
  nothing_new: boolean;
}

/** Turn the task list into the results of scheduled runs, newest first. */
export function scheduleResults(tasks: CognitionTask[]): ScheduleResult[] {
  return tasks
    .filter((t) => !!t.payload?.schedule_id)
    .map((t) => {
      const r = t.result ?? {};
      const findings = typeof r.findings === "string" ? r.findings.trim() : "";
      const reason = typeof r.reason === "string" ? r.reason.trim() : "";
      const text = t.status === "failed" ? (t.error_message || "The run failed.") : (findings || reason || (t.status === "complete" ? "The run finished without a write-up." : ""));
      return {
        task_id: t.id,
        schedule_id: String(t.payload?.schedule_id),
        title: String(t.payload?.schedule_title || t.payload?.topic || "Scheduled run"),
        status: t.status,
        when: t.completed_at ?? t.created_at ?? null,
        text,
        crystal_id: (typeof r.crystal_id === "string" && r.crystal_id) || null,
        nothing_new: r.action === "no_actionable_findings",
      };
    })
    .sort((a, b) => (b.when ?? "").localeCompare(a.when ?? ""));
}

const STATUS: Record<string, { label: string; variant: StatusVariant }> = {
  complete: { label: "Completed", variant: "ok" },
  failed: { label: "Failed", variant: "bad" },
  pending: { label: "Queued", variant: "info" },
  running: { label: "Running", variant: "info" },
  cancelled: { label: "Cancelled", variant: "neutral" },
};

function ResultCard({ r }: { r: ScheduleResult }) {
  const [open, setOpen] = useState(false);
  const long = r.text.length > PREVIEW_CHARS;
  const collapsed = long && !open;
  const st = STATUS[r.status] ?? { label: r.status, variant: "neutral" as StatusVariant };
  return (
    <li className="px-3 py-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[13.5px] font-medium text-fg">{r.title}</span>
        <StatusPill variant={r.nothing_new ? "neutral" : st.variant} raw={r.status}>{r.nothing_new ? "Nothing new" : st.label}</StatusPill>
        {r.when && <span className="text-[12px] text-fg-3" title={fmtDateTime(r.when)}>{fmtRelative(r.when)}</span>}
      </div>
      {r.text && (
        <div data-collapsed={collapsed ? "true" : "false"} className={"relative mt-2 text-[13px] leading-relaxed text-fg-2" + (collapsed ? " max-h-56 overflow-hidden" : "")}>
          {r.status === "failed" ? <p className="whitespace-pre-wrap">{r.text}</p> : <Markdown text={r.text} />}
          {collapsed && <div aria-hidden="true" className="pointer-events-none absolute inset-x-0 bottom-0 h-10 bg-gradient-to-t from-surface to-transparent" />}
        </div>
      )}
      <div className="mt-2 flex flex-wrap gap-2">
        {long && <button type="button" onClick={() => setOpen((v) => !v)} className={BTN_SECONDARY}>{open ? "Show less" : "Show all"}</button>}
        {r.crystal_id && <a href={bankHref(r.crystal_id, r.title)} className={BTN_SECONDARY}>Open in bank</a>}
      </div>
    </li>
  );
}

export function ScheduleResults({ customerId }: { customerId: string }) {
  const tasks = useQuery({
    queryKey: ["schedule-results", customerId],
    queryFn: () => api.listCognitionTasks(customerId, { limit: 200 }),
    refetchInterval: 30_000,
  });
  const results = scheduleResults(tasks.data?.items ?? []);
  return (
    <section aria-labelledby="schedule-results-title">
      <div className="mb-2">
        <h2 id="schedule-results-title" className="text-[15px] font-semibold text-fg">Results{results.length ? ` ${results.length}` : ""}</h2>
        <p className="text-[12.5px] text-fg-3">What each run produced, newest first. Every result is also a crystal in your bank, so your connected clients can recall it.</p>
      </div>
      {tasks.isError && <ErrorBanner title="Couldn't load results" error={tasks.error} onRetry={() => tasks.refetch()} />}
      {tasks.isLoading && <Skeleton rows={3} />}
      {!tasks.isLoading && !tasks.isError && results.length === 0 && (
        <EmptyState icon={FileText} title="No results yet" help="The first run's write-up lands here when it finishes." />
      )}
      {results.length > 0 && (
        <ul className="divide-y divide-line rounded-xl border border-line bg-surface">
          {results.map((r) => <ResultCard key={r.task_id} r={r} />)}
        </ul>
      )}
    </section>
  );
}
