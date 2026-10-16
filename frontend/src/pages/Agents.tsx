// Agents — Console Refine PR 13 (docs/CONSOLE_UX_AUDIT.md 5.11). Everything
// CRYS does for a team: live sessions with turn-by-turn activity, the
// daemon's background queue, and the runs that failed. Platform-admin only
// (the route is not mounted for tenants), so session cost may show dollars.
//
// Honest limits (recorded in the handoff): approval decisions must be
// SIGNED with an operator key and verified by the agent
// (endpoints/control.py), which the console session cannot do — the
// banner shows what a session is waiting on and says where to answer it.
import { Fragment, ReactNode, useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Bot, ChevronDown, ChevronRight } from "lucide-react";
import { api, errorMessage } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";
import { EmptyState, ErrorBanner, PageHeader, Segmented, Skeleton, StatusPill, useToast } from "@/components/ui";
import type { StatusVariant } from "@/components/ui";
import type { AgentEvent, AgentGap, AgentSession, AgentTask, OperatorRow } from "@/lib/types";
import { cn, fmtDateTime, fmtNum, fmtRelative, truncate } from "@/lib/utils";

// The list polls so the view stays live — agents heartbeat at turn
// boundaries and on a timer, and stale ones flip to crashed server-side.
export const REFRESH_MS = 2000;
const STALE_MS = 2 * 60_000;

const LIVE_STATUSES = new Set(["running", "awaiting_approval"]);

export const SESSION_STATUS: Record<string, { label: string; variant: StatusVariant }> = {
  running: { label: "Working", variant: "ok" },
  awaiting_approval: { label: "Needs approval", variant: "warn" },
  idle: { label: "Idle", variant: "info" },
  crashed: { label: "Crashed", variant: "bad" },
  exited: { label: "Exited", variant: "neutral" },
};

export const TASK_STATUS: Record<string, { label: string; variant: StatusVariant }> = {
  queued: { label: "Queued", variant: "info" },
  running: { label: "Running", variant: "ok" },
  done: { label: "Done", variant: "neutral" },
  failed: { label: "Failed", variant: "bad" },
  cancelled: { label: "Cancelled", variant: "neutral" },
};

export const GAP_STATUS: Record<string, { label: string; variant: StatusVariant }> = {
  open: { label: "Open", variant: "warn" },
  retrying: { label: "Retrying", variant: "info" },
  filled: { label: "Filled", variant: "ok" },
  needs_operator: { label: "Needs an operator", variant: "bad" },
};

export const COMMAND_STATUS: Record<string, string> = { pending: "Waiting for the agent", consumed: "Delivered", voided: "Voided" };

export function fmtUsd(micro: number | null | undefined): string {
  if (micro == null) return "$0.00";
  return "$" + (micro / 1_000_000).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 });
}

// "4s ago" while fresh, then minutes, then the shared relative format.
export function heartbeatLabel(iso: string | null | undefined, now: number): string {
  if (!iso) return "never";
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return "never";
  const s = Math.max(0, Math.round((now - t) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  return fmtRelative(iso, now);
}

export function isStale(iso: string | null | undefined, now: number): boolean {
  if (!iso) return true;
  const t = new Date(iso).getTime();
  return Number.isNaN(t) || now - t > STALE_MS;
}

function useNow(ms: number): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), ms);
    return () => window.clearInterval(id);
  }, [ms]);
  return now;
}

// What a waiting session is asking for, from whatever the agent put in its
// payload (the shape is the agent's; we read the common keys defensively).
export function approvalSummary(payload: Record<string, unknown> | null | undefined): { action: string; command: string | null; reason: string | null } {
  const p = payload ?? {};
  const str = (k: string) => (typeof p[k] === "string" && (p[k] as string).trim() ? (p[k] as string).trim() : null);
  const command = str("command") ?? str("cmd") ?? str("tool_input");
  const action = str("action") ?? str("kind") ?? str("tool") ?? (command ? "run a command" : "continue");
  return { action, command, reason: str("reason") ?? str("why") ?? str("description") };
}

export function operatorLabel(id: string | null, names: Map<string, string>): string {
  if (!id) return "Team key";
  return names.get(id) ?? truncate(id, 14);
}

type Tab = "live" | "queue" | "failed";

export function Agents() {
  const { selectedCustomerId } = useSelectedCustomer();
  const [tab, setTab] = useState<Tab>("live");
  const now = useNow(1000);

  const sessions = useQuery({
    queryKey: ["sessions", selectedCustomerId],
    queryFn: () => api.listSessions(selectedCustomerId!),
    enabled: !!selectedCustomerId,
    refetchInterval: REFRESH_MS,
  });
  const tasks = useQuery({
    queryKey: ["agent-tasks", selectedCustomerId],
    queryFn: () => api.listAgentTasks(selectedCustomerId!),
    enabled: !!selectedCustomerId,
    refetchInterval: REFRESH_MS * 2,
  });
  const gaps = useQuery({
    queryKey: ["agent-gaps", selectedCustomerId],
    queryFn: () => api.listAgentGaps(selectedCustomerId!),
    enabled: !!selectedCustomerId,
    refetchInterval: REFRESH_MS * 2,
  });
  const operators = useQuery({
    queryKey: ["operators", selectedCustomerId],
    queryFn: () => api.listOperators(selectedCustomerId!),
    enabled: !!selectedCustomerId,
  });

  if (!selectedCustomerId) {
    return (
      <>
        <PageHeader title="Agents" />
        <EmptyState icon={Bot} title="No customer selected" help="Pick a customer above to see its agents." />
      </>
    );
  }

  const rows = sessions.data?.sessions ?? [];
  const names = new Map<string, string>((operators.data?.operators ?? []).map((o: OperatorRow) => [o.id, o.display_name]));
  const waiting = rows.filter((s) => s.effective_status === "awaiting_approval");
  const crashed = rows.filter((s) => s.effective_status === "crashed").length;
  const taskRows = tasks.data?.tasks ?? [];
  const gapRows = gaps.data?.gaps ?? [];
  const queueRows = taskRows.filter((t) => t.status !== "failed");
  const failedTasks = taskRows.filter((t) => t.status === "failed");
  const failedGaps = gapRows.filter((g) => g.status !== "filled");
  const queueCount = queueRows.filter((t) => t.status === "queued" || t.status === "running").length;
  const failedCount = failedTasks.length + failedGaps.length;

  const meta = sessions.isLoading ? "Loading…" : `${rows.length} session${rows.length === 1 ? "" : "s"} · ${waiting.length} waiting for approval · ${crashed} crashed`;

  return (
    <div>
      <PageHeader title="Agents" meta={meta}
        actions={
          <span className="inline-flex items-center gap-1.5 text-[12px] text-fg-3" aria-live="off">
            <span aria-hidden className={cn("h-1.5 w-1.5 rounded-full", sessions.isError ? "bg-bad" : "bg-ok animate-pulse")} />
            Live · updates every {REFRESH_MS / 1000}s
          </span>
        } />

      <Segmented label="View" value={tab} onChange={setTab} className="mb-3"
        options={[
          { value: "live", label: `Live ${rows.length}` },
          { value: "queue", label: `Queue ${queueCount}` },
          { value: "failed", label: `Failed runs ${failedCount}` },
        ]} />

      {tab === "live" && (
        <>
          {waiting.map((s) => <ApprovalBanner key={s.session_id} session={s} names={names} />)}
          {sessions.isError && <ErrorBanner title="Couldn't load sessions" error={sessions.error} onRetry={() => sessions.refetch()} />}
          {sessions.isLoading && <Skeleton rows={4} />}
          {!sessions.isLoading && !sessions.isError && rows.length === 0 && (
            <EmptyState icon={Bot} title="No agent sessions yet"
              help="A CRYS session signed into this team appears here once it heartbeats, with its status, current action and turn-by-turn activity. A killed session shows as crashed." />
          )}
          {rows.length > 0 && <SessionTable customerId={selectedCustomerId} sessions={rows} names={names} now={now} />}
        </>
      )}

      {tab === "queue" && (
        <>
          {tasks.isError && <ErrorBanner title="Couldn't load the queue" error={tasks.error} onRetry={() => tasks.refetch()} />}
          {tasks.isLoading && <Skeleton rows={4} />}
          {!tasks.isLoading && !tasks.isError && queueRows.length === 0 && (
            <EmptyState icon={Bot} title="Queue is empty"
              help="Background tasks (from --queue or the agent's queue_task tool) show up here once a daemon is running. Start one with: python -m crys --daemon" />
          )}
          {queueRows.length > 0 && <TaskTable tasks={queueRows} />}
        </>
      )}

      {tab === "failed" && (
        <FailedRuns customerId={selectedCustomerId} tasks={failedTasks} gaps={failedGaps} loading={tasks.isLoading || gaps.isLoading} />
      )}
    </div>
  );
}

function ApprovalBanner({ session, names }: { session: AgentSession; names: Map<string, string> }) {
  const a = approvalSummary(session.awaiting_payload);
  return (
    <section role="status" aria-label="Approval needed" className="mb-3 rounded-xl border border-warn/40 bg-warn-soft px-3 py-2.5 text-[13px] text-fg">
      <p className="font-medium">{operatorLabel(session.operator_id, names)}'s session wants to {a.action}</p>
      {a.command && <pre className="mt-1 overflow-x-auto rounded-md bg-surface px-2 py-1 font-mono text-[12px] text-fg">{a.command}</pre>}
      {a.reason && <p className="mt-1 text-fg-2">Reason given: {a.reason}</p>}
      <p className="mt-1 text-[12px] text-fg-3">
        {session.host ?? "unknown host"}{session.project_dir ? ` · ${session.project_dir}` : ""}
      </p>
      <p className="mt-1.5 text-[12.5px] text-fg-2">
        Approve or deny from the terminal running this session. A decision has to be signed with an operator key, which this console doesn't hold.
      </p>
    </section>
  );
}

function SessionTable({ customerId, sessions, names, now }: { customerId: string; sessions: AgentSession[]; names: Map<string, string>; now: number }) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  return (
    <div className="overflow-x-auto rounded-xl border border-line bg-surface">
      <table className="w-full min-w-[360px] text-[13px]">
        <thead>
          <tr className="text-left text-[11px] uppercase tracking-[0.08em] text-fg-3">
            <th className="px-3 py-2 font-medium">Status</th>
            <th className="px-3 py-2 font-medium">Session</th>
            <th className="hidden px-3 py-2 font-medium min-[720px]:table-cell">Operator</th>
            <th className="px-3 py-2 font-medium">Current action</th>
            <th className="hidden px-3 py-2 font-medium min-[720px]:table-cell">Host</th>
            <th className="hidden px-3 py-2 font-medium min-[720px]:table-cell">Heartbeat</th>
          </tr>
        </thead>
        <tbody>
          {sessions.map((s) => {
            const open = expandedId === s.session_id;
            const st = SESSION_STATUS[s.effective_status] ?? { label: s.effective_status, variant: "neutral" as StatusVariant };
            const live = LIVE_STATUSES.has(s.effective_status);
            const stale = isStale(s.last_heartbeat_at, now);
            return (
              <Fragment key={s.session_id}>
                <tr className={cn("cursor-pointer border-t border-line hover:bg-inset", open && "bg-inset")} aria-expanded={open}
                  onClick={() => setExpandedId(open ? null : s.session_id)}>
                  <td className="whitespace-nowrap px-3 py-2">
                    <span className="inline-flex items-center gap-1">
                      {open ? <ChevronDown aria-hidden className="h-3.5 w-3.5 text-fg-3" /> : <ChevronRight aria-hidden className="h-3.5 w-3.5 text-fg-3" />}
                      <StatusPill variant={st.variant} raw={s.effective_status} dot={live}>{st.label}</StatusPill>
                    </span>
                  </td>
                  <td className="px-3 py-2">
                    <div className="font-mono text-[12px] text-fg">{truncate(s.session_id, 22)}</div>
                    {s.project_dir && <div className="text-[11px] text-fg-3">{truncate(s.project_dir, 36)}</div>}
                  </td>
                  <td className="hidden px-3 py-2 text-fg-2 min-[720px]:table-cell" title={s.operator_id ?? undefined}>{operatorLabel(s.operator_id, names)}</td>
                  <td className="max-w-[280px] truncate px-3 py-2 text-fg" title={s.current_action ?? undefined}>{s.current_action ? truncate(s.current_action, 60) : <span className="text-fg-3">—</span>}</td>
                  <td className="hidden px-3 py-2 text-[12px] text-fg-2 min-[720px]:table-cell">{s.host ?? "—"}{s.pid != null && <span className="text-fg-3"> · {s.pid}</span>}</td>
                  <td className={cn("hidden whitespace-nowrap px-3 py-2 text-[12px] min-[720px]:table-cell", stale ? "text-bad" : "text-fg-2")} title={s.last_heartbeat_at ? fmtDateTime(s.last_heartbeat_at) : undefined}>
                    {heartbeatLabel(s.last_heartbeat_at, now)}
                  </td>
                </tr>
                {open && (
                  <tr className="border-t border-line bg-inset/60">
                    <td colSpan={6} className="px-3 py-3"><SessionDetail customerId={customerId} session={s} now={now} /></td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function SessionDetail({ customerId, session, now }: { customerId: string; session: AgentSession; now: number }) {
  const live = LIVE_STATUSES.has(session.effective_status);
  const events = useQuery({
    queryKey: ["session-events", customerId, session.session_id],
    queryFn: () => api.getAgentEvents(customerId, session.session_id),
    refetchInterval: live ? REFRESH_MS : false,
  });
  const deps = useQuery({
    queryKey: ["session-deps", customerId, session.session_id],
    queryFn: () => api.getSessionDependencies(customerId, session.session_id),
  });
  const cmds = useQuery({
    queryKey: ["session-cmds", customerId, session.session_id],
    queryFn: () => api.getSessionCommands(customerId, session.session_id),
  });
  const evs = events.data?.events ?? [];
  const cost = evs.reduce((sum, e) => sum + (e.cost_micro_usd ?? 0), 0);
  const tokens = evs.reduce((sum, e) => sum + (e.tokens_input ?? 0) + (e.tokens_output ?? 0), 0);

  return (
    <div className="space-y-4">
      <dl className="grid grid-cols-1 gap-x-6 gap-y-2 min-[520px]:grid-cols-2 min-[720px]:grid-cols-4">
        <Field label="Model" value={session.model ?? "—"} />
        <Field label="Started" value={session.started_at ? fmtDateTime(session.started_at) : "—"} />
        <Field label="Session cost" value={cost > 0 ? `${fmtUsd(cost)}${tokens > 0 ? ` · ${fmtNum(tokens)} tokens` : ""}` : "—"} />
        <Field label="Last heartbeat" value={session.last_heartbeat_at ? `${heartbeatLabel(session.last_heartbeat_at, now)} · ${session.status}` : session.status} />
      </dl>

      <div>
        <p className="mb-1.5 text-[11px] font-medium uppercase tracking-[0.08em] text-fg-3">Activity</p>
        {events.isLoading && <p className="text-[12.5px] text-fg-3">Loading…</p>}
        {!events.isLoading && evs.length === 0 && <p className="text-[12.5px] text-fg-3">No turns recorded yet for this session.</p>}
        {evs.length > 0 && <TurnTimeline events={evs} />}
      </div>

      <div>
        <p className="mb-1.5 text-[11px] font-medium uppercase tracking-[0.08em] text-fg-3">Dependencies</p>
        {deps.data && deps.data.dependencies.length === 0 && <p className="text-[12.5px] text-fg-3">None.</p>}
        {deps.data && deps.data.dependencies.length > 0 && (
          <ul className="space-y-1">
            {deps.data.dependencies.map((d) => (
              <li key={d.dependency_id} className="flex flex-wrap items-center gap-2 text-[12.5px]">
                <span className="rounded bg-inset px-1.5 py-0.5 font-medium text-fg-2">{d.kind.replace(/_/g, " ")}</span>
                <span className="text-fg-2">{truncate(d.descriptor, 50)}</span>
                {d.pid != null && <span className="text-fg-3">pid {d.pid}</span>}
                <StatusPill variant={d.status === "active" ? "ok" : d.status === "orphaned" ? "bad" : "neutral"} raw={d.status}>
                  {d.status === "active" ? "Active" : d.status === "orphaned" ? "Orphaned" : "Exited"}
                </StatusPill>
              </li>
            ))}
          </ul>
        )}
      </div>

      {cmds.data && cmds.data.commands.length > 0 && (
        <div>
          <p className="mb-1.5 text-[11px] font-medium uppercase tracking-[0.08em] text-fg-3">Control commands</p>
          <ul className="space-y-1">
            {cmds.data.commands.map((c) => (
              <li key={c.id} className="flex flex-wrap items-center gap-2 text-[12.5px]">
                <span className="font-medium text-fg">{c.command_type.replace(/_/g, " ")}</span>
                {c.decision && <span className="text-fg-2">{c.decision}</span>}
                <StatusPill variant={c.status === "pending" ? "warn" : c.status === "consumed" ? "ok" : "neutral"} raw={c.status}>{COMMAND_STATUS[c.status] ?? c.status}</StatusPill>
                {c.created_at && <span className="text-fg-3">{fmtRelative(c.created_at, now)}</span>}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

// Group the event stream into turns; each turn is a card whose inner events
// (recall, read, edit, bash, store…) render as a step list with durations.
function TurnTimeline({ events }: { events: AgentEvent[] }) {
  const turns = new Map<number, AgentEvent[]>();
  for (const e of events) {
    const k = e.turn_index ?? -1;
    if (!turns.has(k)) turns.set(k, []);
    turns.get(k)!.push(e);
  }
  const ordered = [...turns.entries()].sort((a, b) => a[0] - b[0]);
  return <ol className="space-y-2">{ordered.map(([idx, evs]) => <TurnCard key={idx} turnIdx={idx} events={evs} />)}</ol>;
}

function fmtDuration(ms: number | null | undefined): string | null {
  if (ms == null) return null;
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)}s`;
}

function TurnCard({ turnIdx, events }: { turnIdx: number; events: AgentEvent[] }) {
  const started = events.find((e) => e.event_type === "turn_started");
  const completed = events.find((e) => e.event_type === "turn_completed");
  const failed = events.find((e) => e.event_type === "turn_failed");
  const steps = events.filter((e) => !["turn_started", "turn_completed", "turn_failed"].includes(e.event_type));
  const prompt = (started?.payload?.prompt as string | undefined) ?? started?.label ?? "";
  const tail = completed ?? failed;
  const summary = (completed?.payload?.summary as string | undefined) ?? "";
  const tools = (completed?.payload?.tools_called as string[] | undefined) ?? [];
  const files = (completed?.payload?.files_written as string[] | undefined) ?? [];
  const tokens = (completed?.tokens_input ?? 0) + (completed?.tokens_output ?? 0);
  const running = !completed && !failed;

  return (
    <li className="rounded-lg border border-line bg-surface px-3 py-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-[11px] font-medium uppercase tracking-[0.08em] text-fg-3">{turnIdx >= 0 ? `Turn ${turnIdx + 1}` : "Session"}</span>
        <span className="flex items-center gap-2 text-[11px] text-fg-3">
          {completed && <span>{fmtUsd(completed.cost_micro_usd)}</span>}
          {completed && tokens > 0 && <span>{fmtNum(tokens)} tokens</span>}
          {fmtDuration(tail?.duration_ms) && <span>{fmtDuration(tail?.duration_ms)}</span>}
          {running && <span className="text-ok">working…</span>}
          {failed && <span className="text-bad">failed</span>}
        </span>
      </div>
      {prompt && <p className="mt-1 text-[12.5px] text-fg">{truncate(prompt, 160)}</p>}
      {steps.length > 0 && (
        <ol className="mt-1.5 space-y-0.5 border-l border-line pl-2">
          {steps.map((e) => {
            const bad = e.status === "error" || e.status === "denied";
            return (
              <li key={e.id} className="flex flex-wrap items-center gap-2 text-[11.5px] text-fg-2">
                <span className="text-fg-3">{(e.phase ?? e.event_type).replace(/_/g, " ")}</span>
                <span>{truncate(e.label || e.event_type, 90)}</span>
                {fmtDuration(e.duration_ms) && <span className="text-fg-3">{fmtDuration(e.duration_ms)}</span>}
                {e.status && e.status !== "ok" && <span className={bad ? "text-bad" : "text-warn"}>{e.status}</span>}
              </li>
            );
          })}
        </ol>
      )}
      {(tools.length > 0 || files.length > 0) && (
        <div className="mt-1.5 flex flex-wrap items-center gap-1">
          {tools.map((t, i) => <span key={i} className="rounded bg-inset px-1.5 py-0.5 text-[11px] text-fg-2">{t}</span>)}
          {files.length > 0 && <span className="text-[11px] text-fg-3">· {files.length} file{files.length > 1 ? "s" : ""} written</span>}
        </div>
      )}
      {summary && <p className="mt-1.5 text-[12.5px] text-fg-2">{truncate(summary, 220)}</p>}
      {failed && <p className="mt-1 text-[12.5px] text-bad">{truncate((failed.payload?.error as string | undefined) ?? failed.label, 180)}</p>}
    </li>
  );
}

function TaskTable({ tasks }: { tasks: AgentTask[] }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-line bg-surface">
      <table className="w-full min-w-[360px] text-[13px]">
        <thead>
          <tr className="text-left text-[11px] uppercase tracking-[0.08em] text-fg-3">
            <th className="px-3 py-2 font-medium">Status</th>
            <th className="px-3 py-2 font-medium">Task</th>
            <th className="hidden px-3 py-2 font-medium min-[720px]:table-cell">Source</th>
            <th className="hidden px-3 py-2 font-medium min-[720px]:table-cell">Created</th>
            <th className="hidden px-3 py-2 font-medium min-[720px]:table-cell">Finished</th>
          </tr>
        </thead>
        <tbody>
          {tasks.map((t) => {
            const st = TASK_STATUS[t.status] ?? { label: t.status, variant: "neutral" as StatusVariant };
            return (
              <tr key={t.id} className="border-t border-line align-top">
                <td className="whitespace-nowrap px-3 py-2"><StatusPill variant={st.variant} raw={t.status} dot={t.status === "running"}>{st.label}</StatusPill></td>
                <td className="px-3 py-2">
                  <div className="text-fg">{truncate(t.task, 90)}</div>
                  {t.branch && <div className="font-mono text-[11px] text-fg-3">{truncate(t.branch, 40)}</div>}
                  {t.status === "done" && t.report && <div className="mt-0.5 text-[11.5px] text-fg-3">{truncate(t.report, 120)}</div>}
                </td>
                <td className="hidden px-3 py-2 text-[12px] text-fg-2 min-[720px]:table-cell">{t.source}{t.recur_seconds != null && <span className="text-fg-3"> · every {t.recur_seconds}s</span>}</td>
                <td className="hidden whitespace-nowrap px-3 py-2 text-[12px] text-fg-3 min-[720px]:table-cell">{t.created_at ? fmtRelative(t.created_at) : "—"}</td>
                <td className="hidden whitespace-nowrap px-3 py-2 text-[12px] text-fg-3 min-[720px]:table-cell">{t.finished_at ? fmtRelative(t.finished_at) : "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function FailedRuns({ customerId, tasks, gaps, loading }: { customerId: string; tasks: AgentTask[]; gaps: AgentGap[]; loading: boolean }) {
  const queryClient = useQueryClient();
  const toast = useToast();
  const [error, setError] = useState<string | null>(null);
  const retry = useMutation({
    mutationFn: (taskId: string) => api.retryAgentTask(customerId, taskId),
    onSuccess: () => {
      setError(null);
      toast("Queued again — the daemon picks it up on its next poll");
      void queryClient.invalidateQueries({ queryKey: ["agent-tasks", customerId] });
    },
    onError: (e) => setError(errorMessage(e, "Couldn't retry that task.")),
  });

  if (loading) return <Skeleton rows={3} />;
  if (tasks.length === 0 && gaps.length === 0) {
    return <EmptyState icon={Bot} title="No failed runs" help="Background tasks that fail, and the research gaps a daemon could not fill, show up here with the error and a Retry." />;
  }
  return (
    <div className="space-y-4">
      {error && <ErrorBanner title="Couldn't retry" message={error} />}
      {tasks.length > 0 && (
        <ul className="space-y-2">
          {tasks.map((t) => (
            <li key={t.id} className="rounded-xl border border-line bg-surface px-3 py-2.5">
              <div className="flex flex-wrap items-start justify-between gap-2">
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <StatusPill variant="bad" raw={t.status}>Failed</StatusPill>
                    <span className="text-[13px] text-fg">{truncate(t.task, 120)}</span>
                  </div>
                  <p className="mt-1 text-[12.5px] text-bad">{t.error ? truncate(t.error, 300) : "No error was recorded."}</p>
                  <p className="mt-0.5 text-[11.5px] text-fg-3">
                    {t.source}{t.branch ? ` · ${t.branch}` : ""}{t.finished_at ? ` · ${fmtRelative(t.finished_at)}` : ""}
                  </p>
                </div>
                <button type="button" onClick={() => retry.mutate(t.id)} disabled={retry.isPending}
                  className="rounded-md border border-line px-2.5 py-1 text-[12.5px] font-medium text-fg-2 hover:bg-inset hover:text-fg disabled:opacity-50">
                  {retry.isPending && retry.variables === t.id ? "Queuing…" : "Retry"}
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
      {gaps.length > 0 && (
        <div>
          <p className="mb-1.5 text-[11px] font-medium uppercase tracking-[0.08em] text-fg-3">Research the daemon could not finish</p>
          <ul className="space-y-1.5">
            {gaps.map((g) => {
              const st = GAP_STATUS[g.status] ?? { label: g.status, variant: "neutral" as StatusVariant };
              return (
                <li key={g.id} className="rounded-xl border border-line bg-surface px-3 py-2">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="text-[13px] text-fg">{truncate(g.subject, 100)}</span>
                    <StatusPill variant={st.variant} raw={g.status}>{st.label}</StatusPill>
                  </div>
                  {g.missing && <p className="mt-0.5 text-[12.5px] text-fg-2">{truncate(g.missing, 200)}</p>}
                  {g.created_at && <p className="mt-0.5 text-[11.5px] text-fg-3">{fmtRelative(g.created_at)}</p>}
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}

function Field({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div>
      <dt className="text-[11px] font-medium uppercase tracking-[0.08em] text-fg-3">{label}</dt>
      <dd className="mt-0.5 text-[13px] text-fg">{value}</dd>
    </div>
  );
}
