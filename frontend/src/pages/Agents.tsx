import { Fragment, ReactNode, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ChevronDown, ChevronRight, RefreshCw } from "lucide-react";
import { api } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";
import { EmptyState, ErrorBanner, LoadingRows } from "@/components/ui";
import type { AgentEvent, AgentSession } from "@/lib/types";
import { cn, fmtDateTime, fmtNum, truncate } from "@/lib/utils";

// Poll so the view stays live — agents heartbeat at turn boundaries + on a
// timer (in-turn), and stale ones flip to crashed server-side.
const REFRESH_MS = 5000;

const LIVE_STATUSES = new Set(["running", "awaiting_approval"]);

function fmtUsd(micro: number | null | undefined): string {
  if (micro == null) return "$0.00";
  return (
    "$" +
    (micro / 1_000_000).toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 4,
    })
  );
}

function StatusPill({ status }: { status: string }) {
  const map: Record<string, { cls: string; dot: string }> = {
    running: { cls: "bg-emerald-50 text-emerald-600 ring-emerald-200/60", dot: "bg-emerald-400" },
    idle: { cls: "bg-sky-50 text-sky-600 ring-sky-200/60", dot: "bg-sky-400" },
    awaiting_approval: { cls: "bg-amber-50 text-amber-600 ring-amber-200/60", dot: "bg-amber-400" },
    crashed: { cls: "bg-red-50 text-red-600 ring-red-200/60", dot: "bg-red-400" },
    exited: { cls: "bg-gray-50 text-gray-500 ring-gray-200/60", dot: "bg-gray-400" },
  };
  const s = map[status] ?? { cls: "bg-gray-50 text-gray-500 ring-gray-200/60", dot: "bg-gray-400" };
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-xs font-medium ring-1 ring-inset", s.cls)}>
      <span className={cn("h-1.5 w-1.5 rounded-full", s.dot, LIVE_STATUSES.has(status) && "animate-pulse")} />
      {status}
    </span>
  );
}

export function Agents() {
  const { selectedCustomerId } = useSelectedCustomer();
  const [tab, setTab] = useState<"live" | "queue">("live");

  if (!selectedCustomerId)
    return <EmptyState title="No customer selected" description="Pick a customer to see its agents." />;

  return (
    <div className="space-y-4">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-lg font-semibold text-gray-900">Agents</h1>
          <p className="text-sm text-gray-500">
            Everything CRYS does — live sessions, turn-by-turn activity, the background queue.
          </p>
        </div>
        <span className="inline-flex items-center gap-1.5 text-xs text-gray-400">
          <RefreshCw className="h-3 w-3" />
          auto-refresh
        </span>
      </div>

      <div className="inline-flex rounded-lg border border-gray-200 bg-white p-0.5 text-sm">
        {(["live", "queue"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={cn(
              "rounded-md px-3 py-1 capitalize transition-colors",
              tab === t ? "bg-brand-50 font-medium text-brand-700" : "text-gray-500 hover:text-gray-800"
            )}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === "live" ? <LiveTab customerId={selectedCustomerId} /> : <QueueTab customerId={selectedCustomerId} />}
    </div>
  );
}

// ─── Live: sessions + per-session timeline ───────────────────────────────

function LiveTab({ customerId }: { customerId: string }) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const list = useQuery({
    queryKey: ["sessions", customerId],
    queryFn: () => api.listSessions(customerId),
    refetchInterval: REFRESH_MS,
  });

  if (list.isError) return <ErrorBanner title="Couldn't load sessions" message={String(list.error)} />;
  const sessions = list.data?.sessions ?? [];

  if (!list.isLoading && sessions.length === 0)
    return (
      <EmptyState
        title="No agent sessions yet"
        description="A CRYS session logged into this team appears here once it heartbeats — with its status, current action, dependencies, and turn-by-turn activity. A killed session shows as crashed."
      />
    );

  return (
    <div className="overflow-hidden rounded-lg border border-gray-200 bg-white">
      <table className="min-w-full divide-y divide-gray-100 text-sm">
        <thead className="bg-gray-50/60">
          <tr>
            <th className="w-8 px-2 py-2.5" />
            <Th>Status</Th>
            <Th>Session</Th>
            <Th>Operator</Th>
            <Th>Current action</Th>
            <Th>Host</Th>
            <Th>Heartbeat</Th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-50">
          {list.isLoading && <LoadingRows rows={5} cols={7} />}
          {!list.isLoading &&
            sessions.map((s) => (
              <Fragment key={s.session_id}>
                <tr
                  onClick={() => setExpandedId(expandedId === s.session_id ? null : s.session_id)}
                  className="cursor-pointer transition-colors hover:bg-gray-50"
                >
                  <td className="px-2 py-2.5 text-gray-400">
                    {expandedId === s.session_id ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                  </td>
                  <td className="px-4 py-2.5"><StatusPill status={s.effective_status} /></td>
                  <td className="px-4 py-2.5">
                    <div className="font-mono text-xs text-gray-700">{truncate(s.session_id, 22)}</div>
                    {s.project_dir && <div className="text-[11px] text-gray-400">{truncate(s.project_dir, 36)}</div>}
                  </td>
                  <td className="px-4 py-2.5 text-gray-600">
                    {s.operator_id ? (
                      <span className="font-mono text-xs">{truncate(s.operator_id, 16)}</span>
                    ) : (
                      <span className="text-gray-300">team</span>
                    )}
                  </td>
                  <td className="px-4 py-2.5 text-gray-700">
                    {s.current_action ? truncate(s.current_action, 40) : <span className="text-gray-300">—</span>}
                  </td>
                  <td className="px-4 py-2.5 text-xs text-gray-500">
                    {s.host ?? "—"}
                    {s.pid != null && <span className="text-gray-300"> · {s.pid}</span>}
                  </td>
                  <td className="whitespace-nowrap px-4 py-2.5 text-xs text-gray-400">
                    {s.last_heartbeat_at ? fmtDateTime(s.last_heartbeat_at) : "—"}
                  </td>
                </tr>
                {expandedId === s.session_id && (
                  <tr className="bg-gray-50/50">
                    <td colSpan={7} className="px-4 py-4">
                      <SessionDetail customerId={customerId} session={s} />
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
        </tbody>
      </table>
    </div>
  );
}

function SessionDetail({ customerId, session }: { customerId: string; session: AgentSession }) {
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
  const sessionCost = evs.reduce((sum, e) => sum + (e.cost_micro_usd ?? 0), 0);

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-x-6 gap-y-2 md:grid-cols-4">
        <Field label="Model" value={session.model ?? "—"} />
        <Field label="Started" value={session.started_at ? fmtDateTime(session.started_at) : "—"} />
        <Field label="Session cost" value={sessionCost > 0 ? fmtUsd(sessionCost) : "—"} />
        <Field label="Self-reported" value={<span className="font-mono text-xs">{session.status}</span>} />
      </div>

      {session.awaiting_payload && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2">
          <p className="text-[11px] uppercase tracking-wider text-amber-700">Awaiting approval</p>
          <pre className="mt-1 overflow-auto text-xs text-amber-800">{JSON.stringify(session.awaiting_payload, null, 2)}</pre>
        </div>
      )}

      <div>
        <p className="mb-1.5 text-[11px] uppercase tracking-wider text-gray-400">Activity</p>
        {events.isLoading && <p className="text-xs text-gray-400">Loading…</p>}
        {!events.isLoading && evs.length === 0 && (
          <p className="text-xs text-gray-300">No turns recorded yet for this session.</p>
        )}
        {evs.length > 0 && <TurnTimeline events={evs} />}
      </div>

      <div>
        <p className="mb-1.5 text-[11px] uppercase tracking-wider text-gray-400">Dependencies</p>
        {deps.data && deps.data.dependencies.length === 0 && <p className="text-xs text-gray-300">None.</p>}
        {deps.data && deps.data.dependencies.length > 0 && (
          <div className="space-y-1">
            {deps.data.dependencies.map((d) => (
              <div key={d.dependency_id} className="flex items-center gap-2 text-xs">
                <span className="inline-flex rounded bg-gray-100 px-1.5 py-0.5 font-medium text-gray-600">{d.kind}</span>
                <span className="text-gray-600">{truncate(d.descriptor, 50)}</span>
                {d.pid != null && <span className="text-gray-400">pid {d.pid}</span>}
                <span className={d.status === "active" ? "text-emerald-600" : d.status === "orphaned" ? "text-red-500" : "text-gray-400"}>
                  {d.status}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      {cmds.data && cmds.data.commands.length > 0 && (
        <div>
          <p className="mb-1.5 text-[11px] uppercase tracking-wider text-gray-400">Control commands</p>
          <div className="space-y-1">
            {cmds.data.commands.map((c) => (
              <div key={c.id} className="flex items-center gap-2 text-xs">
                <span className="font-medium text-gray-700">{c.command_type}</span>
                {c.decision && <span className="text-gray-500">{c.decision}</span>}
                <span
                  className={cn(
                    "rounded px-1.5 py-0.5",
                    c.status === "pending" ? "bg-amber-50 text-amber-600" : c.status === "consumed" ? "bg-emerald-50 text-emerald-600" : "bg-gray-100 text-gray-500"
                  )}
                >
                  {c.status}
                </span>
                {c.created_at && <span className="text-gray-400">{fmtDateTime(c.created_at)}</span>}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Group the event stream into turns and render each as a card. Forward-
// compatible: any non-turn events (tool/subagent/crystal/gap, once P1c emits
// them) render as inner lines under their turn.
function TurnTimeline({ events }: { events: AgentEvent[] }) {
  const turns = new Map<number, AgentEvent[]>();
  for (const e of events) {
    const k = e.turn_index ?? -1;
    if (!turns.has(k)) turns.set(k, []);
    turns.get(k)!.push(e);
  }
  const ordered = [...turns.entries()].sort((a, b) => a[0] - b[0]);
  return (
    <div className="space-y-2">
      {ordered.map(([turnIdx, evs]) => (
        <TurnCard key={turnIdx} turnIdx={turnIdx} events={evs} />
      ))}
    </div>
  );
}

function TurnCard({ turnIdx, events }: { turnIdx: number; events: AgentEvent[] }) {
  const started = events.find((e) => e.event_type === "turn_started");
  const completed = events.find((e) => e.event_type === "turn_completed");
  const failed = events.find((e) => e.event_type === "turn_failed");
  const inner = events.filter((e) => !["turn_started", "turn_completed", "turn_failed"].includes(e.event_type));

  const prompt = (started?.payload?.prompt as string | undefined) ?? started?.label ?? "";
  const tail = completed ?? failed;
  const summary = (completed?.payload?.summary as string | undefined) ?? "";
  const tools = (completed?.payload?.tools_called as string[] | undefined) ?? [];
  const files = (completed?.payload?.files_written as string[] | undefined) ?? [];
  const tokTotal = (completed?.tokens_input ?? 0) + (completed?.tokens_output ?? 0);
  const dur = tail?.duration_ms != null ? `${(tail.duration_ms / 1000).toFixed(1)}s` : null;
  const running = !completed && !failed;

  return (
    <div className="rounded-lg border border-gray-200 bg-white px-3 py-2">
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-medium uppercase tracking-wider text-gray-400">
          {turnIdx >= 0 ? `Turn ${turnIdx + 1}` : "Session"}
        </span>
        <span className="flex items-center gap-2 text-[11px] text-gray-400">
          {completed && <span>{fmtUsd(completed.cost_micro_usd)}</span>}
          {completed && tokTotal > 0 && <span>{fmtNum(tokTotal)} tok</span>}
          {dur && <span>{dur}</span>}
          {running && <span className="text-emerald-500">running…</span>}
          {failed && <span className="text-red-500">failed</span>}
        </span>
      </div>

      {prompt && <p className="mt-1 text-xs text-gray-700">▸ {truncate(prompt, 160)}</p>}

      {inner.length > 0 && (
        <div className="mt-1.5 space-y-0.5 border-l border-gray-100 pl-2">
          {inner.map((e) => (
            <EventLine key={e.id} e={e} />
          ))}
        </div>
      )}

      {(tools.length > 0 || files.length > 0) && (
        <div className="mt-1.5 flex flex-wrap items-center gap-1">
          {tools.map((t, i) => (
            <span key={i} className="rounded bg-gray-100 px-1.5 py-0.5 text-[11px] text-gray-600">{t}</span>
          ))}
          {files.length > 0 && (
            <span className="text-[11px] text-gray-400">· {files.length} file{files.length > 1 ? "s" : ""} written</span>
          )}
        </div>
      )}

      {summary && <p className="mt-1.5 text-xs text-gray-500">{truncate(summary, 220)}</p>}
      {failed && (
        <p className="mt-1 text-xs text-red-500">{truncate((failed.payload?.error as string | undefined) ?? failed.label, 180)}</p>
      )}
    </div>
  );
}

function EventLine({ e }: { e: AgentEvent }) {
  const bad = e.status === "error" || e.status === "denied";
  return (
    <div className="flex items-center gap-2 text-[11px] text-gray-500">
      <span className="text-gray-300">{e.phase ?? e.event_type}</span>
      <span className="text-gray-600">{truncate(e.label || e.event_type, 90)}</span>
      {e.status && e.status !== "ok" && (
        <span className={bad ? "text-red-500" : "text-amber-500"}>{e.status}</span>
      )}
    </div>
  );
}

// ─── Queue: daemon tasks + agent-run gaps ────────────────────────────────

function TaskPill({ status }: { status: string }) {
  const map: Record<string, string> = {
    queued: "bg-sky-50 text-sky-600",
    running: "bg-emerald-50 text-emerald-600",
    done: "bg-gray-100 text-gray-600",
    failed: "bg-red-50 text-red-600",
  };
  return (
    <span className={cn("inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-medium", map[status] ?? "bg-gray-100 text-gray-500")}>
      {status === "running" && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />}
      {status}
    </span>
  );
}

function QueueTab({ customerId }: { customerId: string }) {
  const tasks = useQuery({
    queryKey: ["agent-tasks", customerId],
    queryFn: () => api.listAgentTasks(customerId),
    refetchInterval: REFRESH_MS,
  });
  const gaps = useQuery({
    queryKey: ["agent-gaps", customerId],
    queryFn: () => api.listAgentGaps(customerId),
    refetchInterval: REFRESH_MS,
  });

  if (tasks.isError) return <ErrorBanner title="Couldn't load the queue" message={String(tasks.error)} />;

  const taskRows = tasks.data?.tasks ?? [];
  const gapRows = gaps.data?.gaps ?? [];
  const empty = !tasks.isLoading && taskRows.length === 0 && gapRows.length === 0;

  if (empty)
    return (
      <EmptyState
        title="Queue is empty"
        description="Background tasks (from --queue or the agent's queue_task tool) and the gaps a daemon retries show up here once the daemon is running. Start one with: python -m crys --daemon"
      />
    );

  return (
    <div className="space-y-6">
      <div>
        <p className="mb-1.5 text-[11px] uppercase tracking-wider text-gray-400">Background tasks</p>
        {taskRows.length === 0 ? (
          <p className="text-xs text-gray-300">No tasks queued.</p>
        ) : (
          <div className="overflow-hidden rounded-lg border border-gray-200 bg-white">
            <table className="min-w-full divide-y divide-gray-100 text-sm">
              <thead className="bg-gray-50/60">
                <tr>
                  <Th>Status</Th>
                  <Th>Task</Th>
                  <Th>Source</Th>
                  <Th>Created</Th>
                  <Th>Finished</Th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {taskRows.map((t) => (
                  <tr key={t.id} className="align-top">
                    <td className="px-4 py-2.5"><TaskPill status={t.status} /></td>
                    <td className="px-4 py-2.5">
                      <div className="text-gray-700">{truncate(t.task, 70)}</div>
                      {t.branch && <div className="font-mono text-[11px] text-gray-400">{truncate(t.branch, 40)}</div>}
                      {t.error && <div className="mt-0.5 text-[11px] text-red-500">{truncate(t.error, 90)}</div>}
                      {t.status === "done" && t.report && (
                        <div className="mt-0.5 text-[11px] text-gray-400">{truncate(t.report, 90)}</div>
                      )}
                    </td>
                    <td className="px-4 py-2.5 text-xs text-gray-500">
                      {t.source}
                      {t.recur_seconds != null && <span className="text-gray-400"> · every {t.recur_seconds}s</span>}
                    </td>
                    <td className="whitespace-nowrap px-4 py-2.5 text-xs text-gray-400">
                      {t.created_at ? fmtDateTime(t.created_at) : "—"}
                    </td>
                    <td className="whitespace-nowrap px-4 py-2.5 text-xs text-gray-400">
                      {t.finished_at ? fmtDateTime(t.finished_at) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {gapRows.length > 0 && (
        <div>
          <p className="mb-1.5 text-[11px] uppercase tracking-wider text-gray-400">Gaps (failed runs)</p>
          <div className="space-y-1.5">
            {gapRows.map((g) => (
              <div key={g.id} className="rounded-lg border border-gray-200 bg-white px-3 py-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-700">{truncate(g.subject, 80)}</span>
                  <span
                    className={cn(
                      "rounded px-1.5 py-0.5 text-[11px]",
                      g.status === "open" ? "bg-amber-50 text-amber-600" : g.status === "needs_operator" ? "bg-red-50 text-red-600" : g.status === "filled" ? "bg-emerald-50 text-emerald-600" : "bg-gray-100 text-gray-500"
                    )}
                  >
                    {g.status}
                  </span>
                </div>
                {g.created_at && <p className="mt-0.5 text-[11px] text-gray-400">{fmtDateTime(g.created_at)}</p>}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function Th({ children }: { children: ReactNode }) {
  return <th className="px-4 py-2.5 text-left text-xs font-medium uppercase tracking-wider text-gray-500">{children}</th>;
}

function Field({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div>
      <p className="text-[11px] uppercase tracking-wider text-gray-400">{label}</p>
      <p className="mt-0.5 text-sm text-gray-900">{value}</p>
    </div>
  );
}
