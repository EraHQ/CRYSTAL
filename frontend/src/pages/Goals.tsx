// Goals (2026-10-24, Q109 A). Anthony: "'goals' its just a long standing
// directive that the model addresses periodically. And consistently works
// at the next step of each goal's plans." A goal is the person's words; a
// frontier model writes its plan; a check-in on the plan's cadence files
// action items, writes drafts and (only when allowed) sends. This page is
// where the person reads the plan, approves drafts and sees what was sent.
import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Circle, Target } from "lucide-react";
import { api, errorMessage } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";
import type { Goal, GoalItem } from "@/lib/types";
import { ConfirmInline, EmptyState, ErrorBanner, PageHeader, StatusPill, Switch, useToast } from "@/components/ui";
import type { StatusVariant } from "@/components/ui";
import { cn, fmtRelative } from "@/lib/utils";
import { BTN_PRIMARY, BTN_QUIET, BTN_SECONDARY, INPUT } from "./settings/shared";

const APP_NAMES: Record<string, string> = {
  "google-mail": "Gmail", "google-calendar": "Google Calendar", "google-drive": "Google Drive", notion: "Notion", slack: "Slack",
};

export function goalState(g: Goal): { label: string; variant: StatusVariant } {
  if (g.status === "active" && !g.planned_at) return { label: "Planning", variant: "info" };
  if (g.status === "active") return { label: "Active", variant: "ok" };
  if (g.status === "paused") return { label: "Paused", variant: "neutral" };
  if (g.status === "done") return { label: "Done", variant: "vio" };
  return { label: g.status === "archived" ? "Archived" : "Draft", variant: "neutral" };
}

export function cadenceText(minutes: number | undefined): string {
  if (!minutes) return "";
  if (minutes < 60) return `every ${minutes} minutes`;
  if (minutes < 1440) { const h = Math.round(minutes / 60); return `every ${h} hour${h === 1 ? "" : "s"}`; }
  if (minutes < 10080) { const d = Math.round(minutes / 1440); return `every ${d} day${d === 1 ? "" : "s"}`; }
  return "every week";
}

/** Where a draft would go, in words. */
export function draftWhere(item: GoalItem): string {
  const t = item.target;
  const app = APP_NAMES[t.app ?? ""] ?? t.app ?? "";
  if (t.action === "send_email") return `${app} to ${t.to}${t.subject ? `, “${t.subject}”` : ""}`;
  if (t.action === "post_message") return `${app} in ${t.channel}`;
  if (t.action === "create_event") return `${app} event at ${t.start ? new Date(t.start).toLocaleString() : "?"}`;
  if (t.action === "append_page") return `${app} page ${t.page_id}`;
  return app;
}

const SEND_LABEL: Record<string, string> = {
  send_email: "Approve and send", post_message: "Approve and post", create_event: "Approve and create", append_page: "Approve and add",
};

export function NewGoal({ customerId, onDone, bare = false }: { customerId: string; onDone: () => void; bare?: boolean }) {
  const qc = useQueryClient();
  const toast = useToast();
  const [title, setTitle] = useState("");
  const [directive, setDirective] = useState("");
  const [maySend, setMaySend] = useState(false);
  const [mayPost, setMayPost] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const create = useMutation({
    mutationFn: () => api.createGoal(customerId, { title: title.trim(), directive: directive.trim(), may_send: maySend, may_post: mayPost }),
    onSuccess: () => { toast("Goal added. Its plan is written within a minute or two."); qc.invalidateQueries({ queryKey: ["goals", customerId] }); onDone(); },
    onError: (e) => setError(errorMessage(e, "Couldn't add the goal.")),
  });
  return (
    <form aria-label="New goal" className={cn("space-y-3", !bare && "rounded-xl border border-line bg-surface p-4")}
      onSubmit={(e) => { e.preventDefault(); create.mutate(); }}>
      {error && <ErrorBanner title="That didn't go through" message={error} />}
      <label className="block text-[12.5px] font-medium text-fg">Name
        <input className={INPUT + " mt-1"} value={title} maxLength={200} placeholder="Inbox under control" onChange={(e) => setTitle(e.target.value)} />
      </label>
      <label className="block text-[12.5px] font-medium text-fg">The goal, in your words
        <textarea className={INPUT + " mt-1 min-h-[96px]"} value={directive} maxLength={4000}
          placeholder="Reply to every email within 3 hours. Draft the replies for me; flag anything from customers."
          onChange={(e) => setDirective(e.target.value)} />
      </label>
      <div className="grid gap-3 min-[720px]:grid-cols-2">
        <Switch checked={maySend} onChange={setMaySend} label="May send email on its own" hint="Off: replies wait for you to approve." />
        <Switch checked={mayPost} onChange={setMayPost} label="May post and create on its own" hint="Slack posts, calendar events, Notion additions. Off: they wait for you." />
      </div>
      <p className="text-[12px] text-fg-3">Goals work through your <Link to="/settings/connections" className="text-accent hover:underline">connected apps</Link>; an app also needs Act on before anything goes out.</p>
      <div className="flex items-center gap-2">
        <button type="submit" className={BTN_PRIMARY} disabled={create.isPending || title.trim().length < 2 || directive.trim().length < 10}>Add goal</button>
        <button type="button" className={BTN_SECONDARY} onClick={onDone}>Cancel</button>
      </div>
    </form>
  );
}

function DraftCard({ item, busy, act }: { item: GoalItem; busy: boolean; act: (body: Parameters<typeof api.goalItemAction>[2]) => void }) {
  const [editing, setEditing] = useState(false);
  const [body, setBody] = useState(item.body);
  return (
    <li className="rounded-lg border border-line bg-inset px-3 py-2.5" data-testid={`draft-${item.id}`}>
      <div className="flex flex-wrap items-center gap-2">
        <StatusPill variant="warn">Draft</StatusPill>
        <span className="min-w-0 flex-1 text-[13.5px] font-medium text-fg">{item.title}</span>
      </div>
      <p className="mt-1 text-[12px] text-fg-3">{draftWhere(item)}</p>
      {editing ? (
        <textarea aria-label="Draft text" className={INPUT + " mt-2 min-h-[120px]"} value={body} onChange={(e) => setBody(e.target.value)} />
      ) : (
        <p className="mt-2 whitespace-pre-wrap text-[13px] text-fg">{item.body}</p>
      )}
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <button type="button" className={BTN_PRIMARY} disabled={busy || !body.trim()}
          onClick={() => act({ action: "send", ...(editing && body !== item.body ? { body } : {}) })}>
          {SEND_LABEL[item.target.action ?? ""] ?? "Approve and send"}
        </button>
        <button type="button" className={BTN_SECONDARY} disabled={busy} onClick={() => setEditing((v) => !v)}>{editing ? "Stop editing" : "Edit"}</button>
        <button type="button" className={BTN_QUIET} disabled={busy} onClick={() => act({ action: "dismiss" })}>Dismiss</button>
      </div>
    </li>
  );
}

function GoalDetail({ customerId, goal }: { customerId: string; goal: Goal }) {
  const qc = useQueryClient();
  const toast = useToast();
  const [error, setError] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const detail = useQuery({ queryKey: ["goal", customerId, goal.id], queryFn: () => api.getGoal(customerId, goal.id), refetchInterval: 30_000 });
  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["goal", customerId, goal.id] });
    qc.invalidateQueries({ queryKey: ["goals", customerId] });
  };
  const update = useMutation({
    mutationFn: (body: Parameters<typeof api.updateGoal>[2]) => api.updateGoal(customerId, goal.id, body),
    onSuccess: () => { setError(null); refresh(); },
    onError: (e) => setError(errorMessage(e, "Couldn't change the goal.")),
  });
  const replan = useMutation({
    mutationFn: () => api.replanGoal(customerId, goal.id),
    onSuccess: () => { setError(null); toast("A fresh plan is on its way. Finished steps stay finished."); refresh(); },
    onError: (e) => setError(errorMessage(e, "Couldn't ask for a new plan.")),
  });
  const runNow = useMutation({
    mutationFn: () => api.runGoalNow(customerId, goal.id),
    onSuccess: () => { setError(null); toast("It checks in within a minute."); refresh(); },
    onError: (e) => setError(errorMessage(e, "Couldn't start a check-in.")),
  });
  const remove = useMutation({
    mutationFn: () => api.deleteGoal(customerId, goal.id),
    onSuccess: () => { toast("Goal deleted."); qc.invalidateQueries({ queryKey: ["goals", customerId] }); },
    onError: (e) => setError(errorMessage(e, "Couldn't delete the goal.")),
  });
  const item = useMutation({
    mutationFn: ({ id, body }: { id: string; body: Parameters<typeof api.goalItemAction>[2] }) => api.goalItemAction(customerId, id, body),
    onSuccess: (_o, v) => {
      setError(null);
      toast(v.body.action === "send" ? "Sent." : v.body.action === "done" ? "Marked done." : v.body.action === "reopen" ? "Reopened." : "Dismissed.");
      refresh();
    },
    onError: (e) => setError(errorMessage(e, "That didn't go through.")),
  });
  const g = detail.data?.goal ?? goal;
  const items = detail.data?.items ?? [];
  const open = items.filter((i) => i.status === "open");
  const notes = open.filter((i) => i.kind === "note");
  const drafts = open.filter((i) => i.kind === "draft");
  const actions = open.filter((i) => i.kind === "action");
  const history = items.filter((i) => i.status !== "open");
  const plan = g.plan;
  const busy = update.isPending || replan.isPending || runNow.isPending || remove.isPending || item.isPending;
  const missing = (plan?.needs.apps ?? []).map((a) => APP_NAMES[a] ?? a);
  return (
    <div className="mt-3 space-y-4 border-t border-line pt-3">
      {error && <ErrorBanner title="That didn't go through" message={error} />}
      {detail.isError && <ErrorBanner title="Couldn't load the goal" error={detail.error} onRetry={() => detail.refetch()} />}
      <p className="whitespace-pre-wrap text-[13px] text-fg-2">“{g.directive}”</p>

      {notes.length > 0 && (
        <div className="rounded-lg border border-warn/40 bg-warn-soft px-3 py-2.5" role="note" aria-label="Needs you">
          <p className="text-[12.5px] font-medium text-fg">This goal needs you</p>
          <ul className="mt-1 space-y-1">
            {notes.map((n) => (
              <li key={n.id} className="flex items-start gap-2 text-[13px] text-fg">
                <span className="min-w-0 flex-1">{n.title}</span>
                <button type="button" className={BTN_QUIET} disabled={busy} onClick={() => item.mutate({ id: n.id, body: { action: "done" } })}>Handled</button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {(drafts.length > 0 || actions.length > 0) && (
        <div>
          <h3 className="text-[13px] font-semibold text-fg">Waiting for you</h3>
          <ul className="mt-2 space-y-2" aria-label="Drafts">
            {drafts.map((d) => (
              <DraftCard key={d.id} item={d} busy={busy} act={(body) => item.mutate({ id: d.id, body })} />
            ))}
          </ul>
          <ul className="mt-2 space-y-1.5" aria-label="Action items">
            {actions.map((a) => (
              <li key={a.id} className="flex flex-wrap items-start gap-2 rounded-lg border border-line px-3 py-2">
                <div className="min-w-0 flex-1">
                  <p className="text-[13px] text-fg">{a.title}</p>
                  {a.body && <p className="text-[12px] text-fg-2">{a.body}</p>}
                  {a.due_at && <p className={cn("text-[12px]", new Date(a.due_at).getTime() < Date.now() ? "text-bad" : "text-fg-3")}>Due {fmtRelative(a.due_at)}</p>}
                </div>
                <button type="button" className={BTN_SECONDARY} disabled={busy} onClick={() => item.mutate({ id: a.id, body: { action: "done" } })}>Done</button>
                <button type="button" className={BTN_QUIET} disabled={busy} onClick={() => item.mutate({ id: a.id, body: { action: "dismiss" } })}>Dismiss</button>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div>
        <h3 className="text-[13px] font-semibold text-fg">Plan</h3>
        {!plan ? (
          <p className="mt-1 text-[13px] text-fg-3">{g.status === "active" ? "Being written now." : "No plan yet."}</p>
        ) : (
          <div className="mt-1 space-y-2">
            <p className="text-[13px] text-fg">{plan.summary}</p>
            <p className="text-[12.5px] text-fg-2"><span className="font-medium text-fg">Working when:</span> {plan.success_measure}</p>
            <ol className="space-y-1" aria-label="Steps">
              {plan.steps.map((s) => (
                <li key={s.n} className="flex items-start gap-2 text-[13px]">
                  {s.done ? <CheckCircle2 aria-label="Done" className="mt-0.5 h-4 w-4 shrink-0 text-ok" /> : <Circle aria-label="Not done" className="mt-0.5 h-4 w-4 shrink-0 text-fg-3" />}
                  <span className="min-w-0">
                    <span className={cn("text-fg", s.done && "text-fg-2")}>{s.title}</span>
                    {s.detail && <span className="block text-[12px] text-fg-3">{s.detail}</span>}
                    {s.note && <span className="block text-[12px] text-fg-2">{s.note}</span>}
                  </span>
                </li>
              ))}
            </ol>
            {plan.standing_instructions && <p className="text-[12.5px] text-fg-2"><span className="font-medium text-fg">Always:</span> {plan.standing_instructions}</p>}
            <p className="text-[12px] text-fg-3">Checks in {cadenceText(plan.check_every_minutes)}{missing.length ? `. Uses ${missing.join(", ")}.` : "."}</p>
          </div>
        )}
      </div>

      {g.last_summary && (
        <div>
          <h3 className="text-[13px] font-semibold text-fg">Last check-in{g.last_run_at ? `, ${fmtRelative(g.last_run_at)}` : ""}</h3>
          <p className="mt-1 whitespace-pre-wrap text-[13px] text-fg-2">{g.last_summary}</p>
          {g.last_task_id && <Link to="/activity?view=runs" className="text-[12px] text-accent hover:underline">See the run in Activity</Link>}
        </div>
      )}

      <div className="grid gap-3 min-[720px]:grid-cols-2" role="group" aria-label="Permissions">
        <Switch checked={g.may_send} disabled={busy} onChange={(v) => update.mutate({ may_send: v })} label="May send email on its own"
          hint={g.may_send ? "Sends replies that follow its instructions; every send is recorded." : "Replies wait for you to approve."} />
        <Switch checked={g.may_post} disabled={busy} onChange={(v) => update.mutate({ may_post: v })} label="May post and create on its own"
          hint={g.may_post ? "Posts, events and page additions go out; each is recorded." : "They wait for you to approve."} />
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {g.status === "active" && <button type="button" className={BTN_SECONDARY} disabled={busy || !g.planned_at} onClick={() => runNow.mutate()}>Check in now</button>}
        {g.status !== "archived" && g.status !== "done" && <button type="button" className={BTN_SECONDARY} disabled={busy} onClick={() => replan.mutate()}>New plan</button>}
        {g.status === "active" && <button type="button" className={BTN_QUIET} disabled={busy} onClick={() => update.mutate({ status: "paused" })}>Pause</button>}
        {(g.status === "paused" || g.status === "done" || g.status === "archived") && (
          <button type="button" className={BTN_QUIET} disabled={busy} onClick={() => update.mutate({ status: "active" })}>Resume</button>
        )}
        {g.status !== "done" && <button type="button" className={BTN_QUIET} disabled={busy} onClick={() => update.mutate({ status: "done" })}>Mark done</button>}
        <span className="flex-1" />
        <button type="button" className="text-[12.5px] text-fg-3 hover:text-bad disabled:opacity-50" disabled={busy} onClick={() => setConfirmDelete(true)}>Delete</button>
      </div>
      <ConfirmInline open={confirmDelete} permanent busy={remove.isPending} confirmLabel="Delete goal"
        message="Delete this goal and everything it filed? What it already sent stays sent." onConfirm={() => remove.mutate()} onCancel={() => setConfirmDelete(false)} />

      {history.length > 0 && (
        <div>
          <button type="button" className={BTN_QUIET} aria-expanded={showHistory} onClick={() => setShowHistory((v) => !v)}>
            {showHistory ? "Hide history" : `History (${history.length})`}
          </button>
          {showHistory && (
            <ul className="mt-2 space-y-1.5" aria-label="History">
              {history.map((h) => (
                <li key={h.id} className="rounded-lg border border-line px-3 py-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <StatusPill variant={h.status === "sent" ? "ok" : h.status === "failed" ? "bad" : "neutral"}>
                      {h.status === "sent" ? "Sent" : h.status === "failed" ? "Failed" : h.status === "done" ? "Done" : "Dismissed"}
                    </StatusPill>
                    <span className="min-w-0 flex-1 text-[13px] text-fg">{h.title}</span>
                    <span className="text-[11.5px] text-fg-3">{h.updated_at ? fmtRelative(h.updated_at) : ""}</span>
                    {(h.status === "done" || h.status === "dismissed") && (
                      <button type="button" className={BTN_QUIET} disabled={busy} onClick={() => item.mutate({ id: h.id, body: { action: "reopen" } })}>Reopen</button>
                    )}
                  </div>
                  {h.kind === "draft" && <p className="mt-1 text-[12px] text-fg-3">{draftWhere(h)}{h.target.sent_by ? ` · sent by ${h.target.sent_by.startsWith("goal:") ? "the goal" : "you"}` : ""}</p>}
                  {h.status === "sent" && <p className="mt-1 whitespace-pre-wrap text-[12.5px] text-fg-2">{h.body}</p>}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

export function GoalsPage() {
  const { selectedCustomerId } = useSelectedCustomer();
  const [params] = useSearchParams();
  const [adding, setAdding] = useState(false);
  // ?goal=<id> (from the Command center) opens that goal.
  const [openId, setOpenId] = useState<string | null>(params.get("goal"));
  const list = useQuery({
    queryKey: ["goals", selectedCustomerId],
    queryFn: () => api.listGoals(selectedCustomerId!),
    enabled: !!selectedCustomerId,
    refetchInterval: 60_000,
  });
  if (!selectedCustomerId) {
    return (
      <>
        <PageHeader title="Goals" />
        <EmptyState icon={Target} title="No customer selected" help="Pick a customer above to see their goals." />
      </>
    );
  }
  const goals = list.data?.goals ?? [];
  const active = goals.filter((g) => g.status === "active").length;
  return (
    <div className="space-y-4">
      <PageHeader title="Goals" meta={list.isLoading ? "Loading…" : `${active} active`}
        actions={!adding && <button type="button" className={BTN_PRIMARY} onClick={() => setAdding(true)}>New goal</button>} />
      {adding && <NewGoal customerId={selectedCustomerId} onDone={() => setAdding(false)} />}
      {list.isError && <ErrorBanner title="Couldn't load goals" error={list.error} onRetry={() => list.refetch()} />}
      {!list.isLoading && !list.isError && goals.length === 0 && !adding && (
        <EmptyState icon={Target} title="No goals yet"
          help="A goal is a standing instruction, like “reply to every email within 3 hours”. Crystal plans it, checks in on it, and drafts the work." />
      )}
      <ul className="space-y-3" aria-label="Goals">
        {goals.map((g) => {
          const st = goalState(g);
          const isOpen = openId === g.id;
          const waiting = (g.open?.draft ?? 0) + (g.open?.action ?? 0) + (g.open?.note ?? 0);
          return (
            <li key={g.id} className="rounded-xl border border-line bg-surface p-4">
              <button type="button" className="flex w-full flex-wrap items-center gap-2 text-left" aria-expanded={isOpen}
                onClick={() => setOpenId(isOpen ? null : g.id)}>
                <span className="min-w-0 flex-1 text-[15px] font-semibold text-fg">{g.title}</span>
                {waiting > 0 && <StatusPill variant="warn">{waiting} waiting</StatusPill>}
                <StatusPill variant={st.variant} dot>{st.label}</StatusPill>
              </button>
              <p className="mt-1 text-[12.5px] text-fg-2">
                {g.plan?.summary ?? g.directive.slice(0, 160)}
              </p>
              <p className="mt-0.5 text-[12px] text-fg-3">
                {g.last_run_at ? `Last check-in ${fmtRelative(g.last_run_at)}` : "Not checked in yet"}
                {g.plan ? ` · ${cadenceText(g.plan.check_every_minutes)}` : ""}
              </p>
              {isOpen && <GoalDetail customerId={selectedCustomerId} goal={g} />}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
