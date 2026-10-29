// Goals (2026-10-24, Q109 A). Anthony: "'goals' its just a long standing
// directive that the model addresses periodically. And consistently works
// at the next step of each goal's plans." A goal is the person's words; a
// frontier model writes its plan; a check-in on the plan's cadence files
// action items, writes drafts and (only when allowed) sends. This page is
// where the person reads the plan, approves drafts and sees what was sent.
import { useState } from "react";
import { Link, Navigate, useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Target } from "lucide-react";
import { api, errorMessage } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";
import type { Goal, GoalItem } from "@/lib/types";
import { EmptyState, ErrorBanner, FilterChips, PageHeader, StatusPill, Switch, useToast } from "@/components/ui";
import type { StatusVariant } from "@/components/ui";
import { cn, fmtRelative } from "@/lib/utils";
import { BTN_PRIMARY, BTN_QUIET, BTN_SECONDARY, ChoiceCard, INPUT } from "./settings/shared";

export const APP_NAMES: Record<string, string> = {
  "google-mail": "Gmail", "google-calendar": "Google Calendar", "google-drive": "Google Drive", notion: "Notion", slack: "Slack",
  "google-sheet": "Google Sheets",
};

export function goalState(g: Goal): { label: string; variant: StatusVariant } {
  if (g.status === "active" && !g.planned_at) return { label: "Planning", variant: "info" };
  if (g.status === "deciding") return { label: "Needs your answers", variant: "warn" };
  if (g.status === "needs_setup") return { label: "Needs setup", variant: "warn" };
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
  if (t.action === "create_spreadsheet") return `${app}: a new spreadsheet, “${item.title}”`;
  if (t.action === "append_rows") return `${app}: rows into ${t.sheet ? `“${t.sheet}” in ` : ""}spreadsheet ${t.spreadsheet_id}`;
  return app;
}

export const MISSING_LABEL: Record<string, string> = {
  app: "App", data_source: "Data source", tool: "Ability", permission: "Permission", info: "Your answer",
};

const SEND_LABEL: Record<string, string> = {
  send_email: "Approve and send", post_message: "Approve and post", create_event: "Approve and create", append_page: "Approve and add",
  create_spreadsheet: "Approve and create", append_rows: "Approve and add rows",
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

/** Planning step 2: the planner's questions, one choice (or your own words) each. */
export function Decisions({ customerId, goal, onError }: { customerId: string; goal: Goal; onError: (m: string | null) => void }) {
  const qc = useQueryClient();
  const toast = useToast();
  const decisions = goal.plan?.decisions ?? [];
  const [picks, setPicks] = useState<Record<number, number | "other">>(
    () => Object.fromEntries(decisions.map((d, i) => [i, d.recommended])));
  const [others, setOthers] = useState<Record<number, string>>({});
  const decide = useMutation({
    mutationFn: () => api.decideGoal(customerId, goal.id, decisions.map((_d, i) => (
      picks[i] === "other" ? { index: i, other: (others[i] ?? "").trim() } : { index: i, choice: picks[i] as number }))),
    onSuccess: () => {
      onError(null);
      toast("Thanks. The plan is written on your answers within a minute or two.");
      qc.invalidateQueries({ queryKey: ["goal", customerId, goal.id] });
      qc.invalidateQueries({ queryKey: ["goals", customerId] });
    },
    onError: (e) => onError(errorMessage(e, "Couldn't save your answers.")),
  });
  const ready = decisions.every((_d, i) => picks[i] !== "other" || (others[i] ?? "").trim().length > 0);
  return (
    <form aria-label="Questions before the plan" className="space-y-4 rounded-lg border border-warn/40 bg-warn-soft px-3 py-3"
      onSubmit={(e) => { e.preventDefault(); decide.mutate(); }}>
      <p className="text-[12.5px] font-medium text-fg">Before it plans, this goal needs a few answers from you.</p>
      {decisions.map((d, i) => (
        <fieldset key={i} className="space-y-1.5">
          <legend className="text-[13.5px] font-medium text-fg">{d.question}</legend>
          {d.why && <p className="text-[12px] text-fg-2">{d.why}</p>}
          <div role="radiogroup" aria-label={d.question} className="grid gap-1.5 min-[720px]:grid-cols-2">
            {d.options.map((o, j) => (
              <ChoiceCard key={j} checked={picks[i] === j} onPick={() => setPicks({ ...picks, [i]: j })}
                title={o.label} body={o.detail || " "} aside={j === d.recommended ? <StatusPill variant="info">Recommended</StatusPill> : undefined} />
            ))}
            <ChoiceCard checked={picks[i] === "other"} onPick={() => setPicks({ ...picks, [i]: "other" })}
              title="Something else" body="Say it in your own words." />
          </div>
          {picks[i] === "other" && (
            <input aria-label={`Your answer: ${d.question}`} className={INPUT} maxLength={400} value={others[i] ?? ""}
              onChange={(e) => setOthers({ ...others, [i]: e.target.value })} />
          )}
        </fieldset>
      ))}
      <button type="submit" className={BTN_PRIMARY} disabled={decide.isPending || !ready}>Plan with these answers</button>
    </form>
  );
}

export function DraftCard({ item, busy, act }: { item: GoalItem; busy: boolean; act: (body: Parameters<typeof api.goalItemAction>[2]) => void }) {
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

type Filter = "all" | "waiting" | "active" | "paused" | "done";

function waitingOnYou(g: Goal): number {
  const items = (g.open?.draft ?? 0) + (g.open?.action ?? 0) + (g.open?.note ?? 0);
  const review = (g.plan?.steps ?? []).filter((s) => s.state === "awaiting_review").length;
  return items + review + (g.status === "deciding" || g.status === "needs_setup" ? 1 : 0);
}

/** One line on where a goal stands. */
export function whereItStands(g: Goal): { line: string; detail: string } {
  if (g.status === "deciding") return { line: `${(g.plan?.decisions ?? []).filter((d) => !d.answer).length} questions before the plan is written`, detail: "Planning" };
  if (!g.plan || !g.plan.steps.length) return { line: g.status === "active" ? "The plan is being written" : "No plan yet", detail: "" };
  const finite = g.plan.steps.filter((s) => !s.repeats);
  const cur = finite.find((s) => !s.done);
  if (!cur) {
    const rep = g.plan.steps.find((s) => s.repeats);
    return { line: rep ? `Repeating · ${rep.title}` : "Every step accepted", detail: `${finite.length} / ${finite.length} steps` };
  }
  const state = cur.state === "awaiting_review" ? "waits for your approval" : cur.state === "rejected" ? "rejected" : "";
  return { line: `Step ${cur.n} of ${finite.length}${state ? ` ${state}` : ""} · ${cur.title}`,
           detail: `${finite.filter((s) => s.done).length} / ${finite.length} steps` };
}

export function GoalsPage() {
  const { selectedCustomerId } = useSelectedCustomer();
  const [params] = useSearchParams();
  const [adding, setAdding] = useState(false);
  const [filter, setFilter] = useState<Filter>("all");
  const list = useQuery({
    queryKey: ["goals", selectedCustomerId],
    queryFn: () => api.listGoals(selectedCustomerId!),
    enabled: !!selectedCustomerId,
    refetchInterval: 60_000,
  });
  // ?goal=<id> (older links, the Command center) opens that goal's page.
  const linked = params.get("goal");
  if (linked) return <Navigate to={`/goals/${encodeURIComponent(linked)}`} replace />;
  if (!selectedCustomerId) {
    return (
      <>
        <PageHeader title="Goals" />
        <EmptyState icon={Target} title="No customer selected" help="Pick a customer above to see their goals." />
      </>
    );
  }
  const goals = list.data?.goals ?? [];
  const count = (f: Filter) => goals.filter((g) => match(g, f)).length;
  function match(g: Goal, f: Filter): boolean {
    if (f === "waiting") return waitingOnYou(g) > 0;
    if (f === "active") return g.status === "active";
    if (f === "paused") return g.status === "paused" || g.status === "needs_setup";
    if (f === "done") return g.status === "done" || g.status === "archived";
    return true;
  }
  const shown = goals.filter((g) => match(g, filter));
  return (
    <div className="space-y-4">
      <PageHeader title="Goals" meta={list.isLoading ? "Loading…" : "Standing work Crystal plans, checks in on, and proves done."}
        actions={!adding && <button type="button" className={BTN_PRIMARY} onClick={() => setAdding(true)}>New goal</button>} />
      {adding && <NewGoal customerId={selectedCustomerId} onDone={() => setAdding(false)} />}
      {list.isError && <ErrorBanner title="Couldn't load goals" error={list.error} onRetry={() => list.refetch()} />}
      {!list.isLoading && !list.isError && goals.length === 0 && !adding && (
        <EmptyState icon={Target} title="No goals yet"
          help="A goal is a standing instruction, like “reply to every email within 3 hours”. Crystal plans it, checks in on it, and drafts the work." />
      )}
      {goals.length > 0 && (
        <FilterChips label="Show" value={filter} onChange={(v) => setFilter((v as Filter) ?? "all")}
          options={([["all", "All"], ["waiting", "Waiting on you"], ["active", "Active"], ["paused", "Paused"], ["done", "Done"]] as const)
            .map(([value, label]) => ({ value, label, count: count(value) }))} />
      )}
      <ul className="space-y-3" aria-label="Goals">
        {shown.map((g) => {
          const st = goalState(g);
          const waiting = waitingOnYou(g);
          const where = whereItStands(g);
          return (
            <li key={g.id}>
              <Link to={`/goals/${encodeURIComponent(g.id)}`} className="block rounded-xl border border-line bg-surface p-4 hover:border-accent/60 hover:bg-inset">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="min-w-0 flex-1 text-[15px] font-semibold text-fg">{g.title}</span>
                  {waiting > 0 && <StatusPill variant="warn">{waiting} waiting on you</StatusPill>}
                  <StatusPill variant={st.variant} dot>{st.label}</StatusPill>
                </div>
                <p className="mt-1 text-[13px] text-fg-2">{where.line}</p>
                <p className="mt-0.5 text-[12px] text-fg-3">
                  {[where.detail, g.last_run_at ? `Last check-in ${fmtRelative(g.last_run_at)}` : "Not checked in yet",
                    g.status === "active" && g.plan ? cadenceText(g.plan.check_every_minutes) : g.status === "paused" ? "Next: paused" : ""]
                    .filter(Boolean).join(" · ")}
                </p>
              </Link>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
