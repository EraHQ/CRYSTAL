// Cognition — Console Refine PR 6 (docs/CONSOLE_UX_AUDIT.md 5.4, bug 7.1).
//
// One page: a header that says what is working and what needs you, an
// "AI capacity today" card for tenants (admins keep the dollar panel), a
// full-width runs table that opens a Drawer at /cognition/run/:runId,
// ONE knowledge-gaps list where every row has its verb, and the review
// queue in the same words Assumptions uses. No dollars for tenants, no
// fixed-height inner scroll, and the gaps list reads one query shape
// (bug 7.1: two shapes shared one query key, so the rail was empty).
import { useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Brain, CheckCircle2, Lightbulb, Search, XCircle } from "lucide-react";
import { api, errorMessage } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useSelectedCustomer } from "@/lib/selected-customer";
import {
  ConfirmInline, EmptyState, ErrorBanner, FilterChips, PageHeader, Skeleton, StatusPill, Switch, useToast,
} from "@/components/ui";
import type { StatusVariant } from "@/components/ui";
import type { CognitionRun, DailyCost, KnowledgeGap, PushQueueItem } from "@/lib/types";
import { cn, fmtRelative } from "@/lib/utils";
import { activeStage } from "@/components/cognition/PipelineStrip";
import { RunDrawer, isActiveStatus, runStatus } from "./CognitionTracker";

// ---------------------------------------------------------- capacity card

// Spend origins (llm_cost_ledger.origin) grouped into the five activities
// a tenant recognizes; anything new lands in "Other" rather than vanishing.
const ACTIVITY_OF: Record<string, string> = {
  direct: "Answering your tools", agent: "Answering your tools", tool_loop: "Answering your tools", inline_research: "Answering your tools",
  code_descriptions: "Reading files", sparse_keys: "Reading files", schema_inference: "Reading files",
  shadow_critic: "Curation", shadow_eval: "Curation", self_critique: "Curation", scan_dedup: "Curation", scan_contradiction: "Curation",
  scan_gap_discovery: "Curation", failure_learning: "Curation", meta_reflection: "Curation", outbound_review: "Curation",
  cognition: "Research", precondition_verify: "Research",
  assumptions: "Assumptions", assumption_verification: "Assumptions",
};
const ACTIVITIES = ["Answering your tools", "Reading files", "Curation", "Research", "Assumptions", "Other"];
const ACTIVITY_FILL = ["bg-accent", "bg-info", "bg-vio", "bg-warn", "bg-ok", "bg-line-strong"];

export function activityShares(cost: DailyCost | undefined): Array<{ label: string; fraction: number }> {
  if (!cost || cost.total_micro_usd <= 0) return [];
  const by = new Map<string, number>();
  for (const f of cost.families) {
    const k = ACTIVITY_OF[f.origin] ?? "Other";
    by.set(k, (by.get(k) ?? 0) + f.cost_micro_usd);
  }
  return ACTIVITIES.filter((a) => by.has(a)).map((a) => ({ label: a, fraction: (by.get(a) ?? 0) / cost.total_micro_usd }));
}

function CapacityCard({ customerId, pct }: { customerId: string; pct: number | null }) {
  const cost = useQuery({ queryKey: ["daily-cost", customerId], queryFn: () => api.dailyCost(customerId), refetchInterval: 60000 });
  const shares = activityShares(cost.data);
  const used = pct ?? 0;
  return (
    <section aria-labelledby="capacity-title" className="rounded-xl border border-line bg-surface p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 id="capacity-title" className="text-[13px] font-medium text-fg-2">AI capacity today</h2>
        <span className="text-[12px] text-fg-3">resets at midnight UTC</span>
      </div>
      <p className="mt-1 text-[32px] font-[650] leading-none tracking-tight text-fg">{pct == null ? "—" : `${pct}%`}</p>
      <div role="img" aria-label={`${used}% of today's capacity used`} className="mt-3 flex h-2 w-full overflow-hidden rounded-full bg-inset">
        {shares.map((s, i) => (
          <span key={s.label} title={`${s.label} · ${Math.round(s.fraction * 100)}% of what was used`}
            className={cn("h-full", ACTIVITY_FILL[ACTIVITIES.indexOf(s.label)] ?? ACTIVITY_FILL[5])}
            style={{ width: `${used * s.fraction}%`, marginLeft: i === 0 ? 0 : 1 }} />
        ))}
      </div>
      {shares.length > 0 && (
        <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[12px] text-fg-2">
          {shares.map((s) => (
            <li key={s.label} className="inline-flex items-center gap-1.5">
              <span aria-hidden className={cn("h-2 w-2 rounded-full", ACTIVITY_FILL[ACTIVITIES.indexOf(s.label)])} />
              {s.label} {Math.round(s.fraction * 100)}%
            </li>
          ))}
        </ul>
      )}
      <p className="mt-2 text-[12px] text-fg-3">
        Background research stops on its own when today's capacity is used up. Recall from your tools never spends capacity, so it keeps working.
      </p>
    </section>
  );
}

function SpendPanel({ customerId }: { customerId: string }) {
  // Admins only (RC-18, Q4=A): dollars per spend family.
  const spend = useQuery({ queryKey: ["daily-cost", customerId], queryFn: () => api.dailyCost(customerId), refetchInterval: 60000 });
  const d = spend.data;
  if (!d || d.families.length === 0) return null;
  const usd = (m: number) => `$${(m / 1_000_000).toFixed(2)}`;
  const budget = d.budget_per_customer_usd;
  const over = budget != null && d.total_micro_usd / 1_000_000 >= budget;
  return (
    <section className="rounded-xl border border-line bg-surface px-4 py-3">
      <div className="flex flex-wrap items-center gap-2 text-[13px]">
        <span className="font-medium text-fg-2">LLM spend today (admin)</span>
        <span className={cn("font-semibold", over ? "text-bad" : "text-fg")}>{usd(d.total_micro_usd)}</span>
        {budget != null && <span className={cn("text-[12px]", over ? "text-bad" : "text-fg-3")}>/ ${budget.toFixed(2)} daily budget{over ? " · background paused" : ""}</span>}
      </div>
      <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-0.5 text-[12px] text-fg-3">
        {d.families.map((f) => (
          <span key={f.origin}><span className="font-mono">{f.origin}</span> <span className="text-fg-2">{usd(f.cost_micro_usd)}</span> ({f.calls})</span>
        ))}
      </div>
    </section>
  );
}

// -------------------------------------------------------------- runs table

type RunFilter = "all" | "working" | "needs_you" | "done" | "gave_up";
const RUN_FILTERS: Array<{ value: RunFilter; label: string; match: (s: string) => boolean }> = [
  { value: "all", label: "All", match: () => true },
  { value: "working", label: "Working", match: isActiveStatus },
  { value: "needs_you", label: "Needs you", match: (s) => s === "needs_human_review" },
  { value: "done", label: "Done", match: (s) => s === "complete" },
  { value: "gave_up", label: "Gave up", match: (s) => s === "failed" || s === "cancelled" },
];

function StageBar({ run }: { run: CognitionRun }) {
  // Four segments (Orchestrate, Research, Compose, Validate): filled up to
  // the active stage; all four for a finished run, none for a stopped one.
  const done = run.status === "complete";
  const stage = done ? 4 : isActiveStatus(run.status) ? activeStage(run.status, {}) + 1 : run.status === "needs_human_review" || run.status === "failed" ? 4 : 0;
  return (
    <span role="img" aria-label={done ? "all stages done" : `stage ${stage} of 4`} className="inline-flex gap-0.5">
      {[0, 1, 2, 3].map((i) => (
        <span key={i} className={cn("h-1.5 w-4 rounded-sm", i < stage ? (done ? "bg-ok" : i === stage - 1 && isActiveStatus(run.status) ? "bg-accent animate-pulse" : "bg-accent") : "bg-inset")} />
      ))}
    </span>
  );
}

function RunsTable({ runs, isLoading, error, onRetry, onOpen, isAdmin }: {
  runs: CognitionRun[]; isLoading: boolean; error: unknown; onRetry: () => void; onOpen: (id: string) => void; isAdmin: boolean;
}) {
  const [filter, setFilter] = useState<RunFilter>("all");
  const counts = useMemo(() => Object.fromEntries(RUN_FILTERS.map((f) => [f.value, runs.filter((r) => f.match(r.status)).length])) as Record<RunFilter, number>, [runs]);
  const shown = runs.filter((r) => RUN_FILTERS.find((f) => f.value === filter)!.match(r.status));
  return (
    <section aria-labelledby="runs-title">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <h2 id="runs-title" className="text-[15px] font-semibold text-fg">Runs</h2>
        <FilterChips label="Run status" value={filter} onChange={(v) => setFilter((v as RunFilter) ?? "all")}
          options={RUN_FILTERS.map((f) => ({ value: f.value, label: f.label, count: counts[f.value] }))} />
      </div>
      {isLoading && <Skeleton rows={4} />}
      {!!error && <ErrorBanner title="Couldn't load runs" error={error} onRetry={onRetry} />}
      {!isLoading && !error && runs.length === 0 && (
        <EmptyState icon={Brain} title="No runs yet" help="A run starts when a knowledge gap is researched or a task is queued. Finished runs stay here." />
      )}
      {!isLoading && runs.length > 0 && shown.length === 0 && <p className="py-4 text-center text-[12.5px] text-fg-3">No runs in this state.</p>}
      {shown.length > 0 && (
        <ul className="divide-y divide-line rounded-xl border border-line bg-surface">
          {shown.map((r) => {
            const st = runStatus(r.status);
            const notes = r.open_critiques ?? 0;
            return (
              <li key={r.id}>
                <button type="button" onClick={() => onOpen(r.id)} className="flex w-full flex-wrap items-center gap-x-3 gap-y-1 px-3 py-2.5 text-left hover:bg-inset">
                  <StatusPill variant={st.variant} raw={r.status} dot={isActiveStatus(r.status)} className="shrink-0">{st.label}</StatusPill>
                  <span className="min-w-0 flex-1 truncate text-[13.5px] font-medium text-fg">{r.goal_title || r.trigger_type}</span>
                  <span className="text-[12px] text-fg-3">
                    attempt {r.attempts}/{r.max_attempts}{(r.cycle ?? 1) > 1 ? ` · cycle ${r.cycle}/${r.cycle_cap ?? 3}` : ""} · {fmtRelative(r.created_at)}{notes ? ` · ${notes} note${notes === 1 ? "" : "s"}` : ""}
                  </span>
                  <StageBar run={r} />
                  <span className="w-20 text-right text-[12px] tabular-nums text-fg-3">
                    {isAdmin ? `$${(r.cost_usd ?? 0).toFixed(3)}` : r.capacity_share_pct != null ? `${r.capacity_share_pct}%` : ""}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}

// ------------------------------------------------------------------- gaps

type GapFilter = "all" | "needs_you" | "researching" | "filled";
const DISPOSITION: Record<string, string> = {
  researchable: "Can be researched", workable: "Needs a judgment call", needs_document: "Needs a document", cycles_exhausted: "Research gave up",
};

export function gapNeedsYou(g: KnowledgeGap): boolean {
  return g.status === "open" && (g.disposition === "needs_document" || g.disposition === "workable" || g.disposition === "cycles_exhausted");
}

function GapsList({ customerId, gaps, isLoading, error, onRetry, researching }: {
  customerId: string; gaps: KnowledgeGap[]; isLoading: boolean; error: unknown; onRetry: () => void; researching: Map<string, string>;
}) {
  const [filter, setFilter] = useState<GapFilter>("all");
  const isResearching = (g: KnowledgeGap) => g.status === "open" && (researching.has(g.id) || (g.last_run_status != null && isActiveStatus(g.last_run_status)));
  const matches = (g: KnowledgeGap, f: GapFilter) =>
    f === "all" ? g.status !== "closed" : f === "needs_you" ? gapNeedsYou(g) && !isResearching(g) : f === "researching" ? isResearching(g) : g.status === "filled";
  const counts = { all: gaps.filter((g) => matches(g, "all")).length, needs_you: gaps.filter((g) => matches(g, "needs_you")).length, researching: gaps.filter((g) => matches(g, "researching")).length, filled: gaps.filter((g) => matches(g, "filled")).length };
  const shown = gaps.filter((g) => matches(g, filter));
  return (
    <section aria-labelledby="gaps-title">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 id="gaps-title" className="text-[15px] font-semibold text-fg">Knowledge gaps</h2>
          <p className="text-[12.5px] text-fg-3">Questions your tools asked that memory could not answer.</p>
        </div>
        <FilterChips label="Gap state" value={filter} onChange={(v) => setFilter((v as GapFilter) ?? "all")}
          options={[{ value: "all", label: "All", count: counts.all }, { value: "needs_you", label: "Needs you", count: counts.needs_you },
            { value: "researching", label: "Researching", count: counts.researching }, { value: "filled", label: "Filled", count: counts.filled }]} />
      </div>
      {isLoading && <Skeleton rows={3} />}
      {!!error && <ErrorBanner title="Couldn't load gaps" error={error} onRetry={onRetry} />}
      {!isLoading && !error && shown.length === 0 && (
        <EmptyState icon={Search} title={gaps.length ? "Nothing here" : "No gaps yet"} help={gaps.length ? "Nothing matches this filter." : "Gaps appear when a tool asks something memory cannot answer."} />
      )}
      <div className="space-y-2">
        {shown.map((g) => <GapRow key={g.id} customerId={customerId} gap={g} researching={isResearching(g) ? (researching.get(g.id) ?? "running") : null} />)}
      </div>
    </section>
  );
}

function GapRow({ customerId, gap, researching }: { customerId: string; gap: KnowledgeGap; researching: string | null }) {
  const qc = useQueryClient();
  const toast = useToast();
  const [answering, setAnswering] = useState(false);
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["knowledge-gaps", customerId] });
    qc.invalidateQueries({ queryKey: ["cognition-tasks", customerId] });
  };
  const research = useMutation({
    mutationFn: () => api.promoteGapToResearch(gap.id),
    onSuccess: () => { setError(null); toast("Queued for research."); refresh(); },
    onError: (e) => setError(errorMessage(e, "Couldn't queue research.")),
  });
  const answer = useMutation({
    mutationFn: () => api.answerGap(gap.id, text.trim()),
    onSuccess: () => { setError(null); setAnswering(false); setText(""); toast("Saved to memory. The gap is filled."); refresh(); },
    onError: (e) => setError(errorMessage(e, "Couldn't save the answer.")),
  });
  const reopen = useMutation({
    mutationFn: () => api.reopenGap(gap.id),
    onSuccess: () => { toast("Gap restored."); refresh(); },
    onError: (e) => setError(errorMessage(e, "Couldn't restore the gap.")),
  });
  const dismiss = useMutation({
    mutationFn: () => api.dismissGap(gap.id),
    onSuccess: () => { setError(null); toast("Gap dismissed.", { action: { label: "Undo", onClick: () => reopen.mutate() }, duration: 6000 }); refresh(); },
    onError: (e) => setError(errorMessage(e, "Couldn't dismiss the gap.")),
  });

  const status: { label: string; variant: StatusVariant } =
    gap.status === "filled" ? { label: "Filled", variant: "ok" }
    : researching ? { label: researching === "running" ? "Researching" : "Queued for research", variant: "info" }
    : gapNeedsYou(gap) ? { label: "Needs you", variant: "warn" }
    : gap.status === "open" ? { label: "Open", variant: "neutral" }
    : { label: gap.status, variant: "neutral" };
  const open = gap.status === "open" && !researching;
  const btn = "rounded-md border border-line px-2.5 py-1 text-[12.5px] font-medium text-fg-2 hover:bg-inset hover:text-fg disabled:opacity-50";
  const primary = "rounded-md bg-accent px-2.5 py-1 text-[12.5px] font-medium text-on-accent hover:bg-accent-hover disabled:opacity-50";

  return (
    <article className="rounded-xl border border-line bg-surface px-3.5 py-3">
      <div className="flex flex-wrap items-center gap-2">
        <StatusPill variant={status.variant} raw={gap.status} dot={researching === "running"}>{status.label}</StatusPill>
        {gap.disposition && <span className="text-[12px] text-fg-3" title={gap.disposition}>{DISPOSITION[gap.disposition] ?? gap.disposition}</span>}
        {(gap.domain || gap.subject) && <span className="text-[12px] text-fg-3">{[gap.domain, gap.subject].filter(Boolean).join(" › ")}</span>}
        <span className="flex-1" />
        <span className="text-[12px] text-fg-3">{fmtRelative(gap.created_at)}</span>
      </div>
      <p className="mt-1.5 whitespace-pre-wrap break-words text-[13.5px] text-fg">{gap.missing}</p>
      {gap.triggering_query && <p className="mt-1 text-[12px] italic text-fg-3">asked: “{gap.triggering_query}”</p>}
      {gap.filled_snippet && (
        <p className="mt-2 rounded-md border border-ok/30 bg-ok-soft px-2.5 py-1.5 text-[12.5px] text-fg"><span className="font-medium text-ok">Filled with: </span>{gap.filled_snippet}</p>
      )}
      {open && (
        <div className="mt-2.5 flex flex-wrap items-center gap-2">
          {gap.disposition === "needs_document" && (
            <Link to="/knowledge" className={primary}>Upload a document</Link>
          )}
          <button type="button" className={gap.disposition === "needs_document" ? btn : primary} onClick={() => setAnswering((v) => !v)} aria-expanded={answering}>Type the answer</button>
          {gap.disposition !== "needs_document" && (
            <button type="button" className={btn} disabled={research.isPending} onClick={() => research.mutate()}>
              {research.isPending ? "Queuing…" : "Research it"}
            </button>
          )}
          {gap.last_run_id && <Link to={`/cognition/run/${gap.last_run_id}`} className={btn}>Last run</Link>}
          <span className="flex-1" />
          <button type="button" className="text-[12.5px] text-fg-3 hover:text-fg" disabled={dismiss.isPending} onClick={() => dismiss.mutate()}>Dismiss</button>
        </div>
      )}
      {open && answering && (
        <form className="mt-2" onSubmit={(e) => { e.preventDefault(); if (text.trim()) answer.mutate(); }}>
          <textarea value={text} onChange={(e) => setText(e.target.value)} rows={3} aria-label="Your answer" autoFocus
            placeholder="Write what your tools should know. It is saved as a fact you stated."
            className="w-full resize-y rounded-md border border-line bg-inset px-2.5 py-1.5 text-[13px] text-fg outline-none focus:border-accent" />
          <div className="mt-1.5 flex justify-end gap-2">
            <button type="button" className={btn} onClick={() => { setAnswering(false); setText(""); }}>Cancel</button>
            <button type="submit" className={primary} disabled={!text.trim() || answer.isPending}>{answer.isPending ? "Saving…" : "Save answer"}</button>
          </div>
        </form>
      )}
      {gap.status === "closed" && <button type="button" className={cn(btn, "mt-2")} onClick={() => reopen.mutate()}>Restore</button>}
      {error && <ErrorBanner title="That didn't go through" message={error} className="mt-2" />}
    </article>
  );
}

// ----------------------------------------------------------- review queue

function ReviewQueue({ customerId }: { customerId: string }) {
  const qc = useQueryClient();
  const [error, setError] = useState<string | null>(null);
  const [rejecting, setRejecting] = useState<string | null>(null);
  const queue = useQuery({ queryKey: ["review-queue", customerId], queryFn: () => api.listReviewQueue(customerId), refetchInterval: 10_000 });
  const items = ((queue.data?.items ?? []) as PushQueueItem[]).filter((i) => i.status === "pending");
  const refresh = () => qc.invalidateQueries({ queryKey: ["review-queue", customerId] });
  const approve = useMutation({ mutationFn: (id: string) => api.approveReviewItem(customerId, id), onSuccess: () => { setError(null); refresh(); }, onError: (e) => setError(errorMessage(e, "Couldn't approve this fact.")) });
  const reject = useMutation({ mutationFn: (id: string) => api.rejectReviewItem(customerId, id), onSuccess: () => { setError(null); setRejecting(null); refresh(); }, onError: (e) => { setRejecting(null); setError(errorMessage(e, "Couldn't reject this fact.")); } });
  return (
    <section aria-labelledby="review-title">
      <div className="mb-2 flex flex-wrap items-end justify-between gap-2">
        <div>
          <h2 id="review-title" className="text-[15px] font-semibold text-fg">Review queue{items.length ? ` ${items.length}` : ""}</h2>
          <p className="text-[12.5px] text-fg-3">Facts Crystal was not sure enough to save.</p>
        </div>
        <Link to="/assumptions" className="text-[12.5px] text-accent hover:underline">Assumptions ›</Link>
      </div>
      {error && <ErrorBanner title="Review queue" message={error} className="mb-2" />}
      {queue.isLoading && <Skeleton rows={2} />}
      {queue.isError && <ErrorBanner title="Couldn't load the review queue" error={queue.error} onRetry={() => queue.refetch()} />}
      {!queue.isLoading && !queue.isError && items.length === 0 && (
        <EmptyState icon={Lightbulb} title="Nothing to review" help="When a tool saves something Crystal is unsure about, it waits here." />
      )}
      <div className="space-y-2">
        {items.map((item) => (
          <article key={item.id} className="rounded-xl border border-line bg-surface px-3.5 py-3">
            <div className="flex flex-wrap items-center gap-2">
              <code className="rounded bg-inset px-1.5 py-0.5 font-mono text-[12px] text-fg-2">{item.key}</code>
              <StatusPill variant="info">{Math.round((item.confidence ?? 0) * 100)}% sure</StatusPill>
              <span className="flex-1" />
              <span className="text-[12px] text-fg-3">{fmtRelative(item.created_at)}</span>
            </div>
            <p className="mt-1.5 whitespace-pre-wrap break-words text-[13.5px] text-fg">{item.value}</p>
            <div className="mt-2.5 flex flex-wrap items-center gap-2">
              <button type="button" onClick={() => approve.mutate(item.id)} disabled={approve.isPending}
                className="inline-flex items-center gap-1 rounded-md bg-accent px-2.5 py-1 text-[12.5px] font-medium text-on-accent hover:bg-accent-hover disabled:opacity-50">
                <CheckCircle2 aria-hidden className="h-3.5 w-3.5" /> Approve
              </button>
              <button type="button" onClick={() => setRejecting(item.id)} disabled={rejecting === item.id}
                className="inline-flex items-center gap-1 rounded-md border border-line px-2.5 py-1 text-[12.5px] font-medium text-fg-2 hover:bg-bad-soft hover:text-bad disabled:opacity-50">
                <XCircle aria-hidden className="h-3.5 w-3.5" /> Reject
              </button>
            </div>
            <ConfirmInline open={rejecting === item.id} message="Reject this fact? It is discarded, not saved." confirmLabel="Reject" busy={reject.isPending}
              onCancel={() => setRejecting(null)} onConfirm={() => reject.mutate(item.id)} />
          </article>
        ))}
      </div>
    </section>
  );
}

// ------------------------------------------------------------------- page

// The Background research switch is the tenant's auto_research spend
// budget (S4, ratified B-1: manual by default). ON writes a cap that no
// tenant reaches in practice, so the plan's daily AI capacity is the
// only bound; OFF writes 0, which the sweep reads as "skip this tenant".
export const BACKGROUND_RESEARCH_ON_CAP = 1_000_000_000_000;

export function Cognition() {
  const { selectedCustomerId } = useSelectedCustomer();
  const { me } = useAuth();
  const isPlatformAdmin = me?.role === "platform_admin" || me?.kind === "platform_admin_key";
  const navigate = useNavigate();
  const { runId } = useParams<{ runId: string }>();
  const qc = useQueryClient();
  const toast = useToast();
  const [switchError, setSwitchError] = useState<string | null>(null);

  const runs = useQuery({ queryKey: ["cognition-environments", selectedCustomerId], queryFn: () => api.listCognitionRuns(selectedCustomerId!), enabled: !!selectedCustomerId, refetchInterval: 5000 });
  const gaps = useQuery({ queryKey: ["knowledge-gaps", selectedCustomerId], queryFn: () => api.listKnowledgeGaps(selectedCustomerId!), enabled: !!selectedCustomerId, refetchInterval: 10_000 });
  const tasks = useQuery({ queryKey: ["cognition-tasks", selectedCustomerId], queryFn: () => api.listCognitionTasks(selectedCustomerId!), enabled: !!selectedCustomerId, refetchInterval: 10_000 });
  const budgets = useQuery({ queryKey: ["budgets", selectedCustomerId], queryFn: () => api.listBudgets(selectedCustomerId!), enabled: !!selectedCustomerId });
  const setResearch = useMutation({
    mutationFn: (on: boolean) => api.upsertBudget(selectedCustomerId!, "auto_research", on ? BACKGROUND_RESEARCH_ON_CAP : 0),
    onSuccess: (_r, on) => { setSwitchError(null); toast(on ? "Background research is on." : "Background research is off."); qc.invalidateQueries({ queryKey: ["budgets", selectedCustomerId] }); },
    onError: (e) => setSwitchError(errorMessage(e, "Couldn't change background research.")),
  });

  if (!selectedCustomerId) {
    return (
      <>
        <PageHeader title="Cognition" />
        <EmptyState icon={Brain} title="No customer selected" help="Pick a customer above to see their runs and gaps." />
      </>
    );
  }

  const runList = runs.data ?? [];
  const gapList = gaps.data?.items ?? [];
  const working = runList.filter((r) => isActiveStatus(r.status)).length;
  const needYou = runList.filter((r) => r.status === "needs_human_review").length + gapList.filter(gapNeedsYou).length;
  const openGaps = gapList.filter((g) => g.status === "open").length;
  const researchOn = (budgets.data?.budgets ?? []).some((b) => b.function === "auto_research" && !(b as { operator_id?: string }).operator_id && b.cap_micro_usd > 0);
  // A pending/running cognition task whose payload names a gap means the
  // gap is queued or being researched; the row says so instead of
  // offering Research again.
  const researching = new Map<string, string>();
  for (const t of tasks.data?.items ?? []) {
    const gid = t.payload?.gap_id;
    if (gid && (t.status === "pending" || t.status === "running")) researching.set(gid, t.status);
  }
  const pct = me?.usage?.ai_capacity_pct ?? null;

  return (
    <div className="space-y-6">
      <PageHeader title="Cognition"
        meta={runs.isLoading || gaps.isLoading ? "Loading…" : `${working} run${working === 1 ? "" : "s"} working · ${needYou} need${needYou === 1 ? "s" : ""} you · ${openGaps} gap${openGaps === 1 ? "" : "s"} open`}
        actions={
          <Switch label="Background research" checked={researchOn} disabled={budgets.isLoading || setResearch.isPending}
            hint={researchOn ? "Open gaps are researched on their own, within today's capacity." : "Gaps wait for you to press Research."}
            onChange={(on) => setResearch.mutate(on)} />
        } />
      {switchError && <ErrorBanner title="Background research" message={switchError} />}
      {isPlatformAdmin ? <SpendPanel customerId={selectedCustomerId} /> : <CapacityCard customerId={selectedCustomerId} pct={pct} />}
      <RunsTable runs={runList} isLoading={runs.isLoading} error={runs.error} onRetry={() => runs.refetch()} onOpen={(id) => navigate(`/cognition/run/${id}`)} isAdmin={isPlatformAdmin} />
      <GapsList customerId={selectedCustomerId} gaps={gapList} isLoading={gaps.isLoading} error={gaps.error} onRetry={() => gaps.refetch()} researching={researching} />
      <ReviewQueue customerId={selectedCustomerId} />
      {runId && <RunDrawer runId={runId} onClose={() => navigate("/cognition")} />}
    </div>
  );
}
