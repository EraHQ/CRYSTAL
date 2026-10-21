// Assumptions — Console Refine PR 7 (docs/CONSOLE_UX_AUDIT.md 5.6).
//
// Crystal connected two or more crystals and drew a conclusion; it is
// held out of recall until approved. The page defaults to what waits,
// gives each card one primary verb (Approve) and one secondary
// (Respond), pushes Verify and Delete right, confirms delete inline,
// links every parent to the bank, threads revisions with a left rule,
// opens the activity feed in a drawer, and deep-links a card at
// /assumptions/:crystalId.
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { History, Lightbulb, Search } from "lucide-react";
import { api, errorMessage } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";
import {
  ConfirmInline, Drawer, EmptyState, ErrorBanner, FilterChips, Meter, PageHeader, Skeleton, StatusPill, Toolbar, useToast,
} from "@/components/ui";
import type { StatusVariant } from "@/components/ui";
import type { Assumption, CurationEvent } from "@/lib/types";
import { cn, fmtRelative } from "@/lib/utils";
import { eventText } from "@/components/activity/events";

type Filter = "pending" | "approved" | "invalid" | "all";

export function assumptionState(a: Assumption): { key: Filter; label: string; variant: StatusVariant } {
  const invalidated = a.quality_tier === "blacklist";
  if (invalidated && a.superseded_by) return { key: "invalid", label: "Superseded", variant: "neutral" };
  if (invalidated) return { key: "invalid", label: "Invalid", variant: "bad" };
  if (!a.recall_gated) return { key: "approved", label: "Approved", variant: "ok" };
  return { key: "pending", label: "Waiting for review", variant: "warn" };
}

// v110: a thread is the root assumption plus every revision tagged with
// it; revisions sit under the root by iteration, indented by distance.
export function threadsOf(items: Assumption[]): { root: Assumption; members: Assumption[] }[] {
  const byRoot = new Map<string, Assumption[]>();
  for (const it of items) {
    const root = it.thread_id || it.id;
    if (!byRoot.has(root)) byRoot.set(root, []);
    byRoot.get(root)!.push(it);
  }
  const out: { root: Assumption; members: Assumption[] }[] = [];
  for (const [rootId, members] of byRoot) {
    const root = members.find((m) => m.id === rootId) ?? members[0];
    const rest = members.filter((m) => m !== root)
      .sort((a, b) => (a.iteration ?? 0) - (b.iteration ?? 0) || String(a.created_at).localeCompare(String(b.created_at)));
    out.push({ root, members: rest });
  }
  out.sort((a, b) => String(b.root.created_at).localeCompare(String(a.root.created_at)));
  return out;
}

function depthOf(item: Assumption, byId: Map<string, Assumption>): number {
  let d = 0;
  let cur: Assumption | undefined = item;
  const seen = new Set<string>();
  while (cur?.responds_to && !seen.has(cur.responds_to)) {
    seen.add(cur.responds_to);
    cur = byId.get(cur.responds_to);
    d += 1;
    if (!cur) break;
  }
  return d;
}


function AssumptionCard({ item, depth, highlighted, busy, onApprove, onVerify, onDelete, onRespond }: {
  item: Assumption; depth: number; highlighted: boolean; busy: boolean;
  onApprove: () => void; onVerify: () => void; onDelete: () => void; onRespond: (text: string) => Promise<void>;
}) {
  const state = assumptionState(item);
  const [responding, setResponding] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [text, setText] = useState("");
  const [sending, setSending] = useState(false);
  const verificationQueued = (item.diagnostic_tags ?? []).some((t) => typeof t === "string" && t.startsWith("verification_task:"));
  const pct = item.confidence == null ? null : Math.round(item.confidence * 100);
  const live = state.key === "pending" || state.key === "approved";
  const submit = async () => {
    if (!text.trim()) return;
    setSending(true);
    try { await onRespond(text.trim()); setText(""); setResponding(false); } finally { setSending(false); }
  };
  const btn = "rounded-md border border-line px-2.5 py-1 text-[12.5px] font-medium text-fg-2 hover:bg-inset hover:text-fg disabled:opacity-50";
  return (
    <article id={`assumption-${item.id}`}
      className={cn("rounded-xl border bg-surface p-4", highlighted ? "border-accent ring-2 ring-accent/40" : "border-line",
        depth > 0 && "ml-4 border-l-2 border-l-line-strong min-[720px]:ml-8")}>
      {item.operator_response && (
        <blockquote className="mb-3 rounded-md border-l-2 border-accent bg-inset px-3 py-2 text-[13px] text-fg">
          <span className="font-medium text-fg-2">You responded: </span>{item.operator_response}
        </blockquote>
      )}
      <div className="flex flex-wrap items-center gap-2">
        <StatusPill variant={state.variant} raw={`${item.quality_tier} · recall_gated ${item.recall_gated}`}>{state.label}</StatusPill>
        {pct != null && (
          <span className="inline-flex items-center gap-1.5 text-[12px] text-fg-2">
            <Meter value={pct} max={100} label={`${pct}% confident`} tone="accent" hideLabel className="w-16" />
            {pct}% confident
          </span>
        )}
        {item.responds_to && <span className="text-[12px] text-fg-3">{item.replaces_parent ? "revises the one above" : "adds to the one above"}</span>}
        <span className="flex-1" />
        <span className="text-[12px] text-fg-3">{fmtRelative(item.created_at)}</span>
      </div>
      <p className="mt-2 whitespace-pre-wrap break-words text-[14px] text-fg">{item.statement || "(no statement)"}</p>
      {item.rationale && (
        <section aria-label="Why" className="mt-2 rounded-lg border border-line bg-inset px-3 py-2 text-[13px]">
          <p className="text-[11px] font-medium uppercase tracking-[0.08em] text-fg-3">Why</p>
          <p className="mt-0.5 whitespace-pre-wrap break-words text-fg">{item.rationale}</p>
        </section>
      )}
      {(item.parents.length > 0 || item.invalidated_parents.length > 0) && (
        <div className="mt-2.5">
          <p className="text-[11px] font-medium uppercase tracking-[0.08em] text-fg-3">Built from</p>
          <div className="mt-1 grid grid-cols-1 gap-1.5 min-[720px]:grid-cols-2">
            {item.parents.map((p) => (
              <Link key={p.id} to={`/bank/${p.id}`} title={p.id}
                className="block min-w-0 rounded-md border border-line bg-inset px-2.5 py-1.5 text-[12.5px] text-fg-2 hover:border-accent/50 hover:text-fg">
                <span className="block truncate font-medium">{p.label || p.summary_text || p.id}</span>
                {p.claims && p.claims.length > 0 && (
                  <span className="mt-0.5 block text-[12px] text-fg-3">“{p.claims[0]}”{p.claims.length > 1 ? ` and ${p.claims.length - 1} more` : ""}</span>
                )}
              </Link>
            ))}
            {item.invalidated_parents.map((id) => (
              <span key={id} className="rounded-md border border-bad/30 bg-bad-soft px-2.5 py-1.5 text-[12px] text-bad line-through" title="This parent was deleted">{id}</span>
            ))}
          </div>
        </div>
      )}
      {live && (
        <div className="mt-3 flex flex-wrap items-center gap-2">
          {state.key === "pending" && (
            <button type="button" disabled={busy} onClick={onApprove}
              className="rounded-md bg-accent px-3 py-1.5 text-[12.5px] font-medium text-on-accent hover:bg-accent-hover disabled:opacity-50">
              Approve
            </button>
          )}
          <button type="button" disabled={busy} onClick={() => setResponding((v) => !v)} aria-expanded={responding} className={btn}>Respond</button>
          <span className="flex-1" />
          <button type="button" disabled={busy || verificationQueued} onClick={onVerify}
            title={verificationQueued ? "A verification run is already queued" : "Queue research to confirm or refute this"}
            className="text-[12.5px] text-fg-3 hover:text-fg disabled:opacity-50">
            {verificationQueued ? "Verifying…" : "Verify"}
          </button>
          <button type="button" disabled={busy || confirming} onClick={() => setConfirming(true)}
            className="text-[12.5px] text-fg-3 hover:text-bad disabled:opacity-50">Delete</button>
        </div>
      )}
      {!live && (
        <div className="mt-3 flex justify-end">
          <button type="button" disabled={busy || confirming} onClick={() => setConfirming(true)} className="text-[12.5px] text-fg-3 hover:text-bad disabled:opacity-50">Delete</button>
        </div>
      )}
      <ConfirmInline open={confirming} message="Delete this assumption?" permanent confirmLabel="Delete" busy={busy}
        onConfirm={() => { setConfirming(false); onDelete(); }} onCancel={() => setConfirming(false)} />
      {responding && (
        <form className="mt-3 border-t border-line pt-3" onSubmit={(e) => { e.preventDefault(); void submit(); }}>
          <textarea value={text} onChange={(e) => setText(e.target.value)} rows={3} maxLength={4000} aria-label="What is actually true?"
            placeholder="What is actually true?"
            className="w-full resize-y rounded-md border border-line bg-inset px-3 py-2 text-[13px] text-fg outline-none focus:border-accent" />
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <span className="text-[12px] text-fg-3">Uses daily AI capacity. Nothing is deleted; revisions join this thread.</span>
            <span className="flex-1" />
            <button type="button" disabled={sending} onClick={() => { setResponding(false); setText(""); }} className={btn}>Cancel</button>
            <button type="submit" disabled={sending || !text.trim()}
              className="rounded-md bg-accent px-3 py-1.5 text-[12.5px] font-medium text-on-accent hover:bg-accent-hover disabled:opacity-50">
              {sending ? "Thinking…" : "Send response"}
            </button>
          </div>
        </form>
      )}
    </article>
  );
}

function ActivityDrawer({ customerId, onClose }: { customerId: string; onClose: () => void }) {
  const activity = useQuery({ queryKey: ["curation-activity", customerId], queryFn: () => api.listCurationActivity(customerId, 100), refetchInterval: 15_000 });
  const events: CurationEvent[] = activity.data?.events ?? [];
  return (
    <Drawer open onClose={onClose} title="Activity">
      <p className="mb-3 text-[12.5px] text-fg-2">What Crystal did to its own knowledge, newest first. Nothing here is silent.</p>
      {activity.isLoading && <Skeleton rows={5} />}
      {activity.isError && <ErrorBanner title="Couldn't load activity" error={activity.error} onRetry={() => activity.refetch()} />}
      {!activity.isLoading && events.length === 0 && <EmptyState icon={History} title="No activity yet" help="Approvals, responses, gap fills and scans land here as they happen." />}
      <ul className="divide-y divide-line">
        {events.map((e) => (
          <li key={e.id} className="flex items-start gap-3 py-2 text-[13px]">
            <span className="mt-0.5 whitespace-nowrap text-[12px] font-medium text-fg-2" title={e.event_type}>{eventText(e.event_type)}</span>
            <span className="min-w-0 flex-1 break-words text-fg">{e.label || ""}</span>
            <span className="whitespace-nowrap text-[12px] text-fg-3">{fmtRelative(e.created_at)}</span>
          </li>
        ))}
      </ul>
    </Drawer>
  );
}

export function Assumptions() {
  const { selectedCustomerId } = useSelectedCustomer();
  const { crystalId } = useParams<{ crystalId: string }>();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const toast = useToast();
  const [filter, setFilter] = useState<Filter>("pending");
  const [q, setQ] = useState("");
  const [activityOpen, setActivityOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrolled = useRef<string | null>(null);

  const assumptions = useQuery({ queryKey: ["assumptions", selectedCustomerId], queryFn: () => api.listAssumptions(selectedCustomerId!), enabled: !!selectedCustomerId, refetchInterval: 15_000 });
  const items = assumptions.data?.items ?? [];
  const threads = useMemo(() => threadsOf(items), [items]);
  const threadRoots = threads.filter((t) => t.members.length).map((t) => t.root.id);
  // Operator responses ride only on the thread read; fetch those for
  // every root that has revisions.
  const threadReads = useQuery({
    queryKey: ["assumption-threads", selectedCustomerId, threadRoots.join(",")],
    queryFn: async () => {
      const out: Record<string, Assumption> = {};
      for (const root of threadRoots) {
        const t = await api.assumptionThread(root);
        for (const m of t.thread as Assumption[]) out[m.id] = m;
      }
      return out;
    },
    enabled: !!selectedCustomerId && threadRoots.length > 0,
  });
  const enrich = (it: Assumption): Assumption => ({ ...it, ...(threadReads.data?.[it.id] ?? {}) });

  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["assumptions", selectedCustomerId] });
    qc.invalidateQueries({ queryKey: ["assumption-threads", selectedCustomerId] });
    qc.invalidateQueries({ queryKey: ["curation-activity", selectedCustomerId] });
  };
  // The nav badge reads ["assumptions", id] and counts pending rows; flip
  // the row in that cache at once so the badge moves with the click.
  const patchCache = (id: string, patch: Partial<Assumption> | null) =>
    qc.setQueryData(["assumptions", selectedCustomerId], (old: { total: number; items: Assumption[] } | undefined) =>
      old ? {
        total: patch ? old.total : Math.max(0, old.total - 1),
        items: patch ? old.items.map((a) => (a.id === id ? { ...a, ...patch } : a)) : old.items.filter((a) => a.id !== id),
      } : old);

  const approve = useMutation({
    mutationFn: (id: string) => api.approveAssumption(id),
    onMutate: (id) => patchCache(id, { recall_gated: false }),
    onSuccess: () => { setError(null); toast("Approved. Your tools can recall it now."); refresh(); },
    onError: (e) => { setError(errorMessage(e, "Couldn't approve this assumption.")); refresh(); },
  });
  const verify = useMutation({
    mutationFn: (id: string) => api.verifyAssumption(id),
    onSuccess: () => { setError(null); toast("Verification queued. The result joins this thread."); refresh(); },
    onError: (e) => setError(errorMessage(e, "Couldn't queue verification.")),
  });
  const respond = useMutation({
    mutationFn: ({ id, text }: { id: string; text: string }) => api.respondAssumption(id, text),
    onSuccess: (out) => { setError(null); toast(out.created?.length ? `${out.created.length} revision${out.created.length === 1 ? "" : "s"} added to the thread.` : "Response saved."); refresh(); },
    onError: (e) => setError(errorMessage(e, "Couldn't send the response.")),
  });
  const remove = useMutation({
    mutationFn: (id: string) => api.deleteCrystal(id),
    onMutate: (id) => patchCache(id, null),
    onSuccess: () => { setError(null); toast("Assumption deleted."); refresh(); },
    onError: (e) => { setError(errorMessage(e, "Couldn't delete this assumption.")); refresh(); },
  });

  // Deep link: show the card whatever its state, scroll to it, highlight it.
  useEffect(() => {
    if (!crystalId || !items.length || scrolled.current === crystalId) return;
    const target = items.find((a) => a.id === crystalId);
    if (!target) return;
    if (assumptionState(target).key !== filter) setFilter("all");
    scrolled.current = crystalId;
    requestAnimationFrame(() => {
      const el = document.getElementById(`assumption-${crystalId}`);
      if (el && typeof el.scrollIntoView === "function") el.scrollIntoView({ block: "center", behavior: "smooth" });
    });
  }, [crystalId, items, filter]);

  if (!selectedCustomerId) {
    return (
      <>
        <PageHeader title="Assumptions" />
        <EmptyState icon={Lightbulb} title="No customer selected" help="Pick a customer above to review their assumptions." />
      </>
    );
  }

  const counts: Record<Filter, number> = {
    pending: items.filter((a) => assumptionState(a).key === "pending").length,
    approved: items.filter((a) => assumptionState(a).key === "approved").length,
    invalid: items.filter((a) => assumptionState(a).key === "invalid").length,
    all: items.length,
  };
  const needle = q.trim().toLowerCase();
  const matchesText = (a: Assumption) => !needle || (a.statement ?? "").toLowerCase().includes(needle) || (a.rationale ?? "").toLowerCase().includes(needle);
  const matchesFilter = (a: Assumption) => filter === "all" || assumptionState(a).key === filter || a.id === crystalId;
  // A thread shows when any member matches; members that do not match stay
  // visible so the thread reads whole.
  const shown = threads.filter((t) => [t.root, ...t.members].some((m) => matchesFilter(m) && matchesText(m)));
  const busy = approve.isPending || verify.isPending || respond.isPending || remove.isPending;

  return (
    <div>
      <PageHeader title="Assumptions"
        meta={assumptions.isLoading ? "Loading…" : `${counts.pending} waiting for review · ${counts.all} in all`}
        actions={
          <button type="button" onClick={() => setActivityOpen(true)}
            className="inline-flex items-center gap-1.5 rounded-md border border-line px-3 py-1.5 text-[13px] font-medium text-fg-2 hover:bg-inset hover:text-fg">
            <History aria-hidden className="h-3.5 w-3.5" /> Activity
          </button>
        }
        toolbar={
          <Toolbar search={
            <label className="relative block">
              <Search aria-hidden className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-fg-3" />
              <input type="search" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search assumptions" placeholder="Search assumptions"
                className="w-full rounded-md border border-line bg-inset py-1.5 pl-8 pr-2 text-[13px] text-fg placeholder:text-fg-3 outline-none focus:border-accent" />
            </label>
          }>
            <FilterChips label="State" value={filter} onChange={(v) => setFilter((v as Filter) ?? "pending")}
              options={[{ value: "pending", label: "Pending", count: counts.pending }, { value: "approved", label: "Approved", count: counts.approved },
                { value: "invalid", label: "Superseded or invalid", count: counts.invalid }, { value: "all", label: "All", count: counts.all }]} />
          </Toolbar>
        } />
      <p className="mb-4 text-[13px] text-fg-2">Crystal connected two or more crystals and drew a conclusion. Approve to let your tools recall it, or respond to correct it.</p>
      {error && <ErrorBanner title="That didn't go through" message={error} className="mb-3" />}
      {assumptions.isLoading && <Skeleton rows={4} />}
      {assumptions.isError && <ErrorBanner title="Couldn't load assumptions" error={assumptions.error} onRetry={() => assumptions.refetch()} />}
      {!assumptions.isLoading && !assumptions.isError && shown.length === 0 && (
        <EmptyState icon={Lightbulb} title={items.length ? "Nothing here" : "No assumptions yet"}
          help={items.length ? "Nothing matches this filter or search." : "Crystal proposes one when two of your crystals imply something neither says."} />
      )}
      <div className="space-y-4">
        {shown.map(({ root, members }) => {
          const byId = new Map<string, Assumption>([root, ...members].map((m) => [m.id, m]));
          return (
            <div key={root.id} className="space-y-2">
              {[root, ...members].map((m) => (
                <AssumptionCard key={m.id} item={enrich(m)} depth={m === root ? 0 : depthOf(m, byId)} highlighted={m.id === crystalId} busy={busy}
                  onApprove={() => approve.mutate(m.id)} onVerify={() => verify.mutate(m.id)} onDelete={() => remove.mutate(m.id)}
                  onRespond={async (text) => { await respond.mutateAsync({ id: m.id, text }); }} />
              ))}
            </div>
          );
        })}
      </div>
      {activityOpen && <ActivityDrawer customerId={selectedCustomerId} onClose={() => setActivityOpen(false)} />}
      {crystalId && <span className="sr-only">Showing assumption {crystalId}. <button type="button" onClick={() => navigate("/assumptions")}>Clear</button></span>}
    </div>
  );
}
