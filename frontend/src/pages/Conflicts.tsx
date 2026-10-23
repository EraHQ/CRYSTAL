// Conflicts — Console Refine PR 7 (docs/CONSOLE_UX_AUDIT.md 5.5).
//
// Two facts the bank holds that cannot both be true. The decision UI
// keeps its shape (two claims, one verb per side) and gains what the
// decision needs: when each claim was learned, where it came from, which
// is newer, one primary verb per side, Undo on every decision (the new
// /reopen route), a Resolved tab with Reopen, and a Backlog column whose
// rows go where each item is handled. Claims stack under 720px.
//
// PR 7b (2026-10-16): a contradiction carries the scanner's own account
// ("Why it looks like a contradiction") and a suggested verb with Apply
// suggestion; a contradiction without one offers Explain (one small
// model call through /explain). 2026-10-21: duplicates carry the same
// block in their own words ("Why these look like the same fact"; "Keep
// claim B; claim A says the same thing").
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Layers, RefreshCw, Scale } from "lucide-react";
import { api, errorMessage } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useSelectedCustomer } from "@/lib/selected-customer";
import { EmptyState, ErrorBanner, FilterChips, PageHeader, Skeleton, StatusPill, useToast } from "@/components/ui";
import type { StatusVariant } from "@/components/ui";
import type { BacklogItem, ConflictFact, ConflictRecommendation, KnowledgeConflict } from "@/lib/types";
import { cn, fmtRelative } from "@/lib/utils";

// Detectors (models/knowledge_conflict.py ConflictDetector) in plain words.
// Q98 B (2026-10-22): implication_scan rows are contradictions found by
// reasoning one step out from a fact (its explanation starts "A implies").
const KIND: Record<string, string> = { contradiction_scan: "Contradiction", dedup_scan: "Duplicate", implication_scan: "Found by reasoning" };
const RESOLUTION: Record<string, { label: string; variant: StatusVariant }> = {
  qualified: { label: "Both are true", variant: "info" },
  superseded: { label: "One kept", variant: "ok" },
  blacklisted: { label: "One marked wrong", variant: "bad" },
  dismissed: { label: "Not a conflict", variant: "neutral" },
};
const SOURCE: Record<string, string> = {
  model_reasoning: "a conversation", failed_reasoning: "a failed attempt", web_search_result: "a web search",
  code_execution_result: "running code", document_chunk: "a document", document_extraction: "a document",
  operator_stated: "you", agent_inferred: "a tool's inference",
};

export function claimSource(f: ConflictFact | null | undefined, provenance: string | null): string {
  if (f?.source_path) return `From ${f.source_path}`;
  if (f?.source_kind) return `From ${SOURCE[f.source_kind] ?? f.source_kind}`;
  return provenance ? `From ${provenance}` : "Source unknown";
}

export function newerSide(c: KnowledgeConflict): "a" | "b" | null {
  const a = c.facts?.a?.learned_at ? Date.parse(c.facts.a.learned_at) : NaN;
  const b = c.facts?.b?.learned_at ? Date.parse(c.facts.b.learned_at) : NaN;
  if (Number.isNaN(a) || Number.isNaN(b) || a === b) return null;
  return a > b ? "a" : "b";
}

// The suggestion in the console's own verbs (PR 7b). A duplicate's verbs
// mean something slightly different (2026-10-21): one copy is redundant,
// both add something, or they are not the same fact.
export function suggestionLabel(rec: ConflictRecommendation, detector = "contradiction_scan"): string {
  const loser = rec.loser ? rec.loser.toUpperCase() : null;
  const kept = rec.loser === "a" ? "B" : "A";
  const dup = detector === "dedup_scan";
  switch (rec.resolution) {
    case "superseded":
      if (!loser) return "Keep one";
      return dup ? `Keep claim ${kept}; claim ${loser} says the same thing` : `Keep claim ${kept}; claim ${loser} is outdated`;
    case "blacklisted": return loser ? `Mark claim ${loser} wrong` : "Mark one wrong";
    case "qualified": return dup ? "Both add something; keep both" : "Both are true";
    case "dismissed": return dup ? "Not a duplicate" : "Not a conflict";
    default: return rec.resolution;
  }
}

const WHY: Record<string, string> = {
  contradiction_scan: "Why it looks like a contradiction",
  dedup_scan: "Why these look like the same fact",
  implication_scan: "What A implies, and how B goes against it",
};

function fmtLearned(iso: string | null | undefined): string {
  if (!iso) return "Learned date unknown";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "Learned date unknown";
  return `Learned ${d.toLocaleDateString(undefined, { month: "short", day: "numeric" })} · ${fmtRelative(iso)}`;
}

// Where a backlog row is handled (5.5 item 6).
export function backlogHref(b: BacklogItem, isAdmin: boolean): string {
  switch (b.kind) {
    case "conflict": return `/conflicts#conflict-${b.id}`;
    case "verification": return "/assumptions";
    case "agent_task": return isAdmin ? "/agents" : "/cognition";
    case "push_review": return "/cognition#review";
    default: return "/cognition";
  }
}
const BACKLOG_KIND: Record<string, string> = {
  gap: "Gap", conflict: "Conflict", cognition_task: "Research", agent_task: "Agent task", push_review: "Review", verification: "Verification",
};

type Side = "a" | "b";

function ClaimCard({ c, side, newer, busy, onKeep, onWrong }: {
  c: KnowledgeConflict; side: Side; newer: boolean; busy: boolean; onKeep: () => void; onWrong: () => void;
}) {
  const claim = side === "a" ? c.claim_a : c.claim_b;
  const prov = side === "a" ? c.provenance_a : c.provenance_b;
  const f = c.facts?.[side] ?? null;
  const open = c.status === "open";
  return (
    <div className={cn("flex flex-col rounded-lg border bg-inset p-3", newer ? "border-accent" : "border-line")}>
      <div className="mb-1.5 flex flex-wrap items-center gap-2 text-[11px]">
        <span className="font-medium uppercase tracking-[0.08em] text-fg-3">Claim {side.toUpperCase()}</span>
        {newer && <StatusPill variant="info">Newer</StatusPill>}
        {f && !f.active && <StatusPill variant="neutral" raw="grating_strength 0">Out of recall</StatusPill>}
      </div>
      <p className="flex-1 whitespace-pre-wrap break-words text-[13.5px] text-fg">{claim}</p>
      <p className="mt-2 text-[12px] text-fg-3">{fmtLearned(f?.learned_at)}</p>
      <p className="text-[12px] text-fg-3">{claimSource(f, prov)}</p>
      {open && (
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <button type="button" disabled={busy} onClick={onKeep}
            className="rounded-md bg-accent px-2.5 py-1 text-[12.5px] font-medium text-on-accent hover:bg-accent-hover disabled:opacity-50"
            title={`Keep claim ${side.toUpperCase()}; the other leaves recall`}>
            Keep this one
          </button>
          <button type="button" disabled={busy} onClick={onWrong}
            className="text-[12.5px] text-fg-3 hover:text-bad disabled:opacity-50"
            title={`Claim ${side.toUpperCase()} is wrong: it leaves recall and the claim is blocked from coming back`}>
            Mark wrong
          </button>
        </div>
      )}
    </div>
  );
}

function Explanation({ c, busy, explaining, onApply, onExplain }: {
  c: KnowledgeConflict; busy: boolean; explaining: boolean; onApply: (rec: ConflictRecommendation) => void; onExplain: () => void;
}) {
  const open = c.status === "open";
  const rec = c.recommendation ?? null;
  if (!c.reasoning) {
    if (!open) return null;
    return (
      <div className="mt-3 flex flex-wrap items-center gap-2 rounded-lg border border-dashed border-line px-3 py-2 text-[12.5px] text-fg-3">
        <span>No explanation yet.</span>
        <button type="button" disabled={busy || explaining} onClick={onExplain}
          className="rounded-md border border-line px-2 py-0.5 font-medium text-fg-2 hover:bg-inset hover:text-fg disabled:opacity-50">
          {explaining ? "Explaining…" : "Explain"}
        </button>
        <span>Uses a little AI capacity.</span>
      </div>
    );
  }
  return (
    <section aria-label="Explanation" className="mt-3 rounded-lg border border-line bg-inset p-3 text-[13px]">
      <p className="text-[11px] font-medium uppercase tracking-[0.08em] text-fg-3">{WHY[c.detector] ?? "Why the scan paired these"}</p>
      <p className="mt-1 whitespace-pre-wrap break-words text-fg">{c.reasoning}</p>
      {rec && (
        <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1.5">
          <p className="text-fg-2">
            <span className="font-medium text-fg">Suggested: {suggestionLabel(rec, c.detector)}.</span>
            {rec.reason ? ` ${rec.reason}` : ""}
          </p>
          {open && RESOLUTION[rec.resolution] && (
            <button type="button" disabled={busy} onClick={() => onApply(rec)}
              className="rounded-md bg-accent px-2.5 py-1 text-[12.5px] font-medium text-on-accent hover:bg-accent-hover disabled:opacity-50">
              Apply suggestion
            </button>
          )}
        </div>
      )}
    </section>
  );
}

function ConflictCard({ c, busy, explaining, onDecide, onReopen, onExplain }: {
  c: KnowledgeConflict; busy: boolean; explaining: boolean;
  onDecide: (resolution: string, loser?: Side) => void; onReopen: () => void; onExplain: () => void;
}) {
  const newer = newerSide(c);
  const open = c.status === "open";
  const res = c.resolution ? RESOLUTION[c.resolution] : null;
  return (
    <article id={`conflict-${c.id}`} className="rounded-xl border border-line bg-surface p-4">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        {c.subject && <span className="text-[13px] font-medium text-fg">{c.subject}</span>}
        <span className="text-[12px] text-fg-3" title={c.detector}>{KIND[c.detector] ?? c.detector}</span>
        {res && <StatusPill variant={res.variant} raw={c.resolution ?? undefined}>{res.label}</StatusPill>}
        <span className="flex-1" />
        <span className="text-[12px] text-fg-3">{open ? `found ${fmtRelative(c.created_at)}` : `decided ${fmtRelative(c.resolved_at ?? c.created_at)}`}</span>
      </div>
      <div className="grid grid-cols-1 gap-3 min-[720px]:grid-cols-2">
        {(["a", "b"] as const).map((side) => (
          <ClaimCard key={side} c={c} side={side} newer={newer === side} busy={busy}
            onKeep={() => onDecide("superseded", side === "a" ? "b" : "a")}
            onWrong={() => onDecide("blacklisted", side)} />
        ))}
      </div>
      <Explanation c={c} busy={busy} explaining={explaining} onExplain={onExplain}
        onApply={(rec) => onDecide(rec.resolution, rec.loser ?? undefined)} />
      <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-line pt-3">
        {open ? (
          <>
            <button type="button" disabled={busy} onClick={() => onDecide("qualified")}
              className="rounded-md border border-line px-2.5 py-1 text-[12.5px] font-medium text-fg-2 hover:bg-inset hover:text-fg disabled:opacity-50"
              title="True under different conditions; both stay in recall">Both are true</button>
            <button type="button" disabled={busy} onClick={() => onDecide("dismissed")}
              className="rounded-md border border-line px-2.5 py-1 text-[12.5px] font-medium text-fg-2 hover:bg-inset hover:text-fg disabled:opacity-50"
              title="Not a real conflict; nothing changes">Not a conflict</button>
          </>
        ) : (
          <button type="button" disabled={busy} onClick={onReopen}
            className="rounded-md border border-line px-2.5 py-1 text-[12.5px] font-medium text-fg-2 hover:bg-inset hover:text-fg disabled:opacity-50">
            Reopen
          </button>
        )}
      </div>
    </article>
  );
}

export function Conflicts() {
  const { selectedCustomerId } = useSelectedCustomer();
  const { me } = useAuth();
  const isPlatformAdmin = me?.role === "platform_admin" || me?.kind === "platform_admin_key";
  const qc = useQueryClient();
  const toast = useToast();
  const [tab, setTab] = useState<"open" | "resolved">("open");
  const [kind, setKind] = useState<string | null>(null);
  const [scanReport, setScanReport] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [explainingId, setExplainingId] = useState<string | null>(null);

  const open = useQuery({ queryKey: ["conflicts", selectedCustomerId], queryFn: () => api.listConflicts(selectedCustomerId!, "open"), enabled: !!selectedCustomerId, refetchInterval: 15_000 });
  const resolved = useQuery({ queryKey: ["conflicts-resolved", selectedCustomerId], queryFn: () => api.listConflicts(selectedCustomerId!, "resolved"), enabled: !!selectedCustomerId });
  const dismissed = useQuery({ queryKey: ["conflicts-dismissed", selectedCustomerId], queryFn: () => api.listConflicts(selectedCustomerId!, "dismissed"), enabled: !!selectedCustomerId });
  const backlog = useQuery({ queryKey: ["backlog", selectedCustomerId], queryFn: () => api.listBacklog(selectedCustomerId!), enabled: !!selectedCustomerId, refetchInterval: 15_000 });
  const lastScan = useQuery({ queryKey: ["last-conflict-scan", selectedCustomerId], queryFn: () => api.listCurationActivity(selectedCustomerId!, 1, "conflict_scan_completed"), enabled: !!selectedCustomerId });

  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["conflicts", selectedCustomerId] });
    qc.invalidateQueries({ queryKey: ["conflicts-resolved", selectedCustomerId] });
    qc.invalidateQueries({ queryKey: ["conflicts-dismissed", selectedCustomerId] });
    qc.invalidateQueries({ queryKey: ["backlog", selectedCustomerId] });
  };
  // The nav badge and the Open count read ["conflicts", id]; drop the row
  // from that cache at once, then let the refetch confirm.
  const dropOpen = (id: string) =>
    qc.setQueryData(["conflicts", selectedCustomerId], (old: { total: number; items: KnowledgeConflict[] } | undefined) =>
      old ? { total: Math.max(0, old.total - 1), items: old.items.filter((c) => c.id !== id) } : old);
  const putOpen = (c: KnowledgeConflict) =>
    qc.setQueryData(["conflicts", selectedCustomerId], (old: { total: number; items: KnowledgeConflict[] } | undefined) =>
      old && !old.items.some((x) => x.id === c.id) ? { total: old.total + 1, items: [c, ...old.items] } : old);

  // An explanation lands on the row in every list cache (facts kept: the
  // explain route returns the conflict without them).
  const patchConflict = (c: KnowledgeConflict) => {
    for (const key of ["conflicts", "conflicts-resolved", "conflicts-dismissed"]) {
      qc.setQueryData([key, selectedCustomerId], (old: { total: number; items: KnowledgeConflict[] } | undefined) =>
        old ? { ...old, items: old.items.map((x) => (x.id === c.id ? { ...x, ...c } : x)) } : old);
    }
  };
  const explain = useMutation({
    mutationFn: (id: string) => api.explainConflict(selectedCustomerId!, id),
    onMutate: (id) => setExplainingId(id),
    onSuccess: (out) => { setError(null); patchConflict(out.conflict); refresh(); },
    onError: (e) => setError(errorMessage(e, "Couldn't explain this conflict.")),
    onSettled: () => setExplainingId(null),
  });

  const reopen = useMutation({
    mutationFn: (id: string) => api.reopenConflict(selectedCustomerId!, id),
    onSuccess: (out) => { setError(null); putOpen(out.conflict); toast("Conflict reopened."); refresh(); },
    onError: (e) => setError(errorMessage(e, "Couldn't reopen this conflict.")),
  });
  const decide = useMutation({
    mutationFn: ({ id, resolution, loser }: { id: string; resolution: string; loser?: Side }) =>
      api.resolveConflict(selectedCustomerId!, id, resolution, loser),
    onMutate: ({ id }) => dropOpen(id),
    onSuccess: (out, { id, resolution }) => {
      setError(null);
      toast(`${RESOLUTION[resolution]?.label ?? "Decided"}.`, { action: { label: "Undo", onClick: () => reopen.mutate(id) }, duration: 6000 });
      void out; refresh();
    },
    onError: (e) => { setError(errorMessage(e, "Couldn't save that decision.")); refresh(); },
  });
  const scan = useMutation({
    mutationFn: () => api.scanConflicts(selectedCustomerId!),
    onSuccess: (data) => {
      const s = data.scan ?? {};
      const n = s.conflicts_found ?? 0;
      const explained = n > 0 ? `, ${s.explained ?? 0} explained` : "";
      setScanReport(`Scanned ${(s.facts_scanned ?? 0).toLocaleString()} facts, ${(s.pairs_evaluated ?? 0).toLocaleString()} pairs checked: ${n} new conflict${n === 1 ? "" : "s"}${explained}.`);
      setError(null); refresh();
      qc.invalidateQueries({ queryKey: ["last-conflict-scan", selectedCustomerId] });
    },
    onError: (e) => setError(errorMessage(e, "The scan could not run.")),
  });

  const openItems = open.data?.items ?? [];
  const history = useMemo(() => [...(resolved.data?.items ?? []), ...(dismissed.data?.items ?? [])]
    .sort((a, b) => String(b.resolved_at ?? b.created_at).localeCompare(String(a.resolved_at ?? a.created_at))), [resolved.data, dismissed.data]);
  const list = (tab === "open" ? openItems : history).filter((c) => !kind || c.detector === kind);
  const kinds = Object.keys(KIND);
  const countKind = (k: string | null) => (tab === "open" ? openItems : history).filter((c) => !k || c.detector === k).length;
  const scanEvent = lastScan.data?.events?.[0];
  const busy = decide.isPending || reopen.isPending || scan.isPending || explain.isPending;

  if (!selectedCustomerId) {
    return (
      <>
        <PageHeader title="Conflicts" />
        <EmptyState icon={Scale} title="No customer selected" help="Pick a customer above to review their conflicts." />
      </>
    );
  }

  const meta = open.isLoading ? "Loading…"
    : `${openItems.length} open · ${scanEvent ? `last scan ${fmtRelative(scanEvent.created_at)} over ${Number(scanEvent.payload?.facts_scanned ?? 0).toLocaleString()} facts` : "not scanned yet"}`;

  return (
    <div>
      <PageHeader title="Conflicts" meta={meta}
        actions={
          <button type="button" onClick={() => scan.mutate()} disabled={busy}
            className="inline-flex items-center gap-1.5 rounded-md bg-accent px-3 py-1.5 text-[13px] font-medium text-on-accent hover:bg-accent-hover disabled:opacity-50">
            <RefreshCw aria-hidden className={cn("h-3.5 w-3.5", scan.isPending && "animate-spin")} /> {scan.isPending ? "Scanning…" : "Scan now"}
          </button>
        } />
      {scanReport && <p role="status" className="mb-3 rounded-lg border border-line bg-surface px-3 py-2 text-[13px] text-fg-2">{scanReport}</p>}
      {error && <ErrorBanner title="That didn't go through" message={error} className="mb-3" />}

      <div className="grid gap-6 min-[1024px]:grid-cols-[minmax(0,1fr)_300px]">
        <section aria-label="Conflicts">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <div role="tablist" aria-label="Conflict state" className="flex gap-4 border-b border-line text-[13px]">
              {([["open", `Open ${openItems.length}`], ["resolved", `Resolved ${history.length}`]] as const).map(([id, label]) => (
                <button key={id} type="button" role="tab" aria-selected={tab === id} onClick={() => setTab(id)}
                  className={cn("-mb-px border-b-2 pb-2", tab === id ? "border-accent font-medium text-accent" : "border-transparent text-fg-2 hover:text-fg")}>
                  {label}
                </button>
              ))}
            </div>
            <FilterChips label="Kind" value={kind ?? "all"} onChange={(v) => setKind(v === "all" || v === null ? null : v)}
              options={[{ value: "all", label: "All", count: countKind(null) }, ...kinds.map((k) => ({ value: k, label: KIND[k] + "s", count: countKind(k), title: k }))]} />
          </div>
          {(tab === "open" ? open.isLoading : resolved.isLoading || dismissed.isLoading) && <Skeleton rows={4} />}
          {open.isError && <ErrorBanner title="Couldn't load conflicts" error={open.error} onRetry={() => open.refetch()} />}
          {!open.isLoading && !open.isError && list.length === 0 && (
            <EmptyState icon={Check} title={tab === "open" ? "No open conflicts" : "No decisions yet"}
              help={tab === "open" ? "Run a scan, or the bank agrees with itself for now." : "Decisions you make on open conflicts land here and can be reopened."} />
          )}
          <div className="space-y-3">
            {list.map((c) => (
              <ConflictCard key={c.id} c={c} busy={busy} explaining={explainingId === c.id}
                onDecide={(resolution, loser) => decide.mutate({ id: c.id, resolution, loser })}
                onReopen={() => reopen.mutate(c.id)} onExplain={() => explain.mutate(c.id)} />
            ))}
          </div>
        </section>

        <aside aria-labelledby="backlog-title" className="min-w-0">
          <h2 id="backlog-title" className="mb-1 flex items-center gap-1.5 text-[13.5px] font-semibold text-fg">
            <Layers aria-hidden className="h-4 w-4 text-fg-3" /> Backlog · {backlog.data?.total ?? 0}
          </h2>
          <p className="mb-2 text-[12px] text-fg-3">Everything waiting on you or on research, highest priority first.</p>
          {backlog.isLoading && <Skeleton rows={3} />}
          {!backlog.isLoading && (backlog.data?.items.length ?? 0) === 0 && <p className="text-[12.5px] text-fg-3">Nothing is waiting.</p>}
          <ul className="space-y-1">
            {(backlog.data?.items ?? []).map((b) => (
              <li key={`${b.kind}-${b.id}`}>
                <Link to={backlogHref(b, isPlatformAdmin)} className="flex items-center gap-2 rounded-md border border-line bg-surface px-2.5 py-1.5 text-[12.5px] hover:bg-inset">
                  <StatusPill variant="neutral" raw={b.kind}>{BACKLOG_KIND[b.kind] ?? b.kind}</StatusPill>
                  <span className="min-w-0 flex-1 truncate text-fg">{b.subject || "—"}</span>
                  <span className="whitespace-nowrap text-fg-3">{fmtRelative(b.created_at)}</span>
                </Link>
              </li>
            ))}
          </ul>
        </aside>
      </div>
    </div>
  );
}
