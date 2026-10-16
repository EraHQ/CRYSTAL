// Conflicts — Console Refine PR 7 (docs/CONSOLE_UX_AUDIT.md 5.5).
//
// Two facts the bank holds that cannot both be true. The decision UI
// keeps its shape (two claims, one verb per side) and gains what the
// decision needs: when each claim was learned, where it came from, which
// is newer, one primary verb per side, Undo on every decision (the new
// /reopen route), a Resolved tab with Reopen, and a Backlog column whose
// rows go where each item is handled. Claims stack under 720px.
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Layers, RefreshCw, Scale } from "lucide-react";
import { api, errorMessage } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useSelectedCustomer } from "@/lib/selected-customer";
import { EmptyState, ErrorBanner, FilterChips, PageHeader, Skeleton, StatusPill, useToast } from "@/components/ui";
import type { StatusVariant } from "@/components/ui";
import type { BacklogItem, ConflictFact, KnowledgeConflict } from "@/lib/types";
import { cn, fmtRelative } from "@/lib/utils";

// Detectors (models/knowledge_conflict.py ConflictDetector) in plain words.
const KIND: Record<string, string> = { contradiction_scan: "Contradiction", dedup_scan: "Duplicate" };
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

function ConflictCard({ c, busy, onDecide, onReopen }: {
  c: KnowledgeConflict; busy: boolean;
  onDecide: (resolution: string, loser?: Side) => void; onReopen: () => void;
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
      setScanReport(`Scanned ${(s.facts_scanned ?? 0).toLocaleString()} facts, ${(s.pairs_evaluated ?? 0).toLocaleString()} pairs checked: ${n} new conflict${n === 1 ? "" : "s"}.`);
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
  const busy = decide.isPending || reopen.isPending || scan.isPending;

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
              <ConflictCard key={c.id} c={c} busy={busy}
                onDecide={(resolution, loser) => decide.mutate({ id: c.id, resolution, loser })}
                onReopen={() => reopen.mutate(c.id)} />
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
