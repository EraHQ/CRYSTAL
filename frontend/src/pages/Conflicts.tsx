import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";
import {
  Scale,
  RefreshCw,
  Layers,
  ArrowRightLeft,
  Ban,
  Check,
  X,
  Equal,
} from "lucide-react";

function TimeAgo({ iso }: { iso: string | null }) {
  if (!iso) return <span className="text-gray-400">—</span>;
  const d = new Date(iso);
  const mins = Math.floor((Date.now() - d.getTime()) / 60000);
  if (mins < 1) return <span className="text-gray-500">just now</span>;
  if (mins < 60) return <span className="text-gray-500">{mins}m ago</span>;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return <span className="text-gray-500">{hrs}h ago</span>;
  return <span className="text-gray-500">{d.toLocaleDateString()}</span>;
}

function EmptyState({
  icon: Icon,
  title,
  description,
}: {
  icon: any;
  title: string;
  description: string;
}) {
  return (
    <div className="flex flex-col items-center justify-center py-8 text-center">
      <Icon className="h-8 w-8 text-gray-300 mb-2" />
      <p className="text-sm font-medium text-gray-500">{title}</p>
      <p className="text-xs text-gray-400 mt-1">{description}</p>
    </div>
  );
}

export function Conflicts() {
  const { selectedCustomerId } = useSelectedCustomer();
  const queryClient = useQueryClient();
  const [scanResult, setScanResult] = useState<string | null>(null);

  const conflicts = useQuery({
    queryKey: ["conflicts", selectedCustomerId],
    queryFn: () => api.listConflicts(selectedCustomerId!, "open"),
    enabled: !!selectedCustomerId,
    refetchInterval: 15_000,
  });

  const backlog = useQuery({
    queryKey: ["backlog", selectedCustomerId],
    queryFn: () => api.listBacklog(selectedCustomerId!),
    enabled: !!selectedCustomerId,
    refetchInterval: 15_000,
  });

  const scanMutation = useMutation({
    mutationFn: () => api.scanConflicts(selectedCustomerId!),
    onSuccess: (data) => {
      const s = data.scan ?? {};
      setScanResult(
        `Scanned ${s.facts_scanned ?? 0} facts, ${s.pairs_evaluated ?? 0} pair(s) checked — ${s.conflicts_found ?? 0} new conflict(s).`
      );
      queryClient.invalidateQueries({ queryKey: ["conflicts"] });
      queryClient.invalidateQueries({ queryKey: ["backlog"] });
    },
    onError: (e: any) => setScanResult(`Scan failed: ${e?.message ?? "error"}`),
  });

  const resolveMutation = useMutation({
    mutationFn: ({
      conflictId,
      resolution,
      loser,
    }: {
      conflictId: string;
      resolution: string;
      loser?: "a" | "b";
    }) => api.resolveConflict(conflictId, resolution, loser),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["conflicts"] });
      queryClient.invalidateQueries({ queryKey: ["backlog"] });
    },
  });

  if (!selectedCustomerId) {
    return (
      <EmptyState
        icon={Scale}
        title="No customer selected"
        description="Pick a customer from the selector to review knowledge conflicts."
      />
    );
  }

  const items = conflicts.data?.items ?? [];
  const busy = resolveMutation.isPending || scanMutation.isPending;

  return (
    <div className="space-y-8">
      {/* Header + scan */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
            <Scale className="h-5 w-5 text-brand-500" /> Conflicts
          </h2>
          <p className="text-sm text-gray-500 mt-1 max-w-2xl">
            Two facts the bank holds that can't both be true. Surfacing is
            automatic; resolving is your call — supersede the outdated one,
            blacklist the wrong one, keep both if they're true under different
            conditions, or dismiss.
          </p>
        </div>
        <button
          onClick={() => scanMutation.mutate()}
          disabled={busy}
          className="inline-flex shrink-0 items-center gap-1.5 px-3 py-2 text-sm font-medium text-brand-700 bg-brand-50 hover:bg-brand-100 rounded-md transition-colors disabled:opacity-50"
        >
          <RefreshCw
            className={`h-4 w-4 ${scanMutation.isPending ? "animate-spin" : ""}`}
          />
          {scanMutation.isPending ? "Scanning…" : "Scan now"}
        </button>
      </div>

      {scanResult && (
        <div className="text-xs text-gray-500 bg-gray-50 border border-gray-200 rounded-md px-3 py-2">
          {scanResult}
        </div>
      )}

      {/* Summary */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-sm text-gray-500 mb-1">
            <Scale className="h-4 w-4" /> Open Conflicts
          </div>
          <div className="text-2xl font-semibold text-gray-900">
            {conflicts.data?.total ?? 0}
          </div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-sm text-gray-500 mb-1">
            <Layers className="h-4 w-4" /> Backlog Items
          </div>
          <div className="text-2xl font-semibold text-gray-900">
            {backlog.data?.total ?? 0}
          </div>
        </div>
      </div>

      {/* Conflicts list */}
      <div className="bg-white border border-gray-200 rounded-lg p-5">
        <div className="flex items-center gap-2 mb-3">
          <Scale className="h-4 w-4 text-gray-500" />
          <h3 className="text-sm font-semibold text-gray-900">Open Conflicts</h3>
          <span className="text-xs text-gray-400">({items.length})</span>
        </div>
        {conflicts.isLoading ? (
          <p className="text-sm text-gray-400">Loading…</p>
        ) : !items.length ? (
          <EmptyState
            icon={Check}
            title="No open conflicts"
            description="Run a scan, or the bank is internally consistent for now."
          />
        ) : (
          <div className="space-y-4">
            {items.map((c: any) => (
              <ConflictCard
                key={c.id}
                conflict={c}
                busy={busy}
                onResolve={(resolution, loser) =>
                  resolveMutation.mutate({ conflictId: c.id, resolution, loser })
                }
              />
            ))}
          </div>
        )}
      </div>

      {/* Backlog */}
      <div className="bg-white border border-gray-200 rounded-lg p-5">
        <div className="flex items-center gap-2 mb-3">
          <Layers className="h-4 w-4 text-gray-500" />
          <h3 className="text-sm font-semibold text-gray-900">Backlog</h3>
          <span className="text-xs text-gray-400">
            ({backlog.data?.total ?? 0})
          </span>
        </div>
        <p className="text-xs text-gray-400 mb-4">
          One ranked view of everything waiting — gaps, conflicts, tasks,
          review, verification — highest priority first.
        </p>
        {!backlog.data?.items.length ? (
          <EmptyState
            icon={Layers}
            title="Backlog empty"
            description="No pending work across the queues."
          />
        ) : (
          <div className="space-y-1.5">
            {backlog.data.items.map((b: any) => (
              <div
                key={`${b.kind}-${b.id}`}
                className="flex items-center gap-2 border border-gray-100 rounded-md px-3 py-2 text-sm"
              >
                <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-600">
                  {b.kind}
                </span>
                <span className="flex-1 min-w-0 truncate text-gray-700">
                  {b.subject || <span className="text-gray-400">—</span>}
                </span>
                <span className="text-xs text-gray-400">p{b.priority_score}</span>
                <TimeAgo iso={b.created_at} />
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ConflictCard({
  conflict: c,
  busy,
  onResolve,
}: {
  conflict: any;
  busy: boolean;
  onResolve: (resolution: string, loser?: "a" | "b") => void;
}) {
  return (
    <div className="border border-gray-200 rounded-lg p-4">
      <div className="flex items-center gap-2 mb-3">
        {c.subject && (
          <span className="text-xs font-medium text-gray-600">{c.subject}</span>
        )}
        <span className="text-xs text-gray-300">·</span>
        <span className="text-xs text-gray-400">{c.detector}</span>
      </div>

      <div className="grid grid-cols-2 gap-3">
        {(["a", "b"] as const).map((side) => {
          const claim = side === "a" ? c.claim_a : c.claim_b;
          const prov = side === "a" ? c.provenance_a : c.provenance_b;
          return (
            <div
              key={side}
              className="flex flex-col border border-gray-100 rounded-md p-3 bg-gray-50/60"
            >
              <div className="text-[10px] font-semibold uppercase tracking-wide text-gray-400 mb-1">
                Claim {side.toUpperCase()}
              </div>
              <p className="text-sm text-gray-800 flex-1">{claim}</p>
              {prov && <p className="text-xs text-gray-400 mt-2">{prov}</p>}
              <div className="flex items-center gap-1.5 mt-3">
                <button
                  onClick={() => onResolve("superseded", side)}
                  disabled={busy}
                  className="inline-flex items-center gap-1 px-2 py-1 text-xs font-medium text-amber-700 bg-amber-50 hover:bg-amber-100 rounded transition-colors disabled:opacity-50"
                  title={`Claim ${side.toUpperCase()} is outdated — deactivate it, keep the other`}
                >
                  <ArrowRightLeft className="h-3 w-3" /> Outdated
                </button>
                <button
                  onClick={() => onResolve("blacklisted", side)}
                  disabled={busy}
                  className="inline-flex items-center gap-1 px-2 py-1 text-xs font-medium text-red-700 bg-red-50 hover:bg-red-100 rounded transition-colors disabled:opacity-50"
                  title={`Claim ${side.toUpperCase()} is wrong — deactivate it and blacklist the claim`}
                >
                  <Ban className="h-3 w-3" /> Wrong
                </button>
              </div>
            </div>
          );
        })}
      </div>

      <div className="flex items-center gap-2 mt-3 pt-3 border-t border-gray-100">
        <button
          onClick={() => onResolve("qualified")}
          disabled={busy}
          className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-blue-700 bg-blue-50 hover:bg-blue-100 rounded-md transition-colors disabled:opacity-50"
          title="Both true under different conditions — keep both active"
        >
          <Equal className="h-3.5 w-3.5" /> Keep both
        </button>
        <button
          onClick={() => onResolve("dismissed")}
          disabled={busy}
          className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-gray-600 bg-gray-50 hover:bg-gray-100 rounded-md transition-colors disabled:opacity-50"
          title="Not a real conflict — dismiss"
        >
          <X className="h-3.5 w-3.5" /> Dismiss
        </button>
        <span className="ml-auto text-xs text-gray-400">
          <TimeAgo iso={c.created_at} />
        </span>
      </div>
    </div>
  );
}
