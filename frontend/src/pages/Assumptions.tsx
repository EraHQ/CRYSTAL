import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";
import {
  Lightbulb,
  Check,
  Trash2,
  ShieldAlert,
  Link2,
  CircleDashed,
  History,
  FlaskConical,
  MessageSquareReply,
  CornerDownRight,
  GitBranch,
} from "lucide-react";

function TimeAgo({ iso }: { iso: string | null }) {
  if (!iso) return <span className="text-gray-400">n/a</span>;
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

function ConfidenceBadge({ value }: { value: number | null }) {
  if (value == null) return null;
  const pct = Math.round(value * 100);
  const tone =
    value >= 0.75
      ? "bg-green-50 text-green-700 border-green-200"
      : value >= 0.5
        ? "bg-amber-50 text-amber-700 border-amber-200"
        : "bg-gray-50 text-gray-600 border-gray-200";
  return (
    <span
      className={`inline-flex items-center px-1.5 py-0.5 rounded text-[11px] font-medium border ${tone}`}
    >
      {pct}% confidence
    </span>
  );
}

function AssumptionCard({
  item,
  busy,
  depth = 0,
  onApprove,
  onDelete,
  onVerify,
  onRespond,
}: {
  item: any;
  busy: boolean;
  depth?: number;
  onApprove: () => void;
  onDelete: () => void;
  onVerify: () => void;
  onRespond: (text: string) => Promise<void>;
}) {
  const invalidated = item.quality_tier === "blacklist";
  const superseded = invalidated && !!item.superseded_by;
  const approved = !item.recall_gated && !invalidated;
  const [responding, setResponding] = useState(false);
  const [text, setText] = useState("");
  const [sending, setSending] = useState(false);
  const submit = async () => {
    if (!text.trim()) return;
    setSending(true);
    try {
      await onRespond(text.trim());
      setText("");
      setResponding(false);
    } finally {
      setSending(false);
    }
  };
  const verificationQueued = (item.diagnostic_tags ?? []).some(
    (t: string) => typeof t === "string" && t.startsWith("verification_task:")
  );
  return (
    <div
      className={`border rounded-lg p-4 ${
        superseded
          ? "border-gray-200 bg-gray-50/60"
          : invalidated ? "border-red-200 bg-red-50/40" : "border-gray-200 bg-white"
      }`}
      style={depth ? { marginLeft: Math.min(depth, 4) * 20 } : undefined}
    >
      {item.operator_response && (
        <div className="mb-2 flex items-start gap-2 rounded-md bg-indigo-50/60 px-2.5 py-1.5 text-[12px] text-indigo-900">
          <CornerDownRight className="mt-0.5 h-3.5 w-3.5 shrink-0 text-indigo-400" />
          <span>
            <span className="font-medium">You responded:</span> {item.operator_response}
          </span>
        </div>
      )}
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2 mb-1.5">
            {superseded ? (
              <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[11px] font-medium bg-gray-100 text-gray-600 border border-gray-200">
                <GitBranch className="h-3 w-3" /> Superseded
              </span>
            ) : invalidated ? (
              <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[11px] font-medium bg-red-50 text-red-700 border border-red-200">
                <ShieldAlert className="h-3 w-3" /> Invalidated
              </span>
            ) : approved ? (
              <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[11px] font-medium bg-green-50 text-green-700 border border-green-200">
                <Check className="h-3 w-3" /> Approved
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[11px] font-medium bg-amber-50 text-amber-700 border border-amber-200">
                <CircleDashed className="h-3 w-3" /> Pending review
              </span>
            )}
            <ConfidenceBadge value={item.confidence} />
            {item.responds_to && (
              <span className="text-[11px] text-gray-400">
                {item.replaces_parent ? "revises the parent" : "additional claim"}
              </span>
            )}
            {item.gap_id && (
              <span className="text-[11px] text-gray-400">
                gap-seeded, {item.gap_id}
              </span>
            )}
          </div>
          <p className="text-sm text-gray-900">{item.statement}</p>
          {item.rationale && (
            <p className="mt-1 text-[12px] italic text-gray-500">{item.rationale}</p>
          )}
          <div className="mt-2 flex flex-wrap items-center gap-1.5">
            <Link2 className="h-3.5 w-3.5 text-gray-400" />
            {(item.parents ?? []).map((p: any) => (
              <span
                key={p.id}
                title={p.id}
                className="inline-flex max-w-[16rem] truncate px-1.5 py-0.5 rounded text-[11px] bg-gray-50 text-gray-600 border border-gray-200"
              >
                {p.summary_text || p.id}
              </span>
            ))}
            {(item.invalidated_parents ?? []).map((id: string) => (
              <span
                key={id}
                className="inline-flex px-1.5 py-0.5 rounded text-[11px] bg-red-50 text-red-600 border border-red-200 line-through"
              >
                {id}
              </span>
            ))}
          </div>
          <div className="mt-2 text-[11px] text-gray-400">
            <TimeAgo iso={item.created_at} />
          </div>
        </div>
        <div className="flex shrink-0 flex-col items-stretch gap-2">
          {!invalidated && !approved && (
            <button
              onClick={onApprove}
              disabled={busy}
              className="inline-flex items-center justify-center gap-1.5 px-3 py-1.5 text-xs font-medium text-green-700 bg-green-50 hover:bg-green-100 border border-green-200 rounded-md transition-colors disabled:opacity-50"
            >
              <Check className="h-3.5 w-3.5" /> Approve
            </button>
          )}
          {!invalidated && (
            <button
              onClick={onVerify}
              disabled={busy || verificationQueued}
              title={
                verificationQueued
                  ? "A verification task has already been queued"
                  : "Queue a research task to confirm or refute this inference"
              }
              className="inline-flex items-center justify-center gap-1.5 px-3 py-1.5 text-xs font-medium text-blue-700 bg-blue-50 hover:bg-blue-100 border border-blue-200 rounded-md transition-colors disabled:opacity-50"
            >
              <FlaskConical className="h-3.5 w-3.5" />{" "}
              {verificationQueued ? "Verifying" : "Verify"}
            </button>
          )}
          {!invalidated && (
            <button
              onClick={() => setResponding((v) => !v)}
              disabled={busy}
              title="Correct this assumption; the model reasons over your response and proposes revisions"
              className="inline-flex items-center justify-center gap-1.5 px-3 py-1.5 text-xs font-medium text-indigo-700 bg-indigo-50 hover:bg-indigo-100 border border-indigo-200 rounded-md transition-colors disabled:opacity-50"
            >
              <MessageSquareReply className="h-3.5 w-3.5" /> Respond
            </button>
          )}
          <button
            onClick={onDelete}
            disabled={busy}
            className="inline-flex items-center justify-center gap-1.5 px-3 py-1.5 text-xs font-medium text-red-700 bg-white hover:bg-red-50 border border-red-200 rounded-md transition-colors disabled:opacity-50"
          >
            <Trash2 className="h-3.5 w-3.5" /> Delete
          </button>
        </div>
      </div>
      {responding && (
        <div className="mt-3 border-t border-gray-100 pt-3">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={3}
            maxLength={4000}
            placeholder="What is actually true? The model reads this with the assumption and its evidence, then proposes revised assumptions you can approve, respond to again, or delete."
            className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm text-gray-900 focus:border-indigo-400 focus:outline-none"
          />
          <div className="mt-2 flex items-center justify-end gap-2">
            <span className="mr-auto text-[11px] text-gray-400">
              Uses daily AI capacity. Nothing is deleted; revisions join this thread.
            </span>
            <button
              onClick={() => { setResponding(false); setText(""); }}
              disabled={sending}
              className="rounded-md border border-gray-300 px-3 py-1.5 text-xs text-gray-700 hover:bg-gray-50"
            >
              Cancel
            </button>
            <button
              onClick={submit}
              disabled={sending || !text.trim()}
              className="rounded-md bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-indigo-700 disabled:opacity-50"
            >
              {sending ? "Reasoning..." : "Send response"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// v110: a thread is the root assumption plus every revision tagged with
// it. Render root first, then revisions by iteration, indented by how
// far they are from the root.
function threadsOf(items: any[]): { root: any; members: any[] }[] {
  const byRoot = new Map<string, any[]>();
  for (const it of items) {
    const root = it.thread_id || it.id;
    if (!byRoot.has(root)) byRoot.set(root, []);
    byRoot.get(root)!.push(it);
  }
  const out: { root: any; members: any[] }[] = [];
  for (const [rootId, members] of byRoot) {
    const root = members.find((m) => m.id === rootId) ?? members[0];
    const rest = members
      .filter((m) => m !== root)
      .sort((a, b) => (a.iteration ?? 0) - (b.iteration ?? 0) || String(a.created_at).localeCompare(String(b.created_at)));
    out.push({ root, members: rest });
  }
  // Newest root first, matching the list's own ordering.
  out.sort((a, b) => String(b.root.created_at).localeCompare(String(a.root.created_at)));
  return out;
}

function depthOf(item: any, byId: Map<string, any>): number {
  let d = 0;
  let cur = item;
  const seen = new Set<string>();
  while (cur?.responds_to && !seen.has(cur.responds_to)) {
    seen.add(cur.responds_to);
    cur = byId.get(cur.responds_to);
    d += 1;
    if (!cur) break;
  }
  return d;
}

export function Assumptions() {
  const { selectedCustomerId } = useSelectedCustomer();
  const queryClient = useQueryClient();
  const [actionError, setActionError] = useState<string | null>(null);

  const assumptions = useQuery({
    queryKey: ["assumptions", selectedCustomerId],
    queryFn: () => api.listAssumptions(selectedCustomerId!),
    enabled: !!selectedCustomerId,
    refetchInterval: 15_000,
  });

  const activity = useQuery({
    queryKey: ["curation-activity", selectedCustomerId],
    queryFn: () => api.listCurationActivity(selectedCustomerId!),
    enabled: !!selectedCustomerId,
    refetchInterval: 15_000,
  });

  const approveMutation = useMutation({
    mutationFn: (crystalId: string) => api.approveAssumption(crystalId),
    onSuccess: () => {
      setActionError(null);
      queryClient.invalidateQueries({ queryKey: ["assumptions"] });
      queryClient.invalidateQueries({ queryKey: ["assumption-threads"] });
      queryClient.invalidateQueries({ queryKey: ["curation-activity"] });
    },
    onError: (e: any) =>
      setActionError(`Approve failed: ${e?.message ?? "unknown error"}`),
  });

  const verifyMutation = useMutation({
    mutationFn: (crystalId: string) => api.verifyAssumption(crystalId),
    onSuccess: () => {
      setActionError(null);
      queryClient.invalidateQueries({ queryKey: ["assumptions"] });
      queryClient.invalidateQueries({ queryKey: ["curation-activity"] });
    },
    onError: (e: any) =>
      setActionError(`Verify failed: ${e?.message ?? "unknown error"}`),
  });

  const respondMutation = useMutation({
    mutationFn: ({ crystalId, text }: { crystalId: string; text: string }) =>
      api.respondAssumption(crystalId, text),
    onSuccess: () => {
      setActionError(null);
      queryClient.invalidateQueries({ queryKey: ["assumptions"] });
      queryClient.invalidateQueries({ queryKey: ["assumption-threads"] });
      queryClient.invalidateQueries({ queryKey: ["curation-activity"] });
    },
    onError: (e: any) =>
      setActionError(`Respond failed: ${e?.message ?? "unknown error"}`),
  });

  const deleteMutation = useMutation({
    mutationFn: (crystalId: string) => api.deleteCrystal(crystalId),
    onSuccess: () => {
      setActionError(null);
      queryClient.invalidateQueries({ queryKey: ["assumptions"] });
      queryClient.invalidateQueries({ queryKey: ["assumption-threads"] });
      queryClient.invalidateQueries({ queryKey: ["curation-activity"] });
    },
    onError: (e: any) =>
      setActionError(`Delete failed: ${e?.message ?? "unknown error"}`),
  });

  const items = assumptions.data?.items ?? [];
  // Threads with revisions carry operator responses only on the thread
  // read; fetch those for every root that has members.
  const threads = threadsOf(items);
  const threadRoots = threads.filter((t) => t.members.length).map((t) => t.root.id);
  const threadReads = useQuery({
    queryKey: ["assumption-threads", selectedCustomerId, threadRoots.join(",")],
    queryFn: async () => {
      const out: Record<string, any> = {};
      for (const root of threadRoots) {
        const t = await api.assumptionThread(root);
        for (const m of t.thread) out[m.id] = m;
      }
      return out;
    },
    enabled: !!selectedCustomerId && threadRoots.length > 0,
  });
  const enrich = (it: any) => ({ ...it, ...(threadReads.data?.[it.id] ?? {}) });

  if (!selectedCustomerId) {
    return (
      <EmptyState
        icon={Lightbulb}
        title="No customer selected"
        description="Pick a customer from the selector to review assumptions."
      />
    );
  }

  const pending = items.filter(
    (i: any) => i.recall_gated && i.quality_tier !== "blacklist"
  );
  const invalidated = items.filter(
    (i: any) => i.quality_tier === "blacklist"
  );
  const busy = approveMutation.isPending || deleteMutation.isPending || verifyMutation.isPending || respondMutation.isPending;
  const respond = (crystalId: string) => async (text: string) => {
    await respondMutation.mutateAsync({ crystalId, text });
  };
  const confirmDelete = (id: string) => {
    if (window.confirm("Delete this assumption crystal? This is permanent.")) {
      deleteMutation.mutate(id);
    }
  };

  return (
    <div className="space-y-8">
      {actionError && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700">{actionError}</div>
      )}
      {/* Header */}
      <div>
        <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
          <Lightbulb className="h-5 w-5 text-brand-500" /> Assumptions
        </h2>
        <p className="text-sm text-gray-500 mt-1 max-w-2xl">
          Bridging inferences the system drew from pairs of crystals, held
          out of recall until you approve them. Respond to correct one: the
          model reasons over your response and the evidence and proposes
          revisions that join the thread. Approving a revision supersedes
          the versions it replaces; additional claims stay open. If a parent
          crystal dies, the assumption is invalidated automatically and
          stays here as the record.
        </p>
      </div>

      {/* Summary */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-sm text-gray-500 mb-1">
            <CircleDashed className="h-4 w-4" /> Pending Review
          </div>
          <div className="text-2xl font-semibold text-gray-900">
            {pending.length}
          </div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-sm text-gray-500 mb-1">
            <ShieldAlert className="h-4 w-4" /> Invalidated or superseded
          </div>
          <div className="text-2xl font-semibold text-gray-900">
            {invalidated.length}
          </div>
        </div>
      </div>

      {/* List */}
      <div className="bg-white border border-gray-200 rounded-lg p-5">
        <div className="flex items-center gap-2 mb-3">
          <Lightbulb className="h-4 w-4 text-gray-500" />
          <h3 className="text-sm font-semibold text-gray-900">Assumptions</h3>
          <span className="text-xs text-gray-400">({items.length})</span>
        </div>
        {assumptions.isLoading ? (
          <p className="text-sm text-gray-400">Loading...</p>
        ) : !items.length ? (
          <EmptyState
            icon={Lightbulb}
            title="No assumptions yet"
            description="The assumptions worker infers bridges from chained crystal pairs and open gaps; the agent can also propose them via the assume tool."
          />
        ) : (
          <div className="space-y-4">
            {threads.map(({ root, members }) => {
              const byId = new Map<string, any>([root, ...members].map((m) => [m.id, m]));
              return (
                <div key={root.id} className="space-y-2">
                  <AssumptionCard
                    item={enrich(root)}
                    busy={busy}
                    onApprove={() => approveMutation.mutate(root.id)}
                    onVerify={() => verifyMutation.mutate(root.id)}
                    onDelete={() => confirmDelete(root.id)}
                    onRespond={respond(root.id)}
                  />
                  {members.map((m) => (
                    <AssumptionCard
                      key={m.id}
                      item={enrich(m)}
                      depth={depthOf(m, byId)}
                      busy={busy}
                      onApprove={() => approveMutation.mutate(m.id)}
                      onVerify={() => verifyMutation.mutate(m.id)}
                      onDelete={() => confirmDelete(m.id)}
                      onRespond={respond(m.id)}
                    />
                  ))}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Activity: the self-curation witness feed (C2 Q3=A). */}
      <div className="bg-white border border-gray-200 rounded-lg p-5">
        <div className="flex items-center gap-2 mb-3">
          <History className="h-4 w-4 text-gray-500" />
          <h3 className="text-sm font-semibold text-gray-900">Activity</h3>
          <span className="text-xs text-gray-400">
            ({activity.data?.events?.length ?? 0})
          </span>
        </div>
        {activity.isLoading ? (
          <p className="text-sm text-gray-400">Loading...</p>
        ) : !(activity.data?.events?.length) ? (
          <EmptyState
            icon={History}
            title="No activity yet"
            description="Assumption and gap lifecycle events land here the moment they happen; nothing the system does to its own knowledge is silent."
          />
        ) : (
          <div className="max-h-72 overflow-y-auto divide-y divide-gray-100">
            {activity.data!.events.map((e: any) => (
              <div key={e.id} className="py-2 flex items-start gap-3">
                <span className="mt-0.5 inline-block text-[10px] font-medium uppercase tracking-wide text-gray-500 bg-gray-100 rounded px-1.5 py-0.5 whitespace-nowrap">
                  {String(e.event_type ?? "").replace(/_/g, " ")}
                </span>
                <span className="text-sm text-gray-700 flex-1">
                  {e.label || e.event_type}
                </span>
                <span className="text-xs text-gray-400 whitespace-nowrap">
                  <TimeAgo iso={e.created_at ?? null} />
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
