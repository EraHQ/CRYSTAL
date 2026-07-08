import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";
import { AlertCircle, CheckCircle2, XCircle, Clock, Search, Brain, Lightbulb, ChevronDown, ChevronUp, ArrowRight, ClipboardList } from "lucide-react";
import { CognitionTracker } from "./CognitionTracker";

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    pending: "bg-yellow-100 text-yellow-800",
    approved: "bg-green-100 text-green-800",
    rejected: "bg-red-100 text-red-800",
    open: "bg-blue-100 text-blue-800",
    acknowledged: "bg-gray-100 text-gray-600",
    filled: "bg-green-100 text-green-800",
    completed: "bg-green-100 text-green-800",
    running: "bg-blue-100 text-blue-800",
    failed: "bg-red-100 text-red-800",
    needs_document: "bg-orange-100 text-orange-800",
    needs_input: "bg-purple-100 text-purple-800",
  };
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${colors[status] ?? "bg-gray-100 text-gray-600"}`}>
      {status}
    </span>
  );
}

function TimeAgo({ iso }: { iso: string | null }) {
  if (!iso) return <span className="text-gray-400">—</span>;
  const d = new Date(iso);
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const mins = Math.floor(diffMs / 60000);
  if (mins < 1) return <span className="text-gray-500">just now</span>;
  if (mins < 60) return <span className="text-gray-500">{mins}m ago</span>;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return <span className="text-gray-500">{hrs}h ago</span>;
  return <span className="text-gray-500">{d.toLocaleDateString()}</span>;
}

function SectionHeader({ icon: Icon, title, count }: { icon: any; title: string; count: number }) {
  return (
    <div className="flex items-center gap-2 mb-3">
      <Icon className="h-4 w-4 text-gray-500" />
      <h3 className="text-sm font-semibold text-gray-900">{title}</h3>
      <span className="text-xs text-gray-400">({count})</span>
    </div>
  );
}

function EmptyState({ icon: Icon, title, description }: { icon: any; title: string; description: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-8 text-center">
      <Icon className="h-8 w-8 text-gray-300 mb-2" />
      <p className="text-sm font-medium text-gray-500">{title}</p>
      <p className="text-xs text-gray-400 mt-1">{description}</p>
    </div>
  );
}

function ExpandableText({ text, maxLength = 120 }: { text: string; maxLength?: number }) {
  const [expanded, setExpanded] = useState(false);
  if (text.length <= maxLength) return <span>{text}</span>;
  return (
    <span>
      {expanded ? text : text.slice(0, maxLength) + "..."}
      <button
        onClick={() => setExpanded(!expanded)}
        className="ml-1 text-brand-600 hover:text-brand-800 text-xs font-medium inline-flex items-center gap-0.5"
      >
        {expanded ? <><ChevronUp className="h-3 w-3" /> less</> : <><ChevronDown className="h-3 w-3" /> more</>}
      </button>
    </span>
  );
}


export function Cognition() {
  const { selectedCustomerId } = useSelectedCustomer();
  const queryClient = useQueryClient();

  // S4: manual gap promotion — the Research click enqueues a cognition
  // task; the worker fills the gap and closes it on success.
  const [promoting, setPromoting] = useState<string | null>(null);
  const promoteGap = async (gapId: string) => {
    setPromoting(gapId);
    try {
      await api.promoteGapToResearch(gapId);
      await queryClient.invalidateQueries({ queryKey: ["cognition-tasks"] });
    } finally {
      setPromoting(null);
    }
  };

  const reviewQueue = useQuery({
    queryKey: ["review-queue", selectedCustomerId],
    queryFn: () => api.listReviewQueue(selectedCustomerId!),
    enabled: !!selectedCustomerId,
    refetchInterval: 10_000,
  });

  const gaps = useQuery({
    queryKey: ["knowledge-gaps", selectedCustomerId],
    queryFn: () => api.listKnowledgeGaps(selectedCustomerId!),
    enabled: !!selectedCustomerId,
    refetchInterval: 10_000,
  });

  const tasks = useQuery({
    queryKey: ["cognition-tasks", selectedCustomerId],
    queryFn: () => api.listCognitionTasks(selectedCustomerId!),
    enabled: !!selectedCustomerId,
    refetchInterval: 10_000,
  });

  const approveMutation = useMutation({
    mutationFn: (itemId: string) => api.approveReviewItem(selectedCustomerId!, itemId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["review-queue"] }),
  });

  const rejectMutation = useMutation({
    mutationFn: (itemId: string) => api.rejectReviewItem(selectedCustomerId!, itemId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["review-queue"] }),
  });

  if (!selectedCustomerId) {
    return (
      <EmptyState
        icon={Brain}
        title="No customer selected"
        description="Pick a customer from the selector to view cognition activity."
      />
    );
  }

  const pendingCount = reviewQueue.data?.items.filter((i: any) => i.status === "pending").length ?? 0;
  const openGaps = gaps.data?.items.filter((i: any) => i.status === "open").length ?? 0;

  // S5 (2026-07-08, redesign P4): ONE gap, ONE pane. "Your Tasks" is
  // reserved for gaps only the human can close (needs_document, open);
  // everything else renders in Knowledge Gaps with its disposition's
  // affordance. The old double-render put the same row in both panes
  // with contradictory instructions.
  const userTasks = (gaps.data?.items ?? [])
    .filter((g: any) => g.status === "open" && g.disposition === "needs_document")
    .map((g: any) => ({
      id: g.id,
      type: "knowledge_gap" as const,
      title: g.missing,
      subject: g.subject,
      domain: g.domain,
      created_at: g.created_at,
      action: "Upload a document or manually add this knowledge to fill the gap.",
    }));
  const gapPaneItems = (gaps.data?.items ?? []).filter(
    (g: any) => !(g.status === "open" && g.disposition === "needs_document")
  );
  // Feedback for the Research click (2026-07-08): a pending/running
  // cognition task whose payload names this gap = the gap is queued or
  // being researched — show that instead of re-offering the button.
  const gapTaskState = new Map<string, string>();
  for (const t of tasks.data?.items ?? []) {
    const gid = t?.payload?.gap_id;
    if (gid && (t.status === "pending" || t.status === "running")) {
      gapTaskState.set(gid, t.status);
    }
  }

  return (
    <div className="space-y-8">
      {/* Summary bar */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-sm text-gray-500 mb-1">
            <Clock className="h-4 w-4" />
            Pending Review
          </div>
          <div className="text-2xl font-semibold text-gray-900">{pendingCount}</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-sm text-gray-500 mb-1">
            <AlertCircle className="h-4 w-4" />
            Knowledge Gaps
          </div>
          <div className="text-2xl font-semibold text-gray-900">{openGaps}</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-sm text-gray-500 mb-1">
            <Search className="h-4 w-4" />
            Research Tasks
          </div>
          <div className="text-2xl font-semibold text-gray-900">{tasks.data?.total ?? 0}</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-sm text-gray-500 mb-1">
            <ClipboardList className="h-4 w-4" />
            Your Tasks
          </div>
          <div className="text-2xl font-semibold text-gray-900">{userTasks.length}</div>
        </div>
      </div>

      {/* Cognition Environments (live tracking) */}
      <CognitionTracker />

      {/* User Tasks */}
      {userTasks.length > 0 && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-5">
          <SectionHeader icon={ClipboardList} title="Your Tasks" count={userTasks.length} />
          <p className="text-xs text-amber-700 mb-4">
            Actions that need your input. The system identified these gaps that require additional documents or manual knowledge entry.
          </p>
          <div className="space-y-3">
            {userTasks.map((task: any) => (
              <div key={task.id} className="bg-white border border-amber-100 rounded-lg p-3">
                <div className="flex items-center gap-2 mb-1">
                  <StatusBadge status="needs_document" />
                  {task.domain && <span className="text-xs text-gray-400">{task.domain}</span>}
                  {task.subject && <span className="text-xs font-medium text-gray-600">{task.subject}</span>}
                </div>
                <p className="text-sm text-gray-700 mb-1">
                  <ExpandableText text={task.title} maxLength={200} />
                </p>
                <p className="text-xs text-amber-600 italic">{task.action}</p>
                <div className="text-xs text-gray-400 mt-1">
                  <TimeAgo iso={task.created_at} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Review Queue */}
      <div className="bg-white border border-gray-200 rounded-lg p-5">
        <SectionHeader icon={Lightbulb} title="Review Queue" count={reviewQueue.data?.total ?? 0} />
        <p className="text-xs text-gray-400 mb-4">
          Knowledge the LLM noticed but wasn't confident enough to auto-commit. Approve to create a crystal, reject to discard.
        </p>
        {!reviewQueue.data?.items.length ? (
          <EmptyState icon={Lightbulb} title="No items" description="The LLM hasn't pushed any medium-confidence observations yet." />
        ) : (
          <div className="space-y-3">
            {reviewQueue.data.items.map((item: any) => (
              <div key={item.id} className="border border-gray-100 rounded-lg p-3">
                <div className="flex items-start justify-between gap-3">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1 flex-wrap">
                      <code className="text-xs bg-gray-50 text-gray-700 px-1.5 py-0.5 rounded font-mono">
                        {item.key}
                      </code>
                      <StatusBadge status={item.status} />
                      <span className="text-xs text-gray-400">conf: {item.confidence}</span>
                    </div>
                    <p className="text-sm text-gray-600">
                      <ExpandableText text={item.value} maxLength={200} />
                    </p>
                    <div className="text-xs text-gray-400 mt-1">
                      <TimeAgo iso={item.created_at} />
                      {item.crystal_id && (
                        <span className="ml-2 text-green-600">
                          → crystal {item.crystal_id.slice(0, 20)}
                        </span>
                      )}
                    </div>
                  </div>
                  {item.status === "pending" && (
                    <div className="flex items-center gap-1.5 flex-shrink-0">
                      <button
                        onClick={() => approveMutation.mutate(item.id)}
                        disabled={approveMutation.isPending}
                        className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-green-700 bg-green-50 hover:bg-green-100 rounded-md transition-colors"
                      >
                        <CheckCircle2 className="h-3.5 w-3.5" />
                        Approve
                      </button>
                      <button
                        onClick={() => rejectMutation.mutate(item.id)}
                        disabled={rejectMutation.isPending}
                        className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-red-700 bg-red-50 hover:bg-red-100 rounded-md transition-colors"
                      >
                        <XCircle className="h-3.5 w-3.5" />
                        Reject
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Knowledge Gaps */}
      <div className="bg-white border border-gray-200 rounded-lg p-5">
        <SectionHeader icon={AlertCircle} title="Knowledge Gaps" count={gapPaneItems.length} />
        <p className="text-xs text-gray-400 mb-4">
          Missing knowledge identified while answering questions. Researchable and workable gaps can be promoted; document-needing gaps live under Your Tasks.
        </p>
        {!gapPaneItems.length ? (
          <EmptyState icon={AlertCircle} title="No gaps" description="The LLM hasn't identified any missing knowledge yet." />
        ) : (
          <div className="space-y-2">
            {gapPaneItems.map((item: any) => (
              <div key={item.id} className="border border-gray-100 rounded-lg p-3">
                <div className="flex items-start justify-between gap-3">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <StatusBadge status={item.status} />
                      {item.domain && <span className="text-xs text-gray-400">{item.domain}</span>}
                      {item.subject && <span className="text-xs font-medium text-gray-600">{item.subject}</span>}
                      {item.disposition && (
                        <span className={"text-[10px] font-medium rounded-full px-1.5 py-0.5 " +
                          (item.disposition === "needs_document"
                            ? "bg-amber-50 text-amber-700"
                            : item.disposition === "workable"
                              ? "bg-purple-50 text-purple-700"
                              : "bg-blue-50 text-blue-700")}>
                          {item.disposition}
                        </span>
                      )}
                      {item.filled_by_crystal_id && (
                        <span className="text-xs text-green-600">
                          → filled by {item.filled_by_crystal_id.slice(0, 20)}
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-gray-700">
                      <ExpandableText text={item.missing} maxLength={200} />
                    </p>
                    {/* S3 provenance (2026-07-08): the anchoring key and
                        the query that missed, when the gap carries them. */}
                    {item.full_key && (
                      <div className="mt-1 text-xs font-mono text-gray-500">
                        {item.full_key}
                      </div>
                    )}
                    {item.triggering_query && (
                      <div className="mt-1 text-xs text-gray-500 italic">
                        asked: “{item.triggering_query}”
                      </div>
                    )}
                    {item.filled_content && (
                      <div className="mt-2 p-2 bg-green-50 border border-green-100 rounded text-sm text-green-800">
                        <span className="text-xs font-medium text-green-600 block mb-1">Filled with:</span>
                        <ExpandableText text={item.filled_content} maxLength={300} />
                      </div>
                    )}
                    <div className="text-xs text-gray-400 mt-1">
                      <TimeAgo iso={item.created_at} />
                    </div>
                  </div>
                  {item.status === "open" && item.disposition !== "needs_document" && (
                    <div className="flex-shrink-0">
                      {gapTaskState.has(item.id) ? (
                        <span className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-indigo-700 bg-indigo-50 rounded-md">
                          <span className="h-1.5 w-1.5 rounded-full bg-indigo-500 animate-pulse" />
                          {gapTaskState.get(item.id) === "running" ? "Researching…" : "Queued"}
                        </span>
                      ) : (
                        <button
                          className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-blue-700 bg-blue-50 hover:bg-blue-100 rounded-md transition-colors disabled:opacity-50"
                          title="Enqueue a research task for this gap"
                          disabled={promoting === item.id}
                          onClick={() => void promoteGap(item.id)}
                        >
                          <ArrowRight className="h-3.5 w-3.5" />
                          {promoting === item.id ? "Queued…" : "Research"}
                        </button>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Research Tasks */}
      <div className="bg-white border border-gray-200 rounded-lg p-5">
        <SectionHeader icon={Search} title="Research Tasks" count={tasks.data?.total ?? 0} />
        <p className="text-xs text-gray-400 mb-4">
          Research and analysis tasks requested by the LLM or queued for background processing.
        </p>
        {!tasks.data?.items.length ? (
          <EmptyState icon={Search} title="No tasks" description="No research requests have been made yet." />
        ) : (
          <div className="space-y-3">
            {tasks.data.items.map((item: any) => (
              <div key={item.id} className="border border-gray-100 rounded-lg p-3">
                <div className="flex items-center gap-2 mb-1">
                  <StatusBadge status={item.status} />
                  <span className="text-xs text-gray-400">{item.task_type}</span>
                  <span className="text-xs text-gray-400">priority: {item.priority}</span>
                </div>
                <p className="text-sm text-gray-700 mb-1">
                  <ExpandableText
                    text={item.payload?.topic ?? JSON.stringify(item.payload)}
                    maxLength={150}
                  />
                </p>
                {item.result && (
                  <div className="mt-2 space-y-2">
                    {item.result.action && (
                      <div className="flex items-center gap-2">
                        <StatusBadge status={item.result.action === "inferred_fact_created" ? "approved" : "needs_document"} />
                        <span className="text-xs text-gray-500">
                          {item.result.action === "inferred_fact_created"
                            ? `Inferred fact created → ${item.result.crystal_id?.slice(0, 20)}`
                            : item.result.recommendation || "No actionable findings"}
                        </span>
                        {typeof item.result.confidence === "number" && (
                          <span className="text-[10px] text-gray-400">
                            confidence {(item.result.confidence * 100).toFixed(0)}%
                          </span>
                        )}
                      </div>
                    )}
                    {item.result.reason && (
                      <div className="p-2 bg-gray-50 border border-gray-100 rounded">
                        <span className="text-xs font-medium text-gray-500 block mb-1">Why:</span>
                        <p className="text-xs text-gray-600">
                          <ExpandableText text={item.result.reason} maxLength={240} />
                        </p>
                      </div>
                    )}
                    {item.result.gap_disposition === "needs_document" && (
                      <p className="text-[11px] text-amber-600 italic">
                        The originating gap was moved to Your Tasks — research concluded it needs a document.
                      </p>
                    )}
                    {item.result.findings && (
                      <div className="p-2 bg-gray-50 border border-gray-100 rounded">
                        <span className="text-xs font-medium text-gray-500 block mb-1">Research findings:</span>
                        <p className="text-sm text-gray-700">
                          <ExpandableText text={item.result.findings} maxLength={300} />
                        </p>
                      </div>
                    )}
                    {!item.result.findings && !item.result.action && (
                      <div className="p-2 bg-gray-50 border border-gray-100 rounded">
                        <span className="text-xs font-medium text-gray-500 block mb-1">Raw result:</span>
                        <p className="text-xs text-gray-600 font-mono">
                          <ExpandableText text={JSON.stringify(item.result)} maxLength={200} />
                        </p>
                      </div>
                    )}
                  </div>
                )}
                {item.error_message && (
                  <p className="text-xs text-red-500 mt-1 bg-red-50 p-2 rounded">
                    {item.error_message}
                  </p>
                )}
                <div className="text-xs text-gray-400 mt-2">
                  <TimeAgo iso={item.created_at} />
                  {item.completed_at && (
                    <span className="ml-2">
                      completed <TimeAgo iso={item.completed_at} />
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
