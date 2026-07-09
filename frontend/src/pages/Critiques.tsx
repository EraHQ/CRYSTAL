// System Critiques (S6, 2026-07-08) — the surface for CRYS's structured
// complaints about ITSELF. The substrate channel reaches every part of
// the system that affects outcomes: tool capability wishes, ingestion
// artifacts, retrieval quality, metacognition misses. Observations are
// recorded and surfaced, never auto-acted (MCR Principle 9).
import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { MessageSquareWarning, ChevronDown, ChevronRight, X } from "lucide-react";
import { api } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";

function SeverityChip({ severity }: { severity: string }) {
  const cls =
    severity === "high"
      ? "bg-red-50 text-red-700"
      : severity === "medium"
        ? "bg-amber-50 text-amber-700"
        : "bg-gray-100 text-gray-600";
  return (
    <span className={`text-[10px] font-medium rounded-full px-1.5 py-0.5 ${cls}`}>
      {severity}
    </span>
  );
}

function TimeAgo({ iso }: { iso?: string }) {
  if (!iso) return null;
  const secs = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  const label =
    secs < 3600 ? `${Math.floor(secs / 60)}m ago`
    : secs < 86400 ? `${Math.floor(secs / 3600)}h ago`
    : `${Math.floor(secs / 86400)}d ago`;
  return <span className="text-xs text-gray-400">{label}</span>;
}

export function Critiques() {
  const { selectedCustomerId } = useSelectedCustomer();
  const queryClient = useQueryClient();
  const [openGroup, setOpenGroup] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  // S11: two critic streams share this page — substrate (complaints
  // about the SYSTEM) and quality (the critics' verdicts on the
  // agent's own responses).
  const [stream, setStream] = useState<"substrate" | "quality">("substrate");

  const refresh = () =>
    queryClient.invalidateQueries({ queryKey: ["substrate-grouped"] }).then(
      () => queryClient.invalidateQueries({ queryKey: ["substrate-flat"] })
    );
  const dismiss = async (itemId: string) => {
    setBusy(itemId);
    try {
      await api.dismissSubstrateObservation(itemId);
      await refresh();
    } finally {
      setBusy(null);
    }
  };
  const dismissAll = async () => {
    setBusy("__all__");
    try {
      await api.dismissAllSubstrateObservations(selectedCustomerId!);
      await refresh();
    } finally {
      setBusy(null);
    }
  };

  const grouped = useQuery({
    queryKey: ["substrate-grouped", selectedCustomerId],
    queryFn: () => api.groupedSubstrateObservations(selectedCustomerId!),
    enabled: !!selectedCustomerId,
    refetchInterval: 15_000,
  });
  const flat = useQuery({
    queryKey: ["substrate-flat", selectedCustomerId],
    queryFn: () => api.listSubstrateObservations(selectedCustomerId!),
    enabled: !!selectedCustomerId,
    refetchInterval: 15_000,
  });

  const groups = grouped.data?.groups ?? [];
  const observations = flat.data?.observations ?? [];
  const bySubsystem = (subsystem: string) =>
    observations.filter(
      (o: any) => (o.action_item?.content?.subsystem ?? "unspecified") === subsystem
    );

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-gray-900">System Critiques</h1>
          <p className="text-sm text-gray-500 mt-1">
            CRYS's structured complaints about its own system — tools it wishes
            were more capable, ingestion artifacts, retrieval friction,
            metacognition misses. Observations are recorded and surfaced, never
            auto-acted. Dismissing hides an observation; the record survives.
          </p>
        </div>
        {stream === "substrate" && groups.length > 0 && (
          <button
            className="flex-shrink-0 rounded-lg border border-gray-300 px-3 py-1.5 text-xs font-medium text-gray-600 hover:bg-gray-50 disabled:opacity-40"
            disabled={busy !== null}
            onClick={() => void dismissAll()}
          >
            {busy === "__all__" ? "Clearing…" : "Clear all"}
          </button>
        )}
      </div>

      <div className="flex gap-1 border-b border-gray-200">
        {([
          ["substrate", "Substrate"],
          ["quality", "Response Quality"],
        ] as const).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setStream(key)}
            className={`px-3 py-1.5 text-sm font-medium rounded-t-md border-b-2 -mb-px ${
              stream === key
                ? "border-indigo-500 text-indigo-700"
                : "border-transparent text-gray-500 hover:text-gray-700"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {stream === "quality" ? (
        <QualitySection customerId={selectedCustomerId!} />
      ) : !groups.length ? (
        <div className="bg-white border border-gray-200 rounded-lg p-10 text-center">
          <MessageSquareWarning className="h-8 w-8 text-gray-300 mx-auto mb-2" />
          <div className="text-sm font-medium text-gray-700">No critiques yet</div>
          <div className="text-xs text-gray-400 mt-1">
            When the agent or the structural scanners find something in the
            system worth complaining about, it lands here.
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          {groups.map((g: any) => {
            const open = openGroup === g.subsystem;
            return (
              <div key={g.subsystem} className="bg-white border border-gray-200 rounded-lg">
                <button
                  className="w-full flex items-center justify-between p-4 text-left"
                  onClick={() => setOpenGroup(open ? null : g.subsystem)}
                >
                  <div className="flex items-center gap-3 min-w-0">
                    {open ? <ChevronDown className="h-4 w-4 text-gray-400" /> : <ChevronRight className="h-4 w-4 text-gray-400" />}
                    <span className="text-sm font-semibold text-gray-900">{g.subsystem}</span>
                    <span className="text-xs text-gray-400">{g.count} observation{g.count === 1 ? "" : "s"}</span>
                    {Object.entries(g.severities ?? {}).map(([sev, n]: any) =>
                      n > 0 ? <SeverityChip key={sev} severity={sev} /> : null
                    )}
                  </div>
                  <TimeAgo iso={g.latest_at} />
                </button>
                {!open && g.latest_complaint && (
                  <div className="px-11 pb-3 text-xs text-gray-500 truncate">
                    {g.latest_complaint}
                  </div>
                )}
                {open && (
                  <div className="border-t border-gray-100 divide-y divide-gray-50">
                    {bySubsystem(g.subsystem).map((o: any) => (
                      <div key={o.action_item.id} className="p-4 pl-11 relative group">
                        <button
                          className="absolute top-3 right-3 p-1 rounded text-gray-300 hover:text-gray-600 hover:bg-gray-100 disabled:opacity-40"
                          title="Dismiss (hides; record survives)"
                          disabled={busy !== null}
                          onClick={() => void dismiss(o.action_item.id)}
                        >
                          <X className="h-3.5 w-3.5" />
                        </button>
                        <div className="flex items-center gap-2 mb-1">
                          <SeverityChip severity={o.action_item?.content?.severity ?? "low"} />
                          <span className="text-[11px] text-gray-400">
                            critic: {o.critique?.critic_role ?? "unknown"}
                            {o.critique?.critic_model ? ` (${o.critique.critic_model})` : ""}
                          </span>
                          <TimeAgo iso={o.action_item?.created_at} />
                        </div>
                        <p className="text-sm text-gray-700">
                          {o.action_item?.content?.complaint ?? "(no complaint text)"}
                        </p>
                        {o.action_item?.content?.crystal_id && (
                          <div className="mt-1 text-xs font-mono text-gray-400">
                            {o.action_item.content.crystal_id}
                          </div>
                        )}
                        {o.trace_summary?.user_query && (
                          <div className="mt-1 text-xs text-gray-500 italic">
                            during: “{o.trace_summary.user_query.slice(0, 140)}”
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// S11 (2026-07-09) — the response-quality stream. Read-only by design:
// observations live inside critique rows and carry no dismissal state;
// this surface exists so the operator can SEE what the shadow and self
// critics think of the agent's work, grouped by failure mode.
const OBS_LABELS: Record<string, string> = {
  assumption_identified: "Assumption not in evidence",
  generalization_from_thin_evidence: "Generalized from thin evidence",
  source_contradiction: "Contradicts a consulted source",
  tool_output_questionable: "Trusted a questionable tool output",
  gap_papered_over: "Gap papered over",
  border_crossing_unflagged: "Evidence→inference unflagged",
  reasoning_skip: "Skipped a reasoning step",
};

function RoleChip({ role }: { role: string }) {
  const cls =
    role === "shadow"
      ? "bg-purple-50 text-purple-700"
      : "bg-blue-50 text-blue-700";
  return (
    <span className={`text-[10px] font-medium rounded-full px-1.5 py-0.5 ${cls}`}>
      {role}
    </span>
  );
}

function QualitySection({ customerId }: { customerId: string }) {
  const [openType, setOpenType] = useState<string | null>(null);
  const grouped = useQuery({
    queryKey: ["quality-grouped", customerId],
    queryFn: () => api.groupedQualityObservations(customerId),
    enabled: !!customerId,
    refetchInterval: 15_000,
  });
  const groups = grouped.data?.groups ?? [];

  if (!groups.length) {
    return (
      <div className="bg-white border border-gray-200 rounded-lg p-10 text-center">
        <MessageSquareWarning className="h-8 w-8 text-gray-300 mx-auto mb-2" />
        <div className="text-sm font-medium text-gray-700">No quality observations</div>
        <div className="text-xs text-gray-400 mt-1">
          When the shadow or self critic flags an assumption, a skipped
          reasoning step, a source contradiction — it lands here. An empty
          stream over real traffic is the good outcome.
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {groups.map((g: any) => {
        const open = openType === g.observation_type;
        return (
          <div key={g.observation_type} className="bg-white border border-gray-200 rounded-lg">
            <button
              className="w-full flex items-center justify-between px-4 py-3 text-left"
              onClick={() => setOpenType(open ? null : g.observation_type)}
            >
              <div className="flex items-center gap-2">
                {open ? <ChevronDown className="h-4 w-4 text-gray-400" /> : <ChevronRight className="h-4 w-4 text-gray-400" />}
                <span className="text-sm font-medium text-gray-800">
                  {OBS_LABELS[g.observation_type] ?? g.observation_type}
                </span>
              </div>
              <span className="text-xs font-semibold text-gray-500 bg-gray-100 rounded-full px-2 py-0.5">
                {g.count}
              </span>
            </button>
            {open && (
              <div className="border-t border-gray-100 divide-y divide-gray-50">
                {(g.latest ?? []).map((o: any, i: number) => (
                  <div key={`${o.critique_id}-${i}`} className="px-4 py-3">
                    <div className="flex items-center gap-2 mb-1">
                      <RoleChip role={o.critic_role} />
                      <TimeAgo iso={o.created_at} />
                      {o.sequence_id && (
                        <span className="text-[10px] text-gray-400 font-mono truncate">
                          seq {o.sequence_id.slice(0, 12)}…
                        </span>
                      )}
                    </div>
                    <div className="text-sm text-gray-700">
                      {o.detail?.text ?? o.detail?.description ?? JSON.stringify(o.detail)}
                    </div>
                    {o.summary_text && (
                      <div className="text-xs text-gray-400 mt-1 line-clamp-2">
                        critic summary: {o.summary_text}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
