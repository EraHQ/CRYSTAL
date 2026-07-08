// System Critiques (S6, 2026-07-08) — the surface for CRYS's structured
// complaints about ITSELF. The substrate channel reaches every part of
// the system that affects outcomes: tool capability wishes, ingestion
// artifacts, retrieval quality, metacognition misses. Observations are
// recorded and surfaced, never auto-acted (MCR Principle 9).
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { MessageSquareWarning, ChevronDown, ChevronRight } from "lucide-react";
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
  const [openGroup, setOpenGroup] = useState<string | null>(null);

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
      <div>
        <h1 className="text-xl font-semibold text-gray-900">System Critiques</h1>
        <p className="text-sm text-gray-500 mt-1">
          CRYS's structured complaints about its own system — tools it wishes
          were more capable, ingestion artifacts, retrieval friction,
          metacognition misses. Observations are recorded and surfaced, never
          auto-acted.
        </p>
      </div>

      {!groups.length ? (
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
                      <div key={o.action_item.id} className="p-4 pl-11">
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
