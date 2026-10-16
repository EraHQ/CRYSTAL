// Knowledge › General knowledge (Console Refine PR 5, 5.3 item 9): the
// shared banks this workspace can subscribe to. A subscribed bank is
// searched together with the workspace's own memories on every recall
// (fact_vector_store: general results merge in with a tie-break so the
// tenant's own knowledge wins ties).
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Globe } from "lucide-react";
import { api, errorMessage, jsonFetch } from "@/lib/api";
import { EmptyState, ErrorBanner, Skeleton } from "@/components/ui";
import { cn } from "@/lib/utils";

interface CrystalType { id: string; display_name: string; scope: string }

export function GeneralBanks({ customerId }: { customerId: string }) {
  const qc = useQueryClient();
  const [error, setError] = useState<string | null>(null);
  const types = useQuery({
    queryKey: ["crystal_types"],
    queryFn: () => jsonFetch<{ items: CrystalType[] }>("/admin/api/crystal_types"),
  });
  const subs = useQuery({
    queryKey: ["subscriptions", customerId],
    queryFn: () => api.listSubscriptions(customerId) as Promise<{ general_crystal_types: string[] }>,
  });
  const toggle = useMutation({
    mutationFn: ({ id, on }: { id: string; on: boolean }) =>
      on ? api.unsubscribeCrystalType(customerId, id) : api.subscribeCrystalType(customerId, id),
    onSuccess: () => { setError(null); qc.invalidateQueries({ queryKey: ["subscriptions", customerId] }); },
    onError: (e) => setError(errorMessage(e, "Couldn't change the subscription.")),
  });

  const general = (types.data?.items ?? []).filter((t) => t.scope === "general");
  const active = new Set(subs.data?.general_crystal_types ?? []);
  const loading = types.isLoading || subs.isLoading;

  return (
    <div>
      <p className="mb-3 text-[12.5px] text-fg-2">
        A subscribed bank is searched together with your own memories on every recall. Your own knowledge wins ties.
      </p>
      {error && <ErrorBanner title="General knowledge" message={error} className="mb-3" />}
      {loading && <Skeleton rows={2} />}
      {(types.isError || subs.isError) && (
        <ErrorBanner title="Couldn't load general knowledge" error={types.error ?? subs.error} onRetry={() => { types.refetch(); subs.refetch(); }} />
      )}
      {!loading && !types.isError && general.length === 0 && (
        <EmptyState icon={Globe} title="No general banks yet" help="Shared banks appear here when the platform publishes one." />
      )}
      {general.length > 0 && (
        <ul className="divide-y divide-line rounded-lg border border-line bg-surface">
          {general.map((t) => {
            const on = active.has(t.id);
            return (
              <li key={t.id} className="flex items-center gap-3 px-4 py-3">
                <div className="min-w-0 flex-1">
                  <p className="text-[13.5px] font-medium text-fg">{t.display_name}</p>
                  <p className="text-[12px] text-fg-3" title={t.id}>
                    {on ? "Included in recall for this workspace." : "Subscribe to include its facts in recall, labeled as general knowledge."}
                  </p>
                </div>
                <button type="button" role="switch" aria-checked={on} aria-label={`${on ? "Unsubscribe from" : "Subscribe to"} ${t.display_name}`}
                  disabled={toggle.isPending} onClick={() => toggle.mutate({ id: t.id, on })}
                  className={cn("relative h-5 w-9 shrink-0 rounded-full transition-colors disabled:opacity-50", on ? "bg-accent" : "bg-line-strong")}>
                  <span aria-hidden className={cn("absolute top-0.5 h-4 w-4 rounded-full bg-surface shadow-card transition-transform", on ? "translate-x-4" : "translate-x-0.5")} />
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
