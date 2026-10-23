// Nav badges (4.1 item 3): open conflicts and pending assumptions, read
// through the SAME query keys and fetchers the two pages use, so the
// sidebar never fetches what a page already holds and both stay in
// step. Hidden at 0.
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";
import type { Badge } from "./nav";

export function isPendingAssumption(item: { recall_gated?: boolean; quality_tier?: string | null; kind?: string }): boolean {
  // Q99 C: consequences have their own section and never count here.
  return !!item.recall_gated && item.quality_tier !== "blacklist" && item.kind !== "consequence";
}

export function useNavBadges(): Record<Badge, number> {
  const { selectedCustomerId } = useSelectedCustomer();
  const conflicts = useQuery({
    queryKey: ["conflicts", selectedCustomerId],
    queryFn: () => api.listConflicts(selectedCustomerId!, "open"),
    enabled: !!selectedCustomerId,
    refetchInterval: 60_000,
  });
  const assumptions = useQuery({
    queryKey: ["assumptions", selectedCustomerId],
    queryFn: () => api.listAssumptions(selectedCustomerId!),
    enabled: !!selectedCustomerId,
    refetchInterval: 60_000,
  });
  const openConflicts = conflicts.data?.total ?? 0;
  const pending = (assumptions.data?.items ?? []).filter(isPendingAssumption).length;
  return { conflicts: openConflicts, assumptions: pending, review: openConflicts + pending };
}
