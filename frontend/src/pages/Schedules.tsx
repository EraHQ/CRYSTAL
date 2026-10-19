// Scheduled runs (launch feature, 2026-10-18; Q78-Q81): its own page and
// menu item. Anthony, 2026-10-18 23:20: "the scheduler doesn't even have
// its own page on the menu. How are people supposed to set one up".
// The section component is shared with nothing else now; the Cognition
// page only links here.
import { CalendarClock } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";
import { EmptyState, PageHeader } from "@/components/ui";
import { Schedules } from "@/components/cognition/Schedules";

export function SchedulesPage() {
  const { selectedCustomerId } = useSelectedCustomer();
  const list = useQuery({
    queryKey: ["schedules", selectedCustomerId],
    queryFn: () => api.listSchedules(selectedCustomerId!),
    enabled: !!selectedCustomerId,
  });
  if (!selectedCustomerId) {
    return (
      <>
        <PageHeader title="Scheduled runs" />
        <EmptyState icon={CalendarClock} title="No customer selected" help="Pick a customer above to see their scheduled runs." />
      </>
    );
  }
  const active = list.data?.active ?? 0;
  const cap = list.data?.max_active;
  return (
    <div className="space-y-6">
      <PageHeader title="Scheduled runs"
        meta={list.isLoading ? "Loading…" : `${active} active${cap ? ` of ${cap} on this plan` : ""}`} />
      <Schedules customerId={selectedCustomerId} />
    </div>
  );
}
