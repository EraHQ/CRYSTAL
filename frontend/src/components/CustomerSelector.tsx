import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";
import { useEffect } from "react";
import { ChevronDown } from "lucide-react";

export function CustomerSelector() {
  const { selectedCustomerId, setSelectedCustomerId } = useSelectedCustomer();
  const { data, isLoading, isError } = useQuery({
    queryKey: ["customers"],
    queryFn: api.listCustomers,
  });

  useEffect(() => {
    if (!selectedCustomerId && data?.items && data.items.length > 0 && !isLoading) {
      setSelectedCustomerId(data.items[0].id);
    }
  }, [data, selectedCustomerId, setSelectedCustomerId, isLoading]);

  if (isLoading) return <div className="px-1 text-xs text-gray-400">Loading…</div>;
  if (isError) return <div className="px-1 text-xs text-red-400">API unreachable</div>;

  const items = data?.items ?? [];
  if (items.length === 0) {
    return (
      <Link to="/onboard" className="block px-1 text-xs font-medium text-brand-400 hover:text-brand-300">
        No customers — onboard one →
      </Link>
    );
  }

  return (
    <div className="relative">
      <select
        value={selectedCustomerId ?? ""}
        onChange={(e) => setSelectedCustomerId(e.target.value || null)}
        className="w-full cursor-pointer appearance-none truncate rounded-lg border border-gray-200 bg-gray-50 py-2 pl-3 pr-8 text-xs font-medium text-gray-800 transition-colors hover:border-gray-300 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-500/25"
      >
        {items.map((c) => (
          <option key={c.id} value={c.id}>
            {c.id} · {c.crystal_count} crystals
          </option>
        ))}
      </select>
      <ChevronDown className="pointer-events-none absolute right-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-gray-400" />
    </div>
  );
}
