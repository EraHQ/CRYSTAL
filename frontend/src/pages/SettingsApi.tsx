// Settings — Console Refine PR 9 (docs/CONSOLE_UX_AUDIT.md 5.8). One
// PageHeader, six tabs each with its own URL (/settings/<tab>), a left
// sub-nav from 720px and a scrolling tab row below it. The tabs live in
// pages/settings/*; Your data is the owner's alone (a platform admin
// cannot erase or delete on a tenant's behalf). "API & Inference" as a
// second title, dollars for tenants and the eleven-section scroll are
// gone.
import { Navigate, NavLink, useParams } from "react-router-dom";
import { Settings as SettingsIcon } from "lucide-react";
import { useAuth } from "@/lib/auth";
import { useSelectedCustomer } from "@/lib/selected-customer";
import { EmptyState, PageHeader } from "@/components/ui";
import { YourData } from "@/components/YourData";
import { cn } from "@/lib/utils";
import { SETTINGS_TABS, type SettingsTab } from "./settings/shared";
import { Connections } from "./settings/Connections";
import { ModelInference } from "./settings/ModelInference";
import { Curation } from "./settings/Curation";
import { ApiKey } from "./settings/ApiKey";
import { Team } from "./settings/Team";

export function planLabel(tier: string | null | undefined): string {
  if (!tier) return "";
  return tier.charAt(0).toUpperCase() + tier.slice(1);
}

export function SettingsApi() {
  const { tab } = useParams();
  const { selectedCustomerId } = useSelectedCustomer();
  const { me } = useAuth();
  const isUser = me?.kind === "user";
  const tabs = SETTINGS_TABS.filter((t) => t.id !== "your-data" || isUser);
  const active = tabs.find((t) => t.id === tab);
  if (!tab || !active) return <Navigate to="/settings/connections" replace />;

  if (!selectedCustomerId) {
    return (
      <>
        <PageHeader title="Settings" />
        <EmptyState icon={SettingsIcon} title="No customer selected" help="Pick a customer above to manage its settings." />
      </>
    );
  }

  const plan = planLabel(me?.subscription_tier);
  const meta = isUser
    ? [plan ? `${plan} plan` : null, me?.email ? `signed in as ${me.email}` : null].filter(Boolean).join(" · ")
    : `Customer ${selectedCustomerId}`;

  return (
    <div>
      <PageHeader title="Settings" meta={meta} />
      <div className="grid gap-4 min-[720px]:grid-cols-[180px_minmax(0,1fr)] min-[720px]:gap-6">
        <nav aria-label="Settings sections" className="-mx-4 overflow-x-auto px-4 min-[720px]:mx-0 min-[720px]:px-0">
          <ul className="flex gap-1 border-b border-line min-[720px]:flex-col min-[720px]:border-b-0">
            {tabs.map((t) => (
              <li key={t.id} className="shrink-0">
                <NavLink to={`/settings/${t.id}`} aria-current={t.id === active.id ? "page" : undefined}
                  className={cn("block whitespace-nowrap px-2.5 py-2 text-[13px] min-[720px]:rounded-md",
                    t.id === active.id
                      ? "-mb-px border-b-2 border-accent font-medium text-accent min-[720px]:mb-0 min-[720px]:border-b-0 min-[720px]:bg-accent-soft min-[720px]:text-fg"
                      : "text-fg-2 hover:text-fg min-[720px]:hover:bg-inset")}>
                  {t.label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
        <div className="min-w-0">
          <TabBody tab={active.id} customerId={selectedCustomerId} />
        </div>
      </div>
    </div>
  );
}

function TabBody({ tab, customerId }: { tab: SettingsTab; customerId: string }) {
  switch (tab) {
    case "connections": return <Connections />;
    case "model": return <ModelInference />;
    case "curation": return <Curation />;
    case "api-key": return <ApiKey />;
    case "team": return <Team />;
    case "your-data": return <YourData customerId={customerId} />;
  }
}
