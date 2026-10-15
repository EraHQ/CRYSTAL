// The sidebar footer (4.1 item 4): the two meters from me.usage, the
// plan pill, the email, and a labeled sign-out button. Admin keeps the
// customer select above it.
import { LogOut } from "lucide-react";
import { CustomerSelector } from "@/components/CustomerSelector";
import { IconButton, Meter, StatusPill } from "@/components/ui";
import type { Me } from "@/lib/auth";
import { tierLabel } from "./nav";

export function AccountBlock({ me, email, isTenant, signedIn, onSignOut }: {
  me: Me | null;
  email: string | null;
  isTenant: boolean;
  signedIn: boolean;
  onSignOut: () => void;
}) {
  const usage = me?.usage;
  const facts = usage?.facts_used ?? null;
  const cap = usage?.fact_cap ?? null;
  const ai = usage?.ai_capacity_pct ?? null;
  return (
    <div className="space-y-3 border-t border-line px-4 pb-4 pt-3">
      {!isTenant && (
        <div>
          <p className="mb-1 px-1 text-[11px] font-medium uppercase tracking-[0.12em] text-fg-3">Customer</p>
          <CustomerSelector />
        </div>
      )}
      {usage && (facts !== null || ai !== null) && (
        <div className="space-y-2.5">
          {facts !== null && cap !== null && cap > 0 ? (
            <Meter label="Memory" value={facts} max={cap} valueLabel={`${facts.toLocaleString()} / ${cap.toLocaleString()}`} />
          ) : facts !== null ? (
            <div className="flex items-baseline justify-between text-[12.5px]">
              <span className="text-fg-2">Memory</span>
              <span className="tabular-nums text-fg-2">{facts.toLocaleString()} · uncapped</span>
            </div>
          ) : null}
          {ai !== null && <Meter label="AI today" value={ai} max={100} valueLabel={`${Math.round(ai)}%`} />}
        </div>
      )}
      <div className="flex items-center gap-2">
        {me?.kind === "user" && <StatusPill variant={me.subscription_tier === "free" || !me.subscription_tier ? "neutral" : "info"} raw={me.subscription_tier ?? undefined}>{tierLabel(me.subscription_tier)}</StatusPill>}
        <span className="min-w-0 flex-1 truncate text-[12px] text-fg-3" title={email ?? undefined}>{email ?? ""}</span>
        {signedIn && <IconButton label="Sign out" icon={LogOut} size="sm" onClick={onSignOut} />}
      </div>
    </div>
  );
}
