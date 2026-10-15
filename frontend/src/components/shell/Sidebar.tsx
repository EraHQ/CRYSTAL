// Desktop navigation (4.1): grouped destinations with section labels,
// badges for things waiting, the brand, and the account block.
import { NavLink } from "react-router-dom";
import { cn } from "@/lib/utils";
import type { Me } from "@/lib/auth";
import { AccountBlock } from "./AccountBlock";
import { CrystalMark } from "./CrystalMark";
import { NAV_GROUPS } from "./nav";
import { useNavBadges } from "./useNavBadges";

export function Sidebar({ me, email, isTenant, signedIn, onSignOut }: {
  me: Me | null;
  email: string | null;
  isTenant: boolean;
  signedIn: boolean;
  onSignOut: () => void;
}) {
  const badges = useNavBadges();
  const groups = NAV_GROUPS.filter((g) => !(isTenant && g.adminOnly));
  return (
    <aside className="hidden w-[230px] shrink-0 flex-col border-r border-line bg-surface min-[720px]:flex">
      <div className="flex items-center gap-2.5 px-5 pb-4 pt-5">
        <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-accent-soft shadow-glow">
          <CrystalMark className="h-5 w-5" />
        </div>
        <span className="text-[15px] font-semibold tracking-tight text-fg">Crystal</span>
      </div>
      <nav aria-label="Main" className="min-h-0 flex-1 space-y-4 overflow-y-auto px-3 pb-3">
        {groups.map((g) => (
          <div key={g.label}>
            <p className="mb-1 px-3 text-[11px] font-medium uppercase tracking-[0.12em] text-fg-3">{g.label}</p>
            <div className="space-y-0.5">
              {g.items.map((item) => {
                const count = item.badge ? badges[item.badge] : 0;
                return (
                  <NavLink key={item.to} to={item.to} end={item.end}
                    className={({ isActive }) => cn(
                      "group relative flex items-center gap-2.5 rounded-lg px-3 py-2 text-[13px] font-medium transition-colors",
                      isActive ? "bg-accent-soft text-accent" : "text-fg-2 hover:bg-inset hover:text-fg",
                    )}>
                    {({ isActive }) => (
                      <>
                        {isActive && <span aria-hidden className="absolute left-0 top-1/2 h-4 w-0.5 -translate-y-1/2 rounded-full bg-prism-accent" />}
                        <item.icon aria-hidden className={cn("h-4 w-4 transition-colors", isActive ? "text-accent" : "text-fg-3 group-hover:text-fg-2")} />
                        <span className="flex-1">{item.label}</span>
                        {count > 0 && (
                          <span className="rounded-full bg-accent px-1.5 py-0.5 text-[11px] font-semibold leading-none text-on-accent tabular-nums"
                            aria-label={`${count} waiting`}>
                            {count > 99 ? "99+" : count}
                          </span>
                        )}
                      </>
                    )}
                  </NavLink>
                );
              })}
            </div>
          </div>
        ))}
      </nav>
      <AccountBlock me={me} email={email} isTenant={isTenant} signedIn={signedIn} onSignOut={onSignOut} />
    </aside>
  );
}
