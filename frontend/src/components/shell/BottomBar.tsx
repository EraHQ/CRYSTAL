// Phone navigation (4.1 item 6): five items under 720px, Review carries
// the combined Conflicts + Assumptions badge, and the bar sits above the
// home indicator (env(safe-area-inset-bottom)).
import { NavLink } from "react-router-dom";
import { cn } from "@/lib/utils";
import { BOTTOM_BAR } from "./nav";
import { useNavBadges } from "./useNavBadges";

export function BottomBar() {
  const badges = useNavBadges();
  return (
    <nav aria-label="Main" className="fixed inset-x-0 bottom-0 z-40 flex border-t border-line bg-surface min-[720px]:hidden"
      style={{ paddingBottom: "env(safe-area-inset-bottom)" }}>
      {BOTTOM_BAR.map((item) => {
        const count = item.badge ? badges[item.badge] : 0;
        return (
          <NavLink key={item.to} to={item.to} end={item.end}
            className={({ isActive }) => cn(
              "relative flex min-w-0 flex-1 flex-col items-center gap-0.5 px-1 py-2 text-[11px] font-medium",
              isActive ? "text-accent" : "text-fg-3",
            )}>
            <span className="relative">
              <item.icon aria-hidden className="h-5 w-5" />
              {count > 0 && (
                <span className="absolute -right-2 -top-1 rounded-full bg-accent px-1 text-[10px] font-semibold leading-4 text-on-accent tabular-nums"
                  aria-label={`${count} waiting`}>
                  {count > 99 ? "99+" : count}
                </span>
              )}
            </span>
            <span className="truncate">{item.label}</span>
          </NavLink>
        );
      })}
    </nav>
  );
}
