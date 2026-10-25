// The console's destinations, grouped (docs/CONSOLE_UX_AUDIT.md 4.1).
// Routes are unchanged from the flat list this replaces; only three
// labels moved to match their page titles (Logs -> Query log, Billing ->
// Plan & usage, Onboard -> Customers). `adminOnly` marks the
// cross-tenant / platform surfaces: a hosted tenant never sees them.
import type { LucideIcon } from "lucide-react";
import {
  Activity, BookOpen, Brain, CalendarClock, CreditCard, Database, History, Lightbulb,
  FlaskConical, Gauge, MessageSquare, MessageSquareWarning, Orbit, Scale, Settings as SettingsIcon, Table2, Target, UserPlus,
} from "lucide-react";

export type Badge = "conflicts" | "assumptions" | "review";

export interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
  end?: boolean;
  badge?: Badge;
  /** A second path prefix that counts as this destination (the bank's
   *  drawer lives at /bank/:id while the list is /bank). */
  also?: string;
}

export interface NavGroup {
  label: string;
  adminOnly?: boolean;
  items: NavItem[];
}

export const NAV_GROUPS: NavGroup[] = [
  {
    label: "Use",
    items: [
      // Command center (2026-10-24): goals, connections and the system at a glance.
      { to: "/", label: "Command center", icon: Gauge, end: true },
      { to: "/playground", label: "Chat", icon: MessageSquare },
      { to: "/knowledge", label: "Knowledge", icon: BookOpen },
      { to: "/bank", label: "Crystal Bank", icon: Database },
      // Lattices (2026-10-25, Q115 A): Crystal storage's tables.
      { to: "/lattices", label: "Lattices", icon: Table2, also: "/lattices/" },
      // Goals (Q109 A, 2026-10-24): standing directives Crystal plans and works.
      { to: "/goals", label: "Goals", icon: Target },
    ],
  },
  {
    label: "Curate",
    items: [
      { to: "/cognition", label: "Cognition", icon: Brain },
      // Scheduled runs (2026-10-18): a brief or a research question on a
      // cadence. Its own destination so people can find it.
      { to: "/schedules", label: "Scheduled runs", icon: CalendarClock },
      { to: "/conflicts", label: "Conflicts", icon: Scale, badge: "conflicts" },
      { to: "/assumptions", label: "Assumptions", icon: Lightbulb, badge: "assumptions" },
    ],
  },
  {
    // Q90=A (2026-10-19): one home for discoverability and auditability —
    // questions, runs and background curation, with where each went.
    label: "Inspect",
    items: [
      { to: "/activity", label: "Activity", icon: History, also: "/activity/" },
      // Mind (2026-10-21): the whole memory as one scene.
      { to: "/mind", label: "Mind", icon: Orbit },
    ],
  },
  {
    label: "Account",
    items: [
      { to: "/settings", label: "Settings", icon: SettingsIcon, also: "/settings/" },
      // L2-S4=B: the money page. Hosted tenants see it; self-host/admin
      // too (it renders the no-subscription state harmlessly).
      { to: "/billing", label: "Plan & usage", icon: CreditCard },
    ],
  },
  {
    label: "Admin",
    adminOnly: true,
    items: [
      { to: "/critiques", label: "Critiques", icon: MessageSquareWarning },
      { to: "/agents", label: "Agents", icon: Activity },
      { to: "/onboard", label: "Customers", icon: UserPlus },
      // The Beta list (2026-10-24): who sees invite-only features.
      { to: "/beta", label: "Beta", icon: FlaskConical },
    ],
  },
];

// Under 720px the sidebar is replaced by a five-item bar (4.1 item 6).
// Review goes to Assumptions and carries the combined Conflicts +
// Assumptions badge.
export const BOTTOM_BAR: NavItem[] = [
  { to: "/", label: "Home", icon: Gauge, end: true },
  { to: "/playground", label: "Chat", icon: MessageSquare },
  { to: "/bank", label: "Bank", icon: Database },
  { to: "/assumptions", label: "Review", icon: Lightbulb, badge: "review" },
  { to: "/settings", label: "Settings", icon: SettingsIcon, also: "/settings/" },
];

export function tierLabel(tier: string | null | undefined): string {
  switch (tier) {
    case "starter": return "Starter";
    case "scale": return "Scale";
    case "free": return "Free";
    default: return "Self-hosted";
  }
}
