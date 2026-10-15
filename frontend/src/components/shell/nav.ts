// The console's destinations, grouped (docs/CONSOLE_UX_AUDIT.md 4.1).
// Routes are unchanged from the flat list this replaces; only three
// labels moved to match their page titles (Logs -> Query log, Billing ->
// Plan & usage, Onboard -> Customers). `adminOnly` marks the
// cross-tenant / platform surfaces: a hosted tenant never sees them.
import type { LucideIcon } from "lucide-react";
import {
  Activity, BookOpen, Brain, CreditCard, Database, Lightbulb, ListOrdered,
  MessageSquare, MessageSquareWarning, Scale, Settings as SettingsIcon, UserPlus,
} from "lucide-react";

export type Badge = "conflicts" | "assumptions" | "review";

export interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
  end?: boolean;
  badge?: Badge;
  /** A second path prefix that counts as this destination (the bank's
   *  drawer lives at /bank/:id while the list is "/"). */
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
      { to: "/playground", label: "Chat", icon: MessageSquare },
      { to: "/knowledge", label: "Knowledge", icon: BookOpen },
      { to: "/", label: "Crystal Bank", icon: Database, end: true, also: "/bank/" },
    ],
  },
  {
    label: "Curate",
    items: [
      { to: "/cognition", label: "Cognition", icon: Brain },
      { to: "/conflicts", label: "Conflicts", icon: Scale, badge: "conflicts" },
      { to: "/assumptions", label: "Assumptions", icon: Lightbulb, badge: "assumptions" },
    ],
  },
  {
    label: "Inspect",
    items: [{ to: "/queries", label: "Query log", icon: ListOrdered }],
  },
  {
    label: "Account",
    items: [
      { to: "/settings", label: "Settings", icon: SettingsIcon },
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
    ],
  },
];

// Under 720px the sidebar is replaced by a five-item bar (4.1 item 6).
// Review goes to Assumptions and carries the combined Conflicts +
// Assumptions badge.
export const BOTTOM_BAR: NavItem[] = [
  { to: "/playground", label: "Chat", icon: MessageSquare },
  { to: "/knowledge", label: "Knowledge", icon: BookOpen },
  { to: "/", label: "Bank", icon: Database, end: true, also: "/bank/" },
  { to: "/assumptions", label: "Review", icon: Lightbulb, badge: "review" },
  { to: "/settings", label: "Settings", icon: SettingsIcon },
];

export function tierLabel(tier: string | null | undefined): string {
  switch (tier) {
    case "starter": return "Starter";
    case "scale": return "Scale";
    case "free": return "Free";
    default: return "Self-hosted";
  }
}
