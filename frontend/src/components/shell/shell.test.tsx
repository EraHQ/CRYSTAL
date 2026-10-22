// @vitest-environment jsdom
// Console Refine PR 3 (docs/CONSOLE_UX_AUDIT.md 4.1, 4.2): the grouped
// nav keeps every route, the badges count what the pages count, the
// phone bar carries the combined Review badge, the brand is "Crystal"
// alone, and Getting started is a card of links with a progress line.
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen } from "@testing-library/react";
import { ReactNode } from "react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/api", () => ({
  api: {
    listConflicts: vi.fn(async () => ({ total: 2, items: [] })),
    listAssumptions: vi.fn(async () => ({
      total: 3,
      items: [
        { id: "a", recall_gated: true, quality_tier: "quarantine" },
        { id: "b", recall_gated: true, quality_tier: "quarantine" },
        { id: "c", recall_gated: true, quality_tier: "blacklist" },
        { id: "d", recall_gated: false, quality_tier: "neutral" },
      ],
    })),
    listCustomers: vi.fn(async () => ({ items: [] })),
    onboardingStatus: vi.fn(async () => ({ connected: true, last_seen_at: null, ai_tools: [] })),
  },
  errorMessage: (e: unknown, fallback = "Something went wrong.") => (e instanceof Error ? e.message : fallback),
}));

const ME = {
  kind: "user", role: "owner", customer_id: "cus_1", user_id: "u", email: "a@b.c",
  subscription_tier: "starter",
  usage: { facts_used: 2354, fact_cap: 50_000, fact_state: "ok", ai_capacity_pct: 38 },
};
vi.mock("@/lib/auth", () => ({ useAuth: () => ({ me: ME, refreshMe: async () => {} }) }));

import { SelectedCustomerProvider } from "@/lib/selected-customer";
import { GettingStarted } from "@/components/GettingStarted";
import { BottomBar } from "./BottomBar";
import { Sidebar } from "./Sidebar";
import { BOTTOM_BAR, NAV_GROUPS, tierLabel } from "./nav";
import { isPendingAssumption } from "./useNavBadges";

function wrap(children: ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return (
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={["/conflicts"]}>
        <SelectedCustomerProvider pinnedId="cus_1">{children}</SelectedCustomerProvider>
      </MemoryRouter>
    </QueryClientProvider>
  );
}

afterEach(cleanup);

describe("nav model", () => {
  it("keeps every route the flat list had and only renames three labels", () => {
    const routes = NAV_GROUPS.flatMap((g) => g.items.map((i) => i.to)).sort();
    expect(routes).toEqual(["/", "/activity", "/agents", "/assumptions", "/billing", "/cognition", "/conflicts", "/critiques",
      "/knowledge", "/mind", "/onboard", "/playground", "/schedules", "/settings"]);
    const labels = Object.fromEntries(NAV_GROUPS.flatMap((g) => g.items.map((i) => [i.to, i.label])));
    expect(labels["/activity"]).toBe("Activity");
    expect(labels["/mind"]).toBe("Mind");
    expect(labels["/billing"]).toBe("Plan & usage");
    expect(labels["/onboard"]).toBe("Customers");
    expect(labels["/schedules"]).toBe("Scheduled runs");
    expect(NAV_GROUPS.map((g) => g.label)).toEqual(["Use", "Curate", "Inspect", "Account", "Admin"]);
    expect(NAV_GROUPS.filter((g) => g.adminOnly).map((g) => g.label)).toEqual(["Admin"]);
    expect(BOTTOM_BAR.map((i) => i.label)).toEqual(["Chat", "Knowledge", "Bank", "Review", "Settings"]);
    expect(BOTTOM_BAR.find((i) => i.label === "Review")?.to).toBe("/assumptions");
  });
  it("a pending assumption is gated and not blacklisted; tier labels are words", () => {
    expect(isPendingAssumption({ recall_gated: true, quality_tier: "quarantine" })).toBe(true);
    expect(isPendingAssumption({ recall_gated: true, quality_tier: "blacklist" })).toBe(false);
    expect(isPendingAssumption({ recall_gated: false, quality_tier: "quarantine" })).toBe(false);
    expect([tierLabel("free"), tierLabel("starter"), tierLabel("scale"), tierLabel(null)]).toEqual(["Free", "Starter", "Scale", "Self-hosted"]);
  });
});

describe("Sidebar", () => {
  it("shows the brand as Crystal, the grouped nav, the badges and the meters; hides Admin for a tenant", async () => {
    render(wrap(<Sidebar me={ME as never} email="a@b.c" isTenant signedIn onSignOut={() => {}} />));
    expect(screen.getByText("Crystal")).toBeTruthy();
    expect(screen.queryByText(/Inspector/)).toBeNull();
    expect(screen.queryByText("Admin")).toBeNull();
    expect(screen.getByRole("link", { name: /Activity/ })).toBeTruthy();
    const badges = await screen.findAllByLabelText(/waiting/);
    expect(badges.map((b) => b.textContent)).toEqual(["2", "2"]);     // conflicts 2, pending assumptions 2 (c is blacklisted, d is approved)
    expect(screen.getByRole("progressbar", { name: "Memory" }).getAttribute("aria-valuenow")).toBe("5");
    expect(screen.getByRole("progressbar", { name: "AI today" }).getAttribute("aria-valuenow")).toBe("38");
    expect(screen.getByText("Starter")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Sign out" })).toBeTruthy();
  });
  it("shows the Admin group for a platform admin", () => {
    render(wrap(<Sidebar me={null} email={null} isTenant={false} signedIn={false} onSignOut={() => {}} />));
    expect(screen.getByText("Admin")).toBeTruthy();
    expect(screen.getByRole("link", { name: /Customers/ })).toBeTruthy();
  });
});

describe("BottomBar", () => {
  it("has five items and the combined Review badge", async () => {
    render(wrap(<BottomBar />));
    const links = screen.getAllByRole("link");
    expect(links.map((l) => l.textContent?.replace(/\d+/, "").trim())).toEqual(["Chat", "Knowledge", "Bank", "Review", "Settings"]);
    expect((await screen.findByLabelText("4 waiting")).textContent).toBe("4");
  });
});

describe("GettingStarted", () => {
  it("is a card of links with a progress line and a hide button", async () => {
    render(wrap(<GettingStarted />));
    expect(await screen.findByText("3 of 4 done")).toBeTruthy();
    expect(screen.getByRole("progressbar", { name: "Getting started" }).getAttribute("aria-valuenow")).toBe("3");
    expect(screen.getByRole("link", { name: /Store your first memory/ }).getAttribute("href")).toBe("/");
    expect(screen.getByRole("link", { name: /Connect a tool/ }).textContent).toContain("Manage connections");
    expect(screen.getByRole("button", { name: "Check now" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Hide getting started" })).toBeTruthy();
  });
  it("collapses to one line on Chat with the next step", async () => {
    render(wrap(<GettingStarted collapsed />));
    expect(await screen.findByText("3 of 4 done")).toBeTruthy();
    expect(screen.getByRole("link", { name: /Next:/ })).toBeTruthy();
  });
});
