// @vitest-environment jsdom
// Console Refine PR 10 (docs/CONSOLE_UX_AUDIT.md 5.9 + 5.16 + bug 4):
// Plan & usage reads its meters with a note per state, the promise sits
// next to them, the cards come from lib/plans.ts with Current and
// Recommended badges and one primary button, the return from Stripe
// shows a confirmation with "What changed", self-host says so instead
// of pretending Free is current; the plan wall is specific from the
// console's own numbers, keeps a non-upgrade path and closes on Escape.
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const m = {
  billingCheckout: vi.fn(async () => ({ checkout_url: "https://stripe.test/c", session_id: "cs_1" })),
  billingPortal: vi.fn(async () => ({ portal_url: "https://stripe.test/p" })),
};
vi.mock("@/lib/api", () => ({
  api: new Proxy({}, { get: (_t, k: string) => (...a: unknown[]) => (m as Record<string, (...x: unknown[]) => unknown>)[k](...a) }),
  errorMessage: (e: unknown, fallback = "Something went wrong.") => (e instanceof Error ? e.message : fallback),
}));
const auth: { me: Record<string, unknown> | null; refreshMe: ReturnType<typeof vi.fn> } = {
  me: null,
  refreshMe: vi.fn(async () => {}),
};
vi.mock("@/lib/auth", () => ({ useAuth: () => auth }));

import { Billing, memoryNote } from "./Billing";
import { PlanWallModal, untilMidnightUtc, wallView } from "@/components/PlanWallModal";
import { PLANS, nextPlan, planFor } from "@/lib/plans";

const freeMe = (over: Record<string, unknown> = {}) => ({
  kind: "user", role: "owner", customer_id: "cus_1", email: "a@example.com", subscription_tier: "free", billing_live: true,
  usage: { facts_used: 2354, fact_cap: 2500, fact_state: "warning", ai_capacity_pct: 37 }, ...over,
});

function mount(path = "/billing") {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <PlanWallModal />
      <Routes>
        <Route path="/billing" element={<Billing />} />
        <Route path="/settings/model" element={<p>Model settings</p>} />
        <Route path="/" element={<p>Bank</p>} />
      </Routes>
    </MemoryRouter>,
  );
}

beforeEach(() => { vi.clearAllMocks(); auth.me = freeMe(); });
afterEach(cleanup);

describe("Plan & usage (Refine PR 10)", () => {
  it("shows the meters with a note per state, the promise, the badges and one primary button", async () => {
    mount();
    expect(screen.getByRole("heading", { level: 1, name: "Plan & usage" })).toBeTruthy();
    expect(screen.getByText("Free plan · capacity resets at midnight UTC")).toBeTruthy();
    expect(screen.getByText("2,354 of 2,500 facts · 94%")).toBeTruthy();
    expect(screen.getByText(/Almost full\. At 2,500, new memories pause/)).toBeTruthy();
    expect(screen.getByText("37% used today")).toBeTruthy();
    expect(screen.getByText(/Whatever your plan, everything you stored stays recallable/)).toBeTruthy();
    const free = within(screen.getByRole("article", { name: "Free plan" }));
    expect(free.getByText("Current")).toBeTruthy();
    const starter = within(screen.getByRole("article", { name: "Starter plan" }));
    expect(starter.getByText("Recommended")).toBeTruthy();
    expect(starter.getByText("$29/mo")).toBeTruthy();
    // Exactly one accent (primary) button on the page.
    const primaries = screen.getAllByRole("button").filter((b) => b.className.includes("bg-accent"));
    expect(primaries.map((b) => b.textContent?.trim())).toEqual(["Upgrade to Starter"]);
    expect(screen.getByRole("button", { name: "Choose Scale" })).toBeTruthy();
    expect(screen.queryByText("Annual")).toBeNull();
    expect(document.body.textContent).not.toContain("crystal facts");
    const assign = vi.fn();
    Object.defineProperty(window, "location", { value: { ...window.location, assign, origin: "http://t" }, writable: true });
    fireEvent.click(screen.getByRole("button", { name: "Upgrade to Starter" }));
    await waitFor(() => expect(m.billingCheckout).toHaveBeenCalled());
    const [success, cancel, plan] = m.billingCheckout.mock.calls[0] as unknown as [string, string, string];
    expect(success.endsWith("billing?upgraded=1")).toBe(true);
    expect(cancel.endsWith("billing")).toBe(true);
    expect(plan).toBe("starter");
    await waitFor(() => expect(assign).toHaveBeenCalledWith("https://stripe.test/c"));
    expect(memoryNote("blocked", 2500)).toBe("Full. New memories are paused. Recall, edits and export still work.");
    expect(memoryNote("ok", null)).toBe("No memory limit on this plan.");
  });

  it("confirms the upgrade when Stripe sends the browser back, with what changed", async () => {
    auth.me = freeMe({ subscription_tier: "starter_29", usage: { facts_used: 2354, fact_cap: 50000, fact_state: "ok", ai_capacity_pct: 4 } });
    mount("/billing?upgraded=1");
    const box = within(screen.getByRole("region", { name: "Upgrade confirmation" }));
    expect(box.getByText("You're on Starter")).toBeTruthy();
    const rows = box.getAllByRole("row").map((r) => r.textContent);
    expect(rows[1]).toContain("Memory limit");
    expect(rows[1]).toContain("2,500 facts");
    expect(rows[1]).toContain("50,000 facts");
    expect(rows[2]).toContain("10x Free");
    expect(rows[3]).toContain("Not included");
    expect(rows[3]).toContain("Included");
    expect(screen.getByText("Starter plan · capacity resets at midnight UTC")).toBeTruthy();
    expect(within(screen.getByRole("article", { name: "Starter plan" })).getByRole("button", { name: "Manage billing" })).toBeTruthy();
    expect(within(screen.getByRole("article", { name: "Scale plan" })).getByRole("button", { name: "Switch to Scale" })).toBeTruthy();
    fireEvent.click(box.getByRole("button", { name: "Back to plans" }));
    await waitFor(() => expect(screen.queryByRole("region", { name: "Upgrade confirmation" })).toBeNull());
  });

  it("polls /v1/me while the webhook lands, and says self-hosted when there is no plan", async () => {
    mount("/billing?upgraded=1");
    expect(screen.getByText(/Confirming your payment with Stripe/)).toBeTruthy();
    await waitFor(() => expect(auth.refreshMe).toHaveBeenCalled());
    cleanup();
    auth.me = { kind: "platform_admin_key", role: "platform_admin", customer_id: null };
    mount();
    expect(screen.getByText("Self-hosted: no plan limits")).toBeTruthy();
    expect(screen.queryByText("Current")).toBeNull();
    expect(planFor("trial_29").id).toBe("starter");
    expect(nextPlan("scale_49_seat")).toBeNull();
    expect(PLANS.map((p) => p.capacityX)).toEqual([1, 10, 50]);
  });

  it("plan wall: specific body from the console's numbers, a Meter, a non-upgrade path, Escape closes", async () => {
    mount();
    window.dispatchEvent(new CustomEvent("crystal:plan-wall", { detail: { code: "memory_full", message: "Memory is full.", status: 402 } }));
    const dialog = within(await screen.findByRole("dialog"));
    expect(dialog.getByText("Your memory is full")).toBeTruthy();
    expect(dialog.getByText("Free has room for 146 more (2,354 of 2,500 used).")).toBeTruthy();
    expect(dialog.getByRole("progressbar", { name: "Memory" })).toBeTruthy();
    expect(dialog.getByText("Starter holds 50,000 facts, 20x Free")).toBeTruthy();
    expect(dialog.getByRole("button", { name: "Upgrade to Starter · $29/mo" })).toBeTruthy();
    expect(dialog.getByText("Esc closes")).toBeTruthy();
    fireEvent.keyDown(document.activeElement ?? document.body, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    // Daily capacity: what is left and when it resets; the other path is the user's own key.
    window.dispatchEvent(new CustomEvent("crystal:plan-wall", { detail: { code: "daily_capacity", message: "x", status: 429 } }));
    const d2 = within(await screen.findByRole("dialog"));
    expect(d2.getByText(/You have 63% of today's capacity left\. It resets at midnight UTC, in \d+ (h \d+ )?min\./)).toBeTruthy();
    fireEvent.click(d2.getByRole("button", { name: "Use my own key" }));
    expect(await screen.findByText("Model settings")).toBeTruthy();
    expect(screen.queryByRole("dialog")).toBeNull();
    // The pure view: monthly names the renewal, never dollars; the model wall offers Sonnet.
    const me = freeMe() as unknown as Parameters<typeof wallView>[1];
    const monthly = wallView({ code: "monthly_budget", message: "", status: 429 }, me, new Date(Date.UTC(2026, 9, 16, 12)));
    expect(monthly.body).toBe("It renews on November 1. Saving and recall keep working until then.");
    expect(monthly.body).not.toContain("$");
    expect(wallView({ code: "model_not_in_plan", message: "m", status: 400 }, me).other).toEqual({ label: "Use Sonnet", to: "/settings/model" });
    expect(untilMidnightUtc(new Date(Date.UTC(2026, 9, 16, 17, 48)))).toBe("6 h 12 min");
  });
});
