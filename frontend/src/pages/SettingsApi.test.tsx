// @vitest-environment jsdom
// Console Refine PR 9 (docs/CONSOLE_UX_AUDIT.md 5.8): six deep-linkable
// tabs under one header; snippets use the product host and point a lost
// key at the API key tab; the model is radio cards with a plan lock;
// a tenant sees no dollars on any tab; Regenerate names the tools that
// break and reveals the key once; Remove on the team is a suspension
// with Undo; Erase and Delete are typed confirms with the facts count.
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { ReactNode } from "react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const state = {
  operators: [
    { id: "op_1", display_name: "Anthony Holley", role: "admin", status: "active", has_identity: true },
    { id: "op_2", display_name: "Dana", role: "operator", status: "active", has_identity: false },
  ],
  model: "claude-sonnet-5",
};
const m = {
  onboardingStatus: vi.fn(async () => ({ connected: true, last_seen_at: new Date(Date.now() - 3600_000).toISOString(), ai_tools: ["cursor", "claude_desktop"] })),
  customerSpend: vi.fn(async () => ({ customer_id: "cus_1", inference_mode: "managed", subscription_tier: "free", totals: {}, managed_month_to_date_micro_usd: 120_000, managed_monthly_cap_micro_usd: 500_000, has_upstream_key: false })),
  getCustomer: vi.fn(async () => ({ id: "cus_1", model_id: state.model, assumptions_explore: null })),
  listBudgets: vi.fn(async () => ({ budgets: [{ function: "auto_research", cap_micro_usd: 0, period: "monthly" }], shadow_critic: { effective_max_per_day: 5, runs_last_24h: 1, cost_micro_usd_last_24h: 1200 } })),
  spendOrigins: vi.fn(async () => ({ origins: [{ origin: "agent", call_count: 3, cost_micro_usd: 999 }] })),
  setInferenceMode: vi.fn(async () => ({})),
  updateUpstreamKey: vi.fn(async () => ({ updated: true })),
  setModel: vi.fn(async (_c: string, id: string) => { state.model = id; return {}; }),
  upsertBudget: vi.fn(async () => ({})),
  setAssumptionsExplore: vi.fn(async () => ({})),
  rotateApiKey: vi.fn(async () => ({ customer_id: "cus_1", api_key: "crys_live_4f2a9b8c7d6e5f4a3b2c1d0e" })),
  listOperators: vi.fn(async () => ({ operators: state.operators })),
  seedOperator: vi.fn(async (_c: string, b: { name: string }) => ({ operator_id: "op_3", display_name: b.name, facts_written: 2 })),
  updateOperator: vi.fn(async (_c: string, id: string, b: { status?: string }) => {
    if (b.status) state.operators = state.operators.map((o) => (o.id === id ? { ...o, status: b.status! } : o));
    return { operator_id: id, display_name: "Dana", facts_written: 0, status: b.status ?? "active" };
  }),
  eraseMyMemories: vi.fn(async () => ({ erased: true, rows: { facts: 1240 } })),
  deleteMyAccount: vi.fn(async () => ({ scheduled: true, purge_after: "2026-10-23T00:00:00Z", subscriptions_cancelled: 1 })),
  restoreMyAccount: vi.fn(async () => ({ restored: true })),
};
vi.mock("@/lib/api", () => ({
  api: new Proxy({}, { get: (_t, k: string) => (...a: unknown[]) => (m as Record<string, (...x: unknown[]) => unknown>)[k](...a) }),
  authedFetch: vi.fn(),
  errorMessage: (e: unknown, fallback = "Something went wrong.") => (e instanceof Error ? e.message : fallback),
}));
const auth = {
  status: "signedIn", reauthProvider: () => "password", reauthenticate: vi.fn(async () => {}), refreshMe: vi.fn(async () => {}),
  me: { kind: "user", role: "owner", customer_id: "cus_1", email: "anthony@example.com", subscription_tier: "free",
        deletion_scheduled_at: undefined as string | undefined, purge_after: undefined as string | undefined,
        allowed_models: ["claude-haiku-5-5", "claude-sonnet-5-5"], usage: { facts_used: 1240, fact_cap: 5000, fact_state: "ok", ai_capacity_pct: 37 } },
};
vi.mock("@/lib/auth", () => ({ useAuth: () => auth }));

import { SelectedCustomerProvider } from "@/lib/selected-customer";
import { ToastProvider } from "@/components/ui";
import { SettingsApi, planLabel } from "./SettingsApi";
import { regenerateImpact } from "./settings/ApiKey";
import { initials } from "./settings/Team";

function mount(path = "/settings") {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const tree = (children: ReactNode) => (
    <QueryClientProvider client={qc}>
      <ToastProvider>
        <MemoryRouter initialEntries={[path]}>
          <SelectedCustomerProvider pinnedId="cus_1">{children}</SelectedCustomerProvider>
        </MemoryRouter>
      </ToastProvider>
    </QueryClientProvider>
  );
  return render(tree(
    <Routes>
      <Route path="/settings" element={<SettingsApi />} />
      <Route path="/settings/:tab" element={<SettingsApi />} />
      <Route path="/billing" element={<p>Billing page</p>} />
    </Routes>
  ));
}

beforeEach(() => {
  vi.clearAllMocks();
  state.operators = [
    { id: "op_1", display_name: "Anthony Holley", role: "admin", status: "active", has_identity: true },
    { id: "op_2", display_name: "Dana", role: "operator", status: "active", has_identity: false },
  ];
  state.model = "claude-sonnet-5";
  auth.me.deletion_scheduled_at = undefined;
});
afterEach(cleanup);

describe("Settings (Refine PR 9)", () => {
  it("opens on Connections with one header, six tab links, the product host and a lost-key path", async () => {
    mount();
    expect(await screen.findByRole("heading", { level: 1, name: "Settings" })).toBeTruthy();
    expect(screen.getByText("Free plan · signed in as anthony@example.com")).toBeTruthy();
    const nav = within(screen.getByRole("navigation", { name: "Settings sections" }));
    expect(nav.getAllByRole("link").map((l) => l.getAttribute("href"))).toEqual([
      "/settings/connections", "/settings/model", "/settings/curation", "/settings/api-key", "/settings/team", "/settings/your-data",
    ]);
    expect(nav.getByRole("link", { name: "Connections" }).getAttribute("aria-current")).toBe("page");
    // Claude Desktop first: OAuth, so the URL alone.
    expect(screen.getByText("https://api.erahq.ai/mcp")).toBeTruthy();
    expect(screen.getByText(/No key needed/)).toBeTruthy();
    fireEvent.click(screen.getByRole("tab", { name: "Cursor" }));
    expect(screen.getByText(/"url": "https:\/\/api\.erahq\.ai\/mcp"/)).toBeTruthy();
    expect(screen.getByRole("link", { name: "Create a new one in API key" }).getAttribute("href")).toBe("/settings/api-key");
    expect(await screen.findByText("Connected")).toBeTruthy();
    expect(screen.getByText(/At signup you said you use Cursor, Claude Desktop/)).toBeTruthy();
    expect(planLabel("starter")).toBe("Starter");
  });

  it("Model & inference: consequences in words, capacity as a percent, Opus locked with Upgrade, no dollars for a tenant", async () => {
    mount("/settings/model");
    expect(await screen.findByText("Crystal pays for the model. Work counts toward your daily AI capacity.")).toBeTruthy();
    expect(screen.getByRole("progressbar", { name: "AI capacity used today" })).toBeTruthy();
    expect(screen.getByText("37%")).toBeTruthy();
    const models = within(await screen.findByRole("radiogroup", { name: "Model" }));
    await waitFor(() => expect(models.getByRole("radio", { name: /Sonnet/ }).getAttribute("aria-checked")).toBe("true"));
    const opus = models.getByRole("radio", { name: /Opus/ });
    expect(opus.getAttribute("aria-disabled")).toBe("true");
    expect(within(opus).getByRole("link", { name: "Upgrade to use" }).getAttribute("href")).toBe("/billing");
    fireEvent.click(models.getByRole("radio", { name: /Haiku/ }));
    await waitFor(() => expect(m.setModel).toHaveBeenCalledWith("cus_1", "claude-haiku-5-5"));
    // Fable is on the platform's list too, locked on Free like Opus.
    expect(within(models.getByRole("radio", { name: /Fable/ })).getByRole("link", { name: "Upgrade to use" })).toBeTruthy();
    // My own key is unavailable until a provider key is stored.
    expect(screen.getByRole("radio", { name: /My own provider key/ }).getAttribute("aria-disabled")).toBe("true");
    fireEvent.click(screen.getByRole("switch", { name: "Autonomous research" }));
    await waitFor(() => expect(m.upsertBudget).toHaveBeenCalledWith("cus_1", "auto_research", 1_000_000_000_000));
    expect(document.body.textContent).not.toContain("$");
  });

  it("API key: Regenerate names the tools that break, needs the word and password, reveals the key once", async () => {
    mount("/settings/api-key");
    fireEvent.click(await screen.findByRole("button", { name: "Regenerate key" }));
    await screen.findByText(/Cursor will lose memory until you paste the new key/);
    expect(screen.getByText(/Claude Desktop keeps working/)).toBeTruthy();
    const confirmBtn = screen.getByRole("button", { name: "Regenerate key" });
    expect((confirmBtn as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(screen.getByLabelText(/Type/), { target: { value: "REGENERATE" } });
    fireEvent.change(screen.getByLabelText("Your password"), { target: { value: "hunter2" } });
    fireEvent.click(screen.getByRole("button", { name: "Regenerate key" }));
    await waitFor(() => expect(m.rotateApiKey).toHaveBeenCalledWith("cus_1"));
    expect(auth.reauthenticate).toHaveBeenCalledWith("hunter2");
    const modal = within(await screen.findByRole("dialog"));
    expect(modal.getByText("crys_live_4f2a9b8c7d6e5f4a3b2c1d0e")).toBeTruthy();
    fireEvent.click(modal.getByRole("button", { name: "I saved it" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(screen.getByText("crys_live_4f…")).toBeTruthy();
    expect(regenerateImpact([]).breaks).toEqual(["Claude Code", "Cursor", "VS Code (Copilot)", "ChatGPT"]);
    expect(regenerateImpact(["zed"]).keeps).toEqual([]);
  });

  it("Team: lists operators, Remove suspends with Undo, Add opens a drawer that seeds", async () => {
    mount("/settings/team");
    const dana = (await screen.findByText("Dana")).closest("li")!;
    expect(within(dana).getByText("No identity yet")).toBeTruthy();
    fireEvent.click(within(dana).getByRole("button", { name: "Remove" }));
    await waitFor(() => expect(m.updateOperator).toHaveBeenCalledWith("cus_1", "op_2", { status: "suspended" }));
    await screen.findByText("Removed");
    fireEvent.click(await screen.findByRole("button", { name: "Undo" }));
    await waitFor(() => expect(m.updateOperator).toHaveBeenCalledWith("cus_1", "op_2", { status: "active" }));
    fireEvent.click(screen.getByRole("button", { name: "Add operator" }));
    const drawer = within(await screen.findByRole("dialog"));
    fireEvent.change(drawer.getByLabelText("Name"), { target: { value: "Sam" } });
    fireEvent.change(drawer.getByLabelText("Identity facts, one per line"), { target: { value: "Runs ops\n\nLikes tea" } });
    expect(drawer.getByText("Crystal recalls these when Sam asks something.")).toBeTruthy();
    fireEvent.click(drawer.getByRole("button", { name: "Add operator" }));
    await waitFor(() => expect(m.seedOperator).toHaveBeenCalledWith("cus_1", { name: "Sam", role: "admin", facts: ["Runs ops", "Likes tea"] }));
    expect(initials("Anthony Holley")).toBe("AH");
  });

  it("Your data: Erase names the facts count and needs ERASE plus the password; Delete needs DELETE", async () => {
    mount("/settings/your-data");
    expect(await screen.findByText("Removes all 1,240 facts. Connections, settings and your account stay.")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Erase all memories" }));
    fireEvent.change(screen.getByLabelText(/Type/), { target: { value: "ERASE" } });
    fireEvent.change(screen.getByLabelText("Your password"), { target: { value: "hunter2" } });
    fireEvent.click(screen.getByRole("button", { name: "Erase everything" }));
    await waitFor(() => expect(m.eraseMyMemories).toHaveBeenCalled());
    expect(await screen.findAllByText("All memories erased (1240 rows). Your account and plan are unchanged.")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Delete my account" }));
    fireEvent.change(screen.getByLabelText(/Type/), { target: { value: "DELETE" } });
    fireEvent.change(screen.getByLabelText("Your password"), { target: { value: "hunter2" } });
    fireEvent.click(screen.getByRole("button", { name: "Delete my account" }));
    await waitFor(() => expect(m.deleteMyAccount).toHaveBeenCalled());
  });

  it("hides Your data from a platform admin and shows the restore banner to a locked owner", async () => {
    auth.me.deletion_scheduled_at = "2026-10-16T00:00:00Z";
    mount("/settings/your-data");
    expect(await screen.findByText("This account is scheduled for deletion.")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Restore my account" }));
    await waitFor(() => expect(m.restoreMyAccount).toHaveBeenCalled());
    cleanup();
    const was = auth.me.kind;
    (auth.me as { kind: string }).kind = "platform_admin_key";
    mount("/settings/your-data");
    await screen.findByRole("heading", { level: 1, name: "Settings" });
    expect(screen.queryByRole("link", { name: "Your data" })).toBeNull();
    expect(screen.getByRole("navigation", { name: "Settings sections" }).querySelector('[aria-current="page"]')?.textContent).toBe("Connections");
    (auth.me as { kind: string }).kind = was;
  });
});
