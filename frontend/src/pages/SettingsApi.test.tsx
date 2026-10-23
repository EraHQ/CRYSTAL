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
  background: null as string | null,
  catalog: { enabled: false, models: [] as { id: string; name: string; context_length: number; band: string; fast: boolean; nvidia: boolean; caps: { json_schema: boolean; tools: boolean } | null }[] },
  rules: [] as Array<{ id: string; rule_type: string; name: string; selector: Record<string, unknown>; conditions: Record<string, unknown>; action: Record<string, unknown>; enabled: boolean; priority: number; fire_count: number; last_fired_at: string | null; created_at: string | null }>,
  steward: false,
};
const m = {
  onboardingStatus: vi.fn(async () => ({ connected: true, last_seen_at: new Date(Date.now() - 3600_000).toISOString(), ai_tools: ["cursor", "claude_desktop"] })),
  customerSpend: vi.fn(async () => ({ customer_id: "cus_1", inference_mode: "managed", subscription_tier: "free", totals: {}, managed_month_to_date_micro_usd: 120_000, managed_monthly_cap_micro_usd: 500_000, has_upstream_key: false })),
  getCustomer: vi.fn(async () => ({ id: "cus_1", model_id: state.model, background_model_id: state.background, assumptions_explore: null })),
  setBackgroundModel: vi.fn(async (_c: string, id: string) => { state.background = id || null; return {}; }),
  modelCatalog: vi.fn(async () => state.catalog),
  listBudgets: vi.fn(async () => ({ budgets: [{ function: "auto_research", cap_micro_usd: 0, period: "monthly" }], shadow_critic: { effective_max_per_day: 5, runs_last_24h: 1, cost_micro_usd_last_24h: 1200 } })),
  spendOrigins: vi.fn(async () => ({ origins: [{ origin: "agent", call_count: 3, cost_micro_usd: 999 }] })),
  setInferenceMode: vi.fn(async () => ({})),
  updateUpstreamKey: vi.fn(async () => ({ updated: true })),
  setModel: vi.fn(async (_c: string, id: string) => { state.model = id; return {}; }),
  upsertBudget: vi.fn(async () => ({})),
  setAssumptionsExplore: vi.fn(async () => ({})),
  stewardStatus: vi.fn(async () => ({ customer_id: "cus_1", enabled: state.steward, platform_enabled: true, today: { sessions: 0, actions: 0, sessions_per_day: 12 }, paused: state.steward ? { code: "daily_capacity", message: "used", at: "2026-10-23T12:00:00Z", resets_at: "2026-10-24T00:00:00Z" } : null, latest: null, actions: [] })),
  setSteward: vi.fn(async (_c: string, enabled: boolean) => { state.steward = enabled; return { customer_id: "cus_1", enabled }; }),
  listRules: vi.fn(async () => ({ rules: state.rules, count: state.rules.length })),
  createRule: vi.fn(async (_c: string, r: { name: string; rule_type: string; selector: Record<string, unknown>; conditions: Record<string, unknown>; action: Record<string, unknown> }) => {
    const rule = { id: `rule_${state.rules.length + 1}`, enabled: true, priority: 100, fire_count: 0, last_fired_at: null, created_at: null, ...r };
    state.rules = [...state.rules, rule]; return { rule };
  }),
  updateRule: vi.fn(async (_c: string, id: string, p: { enabled?: boolean }) => { state.rules = state.rules.map((r) => (r.id === id ? { ...r, ...p } : r)); return { rule: state.rules.find((r) => r.id === id) }; }),
  deleteRule: vi.fn(async (_c: string, id: string) => { state.rules = state.rules.filter((r) => r.id !== id); return { deleted: true, rule_id: id }; }),
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
import { ruleText } from "./settings/Curation";
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
  state.background = null;
  state.catalog = { enabled: false, models: [] };
  state.rules = [];
  state.steward = false;
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
    const models = within(await screen.findByRole("radiogroup", { name: "Answers model" }));
    // The stored legacy id selects its 5.5 card.
    await waitFor(() => expect(models.getByRole("radio", { name: /Sonnet/ }).getAttribute("aria-checked")).toBe("true"));
    const opus = models.getByRole("radio", { name: /Opus/ });
    expect(opus.getAttribute("aria-disabled")).toBe("true");
    expect(within(opus).getByRole("link", { name: "Upgrade to use" }).getAttribute("href")).toBe("/billing");
    fireEvent.click(models.getByRole("radio", { name: /Haiku/ }));
    await waitFor(() => expect(m.setModel).toHaveBeenCalledWith("cus_1", "claude-haiku-5-5"));
    // Fable is on the platform's list too, locked on Free like Opus.
    expect(within(models.getByRole("radio", { name: /Fable/ })).getByRole("link", { name: "Upgrade to use" })).toBeTruthy();
    // Q87: the second row picks the background-curation model; nothing picked = Crystal's tier map.
    const where = within(screen.getByLabelText("What runs where"));
    expect(where.getByText(/Crystal's tier map/)).toBeTruthy();
    const bg = within(screen.getByRole("radiogroup", { name: "Background curation model" }));
    expect(bg.getAllByRole("radio").every((r) => r.getAttribute("aria-checked") === "false")).toBe(true);
    fireEvent.click(bg.getByRole("radio", { name: /Sonnet/ }));
    await waitFor(() => expect(m.setBackgroundModel).toHaveBeenCalledWith("cus_1", "claude-sonnet-5-5"));
    await waitFor(() => expect(bg.getByRole("radio", { name: /Sonnet/ }).getAttribute("aria-checked")).toBe("true"));
    expect(where.getByText("Sonnet, quick steps on Haiku")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Use Crystal's default" }));
    await waitFor(() => expect(m.setBackgroundModel).toHaveBeenCalledWith("cus_1", ""));
    // M3: no Nemotron ids offered by the platform, so no Nemotron cards.
    expect(models.queryByRole("radio", { name: /Nemotron/ })).toBeNull();
    // My own key is unavailable until a provider key is stored.
    expect(screen.getByRole("radio", { name: /My own provider key/ }).getAttribute("aria-disabled")).toBe("true");
    fireEvent.click(screen.getByRole("switch", { name: "Autonomous research" }));
    await waitFor(() => expect(m.upsertBudget).toHaveBeenCalledWith("cus_1", "auto_research", 1_000_000_000_000));
    expect(document.body.textContent).not.toContain("$");
  });

  it("Model & inference: Nemotron cards appear only when the platform offers the ids, and the summary names the light sibling", async () => {
    auth.me.allowed_models = ["claude-haiku-5-5", "claude-sonnet-5-5", "nvidia/Nemotron-3_5-Lightning", "nvidia/nemotron-3-super-120b-a12b"];
    try {
      mount("/settings/model");
      const models = within(await screen.findByRole("radiogroup", { name: "Answers model" }));
      const superCard = await models.findByRole("radio", { name: /Nemotron Super/ });
      expect(superCard.getAttribute("aria-disabled")).not.toBe("true");
      expect(within(superCard).queryByRole("link", { name: "Upgrade to use" })).toBeNull();
      expect(within(superCard).getByText(/Nebius Token Factory/)).toBeTruthy();
      expect(models.getByRole("radio", { name: /Nemotron Lightning/ })).toBeTruthy();
      fireEvent.click(superCard);
      await waitFor(() => expect(m.setModel).toHaveBeenCalledWith("cus_1", "nvidia/nemotron-3-super-120b-a12b"));
      const bg = within(screen.getByRole("radiogroup", { name: "Background curation model" }));
      fireEvent.click(bg.getByRole("radio", { name: /Nemotron Super/ }));
      await waitFor(() => expect(m.setBackgroundModel).toHaveBeenCalledWith("cus_1", "nvidia/nemotron-3-super-120b-a12b"));
      await waitFor(() => expect(within(screen.getByLabelText("What runs where")).getByText("Nemotron Super, quick steps on Nemotron Lightning")).toBeTruthy());
    } finally {
      auth.me.allowed_models = ["claude-haiku-5-5", "claude-sonnet-5-5"];
    }
  });

  it("Model & inference: any Token Factory model is a pick from the catalog, with what the probe found", async () => {
    // Hackathon item 1 (2026-10-22): the catalog picker lists text models
    // not already drawn as cards, with context and a cost band (no
    // dollars); a pick calls the same route; the probe's record shows
    // as JSON / Tools chips next to the running model.
    auth.me.allowed_models = ["claude-haiku-5-5", "claude-sonnet-5-5", "nvidia/nemotron-3-super-120b-a12b"];
    state.catalog = { enabled: true, models: [
      { id: "nvidia/nemotron-3-super-120b-a12b", name: "nemotron-3-super-120b-a12b", context_length: 262144, band: "standard", fast: false, nvidia: true, caps: null },
      { id: "nvidia/Nemotron-3-Ultra-550b-a55b", name: "Nemotron-3-Ultra-550b-a55b", context_length: 1048576, band: "heavy", fast: false, nvidia: true, caps: null },
      { id: "deepseek-ai/DeepSeek-V4-Pro", name: "DeepSeek-V4-Pro", context_length: 1000000, band: "heavy", fast: false, nvidia: false, caps: { json_schema: false, tools: true } },
    ] };
    state.background = "deepseek-ai/DeepSeek-V4-Pro";
    try {
      mount("/settings/model");
      const pick = (await screen.findByLabelText("Answers: any Token Factory model")) as HTMLSelectElement;
      const options = Array.from(pick.options).map((o) => o.textContent);
      // Super is a card already, so the picker starts after it; context and band read in words, never dollars.
      expect(options).toEqual(["Choose a model", "Nemotron Ultra · 1M context · heavy cost", "DeepSeek-V4-Pro · 1M context · heavy cost"]);
      expect(screen.queryByText(/\$/)).toBeNull();
      fireEvent.change(pick, { target: { value: "nvidia/Nemotron-3-Ultra-550b-a55b" } });
      fireEvent.click(within(pick.closest("form")!).getByRole("button", { name: "Use" }));
      await waitFor(() => expect(m.setModel).toHaveBeenCalledWith("cus_1", "nvidia/Nemotron-3-Ultra-550b-a55b"));
      // The background pick came from the catalog: named, light sibling noted, chips from its probe.
      const where = within(screen.getByLabelText("What runs where"));
      expect(where.getByText(/DeepSeek-V4-Pro, quick steps on/)).toBeTruthy();
      const chips = within(where.getByRole("group", { name: "What DeepSeek-V4-Pro can do" }));
      expect(chips.getByText("JSON no")).toBeTruthy();
      expect(chips.getByText("Tools yes")).toBeTruthy();
    } finally {
      auth.me.allowed_models = ["claude-haiku-5-5", "claude-sonnet-5-5"];
    }
  });

  it("Model & inference: a Token Factory key is saved with its provider, and on it the tenant picks from the catalog", async () => {
    // Item 2 (2026-10-22): the key form carries the provider; a Nebius key
    // saves as "nebius" and the toast says how many models it can see. On
    // that key (byok + nebius) the tenant gets the same cards and catalog
    // picker as a managed tenant, with no plan locks.
    m.updateUpstreamKey.mockImplementationOnce(async () => ({ updated: true, provider: "nebius", models_seen: 28, moved: {} }));
    mount("/settings/model");
    const providerPick = (await screen.findByLabelText("Key provider")) as HTMLSelectElement;
    fireEvent.change(providerPick, { target: { value: "nebius" } });
    fireEvent.change(screen.getByPlaceholderText("Token Factory API key"), { target: { value: "tf-key-123" } });
    fireEvent.click(screen.getByRole("button", { name: "Save key" }));
    await waitFor(() => expect(m.updateUpstreamKey).toHaveBeenCalledWith("cus_1", "tf-key-123", "nebius", undefined));
    expect(await screen.findByText("Token Factory key saved (28 models available).")).toBeTruthy();

    // Now the tenant runs on that key: cards come from the catalog's NVIDIA ids, unlocked, picker below.
    cleanup();
    m.customerSpend.mockImplementation(async () => ({ customer_id: "cus_1", inference_mode: "byok", subscription_tier: "free", totals: {}, managed_month_to_date_micro_usd: 0, managed_monthly_cap_micro_usd: 0, has_upstream_key: true }));
    m.getCustomer.mockImplementation(async () => ({ id: "cus_1", provider: "nebius", model_id: "nvidia/nemotron-3-super-120b-a12b", background_model_id: null, assumptions_explore: null }));
    state.catalog = { enabled: true, models: [
      { id: "nvidia/Nemotron-3_5-Lightning", name: "Nemotron-3_5-Lightning", context_length: 1048576, band: "light", fast: false, nvidia: true, caps: null },
      { id: "nvidia/nemotron-3-super-120b-a12b", name: "nemotron-3-super-120b-a12b", context_length: 262144, band: "standard", fast: false, nvidia: true, caps: { json_schema: true, tools: true } },
      { id: "zai-org/GLM-5.3", name: "GLM-5.3", context_length: 1048576, band: "standard", fast: false, nvidia: false, caps: null },
    ] };
    try {
      mount("/settings/model");
      const models = within(await screen.findByRole("radiogroup", { name: "Answers model" }));
      const superCard = await models.findByRole("radio", { name: /Nemotron Super/ });
      expect(superCard.getAttribute("aria-checked")).toBe("true");
      expect(models.queryByRole("radio", { name: /Sonnet/ })).toBeNull();
      expect(within(superCard).queryByRole("link", { name: "Upgrade to use" })).toBeNull();
      const pick = screen.getByLabelText("Background: any Token Factory model") as HTMLSelectElement;
      expect(Array.from(pick.options).map((o) => o.textContent)).toEqual(["Choose a model", "GLM-5.3 · 1M context · standard cost"]);
      expect(screen.getByText(/A Nebius Token Factory key is stored/)).toBeTruthy();
      const where = within(screen.getByLabelText("What runs where"));
      expect(within(where.getByRole("group", { name: "What Nemotron Super can do" })).getByText("Tools yes")).toBeTruthy();
    } finally {
      m.customerSpend.mockImplementation(async () => ({ customer_id: "cus_1", inference_mode: "managed", subscription_tier: "free", totals: {}, managed_month_to_date_micro_usd: 120_000, managed_monthly_cap_micro_usd: 500_000, has_upstream_key: false }));
      m.getCustomer.mockImplementation(async () => ({ id: "cus_1", model_id: state.model, background_model_id: state.background, assumptions_explore: null }));
    }
  });

  // Q100 A (2026-10-23): "Let Crystal run itself" lives under Curation;
  // once on, a paused day says so right under the switch.
  it("Curation: the Steward switch flips and shows the refill ask when paused", async () => {
    mount("/settings/curation");
    const section = await screen.findByRole("region", { name: "Let Crystal run itself" });
    const sw = await within(section).findByRole("switch", { name: "Let Crystal run itself" });
    expect(sw.getAttribute("aria-checked")).toBe("false");
    expect(within(section).getByText("Off. You approve, settle and research by hand.")).toBeTruthy();
    fireEvent.click(sw);
    await waitFor(() => expect(m.setSteward).toHaveBeenCalledWith("cus_1", true));
    expect(await within(section).findByText(/Today's AI capacity is used/)).toBeTruthy();
    expect(within(section).getByRole("link", { name: "Activity › Steward" }).getAttribute("href")).toBe("/activity?view=steward");
  });

  // Q99 C (2026-10-23): the first approval rule lives under Curation as
  // Automation: no rules means a person approves; a rule is one sentence.
  it("Curation: Automation lists rules in words, adds the approval rule, switches it off, deletes it", async () => {
    state.rules = [{ id: "rule_a", rule_type: "approval", name: "Approve after 2 agree", selector: { kind: "consequence" }, conditions: { min_consistent_edges: 2, no_contradicted_edges: true }, action: { approve: true }, enabled: true, priority: 100, fire_count: 3, last_fired_at: new Date(Date.now() - 3600_000).toISOString(), created_at: null }];
    mount("/settings/curation");
    const section = await screen.findByRole("region", { name: "Automation" });
    expect(await within(section).findByText("Approve a consequence once 2 stated facts agree with it and nothing contradicts it.")).toBeTruthy();
    expect(within(section).getByText(/fired 3 times, last/)).toBeTruthy();
    fireEvent.click(within(section).getByRole("switch", { name: "On" }));
    await waitFor(() => expect(m.updateRule).toHaveBeenCalledWith("cus_1", "rule_a", { enabled: false }));
    fireEvent.click(within(section).getByRole("button", { name: "Add a rule" }));
    const form = within(section).getByRole("form", { name: "New rule" });
    fireEvent.change(within(form).getByLabelText("How many facts must agree"), { target: { value: "3" } });
    fireEvent.click(within(form).getByRole("switch", { name: "Only if nothing contradicts it" }));
    fireEvent.click(within(form).getByRole("button", { name: "Save rule" }));
    await waitFor(() => expect(m.createRule).toHaveBeenCalledWith("cus_1", {
      rule_type: "approval", name: "Approve after 3 agree", selector: { kind: "consequence" },
      conditions: { min_consistent_edges: 3, no_contradicted_edges: false }, action: { approve: true },
    }));
    expect(await within(section).findByText("Approve a consequence once 3 stated facts agree with it.")).toBeTruthy();
    fireEvent.click(within(section).getAllByRole("button", { name: "Delete" })[0]);
    fireEvent.click(within(within(section).getAllByRole("group", { name: "Confirm" })[0]).getByRole("button", { name: "Delete" }));
    await waitFor(() => expect(m.deleteRule).toHaveBeenCalledWith("cus_1", "rule_a"));
    expect(ruleText({ id: "x", rule_type: "approval", name: "n", selector: { kind: "any" }, conditions: { min_consistent_edges: 1 }, action: { approve: true }, enabled: true, priority: 100, fire_count: 0, last_fired_at: null, created_at: null }))
      .toBe("Approve an assumption or consequence once 1 stated fact agrees with it and nothing contradicts it.");
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
