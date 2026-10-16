// @vitest-environment jsdom
// Console Refine PR 13 (docs/CONSOLE_UX_AUDIT.md 5.10–5.12): the three
// platform-admin pages. Critiques groups by subsystem with exact counts,
// filters by severity, dismisses with Undo (restore) and "Dismiss shown"
// over what is shown; Agents speaks in words, names operators, ticks the
// heartbeat, shows a waiting session's ask without claiming to approve it,
// and retries a failed task; New customer hides the fields with no server
// side, validates a base URL inline, flips a keyless workspace to the
// managed key, and gates "Use this customer" on the copy checkbox.
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { ReactNode } from "react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { AgentTask } from "@/lib/types";

const obs = (id: string, over: Record<string, unknown> = {}) => ({
  action_item: { id, created_at: "2026-10-16T11:00:00Z", content: { subsystem: "ingestion", severity: "medium", complaint: `complaint ${id}` }, ...(over.action_item as object ?? {}) },
  critique: { critic_role: "structural", critic_model: "scan" },
  trace_summary: { sequence_id: "seq_abcdef123456789", turn_index: 2 },
  ...over,
});
let substrate = [
  obs("ai_1", { action_item: { id: "ai_1", created_at: "2026-10-16T11:00:00Z", content: { subsystem: "ingestion", severity: "high", complaint: "blob-shaped fact", crystal_id: "crys_blob" } } }),
  obs("ai_2"),
  obs("ai_3", { action_item: { id: "ai_3", created_at: "2026-10-15T11:00:00Z", content: { subsystem: "retrieval", severity: "low", complaint: "weak recall" } } }),
];
const sessions = [
  { session_id: "sess_1", team_id: "cus_1", operator_id: "op_1", host: "mbp", pid: 42, project_dir: "/repo", model: "claude-sonnet-4-5", status: "awaiting_approval", effective_status: "awaiting_approval", is_stale: false,
    current_action: "git push", awaiting_payload: { action: "run a command", command: "git push origin main", reason: "ship it" }, parent_session_id: null, started_at: "2026-10-16T10:00:00Z", last_heartbeat_at: new Date(Date.now() - 4000).toISOString(), cost_usd_cumulative: null },
  { session_id: "sess_2", team_id: "cus_1", operator_id: null, host: "ci", pid: null, project_dir: null, model: null, status: "running", effective_status: "crashed", is_stale: true,
    current_action: null, awaiting_payload: null, parent_session_id: null, started_at: null, last_heartbeat_at: new Date(Date.now() - 10 * 60_000).toISOString(), cost_usd_cumulative: null },
];
let tasks: AgentTask[] = [
  { id: "atask_f", customer_id: "cus_1", project_dir: "/repo", task: "fix the build", branch: "main", status: "failed", source: "cli", run_at: null, recur_seconds: null, parent_task_id: null, series_failures: 0, report: null, error: "pytest exited 1", log_path: null, created_at: "2026-10-16T09:00:00Z", started_at: null, finished_at: "2026-10-16T09:05:00Z" },
  { id: "atask_q", customer_id: "cus_1", project_dir: "/repo", task: "write docs", branch: null, status: "queued", source: "agent", run_at: null, recur_seconds: null, parent_task_id: null, series_failures: 0, report: null, error: null, log_path: null, created_at: "2026-10-16T09:10:00Z", started_at: null, finished_at: null },
];
const { FakeApiError } = vi.hoisted(() => {
  class FakeApiError extends Error { constructor(public status: number, public body: unknown, message: string) { super(message); } }
  return { FakeApiError };
});
const m = {
  listSubstrateObservations: vi.fn(async () => ({ total: substrate.length, observations: substrate })),
  groupedQualityObservations: vi.fn(async () => ({ total_groups: 1, groups: [{ observation_type: "reasoning_skip", count: 2, latest: [{ critique_id: "cr1", critic_role: "shadow", created_at: "2026-10-16T11:00:00Z", detail: { text: "skipped the check" } }] }] })),
  listQualityObservations: vi.fn(async () => ({ total: 2, observations: [] })),
  dismissSubstrateObservation: vi.fn(async (id: string) => { substrate = substrate.filter((o) => o.action_item.id !== id); return { id, status: "dropped" }; }),
  restoreSubstrateObservation: vi.fn(async (id: string) => { substrate = [...substrate, obs(id)]; return { id, status: "deferred" }; }),
  listSessions: vi.fn(async () => ({ sessions, count: sessions.length })),
  listAgentTasks: vi.fn(async () => ({ tasks, count: tasks.length })),
  listAgentGaps: vi.fn(async () => ({ gaps: [{ id: "gap_1", customer_id: "cus_1", subject: "find the deploy doc", missing: "no doc", status: "needs_operator", created_at: "2026-10-16T08:00:00Z", resolved_at: null, filled_by_crystal_id: null }], count: 1 })),
  listOperators: vi.fn(async () => ({ operators: [{ id: "op_1", display_name: "Dana", role: "operator", status: "active", has_identity: true }] })),
  getAgentEvents: vi.fn(async () => ({ events: [], count: 0 })),
  getSessionDependencies: vi.fn(async () => ({ dependencies: [], count: 0 })),
  getSessionCommands: vi.fn(async () => ({ commands: [], count: 0 })),
  retryAgentTask: vi.fn(async (_c: string, id: string) => { const t = { ...tasks[0], id: "atask_r", status: "queued", parent_task_id: id, source: "retry", error: null }; tasks = [t, ...tasks]; return { task: t }; }),
  listCustomers: vi.fn(async () => ({ items: [{ id: "cus_1", provider: "anthropic", model_id: "m", created_at: "2026-10-01T00:00:00Z", crystal_count: 3 }] })),
  createCustomer: vi.fn(async (body: { base_url?: string }) => {
    if (body.base_url?.includes("bad")) throw new FakeApiError(400, {}, "self_hosted provider requires base_url");
    return { id: "cus_new", api_key: "cc_live_secret", provider: "anthropic", model_id: "claude-sonnet-4-5-20250929" };
  }),
  setInferenceMode: vi.fn(async () => ({ customer_id: "cus_new", inference_mode: "managed" })),
};
vi.mock("@/lib/api", () => ({
  api: new Proxy({}, { get: (_t, k: string) => (...a: unknown[]) => (m as Record<string, (...x: unknown[]) => unknown>)[k](...a) }),
  ApiError: FakeApiError,
  errorMessage: (e: unknown, fallback = "Something went wrong.") => (e instanceof Error ? e.message : fallback),
}));
const auth = { me: { kind: "platform_admin_key", role: "platform_admin", customer_id: null } };
vi.mock("@/lib/auth", () => ({ useAuth: () => auth }));

import { SelectedCustomerProvider } from "@/lib/selected-customer";
import { ToastProvider } from "@/components/ui";
import { Critiques, groupBySubsystem, subsystemLabel } from "./Critiques";
import { Agents, approvalSummary, heartbeatLabel, isStale } from "./Agents";
import { Onboard, advancedSummary, baseUrlHint, fieldFor } from "./Onboard";

function mount(ui: ReactNode, path: string) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}><ToastProvider><MemoryRouter initialEntries={[path]}>
      <SelectedCustomerProvider pinnedId="cus_1">{ui}</SelectedCustomerProvider>
    </MemoryRouter></ToastProvider></QueryClientProvider>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  substrate = [
    obs("ai_1", { action_item: { id: "ai_1", created_at: "2026-10-16T11:00:00Z", content: { subsystem: "ingestion", severity: "high", complaint: "blob-shaped fact", crystal_id: "crys_blob" } } }),
    obs("ai_2"),
    obs("ai_3", { action_item: { id: "ai_3", created_at: "2026-10-15T11:00:00Z", content: { subsystem: "retrieval", severity: "low", complaint: "weak recall" } } }),
  ];
  tasks = [tasks.find((t) => t.id === "atask_f")!, tasks.find((t) => t.id === "atask_q")!];
});
afterEach(cleanup);

describe("Critiques (Refine PR 13, 5.10)", () => {
  it("groups by subsystem with exact counts, filters by severity, and dismisses with Undo", async () => {
    mount(<Critiques />, "/critiques");
    expect(await screen.findByText("3 open about tools and pipeline · 2 answer-quality flags this week")).toBeTruthy();
    expect(screen.getByRole("radiogroup", { name: "Stream" })).toBeTruthy();
    const ingestion = within(screen.getByRole("region", { name: "Ingestion" }));
    expect(ingestion.getByText(/2 open · 1 high · 1 medium · latest/)).toBeTruthy();
    expect(screen.getByRole("region", { name: "Retrieval" })).toBeTruthy();
    const sev = within(screen.getByRole("group", { name: "Severity" }));
    expect(sev.getByRole("button", { name: /High/ }).textContent).toContain("1");
    fireEvent.click(sev.getByRole("button", { name: /Low/ }));
    expect(screen.queryByRole("region", { name: "Ingestion" })).toBeNull();
    fireEvent.click(sev.getByRole("button", { name: /All/ }));
    fireEvent.click(within(screen.getByRole("region", { name: "Ingestion" })).getByRole("button", { name: /Ingestion/ }));
    const row = screen.getByText("blob-shaped fact").closest("li")!;
    expect(within(row).getByText("High").closest("[title]")?.getAttribute("title")).toBe("high");
    expect(within(row).getByText(/structural \(scan\)/)).toBeTruthy();
    expect(within(row).getByText(/During: turn 3 of conversation/)).toBeTruthy();
    expect(within(row).getByRole("link", { name: /Crystal crys_blob/ }).getAttribute("href")).toBe("/bank/crys_blob");
    fireEvent.click(within(row).getByRole("button", { name: "Dismiss" }));
    await waitFor(() => expect(m.dismissSubstrateObservation).toHaveBeenCalledWith("ai_1"));
    await waitFor(() => expect(screen.queryByText("blob-shaped fact")).toBeNull());
    expect(await screen.findByText("Observation dismissed")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Undo" }));
    await waitFor(() => expect(m.restoreSubstrateObservation).toHaveBeenCalledWith("ai_1"));
    await waitFor(() => expect(screen.getByText("3 open about tools and pipeline · 2 answer-quality flags this week")).toBeTruthy());
  });

  it("Dismiss shown confirms inline and dismisses only what is shown", async () => {
    mount(<Critiques />, "/critiques");
    await screen.findByText(/3 open about tools/);
    fireEvent.click(within(screen.getByRole("group", { name: "Severity" })).getByRole("button", { name: /Medium/ }));
    fireEvent.click(screen.getByRole("button", { name: "Dismiss shown" }));
    const confirm = within(screen.getByRole("group", { name: "Confirm" }));
    expect(confirm.getByText("Dismiss the 1 observation shown? They stay in the record and can be restored.")).toBeTruthy();
    fireEvent.click(confirm.getByRole("button", { name: "Dismiss" }));
    await waitFor(() => expect(m.dismissSubstrateObservation).toHaveBeenCalledTimes(1));
    expect(m.dismissSubstrateObservation).toHaveBeenCalledWith("ai_2");
    expect(document.body.textContent).not.toContain("Clear all");
    expect(groupBySubsystem([]).length).toBe(0);
    expect(subsystemLabel("tool_capability")).toBe("Tool capability");
  });

  it("the answer-quality tab is read-only and uses words", async () => {
    mount(<Critiques />, "/critiques");
    await screen.findByText(/3 open about tools/);
    fireEvent.click(screen.getByRole("radio", { name: "Answer quality" }));
    fireEvent.click(await screen.findByRole("button", { name: /Skipped a reasoning step/ }));
    expect(screen.getByText("skipped the check")).toBeTruthy();
    expect(screen.getByText("Shadow critic").closest("[title]")?.getAttribute("title")).toBe("shadow");
    expect(screen.queryByRole("button", { name: "Dismiss" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Dismiss shown" })).toBeNull();
  });
});

describe("Agents (Refine PR 13, 5.11)", () => {
  it("counts in the meta, speaks in words, names operators, ticks the heartbeat and shows the ask honestly", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      mount(<Agents />, "/agents");
      expect(await screen.findByText("2 sessions · 1 waiting for approval · 1 crashed")).toBeTruthy();
      expect(screen.getByText(/Live · updates every 2s/)).toBeTruthy();
      const banner = within(await screen.findByRole("status", { name: "Approval needed" }));
      expect(banner.getByText("Dana's session wants to run a command")).toBeTruthy();
      expect(banner.getByText("git push origin main")).toBeTruthy();
      expect(banner.getByText("Reason given: ship it")).toBeTruthy();
      expect(banner.getByText(/signed with an operator key/)).toBeTruthy();
      expect(banner.queryByRole("button", { name: /Approve/ })).toBeNull();
      const table = within(screen.getByRole("table"));
      const row1 = table.getByText("Needs approval").closest("tr")!;
      expect(within(row1).getByText("Needs approval").closest("[title]")?.getAttribute("title")).toBe("awaiting_approval");
      expect(within(row1).getByText("Dana")).toBeTruthy();
      expect(within(row1).getByText(/^\ds ago$/)).toBeTruthy();
      const row2 = table.getByText("Crashed").closest("tr")!;
      expect(within(row2).getByText("Team key")).toBeTruthy();
      expect(within(row2).getByText("10 min ago").className).toContain("text-bad");
      expect(document.body.textContent).not.toContain("op_1");
      await act(async () => { await vi.advanceTimersByTimeAsync(2100); });
      expect(within(row1).getByText(/^\ds ago$/)).toBeTruthy();
    } finally {
      vi.useRealTimers();
    }
    expect(heartbeatLabel(new Date(Date.now() - 4000).toISOString(), Date.now())).toBe("4s ago");
    expect(isStale(new Date(Date.now() - 3 * 60_000).toISOString(), Date.now())).toBe(true);
    expect(approvalSummary({ cmd: "rm -rf build" })).toEqual({ action: "run a command", command: "rm -rf build", reason: null });
  });

  it("splits Queue from Failed runs and retries a failed task", async () => {
    mount(<Agents />, "/agents");
    await screen.findByText(/2 sessions/);
    fireEvent.click(screen.getByRole("radio", { name: "Queue 1" }));
    expect(await screen.findByText("write docs")).toBeTruthy();
    expect(screen.queryByText("fix the build")).toBeNull();
    fireEvent.click(screen.getByRole("radio", { name: "Failed runs 2" }));
    const failed = (await screen.findByText("fix the build")).closest("li")!;
    expect(within(failed).getByText("pytest exited 1")).toBeTruthy();
    expect(screen.getByText("Needs an operator").closest("[title]")?.getAttribute("title")).toBe("needs_operator");
    fireEvent.click(within(failed).getByRole("button", { name: "Retry" }));
    await waitFor(() => expect(m.retryAgentTask).toHaveBeenCalledWith("cus_1", "atask_f"));
    expect(await screen.findByText(/Queued again/)).toBeTruthy();
    fireEvent.click(screen.getByRole("radio", { name: /Queue/ }));
    await waitFor(() => expect(screen.getByRole("radio", { name: "Queue 2" })).toBeTruthy());
  });
});

describe("New customer (Refine PR 13, 5.12)", () => {
  it("hides the fields with no server side, validates the URL inline, flips a keyless workspace to managed and gates Use on the checkbox", async () => {
    mount(<Onboard />, "/onboard");
    expect(await screen.findByText("Creates a workspace with its own memory. 1 customer so far.")).toBeTruthy();
    expect(screen.queryByLabelText(/Owner email/)).toBeNull();
    expect(screen.queryByLabelText(/Workspace name/)).toBeNull();
    expect(screen.queryByText(/welcome email/i)).toBeNull();
    const adv = screen.getByRole("button", { name: /Advanced/ });
    expect(adv.textContent).toContain("managed Anthropic · text injection · shadow 5%");
    fireEvent.click(adv);
    fireEvent.change(screen.getByLabelText("Provider"), { target: { value: "self_hosted" } });
    fireEvent.change(screen.getByLabelText("Base URL"), { target: { value: "htps://models.example.com/v1" } });
    expect(screen.getByRole("alert").textContent).toBe('That URL starts with "htps". Did you mean https://?');
    fireEvent.click(screen.getByRole("button", { name: "Create customer" }));
    expect(m.createCustomer).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText("Base URL"), { target: { value: "https://models.example.com/v1" } });
    expect(screen.queryByRole("alert")).toBeNull();
    fireEvent.change(screen.getByLabelText("Provider"), { target: { value: "anthropic" } });
    fireEvent.click(screen.getByRole("button", { name: "Create customer" }));
    expect(await screen.findByText("Workspace cus_new is ready")).toBeTruthy();
    expect(m.createCustomer.mock.calls[0][0]).toMatchObject({ provider: "anthropic", api_key_ref: "", model_id: "claude-sonnet-4-5-20250929", shadow_sample_rate: 0.05 });
    await waitFor(() => expect(m.setInferenceMode).toHaveBeenCalledWith("cus_new", "managed"));
    expect(screen.getByText("API key · shown once")).toBeTruthy();
    expect(screen.getByText("cc_live_secret")).toBeTruthy();
    const use = screen.getByRole("button", { name: "Use this customer" }) as HTMLButtonElement;
    expect(use.disabled).toBe(true);
    fireEvent.click(screen.getByLabelText("I copied the key somewhere safe"));
    expect(use.disabled).toBe(false);
    expect(baseUrlHint("models.example.com")).toBe("Enter a full URL, starting with https://");
    expect(baseUrlHint("https://x.test")).toBeNull();
    expect(advancedSummary({ provider: "openai", key: "sk-1", injection: "none", shadow: "0.2" })).toBe("own key OpenAI · no injection · shadow 20%");
    expect(fieldFor("self_hosted provider requires base_url")).toBe("base_url");
  });

  it("renders a server error beside its field, never as JSON", async () => {
    mount(<Onboard />, "/onboard");
    await screen.findByText(/1 customer so far/);
    fireEvent.click(screen.getByRole("button", { name: /Advanced/ }));
    fireEvent.change(screen.getByLabelText("Provider"), { target: { value: "self_hosted" } });
    fireEvent.change(screen.getByLabelText("Base URL"), { target: { value: "https://bad.example.com" } });
    fireEvent.click(screen.getByRole("button", { name: "Create customer" }));
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toBe("self_hosted provider requires base_url");
    expect(alert.closest("div")?.querySelector("#nc-url")).toBeTruthy();
    expect(document.body.textContent).not.toContain("{");
  });
});
