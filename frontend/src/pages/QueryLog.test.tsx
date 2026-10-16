// @vitest-environment jsdom
// Console Refine PR 11 (docs/CONSOLE_UX_AUDIT.md 5.7): the log names the
// tool that asked and the memories that answered as text; searching a
// memory phrase finds the queries it answered; match and tool chips
// count what is loaded; no fact id reaches the table; Load more pages
// through the rest; the raw metrics are admin-only.
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { ReactNode } from "react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const Q = (over: Record<string, unknown>) => ({
  id: "q", timestamp: "2026-10-16T12:00:00Z", query_text: "what is the rate?", match_type: "high", injection_method: "mcp_tool",
  matched_facts: ["fq1"], matched_memories: [{ id: "fq1", text: "the rate is 95" }], response_text: null, prompt_tokens: null,
  completion_tokens: null, prompt_token_overhead: null, shadow_ran: false, shadow_delta: null, concept_top_config: null,
  concept_top_score: null, concept_payload: null, latency_ms: 42, client: "Claude-User", ...over,
});
const page1 = [
  Q({ id: "q1" }),
  Q({ id: "q2", query_text: "deploy windows?", match_type: "medium", client: "Cursor", matched_facts: ["fa", "fb"],
      matched_memories: [{ id: "fa", text: "Friday deploys carry risk" }, { id: "fb", text: "deploys banned in March" }], response_text: "Avoid Fridays.", prompt_tokens: 1200 }),
  Q({ id: "q3", query_text: "console asked", match_type: "none", client: "console", matched_facts: [], matched_memories: [] }),
];
const m = {
  listQueryLogs: vi.fn(async (_c: string, o: { offset?: number }) => (o.offset ? { total: 4, items: [Q({ id: "q4", query_text: "older one", client: null })] } : { total: 4, items: page1 })),
};
vi.mock("@/lib/api", () => ({
  api: new Proxy({}, { get: (_t, k: string) => (...a: unknown[]) => (m as Record<string, (...x: unknown[]) => unknown>)[k](...a) }),
  errorMessage: (e: unknown, fallback = "Something went wrong.") => (e instanceof Error ? e.message : fallback),
}));
const auth = { me: { kind: "user", role: "owner", customer_id: "cus_1" } as { kind: string; role: string; customer_id: string } };
vi.mock("@/lib/auth", () => ({ useAuth: () => auth }));

import { SelectedCustomerProvider } from "@/lib/selected-customer";
import { ToastProvider } from "@/components/ui";
import { QueryLog, matchesSearch, toolLabel } from "./QueryLog";

function mount() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const tree = (children: ReactNode) => (
    <QueryClientProvider client={qc}><ToastProvider><MemoryRouter initialEntries={["/queries"]}>
      <SelectedCustomerProvider pinnedId="cus_1">{children}</SelectedCustomerProvider>
    </MemoryRouter></ToastProvider></QueryClientProvider>
  );
  return render(tree(<QueryLog />));
}

beforeEach(() => { vi.clearAllMocks(); auth.me = { kind: "user", role: "owner", customer_id: "cus_1" }; });
afterEach(cleanup);

describe("Query log (Refine PR 11)", () => {
  it("names the tool and the memories that answered, never an id, and counts the chips", async () => {
    mount();
    expect(await screen.findByText("4 questions · showing the latest 3")).toBeTruthy();
    const table = within(screen.getByRole("table"));
    const row = table.getByText("deploy windows?").closest("tr")!;
    expect(within(row).getByText("Cursor")).toBeTruthy();
    expect(within(row).getByText("Partial").closest("[title]")?.getAttribute("title")).toBe("medium");
    expect(within(row).getByText(/Friday deploys carry risk/)).toBeTruthy();
    expect(within(row).getByText("+1 more")).toBeTruthy();
    expect(table.getByText("Claude")).toBeTruthy();
    expect(table.getByText("Console")).toBeTruthy();
    expect(document.body.textContent).not.toContain("fq1");
    const chips = within(screen.getByRole("group", { name: "Match" }));
    expect(chips.getByRole("button", { name: /Strong/ }).textContent).toContain("1");
    expect(chips.getByRole("button", { name: /None/ }).getAttribute("title")).toBe("none");
    const tools = within(screen.getByRole("group", { name: "Tool" }));
    expect(tools.getAllByRole("button").map((b) => b.textContent?.replace(/\d+$/, "").trim())).toEqual(["Any", "Claude", "Cursor", "Console"]);
    fireEvent.click(tools.getByRole("button", { name: /Cursor/ }));
    expect(screen.queryByText("what is the rate?")).toBeNull();
    expect(screen.getByText("deploy windows?")).toBeTruthy();
  });

  it("finds a query by the memory that answered it, expands to the details, and hides raw metrics from tenants", async () => {
    mount();
    await screen.findByText("deploy windows?");
    fireEvent.change(screen.getByRole("textbox", { name: "Search" }), { target: { value: "banned in march" } });
    expect(screen.queryByText("what is the rate?")).toBeNull();
    const row = screen.getByText("deploy windows?").closest("tr")!;
    fireEvent.click(row);
    expect(await screen.findByText("deploys banned in March")).toBeTruthy();
    expect(screen.getByText("Asked by").nextElementSibling?.textContent).toBe("Cursor");
    expect(screen.getByText("1,200 tokens")).toBeTruthy();
    expect(screen.getByText("Memories checked").nextElementSibling?.textContent).toBe("2");
    expect(screen.getByText("See full answer")).toBeTruthy();
    expect(screen.queryByText("Raw (admin)")).toBeNull();
    expect(matchesSearch(page1[1] as never, "FRIDAY")).toBe(true);
    expect(matchesSearch(page1[1] as never, "nope")).toBe(false);
    expect(toolLabel("python-httpx")).toBe("Python client");
    expect(toolLabel(null)).toBe("Unknown");
    expect(toolLabel("SomethingElse")).toBe("SomethingElse");
  });

  it("Load more pages through the rest, and a platform admin gets the Raw disclosure", async () => {
    auth.me = { kind: "platform_admin_key", role: "platform_admin", customer_id: "" };
    mount();
    fireEvent.click(await screen.findByRole("button", { name: "Load more (1 older)" }));
    await waitFor(() => expect(m.listQueryLogs).toHaveBeenCalledWith("cus_1", { offset: 3, limit: 100 }));
    expect(await screen.findByText("older one")).toBeTruthy();
    expect(screen.queryByRole("button", { name: /Load more/ })).toBeNull();
    fireEvent.click(screen.getByText("older one").closest("tr")!);
    expect(await screen.findByText("Raw (admin)")).toBeTruthy();
    expect(within(screen.getByRole("table")).getAllByText("Unknown").length).toBeGreaterThan(0);
  });
});
