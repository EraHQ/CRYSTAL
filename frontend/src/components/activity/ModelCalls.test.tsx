// @vitest-environment jsdom
// Model calls: every ledger row in words with tokens; a run's call links
// to its pipeline; dollars for platform admins only.
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ModelCall } from "@/lib/types";

const calls: ModelCall[] = [
  { id: "l1", session_id: "env_abc", parent_session_id: null, origin: "cognition", model: "nvidia/nemotron-3-super-120b-a12b", input_tokens: 120000, output_tokens: 4000, cache_creation_tokens: 0, cache_read_tokens: 0, cost_micro_usd: 39600, created_at: "2026-10-19T22:00:00Z" },
  { id: "l2", session_id: "seq_1", parent_session_id: null, origin: "agent", model: "claude-sonnet-5-5", input_tokens: 3000, output_tokens: 200, cache_creation_tokens: 0, cache_read_tokens: 10000, cost_micro_usd: 1200, created_at: "2026-10-19T21:00:00Z" },
  { id: "l3", session_id: null, parent_session_id: null, origin: "scan_contradiction", model: "claude-haiku-5-5", input_tokens: 500, output_tokens: 50, cache_creation_tokens: 0, cache_read_tokens: 0, cost_micro_usd: 100, created_at: "2026-10-19T20:00:00Z" },
];
const m = { listModelCalls: vi.fn(async () => ({ calls, count: calls.length })) };
vi.mock("@/lib/api", () => ({
  api: new Proxy({}, { get: (_t, k: string) => (...a: unknown[]) => (m as Record<string, (...x: unknown[]) => unknown>)[k](...a) }),
  errorMessage: (e: unknown, fallback = "Something went wrong.") => (e instanceof Error ? e.message : fallback),
}));

import { ModelCalls } from "./ModelCalls";

afterEach(cleanup);

function mount(admin: boolean) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={qc}><MemoryRouter><ModelCalls customerId="cus_1" admin={admin} /></MemoryRouter></QueryClientProvider>);
}

describe("ModelCalls", () => {
  it("lists calls in words, links a run, filters by activity, and hides dollars from a tenant", async () => {
    mount(false);
    expect(await screen.findByText("3 calls · the latest 200 · 137,750 tokens shown")).toBeTruthy();
    const table = within(screen.getByRole("table"));
    expect(table.getByText("Research run")).toBeTruthy();
    expect(table.getByText("Conflict scan")).toBeTruthy();
    expect(table.getByText("Nemotron Super")).toBeTruthy();
    expect(table.getByText("120,000")).toBeTruthy();
    expect((table.getByRole("link", { name: "a run" }) as HTMLAnchorElement).getAttribute("href")).toBe("/activity/run/env_abc");
    expect(table.getByText("a conversation")).toBeTruthy();
    expect(document.body.textContent).not.toMatch(/\$/);
    const chips = within(screen.getByRole("group", { name: "Activity" }));
    fireEvent.click(chips.getByRole("button", { name: /Curation/ }));
    expect(screen.queryByText("Research run")).toBeNull();
    expect(screen.getByText("Conflict scan")).toBeTruthy();
  });
  it("shows cost to a platform admin", async () => {
    mount(true);
    const table = within(await screen.findByRole("table"));
    expect(table.getByText("$0.0396")).toBeTruthy();
  });
});
