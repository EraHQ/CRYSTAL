// @vitest-environment jsdom
// Lattices (2026-10-25; Q115 A, Q116 A): the list shows each table and the
// row cap; a lattice browses by entity, searches, sorts, and opens a row
// whose cells can be corrected, with live cells' history and staleness.
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Lattice, LatticeRow } from "@/lib/types";

const players: Lattice = {
  lattice_id: "lat_1", name: "NBA players", description: "Season shooting by player.", rows: 3, key_columns: ["player"],
  columns: [
    { key: "player", label: "Player", type: "text", kind: "snapshot", entity: true },
    { key: "team", label: "Team", type: "text", kind: "snapshot", entity: true },
    { key: "fg_pct", label: "FG %", type: "number", kind: "live", entity: false, refresh_minutes: 1440 },
    { key: "profile", label: "Profile", type: "url", kind: "snapshot", entity: false },
  ],
  updated_at: new Date().toISOString(),
};
const rows: LatticeRow[] = [
  { row_id: "lr_1", values: { player: "Jayson Tatum", team: "Celtics", fg_pct: 46.3, profile: "https://www.nba.com/player/1" }, updated_at: null, stale: ["fg_pct"] },
  { row_id: "lr_2", values: { player: "Jaylen Brown", team: "Celtics", fg_pct: 49, profile: null }, updated_at: null },
];
const full: LatticeRow = {
  ...rows[0],
  cells: { fg_pct: { as_of: new Date(Date.now() - 2 * 86_400_000).toISOString(), source: "nba.com", history: [{ value: 45.1, as_of: null }] } },
};
const m = {
  listLattices: vi.fn(async () => ({ lattices: [players], count: 1, rows: { used: 3, cap: 5000 } })),
  getLattice: vi.fn(async () => ({ lattice: players, entities: [
    { column: "player", label: "Player", values: [{ value: "Jayson Tatum", rows: 1 }, { value: "Jaylen Brown", rows: 1 }] },
    { column: "team", label: "Team", values: [{ value: "Celtics", rows: 2 }, { value: "Lakers", rows: 1 }] },
  ] })),
  latticeRows: vi.fn(async () => ({ total: 2, offset: 0, rows })),
  latticeRow: vi.fn(async () => ({ row: full })),
  editLatticeCell: vi.fn(async () => ({ row: full, action_id: "sa_1" })),
  deleteLatticeRows: vi.fn(async () => ({ deleted: 1 })),
  deleteLattice: vi.fn(async () => ({ deleted: true, rows: 3 })),
  aggregateLattice: vi.fn(async () => ({ lattice: "NBA players", group_by: ["team"], metrics: ["count", "avg_fg_pct"],
    groups: [{ team: "Celtics", count: 2, avg_fg_pct: 47.65 }, { team: "Lakers", count: 1, avg_fg_pct: 52 }], total_groups: 2 })),
  exportLatticeCsv: vi.fn(async () => undefined),
};
vi.mock("@/lib/api", () => ({
  api: new Proxy({}, { get: (_t, k: string) => (...a: unknown[]) => (m as Record<string, (...x: unknown[]) => unknown>)[k](...a) }),
  errorMessage: (e: unknown, fallback = "Something went wrong.") => (e instanceof Error ? e.message : fallback),
}));
vi.mock("@/lib/selected-customer", () => ({ useSelectedCustomer: () => ({ selectedCustomerId: "cus_1" }) }));

import { ToastProvider } from "@/components/ui";
import { LatticesPage, cellText } from "./Lattices";

function mount(path: string) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}><ToastProvider><MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/lattices" element={<LatticesPage />} />
        <Route path="/lattices/:latticeId" element={<LatticesPage />} />
      </Routes>
    </MemoryRouter></ToastProvider></QueryClientProvider>,
  );
}

afterEach(() => { cleanup(); vi.clearAllMocks(); });

describe("Lattices (Q115 A)", () => {
  it("words for cells", () => {
    expect(cellText(players.columns[2], 46.3)).toBe("46.3");
    expect(cellText({ ...players.columns[2], type: "bool" }, false)).toBe("No");
    expect(cellText(players.columns[0], null)).toBe("");
  });

  it("lists each lattice with its rows and the plan's row cap", async () => {
    mount("/lattices");
    const card = await screen.findByRole("link", { name: /NBA players/ });
    expect(card.getAttribute("href")).toBe("/lattices/lat_1");
    expect(card.textContent).toContain("3 rows");
    expect(card.textContent).toContain("1 live");
    expect(card.textContent).toContain("browse by Player, Team");
    expect(screen.getByRole("progressbar", { name: "Lattice rows on your plan" })).toBeTruthy();
  });

  it("browses by entity, searches and sorts through the server", async () => {
    mount("/lattices/lat_1");
    expect(await screen.findByRole("heading", { name: "NBA players" })).toBeTruthy();
    const table = await screen.findByRole("table");
    expect(within(table).getByText("Jayson Tatum")).toBeTruthy();
    // A stale live cell says so; a url shows its host.
    expect(within(table).getByTitle("Due for a refresh").textContent).toBe("46.3");
    expect(within(table).getByRole("link", { name: "www.nba.com" })).toBeTruthy();
    // Browse by Team, then pick Celtics.
    const browse = screen.getByRole("region", { name: "Browse by" });
    fireEvent.click(within(browse).getByRole("button", { name: "Team" }));
    fireEvent.click(await within(browse).findByRole("button", { name: /Celtics/ }));
    await waitFor(() => expect(m.latticeRows).toHaveBeenLastCalledWith("cus_1", "lat_1", expect.objectContaining({ column: "team", value: "Celtics", offset: 0 })));
    fireEvent.change(screen.getByRole("textbox", { name: "Search rows" }), { target: { value: "jay" } });
    await waitFor(() => expect(m.latticeRows).toHaveBeenLastCalledWith("cus_1", "lat_1", expect.objectContaining({ search: "jay" })));
    fireEvent.click(within(table).getByRole("button", { name: /FG %/ }));
    await waitFor(() => expect(m.latticeRows).toHaveBeenLastCalledWith("cus_1", "lat_1", expect.objectContaining({ sort: "fg_pct", desc: false })));
    fireEvent.click(within(table).getByRole("button", { name: /FG %/ }));
    await waitFor(() => expect(m.latticeRows).toHaveBeenLastCalledWith("cus_1", "lat_1", expect.objectContaining({ sort: "fg_pct", desc: true })));
  });

  it("opens a row, shows a live cell's history and corrects a cell", async () => {
    mount("/lattices/lat_1");
    fireEvent.click(await screen.findByRole("row", { name: /Row Jayson Tatum/ }));
    const cells = await screen.findByRole("list", { name: "FG % before" });
    expect(cells.textContent).toContain("45.1");
    expect(screen.getByText(/As of .* · nba\.com · refreshes every 1 d/)).toBeTruthy();
    expect(screen.getAllByText("Due for a refresh").length).toBeGreaterThan(0);
    // The key column can't be edited; FG % can.
    const edits = screen.getAllByRole("button", { name: "Edit" });
    expect(edits.length).toBe(3);
    fireEvent.click(edits[1]);
    const input = screen.getByRole("textbox", { name: "FG %" });
    fireEvent.change(input, { target: { value: "47.2" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(m.editLatticeCell).toHaveBeenCalledWith("cus_1", "lat_1", "lr_1", "fg_pct", "47.2"));
  });

  it("analyzes the rows in view and downloads them as CSV", async () => {
    mount("/lattices/lat_1");
    await screen.findByRole("table");
    const browse = screen.getByRole("region", { name: "Browse by" });
    fireEvent.click(within(browse).getByRole("button", { name: "Team" }));
    fireEvent.click(await within(browse).findByRole("button", { name: /Celtics/ }));
    fireEvent.click(screen.getByRole("button", { name: "Analyze" }));
    const panel = screen.getByRole("region", { name: "Analyze" });
    fireEvent.change(within(panel).getByRole("combobox", { name: "Group by" }), { target: { value: "team" } });
    fireEvent.click(within(panel).getByRole("button", { name: "Run" }));
    await waitFor(() => expect(m.aggregateLattice).toHaveBeenCalledWith("cus_1", "lat_1", {
      group_by: ["team"], metrics: ["avg:fg_pct"], sort: "avg_fg_pct",
      filters: [{ column: "team", op: "eq", value: "Celtics" }], search: undefined,
    }));
    const results = await within(panel).findByRole("table", { name: "Results" });
    expect(within(results).getByText("Average of FG %")).toBeTruthy();
    expect(within(results).getByText("47.65")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Download CSV" }));
    await waitFor(() => expect(m.exportLatticeCsv).toHaveBeenCalledWith("cus_1", "lat_1",
      { filters: [{ column: "team", op: "eq", value: "Celtics" }], search: undefined }, "nba-players.csv"));
  });

  it("deletes a lattice only after confirming", async () => {
    mount("/lattices/lat_1");
    fireEvent.click(await screen.findByRole("button", { name: "Delete lattice" }));
    expect(m.deleteLattice).not.toHaveBeenCalled();
    const confirm = await screen.findAllByRole("button", { name: "Delete lattice" });
    fireEvent.click(confirm[confirm.length - 1]);
    await waitFor(() => expect(m.deleteLattice).toHaveBeenCalledWith("cus_1", "lat_1"));
  });
});
