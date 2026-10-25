// Lattices (2026-10-25; Q115 A, Q116 A, Q117 A). Anthony: "Tables carry
// structured data with child crystals that are editable... if the table
// changes with tatums shot %. That isnt a static fact." A lattice is a
// named table in Crystal storage that goals and AI tools fill. Snapshot
// cells are true once and fixed; live cells change, keep their history and
// show when they are due for a refresh. Browse by entity (a team, a
// player), search, sort, and correct any cell.
import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Table2 } from "lucide-react";
import { api, errorMessage } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";
import type { Lattice, LatticeAggregate, LatticeColumn, LatticeRow, LatticeValue } from "@/lib/types";
import {
  ConfirmInline, Drawer, EmptyState, ErrorBanner, FilterChips, Meter, PageHeader, StatusPill, Toolbar, useToast,
} from "@/components/ui";
import { cn, fmtNum, fmtRelative } from "@/lib/utils";
import { BTN_PRIMARY, BTN_QUIET, BTN_SECONDARY, INPUT } from "./settings/shared";

const PAGE = 50;

/** One cell as text. */
export function cellText(col: LatticeColumn, v: LatticeValue | undefined): string {
  if (v === null || v === undefined || v === "") return "";
  if (col.type === "bool") return v ? "Yes" : "No";
  if (col.type === "number" && typeof v === "number") return v.toLocaleString(undefined, { maximumFractionDigits: 3 });
  return String(v);
}

function Cell({ col, value, stale }: { col: LatticeColumn; value: LatticeValue | undefined; stale: boolean }) {
  const text = cellText(col, value);
  if (!text) return <span className="text-fg-3">–</span>;
  if (col.type === "url" && typeof value === "string") {
    let host = value;
    try { host = new URL(value).hostname; } catch { /* keep the raw text */ }
    return <a href={value} target="_blank" rel="noreferrer" className="text-accent hover:underline" onClick={(e) => e.stopPropagation()}>{host}</a>;
  }
  return <span className={cn(stale && "text-warn")} title={stale ? "Due for a refresh" : undefined}>{text}</span>;
}

function refreshText(minutes?: number): string {
  if (!minutes) return "";
  if (minutes < 60) return `every ${minutes} min`;
  if (minutes < 1440) return `every ${Math.round(minutes / 60)} h`;
  return `every ${Math.round(minutes / 1440)} d`;
}

function LatticeCard({ lat }: { lat: Lattice }) {
  const entities = lat.columns.filter((c) => c.entity).map((c) => c.label);
  const live = lat.columns.filter((c) => c.kind === "live").length;
  return (
    <li>
      <Link to={`/lattices/${encodeURIComponent(lat.lattice_id)}`}
        className="block rounded-xl border border-line bg-surface p-4 hover:border-accent/60 hover:bg-inset">
        <div className="flex flex-wrap items-center gap-2">
          <span className="min-w-0 flex-1 text-[15px] font-semibold text-fg">{lat.name}</span>
          <span className="text-[12.5px] tabular-nums text-fg-2">{fmtNum(lat.rows)} rows</span>
        </div>
        {lat.description && <p className="mt-1 text-[12.5px] text-fg-2">{lat.description}</p>}
        <p className="mt-1 text-[12px] text-fg-3">
          {lat.columns.length} columns{live ? ` · ${live} live` : ""}
          {entities.length ? ` · browse by ${entities.join(", ")}` : ""}
          {lat.updated_at ? ` · updated ${fmtRelative(lat.updated_at)}` : ""}
        </p>
      </Link>
    </li>
  );
}

function LatticeList({ customerId }: { customerId: string }) {
  const list = useQuery({ queryKey: ["lattices", customerId], queryFn: () => api.listLattices(customerId), refetchInterval: 60_000 });
  const lattices = list.data?.lattices ?? [];
  const used = list.data?.rows.used ?? 0;
  const cap = list.data?.rows.cap ?? null;
  return (
    <div className="space-y-4">
      <PageHeader title="Lattices" meta={list.isLoading ? "Loading…" : `${lattices.length} lattice${lattices.length === 1 ? "" : "s"} · ${fmtNum(used)} rows`} />
      <p className="text-[13px] text-fg-2">
        Crystal storage's tables. Goals and your AI tools keep rows here, like games, prices and stats. Snapshot cells are fixed once
        they're true; live cells update and keep what they held before.
      </p>
      {cap !== null && (
        <Meter className="max-w-md" value={used} max={cap} label="Lattice rows on your plan" valueLabel={`${fmtNum(used)} of ${fmtNum(cap)}`} />
      )}
      {list.isError && <ErrorBanner title="Couldn't load lattices" error={list.error} onRetry={() => list.refetch()} />}
      {!list.isLoading && !list.isError && lattices.length === 0 && (
        <EmptyState icon={Table2} title="No lattices yet"
          help="When a goal or your AI keeps many records with the same fields, they land here as a table you can browse and correct." />
      )}
      <ul className="grid gap-3 min-[900px]:grid-cols-2" aria-label="Lattices">
        {lattices.map((l) => <LatticeCard key={l.lattice_id} lat={l} />)}
      </ul>
    </div>
  );
}

function RowDrawer({ customerId, lat, rowId, onClose }: { customerId: string; lat: Lattice; rowId: string; onClose: () => void }) {
  const qc = useQueryClient();
  const toast = useToast();
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const row = useQuery({ queryKey: ["lattice-row", customerId, lat.lattice_id, rowId], queryFn: () => api.latticeRow(customerId, lat.lattice_id, rowId) });
  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["lattice-row", customerId, lat.lattice_id, rowId] });
    qc.invalidateQueries({ queryKey: ["lattice-rows", customerId, lat.lattice_id] });
    qc.invalidateQueries({ queryKey: ["lattice", customerId, lat.lattice_id] });
  };
  const edit = useMutation({
    mutationFn: ({ column, value }: { column: string; value: string }) => api.editLatticeCell(customerId, lat.lattice_id, rowId, column, value),
    onSuccess: () => { setError(null); setEditing(null); toast("Saved."); refresh(); },
    onError: (e) => setError(errorMessage(e, "Couldn't save that cell.")),
  });
  const remove = useMutation({
    mutationFn: () => api.deleteLatticeRows(customerId, lat.lattice_id, [rowId]),
    onSuccess: () => { toast("Row deleted."); refresh(); qc.invalidateQueries({ queryKey: ["lattices", customerId] }); onClose(); },
    onError: (e) => setError(errorMessage(e, "Couldn't delete the row.")),
  });
  const r: LatticeRow | undefined = row.data?.row;
  const title = r ? lat.key_columns.map((k) => cellText(lat.columns.find((c) => c.key === k)!, r.values[k])).join(" · ") : "Row";
  return (
    <Drawer open onClose={onClose} title={title}
      footer={
        <div className="flex items-center gap-2">
          <span className="flex-1" />
          <button type="button" className="text-[12.5px] text-fg-3 hover:text-bad disabled:opacity-50" disabled={remove.isPending}
            onClick={() => setConfirmDelete(true)}>Delete row</button>
        </div>
      }>
      {error && <ErrorBanner title="That didn't go through" message={error} />}
      {row.isError && <ErrorBanner title="Couldn't load the row" error={row.error} onRetry={() => row.refetch()} />}
      <ConfirmInline open={confirmDelete} permanent busy={remove.isPending} confirmLabel="Delete row"
        message="Delete this row? It can't be brought back." onConfirm={() => remove.mutate()} onCancel={() => setConfirmDelete(false)} />
      {r && (
        <dl className="space-y-3" aria-label="Cells">
          {lat.columns.map((c) => {
            const meta = r.cells?.[c.key];
            const isKey = lat.key_columns.includes(c.key);
            const stale = r.stale?.includes(c.key) ?? false;
            return (
              <div key={c.key} className="rounded-lg border border-line px-3 py-2">
                <dt className="flex flex-wrap items-center gap-2 text-[12px] text-fg-3">
                  <span className="font-medium text-fg-2">{c.label}</span>
                  {c.kind === "live" ? <StatusPill variant="info">Live</StatusPill> : <StatusPill variant="neutral">Snapshot</StatusPill>}
                  {stale && <StatusPill variant="warn">Due for a refresh</StatusPill>}
                  {isKey && <span>identifies the row</span>}
                </dt>
                <dd className="mt-1">
                  {editing === c.key ? (
                    <form className="flex flex-wrap items-center gap-2" aria-label={`Edit ${c.label}`}
                      onSubmit={(e) => { e.preventDefault(); edit.mutate({ column: c.key, value: draft }); }}>
                      <input aria-label={c.label} className={INPUT + " min-w-0 flex-1"} value={draft} autoFocus
                        placeholder={c.type === "date" ? "YYYY-MM-DD" : c.type === "bool" ? "yes or no" : ""}
                        onChange={(e) => setDraft(e.target.value)} />
                      <button type="submit" className={BTN_PRIMARY} disabled={edit.isPending}>Save</button>
                      <button type="button" className={BTN_QUIET} onClick={() => setEditing(null)}>Cancel</button>
                    </form>
                  ) : (
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="min-w-0 flex-1 break-words text-[13.5px] text-fg"><Cell col={c} value={r.values[c.key]} stale={stale} /></span>
                      {!isKey && (
                        <button type="button" className={BTN_QUIET}
                          onClick={() => { setEditing(c.key); setDraft(cellText(c, r.values[c.key]).replace(/,/g, "")); }}>Edit</button>
                      )}
                    </div>
                  )}
                  {meta?.as_of && (
                    <p className="mt-1 text-[11.5px] text-fg-3">
                      As of {fmtRelative(meta.as_of)}{meta.source ? ` · ${meta.source}` : ""}{meta.corrected_by ? " · corrected" : ""}
                      {c.kind === "live" && c.refresh_minutes ? ` · refreshes ${refreshText(c.refresh_minutes)}` : ""}
                    </p>
                  )}
                  {(meta?.history?.length ?? 0) > 0 && (
                    <ul className="mt-1 space-y-0.5 text-[11.5px] text-fg-3" aria-label={`${c.label} before`}>
                      {meta!.history!.map((h, i) => (
                        <li key={i}>Before: <span className="text-fg-2">{cellText(c, h.value) || "–"}</span>{h.as_of ? `, as of ${fmtRelative(h.as_of)}` : ""}</li>
                      ))}
                    </ul>
                  )}
                </dd>
              </div>
            );
          })}
        </dl>
      )}
    </Drawer>
  );
}

const FN_LABEL: Record<string, string> = { count: "Count of rows", sum: "Sum", avg: "Average", min: "Minimum", max: "Maximum" };

/** Q121 A: group by a column and count, sum or average a number column, over the rows in view. */
function Analyze({ customerId, lat, filters, search }: {
  customerId: string; lat: Lattice; filters?: { column: string; op: string; value: string }[]; search?: string;
}) {
  const textCols = lat.columns.filter((c) => c.type !== "number");
  const numCols = lat.columns.filter((c) => c.type === "number");
  const [groupBy, setGroupBy] = useState<string>((lat.columns.find((c) => c.entity) ?? textCols[0])?.key ?? "");
  const [fn, setFn] = useState<string>(numCols.length ? "avg" : "count");
  const [column, setColumn] = useState<string>(numCols[0]?.key ?? "");
  const [error, setError] = useState<string | null>(null);
  const run = useMutation({
    mutationFn: (): Promise<LatticeAggregate> => {
      const metrics = fn === "count" ? ["count"] : [`${fn}:${column}`];
      return api.aggregateLattice(customerId, lat.lattice_id, {
        group_by: groupBy ? [groupBy] : [], metrics, sort: fn === "count" ? "count" : `${fn}_${column}`, filters, search,
      });
    },
    onSuccess: () => setError(null),
    onError: (e) => setError(errorMessage(e, "Couldn't run that.")),
  });
  const label = (k: string) => lat.columns.find((c) => c.key === k)?.label ?? k;
  const metricLabel = (m: string) => {
    if (m === "count") return "Rows";
    const [f, ...rest] = m.split("_");
    return `${FN_LABEL[f] ?? f} of ${label(rest.join("_"))}`;
  };
  const out = run.data;
  return (
    <section aria-label="Analyze" className="space-y-3 rounded-lg border border-line bg-surface p-3">
      <form className="flex flex-wrap items-end gap-2" onSubmit={(e) => { e.preventDefault(); run.mutate(); }}>
        <label className="text-[12px] text-fg-2">Group by
          <select aria-label="Group by" className={INPUT + " mt-1"} value={groupBy} onChange={(e) => setGroupBy(e.target.value)}>
            <option value="">Nothing (totals)</option>
            {lat.columns.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}
          </select>
        </label>
        <label className="text-[12px] text-fg-2">Measure
          <select aria-label="Measure" className={INPUT + " mt-1"} value={fn} onChange={(e) => setFn(e.target.value)}>
            <option value="count">Count of rows</option>
            {numCols.length > 0 && ["avg", "sum", "min", "max"].map((f) => <option key={f} value={f}>{FN_LABEL[f]}</option>)}
          </select>
        </label>
        {fn !== "count" && (
          <label className="text-[12px] text-fg-2">Of
            <select aria-label="Of column" className={INPUT + " mt-1"} value={column} onChange={(e) => setColumn(e.target.value)}>
              {numCols.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}
            </select>
          </label>
        )}
        <button type="submit" className={BTN_PRIMARY} disabled={run.isPending || (fn !== "count" && !column)}>Run</button>
      </form>
      <p className="text-[12px] text-fg-3">Uses the rows in view: your search and browse filter apply.</p>
      {error && <ErrorBanner title="That didn't go through" message={error} />}
      {out && (
        <div className="overflow-x-auto">
          <table className="w-full min-w-max border-collapse text-[13px]" aria-label="Results">
            <thead>
              <tr className="border-b border-line text-left">
                {out.group_by.map((g) => <th key={g} scope="col" className="px-3 py-1.5 font-medium text-fg-2">{label(g)}</th>)}
                {out.metrics.map((m) => <th key={m} scope="col" className="px-3 py-1.5 text-right font-medium text-fg-2">{metricLabel(m)}</th>)}
              </tr>
            </thead>
            <tbody>
              {out.groups.map((row, i) => (
                <tr key={i} className="border-b border-line last:border-0">
                  {out.group_by.map((g) => <td key={g} className="px-3 py-1 text-fg">{row[g] === null || row[g] === "" ? "–" : String(row[g])}</td>)}
                  {out.metrics.map((m) => (
                    <td key={m} className="px-3 py-1 text-right tabular-nums text-fg">
                      {row[m] === null || row[m] === undefined ? "–" : Number(row[m]).toLocaleString(undefined, { maximumFractionDigits: 2 })}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          {out.total_groups > out.groups.length && (
            <p className="mt-1 text-[12px] text-fg-3">Showing the top {out.groups.length} of {fmtNum(out.total_groups)} groups.</p>
          )}
        </div>
      )}
    </section>
  );
}

function LatticeView({ customerId, latticeId }: { customerId: string; latticeId: string }) {
  const qc = useQueryClient();
  const toast = useToast();
  const navigate = useNavigate();
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState<{ key: string; desc: boolean } | null>(null);
  const [offset, setOffset] = useState(0);
  const [browseCol, setBrowseCol] = useState<string | null>(null);
  const [filter, setFilter] = useState<{ column: string; value: string } | null>(null);
  const [openRow, setOpenRow] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [exporting, setExporting] = useState(false);
  const head = useQuery({ queryKey: ["lattice", customerId, latticeId], queryFn: () => api.getLattice(customerId, latticeId) });
  const q = { search: search.trim() || undefined, sort: sort?.key, desc: sort?.desc, offset, limit: PAGE, column: filter?.column, value: filter?.value };
  const rows = useQuery({
    queryKey: ["lattice-rows", customerId, latticeId, q],
    queryFn: () => api.latticeRows(customerId, latticeId, q),
    placeholderData: (prev) => prev,
  });
  const remove = useMutation({
    mutationFn: () => api.deleteLattice(customerId, latticeId),
    onSuccess: (out) => {
      toast(`Lattice deleted with its ${fmtNum(out.rows)} rows.`);
      qc.invalidateQueries({ queryKey: ["lattices", customerId] });
      navigate("/lattices");
    },
    onError: (e) => setError(errorMessage(e, "Couldn't delete the lattice.")),
  });
  const lat = head.data?.lattice;
  const viewFilters = {
    filters: filter ? [{ column: filter.column, op: "eq", value: filter.value }] : undefined,
    search: search.trim() || undefined,
  };
  const entities = head.data?.entities ?? [];
  const col = browseCol ?? entities[0]?.column ?? null;
  const values = entities.find((e) => e.column === col)?.values ?? [];
  const total = rows.data?.total ?? 0;
  const list = rows.data?.rows ?? [];
  const toggleSort = (key: string) => {
    setOffset(0);
    setSort((s) => (s?.key !== key ? { key, desc: false } : s.desc ? null : { key, desc: true }));
  };
  if (head.isError) {
    return (
      <>
        <PageHeader title="Lattice" actions={<Link to="/lattices" className={BTN_SECONDARY}><ArrowLeft className="h-4 w-4" />All lattices</Link>} />
        <ErrorBanner title="Couldn't load the lattice" error={head.error} onRetry={() => head.refetch()} />
      </>
    );
  }
  return (
    <div className="space-y-4">
      <PageHeader title={lat?.name ?? "Lattice"}
        meta={lat ? `${fmtNum(lat.rows)} rows · ${lat.columns.length} columns${lat.updated_at ? ` · updated ${fmtRelative(lat.updated_at)}` : ""}` : "Loading…"}
        actions={<Link to="/lattices" className={BTN_SECONDARY}><ArrowLeft className="h-4 w-4" />All lattices</Link>} />
      {lat?.description && <p className="text-[13px] text-fg-2">{lat.description}</p>}
      {error && <ErrorBanner title="That didn't go through" message={error} />}

      {entities.length > 0 && (
        <section aria-label="Browse by" className="space-y-2 rounded-lg border border-line bg-surface p-3">
          {entities.length > 1 && (
            <FilterChips label="Browse by" value={col} onChange={(v) => { setBrowseCol(v ?? entities[0].column); }}
              options={entities.map((e) => ({ value: e.column, label: e.label }))} />
          )}
          <FilterChips label={entities.find((e) => e.column === col)?.label ?? "Values"}
            value={filter && filter.column === col ? filter.value : null}
            onChange={(v) => { setOffset(0); setFilter(v && col ? { column: col, value: v } : null); }}
            options={values.map((v) => ({ value: v.value, label: v.value, count: v.rows }))} />
        </section>
      )}

      <Toolbar search={
        <input aria-label="Search rows" className={INPUT} placeholder="Search rows" value={search}
          onChange={(e) => { setOffset(0); setSearch(e.target.value); }} />
      }>
        {filter && (
          <button type="button" className={BTN_QUIET} onClick={() => { setOffset(0); setFilter(null); }}>
            Clear {entities.find((e) => e.column === filter.column)?.label ?? filter.column}: {filter.value}
          </button>
        )}
        <button type="button" className={BTN_SECONDARY} aria-expanded={analyzing} onClick={() => setAnalyzing((v) => !v)}>Analyze</button>
        <button type="button" className={BTN_SECONDARY} disabled={exporting || !lat}
          onClick={async () => {
            if (!lat) return;
            setExporting(true);
            try {
              await api.exportLatticeCsv(customerId, latticeId, viewFilters,
                `${lat.name.toLowerCase().replace(/[^a-z0-9_-]+/g, "-")}.csv`);
              setError(null);
            } catch (e) {
              setError(errorMessage(e, "Couldn't download the CSV."));
            } finally {
              setExporting(false);
            }
          }}>Download CSV</button>
      </Toolbar>
      {analyzing && lat && <Analyze customerId={customerId} lat={lat} filters={viewFilters.filters} search={viewFilters.search} />}

      {rows.isError && <ErrorBanner title="Couldn't load rows" error={rows.error} onRetry={() => rows.refetch()} />}
      {lat && (
        <div className="overflow-x-auto rounded-lg border border-line bg-surface">
          <table className="w-full min-w-max border-collapse text-[13px]">
            <thead>
              <tr className="border-b border-line text-left">
                {lat.columns.map((c) => (
                  <th key={c.key} scope="col" className="whitespace-nowrap px-3 py-2 font-medium text-fg-2"
                    aria-sort={sort?.key === c.key ? (sort.desc ? "descending" : "ascending") : "none"}>
                    <button type="button" className="inline-flex items-center gap-1 hover:text-fg" onClick={() => toggleSort(c.key)}>
                      {c.label}
                      {c.kind === "live" && <span className="rounded bg-accent-soft px-1 text-[10.5px] text-accent" title={c.refresh_minutes ? `Live, refreshes ${refreshText(c.refresh_minutes)}` : "Live"}>live</span>}
                      {sort?.key === c.key && <span aria-hidden>{sort.desc ? "↓" : "↑"}</span>}
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {list.map((r) => (
                <tr key={r.row_id} className="cursor-pointer border-b border-line last:border-0 hover:bg-inset" tabIndex={0}
                  aria-label={`Row ${lat.key_columns.map((k) => String(r.values[k] ?? "")).join(" ")}`}
                  onClick={() => setOpenRow(r.row_id)} onKeyDown={(e) => { if (e.key === "Enter") setOpenRow(r.row_id); }}>
                  {lat.columns.map((c) => (
                    <td key={c.key} className={cn("max-w-[280px] truncate px-3 py-1.5 text-fg", c.type === "number" && "text-right tabular-nums")}>
                      <Cell col={c} value={r.values[c.key]} stale={r.stale?.includes(c.key) ?? false} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          {!rows.isLoading && list.length === 0 && (
            <p className="px-3 py-6 text-center text-[13px] text-fg-3">{search || filter ? "No rows match." : "No rows yet."}</p>
          )}
        </div>
      )}
      {total > 0 && (
        <div className="flex flex-wrap items-center gap-2 text-[12.5px] text-fg-3">
          <span className="flex-1">Rows {fmtNum(offset + 1)} to {fmtNum(Math.min(offset + PAGE, total))} of {fmtNum(total)}</span>
          <button type="button" className={BTN_SECONDARY} disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>Previous</button>
          <button type="button" className={BTN_SECONDARY} disabled={offset + PAGE >= total} onClick={() => setOffset(offset + PAGE)}>Next</button>
        </div>
      )}

      <div className="flex items-center">
        <span className="flex-1" />
        <button type="button" className="text-[12.5px] text-fg-3 hover:text-bad disabled:opacity-50" disabled={remove.isPending}
          onClick={() => setConfirmDelete(true)}>Delete lattice</button>
      </div>
      <ConfirmInline open={confirmDelete} permanent busy={remove.isPending} confirmLabel="Delete lattice"
        message={`Delete ${lat?.name ?? "this lattice"} and all ${fmtNum(lat?.rows ?? 0)} rows? Goals that write to it will make a new one.`}
        onConfirm={() => remove.mutate()} onCancel={() => setConfirmDelete(false)} />

      {openRow && lat && <RowDrawer customerId={customerId} lat={lat} rowId={openRow} onClose={() => setOpenRow(null)} />}
    </div>
  );
}

export function LatticesPage() {
  const { selectedCustomerId } = useSelectedCustomer();
  const { latticeId } = useParams();
  if (!selectedCustomerId) {
    return (
      <>
        <PageHeader title="Lattices" />
        <EmptyState icon={Table2} title="No customer selected" help="Pick a customer above to see their lattices." />
      </>
    );
  }
  return latticeId
    ? <LatticeView key={latticeId} customerId={selectedCustomerId} latticeId={latticeId} />
    : <LatticeList customerId={selectedCustomerId} />;
}
