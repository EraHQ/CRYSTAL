// Crystal Bank v2 — Shelves + Reader (ratified 2026-07-15).
//
// Shelves: crystals grouped by a switchable axis (Domain / Source /
// Kind / Quality) derived from sparse-key parts — collapsible shelf
// headers with counts, cards not rows. Reader: one crystal at full
// fidelity — facts as cards with the FULL claim text (zero-truncation
// rule extends to the bank: no clamps, long facts just grow), inline
// edit (supersede) + retire actions, and the append-only Ledger tab
// (fact_ledger — before→after with actor/timestamp). Raw JSON is
// demoted to the last tab. Constellation (the graph) is the next
// slice and gets its own tab then.
import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Archive, ArrowLeft, ChevronDown, ChevronRight, Pencil, Search, X,
} from "lucide-react";
import { api, authedFetch } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";
import { EmptyState, ErrorBanner, JsonView, LoadingRows, QualityPill } from "@/components/ui";
import type { CrystalSummary, FactSummary } from "@/lib/types";
import { cn, fmtDateTime, truncate } from "@/lib/utils";
import { Constellation } from "@/components/bank/Constellation";

type Kind = "reflection" | "pattern" | "ingested" | "knowledge";

function classify(c: CrystalSummary): { kind: Kind; breadcrumb: string; title: string } {
  const segs = (c.headline_key ?? "").split("|").map((s) => s.trim()).filter(Boolean);
  const first = segs[0] ?? "";

  let kind: Kind = "knowledge";
  if (c.crystal_type === "reflection" || first === "Reflections") kind = "reflection";
  else if (first === "General") kind = "pattern";
  else if (first === "Code" || c.headline_source_kind === "document_chunk" || c.build_method === "content_chunk")
    kind = "ingested";

  let title: string;
  let breadcrumb = "";
  if (segs.length > 0) {
    title = segs[segs.length - 1];
    breadcrumb = segs.slice(0, -1).join(" › ");
  } else {
    title = c.summary_text ? truncate(c.summary_text, 80) : c.crystal_type ?? "(untitled crystal)";
  }
  return { kind, breadcrumb, title };
}

const KIND_LABEL: Record<Kind, string> = {
  reflection: "Reflection", pattern: "Pattern",
  ingested: "Ingested", knowledge: "Knowledge",
};
const KIND_CLS: Record<Kind, string> = {
  reflection: "bg-violet-50 text-violet-700 ring-violet-200/60",
  pattern: "bg-brand-50 text-brand-700 ring-brand-200/60",
  ingested: "bg-amber-50 text-amber-700 ring-amber-200/60",
  knowledge: "bg-sky-50 text-sky-700 ring-sky-200/60",
};

// ------------------------------------------------------ shelf grouping

type Axis = "domain" | "source" | "kind" | "quality";
const AXES: Array<{ id: Axis; label: string }> = [
  { id: "domain", label: "Domain" },
  { id: "source", label: "Source" },
  { id: "kind", label: "Kind" },
  { id: "quality", label: "Quality" },
];

interface Shelved {
  c: CrystalSummary;
  kind: Kind;
  breadcrumb: string;
  title: string;
}

function shelfKey(item: Shelved, axis: Axis): string {
  // Observed key shape: leading segment is the broad domain
  // ("Python", "Docs"), trailing segment is the crystal's own subject
  // — grouping by the LAST segment gave every crystal a private shelf
  // (2026-07-15 fix).
  const segs = (item.c.headline_key ?? "").split("|").map((s) => s.trim()).filter(Boolean);
  if (axis === "domain") return segs[0] || "Unsorted";
  if (axis === "source")
    return segs.length > 2 ? segs.slice(0, 2).join(" › ") : segs[0] || "Unsorted";
  if (axis === "kind") return KIND_LABEL[item.kind];
  return item.c.quality_tier || "untiered";
}

// ---------------------------------------------------------- ledger api

interface LedgerEntry {
  id: string; op: string; actor: string;
  fact_id: string; successor_fact_id: string | null;
  before_prompt: string | null; before_text: string | null;
  after_text: string | null; created_at: string | null;
}

async function fetchLedger(crystalId: string): Promise<LedgerEntry[]> {
  const res = await authedFetch(`/admin/api/crystals/${encodeURIComponent(crystalId)}/ledger`);
  if (!res.ok) throw new Error(`${res.status}`);
  return (await res.json()).ledger ?? [];
}

async function postFactOp(
  crystalId: string, factId: string, op: "supersede" | "retire",
  body?: { text: string; prompt_text?: string },
): Promise<void> {
  const res = await authedFetch(
    `/admin/api/crystals/${encodeURIComponent(crystalId)}/facts/${encodeURIComponent(factId)}/${op}`,
    { method: "POST", body: JSON.stringify(body ?? {}) });
  if (!res.ok) throw new Error(`${res.status}`);
}

// ------------------------------------------------------------ fact card

function FactCard({ crystalId, fact }: { crystalId: string; fact: FactSummary }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(fact.claim_text);
  const [confirmRetire, setConfirmRetire] = useState(false);
  const qc = useQueryClient();
  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ["crystal"] });
    qc.invalidateQueries({ queryKey: ["crystal-ledger", crystalId] });
    qc.invalidateQueries({ queryKey: ["crystals"] });
  };
  const supersede = useMutation({
    mutationFn: () => postFactOp(crystalId, fact.id, "supersede", { text: draft.trim() }),
    onSuccess: () => { setEditing(false); invalidate(); },
  });
  const retire = useMutation({
    mutationFn: () => postFactOp(crystalId, fact.id, "retire"),
    onSuccess: invalidate,
  });

  return (
    <div className="bg-white border border-gray-200 rounded-xl px-4 py-3">
      {editing ? (
        <div>
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            rows={Math.max(3, Math.ceil(draft.length / 90))}
            className="w-full text-sm text-gray-800 border border-gray-200 rounded p-2 outline-none focus:border-indigo-300 resize-y"
          />
          <div className="flex items-center justify-between mt-1.5">
            <span className="text-[10px] text-gray-400">
              Saving replaces this fact and records the change in the ledger — the original text is kept forever.
            </span>
            <div className="flex gap-2">
              <button onClick={() => { setEditing(false); setDraft(fact.claim_text); }}
                className="text-xs px-2 py-1 rounded border border-gray-200 text-gray-500 hover:bg-gray-50">Cancel</button>
              <button onClick={() => supersede.mutate()}
                disabled={!draft.trim() || draft.trim() === fact.claim_text || supersede.isPending}
                className="text-xs px-2 py-1 rounded bg-indigo-600 text-white disabled:opacity-40 hover:bg-indigo-700">
                {supersede.isPending ? "Saving…" : "Save as new version"}
              </button>
            </div>
          </div>
          {supersede.isError && <p className="text-[10px] text-red-600 mt-1">Couldn't save — try again.</p>}
        </div>
      ) : (
        <>
          {/* Full claim text — never clamped (zero-truncation rule). */}
          <p className="text-sm text-gray-800 whitespace-pre-wrap break-words leading-relaxed">{fact.claim_text}</p>
          {fact.prompt_text && (
            <p className="text-[11px] text-gray-400 mt-1.5 whitespace-pre-wrap break-words">{fact.prompt_text}</p>
          )}
          <div className="flex items-center gap-2.5 mt-2 text-[11px] text-gray-400 flex-wrap">
            <span>{fact.pair_type}</span><span>·</span><span>{fact.source_kind}</span>
            <span>·</span><span>{fmtDateTime(fact.created_at)}</span>
            <span className="flex-1" />
            {confirmRetire ? (
              <span className="inline-flex items-center gap-2">
                <span className="text-amber-700">Retire this fact? Its text stays in the ledger.</span>
                <button onClick={() => retire.mutate()} disabled={retire.isPending}
                  className="text-amber-700 font-medium hover:text-amber-900">
                  {retire.isPending ? "Retiring…" : "Confirm"}
                </button>
                <button onClick={() => setConfirmRetire(false)} className="text-gray-400 hover:text-gray-600">
                  <X className="h-3 w-3" />
                </button>
              </span>
            ) : (
              <>
                <button onClick={() => setEditing(true)}
                  className="inline-flex items-center gap-1 text-gray-500 hover:text-gray-700">
                  <Pencil className="h-3 w-3" /> edit
                </button>
                <button onClick={() => setConfirmRetire(true)}
                  className="inline-flex items-center gap-1 text-gray-500 hover:text-gray-700">
                  <Archive className="h-3 w-3" /> retire
                </button>
              </>
            )}
          </div>
        </>
      )}
    </div>
  );
}

// -------------------------------------------------------------- ledger

const OP_CLS: Record<string, string> = {
  supersede: "bg-sky-50 text-sky-700",
  retire: "bg-amber-50 text-amber-700",
};

function LedgerTimeline({ entries }: { entries: LedgerEntry[] }) {
  if (!entries.length) {
    return (
      <p className="text-xs text-gray-400">
        No changes yet. Every edit or retirement lands here permanently — the ledger is append-only and can't be rewritten.
      </p>
    );
  }
  return (
    <div className="border-l-2 border-gray-200 pl-4 space-y-4">
      {entries.map((e) => (
        <div key={e.id}>
          <div className="flex items-center gap-2 text-[11px] mb-1">
            <span className={cn("px-1.5 rounded", OP_CLS[e.op] ?? "bg-gray-100 text-gray-600")}>{e.op}</span>
            <span className="text-gray-400">{e.actor}</span>
            <span className="text-gray-300">{e.created_at ? fmtDateTime(e.created_at) : ""}</span>
            <span className="text-gray-300 font-mono">{e.fact_id}</span>
          </div>
          {e.before_text && (
            <p className="text-xs text-gray-500 line-through whitespace-pre-wrap break-words">{e.before_text}</p>
          )}
          {e.after_text && (
            <p className="text-xs text-gray-700 whitespace-pre-wrap break-words">→ {e.after_text}</p>
          )}
        </div>
      ))}
    </div>
  );
}

// -------------------------------------------------------------- reader

function CrystalReader({ crystalId, onBack }: { crystalId: string; onBack: () => void }) {
  const { selectedCustomerId } = useSelectedCustomer();
  const [tab, setTab] = useState<"facts" | "ledger" | "raw">("facts");

  const detail = useQuery({
    queryKey: ["crystal", selectedCustomerId, crystalId],
    queryFn: () => api.getCrystal(selectedCustomerId!, crystalId),
    enabled: !!selectedCustomerId,
  });
  const ledger = useQuery({
    queryKey: ["crystal-ledger", crystalId],
    queryFn: () => fetchLedger(crystalId),
  });

  const d = detail.data;
  const entries = ledger.data ?? [];

  return (
    <div>
      <button onClick={onBack}
        className="inline-flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-700 mb-3">
        <ArrowLeft className="h-3.5 w-3.5" /> All crystals
      </button>
      {detail.isLoading && <LoadingRows rows={4} />}
      {detail.isError && <ErrorBanner title="Couldn't load crystal" message={String(detail.error)} />}
      {d && (
        <>
          <div className="flex items-baseline gap-2.5 flex-wrap">
            <h2 className="text-base font-semibold text-gray-900">{d.summary_text ? truncate(d.summary_text, 90) : d.id}</h2>
            <QualityPill tier={d.quality_tier} />
            {/* Gate D4a: crystal operations — the curator outranks the
                heuristics. Tier select whitelists a quarantined crystal;
                Delete removes residue no replace path can reach. */}
            <span className="ml-auto flex items-center gap-2">
              <select
                value={d.quality_tier ?? "neutral"}
                onChange={async (e) => {
                  const tier = e.target.value;
                  await authedFetch(`/admin/api/crystals/${encodeURIComponent(d.id)}/tier`, {
                    method: "PATCH",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ tier }),
                  });
                  detail.refetch();
                }}
                className="rounded border border-gray-200 bg-white px-2 py-1 text-xs text-gray-600 focus:outline-none focus:border-brand-500"
                title="Set quality tier">
                <option value="verified">verified</option>
                <option value="neutral">neutral</option>
                <option value="quarantine">quarantine</option>
                <option value="blacklist">blacklist</option>
              </select>
              <button
                onClick={async () => {
                  if (!window.confirm(`Delete crystal ${d.id} and all ${d.fact_count} facts? This cannot be undone.`)) return;
                  await authedFetch(`/admin/api/crystals/${encodeURIComponent(d.id)}`, { method: "DELETE" });
                  window.history.back();
                }}
                className="rounded border border-red-200 bg-red-50 px-2 py-1 text-xs font-medium text-red-600 hover:bg-red-100"
                title="Delete crystal">
                Delete
              </button>
            </span>
          </div>
          <p className="text-xs text-gray-400 mt-0.5 mb-4">
            {d.fact_count} facts · {d.build_method} · created {fmtDateTime(d.created_at)}
            {d.parent_crystal_id ? ` · child of ${d.parent_crystal_id}` : ""}
          </p>

          <div className="flex items-center gap-4 border-b border-gray-100 mb-4 text-xs">
            {([
              ["facts", `Facts (${(d.facts ?? []).length})`],
              ["ledger", `Ledger${entries.length ? ` (${entries.length})` : ""}`],
              ["raw", "Raw"],
            ] as const).map(([id, label]) => (
              <button key={id} onClick={() => setTab(id)}
                className={cn("pb-2 -mb-px border-b-2",
                  tab === id ? "border-indigo-500 text-indigo-700 font-medium"
                             : "border-transparent text-gray-400 hover:text-gray-600")}>
                {label}
              </button>
            ))}
          </div>

          {tab === "facts" && (
            <div className="space-y-2.5 max-w-3xl">
              {(d.facts ?? []).map((f) => (
                <FactCard key={f.id} crystalId={crystalId} fact={f} />
              ))}
              {!(d.facts ?? []).length && <p className="text-xs text-gray-400">No facts in this crystal.</p>}
            </div>
          )}
          {tab === "ledger" && (
            <div className="max-w-3xl">
              {ledger.isError && <ErrorBanner title="Couldn't load ledger" message={String(ledger.error)} />}
              <LedgerTimeline entries={entries} />
            </div>
          )}
          {tab === "raw" && <JsonView value={d} />}
        </>
      )}
    </div>
  );
}

// -------------------------------------------------------------- shelves

const SHELF_FETCH_LIMIT = 200;

export function BankBrowser() {
  const { selectedCustomerId } = useSelectedCustomer();
  const [axis, setAxis] = useState<Axis>("domain");
  const [q, setQ] = useState("");
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const [openId, setOpenId] = useState<string | null>(null);
  const [view, setView] = useState<"shelves" | "graph">("shelves");

  const list = useQuery({
    queryKey: ["crystals", selectedCustomerId, 0, SHELF_FETCH_LIMIT],
    queryFn: () => api.listCrystals(selectedCustomerId!, { offset: 0, limit: SHELF_FETCH_LIMIT }),
    enabled: !!selectedCustomerId,
  });

  const items: CrystalSummary[] = list.data?.items ?? [];
  const shelves = useMemo(() => {
    const classified: Shelved[] = items.map((c) => ({ c, ...classify(c) }));
    const needle = q.trim().toLowerCase();
    const filtered = needle
      ? classified.filter((it) =>
          it.title.toLowerCase().includes(needle) ||
          it.breadcrumb.toLowerCase().includes(needle) ||
          (it.c.headline_key ?? "").toLowerCase().includes(needle) ||
          (it.c.summary_text ?? "").toLowerCase().includes(needle))
      : classified;
    const groups = new Map<string, Shelved[]>();
    for (const it of filtered) {
      const k = shelfKey(it, axis);
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k)!.push(it);
    }
    return [...groups.entries()].sort((a, b) => b[1].length - a[1].length);
  }, [items, axis, q]);

  if (openId) return <CrystalReader crystalId={openId} onBack={() => setOpenId(null)} />;

  return (
    <div>
      <div className="flex items-center gap-3 flex-wrap mb-4">
        <h2 className="text-base font-semibold text-gray-900">Crystal Bank</h2>
        <span className="text-xs text-gray-400">{list.data?.total ?? items.length} crystals</span>
        <div className="flex gap-1 ml-2">
          {(["shelves", "graph"] as const).map((v) => (
            <button key={v} onClick={() => setView(v)}
              className={cn("text-[11px] px-2.5 py-1 rounded border capitalize",
                view === v ? "bg-gray-800 border-gray-800 text-white"
                           : "border-gray-200 text-gray-500 hover:border-gray-300")}>
              {v === "graph" ? "Constellation" : "Shelves"}
            </button>
          ))}
        </div>
        <span className="flex-1" />
        <div className="flex gap-1">
          {AXES.map((a) => (
            <button key={a.id} onClick={() => setAxis(a.id)}
              className={cn("text-[11px] px-2.5 py-1 rounded border",
                axis === a.id ? "bg-indigo-50 border-indigo-300 text-indigo-700"
                              : "border-gray-200 text-gray-500 hover:border-gray-300")}>
              {a.label}
            </button>
          ))}
        </div>
        <div className="relative">
          <Search className="h-3.5 w-3.5 text-gray-300 absolute left-2 top-1/2 -translate-y-1/2" />
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search name or key"
            className="text-xs border border-gray-200 rounded pl-7 pr-2 py-1.5 outline-none focus:border-indigo-300 w-48" />
        </div>
      </div>

      {list.isLoading && <LoadingRows rows={6} />}
      {list.isError && <ErrorBanner title="Couldn't load crystals" message={String(list.error)} />}
      {!list.isLoading && !items.length && (
        <EmptyState title="No crystals yet"
          description="Upload documents or chat through the proxy and crystals will appear here." />
      )}

      {view === "graph" && !!items.length && (
        <Constellation
          customerId={selectedCustomerId!}
          crystals={shelves.flatMap(([group, members]) =>
            members.map((m) => ({
              id: m.c.id, title: m.title, group,
              factCount: m.c.fact_count ?? 1,
              tier: m.c.quality_tier ?? "untiered",
            })))}
          onOpen={setOpenId}
        />
      )}

      {view === "shelves" && shelves.map(([name, members]) => {
        const isCollapsed = collapsed[name] ?? false;
        const factTotal = members.reduce((n, m) => n + (m.c.fact_count ?? 0), 0);
        return (
          <div key={name} className="mb-5">
            <button onClick={() => setCollapsed({ ...collapsed, [name]: !isCollapsed })}
              className="w-full flex items-center gap-2 mb-2 group">
              {isCollapsed
                ? <ChevronRight className="h-3.5 w-3.5 text-gray-400" />
                : <ChevronDown className="h-3.5 w-3.5 text-gray-400" />}
              <span className="text-sm font-medium text-gray-800">{name}</span>
              <span className="text-[11px] text-gray-400">{members.length} crystal{members.length === 1 ? "" : "s"} · {factTotal} facts</span>
              <span className="flex-1 border-t border-gray-100 group-hover:border-gray-200" />
            </button>
            {!isCollapsed && (
              <div className="grid gap-2.5" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))" }}>
                {members.map(({ c, kind, breadcrumb, title }) => (
                  <button key={c.id} onClick={() => setOpenId(c.id)}
                    className="text-left bg-white border border-gray-200 rounded-xl px-3.5 py-3 hover:border-indigo-300 hover:bg-indigo-50/30 transition-colors">
                    <div className="text-[13px] font-medium text-gray-800 leading-snug">{title}</div>
                    {breadcrumb && <div className="text-[11px] text-gray-400 mt-0.5">{breadcrumb}</div>}
                    <div className="flex items-center gap-1.5 mt-2 text-[11px] flex-wrap">
                      <span className={cn("px-1.5 py-0.5 rounded ring-1 ring-inset", KIND_CLS[kind])}>{KIND_LABEL[kind]}</span>
                      <span className="text-gray-500">{c.fact_count} fact{c.fact_count === 1 ? "" : "s"}</span>
                      <QualityPill tier={c.quality_tier} />
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
