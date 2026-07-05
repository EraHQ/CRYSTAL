import { Fragment, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Search } from "lucide-react";
import { api } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";
import { Disclosure, EmptyState, ErrorBanner, JsonView, LoadingRows, QualityPill, CrystalButton } from "@/components/ui";
import type { CrystalSummary } from "@/lib/types";
import { cn, fmtDateTime, truncate } from "@/lib/utils";

const PAGE_SIZE = 50;

type Kind = "reflection" | "pattern" | "ingested" | "knowledge";

// Classify a crystal from its representative sparse key + type metadata.
// The key's leading segment is the strongest signal: Reflections|… are
// lessons the agent earned from fail→fix, General|… are patterns it saved,
// Code|… (or a document_chunk/content_chunk crystal) is ingested material.
// Everything else is ordinary Q&A knowledge.
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

const isAgentMade = (k: Kind) => k === "reflection" || k === "pattern";

const KIND_STYLE: Record<Kind, { cls: string; label: string }> = {
  reflection: { cls: "bg-violet-50 text-violet-700 ring-violet-200/60", label: "Reflection" },
  pattern: { cls: "bg-brand-50 text-brand-700 ring-brand-200/60", label: "Pattern" },
  ingested: { cls: "bg-amber-50 text-amber-700 ring-amber-200/60", label: "Ingested" },
  knowledge: { cls: "bg-sky-50 text-sky-700 ring-sky-200/60", label: "Knowledge" },
};

function KindBadge({ kind }: { kind: Kind }) {
  const s = KIND_STYLE[kind];
  return (
    <span className={cn("inline-flex rounded-md px-2 py-0.5 text-[11px] font-medium ring-1 ring-inset", s.cls)}>
      {s.label}
    </span>
  );
}

type Filter = "all" | "agent" | "ingested" | "knowledge";

export function BankBrowser() {
  const { selectedCustomerId } = useSelectedCustomer();
  const [page, setPage] = useState(0);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<Filter>("all");
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const offset = page * PAGE_SIZE;

  const list = useQuery({ queryKey: ["crystals", selectedCustomerId, offset, PAGE_SIZE], queryFn: () => api.listCrystals(selectedCustomerId!, { offset, limit: PAGE_SIZE }), enabled: !!selectedCustomerId });
  const detail = useQuery({ queryKey: ["crystal", selectedCustomerId, expandedId], queryFn: () => api.getCrystal(selectedCustomerId!, expandedId!), enabled: !!selectedCustomerId && !!expandedId });

  const items = list.data?.items ?? [];
  const total = list.data?.total ?? 0;

  // Classify once per loaded page; counts power the filter pills.
  const classified = useMemo(() => items.map((c) => ({ c, ...classify(c) })), [items]);
  const counts = useMemo(() => {
    const out = { all: classified.length, agent: 0, ingested: 0, knowledge: 0 };
    for (const r of classified) {
      if (isAgentMade(r.kind)) out.agent++;
      else if (r.kind === "ingested") out.ingested++;
      else out.knowledge++;
    }
    return out;
  }, [classified]);

  if (!selectedCustomerId) return <EmptyState title="No customer selected" description="Select a customer to browse crystals." />;
  if (list.isError) return <ErrorBanner title="Couldn't load crystals" message={String(list.error)} />;

  const q = search.toLowerCase();
  const filtered = classified.filter(({ c, kind, title, breadcrumb }) => {
    if (filter === "agent" && !isAgentMade(kind)) return false;
    if (filter === "ingested" && kind !== "ingested") return false;
    if (filter === "knowledge" && kind !== "knowledge") return false;
    if (!q) return true;
    return (
      title.toLowerCase().includes(q) ||
      breadcrumb.toLowerCase().includes(q) ||
      (c.summary_text ?? "").toLowerCase().includes(q) ||
      (c.headline_key ?? "").toLowerCase().includes(q)
    );
  });
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  const PILLS: { id: Filter; label: string; n: number }[] = [
    { id: "all", label: "All", n: counts.all },
    { id: "agent", label: "Agent-made", n: counts.agent },
    { id: "ingested", label: "Ingested", n: counts.ingested },
    { id: "knowledge", label: "Knowledge", n: counts.knowledge },
  ];

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold text-gray-900">Crystal Bank</h1>
          <p className="text-sm text-gray-500">{total} crystal{total !== 1 ? "s" : ""} for <code className="font-mono text-brand-600 text-xs">{selectedCustomerId}</code></p>
        </div>
        <div className="relative">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
          <input type="search" value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search name or key…"
            className="rounded-lg border border-gray-200 bg-white py-1.5 pl-9 pr-3 text-sm focus:outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-500/20 w-64" />
        </div>
      </div>

      <div className="inline-flex rounded-lg border border-gray-200 bg-white p-0.5 text-sm">
        {PILLS.map((p) => (
          <button
            key={p.id}
            onClick={() => setFilter(p.id)}
            className={cn(
              "rounded-md px-3 py-1 transition-colors",
              filter === p.id ? "bg-brand-50 font-medium text-brand-700" : "text-gray-500 hover:text-gray-800"
            )}
          >
            {p.label} <span className="text-gray-400">{p.n}</span>
          </button>
        ))}
      </div>

      <div className="rounded-lg border border-gray-200 bg-white overflow-hidden">
        <table className="min-w-full divide-y divide-gray-100 text-sm">
          <thead className="bg-gray-50/60">
            <tr>
              <th className="px-4 py-2.5 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Name</th>
              <th className="px-4 py-2.5 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Kind</th>
              <th className="px-4 py-2.5 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Facts</th>
              <th className="px-4 py-2.5 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Quality</th>
              <th className="px-4 py-2.5 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Created</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-50">
            {list.isLoading && <LoadingRows rows={5} cols={5} />}
            {!list.isLoading && filtered.map(({ c, kind, breadcrumb, title }) => (
              <Fragment key={c.id}>
                <tr onClick={() => setExpandedId(expandedId === c.id ? null : c.id)} className="cursor-pointer hover:bg-gray-50 transition-colors">
                  <td className="px-4 py-2.5">
                    {breadcrumb && <div className="text-[11px] text-gray-400">{breadcrumb}</div>}
                    <div className="font-medium text-gray-800">{truncate(title, 70)}</div>
                    {c.headline_claim && <div className="text-[11px] text-gray-400">{truncate(c.headline_claim, 90)}</div>}
                  </td>
                  <td className="px-4 py-2.5"><KindBadge kind={kind} /></td>
                  <td className="px-4 py-2.5 text-gray-600">{c.fact_count}</td>
                  <td className="px-4 py-2.5"><QualityPill tier={c.quality_tier} /></td>
                  <td className="px-4 py-2.5 text-xs text-gray-400">{fmtDateTime(c.created_at)}</td>
                </tr>
                {expandedId === c.id && (
                  <tr className="bg-gray-50/50">
                    <td colSpan={5} className="px-4 py-4">
                      {detail.isLoading && <p className="text-sm text-gray-400">Loading…</p>}
                      {detail.isError && <p className="text-sm text-red-600">Error: {String(detail.error)}</p>}
                      {detail.data && (
                        <div className="space-y-3">
                          <div className="flex items-center gap-2 text-[11px] text-gray-400">
                            <span className="font-mono">{c.id}</span>
                            {c.crystal_type && <span className="rounded bg-gray-100 px-1.5 py-0.5 font-mono text-gray-600">{c.crystal_type}</span>}
                          </div>
                          <div className="text-sm"><p className="font-medium text-gray-700">Summary</p><p className="mt-1 text-gray-600 whitespace-pre-wrap">{detail.data.summary_text ?? "—"}</p></div>
                          {(detail.data.facts ?? []).length > 0 && (
                            <div className="text-sm">
                              <p className="font-medium text-gray-700">Facts ({detail.data.facts.length})</p>
                              <div className="mt-1 rounded-md border border-gray-200 bg-white overflow-x-auto">
                                <table className="min-w-full divide-y divide-gray-100 text-xs">
                                  <thead className="bg-gray-50/60">
                                    <tr>
                                      <th className="px-3 py-2 text-left font-medium text-gray-500">Key (wide › specific)</th>
                                      <th className="px-3 py-2 text-left font-medium text-gray-500">Type</th>
                                      <th className="px-3 py-2 text-left font-medium text-gray-500">Claim</th>
                                    </tr>
                                  </thead>
                                  <tbody className="divide-y divide-gray-50">
                                    {detail.data.facts.map((f) => (
                                      <tr key={f.id}>
                                        <td className="px-3 py-2 font-mono text-gray-700 whitespace-nowrap">
                                          {f.prompt_text
                                            ? f.prompt_text.split("|").map((s) => s.trim()).filter(Boolean).join(" › ")
                                            : <span className="text-gray-300">—</span>}
                                        </td>
                                        <td className="px-3 py-2"><span className="rounded bg-gray-100 px-1.5 py-0.5 font-mono text-[11px] text-gray-600">{f.pair_type}</span></td>
                                        <td className="px-3 py-2 text-gray-600">{truncate(f.claim_text ?? "", 90)}</td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                            </div>
                          )}
                          <Disclosure title="Full detail"><JsonView value={detail.data} /></Disclosure>
                        </div>
                      )}
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
            {!list.isLoading && filtered.length === 0 && (
              <tr><td colSpan={5} className="px-4 py-12 text-center text-sm text-gray-400">
                {search || filter !== "all" ? "No crystals match on this page." : "No crystals yet."}
              </td></tr>
            )}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between text-sm">
          <p className="text-gray-400">Page {page + 1}/{totalPages}</p>
          <div className="flex gap-2">
            <CrystalButton variant="secondary" size="sm" disabled={page === 0} onClick={() => setPage((p) => p - 1)}>Previous</CrystalButton>
            <CrystalButton variant="secondary" size="sm" disabled={page >= totalPages - 1} onClick={() => setPage((p) => p + 1)}>Next</CrystalButton>
          </div>
        </div>
      )}
    </div>
  );
}
