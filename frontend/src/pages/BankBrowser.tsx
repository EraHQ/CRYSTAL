// Crystal Bank — Console Refine PR 4 (docs/CONSOLE_UX_AUDIT.md 5.1).
//
// The list loads 100 crystals at a time behind "Load more"; the search
// term and the trust chips run on the server (GET .../crystals?q&tier),
// so a search reaches crystals past the loaded page instead of silently
// missing them. The reader is a Drawer at /bank/:crystalId over the same
// list, so a link opens it directly and closing keeps the filters and
// scroll. Trust is a labeled Segmented with one consequence line that
// says what the tier really does: tiers are a SIGNAL to the model, never
// a ranking weight or a recall filter (retrieval/tier_signal.py), so
// "Blocked" still reaches recall carrying "do not rely on it". No native
// dialogs anywhere; every failure lands in an ErrorBanner.
import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Archive, BookOpen, ChevronDown, ChevronRight, Database, FileText, Link2, MoreHorizontal, Pencil, Search,
  Settings as SettingsIcon, X,
} from "lucide-react";
import { api, errorMessage } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";
import {
  ConfirmInline, Drawer, EmptyState, ErrorBanner, FilterChips, IconButton, JsonView, Skeleton,
  PageHeader, Segmented, StatusPill, Toolbar, useToast,
} from "@/components/ui";
import type { StatusVariant } from "@/components/ui";
import type { CrystalDetail, CrystalSummary, FactSummary, LedgerEntry } from "@/lib/types";
import { cn, fmtDateTime, fmtRelative, truncate } from "@/lib/utils";
import { Constellation } from "@/components/bank/Constellation";

// ------------------------------------------------------------ classify

type Kind = "reflection" | "pattern" | "ingested" | "knowledge";

type Identity = Pick<CrystalSummary, "headline_key" | "crystal_type" | "headline_source_kind" | "build_method" | "source_path" | "summary_text">;

export function classify(c: Identity): { kind: Kind; breadcrumb: string; title: string } {
  const segs = (c.headline_key ?? "").split("|").map((s) => s.trim()).filter(Boolean);
  const first = segs[0] ?? "";

  let kind: Kind = "knowledge";
  if (c.crystal_type === "reflection" || first === "Reflections") kind = "reflection";
  else if (first === "General") kind = "pattern";
  else if (first === "Code" || c.headline_source_kind === "document_chunk" || c.build_method === "content_chunk")
    kind = "ingested";

  let title: string;
  let breadcrumb = "";
  // Gate D6: file crystals label by PATH — the headline fact's last
  // key segment is '<module>' for every code file.
  if (kind === "ingested" && c.source_path) {
    const parts = c.source_path.split("/");
    title = parts[parts.length - 1];
    breadcrumb = parts.slice(0, -1).join(" › ");
  } else if (segs.length > 0) {
    title = segs[segs.length - 1];
    breadcrumb = segs.slice(0, -1).join(" › ");
  } else {
    title = c.summary_text ? truncate(c.summary_text, 80) : c.crystal_type ?? "(untitled crystal)";
  }
  return { kind, breadcrumb, title };
}

// 5.1 item 4: the loud Kind badge becomes quiet provenance text.
const KIND_TEXT: Record<Kind, string> = {
  reflection: "Reflected", pattern: "Learned", ingested: "From a file", knowledge: "Learned",
};
const KIND_LABEL: Record<Kind, string> = {
  reflection: "Reflections", pattern: "Patterns", ingested: "Files", knowledge: "Knowledge",
};

// ---------------------------------------------------------------- trust

// Labels for the four QualityTier values (models/crystal.py). The
// consequence lines come from retrieval/tier_signal.py's legend and
// scan/tier_promotion.py: a tier changes what the model is TOLD, not
// what is recalled.
export const TRUST: Record<string, { label: string; variant: StatusVariant; line: string }> = {
  whitelist: {
    label: "Trusted", variant: "ok",
    line: "Your tools treat it as settled. It drops back to Normal if it stops being cited or gets contested.",
  },
  neutral: {
    label: "Normal", variant: "neutral",
    line: "Recalled normally. Your tools are told it is not yet strongly vetted.",
  },
  quarantine: {
    label: "Needs review", variant: "warn",
    line: "Recalled, but your tools are told to treat it with care and verify before relying on it.",
  },
  blacklist: {
    label: "Blocked", variant: "bad",
    line: "Still recalled, but flagged: your tools are told not to rely on it. Kept for history.",
  },
};
const TIER_ORDER = ["whitelist", "neutral", "quarantine", "blacklist"] as const;
type Tier = (typeof TIER_ORDER)[number];

function trustOf(tier: string | null | undefined) {
  return TRUST[tier ?? ""] ?? { label: tier || "Untiered", variant: "neutral" as StatusVariant, line: "" };
}

// Fact provenance (models/crystal.py SourceKind), in plain words; the
// raw kind rides in `title`.
const SOURCE_TEXT: Record<string, string> = {
  model_reasoning: "From a conversation",
  failed_reasoning: "From a failed attempt",
  web_search_result: "From a web search",
  code_execution_result: "From running code",
  document_chunk: "From a document, verbatim",
  document_extraction: "From a document",
  operator_stated: "Stated by you",
  agent_inferred: "Inferred by a tool",
};

// ------------------------------------------------------ shelf grouping

type Axis = "domain" | "source" | "kind";
const AXES: Array<{ value: Axis; label: string }> = [
  { value: "domain", label: "Domain" },
  { value: "source", label: "Source" },
  { value: "kind", label: "Kind" },
];

interface Shelved { c: CrystalSummary; kind: Kind; breadcrumb: string; title: string }

function shelfKey(item: Shelved, axis: Axis): string {
  // The leading key segment is the broad domain ("Python", "Docs");
  // grouping by the LAST segment gave every crystal a private shelf.
  const segs = (item.c.headline_key ?? "").split("|").map((s) => s.trim()).filter(Boolean);
  if (axis === "domain") return segs[0] || "Unsorted";
  if (axis === "source")
    return segs.length > 2 ? segs.slice(0, 2).join(" › ") : segs[0] || "Unsorted";
  return KIND_LABEL[item.kind];
}

const PAGE = 100;
const MAP_CAP = 80;

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

// ------------------------------------------------------------ fact card

function FactCard({ crystalId, fact }: { crystalId: string; fact: FactSummary }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(fact.claim_text);
  const [confirmRetire, setConfirmRetire] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const qc = useQueryClient();
  const toast = useToast();
  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ["crystal"] });
    qc.invalidateQueries({ queryKey: ["crystal-ledger", crystalId] });
    qc.invalidateQueries({ queryKey: ["crystals"] });
  };
  const supersede = useMutation({
    mutationFn: () => api.supersedeFact(crystalId, fact.id, draft.trim()),
    onSuccess: () => {
      setEditing(false); setError(null); invalidate();
      toast("Saved as a new version. The original stays in History.");
    },
    onError: (e) => setError(errorMessage(e, "Couldn't save this fact.")),
  });
  const retire = useMutation({
    mutationFn: () => api.retireFact(crystalId, fact.id),
    onSuccess: () => {
      setConfirmRetire(false); setError(null); invalidate();
      toast("Fact retired. Its text stays in History.");
    },
    onError: (e) => setError(errorMessage(e, "Couldn't retire this fact.")),
  });
  const hits = fact.hit_count ?? 0;
  const source = SOURCE_TEXT[fact.source_kind] ?? fact.source_kind;

  return (
    <div className="rounded-xl border border-line bg-surface px-3.5 py-3">
      {error && <ErrorBanner title="That didn't go through" message={error} className="mb-2" />}
      {editing ? (
        <div>
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            rows={Math.max(3, Math.ceil(draft.length / 70))}
            aria-label="Fact text"
            className="w-full resize-y rounded-md border border-line bg-inset p-2 text-[13px] text-fg outline-none focus:border-accent"
          />
          <p className="mt-1.5 text-[11.5px] text-fg-3">
            Saving replaces this fact with a new version. The original text stays in History.
          </p>
          <div className="mt-2 flex justify-end gap-2">
            <button type="button" onClick={() => { setEditing(false); setDraft(fact.claim_text); }}
              className="rounded-md border border-line px-2.5 py-1 text-[12.5px] text-fg-2 hover:bg-inset">Cancel</button>
            <button type="button" onClick={() => supersede.mutate()}
              disabled={!draft.trim() || draft.trim() === fact.claim_text || supersede.isPending}
              className="rounded-md bg-accent px-2.5 py-1 text-[12.5px] font-medium text-on-accent hover:bg-accent-hover disabled:opacity-40">
              {supersede.isPending ? "Saving…" : "Save as new version"}
            </button>
          </div>
        </div>
      ) : (
        <>
          {/* Full claim text — never clamped (zero-truncation rule). */}
          <p className="whitespace-pre-wrap break-words text-[13.5px] leading-relaxed text-fg">{fact.claim_text}</p>
          {fact.prompt_text && (
            <p className="mt-1.5 whitespace-pre-wrap break-words text-[11.5px] text-fg-3">{fact.prompt_text}</p>
          )}
          <div className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-[11.5px] text-fg-3">
            <span title={fact.source_kind}>{source}</span>
            <span aria-hidden>·</span>
            <span>Recalled {hits} time{hits === 1 ? "" : "s"}</span>
            <span className="flex-1" />
            {!confirmRetire && (
              <>
                <IconButton label="Edit" icon={Pencil} size="sm" onClick={() => setEditing(true)} />
                <IconButton label="Retire" icon={Archive} size="sm" onClick={() => setConfirmRetire(true)} />
              </>
            )}
          </div>
          <ConfirmInline
            open={confirmRetire}
            message="Retire this fact? It leaves recall for good; its text stays in History."
            confirmLabel="Retire"
            busy={retire.isPending}
            onCancel={() => setConfirmRetire(false)}
            onConfirm={() => retire.mutate()}
          />
        </>
      )}
    </div>
  );
}

// -------------------------------------------------------------- history

const OP_VARIANT: Record<string, StatusVariant> = { supersede: "info", retire: "warn" };

function HistoryTimeline({ entries }: { entries: LedgerEntry[] }) {
  if (!entries.length) {
    return (
      <p className="text-[12.5px] text-fg-3">
        No changes yet. Every edit or retirement lands here permanently — History is append-only.
      </p>
    );
  }
  return (
    <div className="space-y-4 border-l-2 border-line pl-4">
      {entries.map((e) => (
        <div key={e.id}>
          <div className="mb-1 flex flex-wrap items-center gap-2 text-[11.5px] text-fg-3">
            <StatusPill variant={OP_VARIANT[e.op] ?? "neutral"}>{e.op === "supersede" ? "Edited" : e.op === "retire" ? "Retired" : e.op}</StatusPill>
            <span>{e.actor}</span>
            <span>{e.created_at ? fmtDateTime(e.created_at) : ""}</span>
          </div>
          {e.before_text && (
            <p className="whitespace-pre-wrap break-words text-[12.5px] text-fg-3 line-through">{e.before_text}</p>
          )}
          {e.after_text && (
            <p className="whitespace-pre-wrap break-words text-[12.5px] text-fg">→ {e.after_text}</p>
          )}
        </div>
      ))}
    </div>
  );
}

// --------------------------------------------------------------- reader

function sourceCount(facts: FactSummary[]): number {
  const docs = new Set<string>();
  let loose = false;
  for (const f of facts) {
    if (f.source_doc_id) docs.add(f.source_doc_id);
    else loose = true;
  }
  return docs.size + (loose ? 1 : 0);
}

export function CrystalDrawer({ crystalId, onClose }: { crystalId: string; onClose: () => void }) {
  const { selectedCustomerId } = useSelectedCustomer();
  const qc = useQueryClient();
  const toast = useToast();
  const [tab, setTab] = useState<"facts" | "history" | "raw">("facts");
  const [menuOpen, setMenuOpen] = useState(false);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [pendingBlock, setPendingBlock] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  const detail = useQuery({
    queryKey: ["crystal", selectedCustomerId, crystalId],
    queryFn: () => api.getCrystal(selectedCustomerId!, crystalId),
    enabled: !!selectedCustomerId,
  });
  const ledger = useQuery({
    queryKey: ["crystal-ledger", crystalId],
    queryFn: () => api.getCrystalLedger(crystalId),
  });
  const setTier = useMutation({
    mutationFn: (tier: Tier) => api.setCrystalTier(crystalId, tier),
    onSuccess: () => {
      setActionError(null); setPendingBlock(false);
      qc.invalidateQueries({ queryKey: ["crystal", selectedCustomerId, crystalId] });
      qc.invalidateQueries({ queryKey: ["crystals"] });
    },
    onError: (e) => setActionError(errorMessage(e, "Couldn't change the trust level.")),
  });
  const remove = useMutation({
    mutationFn: () => api.deleteCrystal(crystalId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["crystals"] });
      toast("Crystal deleted.");
      onClose();
    },
    onError: (e) => { setConfirmingDelete(false); setActionError(errorMessage(e, "Couldn't delete this crystal.")); },
  });

  const d: CrystalDetail | undefined = detail.data;
  const identity = d ? classify(d) : null;
  const entries = ledger.data ?? [];
  const facts = d?.facts ?? [];
  const tier = d?.quality_tier ?? "neutral";
  const trust = trustOf(tier);

  const copyLink = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      toast("Link copied.");
    } catch {
      toast("Couldn't copy the link — copy it from the address bar.");
    }
  };

  const onTrust = (v: Tier) => {
    if (v === tier) return;
    setActionError(null);
    if (v === "blacklist") { setPendingBlock(true); return; }
    setPendingBlock(false);
    setTier.mutate(v);
  };

  const tabs = [
    ["facts", `Facts ${facts.length}`],
    ["history", `History ${entries.length}`],
    ["raw", "Raw"],
  ] as const;

  return (
    <Drawer open onClose={onClose} title={identity?.title ?? (detail.isLoading ? "Loading…" : "Crystal")}>
      <nav className="mb-3 text-[12.5px]">
        <button type="button" onClick={onClose} className="text-fg-2 hover:text-fg">‹ All crystals</button>
      </nav>
      {detail.isLoading && <Skeleton rows={4} />}
      {detail.isError && <ErrorBanner title="Couldn't load this crystal" error={detail.error} onRetry={() => detail.refetch()} />}
      {actionError && <ErrorBanner title="That didn't go through" message={actionError} className="mb-3" />}
      {d && identity && (
        <>
          <div className="flex items-start gap-2">
            <div className="min-w-0 flex-1">
              {identity.breadcrumb && <p className="truncate text-[11.5px] text-fg-3">{identity.breadcrumb}</p>}
              <p className="mt-0.5 text-[12.5px] text-fg-2">
                {d.fact_count} fact{d.fact_count === 1 ? "" : "s"} · learned from {sourceCount(facts)} source{sourceCount(facts) === 1 ? "" : "s"} · created {fmtDateTime(d.created_at)}
              </p>
            </div>
            <div className="relative flex shrink-0 items-center gap-1">
              <IconButton label="Copy link" icon={Link2} onClick={() => void copyLink()} />
              <IconButton label="More" icon={MoreHorizontal} aria-haspopup="menu" aria-expanded={menuOpen}
                onClick={() => setMenuOpen((o) => !o)} />
              {menuOpen && (
                <div role="menu" className="absolute right-0 top-full z-10 mt-1 min-w-[160px] rounded-lg border border-line bg-surface p-1 shadow-pop">
                  <button type="button" role="menuitem"
                    onClick={() => { setMenuOpen(false); setConfirmingDelete(true); }}
                    className="w-full rounded-md px-2.5 py-1.5 text-left text-[13px] text-bad hover:bg-bad-soft">
                    Delete crystal…
                  </button>
                </div>
              )}
            </div>
          </div>
          <ConfirmInline
            open={confirmingDelete}
            message={`Delete this crystal and all ${d.fact_count} of its facts?`}
            permanent
            confirmLabel="Delete crystal"
            busy={remove.isPending}
            onCancel={() => setConfirmingDelete(false)}
            onConfirm={() => remove.mutate()}
          />

          <section className="mt-4" aria-labelledby="trust-label">
            <div className="mb-1.5 flex items-center gap-2">
              <span id="trust-label" className="text-[12px] font-medium uppercase tracking-[0.08em] text-fg-3">Trust</span>
              <StatusPill variant={trust.variant} raw={tier}>{trust.label}</StatusPill>
            </div>
            <Segmented<Tier>
              label="Trust level" size="sm" className="flex-wrap"
              options={TIER_ORDER.map((t) => ({ value: t, label: TRUST[t].label }))}
              value={(TIER_ORDER as readonly string[]).includes(tier) ? (tier as Tier) : "neutral"}
              onChange={onTrust}
            />
            <p className="mt-1.5 text-[12px] text-fg-3">{pendingBlock ? TRUST.blacklist.line : trust.line}</p>
            <ConfirmInline
              open={pendingBlock}
              message="Block this crystal? Your tools will be told not to rely on it. It stays in the bank and in History."
              confirmLabel="Block"
              busy={setTier.isPending}
              onCancel={() => setPendingBlock(false)}
              onConfirm={() => setTier.mutate("blacklist")}
            />
          </section>

          <div role="tablist" aria-label="Crystal sections" className="mt-5 mb-3 flex gap-4 border-b border-line text-[12.5px]">
            {tabs.map(([id, label]) => (
              <button key={id} type="button" role="tab" aria-selected={tab === id} onClick={() => setTab(id)}
                className={cn("-mb-px border-b-2 pb-2",
                  tab === id ? "border-accent font-medium text-accent" : "border-transparent text-fg-3 hover:text-fg-2")}>
                {label}
              </button>
            ))}
          </div>

          {tab === "facts" && (
            <div className="space-y-2.5">
              {facts.map((f) => <FactCard key={f.id} crystalId={crystalId} fact={f} />)}
              {!facts.length && <p className="text-[12.5px] text-fg-3">No facts in this crystal.</p>}
            </div>
          )}
          {tab === "history" && (
            <div>
              {ledger.isError && <ErrorBanner title="Couldn't load History" error={ledger.error} onRetry={() => ledger.refetch()} />}
              <HistoryTimeline entries={entries} />
            </div>
          )}
          {tab === "raw" && <JsonView value={d} />}
        </>
      )}
    </Drawer>
  );
}

// ----------------------------------------------------------------- page

export function BankBrowser() {
  const { selectedCustomerId } = useSelectedCustomer();
  const { crystalId } = useParams<{ crystalId: string }>();
  const navigate = useNavigate();
  const [view, setView] = useState<"shelves" | "map">("shelves");
  const [axis, setAxis] = useState<Axis>("domain");
  const [q, setQ] = useState("");
  const term = useDebounced(q.trim(), 300);
  const [tier, setTierFilter] = useState<string | null>(null);
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  // Refine PR 5 (5.3 item 4): /?doc=<id>&from=<label> lists the crystals
  // born from one document ("View {n} crystals" on a Knowledge card).
  const [params, setParams] = useSearchParams();
  const docId = params.get("doc") || null;
  const docLabel = params.get("from") || null;

  const list = useInfiniteQuery({
    queryKey: ["crystals", selectedCustomerId, term, tier, docId],
    queryFn: ({ pageParam }) =>
      api.listCrystals(selectedCustomerId!, {
        offset: pageParam, limit: PAGE, q: term, tier: tier ?? undefined, documentId: docId ?? undefined,
      }),
    initialPageParam: 0,
    getNextPageParam: (last, pages) => {
      const loaded = pages.reduce((n, p) => n + p.items.length, 0);
      return loaded < last.total && last.items.length > 0 ? loaded : undefined;
    },
    enabled: !!selectedCustomerId,
  });

  const pages = list.data?.pages ?? [];
  const items: CrystalSummary[] = useMemo(() => pages.flatMap((p) => p.items), [pages]);
  const total = pages.length ? pages[pages.length - 1].total : 0;
  const bank = pages[0]?.bank;
  const filtered = !!term || !!tier || !!docId;

  const shelves = useMemo(() => {
    const classified: Shelved[] = items.map((c) => ({ c, ...classify(c) }));
    const groups = new Map<string, Shelved[]>();
    for (const it of classified) {
      const k = shelfKey(it, axis);
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k)!.push(it);
    }
    return [...groups.entries()].sort((a, b) => b[1].length - a[1].length);
  }, [items, axis]);

  // 5.1 item 8: the Map keeps the 80 most recalled crystals of what is
  // loaded (ties broken by fact count) so it never becomes a hairball.
  const mapCrystals = useMemo(() => {
    const classified = items.map((c) => ({ c, ...classify(c) }));
    return [...classified]
      .sort((a, b) => (b.c.recall_hits ?? 0) - (a.c.recall_hits ?? 0) || (b.c.fact_count ?? 0) - (a.c.fact_count ?? 0))
      .slice(0, MAP_CAP)
      .map((m) => ({
        id: m.c.id, title: m.title, group: shelfKey(m, axis),
        factCount: m.c.fact_count ?? 1, tier: m.c.quality_tier ?? "untiered",
        recallHits: m.c.recall_hits ?? 0,
      }));
  }, [items, axis]);

  const chipOptions = [
    { value: "all", label: "All", count: bank?.crystals },
    ...TIER_ORDER.map((t) => ({ value: t, label: TRUST[t].label, count: bank?.tier_counts?.[t] ?? 0, title: t })),
  ];

  if (!selectedCustomerId) {
    return (
      <>
        <PageHeader title="Crystal Bank" />
        <EmptyState icon={Database} title="No customer selected" help="Pick a customer above to browse their bank." />
      </>
    );
  }

  const meta = bank
    ? `${bank.crystals.toLocaleString()} crystal${bank.crystals === 1 ? "" : "s"} · ${bank.facts.toLocaleString()} fact${bank.facts === 1 ? "" : "s"}${bank.updated_at ? ` · updated ${fmtRelative(bank.updated_at)}` : ""}`
    : list.isLoading ? "Loading…" : undefined;

  return (
    <div>
      <PageHeader
        title="Crystal Bank"
        meta={meta}
        actions={
          <div title={`Map shows the ${MAP_CAP} most recalled crystals`}>
            <Segmented<"shelves" | "map"> label="View" value={view} onChange={setView}
              options={[{ value: "shelves", label: "Shelves" }, { value: "map", label: "Map" }]} />
          </div>
        }
        toolbar={
          <Toolbar
            search={
              <label className="relative block">
                <Search aria-hidden className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-fg-3" />
                <input
                  type="search" value={q} onChange={(e) => setQ(e.target.value)}
                  aria-label="Search crystals"
                  placeholder={`Search all ${(bank?.crystals ?? total).toLocaleString()} crystals by name, key or fact`}
                  className="w-full rounded-md border border-line bg-inset py-1.5 pl-8 pr-2 text-[13px] text-fg placeholder:text-fg-3 outline-none focus:border-accent"
                />
              </label>
            }>
            <FilterChips label="Trust" options={chipOptions} value={tier ?? "all"}
              onChange={(v) => setTierFilter(v === null || v === "all" ? null : v)} />
            <Segmented<Axis> label="Group by" size="sm" className="hidden min-[720px]:inline-flex"
              options={AXES} value={axis} onChange={setAxis} />
          </Toolbar>
        }
      />

      {docId && (
        <div className="mb-3 flex flex-wrap items-center gap-2 text-[12.5px] text-fg-2">
          <FileText aria-hidden className="h-3.5 w-3.5 text-fg-3" />
          <span>Crystals from <span className="font-medium text-fg">{docLabel ?? "one document"}</span></span>
          <IconButton label="Show the whole bank" icon={X} size="sm" onClick={() => setParams({})} />
        </div>
      )}
      {list.isLoading && <Skeleton rows={6} />}
      {list.isError && <ErrorBanner title="Couldn't load crystals" error={list.error} onRetry={() => list.refetch()} />}

      {!list.isLoading && !list.isError && !items.length && !filtered && (
        <EmptyState icon={Database} title="No crystals yet"
          help="Connect a tool or upload a document. Crystals appear here as your tools save memories."
          action={
            <div className="flex flex-wrap justify-center gap-2">
              <Link to="/settings" className="inline-flex items-center gap-1.5 rounded-md bg-accent px-3 py-1.5 text-[13px] font-medium text-on-accent hover:bg-accent-hover">
                <SettingsIcon aria-hidden className="h-3.5 w-3.5" /> Settings › Connections
              </Link>
              <Link to="/knowledge" className="inline-flex items-center gap-1.5 rounded-md border border-line px-3 py-1.5 text-[13px] font-medium text-fg-2 hover:bg-inset hover:text-fg">
                <BookOpen aria-hidden className="h-3.5 w-3.5" /> Knowledge
              </Link>
            </div>
          } />
      )}
      {!list.isLoading && !list.isError && !items.length && filtered && (
        <EmptyState icon={Search} title={docId ? "No crystals from this document yet" : "No crystals match"}
          help={docId ? "Crystals appear here once the document is approved and crystallized." : "Try another term or clear the trust filter."}
          action={
            <button type="button" onClick={() => { setQ(""); setTierFilter(null); setParams({}); }}
              className="rounded-md border border-line px-3 py-1.5 text-[13px] font-medium text-fg-2 hover:bg-inset hover:text-fg">
              Clear search and filters
            </button>
          } />
      )}

      {view === "map" && !!items.length && (
        <Constellation customerId={selectedCustomerId} crystals={mapCrystals} onOpen={(id) => navigate(`/bank/${id}`)} />
      )}

      {view === "shelves" && shelves.map(([name, members]) => {
        const isCollapsed = collapsed[name] ?? false;
        const factTotal = members.reduce((n, m) => n + (m.c.fact_count ?? 0), 0);
        return (
          <section key={name} className="mb-5">
            <button type="button" onClick={() => setCollapsed({ ...collapsed, [name]: !isCollapsed })}
              aria-expanded={!isCollapsed}
              className="group mb-2 flex w-full items-center gap-2 text-left">
              {isCollapsed
                ? <ChevronRight aria-hidden className="h-3.5 w-3.5 text-fg-3" />
                : <ChevronDown aria-hidden className="h-3.5 w-3.5 text-fg-3" />}
              <span className="text-[13.5px] font-medium text-fg">{name}</span>
              <span className="text-[11.5px] text-fg-3">{members.length} crystal{members.length === 1 ? "" : "s"} · {factTotal} facts</span>
              <span className="flex-1 border-t border-line group-hover:border-line-strong" />
            </button>
            {!isCollapsed && (
              <div className="grid gap-2.5" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))" }}>
                {members.map(({ c, kind, breadcrumb, title }) => {
                  const t = trustOf(c.quality_tier);
                  const when = fmtRelative(c.last_activity ?? c.created_at);
                  return (
                    <button key={c.id} type="button" onClick={() => navigate(`/bank/${c.id}`)}
                      className="rounded-xl border border-line bg-surface px-3.5 py-3 text-left transition-colors hover:border-accent/50 hover:bg-accent-soft/30">
                      <div className="flex items-start gap-2">
                        <div className="min-w-0 flex-1">
                          <div className="break-words text-[13.5px] font-medium leading-snug text-fg">{title}</div>
                          {breadcrumb && <div className="mt-0.5 truncate text-[11.5px] text-fg-3">{breadcrumb}</div>}
                        </div>
                        <StatusPill variant={t.variant} raw={c.quality_tier} className="shrink-0">{t.label}</StatusPill>
                      </div>
                      <div className="mt-2 text-[11.5px] text-fg-3">
                        {c.fact_count} fact{c.fact_count === 1 ? "" : "s"} · {KIND_TEXT[kind]}{when ? ` · ${when}` : ""}
                      </div>
                    </button>
                  );
                })}
              </div>
            )}
          </section>
        );
      })}

      {!!items.length && (
        <footer className="mt-2 flex flex-wrap items-center justify-between gap-2 text-[12.5px] text-fg-3">
          <span>Showing {items.length.toLocaleString()} of {total.toLocaleString()} crystal{total === 1 ? "" : "s"}.</span>
          {list.hasNextPage && (
            <button type="button" onClick={() => list.fetchNextPage()} disabled={list.isFetchingNextPage}
              className="rounded-md border border-line px-3 py-1.5 text-[13px] font-medium text-fg-2 hover:bg-inset hover:text-fg disabled:opacity-50">
              {list.isFetchingNextPage ? "Loading…" : `Load ${Math.min(PAGE, total - items.length)} more`}
            </button>
          )}
        </footer>
      )}

      {crystalId && <CrystalDrawer crystalId={crystalId} onClose={() => navigate("/")} />}
    </div>
  );
}
