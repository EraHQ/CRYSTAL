import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Upload, Loader2, CheckCircle, AlertCircle, Trash2,
  Globe, FileText, Gem, Zap, FolderOpen,
  Cloud, Check, ChevronDown, ArrowLeft, Pencil, X, Save,
} from "lucide-react";
import { api, authedFetch } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";
import { EmptyState, CrystalButton, TypeBadge } from "@/components/ui";

interface DocumentItem {
  id: string; label: string; status: string; char_count: number;
  crystals_written: number; items_extracted: number;
  content_chunks_count?: number;
  created_at: string; crystallized_at?: string;
}

interface CrystallizeItem {
  key: string; sparse_key?: string; value: string; type: string; crystal_id?: string;
}

type Phase = "grid" | "crystallizing" | "review";

export function KnowledgeManager() {
  const { selectedCustomerId } = useSelectedCustomer();
  const queryClient = useQueryClient();
  const [phase, setPhase] = useState<Phase>("grid");
  const [reviewItems, setReviewItems] = useState<CrystallizeItem[]>([]);
  const [crystallizingDoc, setCrystallizingDoc] = useState<string | null>(null);
  const [crystallizeProgress, setCrystallizeProgress] = useState(0);
  const [reviewingDocId, setReviewingDocId] = useState<string | null>(null);
  const [reviewReadOnly, setReviewReadOnly] = useState(false);

  // K1 (2026-07-08): this page hung off the deprecated admin_key fetch
  // (410 Gone since no-plaintext, 2026-06-13) — every query below was
  // permanently disabled and the tab showed "No documents yet" for all
  // customers regardless of database contents. All calls now ride the
  // console session via require_customer_or_console.
  const documents = useQuery({
    queryKey: ["documents", selectedCustomerId],
    queryFn: () => api.listDocuments(selectedCustomerId!) as Promise<{ total: number; documents: DocumentItem[] }>,
    enabled: !!selectedCustomerId,
    refetchInterval: 15000,
  });

  const subscriptions = useQuery({
    queryKey: ["subscriptions", selectedCustomerId],
    queryFn: () => api.listSubscriptions(selectedCustomerId!) as Promise<{ general_crystal_types: string[] }>,
    enabled: !!selectedCustomerId,
  });

  const crystalTypes = useQuery({
    queryKey: ["crystal_types"],
    queryFn: async () => {
      const res = await fetch("/admin/api/crystal_types");
      if (!res.ok) return { items: [] };
      return res.json() as Promise<{ items: Array<{ id: string; display_name: string; scope: string }> }>;
    },
    enabled: !!selectedCustomerId,
  });

  const docs = documents.data?.documents || [];
  const hasPending = docs.some((d) => d.status === "pending");
  const subscribedTypes = subscriptions.data?.general_crystal_types || [];
  const generalTypes = (crystalTypes.data?.items || []).filter((t) => t.scope === "general");

  const handleToggleSubscription = async (typeId: string, isSubscribed: boolean) => {
    if (!selectedCustomerId) return;
    if (isSubscribed) {
      await api.unsubscribeCrystalType(selectedCustomerId, typeId);
    } else {
      await api.subscribeCrystalType(selectedCustomerId, typeId);
    }
    queryClient.invalidateQueries({ queryKey: ["subscriptions", selectedCustomerId] });
  };

  // Paths we never ingest from a folder pick: VCS internals, caches,
  // dependency trees, hidden files. Kept deliberately small — the
  // review queue is the real gate; this just avoids obvious noise.
  const isJunkPath = (p: string) =>
    p.split("/").some(
      (seg) =>
        seg.startsWith(".") ||
        seg === "__pycache__" ||
        seg === "node_modules" ||
        seg === "dist" ||
        seg === "build" ||
        seg.endsWith(".egg-info")
    );

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || !selectedCustomerId) return;
    // Gate D6 (authority grammar, ratified 2026-07-18, amends C1):
    // repo identity is repo://<authority>/<path>. The picked root is
    // the default authority — "the largest parent gives you the full
    // scope of a repo" — and naming it explicitly survives clone
    // renames and keeps two same-shaped repos from colliding. Imports
    // resolve only within their authority.
    const fileArr = Array.from(files);
    const firstRel = (fileArr[0] as File & { webkitRelativePath?: string })
      ?.webkitRelativePath;
    let authority: string | null = null;
    if (firstRel && firstRel.includes("/")) {
      const root = firstRel.split("/")[0];
      const named = window.prompt(
        "Source name for this upload (the repo identity — imports resolve within it):",
        root
      );
      if (named === null) { e.target.value = ""; return; }
      authority = (named.trim() || root);
    }
    for (const file of fileArr) {
      const rel = (file as File & { webkitRelativePath?: string })
        .webkitRelativePath;
      let label = rel && rel.length > 0 ? rel : file.name;
      if (rel && isJunkPath(rel)) continue;
      if (rel && authority) {
        const parts = rel.split("/");
        parts[0] = authority;
        label = parts.join("/");
      }
      const formData = new FormData();
      formData.append("file", file);
      formData.append("label", label);
      await api.uploadDocumentFile(selectedCustomerId, formData);
    }
    e.target.value = "";
    queryClient.invalidateQueries({ queryKey: ["documents", selectedCustomerId] });
  };

  const handleCrystallize = async () => {
    if (!selectedCustomerId) return;
    setPhase("crystallizing");
    setCrystallizeProgress(0);
    const pendingDocs = docs.filter((d) => d.status === "pending");
    const allItems: CrystallizeItem[] = [];
    for (let i = 0; i < pendingDocs.length; i++) {
      const doc = pendingDocs[i];
      setCrystallizingDoc(doc.label || "Untitled");
      setCrystallizeProgress(Math.round((i / pendingDocs.length) * 100));
      try {
        const data = await api.crystallizeDocument(selectedCustomerId, doc.id);
        allItems.push(...(data.items || []));
        setCrystallizeProgress(Math.round(((i + 1) / pendingDocs.length) * 100));
      } catch {}
    }
    setCrystallizeProgress(100);
    setReviewItems(allItems);
    queryClient.invalidateQueries({ queryKey: ["documents", selectedCustomerId] });
    setTimeout(() => setPhase("review"), 600);
  };

  const handleDelete = async (docId: string) => {
    if (!selectedCustomerId) return;
    await api.deleteDocument(selectedCustomerId, docId);
    queryClient.invalidateQueries({ queryKey: ["documents", selectedCustomerId] });
  };

  if (!selectedCustomerId) return <EmptyState title="No customer selected" description="Select a customer to manage knowledge." />;

  // ── Document Review Panel ──
  if (reviewingDocId) {
    return (
      <DocumentReviewPanel
        documentId={reviewingDocId}
        customerId={selectedCustomerId}
        readOnly={reviewReadOnly}
        onBack={() => {
          setReviewingDocId(null);
          queryClient.invalidateQueries({ queryKey: ["documents", selectedCustomerId] });
        }}
        onApprove={async (docId, items, chunks, includeChunks) => {
          // Fire and forget — navigate back immediately
          api.approveDocument(selectedCustomerId, docId, {
            items, content_chunks: chunks, include_chunks: includeChunks,
          }).then(() => {
            queryClient.invalidateQueries({ queryKey: ["documents", selectedCustomerId] });
          });
          setReviewingDocId(null);
          queryClient.invalidateQueries({ queryKey: ["documents", selectedCustomerId] });
        }}
        approving={false}
      />
    );
  }

  // ── Crystallizing ──
  if (phase === "crystallizing") {
    return (
      <div className="flex h-[calc(100vh-10rem)] items-center justify-center">
        <div className="text-center space-y-6">
          <div className="relative mx-auto w-20 h-20">
            <div className="absolute inset-0 rounded-2xl bg-brand-100 animate-ping" style={{ animationDuration: "2.5s" }} />
            <div className="absolute inset-2 rounded-xl bg-brand-50 animate-ping" style={{ animationDuration: "1.8s", animationDelay: "0.3s" }} />
            <div className="absolute inset-4 rounded-lg bg-gradient-to-br from-brand-500 to-violet-500 flex items-center justify-center shadow-glow">
              <Gem className="h-6 w-6 text-zinc-50 animate-pulse" />
            </div>
          </div>
          <div>
            <h2 className="text-base font-semibold text-gray-900">Crystallizing</h2>
            <p className="text-sm text-gray-500 mt-1">{crystallizingDoc || "Preparing..."}</p>
          </div>
          <div className="w-64 mx-auto">
            <div className="h-1.5 rounded-full bg-gray-100 overflow-hidden">
              <div className="h-full rounded-full bg-brand-600 transition-all duration-500" style={{ width: `${crystallizeProgress}%` }} />
            </div>
            <p className="text-xs text-gray-400 mt-2">{crystallizeProgress}%</p>
          </div>
        </div>
      </div>
    );
  }

  // ── Review ──
  if (phase === "review") {
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-base font-semibold text-gray-900">Documents ready for review</h1>
            <p className="text-sm text-gray-500">Open each document below and click Review to approve it into your bank.</p>
          </div>
          <CrystalButton onClick={() => { setPhase("grid"); setReviewItems([]); }}>
            <CheckCircle className="h-4 w-4" /> Done
          </CrystalButton>
        </div>
        <div className="rounded-lg border border-gray-200 bg-white overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-100 bg-gray-50/50">
                <th className="px-4 py-2.5 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-[90px]">Type</th>
                <th className="px-4 py-2.5 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Key</th>
                <th className="px-4 py-2.5 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Value</th>
              </tr>
            </thead>
            <tbody>
              {reviewItems.map((item, i) => (
                <tr key={i} className="border-b border-gray-50 hover:bg-gray-50/50">
                  <td className="px-4 py-2.5"><TypeBadge type={item.type} /></td>
                  <td className="px-4 py-2.5 font-mono text-xs text-gray-500 max-w-[200px] truncate">{item.key}</td>
                  <td className="px-4 py-2.5 text-gray-700 max-w-[400px]"><div className="line-clamp-2">{item.value}</div></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  // ── Grid ──
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold text-gray-900">Knowledge</h1>
          <p className="text-sm text-gray-500 mt-0.5">Upload documents and manage your knowledge base</p>
        </div>
        <div className="flex items-center gap-2">
          <label className="cursor-pointer inline-flex items-center gap-1.5 rounded-lg px-3.5 py-2 text-sm font-medium bg-white text-gray-700 shadow-card border border-gray-200 hover:bg-gray-50 hover:shadow-card-hover transition-all">
            <Upload className="h-4 w-4" /> Upload
            <input type="file" className="hidden" accept=".pdf,.docx,.txt,.md,.py,.pyi,.js,.jsx,.ts,.tsx,.go,.rs,.java,.rb,.c,.h,.cpp,.cs,.php,.swift,.kt,.sh" multiple onChange={handleFileUpload} />
          </label>
            <label className="cursor-pointer inline-flex items-center gap-1.5 rounded-lg px-3.5 py-2 text-sm font-medium border border-gray-200 text-gray-600 hover:bg-gray-50 transition-all" title="Upload a folder — relative paths become source identity (repo://path), which is what lets imports resolve into chains">
              <Upload className="h-4 w-4" /> Upload folder
              <input type="file" className="hidden" multiple onChange={handleFileUpload} {...({ webkitdirectory: "" } as any)} />
            </label>
          <CrystalButton onClick={handleCrystallize} disabled={!hasPending}>
            <Zap className="h-4 w-4" /> Crystallize Now
          </CrystalButton>
        </div>
      </div>

      {/* Documents */}
      {docs.length === 0 ? (
        <EmptyState
          title="No documents yet"
          description="Upload files or connect Google Drive. Documents are extracted and queued for your review before crystallization."
          action={
            <div className="flex items-center justify-center gap-2">
            <label className="cursor-pointer inline-flex items-center gap-1.5 rounded-lg px-3.5 py-2 text-sm font-medium bg-brand-600 text-zinc-50 shadow-glow hover:bg-brand-500 transition-all">
              <Upload className="h-4 w-4" /> Upload
              <input type="file" className="hidden" accept=".pdf,.docx,.txt,.md,.py,.pyi,.js,.jsx,.ts,.tsx,.go,.rs,.java,.rb,.c,.h,.cpp,.cs,.php,.swift,.kt,.sh" multiple onChange={handleFileUpload} />
            </label>
            <label className="cursor-pointer inline-flex items-center gap-1.5 rounded-lg px-3.5 py-2 text-sm font-medium border border-gray-200 text-gray-600 hover:bg-gray-50 transition-all" title="Upload a folder — relative paths become source identity (repo://path), which is what lets imports resolve into chains">
              <Upload className="h-4 w-4" /> Upload folder
              <input type="file" className="hidden" multiple onChange={handleFileUpload} {...({ webkitdirectory: "" } as any)} />
            </label>
            </div>
          }
        />
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {docs.map((doc) => (
            <div key={doc.id} className="group rounded-lg border border-gray-200 bg-white p-4 shadow-card hover:shadow-card-hover transition-shadow">
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-2 min-w-0">
                  <FileText className="h-4 w-4 text-gray-400 shrink-0" />
                  <span className="text-sm font-medium text-gray-900 truncate">{doc.label || "Untitled"}</span>
                </div>
                <button onClick={() => handleDelete(doc.id)}
                  className="p-1 rounded text-gray-300 hover:text-red-500 hover:bg-red-50 opacity-0 group-hover:opacity-100 transition-all" title="Delete">
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-400">{(doc.char_count / 1000).toFixed(1)}k chars</span>
                {doc.status === "pending" && (
                  <span className="inline-flex items-center gap-1 rounded-full bg-amber-50 px-2 py-0.5 text-[11px] font-medium text-amber-700">
                    <div className="h-1 w-1 rounded-full bg-amber-400 animate-pulse" /> Queued
                  </span>
                )}
                {doc.status === "extracting" && (
                  <span className="inline-flex items-center gap-1 rounded-full bg-brand-50 px-2 py-0.5 text-[11px] font-medium text-brand-700">
                    <Loader2 className="h-3 w-3 animate-spin" /> Extracting
                  </span>
                )}
                {doc.status === "review" && (
                  <CrystalButton size="sm" onClick={() => { setReviewReadOnly(false); setReviewingDocId(doc.id); }}>
                    <Pencil className="h-3 w-3" />
                    Review ({doc.items_extracted} items · {doc.content_chunks_count ?? 0} chunks)
                  </CrystalButton>
                )}
                {doc.status === "crystallizing" && (
                  <span className="inline-flex items-center gap-1 rounded-full bg-brand-50 px-2 py-0.5 text-[11px] font-medium text-brand-700">
                    <Loader2 className="h-3 w-3 animate-spin" /> Crystallizing
                  </span>
                )}
                {doc.status === "crystallized" && (
                  <span className="inline-flex items-center gap-2">
                    <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2 py-0.5 text-[11px] font-medium text-emerald-700">
                      <Gem className="h-3 w-3" /> {doc.crystals_written}
                    </span>
                    <button
                      onClick={() => { setReviewReadOnly(true); setReviewingDocId(doc.id); }}
                      className="text-[11px] text-gray-400 hover:text-gray-600 underline decoration-dotted"
                    >
                      View items
                    </button>
                  </span>
                )}
                {doc.status === "error" && (
                  <span className="inline-flex items-center gap-1 rounded-full bg-red-50 px-2 py-0.5 text-[11px] font-medium text-red-700">
                    <AlertCircle className="h-3 w-3" /> Error
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Crystal divider */}
      <div className="crystal-divider" />

      {/* Google Drive */}
      {/* K1 note: DriveConnector still speaks Key A and has been dead
          since no-plaintext (2026-06-13) like the rest of this page was.
          It renders its disconnected state here; the G1 connector
          refactor rebuilds this surface on console auth. */}
      <DriveConnector adminKey={undefined} onImportComplete={() => queryClient.invalidateQueries({ queryKey: ["documents", selectedCustomerId] })} />

      {/* Crystal divider */}
      <div className="crystal-divider" />

      {/* General Knowledge */}
      <div>
        <div className="flex items-center gap-2.5 mb-3">
          <Globe className="h-4 w-4 text-brand-500" />
          <h2 className="text-sm font-semibold text-gray-900">General Knowledge Banks</h2>
        </div>

        {generalTypes.length === 0 ? (
          <p className="text-sm text-gray-400">No general banks registered.</p>
        ) : (
          <div className="rounded-lg border border-gray-200 bg-white divide-y divide-gray-100">
            {generalTypes.map((ct) => {
              const active = subscribedTypes.includes(ct.id);
              return (
                <div key={ct.id} className="flex items-center justify-between px-4 py-3">
                  <div>
                    <div className="text-sm font-medium text-gray-900">{ct.display_name}</div>
                    <div className="text-xs text-gray-400 font-mono">{ct.id}</div>
                  </div>
                  <button onClick={() => handleToggleSubscription(ct.id, active)}
                    className={`toggle-track relative w-9 h-5 rounded-full ${active ? "bg-brand-600" : "bg-gray-200"}`}
                    data-active={active}>
                    <div className={`toggle-thumb absolute top-0.5 h-4 w-4 rounded-full bg-white shadow-sm ${active ? "translate-x-4" : "translate-x-0.5"}`} />
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Document Review Panel ──

interface ReviewChunk {
  index: number;
  label: string;
  text: string;
  char_count: number;
  // Gate D3: the describer's judgment (or the curator's) — editable
  // here; steers the chunk's retrieval embedding on every doc type,
  // and becomes a queryable purpose fact for code.
  description?: string | null;
  // Gate D4: the C2 screen's chunk-time findings — surfaced here so
  // the curator's approve is an informed verdict (option C).
  injection_hits?: string[] | null;
}

interface ReviewItem {
  key: string;
  sparse_key?: string;
  value: string;
  type: string;
}

function DocumentReviewPanel({
  documentId, customerId, onBack, onApprove, approving, readOnly = false,
}: {
  documentId: string;
  customerId: string;
  onBack: () => void;
  onApprove: (docId: string, items: ReviewItem[], chunks: ReviewChunk[], includeChunks: boolean) => void;
  approving: boolean;
  readOnly?: boolean;
}) {
  const [tab, setTab] = useState<"chunks" | "items">("items");
  const [includeChunks, setIncludeChunks] = useState(true);
  const [editingIdx, setEditingIdx] = useState<number | null>(null);
  const [editKey, setEditKey] = useState("");
  const [editValue, setEditValue] = useState("");
  const [editType, setEditType] = useState("");
  const [expandedChunk, setExpandedChunk] = useState<number | null>(null);

  const review = useQuery({
    queryKey: ["doc_review", documentId],
    queryFn: async () => {
      const res = await authedFetch(
        `/v1/documents/${documentId}/review?customer_id=${encodeURIComponent(customerId)}`
      );
      if (!res.ok) throw new Error("Failed to load review data");
      return res.json() as Promise<{
        document_id: string; label: string; status: string;
        detected_type: string; confirmed_type: string | null;
        content_chunks: ReviewChunk[]; extracted_items: ReviewItem[];
        char_count: number; items_extracted: number;
        comprehension?: {
          imports: { module: string; resolved_path: string | null }[];
        } | null;
      }>;
    },
  });

  const [localItems, setLocalItems] = useState<ReviewItem[] | null>(null);
  const [localChunks, setLocalChunks] = useState<ReviewChunk[] | null>(null);

  const items = localItems ?? review.data?.extracted_items ?? [];
  const chunks = localChunks ?? review.data?.content_chunks ?? [];

  if (review.data && localItems === null) {
    setLocalItems(review.data.extracted_items || []);
    setLocalChunks(review.data.content_chunks || []);
  }

  const handleDeleteItem = (idx: number) => {
    setLocalItems((prev) => (prev || []).filter((_, i) => i !== idx));
  };

  const handleStartEdit = (idx: number) => {
    const item = items[idx];
    setEditingIdx(idx);
    setEditKey(item.key);
    setEditValue(item.value);
    setEditType(item.type);
  };

  const handleSaveEdit = () => {
    if (editingIdx === null) return;
    setLocalItems((prev) => {
      const next = [...(prev || [])];
      next[editingIdx] = { ...next[editingIdx], key: editKey, value: editValue, type: editType };
      return next;
    });
    setEditingIdx(null);
  };

  const handleCancelEdit = () => setEditingIdx(null);

  const handleDeleteChunk = (idx: number) => {
    setLocalChunks((prev) => (prev || []).filter((_, i) => i !== idx));
  };

  const handleChunkDescription = (idx: number, value: string) => {
    setLocalChunks((prev) => {
      const next = [...(prev || [])];
      next[idx] = { ...next[idx], description: value };
      return next;
    });
  };

  if (review.isLoading) {
    return <div className="flex items-center justify-center h-64"><Loader2 className="h-6 w-6 animate-spin text-brand-500" /></div>;
  }

  if (review.isError) {
    return (
      <div className="space-y-4">
        <button onClick={onBack} className="flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700"><ArrowLeft className="h-4 w-4" /> Back</button>
        <div className="rounded-lg bg-red-50 border border-red-200 px-4 py-3 text-sm text-red-700">Failed to load review data.</div>
      </div>
    );
  }

  const data = review.data!;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button onClick={onBack} className="p-1.5 rounded-lg hover:bg-gray-100 text-gray-500 hover:text-gray-700">
            <ArrowLeft className="h-4 w-4" />
          </button>
          <div>
            <h2 className="text-lg font-semibold text-gray-900">{data.label}</h2>
            <div className="flex items-center gap-3 text-xs text-gray-400">
              <span>{(data.char_count / 1000).toFixed(1)}k chars</span>
              <span className="inline-flex items-center gap-1 rounded bg-brand-50 px-1.5 py-0.5 text-brand-700 font-medium">{data.detected_type}</span>
              <span>{items.length} knowledge items</span>
              <span>{chunks.length} content chunks</span>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {readOnly ? (
            <span className="text-xs text-gray-400">Crystallized — read-only view of what was approved</span>
          ) : (
            <>
              <label className="flex items-center gap-2 text-xs text-gray-600">
                <input type="checkbox" checked={includeChunks} onChange={(e) => setIncludeChunks(e.target.checked)} className="rounded border-gray-300" />
                Include content chunks
              </label>
              <CrystalButton onClick={() => onApprove(documentId, items, chunks, includeChunks)} disabled={approving}>
                {approving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />}
                Approve & Crystallize ({items.length + (includeChunks ? chunks.length : 0)})
              </CrystalButton>
            </>
          )}
        </div>
      </div>

      {/* Comprehension (Gate D3): mechanism on display — deterministic
          structure the ingest derived, no approve ceremony on facts a
          regex cannot get wrong. */}
      {(data.comprehension?.imports?.length ||
        (localChunks || []).some((c) => (c.injection_hits || []).length)) ? (
        <div className="rounded-lg border border-gray-200 bg-gray-50/50 px-4 py-3 space-y-2.5">
          {data.comprehension?.imports?.length ? (
            <div>
              <div className="text-xs font-medium text-gray-500 uppercase mb-2">Comprehension · imports</div>
              <div className="flex flex-wrap gap-1.5">
                {data.comprehension.imports.map((imp) => (
                  <span key={imp.module}
                    className={`inline-flex items-center gap-1 rounded px-2 py-0.5 text-xs font-mono ${imp.resolved_path ? "bg-brand-50 text-brand-700 border border-brand-200" : "bg-gray-100 text-gray-500 border border-gray-200"}`}>
                    {imp.module}
                    {imp.resolved_path && <span className="text-[10px] text-brand-500">→ {imp.resolved_path}</span>}
                  </span>
                ))}
              </div>
            </div>
          ) : null}
          {/* Gate D4 (option C): the screen's findings, shown BEFORE the
              verdict — approving with these visible is the un-quarantine;
              deleting the flagged chunk is the other honest exit. */}
          {(localChunks || []).some((c) => (c.injection_hits || []).length) ? (
            <div>
              <div className="text-xs font-medium text-amber-600 uppercase mb-2">⚠ Instruction-shaped text</div>
              <div className="space-y-1">
                {(localChunks || []).map((c, idx) =>
                  (c.injection_hits || []).length ? (
                    <div key={idx} className="text-xs text-amber-700">
                      #{c.index} {c.label} — <span className="font-mono">{(c.injection_hits || []).join(", ")}</span>
                    </div>
                  ) : null
                )}
              </div>
              <div className="text-[11px] text-gray-400 mt-1.5">Approving stores these chunks as reviewed (crystal born neutral); delete a chunk to exclude it.</div>
            </div>
          ) : null}
        </div>
      ) : null}

      {/* Tabs */}
      <div className="flex gap-1 border-b border-gray-200">
        <button onClick={() => setTab("items")}
          className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${tab === "items" ? "border-brand-600 text-brand-700" : "border-transparent text-gray-500 hover:text-gray-700"}`}>
          Knowledge Items ({items.length})
        </button>
        <button onClick={() => setTab("chunks")}
          className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${tab === "chunks" ? "border-brand-600 text-brand-700" : "border-transparent text-gray-500 hover:text-gray-700"}`}>
          Content Chunks ({chunks.length})
        </button>
      </div>

      {/* Knowledge Items */}
      {tab === "items" && (
        <div className="space-y-2">
          {items.length === 0 && (
            <div className="rounded-lg bg-gray-50 border border-gray-200 px-4 py-8 text-center text-sm text-gray-400">No knowledge items extracted.</div>
          )}
          {items.map((item, idx) => (
            <div key={idx} className="rounded-lg border border-gray-200 bg-white overflow-hidden">
              {editingIdx === idx ? (
                <div className="p-3 space-y-2">
                  <div className="flex items-center gap-2">
                    <select value={editType} onChange={(e) => setEditType(e.target.value)}
                      className="rounded border border-gray-200 px-2 py-1 text-xs text-gray-600 bg-white">
                      <option value="fact">fact</option><option value="entity">entity</option>
                      <option value="relationship">relationship</option><option value="process">process</option>
                      <option value="definition">definition</option><option value="qa">Q&A</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-xs text-gray-500 mb-0.5 block">Key (retrieval query)</label>
                    <input value={editKey} onChange={(e) => setEditKey(e.target.value)}
                      className="w-full rounded border border-gray-200 px-2.5 py-1.5 text-sm text-gray-700 focus:outline-none focus:border-brand-500 focus:ring-1 focus:ring-brand-500/20" />
                  </div>
                  <div>
                    <label className="text-xs text-gray-500 mb-0.5 block">Value (answer/content)</label>
                    <textarea value={editValue} onChange={(e) => setEditValue(e.target.value)} rows={3}
                      className="w-full rounded border border-gray-200 px-2.5 py-1.5 text-sm text-gray-700 focus:outline-none focus:border-brand-500 focus:ring-1 focus:ring-brand-500/20" />
                  </div>
                  <div className="flex gap-2 justify-end">
                    <CrystalButton size="sm" variant="ghost" onClick={handleCancelEdit}><X className="h-3 w-3" /> Cancel</CrystalButton>
                    <CrystalButton size="sm" onClick={handleSaveEdit}><Save className="h-3 w-3" /> Save</CrystalButton>
                  </div>
                </div>
              ) : (
                <div className="flex items-start gap-3 p-3 group">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="inline-flex items-center rounded bg-gray-100 px-1.5 py-0.5 text-[10px] font-medium text-gray-500 uppercase">{item.type}</span>
                      <span className="text-sm font-medium text-gray-800">{item.key}</span>
                    </div>
                    <p className="text-sm text-gray-600 leading-relaxed">{item.value}</p>
                  </div>
                  <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
                    <button onClick={() => handleStartEdit(idx)} className="p-1 rounded text-gray-300 hover:text-brand-600 hover:bg-brand-50" title="Edit"><Pencil className="h-3.5 w-3.5" /></button>
                    <button onClick={() => handleDeleteItem(idx)} className="p-1 rounded text-gray-300 hover:text-red-500 hover:bg-red-50" title="Delete"><Trash2 className="h-3.5 w-3.5" /></button>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Content Chunks */}
      {tab === "chunks" && (
        <div className="space-y-2">
          {chunks.length === 0 && (
            <div className="rounded-lg bg-gray-50 border border-gray-200 px-4 py-8 text-center text-sm text-gray-400">No content chunks generated.</div>
          )}
          {chunks.map((chunk, idx) => (
            <div key={idx} className="rounded-lg border border-gray-200 bg-white overflow-hidden">
              <div className="flex items-center justify-between px-3 py-2 bg-gray-50/50 border-b border-gray-100">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono text-gray-400">#{chunk.index}</span>
                  <span className="text-sm font-medium text-gray-700">{chunk.label}</span>
                  {(chunk.description ?? "").trim() && (
                    <span className="inline-flex items-center rounded bg-brand-50 px-1.5 py-0.5 text-[10px] font-medium text-brand-600" title={chunk.description ?? ""}>desc</span>
                  )}
                  {(chunk.injection_hits || []).length > 0 && (
                    <span className="inline-flex items-center rounded bg-amber-50 px-1.5 py-0.5 text-[10px] font-medium text-amber-600" title={(chunk.injection_hits || []).join(", ")}>⚠</span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-400">{(chunk.char_count / 1000).toFixed(1)}k</span>
                  <button onClick={() => setExpandedChunk(expandedChunk === idx ? null : idx)}
                    className="text-xs text-brand-600 hover:text-brand-800 font-medium">
                    {expandedChunk === idx ? "Collapse" : "Expand"}
                  </button>
                  <button onClick={() => handleDeleteChunk(idx)} className="p-1 rounded text-gray-300 hover:text-red-500 hover:bg-red-50" title="Remove">
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>
              {expandedChunk === idx && (
                <div className="px-3 py-2 space-y-2">
                  <div className="max-h-[300px] overflow-y-auto">
                    <pre className="text-xs text-gray-600 whitespace-pre-wrap font-mono leading-relaxed">{chunk.text}</pre>
                  </div>
                  <div>
                    <label className="text-xs text-gray-500 mb-0.5 block">
                      Description <span className="text-gray-400">(steers retrieval; becomes a purpose fact for code)</span>
                    </label>
                    <textarea value={chunk.description ?? ""} rows={2}
                      onChange={(e) => handleChunkDescription(idx, e.target.value)}
                      placeholder="What this chunk is for — empty means none"
                      className="w-full rounded border border-gray-200 px-2.5 py-1.5 text-xs text-gray-700 focus:outline-none focus:border-brand-500 focus:ring-1 focus:ring-brand-500/20" />
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}


// ── Drive Connector (real OAuth + folder picker + watched folders) ──

interface DriveConnection {
  id: string;
  email: string | null;
  status: string;
  last_synced_at: string | null;
  created_at: string;
}

interface WatchedFolder {
  id: string;
  folder_id: string;
  folder_name: string;
  folder_path: string | null;
  contains_phi: boolean;
  sync_interval_minutes: number;
  last_checked_at: string | null;
  last_file_count: number | null;
  status: string;
}

interface WatchedFile {
  id: string;
  file_id: string;
  file_name: string;
  mime_type: string | null;
  contains_phi: boolean;
  sync_interval_minutes: number;
  last_checked_at: string | null;
  last_modified_at: string | null;
  status: string;
}

interface BrowseItem {
  id: string;
  name: string;
  type?: string;       // "folder" for folders
  mimeType?: string;   // for files
  modifiedTime?: string;
}

function DriveConnector({ adminKey }: { adminKey?: string; onImportComplete: () => void }) {
  const [expanded, setExpanded] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [browsing, setBrowsing] = useState<string | null>(null); // connection_id when browsing
  const [browseParent, setBrowseParent] = useState("root");
  const [browseStack, setBrowseStack] = useState<Array<{ id: string; name: string }>>([]);
  const [browseItems, setBrowseItems] = useState<{ folders: BrowseItem[]; files: BrowseItem[] }>({ folders: [], files: [] });
  const [browseLoading, setBrowseLoading] = useState(false);
  const [addingFolder, setAddingFolder] = useState<string | null>(null); // folder_id being added
  const [phiFlag, setPhiFlag] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const queryClient = useQueryClient();

  // Fetch connections
  const connections = useQuery({
    queryKey: ["gdrive_connections", adminKey],
    queryFn: async () => {
      if (!adminKey) return { connections: [] };
      const res = await fetch("/v1/connectors/gdrive/connections", {
        headers: { Authorization: `Bearer ${adminKey}` },
      });
      if (!res.ok) return { connections: [] };
      return res.json() as Promise<{ connections: DriveConnection[] }>;
    },
    enabled: !!adminKey && expanded,
  });

  // Fetch watched folders for the first active connection
  const activeConn = (connections.data?.connections || []).find((c) => c.status === "active");

  const watchedFolders = useQuery({
    queryKey: ["gdrive_watched_folders", adminKey, activeConn?.id],
    queryFn: async () => {
      if (!adminKey || !activeConn) return { folders: [] };
      const res = await fetch(`/v1/connectors/gdrive/${activeConn.id}/folders`, {
        headers: { Authorization: `Bearer ${adminKey}` },
      });
      if (!res.ok) return { folders: [] };
      return res.json() as Promise<{ folders: WatchedFolder[] }>;
    },
    enabled: !!adminKey && !!activeConn,
  });

  // Fetch watched files
  const watchedFiles = useQuery({
    queryKey: ["gdrive_watched_files", adminKey, activeConn?.id],
    queryFn: async () => {
      if (!adminKey || !activeConn) return { files: [] };
      const res = await fetch(`/v1/connectors/gdrive/${activeConn.id}/watched-files`, {
        headers: { Authorization: `Bearer ${adminKey}` },
      });
      if (!res.ok) return { files: [] };
      return res.json() as Promise<{ files: WatchedFile[] }>;
    },
    enabled: !!adminKey && !!activeConn,
  });

  // Start OAuth flow
  const handleConnect = async () => {
    if (!adminKey) return;
    setConnecting(true);
    try {
      const res = await fetch("/v1/connectors/gdrive/auth-url", {
        headers: { Authorization: `Bearer ${adminKey}` },
      });
      if (!res.ok) {
        setMessage("Could not generate auth URL. Check Google OAuth credentials.");
        setConnecting(false);
        return;
      }
      const data = await res.json();
      // Redirect to Google consent
      window.location.href = data.auth_url;
    } catch {
      setMessage("Failed to connect.");
      setConnecting(false);
    }
  };

  // Disconnect
  const handleDisconnect = async (connId: string) => {
    if (!adminKey) return;
    await fetch(`/v1/connectors/gdrive/${connId}`, {
      method: "DELETE",
      headers: { Authorization: `Bearer ${adminKey}` },
    });
    queryClient.invalidateQueries({ queryKey: ["gdrive_connections"] });
    queryClient.invalidateQueries({ queryKey: ["gdrive_watched_folders"] });
  };

  // Browse folders
  const handleBrowse = async (connectionId: string, parentId: string = "root") => {
    if (!adminKey) return;
    setBrowsing(connectionId);
    setBrowseLoading(true);
    setBrowseParent(parentId);
    try {
      const res = await fetch(
        `/v1/connectors/gdrive/${connectionId}/browse?parent_id=${encodeURIComponent(parentId)}`,
        { headers: { Authorization: `Bearer ${adminKey}` } }
      );
      if (res.ok) {
        const data = await res.json();
        setBrowseItems({ folders: data.folders || [], files: data.files || [] });
      } else {
        setMessage("Could not browse Drive. Token may have expired.");
        setBrowsing(null);
      }
    } catch {
      setMessage("Browse failed.");
      setBrowsing(null);
    } finally {
      setBrowseLoading(false);
    }
  };

  // Navigate into a subfolder
  const navigateInto = (folderId: string, _folderName: string) => {
    if (!browsing) return;
    setBrowseStack((s) => [...s, { id: browseParent, name: browseStack.length === 0 ? "My Drive" : browseStack[browseStack.length - 1]?.name || "" }]);
    handleBrowse(browsing, folderId);
  };

  // Navigate back
  const navigateBack = () => {
    if (!browsing || browseStack.length === 0) return;
    const prev = browseStack[browseStack.length - 1];
    setBrowseStack((s) => s.slice(0, -1));
    handleBrowse(browsing, prev.id);
  };

  // Add a watched folder
  const handleWatchFolder = async (folderId: string, folderName: string) => {
    if (!adminKey || !browsing) return;
    setAddingFolder(folderId);
    try {
      const path = browseStack.map((s) => s.name).join("/") + "/" + folderName;
      await fetch(`/v1/connectors/gdrive/${browsing}/folders`, {
        method: "POST",
        headers: { Authorization: `Bearer ${adminKey}`, "Content-Type": "application/json" },
        body: JSON.stringify({
          folder_id: folderId,
          folder_name: folderName,
          folder_path: path,
          contains_phi: phiFlag,
          sync_interval_minutes: 60,
        }),
      });
      queryClient.invalidateQueries({ queryKey: ["gdrive_watched_folders"] });
      setMessage(`Watching "${folderName}" — sync worker will check every hour`);
    } catch {
      setMessage("Failed to add folder.");
    } finally {
      setAddingFolder(null);
    }
  };

  // Remove a watched folder
  const handleUnwatch = async (watchId: string) => {
    if (!adminKey) return;
    await fetch(`/v1/connectors/gdrive/folders/${watchId}`, {
      method: "DELETE",
      headers: { Authorization: `Bearer ${adminKey}` },
    });
    queryClient.invalidateQueries({ queryKey: ["gdrive_watched_folders"] });
  };

  // Watch a specific file
  const handleWatchFile = async (fileId: string, fileName: string, mimeType: string) => {
    if (!adminKey || !browsing) return;
    setAddingFolder(fileId); // reuse state for loading indicator
    try {
      await fetch(`/v1/connectors/gdrive/${browsing}/files`, {
        method: "POST",
        headers: { Authorization: `Bearer ${adminKey}`, "Content-Type": "application/json" },
        body: JSON.stringify({
          file_id: fileId,
          file_name: fileName,
          mime_type: mimeType,
          contains_phi: phiFlag,
          sync_interval_minutes: 60,
        }),
      });
      queryClient.invalidateQueries({ queryKey: ["gdrive_watched_files"] });
      setMessage(`Watching "${fileName}" for changes`);
    } catch {
      setMessage("Failed to watch file.");
    } finally {
      setAddingFolder(null);
    }
  };

  // Remove a watched file
  const handleUnwatchFile = async (watchId: string) => {
    if (!adminKey) return;
    await fetch(`/v1/connectors/gdrive/watched-files/${watchId}`, {
      method: "DELETE",
      headers: { Authorization: `Bearer ${adminKey}` },
    });
    queryClient.invalidateQueries({ queryKey: ["gdrive_watched_files"] });
  };

  const conns = connections.data?.connections || [];
  const watched = watchedFolders.data?.folders || [];
  const watchedF = watchedFiles.data?.files || [];
  const alreadyWatching = new Set([
    ...watched.map((w) => w.folder_id),
    ...watchedF.map((w) => w.file_id),
  ]);

  return (
    <div>
      <button onClick={() => setExpanded(!expanded)} className="flex items-center justify-between w-full text-left group">
        <div className="flex items-center gap-2.5">
          <Cloud className="h-4 w-4 text-brand-500" />
          <h2 className="text-sm font-semibold text-gray-900">Google Drive</h2>
          {activeConn && (
            <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2 py-0.5 text-[11px] font-medium text-emerald-700">
              <CheckCircle className="h-3 w-3" /> Connected
            </span>
          )}
        </div>
        <ChevronDown className={`h-4 w-4 text-gray-400 transition-transform ${expanded ? "rotate-180" : ""}`} />
      </button>

      {expanded && (
        <div className="mt-3 space-y-4">

          {/* No connections — show connect button */}
          {conns.length === 0 && (
            <div className="rounded-lg border border-dashed border-gray-300 bg-white p-6 text-center">
              <Cloud className="h-8 w-8 text-gray-300 mx-auto mb-3" />
              <p className="text-sm text-gray-600 mb-1">Connect your Google Drive</p>
              <p className="text-xs text-gray-400 mb-4">The system will monitor selected folders and automatically crystallize new or updated documents.</p>
              <CrystalButton onClick={handleConnect} disabled={connecting}>
                {connecting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Cloud className="h-4 w-4" />}
                Connect Google Drive
              </CrystalButton>
            </div>
          )}

          {/* Active connection */}
          {activeConn && (
            <div className="rounded-lg border border-gray-200 bg-white">
              <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100">
                <div className="flex items-center gap-3">
                  <div className="h-8 w-8 rounded-lg bg-blue-50 flex items-center justify-center">
                    <Cloud className="h-4 w-4 text-blue-600" />
                  </div>
                  <div>
                    <div className="text-sm font-medium text-gray-900">{activeConn.email || "Google Drive"}</div>
                    <div className="text-xs text-gray-400">
                      {activeConn.last_synced_at ? `Last synced ${new Date(activeConn.last_synced_at).toLocaleString()}` : "Never synced"}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <CrystalButton size="sm" variant="secondary" onClick={() => handleBrowse(activeConn.id)}>
                    <FolderOpen className="h-3.5 w-3.5" /> Browse
                  </CrystalButton>
                  <CrystalButton size="sm" variant="ghost" onClick={() => handleDisconnect(activeConn.id)}>
                    <Trash2 className="h-3.5 w-3.5" /> Disconnect
                  </CrystalButton>
                </div>
              </div>

              {/* Watched folders */}
              {watched.length > 0 && (
                <div className="divide-y divide-gray-50">
                  {watched.map((wf) => (
                    <div key={wf.id} className="flex items-center justify-between px-4 py-2.5">
                      <div className="flex items-center gap-2.5 min-w-0">
                        <FolderOpen className="h-4 w-4 text-amber-500 shrink-0" />
                        <div className="min-w-0">
                          <div className="text-sm text-gray-700 truncate">{wf.folder_name}</div>
                          <div className="text-xs text-gray-400 flex items-center gap-2">
                            {wf.folder_path && <span className="truncate max-w-[200px]">{wf.folder_path}</span>}
                            {wf.contains_phi && (
                              <span className="inline-flex items-center rounded bg-red-50 px-1.5 py-0.5 text-[10px] font-semibold text-red-600 uppercase">PHI</span>
                            )}
                            {wf.last_file_count != null && <span>{wf.last_file_count} files</span>}
                            <span>every {wf.sync_interval_minutes}m</span>
                          </div>
                        </div>
                      </div>
                      <button onClick={() => handleUnwatch(wf.id)} className="p-1 rounded text-gray-300 hover:text-red-500 hover:bg-red-50" title="Stop watching">
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {/* Watched files */}
              {watchedF.length > 0 && (
                <div className="divide-y divide-gray-50">
                  {watchedF.map((wf) => (
                    <div key={wf.id} className="flex items-center justify-between px-4 py-2.5">
                      <div className="flex items-center gap-2.5 min-w-0">
                        <FileText className="h-4 w-4 text-brand-500 shrink-0" />
                        <div className="min-w-0">
                          <div className="text-sm text-gray-700 truncate">{wf.file_name}</div>
                          <div className="text-xs text-gray-400 flex items-center gap-2">
                            {wf.contains_phi && (
                              <span className="inline-flex items-center rounded bg-red-50 px-1.5 py-0.5 text-[10px] font-semibold text-red-600 uppercase">PHI</span>
                            )}
                            <span>every {wf.sync_interval_minutes}m</span>
                            {wf.last_checked_at && <span>checked {new Date(wf.last_checked_at).toLocaleString()}</span>}
                          </div>
                        </div>
                      </div>
                      <button onClick={() => handleUnwatchFile(wf.id)} className="p-1 rounded text-gray-300 hover:text-red-500 hover:bg-red-50" title="Stop watching">
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {watched.length === 0 && watchedF.length === 0 && !browsing && (
                <div className="px-4 py-4 text-center">
                  <p className="text-xs text-gray-400">No folders being watched. Click Browse to select folders.</p>
                </div>
              )}
            </div>
          )}

          {/* Folder browser modal */}
          {browsing && (
            <div className="rounded-lg border border-brand-200 bg-white overflow-hidden">
              <div className="flex items-center justify-between px-4 py-2.5 bg-brand-50/50 border-b border-brand-100">
                <div className="flex items-center gap-2">
                  {browseStack.length > 0 && (
                    <button onClick={navigateBack} className="p-1 rounded hover:bg-brand-100 text-brand-600">
                      <ChevronDown className="h-4 w-4 rotate-90" />
                    </button>
                  )}
                  <span className="text-sm font-medium text-brand-800">
                    {browseStack.length === 0 ? "My Drive" : browseStack[browseStack.length - 1]?.name || "My Drive"} /
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <label className="flex items-center gap-1.5 text-xs text-gray-600">
                    <input type="checkbox" checked={phiFlag} onChange={(e) => setPhiFlag(e.target.checked)} className="rounded border-gray-300" />
                    Contains PHI
                  </label>
                  <button onClick={() => { setBrowsing(null); setBrowseStack([]); }} className="text-xs text-gray-400 hover:text-gray-600">Close</button>
                </div>
              </div>

              {browseLoading ? (
                <div className="px-4 py-8 text-center"><Loader2 className="h-5 w-5 animate-spin text-brand-400 mx-auto" /></div>
              ) : (
                <div className="divide-y divide-gray-50 max-h-[350px] overflow-y-auto">
                  {browseItems.folders.map((f) => (
                    <div key={f.id} className="flex items-center justify-between px-4 py-2.5 hover:bg-gray-50 transition-colors">
                      <button onClick={() => navigateInto(f.id, f.name)} className="flex items-center gap-2.5 min-w-0 text-left flex-1">
                        <FolderOpen className="h-4 w-4 text-amber-500 shrink-0" />
                        <span className="text-sm text-gray-700 truncate">{f.name}</span>
                      </button>
                      {alreadyWatching.has(f.id) ? (
                        <span className="text-xs text-emerald-600 font-medium flex items-center gap-1"><CheckCircle className="h-3 w-3" /> Watching</span>
                      ) : (
                        <CrystalButton size="sm" variant="secondary" onClick={() => handleWatchFolder(f.id, f.name)} disabled={addingFolder === f.id}>
                          {addingFolder === f.id ? <Loader2 className="h-3 w-3 animate-spin" /> : <Check className="h-3 w-3" />}
                          Watch
                        </CrystalButton>
                      )}
                    </div>
                  ))}
                  {browseItems.files.map((f) => (
                    <div key={f.id} className="flex items-center justify-between px-4 py-2.5 hover:bg-gray-50 transition-colors">
                      <div className="flex items-center gap-2.5 min-w-0 flex-1">
                        <FileText className="h-4 w-4 text-gray-400 shrink-0" />
                        <span className="text-sm text-gray-700 truncate">{f.name}</span>
                        <span className="text-xs text-gray-400 ml-auto mr-3">{f.mimeType?.split(".").pop()?.replace("apps.", "")}</span>
                      </div>
                      {alreadyWatching.has(f.id) ? (
                        <span className="text-xs text-emerald-600 font-medium flex items-center gap-1"><CheckCircle className="h-3 w-3" /> Watching</span>
                      ) : (
                        <CrystalButton size="sm" variant="secondary" onClick={() => handleWatchFile(f.id, f.name, f.mimeType || "")} disabled={addingFolder === f.id}>
                          {addingFolder === f.id ? <Loader2 className="h-3 w-3 animate-spin" /> : <Check className="h-3 w-3" />}
                          Watch
                        </CrystalButton>
                      )}
                    </div>
                  ))}
                  {browseItems.folders.length === 0 && browseItems.files.length === 0 && (
                    <div className="px-4 py-6 text-center text-xs text-gray-400">This folder is empty</div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Error connection */}
          {conns.some((c) => c.status === "auth_error") && (
            <div className="rounded-lg bg-amber-50 border border-amber-200 px-3 py-2">
              <p className="text-xs text-amber-700">Drive connection needs re-authorization. <button onClick={handleConnect} className="underline font-medium">Reconnect</button></p>
            </div>
          )}

          {/* Message */}
          {message && (
            <div className="rounded-lg bg-gray-50 border border-gray-200 px-3 py-2 flex items-center justify-between">
              <p className="text-xs text-gray-600">{message}</p>
              <button onClick={() => setMessage(null)} className="text-xs text-gray-400 hover:text-gray-600 ml-2">×</button>
            </div>
          )}
        </div>
      )}
      <WatchedSourcesPanel />
      <DataShapesPanel />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Watched Sources (Gate M slice 5): register a repo once, the bank
// feeds itself. Auto watches crystallize born-quarantine (the tier
// system reviews unattended ingest); gated watches queue for review.
// ---------------------------------------------------------------------------

interface Watch {
  id: string; scheme: string; source_name: string;
  config: { repo?: string; branch?: string };
  cadence_minutes: number; review_mode: string; status: string;
  has_token: boolean; last_state: { head?: string } | null;
  last_checked_at: string | null; last_error: string | null;
  sync_state?: string; inflight?: number;
}

interface WatchActivity {
  event_type: string; label: string;
  payload: Record<string, unknown> | null; created_at: string | null;
}

function relTime(iso: string | null): string {
  if (!iso) return "never";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

const STATE_STYLES: Record<string, { cls: string; label: string; pulse?: boolean }> = {
  syncing:   { cls: "bg-brand-50 text-brand-600 border border-brand-200", label: "Syncing", pulse: true },
  synced:    { cls: "bg-emerald-50 text-emerald-600 border border-emerald-200", label: "Synced" },
  attention: { cls: "bg-red-50 text-red-600 border border-red-200", label: "Attention" },
  waiting:   { cls: "bg-gray-50 text-gray-500 border border-gray-200", label: "Waiting for first sync" },
  paused:    { cls: "bg-gray-100 text-gray-500 border border-gray-200", label: "Paused" },
};

function WatchStateChip({ w }: { w: Watch }) {
  const st = STATE_STYLES[w.sync_state ?? "waiting"] ?? STATE_STYLES.waiting;
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[10px] font-medium ${st.cls}`}>
      {st.pulse && (
        <span className="relative flex h-1.5 w-1.5">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-brand-400 opacity-75" />
          <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-brand-500" />
        </span>
      )}
      {st.label}{w.sync_state === "syncing" && w.inflight ? ` · ${w.inflight} in flight` : ""}
    </span>
  );
}

function WatchActivityDrawer({ watchId, customerId }: { watchId: string; customerId: string }) {
  const activity = useQuery({
    queryKey: ["watch-activity", watchId],
    queryFn: async (): Promise<{ activity: WatchActivity[] }> => {
      const res = await authedFetch(
        `/admin/api/watches/${encodeURIComponent(watchId)}/activity?customer_id=${encodeURIComponent(customerId)}`
      );
      return res.json();
    },
    refetchInterval: 15000,
  });
  const rows = activity.data?.activity ?? [];
  if (activity.isLoading) return <p className="text-[11px] text-gray-400 px-3 py-2">Loading activity…</p>;
  if (rows.length === 0) return <p className="text-[11px] text-gray-400 px-3 py-2">No activity yet.</p>;
  return (
    <div className="max-h-56 overflow-y-auto border-t border-gray-100 mt-1">
      {rows.map((a, i) => (
        <div key={`${a.created_at ?? ""}-${i}`} className="flex items-center gap-2 px-3 py-1 text-[11px]">
          <span className={`h-1.5 w-1.5 rounded-full ${
            a.event_type === "cycle_completed" ? "bg-emerald-400"
            : a.event_type === "error" ? "bg-red-400"
            : a.event_type === "file_retired" ? "bg-amber-400"
            : a.event_type === "sync_started" ? "bg-gray-300"
            : "bg-brand-400"}`} />
          <span className="text-gray-400 whitespace-nowrap">{a.event_type.replace(/_/g, " ")}</span>
          <span className="font-mono text-gray-600 truncate">{a.label}</span>
          <span className="ml-auto text-gray-300 whitespace-nowrap">{relTime(a.created_at)}</span>
        </div>
      ))}
    </div>
  );
}

export function WatchedSourcesPanel() {
  const { selectedCustomerId } = useSelectedCustomer();
  const queryClient = useQueryClient();
  const [adding, setAdding] = useState(false);
  const [openActivity, setOpenActivity] = useState<string | null>(null);
  const [form, setForm] = useState({
    source_name: "", repo: "", branch: "master",
    review_mode: "auto", cadence_minutes: 15, token: "",
  });

  const watches = useQuery({
    queryKey: ["watches", selectedCustomerId],
    queryFn: async (): Promise<{ watches: Watch[] }> => {
      const res = await authedFetch(
        `/admin/api/watches?customer_id=${encodeURIComponent(selectedCustomerId!)}`
      );
      return res.json();
    },
    enabled: !!selectedCustomerId,
    // Live chip: keep polling while anything is mid-sync.
    refetchInterval: (q) =>
      (q.state.data?.watches ?? []).some((w) => w.sync_state === "syncing")
        ? 10000 : 60000,
  });

  const refresh = () =>
    queryClient.invalidateQueries({ queryKey: ["watches", selectedCustomerId] });

  const act = async (id: string, body: object, method = "PATCH") => {
    const res = await authedFetch(
      `/admin/api/watches/${encodeURIComponent(id)}?customer_id=${encodeURIComponent(selectedCustomerId!)}`,
      { method, headers: { "Content-Type": "application/json" },
        body: method === "DELETE" ? undefined : JSON.stringify(body) }
    );
    if (!res.ok) { window.alert(`Watch action failed (${res.status})`); return; }
    refresh();
  };

  const create = async () => {
    if (!form.source_name.trim() || !form.repo.trim()) {
      window.alert("Source name and repo are required."); return;
    }
    const res = await authedFetch(
      `/admin/api/watches?customer_id=${encodeURIComponent(selectedCustomerId!)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scheme: "git",
          source_name: form.source_name.trim(),
          config: { repo: form.repo.trim(), branch: form.branch.trim() || "master" },
          review_mode: form.review_mode,
          cadence_minutes: form.cadence_minutes,
          token: form.token || undefined,
        }),
      }
    );
    if (!res.ok) {
      const detail = await res.text();
      window.alert(`Watch creation failed (${res.status}): ${detail.slice(0, 200)}`);
      return;
    }
    setForm({ source_name: "", repo: "", branch: "master",
              review_mode: "auto", cadence_minutes: 15, token: "" });
    setAdding(false);
    refresh();
  };

  const rows = watches.data?.watches ?? [];

  return (
    <div className="mt-8 rounded-xl border border-gray-200 bg-white shadow-card p-5">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="text-sm font-semibold text-gray-800">Watched Sources</h3>
          <p className="text-xs text-gray-400">The bank keeps itself current — a push becomes crystals.</p>
        </div>
        <button onClick={() => setAdding((v) => !v)}
          className="rounded-lg px-3 py-1.5 text-xs font-medium border border-gray-200 text-gray-600 hover:bg-gray-50">
          {adding ? "Cancel" : "Add watch"}
        </button>
      </div>

      {adding && (
        <div className="mb-4 grid grid-cols-2 gap-2 rounded-lg border border-gray-100 bg-gray-50/50 p-3">
          <input placeholder="Source name (the authority, e.g. crystal-cache-v2)"
            value={form.source_name}
            onChange={(e) => setForm({ ...form, source_name: e.target.value })}
            className="col-span-2 rounded border border-gray-200 px-2.5 py-1.5 text-xs" />
          <input placeholder="Repo (owner/name or GitHub URL)"
            value={form.repo}
            onChange={(e) => setForm({ ...form, repo: e.target.value })}
            className="col-span-2 rounded border border-gray-200 px-2.5 py-1.5 text-xs" />
          <input placeholder="Branch" value={form.branch}
            onChange={(e) => setForm({ ...form, branch: e.target.value })}
            className="rounded border border-gray-200 px-2.5 py-1.5 text-xs" />
          <select value={form.review_mode}
            onChange={(e) => setForm({ ...form, review_mode: e.target.value })}
            className="rounded border border-gray-200 px-2 py-1.5 text-xs text-gray-600">
            <option value="auto">auto — crystallize (born quarantine)</option>
            <option value="gated">gated — queue for review</option>
          </select>
          <input type="password" placeholder="Token (private repos; stored encrypted)"
            value={form.token}
            onChange={(e) => setForm({ ...form, token: e.target.value })}
            className="rounded border border-gray-200 px-2.5 py-1.5 text-xs" />
          <input type="number" min={1} value={form.cadence_minutes}
            onChange={(e) => setForm({ ...form, cadence_minutes: Number(e.target.value) || 15 })}
            title="Cadence (minutes)"
            className="rounded border border-gray-200 px-2.5 py-1.5 text-xs" />
          <button onClick={create}
            className="col-span-2 rounded-lg bg-brand-600 px-3 py-1.5 text-xs font-medium text-zinc-50 hover:bg-brand-500">
            Register watch
          </button>
        </div>
      )}

      {rows.length === 0 && !adding ? (
        <p className="text-xs text-gray-400">No watched sources yet.</p>
      ) : (
        <div className="space-y-1.5">
          {rows.map((w) => (
            <div key={w.id} className="rounded-lg border border-gray-100">
              <div className="flex items-center gap-2.5 px-3 py-2">
                <span className="rounded bg-gray-100 px-1.5 py-0.5 text-[10px] font-mono text-gray-500">{w.scheme}</span>
                <span className="text-sm font-medium text-gray-700">{w.source_name}</span>
                <span className="text-xs text-gray-400 font-mono truncate">{w.config?.repo}@{w.config?.branch || "master"}</span>
                <WatchStateChip w={w} />
                <span className="rounded bg-gray-50 px-1.5 py-0.5 text-[10px] text-gray-400">{w.review_mode}</span>
                <span className="text-[10px] text-gray-300 whitespace-nowrap">
                  checked {relTime(w.last_checked_at)}
                  {w.last_state?.head ? ` · ${w.last_state.head.slice(0, 7)}` : ""}
                </span>
                <span className="ml-auto flex items-center gap-1.5">
                  <button onClick={() => setOpenActivity(openActivity === w.id ? null : w.id)}
                    className="rounded border border-gray-200 px-2 py-0.5 text-[10px] text-gray-500 hover:bg-gray-50">
                    {openActivity === w.id ? "Hide activity" : "Activity"}
                  </button>
                  <button onClick={() => act(w.id, { action: "sync_now" })}
                    className="rounded border border-gray-200 px-2 py-0.5 text-[10px] text-gray-500 hover:bg-gray-50">Sync now</button>
                  <button onClick={() => act(w.id, { status: w.status === "active" ? "paused" : "active" })}
                    className="rounded border border-gray-200 px-2 py-0.5 text-[10px] text-gray-500 hover:bg-gray-50">
                    {w.status === "active" ? "Pause" : "Resume"}
                  </button>
                  <button onClick={() => {
                      if (window.confirm(`Remove the watch on ${w.source_name}? Its crystals stay.`))
                        act(w.id, {}, "DELETE");
                    }}
                    className="rounded border border-red-200 bg-red-50 px-2 py-0.5 text-[10px] text-red-500 hover:bg-red-100">Remove</button>
                </span>
              </div>
              {w.last_error && (
                <p className="px-3 pb-1.5 text-[10px] text-red-500" title={w.last_error}>⚠ {w.last_error.slice(0, 140)}</p>
              )}
              {openActivity === w.id && (
                <WatchActivityDrawer watchId={w.id} customerId={selectedCustomerId!} />
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}


// --- Gate G3 (2026-07-23): Data shapes — the schema review surface ---------
// G-Q2=A: the status column IS the review queue; this panel is both the
// proposal review and the lifetime mapping editor. The samples pane renders
// records THROUGH the mapping via the same apply_mapping the pipeline runs.

interface SourceSchemaItem {
  id: string; schema_hash: string; status: string;
  mapping: { version?: number; roles?: Record<string, string>; subject?: string | null; domain?: string | null };
  sample: unknown[]; label: string | null; parked_count: number;
  created_at: string | null; updated_at: string | null;
}

interface SchemaPreviewItem {
  key: string; sparse_key: string; value: string; type: string; citation: string;
}

const SCHEMA_ROLES = ["key", "value", "locator", "timestamp", "skip"];

const SCHEMA_BADGES: Record<string, string> = {
  proposed: "bg-amber-50 text-amber-600 border border-amber-200",
  approved: "bg-emerald-50 text-emerald-600 border border-emerald-200",
  rejected: "bg-gray-100 text-gray-500 border border-gray-200",
};

function SchemaCard({ s, customerId }: { s: SourceSchemaItem; customerId: string }) {
  const queryClient = useQueryClient();
  const [expanded, setExpanded] = useState(s.status === "proposed");
  const [roles, setRoles] = useState<Record<string, string>>(s.mapping?.roles ?? {});
  const [dirty, setDirty] = useState(false);
  const [busy, setBusy] = useState(false);

  const candidateMapping = {
    version: 1, roles,
    subject: s.mapping?.subject ?? null,
    domain: s.mapping?.domain ?? "General",
  };

  const preview = useQuery({
    queryKey: ["schema-preview", s.id, JSON.stringify(roles)],
    enabled: expanded && Object.keys(roles).length > 0 && s.sample.length > 0,
    queryFn: async (): Promise<{ items: SchemaPreviewItem[] }> => {
      const res = await authedFetch(`/admin/api/source-schemas/preview?customer_id=${encodeURIComponent(customerId)}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ mapping: candidateMapping, sample: s.sample, label: s.label ?? "" }),
      });
      if (!res.ok) return { items: [] };
      return res.json();
    },
  });

  const refresh = () => {
    queryClient.invalidateQueries({ queryKey: ["source-schemas"] });
    queryClient.invalidateQueries({ queryKey: ["documents"] });
    queryClient.invalidateQueries({ queryKey: ["watches"] });
  };

  const post = async (action: "approve" | "reject") => {
    setBusy(true);
    try {
      const res = await authedFetch(
        `/admin/api/source-schemas/${encodeURIComponent(s.id)}/${action}?customer_id=${encodeURIComponent(customerId)}`,
        { method: "POST" },
      );
      if (!res.ok) window.alert(`${action} failed (${res.status})`);
      refresh();
    } finally { setBusy(false); }
  };

  const saveMapping = async () => {
    setBusy(true);
    try {
      const res = await authedFetch(
        `/admin/api/source-schemas/${encodeURIComponent(s.id)}/mapping?customer_id=${encodeURIComponent(customerId)}`,
        {
          method: "PUT",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ mapping: candidateMapping }),
        },
      );
      if (!res.ok) window.alert(`Save failed (${res.status})`);
      else setDirty(false);
      refresh();
    } finally { setBusy(false); }
  };

  const shortHash = `${s.schema_hash.slice(0, 4)}…${s.schema_hash.slice(-4)}`;
  const badge = SCHEMA_BADGES[s.status] ?? SCHEMA_BADGES.rejected;
  const previewItems = preview.data?.items ?? [];

  return (
    <div className="rounded-lg border border-gray-200 bg-white">
      <div className="flex items-center gap-2 px-3 py-2 cursor-pointer" onClick={() => setExpanded((v) => !v)}>
        <span className="font-mono text-xs text-gray-700 truncate">{s.label ?? "(unnamed shape)"}</span>
        <span className={`rounded-full px-2 py-0.5 text-[10px] font-medium ${badge}`}>
          {s.status.charAt(0).toUpperCase() + s.status.slice(1)}
        </span>
        <span className="font-mono text-[10px] text-gray-300">{shortHash}</span>
        {s.parked_count > 0 && (
          <span className="text-[11px] text-amber-600">{s.parked_count} waiting</span>
        )}
        <ChevronDown className={`ml-auto h-3.5 w-3.5 text-gray-400 transition-transform ${expanded ? "rotate-180" : ""}`} />
      </div>
      {expanded && (
        <div className="border-t border-gray-100 px-3 py-2">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-[10px] uppercase tracking-wide text-gray-400 mb-1">Field mapping</p>
              {Object.keys(roles).length === 0 && (
                <p className="text-[11px] text-gray-400">No mapping was inferred (the model was unavailable at first contact). Reject this shape, or re-upload the file once a model is configured to get a fresh proposal.</p>
              )}
              <table className="w-full">
                <tbody>
                  {Object.entries(roles).map(([path, role]) => (
                    <tr key={path}>
                      <td className="py-0.5 pr-2 font-mono text-[11px] text-gray-600 truncate max-w-[180px]" title={path}>{path}</td>
                      <td className="py-0.5 text-right">
                        <select
                          value={role}
                          onChange={(e) => {
                            setRoles((r) => ({ ...r, [path]: e.target.value }));
                            setDirty(true);
                          }}
                          className="rounded border border-gray-200 bg-gray-50 px-1.5 py-0.5 text-[11px] text-gray-700"
                        >
                          {SCHEMA_ROLES.map((r) => <option key={r} value={r}>{r}</option>)}
                        </select>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-1.5 text-[10px] text-gray-400">
                Subject: <span className="font-mono">{s.mapping?.subject ?? "(none)"}</span> · Domain: {s.mapping?.domain ?? "General"}
              </p>
            </div>
            <div>
              <p className="text-[10px] uppercase tracking-wide text-gray-400 mb-1">Samples through this mapping</p>
              {preview.isFetching && <p className="text-[11px] text-gray-400">Rendering…</p>}
              {!preview.isFetching && previewItems.length === 0 && (
                <p className="text-[11px] text-gray-400">No facts produced. Mark at least one path as value.</p>
              )}
              <div className="space-y-1.5">
                {previewItems.slice(0, 3).map((it, i) => (
                  <div key={i} className="rounded border border-gray-100 bg-gray-50 px-2 py-1.5">
                    <p className="text-[11px] font-medium text-gray-700">{it.key}</p>
                    <p className="text-[11px] text-gray-500">{it.value}</p>
                    <p className="font-mono text-[10px] text-gray-400 truncate" title={it.sparse_key}>{it.sparse_key}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="mt-2.5 flex items-center gap-2">
            {s.status === "proposed" && (
              <button disabled={busy}
                onClick={() => post("approve")}
                className="rounded bg-gray-900 px-2.5 py-1 text-[11px] font-medium text-white hover:bg-gray-700 disabled:opacity-50">
                Approve{s.parked_count > 0 ? ` and release ${s.parked_count}` : ""}
              </button>
            )}
            {s.status === "proposed" && (
              <button disabled={busy}
                onClick={() => { if (window.confirm("Reject this shape? Waiting documents park permanently.")) post("reject"); }}
                className="rounded border border-gray-200 px-2.5 py-1 text-[11px] text-gray-600 hover:bg-gray-50 disabled:opacity-50">
                Reject shape
              </button>
            )}
            {dirty && (
              <button disabled={busy}
                onClick={saveMapping}
                className="rounded border border-brand-200 bg-brand-50 px-2.5 py-1 text-[11px] text-brand-600 hover:bg-brand-100 disabled:opacity-50">
                Save mapping
              </button>
            )}
            <span className="ml-auto text-[10px] text-gray-300">Edits apply to future arrivals</span>
          </div>
        </div>
      )}
    </div>
  );
}

export function DataShapesPanel() {
  const { selectedCustomerId } = useSelectedCustomer();
  const schemas = useQuery({
    queryKey: ["source-schemas", selectedCustomerId],
    queryFn: async (): Promise<{ schemas: SourceSchemaItem[] }> => {
      const res = await authedFetch(
        `/admin/api/source-schemas?customer_id=${encodeURIComponent(selectedCustomerId!)}`,
      );
      return res.json();
    },
    enabled: !!selectedCustomerId,
    refetchInterval: 15000,
  });
  const rows = schemas.data?.schemas ?? [];
  if (!selectedCustomerId || (rows.length === 0 && !schemas.isLoading)) return null;
  const proposed = rows.filter((s) => s.status === "proposed");
  const settled = rows.filter((s) => s.status !== "proposed");
  return (
    <div className="mt-6">
      <div className="flex items-center gap-2 mb-2">
        <h3 className="text-sm font-semibold text-gray-800">Data shapes</h3>
        {proposed.length > 0 && (
          <span className="rounded-full bg-amber-50 border border-amber-200 px-2 py-0.5 text-[10px] font-medium text-amber-600">
            {proposed.length} awaiting review
          </span>
        )}
      </div>
      <p className="text-[11px] text-gray-400 mb-2">
        One approval per shape of data, ever. Approved shapes ingest mechanically with zero model calls.
      </p>
      <div className="space-y-2">
        {[...proposed, ...settled].map((s) => (
          <SchemaCard key={s.id} s={s} customerId={selectedCustomerId} />
        ))}
      </div>
    </div>
  );
}
