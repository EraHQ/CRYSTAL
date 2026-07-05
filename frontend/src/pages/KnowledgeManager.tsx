import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Upload, Loader2, CheckCircle, AlertCircle, Trash2,
  Globe, FileText, Gem, Zap, FolderOpen,
  Cloud, Check, ChevronDown, ArrowLeft, Pencil, X, Save,
} from "lucide-react";
import { api } from "@/lib/api";
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

  const adminKey = useQuery({
    queryKey: ["admin_key", selectedCustomerId],
    queryFn: () => api.getAdminKey(selectedCustomerId!),
    enabled: !!selectedCustomerId,
  });

  const documents = useQuery({
    queryKey: ["documents", selectedCustomerId],
    queryFn: async () => {
      if (!adminKey.data) throw new Error("No admin key");
      const res = await fetch("/v1/documents", {
        headers: { Authorization: `Bearer ${adminKey.data.api_key}`, "Content-Type": "application/json" },
      });
      if (!res.ok) throw new Error(`${res.status}`);
      return res.json() as Promise<{ total: number; documents: DocumentItem[] }>;
    },
    enabled: !!selectedCustomerId && !!adminKey.data,
    refetchInterval: 15000,
  });

  const subscriptions = useQuery({
    queryKey: ["subscriptions", selectedCustomerId],
    queryFn: async () => {
      if (!adminKey.data) throw new Error("No admin key");
      const res = await fetch("/v1/subscriptions", {
        headers: { Authorization: `Bearer ${adminKey.data.api_key}` },
      });
      if (!res.ok) throw new Error(`${res.status}`);
      return res.json() as Promise<{ general_crystal_types: string[] }>;
    },
    enabled: !!selectedCustomerId && !!adminKey.data,
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
    if (!adminKey.data) return;
    if (isSubscribed) {
      await fetch(`/v1/subscribe/${encodeURIComponent(typeId)}`, { method: "DELETE", headers: { Authorization: `Bearer ${adminKey.data.api_key}` } });
    } else {
      await fetch("/v1/subscribe", { method: "POST", headers: { Authorization: `Bearer ${adminKey.data.api_key}`, "Content-Type": "application/json" }, body: JSON.stringify({ crystal_type: typeId }) });
    }
    queryClient.invalidateQueries({ queryKey: ["subscriptions", selectedCustomerId] });
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || !adminKey.data || !selectedCustomerId) return;
    for (const file of Array.from(files)) {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("label", file.name.replace(/\.[^.]+$/, ""));
      await fetch("/v1/documents/upload", { method: "POST", headers: { Authorization: `Bearer ${adminKey.data.api_key}` }, body: formData });
    }
    e.target.value = "";
    queryClient.invalidateQueries({ queryKey: ["documents", selectedCustomerId] });
  };

  const handleCrystallize = async () => {
    if (!adminKey.data || !selectedCustomerId) return;
    setPhase("crystallizing");
    setCrystallizeProgress(0);
    const pendingDocs = docs.filter((d) => d.status === "pending");
    const allItems: CrystallizeItem[] = [];
    for (let i = 0; i < pendingDocs.length; i++) {
      const doc = pendingDocs[i];
      setCrystallizingDoc(doc.label || "Untitled");
      setCrystallizeProgress(Math.round((i / pendingDocs.length) * 100));
      try {
        const res = await fetch(`/v1/documents/${doc.id}/crystallize`, {
          method: "POST", headers: { Authorization: `Bearer ${adminKey.data.api_key}`, "Content-Type": "application/json" },
        });
        if (res.ok) { const data = await res.json(); allItems.push(...(data.items || [])); }
        setCrystallizeProgress(Math.round(((i + 1) / pendingDocs.length) * 100));
      } catch {}
    }
    setCrystallizeProgress(100);
    setReviewItems(allItems);
    queryClient.invalidateQueries({ queryKey: ["documents", selectedCustomerId] });
    setTimeout(() => setPhase("review"), 600);
  };

  const handleDelete = async (docId: string) => {
    if (!adminKey.data) return;
    await fetch(`/v1/documents/${docId}`, { method: "DELETE", headers: { Authorization: `Bearer ${adminKey.data.api_key}` } });
    queryClient.invalidateQueries({ queryKey: ["documents", selectedCustomerId] });
  };

  if (!selectedCustomerId) return <EmptyState title="No customer selected" description="Select a customer to manage knowledge." />;

  // ── Document Review Panel ──
  if (reviewingDocId) {
    return (
      <DocumentReviewPanel
        documentId={reviewingDocId}
        adminKey={adminKey.data?.api_key || ""}
        onBack={() => {
          setReviewingDocId(null);
          queryClient.invalidateQueries({ queryKey: ["documents", selectedCustomerId] });
        }}
        onApprove={async (docId, items, chunks, includeChunks) => {
          // Fire and forget — navigate back immediately
          fetch(`/v1/documents/${docId}/approve`, {
            method: "POST",
            headers: { Authorization: `Bearer ${adminKey.data?.api_key}`, "Content-Type": "application/json" },
            body: JSON.stringify({ items, content_chunks: chunks, include_chunks: includeChunks }),
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
            <label className="cursor-pointer inline-flex items-center gap-1.5 rounded-lg px-3.5 py-2 text-sm font-medium bg-brand-600 text-zinc-50 shadow-glow hover:bg-brand-500 transition-all">
              <Upload className="h-4 w-4" /> Upload
              <input type="file" className="hidden" accept=".pdf,.docx,.txt,.md,.py,.pyi,.js,.jsx,.ts,.tsx,.go,.rs,.java,.rb,.c,.h,.cpp,.cs,.php,.swift,.kt,.sh" multiple onChange={handleFileUpload} />
            </label>
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
                  <CrystalButton size="sm" onClick={() => setReviewingDocId(doc.id)}>
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
                  <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2 py-0.5 text-[11px] font-medium text-emerald-700">
                    <Gem className="h-3 w-3" /> {doc.crystals_written}
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
      <DriveConnector adminKey={adminKey.data?.api_key} onImportComplete={() => queryClient.invalidateQueries({ queryKey: ["documents", selectedCustomerId] })} />

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
}

interface ReviewItem {
  key: string;
  sparse_key?: string;
  value: string;
  type: string;
}

function DocumentReviewPanel({
  documentId, adminKey, onBack, onApprove, approving,
}: {
  documentId: string;
  adminKey: string;
  onBack: () => void;
  onApprove: (docId: string, items: ReviewItem[], chunks: ReviewChunk[], includeChunks: boolean) => void;
  approving: boolean;
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
      const res = await fetch(`/v1/documents/${documentId}/review`, {
        headers: { Authorization: `Bearer ${adminKey}` },
      });
      if (!res.ok) throw new Error("Failed to load review data");
      return res.json() as Promise<{
        document_id: string; label: string; status: string;
        detected_type: string; confirmed_type: string | null;
        content_chunks: ReviewChunk[]; extracted_items: ReviewItem[];
        char_count: number; items_extracted: number;
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
          <label className="flex items-center gap-2 text-xs text-gray-600">
            <input type="checkbox" checked={includeChunks} onChange={(e) => setIncludeChunks(e.target.checked)} className="rounded border-gray-300" />
            Include content chunks
          </label>
          <CrystalButton onClick={() => onApprove(documentId, items, chunks, includeChunks)} disabled={approving}>
            {approving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Check className="h-4 w-4" />}
            Approve & Crystallize ({items.length + (includeChunks ? chunks.length : 0)})
          </CrystalButton>
        </div>
      </div>

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
                <div className="px-3 py-2 max-h-[300px] overflow-y-auto">
                  <pre className="text-xs text-gray-600 whitespace-pre-wrap font-mono leading-relaxed">{chunk.text}</pre>
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
    </div>
  );
}
