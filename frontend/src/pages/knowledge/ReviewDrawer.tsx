// Knowledge › review drawer (Console Refine PR 5, 5.3 item 6): what was
// read from one document, as a Drawer at /knowledge/review/:docId. Items
// and chunks can be pruned and edited before approval; the approve
// button counts what is still selected and is impossible at zero (the
// server refuses an explicit empty selection too, RC-16). Approve awaits
// the response; a failure stays in the drawer as an ErrorBanner.
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Pencil, Trash2 } from "lucide-react";
import { api, errorMessage } from "@/lib/api";
import { Drawer, ErrorBanner, IconButton, Skeleton, StatusPill, useToast } from "@/components/ui";
import type { ReviewChunk, ReviewItem } from "@/lib/types";
import { cn } from "@/lib/utils";
import { fmtChars } from "./shared";

const ITEM_TYPES = ["fact", "entity", "relationship", "process", "definition", "qa"];

export function ReviewDrawer({ customerId, docId, readOnly, onClose }: {
  customerId: string;
  docId: string;
  readOnly: boolean;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const toast = useToast();
  const [tab, setTab] = useState<"items" | "chunks">("items");
  const [includeChunks, setIncludeChunks] = useState(true);
  const [items, setItems] = useState<ReviewItem[] | null>(null);
  const [chunks, setChunks] = useState<ReviewChunk[] | null>(null);
  const [editing, setEditing] = useState<number | null>(null);
  const [draft, setDraft] = useState<ReviewItem>({ key: "", value: "", type: "fact" });
  const [expanded, setExpanded] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const review = useQuery({
    queryKey: ["doc_review", customerId, docId],
    queryFn: () => api.getDocumentReview(customerId, docId),
  });
  const data = review.data;
  const liveItems = items ?? data?.extracted_items ?? [];
  const liveChunks = chunks ?? data?.content_chunks ?? [];
  const selected = liveItems.length + (includeChunks ? liveChunks.length : 0);

  const approve = useMutation({
    mutationFn: () => api.approveDocument(customerId, docId, {
      items: liveItems, content_chunks: includeChunks ? liveChunks : [], include_chunks: includeChunks,
    }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["documents", customerId] });
      toast(`Crystallizing ${data?.label ?? "the document"}. It shows up in the Bank when done.`);
      onClose();
    },
    onError: (e) => setError(errorMessage(e, "Approve failed.")),
  });

  const flagged = liveChunks.filter((c) => (c.injection_hits ?? []).length > 0);

  const startEdit = (idx: number) => { setEditing(idx); setDraft({ ...liveItems[idx] }); };
  const saveEdit = () => {
    if (editing === null) return;
    const next = [...liveItems];
    next[editing] = { ...next[editing], key: draft.key, value: draft.value, type: draft.type };
    setItems(next);
    setEditing(null);
  };
  const removeItem = (idx: number) => setItems(liveItems.filter((_, i) => i !== idx));
  const removeChunk = (idx: number) => setChunks(liveChunks.filter((_, i) => i !== idx));
  const describeChunk = (idx: number, description: string) => {
    const next = [...liveChunks];
    next[idx] = { ...next[idx], description };
    setChunks(next);
  };

  const footer = readOnly ? (
    <p className="text-[12.5px] text-fg-3">Crystallized. This is what was approved; the crystals live in the Bank.</p>
  ) : (
    <div className="flex flex-wrap items-center gap-3">
      <label className="flex items-center gap-2 text-[12.5px] text-fg-2">
        <input type="checkbox" checked={includeChunks} onChange={(e) => setIncludeChunks(e.target.checked)}
          className="h-3.5 w-3.5 rounded border-line" />
        Include content chunks
      </label>
      <span className="flex-1" />
      <button type="button" onClick={() => approve.mutate()} disabled={selected === 0 || approve.isPending || !data}
        title={selected === 0 ? "Nothing is selected. Keep at least one item or chunk." : undefined}
        className="rounded-md bg-accent px-3 py-1.5 text-[13px] font-medium text-on-accent hover:bg-accent-hover disabled:opacity-40">
        {approve.isPending ? "Approving…" : `Approve & crystallize ${selected}`}
      </button>
    </div>
  );

  return (
    <Drawer open onClose={onClose} title={data?.label ?? (review.isLoading ? "Loading…" : "Review")}
      className="min-[720px]:w-[640px]" footer={footer}>
      <nav className="mb-3 text-[12.5px]">
        <button type="button" onClick={onClose} className="text-fg-2 hover:text-fg">‹ Documents</button>
      </nav>
      {review.isLoading && <Skeleton rows={4} />}
      {review.isError && <ErrorBanner title="Couldn't load this document" error={review.error} onRetry={() => review.refetch()} />}
      {error && <ErrorBanner title="That didn't go through" message={error} className="mb-3" />}
      {data && (
        <>
          <p className="text-[12.5px] text-fg-2">
            <span title="detected type">{data.detected_type}</span> · {liveItems.length} item{liveItems.length === 1 ? "" : "s"} · {liveChunks.length} chunk{liveChunks.length === 1 ? "" : "s"} · {fmtChars(data.char_count)}
          </p>

          {!!data.comprehension?.imports?.length && (
            <section className="mt-3 rounded-lg border border-line bg-inset px-3 py-2">
              <p className="text-[11px] font-medium uppercase tracking-[0.08em] text-fg-3">Imports found</p>
              <div className="mt-1.5 flex flex-wrap gap-1.5">
                {data.comprehension.imports.map((imp) => (
                  <span key={imp.module} title={imp.resolved_path ?? "not resolved in this upload"}
                    className={cn("rounded-md px-1.5 py-0.5 font-mono text-[11.5px] ring-1 ring-inset",
                      imp.resolved_path ? "bg-accent-soft text-accent ring-accent/30" : "bg-surface text-fg-3 ring-line")}>
                    {imp.module}
                  </span>
                ))}
              </div>
            </section>
          )}
          {flagged.length > 0 && (
            <section className="mt-3 rounded-lg border border-warn/40 bg-warn-soft px-3 py-2 text-[12.5px]">
              <p className="font-medium text-warn">Instruction-shaped text in {flagged.length} chunk{flagged.length === 1 ? "" : "s"}</p>
              <ul className="mt-1 space-y-0.5 text-fg-2">
                {flagged.map((c) => (
                  <li key={c.index}>#{c.index} {c.label}: <span className="font-mono text-[11.5px]">{(c.injection_hits ?? []).join(", ")}</span></li>
                ))}
              </ul>
              <p className="mt-1 text-fg-3">Approving keeps them as reviewed content. Remove a chunk to leave it out.</p>
            </section>
          )}

          <div role="tablist" aria-label="Review sections" className="mt-4 mb-3 flex gap-4 border-b border-line text-[12.5px]">
            {([["items", `Items ${liveItems.length}`], ["chunks", `Chunks ${liveChunks.length}`]] as const).map(([id, label]) => (
              <button key={id} type="button" role="tab" aria-selected={tab === id} onClick={() => setTab(id)}
                className={cn("-mb-px border-b-2 pb-2", tab === id ? "border-accent font-medium text-accent" : "border-transparent text-fg-3 hover:text-fg-2")}>
                {label}
              </button>
            ))}
          </div>

          {tab === "items" && (
            <div className="space-y-2">
              {liveItems.length === 0 && <p className="text-[12.5px] text-fg-3">No items were extracted from this document.</p>}
              {liveItems.map((item, idx) => (
                <div key={`${idx}-${item.key}`} className="rounded-lg border border-line bg-surface p-3">
                  {editing === idx ? (
                    <div className="space-y-2">
                      <label className="block text-[12px] text-fg-2">Type
                        <select value={draft.type} onChange={(e) => setDraft({ ...draft, type: e.target.value })}
                          className="mt-0.5 block rounded-md border border-line bg-inset px-2 py-1 text-[12.5px] text-fg">
                          {ITEM_TYPES.map((t) => <option key={t} value={t}>{t === "qa" ? "Q&A" : t}</option>)}
                        </select>
                      </label>
                      <label className="block text-[12px] text-fg-2">Key (what it is about)
                        <input value={draft.key} onChange={(e) => setDraft({ ...draft, key: e.target.value })}
                          className="mt-0.5 block w-full rounded-md border border-line bg-inset px-2 py-1 text-[13px] text-fg outline-none focus:border-accent" />
                      </label>
                      <label className="block text-[12px] text-fg-2">Value (the memory)
                        <textarea value={draft.value} rows={3} onChange={(e) => setDraft({ ...draft, value: e.target.value })}
                          className="mt-0.5 block w-full resize-y rounded-md border border-line bg-inset px-2 py-1 text-[13px] text-fg outline-none focus:border-accent" />
                      </label>
                      <div className="flex justify-end gap-2">
                        <button type="button" onClick={() => setEditing(null)}
                          className="rounded-md border border-line px-2.5 py-1 text-[12.5px] text-fg-2 hover:bg-inset">Cancel</button>
                        <button type="button" onClick={saveEdit} disabled={!draft.key.trim() || !draft.value.trim()}
                          className="rounded-md bg-accent px-2.5 py-1 text-[12.5px] font-medium text-on-accent hover:bg-accent-hover disabled:opacity-40">Save</button>
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-start gap-2">
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <StatusPill variant="neutral">{item.type === "qa" ? "Q&A" : item.type}</StatusPill>
                          <span className="break-words text-[13px] font-medium text-fg">{item.key}</span>
                        </div>
                        <p className="mt-1 whitespace-pre-wrap break-words text-[13px] leading-relaxed text-fg-2">{item.value}</p>
                      </div>
                      {!readOnly && (
                        <div className="flex shrink-0 items-center">
                          <IconButton label="Edit" icon={Pencil} size="sm" onClick={() => startEdit(idx)} />
                          <IconButton label="Remove" icon={Trash2} size="sm" variant="danger" onClick={() => removeItem(idx)} />
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {tab === "chunks" && (
            <div className="space-y-2">
              {liveChunks.length === 0 && <p className="text-[12.5px] text-fg-3">No content chunks were made.</p>}
              {liveChunks.map((chunk, idx) => (
                <div key={`${idx}-${chunk.index}`} className="rounded-lg border border-line bg-surface">
                  <div className="flex flex-wrap items-center gap-2 px-3 py-2">
                    <span className="font-mono text-[11.5px] text-fg-3">#{chunk.index}</span>
                    <span className="min-w-0 flex-1 truncate text-[13px] font-medium text-fg" title={chunk.label}>{chunk.label}</span>
                    {(chunk.description ?? "").trim() && <StatusPill variant="info" raw={chunk.description ?? ""}>Described</StatusPill>}
                    {(chunk.injection_hits ?? []).length > 0 && <StatusPill variant="warn" raw={(chunk.injection_hits ?? []).join(", ")}>Flagged</StatusPill>}
                    <span className="text-[11.5px] text-fg-3">{fmtChars(chunk.char_count)}</span>
                    <button type="button" onClick={() => setExpanded(expanded === idx ? null : idx)} aria-expanded={expanded === idx}
                      className="text-[12px] font-medium text-accent hover:underline">
                      {expanded === idx ? "Collapse" : "Expand"}
                    </button>
                    {!readOnly && <IconButton label="Remove" icon={Trash2} size="sm" variant="danger" onClick={() => removeChunk(idx)} />}
                  </div>
                  {expanded === idx && (
                    <div className="space-y-2 border-t border-line px-3 py-2">
                      <pre className="max-h-[300px] overflow-y-auto whitespace-pre-wrap font-mono text-[12px] leading-relaxed text-fg-2">{chunk.text}</pre>
                      {!readOnly && (
                        <label className="block text-[12px] text-fg-2">
                          Description <span className="text-fg-3">(steers recall; becomes a purpose fact for code)</span>
                          <textarea value={chunk.description ?? ""} rows={2} onChange={(e) => describeChunk(idx, e.target.value)}
                            placeholder="What this chunk is for"
                            className="mt-0.5 block w-full resize-y rounded-md border border-line bg-inset px-2 py-1 text-[12.5px] text-fg outline-none focus:border-accent" />
                        </label>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </Drawer>
  );
}
