// Knowledge › Documents (Console Refine PR 5, 5.3 items 3 to 5): the
// upload progress rows, the crystallize progress line and the document
// grid. Every card carries a status pill in plain words, one primary
// action for that status, and an always-visible Remove that asks inline.
import { ReactNode, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { FileText, Upload, X } from "lucide-react";
import { api, errorMessage } from "@/lib/api";
import { ConfirmInline, EmptyState, ErrorBanner, IconButton, Skeleton, Meter, StatusPill } from "@/components/ui";
import type { DocumentItem } from "@/lib/types";
import { fmtRelative } from "@/lib/utils";
import { DOC_STATUS, fmtChars } from "./shared";
import type { UploadRow } from "./uploads";

export interface CrystallizeProgress { label: string; done: number; total: number }

export function Documents({
  customerId, docs, isLoading, error, onRetry, uploads, onCancelUpload, onDismissUpload,
  crystallizing, onReview, onViewCrystals, uploadControls,
}: {
  customerId: string;
  docs: DocumentItem[];
  isLoading: boolean;
  error: unknown;
  onRetry: () => void;
  uploads: UploadRow[];
  onCancelUpload: (id: string) => void;
  onDismissUpload: (id: string) => void;
  crystallizing: CrystallizeProgress | null;
  onReview: (doc: DocumentItem, readOnly: boolean) => void;
  onViewCrystals: (doc: DocumentItem) => void;
  /** The Upload / Upload folder controls, reused by the empty state. */
  uploadControls: ReactNode;
}) {
  const active = uploads.filter((u) => u.state === "uploading" || u.state === "queued");
  const activeTotal = uploads.length;
  return (
    <div>
      {uploads.length > 0 && (
        <ul className="mb-4 space-y-1.5" aria-label="Uploads">
          {uploads.map((u, i) => (
            <li key={u.id} className="flex flex-wrap items-center gap-2 rounded-lg border border-line bg-surface px-3 py-2 text-[12.5px]">
              <Upload aria-hidden className="h-3.5 w-3.5 shrink-0 text-fg-3" />
              <span className="min-w-0 flex-1 truncate text-fg" title={u.label}>{u.label}</span>
              {u.state === "uploading" && (
                <>
                  <span className="text-fg-3">uploading, {i + 1} of {activeTotal} · {Math.round(u.fraction * 100)}%</span>
                  <Meter value={u.fraction * 100} max={100} label={`Uploading ${u.label}`} tone="accent" hideLabel className="w-24" />
                  <button type="button" onClick={() => onCancelUpload(u.id)}
                    className="rounded-md border border-line px-2 py-0.5 text-[12px] text-fg-2 hover:bg-inset hover:text-fg">Cancel</button>
                </>
              )}
              {u.state === "queued" && (
                <>
                  <span className="text-fg-3">waiting, {i + 1} of {activeTotal}</span>
                  <button type="button" onClick={() => onCancelUpload(u.id)}
                    className="rounded-md border border-line px-2 py-0.5 text-[12px] text-fg-2 hover:bg-inset hover:text-fg">Cancel</button>
                </>
              )}
              {u.state === "done" && <StatusPill variant="ok">Uploaded</StatusPill>}
              {u.state === "cancelled" && <StatusPill variant="neutral">Cancelled</StatusPill>}
              {u.state === "failed" && (
                <>
                  <span className="basis-full text-bad" role="alert">{u.error}</span>
                  <IconButton label="Dismiss" icon={X} size="sm" onClick={() => onDismissUpload(u.id)} />
                </>
              )}
            </li>
          ))}
          {active.length > 0 && <li className="sr-only" aria-live="polite">{active.length} uploads in progress</li>}
        </ul>
      )}

      {crystallizing && (
        <div className="mb-4 rounded-lg border border-line bg-surface px-3 py-2 text-[12.5px]" aria-live="polite">
          <div className="flex items-center justify-between gap-3">
            <span className="text-fg">Reading <span className="font-medium">{crystallizing.label}</span></span>
            <span className="text-fg-3">{crystallizing.done} of {crystallizing.total}</span>
          </div>
          <Meter value={crystallizing.done} max={crystallizing.total} label="Crystallize progress" tone="accent" hideLabel className="mt-1.5" />
        </div>
      )}

      {isLoading && <Skeleton rows={4} />}
      {!!error && <ErrorBanner title="Couldn't load documents" error={error} onRetry={onRetry} />}

      {!isLoading && !error && docs.length === 0 && (
        <EmptyState icon={FileText} title="No documents yet"
          help="Upload a file, a folder, or a photo of a whiteboard. You review what was read before it becomes memory."
          action={<div className="flex flex-wrap justify-center gap-2">{uploadControls}</div>} />
      )}

      {docs.length > 0 && (
        <div className="grid gap-3" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))" }}>
          {docs.map((doc) => (
            <DocumentCard key={doc.id} customerId={customerId} doc={doc}
              onReview={onReview} onViewCrystals={onViewCrystals} />
          ))}
        </div>
      )}
    </div>
  );
}

function DocumentCard({ customerId, doc, onReview, onViewCrystals }: {
  customerId: string;
  doc: DocumentItem;
  onReview: (doc: DocumentItem, readOnly: boolean) => void;
  onViewCrystals: (doc: DocumentItem) => void;
}) {
  const qc = useQueryClient();
  const [confirming, setConfirming] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const remove = useMutation({
    mutationFn: () => api.deleteDocument(customerId, doc.id),
    onSuccess: () => { setConfirming(false); qc.invalidateQueries({ queryKey: ["documents", customerId] }); },
    onError: (e) => { setConfirming(false); setErr(errorMessage(e, "Couldn't remove this document.")); },
  });
  const status = DOC_STATUS[doc.status] ?? { label: doc.status, variant: "neutral" as const };
  const busy = doc.status === "approved" || doc.status === "crystallizing";
  const n = doc.crystals_written ?? 0;

  return (
    <article className="flex flex-col rounded-xl border border-line bg-surface p-3.5">
      <div className="flex items-start gap-2">
        <FileText aria-hidden className="mt-0.5 h-4 w-4 shrink-0 text-fg-3" />
        <div className="min-w-0 flex-1">
          <h3 className="truncate text-[13.5px] font-medium text-fg" title={doc.label}>{doc.label || "Untitled"}</h3>
          <p className="mt-0.5 text-[11.5px] text-fg-3">{fmtChars(doc.char_count)} · added {fmtRelative(doc.created_at)}</p>
        </div>
        <StatusPill variant={status.variant} raw={doc.status} dot={busy} className="shrink-0">
          {doc.status === "crystallized" ? `Crystallized · ${n} crystal${n === 1 ? "" : "s"}` : status.label}
        </StatusPill>
      </div>

      {doc.status === "error" && (
        <p className="mt-2 text-[12px] text-bad">
          {doc.error_message || "The file could not be read."} Remove it and upload a text version.
        </p>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-2">
        {doc.status === "review" && (
          <button type="button" onClick={() => onReview(doc, false)}
            className="rounded-md bg-accent px-2.5 py-1 text-[12.5px] font-medium text-on-accent hover:bg-accent-hover">
            Review {doc.items_extracted} item{doc.items_extracted === 1 ? "" : "s"}
          </button>
        )}
        {doc.status === "crystallized" && (
          <>
            <button type="button" onClick={() => onViewCrystals(doc)}
              className="rounded-md border border-line px-2.5 py-1 text-[12.5px] font-medium text-fg-2 hover:bg-inset hover:text-fg">
              View {n} crystal{n === 1 ? "" : "s"}
            </button>
            <button type="button" onClick={() => onReview(doc, true)}
              className="text-[12px] text-fg-3 underline decoration-dotted hover:text-fg-2">
              What was approved
            </button>
          </>
        )}
        {doc.status === "pending" && <span className="text-[12px] text-fg-3">Waiting for Crystallize.</span>}
        {busy && <span className="text-[12px] text-fg-3">Writing memories…</span>}
        <span className="flex-1" />
        <button type="button" onClick={() => setConfirming(true)} disabled={confirming || busy}
          className="text-[12px] text-fg-3 hover:text-bad disabled:opacity-50">
          Remove
        </button>
      </div>
      <ConfirmInline
        open={confirming}
        message={n > 0 ? `Remove this document? Its ${n} crystal${n === 1 ? "" : "s"} stay in the bank.` : "Remove this document?"}
        confirmLabel="Remove"
        busy={remove.isPending}
        onCancel={() => setConfirming(false)}
        onConfirm={() => remove.mutate()}
      />
      {err && <ErrorBanner title="That didn't go through" message={err} className="mt-2" />}
    </article>
  );
}
