// Knowledge — Console Refine PR 5 (docs/CONSOLE_UX_AUDIT.md 5.3).
//
// One PageHeader (counts in the meta, Upload / Upload folder / primary
// "Crystallize {n} queued"), four tabs with counts that deep-link as
// /knowledge?tab=…, and the review drawer at /knowledge/review/:docId.
// The tabs live in ./knowledge/*; uploads run in parallel with a
// progress row each; nothing on this page opens a native dialog.
import { ReactNode, useCallback, useRef, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { FolderUp, Upload, Zap } from "lucide-react";
import { api, errorMessage } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";
import { EmptyState, ErrorBanner, PageHeader, useToast } from "@/components/ui";
import type { DocumentItem } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Documents, type CrystallizeProgress } from "./knowledge/Documents";
import { ReviewDrawer } from "./knowledge/ReviewDrawer";
import { WatchedRepos } from "./knowledge/WatchedRepos";
import { DataShapes } from "./knowledge/DataShapes";
import { GeneralBanks } from "./knowledge/GeneralBanks";
import { GoogleDrivePanel } from "./knowledge/GoogleDrivePanel";
import { useUploads } from "./knowledge/uploads";
import { TABS, type KnowledgeTab } from "./knowledge/shared";

// LAUNCH HIDE (DRIVE Q1=A, 2026-09-27): the Drive watcher rides the
// RESTRICTED drive.readonly scope, which forces Google's heavyweight
// app verification. Hidden for launch; flip this to restore the panel.
const DRIVE_WATCHER_ENABLED = false;

// Item 4 (2026-10-22): images (png, jpg, webp) upload too; a vision model
// reads them and the text goes through review like any document.
const ACCEPT = ".pdf,.docx,.pptx,.rtf,.odt,.epub,.txt,.md,.html,.htm,.xlsx,.csv,.tsv,.json,.jsonl,.ndjson,.eml,.mbox,.vtt,.srt,.ipynb,.py,.pyi,.js,.jsx,.ts,.tsx,.go,.rs,.java,.rb,.c,.h,.cpp,.cs,.php,.swift,.kt,.sh,.png,.jpg,.jpeg,.webp";

export function KnowledgeManager() {
  const { selectedCustomerId } = useSelectedCustomer();
  const qc = useQueryClient();
  const toast = useToast();
  const navigate = useNavigate();
  const { docId } = useParams<{ docId: string }>();
  const [params, setParams] = useSearchParams();
  const tabParam = params.get("tab");
  const tab: KnowledgeTab = TABS.some((t) => t.id === tabParam) ? (tabParam as KnowledgeTab) : "documents";
  const readOnlyReview = params.get("view") === "approved";

  const [pageError, setPageError] = useState<string | null>(null);
  const [pendingFolder, setPendingFolder] = useState<{ files: File[]; root: string } | null>(null);
  const [folderName, setFolderName] = useState("");
  const [crystallizing, setCrystallizing] = useState<CrystallizeProgress | null>(null);
  const folderInput = useRef<HTMLInputElement>(null);

  const documents = useQuery({
    queryKey: ["documents", selectedCustomerId],
    queryFn: () => api.listDocuments(selectedCustomerId!) as Promise<{ documents: DocumentItem[] }>,
    enabled: !!selectedCustomerId,
    refetchInterval: 15000,
  });
  const watches = useQuery({
    queryKey: ["watches", selectedCustomerId],
    queryFn: () => api.listWatches(selectedCustomerId!),
    enabled: !!selectedCustomerId,
  });
  const schemas = useQuery({
    queryKey: ["source-schemas", selectedCustomerId],
    queryFn: () => api.listSourceSchemas(selectedCustomerId!),
    enabled: !!selectedCustomerId,
    refetchInterval: 15000,
  });

  const refreshDocs = useCallback(
    () => { qc.invalidateQueries({ queryKey: ["documents", selectedCustomerId] }); },
    [qc, selectedCustomerId],
  );
  const uploads = useUploads(selectedCustomerId, refreshDocs);

  const docs = documents.data?.documents ?? [];
  const queued = docs.filter((d) => d.status === "pending");
  const reviewWaiting = docs.filter((d) => d.status === "review").length;
  const repos = watches.data ?? [];
  const reposAttention = repos.filter((w) => w.sync_state === "attention").length;
  const shapesWaiting = (schemas.data ?? []).filter((s) => s.status === "proposed").length;

  const onPick = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    e.target.value = "";
    if (!files.length) return;
    const firstRel = (files[0] as File & { webkitRelativePath?: string }).webkitRelativePath;
    if (firstRel && firstRel.includes("/")) {
      // Gate D6: a picked folder's root is the source authority
      // (repo://<authority>/<path>); the name is asked inline (4.3).
      const root = firstRel.split("/")[0];
      setPendingFolder({ files, root });
      setFolderName(root);
      return;
    }
    void uploads.start(files, null);
  };

  const crystallize = async () => {
    if (!selectedCustomerId || queued.length === 0 || crystallizing) return;
    setPageError(null);
    let ready = 0;
    const failures: string[] = [];
    for (let i = 0; i < queued.length; i++) {
      const doc = queued[i];
      setCrystallizing({ label: doc.label || "Untitled", done: i, total: queued.length });
      try {
        await api.crystallizeDocument(selectedCustomerId, doc.id);
        ready += 1;
      } catch (err) {
        failures.push(`${doc.label || "Untitled"}: ${errorMessage(err, "could not be read.")}`);
      }
    }
    setCrystallizing(null);
    refreshDocs();
    if (failures.length) setPageError(failures.join(" · "));
    if (ready) toast(`${ready} document${ready === 1 ? "" : "s"} ready for review.`);
  };

  const uploadControls: ReactNode = (
    <>
      <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-md border border-line px-3 py-1.5 text-[13px] font-medium text-fg-2 hover:bg-inset hover:text-fg">
        <Upload aria-hidden className="h-3.5 w-3.5" /> Upload
        <input type="file" className="sr-only" accept={ACCEPT} multiple onChange={onPick} />
      </label>
      <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-md border border-line px-3 py-1.5 text-[13px] font-medium text-fg-2 hover:bg-inset hover:text-fg"
        title="Relative paths become the source identity, which lets imports resolve across files">
        <FolderUp aria-hidden className="h-3.5 w-3.5" /> Upload folder
        <input ref={folderInput} type="file" className="sr-only" multiple onChange={onPick} {...({ webkitdirectory: "" } as Record<string, string>)} />
      </label>
    </>
  );

  if (!selectedCustomerId) {
    return (
      <>
        <PageHeader title="Knowledge" />
        <EmptyState title="No customer selected" help="Pick a customer above to manage their knowledge." />
      </>
    );
  }

  const counts: Record<KnowledgeTab, number> = {
    documents: reviewWaiting, repos: reposAttention, shapes: shapesWaiting, general: 0,
  };
  const meta = `${docs.length} document${docs.length === 1 ? "" : "s"} · ${repos.length} watched repo${repos.length === 1 ? "" : "s"} · ${shapesWaiting} data shape${shapesWaiting === 1 ? "" : "s"} waiting`;
  const setTab = (next: KnowledgeTab) => {
    const p = new URLSearchParams(params);
    if (next === "documents") p.delete("tab"); else p.set("tab", next);
    setParams(p);
  };

  return (
    <div>
      <PageHeader
        title="Knowledge"
        meta={documents.isLoading ? "Loading…" : meta}
        actions={
          <>
            {uploadControls}
            <button type="button" onClick={() => void crystallize()} disabled={queued.length === 0 || !!crystallizing}
              title={queued.length === 0 ? "Nothing is queued. Upload a document first." : undefined}
              className="inline-flex items-center gap-1.5 rounded-md bg-accent px-3 py-1.5 text-[13px] font-medium text-on-accent hover:bg-accent-hover disabled:opacity-40">
              <Zap aria-hidden className="h-3.5 w-3.5" /> {crystallizing ? "Reading…" : `Crystallize ${queued.length} queued`}
            </button>
          </>
        }
      />

      {pageError && <ErrorBanner title="Some documents could not be read" message={pageError} className="mb-3" />}

      {pendingFolder && (
        <form className="mb-4 rounded-lg border border-line bg-surface p-3"
          onSubmit={(e) => {
            e.preventDefault();
            const folder = pendingFolder;
            setPendingFolder(null);
            void uploads.start(folder.files, folderName.trim() || folder.root);
          }}>
          <label htmlFor="folder-source-name" className="block text-[13px] font-medium text-fg">Name this source</label>
          <p className="mt-0.5 text-[12.5px] text-fg-2">
            Imports resolve within it and the name survives a clone rename. {pendingFolder.files.length} files picked.
          </p>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <input id="folder-source-name" value={folderName} onChange={(e) => setFolderName(e.target.value)} autoFocus maxLength={200}
              className="min-w-[200px] flex-1 rounded-md border border-line bg-inset px-2.5 py-1.5 text-[13px] text-fg outline-none focus:border-accent" />
            <button type="button" onClick={() => setPendingFolder(null)}
              className="rounded-md border border-line px-2.5 py-1.5 text-[12.5px] font-medium text-fg-2 hover:bg-inset hover:text-fg">Cancel</button>
            <button type="submit" className="rounded-md bg-accent px-3 py-1.5 text-[12.5px] font-semibold text-on-accent hover:bg-accent-hover">Upload folder</button>
          </div>
        </form>
      )}

      <div role="tablist" aria-label="Knowledge sections" className="mb-4 flex gap-1 overflow-x-auto border-b border-line">
        {TABS.map((t) => {
          const n = counts[t.id];
          const on = tab === t.id;
          return (
            <button key={t.id} type="button" role="tab" aria-selected={on} onClick={() => setTab(t.id)}
              className={cn("-mb-px inline-flex shrink-0 items-center gap-1.5 border-b-2 px-3 py-2 text-[13px]",
                on ? "border-accent font-medium text-accent" : "border-transparent text-fg-2 hover:text-fg")}>
              {t.label}
              {n > 0 && (
                <span className="rounded-full bg-accent px-1.5 py-0.5 text-[11px] font-semibold leading-none text-on-accent tabular-nums" aria-label={`${n} waiting`}>{n}</span>
              )}
            </button>
          );
        })}
      </div>

      {tab === "documents" && (
        <Documents customerId={selectedCustomerId} docs={docs} isLoading={documents.isLoading} error={documents.error}
          onRetry={() => documents.refetch()} uploads={uploads.rows} onCancelUpload={uploads.cancel} onDismissUpload={uploads.dismiss}
          crystallizing={crystallizing} uploadControls={uploadControls}
          onReview={(doc, readOnly) => navigate(`/knowledge/review/${doc.id}${readOnly ? "?view=approved" : ""}`)}
          onViewCrystals={(doc) => navigate(`/?doc=${encodeURIComponent(doc.id)}&from=${encodeURIComponent(doc.label || "Untitled")}`)} />
      )}
      {tab === "repos" && (
        <>
          {DRIVE_WATCHER_ENABLED && <GoogleDrivePanel />}
          <WatchedRepos customerId={selectedCustomerId} />
        </>
      )}
      {tab === "shapes" && <DataShapes customerId={selectedCustomerId} />}
      {tab === "general" && <GeneralBanks customerId={selectedCustomerId} />}

      {docId && (
        <ReviewDrawer customerId={selectedCustomerId} docId={docId} readOnly={readOnlyReview}
          onClose={() => { refreshDocs(); navigate(`/knowledge${tabParam ? `?tab=${tabParam}` : ""}`); }} />
      )}
    </div>
  );
}
