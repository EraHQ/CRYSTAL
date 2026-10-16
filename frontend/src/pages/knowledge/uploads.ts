// Console Refine PR 5 (5.3 item 3): parallel uploads with a progress row
// per file and Cancel. Three at a time; a plan wall opens the modal from
// the request layer and stops the rest, since they would fail the same
// way; any other failure stays on its row with the cause.
import { useCallback, useRef, useState } from "react";
import { api, errorMessage } from "@/lib/api";
import { explainUploadError } from "./shared";

export interface UploadRow {
  id: string;
  label: string;
  fraction: number;
  state: "queued" | "uploading" | "done" | "failed" | "cancelled";
  error?: string;
}

const PARALLEL = 3;

// Paths never ingested from a folder pick: VCS internals, caches,
// dependency trees, hidden files. The review queue is the real gate.
export function isJunkPath(p: string): boolean {
  return p.split("/").some(
    (seg) => seg.startsWith(".") || seg === "__pycache__" || seg === "node_modules" || seg === "dist" || seg === "build" || seg.endsWith(".egg-info"),
  );
}

export function uploadLabel(file: File, authority: string | null): string | null {
  const rel = (file as File & { webkitRelativePath?: string }).webkitRelativePath;
  if (rel && isJunkPath(rel)) return null;
  if (rel && authority) {
    const parts = rel.split("/");
    parts[0] = authority;
    return parts.join("/");
  }
  return rel && rel.length > 0 ? rel : file.name;
}

export function useUploads(customerId: string | null, onSettled: () => void) {
  const [rows, setRows] = useState<UploadRow[]>([]);
  const controllers = useRef(new Map<string, AbortController>());
  const cancelled = useRef(new Set<string>());
  const seq = useRef(0);

  const patch = useCallback((id: string, p: Partial<UploadRow>) => {
    setRows((rs) => rs.map((r) => (r.id === id ? { ...r, ...p } : r)));
  }, []);

  const start = useCallback(async (files: File[], authority: string | null) => {
    if (!customerId) return;
    const jobs: Array<{ id: string; file: File; label: string }> = [];
    for (const file of files) {
      const label = uploadLabel(file, authority);
      if (!label) continue;
      jobs.push({ id: `u${++seq.current}`, file, label });
    }
    setRows((rs) => [...rs, ...jobs.map((j) => ({ id: j.id, label: j.label, fraction: 0, state: "queued" as const }))]);
    let stop = false;
    let next = 0;
    const worker = async () => {
      while (!stop && next < jobs.length) {
        const job = jobs[next++];
        if (cancelled.current.has(job.id)) continue;
        const ctl = new AbortController();
        controllers.current.set(job.id, ctl);
        patch(job.id, { state: "uploading" });
        const form = new FormData();
        form.append("file", job.file);
        form.append("label", job.label);
        try {
          await api.uploadDocumentFile(customerId, form, {
            signal: ctl.signal,
            onProgress: (f) => patch(job.id, { fraction: f }),
          });
          patch(job.id, { state: "done", fraction: 1 });
        } catch (err) {
          const status = (err as { status?: number }).status;
          if (ctl.signal.aborted) {
            patch(job.id, { state: "cancelled" });
          } else {
            patch(job.id, { state: "failed", error: explainUploadError(errorMessage(err, "Upload failed.")) });
            // A wall (402) stops the batch: the rest would hit the same wall.
            if (status === 402) stop = true;
          }
        } finally {
          controllers.current.delete(job.id);
        }
      }
    };
    await Promise.all(Array.from({ length: Math.min(PARALLEL, jobs.length) }, worker));
    if (stop) {
      setRows((rs) => rs.map((r) => (r.state === "queued" ? { ...r, state: "cancelled" } : r)));
    }
    onSettled();
    // Finished rows leave the list once the document grid has them.
    setTimeout(() => setRows((rs) => rs.filter((r) => r.state !== "done" && r.state !== "cancelled")), 1500);
  }, [customerId, onSettled, patch]);

  const cancel = useCallback((id: string) => {
    cancelled.current.add(id);
    controllers.current.get(id)?.abort();
    setRows((rs) => rs.map((r) => (r.id === id && r.state === "queued" ? { ...r, state: "cancelled" } : r)));
  }, []);

  const dismiss = useCallback((id: string) => setRows((rs) => rs.filter((r) => r.id !== id)), []);

  return { rows, start, cancel, dismiss };
}
