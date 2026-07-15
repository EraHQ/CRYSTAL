// CritiquePanel — read + write critiques pinned to one anatomy node
// (Q2B). The write is the one console write tenants may make; open
// critiques feed the orchestrator on retries and future runs of the
// same trigger, so the composer copy says what it does.
import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, MessageSquarePlus, RotateCcw } from "lucide-react";
import {
  Critique, critiquesUnder, patchCritique, postCritique,
} from "./critiques";

function relTime(ts: string | null): string {
  if (!ts) return "";
  const s = Math.max(0, (Date.now() - new Date(ts).getTime()) / 1000);
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export function CritiquePanel({ envId, targetPath, critiques }: {
  envId: string;
  targetPath: string;
  critiques: Critique[];
}) {
  const [draft, setDraft] = useState("");
  const qc = useQueryClient();
  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ["run-critiques", envId] });
    qc.invalidateQueries({ queryKey: ["cognition-environments"] });
  };
  const create = useMutation({
    mutationFn: () => postCritique(envId, targetPath, draft.trim()),
    onSuccess: () => { setDraft(""); invalidate(); },
  });
  const flip = useMutation({
    mutationFn: ({ id, status }: { id: string; status: "open" | "resolved" }) =>
      patchCritique(id, status),
    onSuccess: invalidate,
  });

  const scoped = critiquesUnder(critiques, targetPath);

  return (
    <div className="space-y-3">
      {scoped.length === 0 && (
        <p className="text-xs text-gray-400">
          No critiques on this part yet. Pin one below — open critiques
          are read by the orchestrator on the next attempt or the next
          run of this task.
        </p>
      )}
      {scoped.map((c) => (
        <div key={c.id}
             className={`rounded border p-2.5 text-xs ${
               c.status === "open"
                 ? "border-amber-200 bg-amber-50"
                 : "border-gray-100 bg-gray-50 opacity-70"}`}>
          <div className="flex items-center gap-2 mb-1">
            <span className="font-mono text-[10px] text-gray-500 bg-white/70 border border-gray-200 rounded px-1">
              {c.target_path}
            </span>
            <span className="text-gray-400">{c.author}</span>
            <span className="text-gray-300">{relTime(c.created_at)}</span>
            <span className="flex-1" />
            {c.status === "open" ? (
              <button
                onClick={() => flip.mutate({ id: c.id, status: "resolved" })}
                className="inline-flex items-center gap-1 text-green-700 hover:text-green-900">
                <CheckCircle2 className="h-3 w-3" /> resolve
              </button>
            ) : (
              <button
                onClick={() => flip.mutate({ id: c.id, status: "open" })}
                className="inline-flex items-center gap-1 text-gray-500 hover:text-gray-700">
                <RotateCcw className="h-3 w-3" /> reopen
              </button>
            )}
          </div>
          <p className="text-gray-700 whitespace-pre-wrap">{c.text}</p>
        </div>
      ))}

      <div className="rounded border border-gray-200 p-2.5">
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          rows={3}
          placeholder={`Critique this ${targetPath === "run" ? "run" : targetPath.replace(/:/g, " ")} — what should be done differently?`}
          className="w-full text-xs text-gray-800 placeholder-gray-300 outline-none resize-y"
        />
        <div className="flex items-center justify-between mt-1">
          <span className="text-[10px] text-gray-400">
            pins to <code className="bg-gray-100 px-1 rounded">{targetPath}</code>
            {" "}· feeds the next orchestrator pass
          </span>
          <button
            onClick={() => create.mutate()}
            disabled={!draft.trim() || create.isPending}
            className="inline-flex items-center gap-1 text-xs px-2 py-1 rounded bg-indigo-600 text-white disabled:opacity-40 hover:bg-indigo-700">
            <MessageSquarePlus className="h-3 w-3" />
            {create.isPending ? "Pinning…" : "Pin critique"}
          </button>
        </div>
        {create.isError && (
          <p className="text-[10px] text-red-600 mt-1">Couldn't save — try again.</p>
        )}
      </div>
    </div>
  );
}
