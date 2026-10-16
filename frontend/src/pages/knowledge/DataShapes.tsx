// Knowledge › Data shapes (Console Refine PR 5, 5.3 item 8; Gate G3).
// One approval per shape of structured file; every later file of the
// same shape is saved without AI calls. Roles carry plain labels, the
// preview shows records through the mapping, Reject asks inline.
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ChevronDown, Table2 } from "lucide-react";
import { api, errorMessage } from "@/lib/api";
import { ConfirmInline, EmptyState, ErrorBanner, Skeleton, StatusPill } from "@/components/ui";
import type { SchemaMapping, SourceSchemaItem } from "@/lib/types";
import { cn } from "@/lib/utils";

const ROLES: Array<{ value: string; label: string }> = [
  { value: "key", label: "Key (what it is about)" },
  { value: "value", label: "Value (the memory)" },
  { value: "locator", label: "Locator" },
  { value: "timestamp", label: "Timestamp" },
  { value: "skip", label: "Skip" },
];

const SHAPE_STATUS: Record<string, { label: string; variant: "warn" | "ok" | "neutral" }> = {
  proposed: { label: "Waiting for approval", variant: "warn" },
  approved: { label: "Approved", variant: "ok" },
  rejected: { label: "Rejected", variant: "neutral" },
};

export function DataShapes({ customerId }: { customerId: string }) {
  const schemas = useQuery({
    queryKey: ["source-schemas", customerId],
    queryFn: () => api.listSourceSchemas(customerId),
    refetchInterval: 15000,
  });
  const rows = schemas.data ?? [];
  const proposed = rows.filter((s) => s.status === "proposed");
  const settled = rows.filter((s) => s.status !== "proposed");
  return (
    <div>
      <p className="mb-3 text-[12.5px] text-fg-2">
        Approve how a kind of file is read once. Every later file of the same shape is saved without AI calls.
      </p>
      {schemas.isLoading && <Skeleton rows={3} />}
      {schemas.isError && <ErrorBanner title="Couldn't load data shapes" error={schemas.error} onRetry={() => schemas.refetch()} />}
      {!schemas.isLoading && !schemas.isError && rows.length === 0 && (
        <EmptyState icon={Table2} title="No data shapes yet"
          help="Upload a CSV, JSON or spreadsheet and its shape appears here for a one-time approval." />
      )}
      <div className="space-y-2">
        {[...proposed, ...settled].map((s) => <ShapeCard key={s.id} s={s} customerId={customerId} />)}
      </div>
    </div>
  );
}

function ShapeCard({ s, customerId }: { s: SourceSchemaItem; customerId: string }) {
  const qc = useQueryClient();
  const [expanded, setExpanded] = useState(s.status === "proposed");
  const [roles, setRoles] = useState<Record<string, string>>(s.mapping?.roles ?? {});
  const [dirty, setDirty] = useState(false);
  const [confirmReject, setConfirmReject] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const mapping: SchemaMapping = { version: 1, roles, subject: s.mapping?.subject ?? null, domain: s.mapping?.domain ?? "General" };
  const preview = useQuery({
    queryKey: ["schema-preview", s.id, JSON.stringify(roles)],
    enabled: expanded && Object.keys(roles).length > 0 && s.sample.length > 0,
    queryFn: () => api.previewSourceSchema(customerId, { mapping, sample: s.sample, label: s.label ?? "" }),
  });
  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["source-schemas", customerId] });
    qc.invalidateQueries({ queryKey: ["documents", customerId] });
    qc.invalidateQueries({ queryKey: ["watches", customerId] });
  };
  const decide = useMutation({
    mutationFn: (action: "approve" | "reject") => api.decideSourceSchema(customerId, s.id, action),
    onSuccess: () => { setError(null); setConfirmReject(false); refresh(); },
    onError: (e) => { setConfirmReject(false); setError(errorMessage(e, "That didn't go through.")); },
  });
  const save = useMutation({
    mutationFn: () => api.saveSourceSchemaMapping(customerId, s.id, mapping),
    onSuccess: () => { setError(null); setDirty(false); refresh(); },
    onError: (e) => setError(errorMessage(e, "Couldn't save the mapping.")),
  });

  const status = SHAPE_STATUS[s.status] ?? { label: s.status, variant: "neutral" as const };
  const items = preview.data?.items ?? [];
  const busy = decide.isPending || save.isPending;
  const panelId = `shape-${s.id}`;

  return (
    <article className="rounded-lg border border-line bg-surface">
      <button type="button" onClick={() => setExpanded((v) => !v)} aria-expanded={expanded} aria-controls={panelId}
        className="flex w-full flex-wrap items-center gap-2 px-3 py-2 text-left">
        <Table2 aria-hidden className="h-3.5 w-3.5 shrink-0 text-fg-3" />
        <span className="min-w-0 flex-1 truncate font-mono text-[12.5px] text-fg" title={s.schema_hash}>{s.label ?? "(unnamed shape)"}</span>
        <StatusPill variant={status.variant} raw={s.status}>{status.label}</StatusPill>
        {s.parked_count > 0 && <span className="text-[12px] text-warn">{s.parked_count} file{s.parked_count === 1 ? "" : "s"} waiting</span>}
        <ChevronDown aria-hidden className={cn("h-3.5 w-3.5 text-fg-3 transition-transform", expanded && "rotate-180")} />
      </button>
      {expanded && (
        <div id={panelId} className="border-t border-line px-3 py-3">
          <div className="grid gap-4 min-[720px]:grid-cols-2">
            <div>
              <p className="mb-1.5 text-[11px] font-medium uppercase tracking-[0.08em] text-fg-3">What each field means</p>
              {Object.keys(roles).length === 0 && (
                <p className="text-[12.5px] text-fg-3">No mapping was proposed (no model was available when the file arrived). Reject this shape, or upload the file again once a model is set in Settings.</p>
              )}
              <div className="space-y-1">
                {Object.entries(roles).map(([path, role]) => (
                  <label key={path} className="flex items-center gap-2 text-[12.5px]">
                    <span className="min-w-0 flex-1 truncate font-mono text-fg-2" title={path}>{path}</span>
                    <select value={role} disabled={s.status === "rejected"}
                      onChange={(e) => { setRoles((r) => ({ ...r, [path]: e.target.value })); setDirty(true); }}
                      className="rounded-md border border-line bg-inset px-1.5 py-0.5 text-[12px] text-fg">
                      {ROLES.map((r) => <option key={r.value} value={r.value}>{r.label}</option>)}
                    </select>
                  </label>
                ))}
              </div>
              <p className="mt-2 text-[11.5px] text-fg-3">
                Subject: <span className="font-mono">{s.mapping?.subject ?? "(none)"}</span> · Domain: {s.mapping?.domain ?? "General"}
              </p>
            </div>
            <div>
              <p className="mb-1.5 text-[11px] font-medium uppercase tracking-[0.08em] text-fg-3">One record, read this way</p>
              {preview.isFetching && <p className="text-[12.5px] text-fg-3">Rendering…</p>}
              {!preview.isFetching && items.length === 0 && (
                <p className="text-[12.5px] text-fg-3">No memory comes out of this mapping. Mark at least one field as the value.</p>
              )}
              {items.slice(0, 1).map((it, i) => (
                <div key={i} className="rounded-md border border-line bg-inset px-2.5 py-2">
                  <p className="text-[12.5px] font-medium text-fg">{it.key}</p>
                  <p className="mt-0.5 whitespace-pre-wrap break-words text-[12.5px] text-fg-2">{it.value}</p>
                  <p className="mt-1 truncate font-mono text-[11px] text-fg-3" title={it.sparse_key}>{it.sparse_key}</p>
                </div>
              ))}
              {items.length > 1 && <p className="mt-1 text-[11.5px] text-fg-3">{items.length - 1} more in the sample read the same way.</p>}
            </div>
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            {s.status === "proposed" && (
              <button type="button" disabled={busy} onClick={() => decide.mutate("approve")}
                className="rounded-md bg-accent px-3 py-1.5 text-[12.5px] font-medium text-on-accent hover:bg-accent-hover disabled:opacity-50">
                {s.parked_count > 0 ? `Approve and save ${s.parked_count}` : "Approve"}
              </button>
            )}
            {s.status === "proposed" && (
              <button type="button" disabled={busy || confirmReject} onClick={() => setConfirmReject(true)}
                className="rounded-md border border-line px-3 py-1.5 text-[12.5px] font-medium text-fg-2 hover:bg-inset hover:text-fg disabled:opacity-50">
                Reject shape
              </button>
            )}
            {dirty && (
              <button type="button" disabled={busy} onClick={() => save.mutate()}
                className="rounded-md border border-accent/40 bg-accent-soft px-3 py-1.5 text-[12.5px] font-medium text-accent hover:bg-accent-soft/70 disabled:opacity-50">
                Save mapping
              </button>
            )}
            <span className="ml-auto text-[11.5px] text-fg-3">Changes apply to files that arrive from now on.</span>
          </div>
          <ConfirmInline open={confirmReject} busy={decide.isPending}
            message={`Reject this shape? ${s.parked_count > 0 ? `The ${s.parked_count} waiting file${s.parked_count === 1 ? "" : "s"} stay parked for good.` : "Files of this shape will not be read."}`}
            confirmLabel="Reject shape" onCancel={() => setConfirmReject(false)} onConfirm={() => decide.mutate("reject")} />
          {error && <ErrorBanner title="Data shape" message={error} className="mt-2" />}
        </div>
      )}
    </article>
  );
}
