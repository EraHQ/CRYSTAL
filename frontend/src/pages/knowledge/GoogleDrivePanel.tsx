// Google Drive watcher panel — moved verbatim out of KnowledgeManager.tsx
// in Console Refine PR 5 (5.3 file split). Still hidden for launch
// (DRIVE Q1=A, 2026-09-27: drive.readonly is a RESTRICTED scope that
// forces Google's app verification); KnowledgeManager keeps the flag.
import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle, ChevronRight, Cloud, Folder, Loader2, Plus, Trash2, X } from "lucide-react";
import { authedFetch, responseMessage } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";
import { ConfirmInline, CrystalButton } from "@/components/ui";
import type { Watch } from "@/lib/types";



// ── Google Drive (DRIVE-Q1=B, 2026-07-24) ──
// The minimal panel: connect (keyless admin auth-url → Google consent),
// list/disconnect connections, and "Watch a Drive folder" via the
// server-side folder browser (DRIVE-Q2 slice 2, 2026-07-27) — the api
// lists folders with the tenant's own stored credential, so no Google
// token ever reaches this frontend. Choosing a folder registers a
// NORMAL source_watch (scheme=gdrive) the standard watches API + sync
// loop serve.

interface DriveConnection {
  id: string;
  email: string | null;
  status: string;
  last_synced_at: string | null;
  created_at: string;
}

interface DriveFolder { id: string; name: string; }

export function GoogleDrivePanel() {
  const { selectedCustomerId } = useSelectedCustomer();
  const queryClient = useQueryClient();
  const [connecting, setConnecting] = useState(false);
  const [path, setPath] = useState<DriveFolder[]>([
    { id: "root", name: "My Drive" },
  ]);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const connections = useQuery({
    queryKey: ["gdrive-connections", selectedCustomerId],
    queryFn: async (): Promise<{ connections: DriveConnection[] }> => {
      const res = await authedFetch(
        `/admin/api/customers/${encodeURIComponent(selectedCustomerId!)}/gdrive/connections`
      );
      if (!res.ok) return { connections: [] };
      return res.json();
    },
    enabled: !!selectedCustomerId,
  });

  const conns = connections.data?.connections ?? [];
  const activeConn = conns.find((c) => c.status === "active") ?? null;

  // Server-side folder browse: the listing for the CURRENT breadcrumb
  // tail. Cached per parent id, so walking back up is instant.
  const currentParent = path[path.length - 1];
  const folderBrowse = useQuery({
    queryKey: ["gdrive-folders", selectedCustomerId, currentParent.id],
    queryFn: async (): Promise<{ folders: DriveFolder[] }> => {
      const res = await authedFetch(
        `/admin/api/customers/${encodeURIComponent(selectedCustomerId!)}/gdrive/folders?parent=${encodeURIComponent(currentParent.id)}`
      );
      if (!res.ok) throw new Error(`${res.status}`);
      return res.json();
    },
    enabled: !!selectedCustomerId && !!activeConn,
    staleTime: 60_000,
  });
  const browseFolders = folderBrowse.data?.folders ?? [];

  // Watched state, from the SERVER, not from a message banner: the same
  // ["watches"] query WatchedSourcesPanel renders — shared cache key, so
  // a watch created or removed in either panel flips both instantly.
  const watchesQ = useQuery({
    queryKey: ["watches", selectedCustomerId],
    queryFn: async (): Promise<{ watches: Watch[] }> => {
      const res = await authedFetch(
        `/admin/api/watches?customer_id=${encodeURIComponent(selectedCustomerId!)}`
      );
      if (!res.ok) throw new Error(`watches ${res.status}`);
      return res.json();
    },
    enabled: !!selectedCustomerId,
  });
  const watchByFolderId = new Map<string, Watch>(
    (watchesQ.data?.watches ?? [])
      .filter((w) => w.scheme === "gdrive" && !!w.config?.folder_id)
      .map((w): [string, Watch] => [w.config.folder_id as string, w])
  );

  const handleConnect = async () => {
    if (!selectedCustomerId) return;
    setConnecting(true);
    setMessage(null);
    try {
      const res = await authedFetch(
        `/admin/api/customers/${encodeURIComponent(selectedCustomerId)}/gdrive/auth-url`
      );
      if (!res.ok) {
        setMessage(`Could not start Google sign-in (${res.status}). Check the OAuth env vars.`);
        return;
      }
      const data = await res.json();
      window.location.href = data.auth_url;
    } catch {
      setMessage("Failed to reach the API for the Google sign-in URL.");
    } finally {
      setConnecting(false);
    }
  };

  // Refine PR 2 (4.3): disconnect and unwatch ask inline, in this
  // panel, instead of a native confirm dialog. One pending confirmation at a time.
  const [confirm, setConfirm] = useState<
    { kind: "disconnect"; id: string } | { kind: "unwatch"; id: string; name: string } | null
  >(null);

  const handleDisconnect = (connId: string) => setConfirm({ kind: "disconnect", id: connId });

  const doDisconnect = async (connId: string) => {
    if (!selectedCustomerId) return;
    setBusy(true);
    try {
      const res = await authedFetch(
        `/admin/api/customers/${encodeURIComponent(selectedCustomerId)}/gdrive/connections/${encodeURIComponent(connId)}`,
        { method: "DELETE" }
      );
      if (!res.ok) { setMessage(await responseMessage(res, "Disconnect failed")); return; }
      queryClient.invalidateQueries({ queryKey: ["gdrive-connections", selectedCustomerId] });
      queryClient.invalidateQueries({ queryKey: ["watches", selectedCustomerId] });
    } finally {
      setBusy(false);
      setConfirm(null);
    }
  };

  const handleWatchFolder = async (folder: DriveFolder) => {
    if (!selectedCustomerId || !activeConn) return;
    // source_name is the label authority and must be slash-free.
    const name = folder.name.trim().replace(/[/\\]/g, "-") ||
      `folder-${folder.id.slice(0, 6)}`;
    setBusy(true);
    setMessage(null);
    try {
      const res = await authedFetch(
        `/admin/api/watches?customer_id=${encodeURIComponent(selectedCustomerId)}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            scheme: "gdrive",
            source_name: `drive-${name}`,
            config: {
              connection_id: activeConn.id,
              folder_id: folder.id,
              folder_name: name,
            },
          }),
        }
      );
      if (!res.ok) {
        const detail = await res.text();
        setMessage(`Watch creation failed (${res.status}): ${detail.slice(0, 160)}`);
        return;
      }
      setMessage(`Watching Drive folder "${name}" — files land as documents on the next sync.`);
      queryClient.invalidateQueries({ queryKey: ["watches", selectedCustomerId] });
    } finally {
      setBusy(false);
    }
  };

  const handleUnwatchFolder = (watchId: string, name: string) => setConfirm({ kind: "unwatch", id: watchId, name });

  const doUnwatch = async (watchId: string) => {
    if (!selectedCustomerId) return;
    setBusy(true);
    setMessage(null);
    try {
      const res = await authedFetch(
        `/admin/api/watches/${encodeURIComponent(watchId)}?customer_id=${encodeURIComponent(selectedCustomerId)}`,
        { method: "DELETE" }
      );
      if (!res.ok) {
        setMessage(await responseMessage(res, "Unwatch failed"));
        return;
      }
      queryClient.invalidateQueries({ queryKey: ["watches", selectedCustomerId] });
    } finally {
      setBusy(false);
      setConfirm(null);
    }
  };

  return (
    <div>
      <div className="flex items-center gap-2.5 mb-3">
        <Cloud className="h-4 w-4 text-brand-500" />
        <h2 className="text-sm font-semibold text-gray-900">Google Drive</h2>
        {activeConn && (
          <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2 py-0.5 text-[11px] font-medium text-emerald-700">
            <CheckCircle className="h-3 w-3" /> Connected
          </span>
        )}
      </div>
      <ConfirmInline
        open={confirm !== null}
        className="mb-3 mt-0"
        busy={busy}
        message={
          confirm?.kind === "disconnect"
            ? "Disconnect Google Drive? Its folder watches are removed too; crystals stay."
            : confirm?.kind === "unwatch"
              ? `Stop watching "${confirm.name}"? Its crystals stay.`
              : ""
        }
        confirmLabel={confirm?.kind === "disconnect" ? "Disconnect" : "Stop watching"}
        onCancel={() => setConfirm(null)}
        onConfirm={() => {
          if (!confirm) return;
          if (confirm.kind === "disconnect") void doDisconnect(confirm.id);
          else void doUnwatch(confirm.id);
        }}
      />

      {conns.length === 0 ? (
        <div className="rounded-lg border border-dashed border-gray-300 bg-surface p-6 text-center">
          <Cloud className="h-8 w-8 text-gray-300 mx-auto mb-3" />
          <p className="text-sm text-gray-600 mb-1">Connect your Google Drive</p>
          <p className="text-xs text-gray-400 mb-4">Watched folders sync through the same loop as every other source — new and changed files become documents automatically.</p>
          <CrystalButton onClick={handleConnect} disabled={connecting}>
            {connecting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Cloud className="h-4 w-4" />}
            Connect Google Drive
          </CrystalButton>
        </div>
      ) : (
        <div className="rounded-lg border border-gray-200 bg-surface divide-y divide-gray-100">
          {conns.map((c) => (
            <div key={c.id} className="flex items-center justify-between px-4 py-3">
              <div className="flex items-center gap-3 min-w-0">
                <div className="h-8 w-8 rounded-lg bg-blue-50 flex items-center justify-center shrink-0">
                  <Cloud className="h-4 w-4 text-blue-600" />
                </div>
                <div className="min-w-0">
                  <div className="text-sm font-medium text-gray-900 truncate">{c.email || "Google Drive"}</div>
                  <div className="text-xs text-gray-400">
                    {c.status !== "active" ? `Status: ${c.status}` : c.last_synced_at ? `Last synced ${new Date(c.last_synced_at).toLocaleString()}` : "Never synced"}
                  </div>
                </div>
              </div>
              <CrystalButton size="sm" variant="ghost" onClick={() => handleDisconnect(c.id)}>
                <Trash2 className="h-3.5 w-3.5" /> Disconnect
              </CrystalButton>
            </div>
          ))}
          {activeConn && (
            <div className="px-4 py-3">
              <p className="text-xs font-medium text-gray-600 mb-2">Watch a Drive folder</p>
              {/* Breadcrumb */}
              <div className="flex flex-wrap items-center gap-1 mb-2 text-xs">
                {path.map((seg, i) => (
                  <span key={seg.id} className="flex items-center gap-1">
                    {i > 0 && <ChevronRight className="h-3 w-3 text-gray-300" />}
                    <button
                      onClick={() => setPath(path.slice(0, i + 1))}
                      className={i === path.length - 1
                        ? "font-medium text-gray-900"
                        : "text-brand-600 hover:underline"}
                    >
                      {seg.name}
                    </button>
                  </span>
                ))}
                <span className="ml-auto" />
                {(() => {
                  const cw = watchByFolderId.get(currentParent.id);
                  return cw ? (
                    <span className="flex items-center gap-1.5">
                      <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2 py-0.5 text-[11px] font-medium text-emerald-700">
                        <CheckCircle className="h-3 w-3" /> Watching
                      </span>
                      <CrystalButton size="sm" variant="ghost" onClick={() => handleUnwatchFolder(cw.id, currentParent.name)} disabled={busy}>
                        <X className="h-3 w-3" /> Unwatch
                      </CrystalButton>
                    </span>
                  ) : (
                    <CrystalButton size="sm" onClick={() => handleWatchFolder(currentParent)} disabled={busy}>
                      {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : <Plus className="h-3 w-3" />}
                      Watch this folder
                    </CrystalButton>
                  );
                })()}
              </div>
              {/* Folder list */}
              <div className="rounded border border-gray-200 divide-y divide-gray-100 max-h-56 overflow-y-auto">
                {folderBrowse.isLoading ? (
                  <div className="flex items-center gap-2 px-3 py-2.5 text-xs text-gray-400">
                    <Loader2 className="h-3.5 w-3.5 animate-spin" /> Loading folders…
                  </div>
                ) : folderBrowse.isError ? (
                  <div className="px-3 py-2.5 text-xs text-red-600">
                    Could not list folders — try reconnecting Drive.
                  </div>
                ) : browseFolders.length === 0 ? (
                  <div className="px-3 py-2.5 text-xs text-gray-400">
                    No subfolders here.
                  </div>
                ) : (
                  browseFolders.map((f) => {
                    const fw = watchByFolderId.get(f.id);
                    return (
                      <div key={f.id} className="flex items-center justify-between px-3 py-2 hover:bg-gray-50">
                        <button
                          onClick={() => setPath([...path, f])}
                          className="flex items-center gap-2 min-w-0 text-left"
                          title="Open folder"
                        >
                          <Folder className="h-4 w-4 text-amber-500 shrink-0" />
                          <span className="text-xs text-gray-800 truncate">{f.name}</span>
                          {fw && (
                            <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-1.5 py-0.5 text-[10px] font-medium text-emerald-700">
                              <CheckCircle className="h-2.5 w-2.5" /> Watching
                            </span>
                          )}
                        </button>
                        {fw ? (
                          <CrystalButton size="sm" variant="ghost" onClick={() => handleUnwatchFolder(fw.id, f.name)} disabled={busy}>
                            <X className="h-3 w-3" /> Unwatch
                          </CrystalButton>
                        ) : (
                          <CrystalButton size="sm" variant="ghost" onClick={() => handleWatchFolder(f)} disabled={busy}>
                            <Plus className="h-3 w-3" /> Watch
                          </CrystalButton>
                        )}
                      </div>
                    );
                  })
                )}
              </div>
              <p className="mt-1.5 text-[11px] text-gray-400">
                Browsed server-side with your connected account — click a name to open it, Watch to register it as a source.
              </p>
            </div>
          )}
        </div>
      )}

      {message && (
        <div className="mt-2 rounded-lg bg-gray-50 border border-gray-200 px-3 py-2 flex items-center justify-between">
          <p className="text-xs text-gray-600">{message}</p>
          <button onClick={() => setMessage(null)} className="text-xs text-gray-400 hover:text-gray-600 ml-2">×</button>
        </div>
      )}
    </div>
  );
}
