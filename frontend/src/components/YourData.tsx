// YourData — Settings › Your data (Refine PR 9, 5.8 item 8; the actions
// are from 2026-10-03: Q34=A export, Q35=A import, Q36=A erase + delete,
// Q37=B seven-day grace with a step-up). Export and Import first, then a
// Danger zone where Erase and Delete each use TypedConfirm (the word plus
// the password when the sign-in is password-based); the server enforces
// both again. The restore banner is readable on bg-warn-soft.
import { useRef, useState } from "react";
import { Download, Loader2, RotateCcw, Trash2, Upload, UserX } from "lucide-react";
import { api, authedFetch, errorMessage } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { TypedConfirm, useToast } from "@/components/ui";
import { BTN_SECONDARY, Section } from "@/pages/settings/shared";

type Busy = "export" | "import" | "erase" | "delete" | "restore" | null;

function Card({ icon: Icon, title, body, children, danger }: {
  icon: typeof Download; title: string; body: string; children: React.ReactNode; danger?: boolean;
}) {
  return (
    <div className={`rounded-lg border p-4 ${danger ? "border-bad/40 bg-bad-soft/40" : "border-line bg-inset"}`}>
      <div className="flex items-start gap-3">
        <div className={`mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${danger ? "bg-bad-soft text-bad" : "bg-surface text-fg-2"}`}>
          <Icon aria-hidden className="h-4 w-4" />
        </div>
        <div className="min-w-0 flex-1">
          <h3 className="text-[13.5px] font-semibold text-fg">{title}</h3>
          <p className="mt-0.5 text-[12.5px] text-fg-2">{body}</p>
          <div className="mt-3">{children}</div>
        </div>
      </div>
    </div>
  );
}

export function YourData({ customerId }: { customerId: string }) {
  const { status: authStatus, reauthProvider, reauthenticate, me, refreshMe } = useAuth();
  const toast = useToast();
  const needsPassword = authStatus === "signedIn" && reauthProvider() === "password";
  const [busy, setBusy] = useState<Busy>(null);
  const [note, setNote] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirm, setConfirm] = useState<"erase" | "delete" | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const deleting = !!me?.deletion_scheduled_at;
  const facts = me?.usage?.facts_used ?? null;

  const run = async (name: Busy, fn: () => Promise<string>) => {
    setBusy(name);
    setError(null);
    setNote(null);
    try {
      const out = await fn();
      setNote(out);
      toast(out);
    } catch (e) {
      setError(errorMessage(e, "Something went wrong."));
    } finally {
      setBusy(null);
    }
  };

  const stepUp = async (password?: string) => {
    if (authStatus === "signedIn") {
      await reauthenticate(needsPassword ? password : undefined);
    }
  };

  const exportBank = () =>
    run("export", async () => {
      const res = await authedFetch("/v1/export/topology", { method: "POST" });
      if (!res.ok) throw new Error(`Export failed (${res.status})`);
      const json = await res.json();
      const blob = new Blob([JSON.stringify(json, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      const day = new Date().toISOString().slice(0, 10);
      a.href = url;
      a.download = `crystal-cache-${customerId}-${day}.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      return `Exported ${json.crystal_count} crystals and ${json.fact_count} facts.`;
    });

  const importBank = (file: File) =>
    run("import", async () => {
      const text = await file.text();
      let parsed: any;
      try {
        parsed = JSON.parse(text);
      } catch {
        throw new Error("That file is not a Crystal Cache export.");
      }
      const data = parsed?.data && typeof parsed.data === "object" ? parsed.data : parsed;
      if (!data || !Array.isArray(data.crystals)) {
        throw new Error("That file is not a Crystal Cache export.");
      }
      // A real bank's export is hundreds of MB (three dense 10,000-dim
      // vectors per crystal, a 768-dim embedding per fact) and still
      // ~90 MB gzipped, past the 32 MiB request limit at the edge. So it
      // travels in parts: crystals with their facts, sized by bytes, then
      // chains, edges, conflicts and citations. The server restores each
      // part against what is already in the bank.
      const PART_BYTES = 16 * 1024 * 1024;
      const factsByCrystal = new Map<string, any[]>();
      for (const f of data.facts ?? []) {
        const k = f.crystal_id;
        if (!factsByCrystal.has(k)) factsByCrystal.set(k, []);
        factsByCrystal.get(k)!.push(f);
      }
      const parts: any[] = [];
      let cur = { crystals: [] as any[], facts: [] as any[] };
      let curBytes = 0;
      const flush = () => {
        if (cur.crystals.length) parts.push({ format: data.format, ...cur, chains: [], edges: [], conflicts: [], citations: [] });
        cur = { crystals: [], facts: [] };
        curBytes = 0;
      };
      for (const c of data.crystals) {
        const facts = factsByCrystal.get(c.id) ?? [];
        const bytes = JSON.stringify(c).length + facts.reduce((n, f) => n + JSON.stringify(f).length, 0);
        if (curBytes + bytes > PART_BYTES && cur.crystals.length) flush();
        cur.crystals.push(c);
        cur.facts.push(...facts);
        curBytes += bytes;
      }
      flush();
      const chunk = (rows: any[], n: number) => {
        const out: any[][] = [];
        for (let i = 0; i < rows.length; i += n) out.push(rows.slice(i, i + n));
        return out;
      };
      for (const rows of chunk(data.chains ?? [], 5000)) parts.push({ format: data.format, crystals: [], facts: [], chains: rows, edges: [], conflicts: [], citations: [] });
      for (const rows of chunk(data.edges ?? [], 5000)) parts.push({ format: data.format, crystals: [], facts: [], chains: [], edges: rows, conflicts: [], citations: [] });
      if ((data.conflicts ?? []).length || (data.citations ?? []).length) {
        parts.push({ format: data.format, crystals: [], facts: [], chains: [], edges: [], conflicts: data.conflicts ?? [], citations: data.citations ?? [] });
      }

      const totals: Record<string, number> = {};
      for (let i = 0; i < parts.length; i++) {
        setNote(`Importing part ${i + 1} of ${parts.length}...`);
        const partText = JSON.stringify(parts[i]);
        let body: BodyInit = partText;
        const headers: Record<string, string> = { "Content-Type": "application/json" };
        if (typeof CompressionStream !== "undefined") {
          const stream = new Blob([partText]).stream().pipeThrough(new CompressionStream("gzip"));
          body = await new Response(stream).blob();
          headers["Content-Encoding"] = "gzip";
        }
        const res = await authedFetch("/v1/import/topology", { method: "POST", headers, body });
        const resBody = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(resBody?.detail ?? `Import failed on part ${i + 1} of ${parts.length} (${res.status})`);
        const c = resBody?.counts ?? resBody;
        for (const [k, v] of Object.entries(c)) {
          if (typeof v === "number") totals[k] = (totals[k] ?? 0) + v;
        }
      }
      return `Imported ${totals.crystals ?? 0} crystals, ${totals.facts ?? 0} facts, ${totals.chains ?? 0} chains, ${totals.edges ?? 0} edges` +
        (totals.skipped_collisions ? ` (${totals.skipped_collisions} already present)` : "") + ` in ${parts.length} part${parts.length === 1 ? "" : "s"}.`;
    });

  const erase = (password?: string) =>
    run("erase", async () => {
      await stepUp(password);
      const out = await api.eraseMyMemories();
      setConfirm(null);
      const n = Object.values(out.rows ?? {}).reduce((a: number, b: unknown) => a + (b as number), 0);
      await refreshMe();
      return `All memories erased (${n} rows). Your account and plan are unchanged.`;
    });

  const del = (password?: string) =>
    run("delete", async () => {
      await stepUp(password);
      const out = await api.deleteMyAccount();
      setConfirm(null);
      await refreshMe();
      const when = out.purge_after ? new Date(out.purge_after).toLocaleDateString() : "seven days";
      return `Your account is scheduled for deletion on ${when}. You can restore it until then.`;
    });

  const restore = () =>
    run("restore", async () => {
      await stepUp();
      await api.restoreMyAccount();
      await refreshMe();
      return "Your account is restored. Regenerate your API key and reconnect your tools.";
    });

  return (
    <div className="space-y-4">
      {deleting && (
        <div role="status" className="rounded-lg border border-warn/40 bg-warn-soft p-4">
          <p className="text-[13px] font-semibold text-fg">This account is scheduled for deletion.</p>
          <p className="mt-1 text-[12.5px] text-fg-2">
            Everything is erased on {me?.purge_after ? new Date(me.purge_after).toLocaleDateString() : "the scheduled date"}. Until then you can export your data or restore the account.
          </p>
          <button type="button" disabled={busy !== null} onClick={() => void restore()}
            className="mt-3 inline-flex items-center gap-1.5 rounded-md bg-warn px-3 py-1.5 text-[13px] font-medium text-canvas hover:bg-warn/90 disabled:opacity-50">
            {busy === "restore" ? <Loader2 aria-hidden className="h-4 w-4 animate-spin" /> : <RotateCcw aria-hidden className="h-4 w-4" />}
            Restore my account
          </button>
        </div>
      )}

      <Section title="Your data" help="Take your memories with you, bring them back, or remove them. Everything here is yours to do at any time.">
        <div className="grid gap-3 min-[720px]:grid-cols-2">
          <Card icon={Download} title="Export" body="Everything in your bank as one JSON file, in the format Import accepts. Includes pending assumptions and their threads.">
            <button type="button" disabled={busy !== null} onClick={() => void exportBank()} className={BTN_SECONDARY}>
              {busy === "export" ? <Loader2 aria-hidden className="h-4 w-4 animate-spin" /> : <Download aria-hidden className="h-4 w-4" />}
              Download my data
            </button>
          </Card>
          <Card icon={Upload} title="Import" body="Restore from a Crystal export. Duplicates are merged, not doubled; nothing already here is overwritten.">
            <input ref={fileRef} type="file" accept="application/json,.json" className="hidden" aria-label="Export file"
              onChange={(e) => { const f = e.target.files?.[0]; if (f) void importBank(f); e.target.value = ""; }} />
            <button type="button" disabled={busy !== null || deleting} onClick={() => fileRef.current?.click()} className={BTN_SECONDARY}>
              {busy === "import" ? <Loader2 aria-hidden className="h-4 w-4 animate-spin" /> : <Upload aria-hidden className="h-4 w-4" />}
              Choose an export file
            </button>
          </Card>
        </div>
        {(note || error) && (
          <p role="status" className={`mt-3 rounded-lg px-3 py-2 text-[12.5px] ${error ? "bg-bad-soft text-bad" : "bg-ok-soft text-ok"}`}>{error ?? note}</p>
        )}
      </Section>

      <Section title="Danger zone" danger help="Both ask for the typed word and your sign-in again. Export first if you might want anything back.">
        <div className="grid gap-3 min-[720px]:grid-cols-2">
          <Card danger icon={Trash2} title="Erase all memories" body={`Removes all ${facts != null ? facts.toLocaleString() : "your"} facts. Connections, settings and your account stay.`}>
            {confirm !== "erase" && (
              <button type="button" disabled={busy !== null || deleting} onClick={() => { setConfirm("erase"); setError(null); }}
                className="rounded-md border border-bad/60 px-3 py-1.5 text-[13px] font-medium text-bad hover:bg-bad-soft disabled:opacity-50">
                Erase all memories
              </button>
            )}
          </Card>
          <Card danger icon={UserX} title="Delete my account" body="Erases memories and closes the workspace after 7 days. You can restore it until then.">
            {confirm !== "delete" && (
              <button type="button" disabled={busy !== null || deleting} onClick={() => { setConfirm("delete"); setError(null); }}
                className="rounded-md bg-bad px-3 py-1.5 text-[13px] font-medium text-canvas hover:bg-bad/90 disabled:opacity-50">
                Delete my account
              </button>
            )}
          </Card>
        </div>
        {confirm === "erase" && (
          <TypedConfirm word="ERASE" confirmLabel="Erase everything" requirePassword={needsPassword} busy={busy === "erase"} error={error ?? undefined}
            onConfirm={(pw) => void erase(pw)} onCancel={() => { setConfirm(null); setError(null); }}
            message={`Erase every memory in this bank${facts != null ? ` (${facts.toLocaleString()} facts)` : ""}? This cannot be undone.`} />
        )}
        {confirm === "delete" && (
          <TypedConfirm word="DELETE" confirmLabel="Delete my account" requirePassword={needsPassword} busy={busy === "delete"} error={error ?? undefined}
            onConfirm={(pw) => void del(pw)} onCancel={() => { setConfirm(null); setError(null); }}
            message="Your subscription is cancelled now and everything is erased in seven days unless you restore the account." />
        )}
      </Section>
    </div>
  );
}
