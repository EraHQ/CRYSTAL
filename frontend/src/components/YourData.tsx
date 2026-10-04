// YourData (2026-10-03; Q34=A export, Q35=A import, Q36=A erase + delete,
// Q37=B seven-day grace with a step-up). Portability and erasure are
// obligations, so the four actions live together on Settings and work
// the same way for every tenant. Erase and Delete re-prove the sign-in
// (password, or a provider popup) exactly like Regenerate key, then
// require the typed word; the server enforces both again.
import { useRef, useState } from "react";
import { Download, Upload, Trash2, UserX, Loader2, RotateCcw } from "lucide-react";
import { api, authedFetch } from "@/lib/api";
import { useAuth } from "@/lib/auth";

type Busy = "export" | "import" | "erase" | "delete" | "restore" | null;

function Card({
  icon: Icon, title, body, children, danger,
}: {
  icon: typeof Download; title: string; body: string; children: React.ReactNode; danger?: boolean;
}) {
  return (
    <div className={`rounded-lg border p-4 ${danger ? "border-red-200 bg-red-50/30" : "border-gray-200 bg-white"}`}>
      <div className="flex items-start gap-3">
        <div className={`mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${danger ? "bg-red-100" : "bg-gray-100"}`}>
          <Icon className={`h-4 w-4 ${danger ? "text-red-600" : "text-gray-600"}`} />
        </div>
        <div className="min-w-0 flex-1">
          <h3 className="text-[13.5px] font-semibold text-gray-900">{title}</h3>
          <p className="mt-0.5 text-[12.5px] text-gray-600">{body}</p>
          <div className="mt-3">{children}</div>
        </div>
      </div>
    </div>
  );
}

export function YourData({ customerId }: { customerId: string }) {
  const { status: authStatus, reauthProvider, reauthenticate, me, refreshMe } = useAuth();
  const needsPassword = authStatus === "signedIn" && reauthProvider() === "password";
  const [busy, setBusy] = useState<Busy>(null);
  const [note, setNote] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirm, setConfirm] = useState<"erase" | "delete" | null>(null);
  const [word, setWord] = useState("");
  const [password, setPassword] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const deleting = !!me?.deletion_scheduled_at;

  const run = async (name: Busy, fn: () => Promise<string>) => {
    setBusy(name);
    setError(null);
    setNote(null);
    try {
      setNote(await fn());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setBusy(null);
    }
  };

  const stepUp = async () => {
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

  const erase = () =>
    run("erase", async () => {
      await stepUp();
      const out = await api.eraseMyMemories();
      setConfirm(null);
      setWord("");
      setPassword("");
      const n = Object.values(out.rows ?? {}).reduce((a: number, b: any) => a + (b as number), 0);
      await refreshMe();
      return `All memories erased (${n} rows). Your account and plan are unchanged.`;
    });

  const del = () =>
    run("delete", async () => {
      await stepUp();
      const out = await api.deleteMyAccount();
      setConfirm(null);
      setWord("");
      setPassword("");
      await refreshMe();
      const when = out.purge_after ? new Date(out.purge_after).toLocaleDateString() : "seven days";
      return `Your account is scheduled for deletion on ${when}. You can restore it until then.`;
    });

  const restore = () =>
    run("restore", async () => {
      await stepUp();
      await api.restoreMyAccount();
      await refreshMe();
      return "Your account is restored. Regenerate your API key above and reconnect your AI tools.";
    });

  const expectedWord = confirm === "erase" ? "ERASE" : "DELETE";
  const confirmReady = word.trim() === expectedWord && (!needsPassword || password.length > 0);

  return (
    <section className="rounded-xl border border-gray-200 bg-white p-5">
      <h2 className="text-[15px] font-semibold text-gray-900">Your data</h2>
      <p className="mt-1 text-[12.5px] text-gray-600">
        Take your memories with you, bring them back, or remove them. Everything
        here is yours to do at any time.
      </p>

      {deleting && (
        <div className="mt-4 rounded-lg border border-amber-300 bg-amber-50 p-4">
          <p className="text-[13px] font-semibold text-amber-900">This account is scheduled for deletion.</p>
          <p className="mt-1 text-[12.5px] text-amber-800">
            Everything is erased on {me?.purge_after ? new Date(me.purge_after).toLocaleDateString() : "the scheduled date"}.
            Until then you can export your data or restore the account.
          </p>
          <button
            disabled={busy !== null}
            onClick={() => void restore()}
            className="mt-3 inline-flex items-center gap-1.5 rounded-lg bg-amber-600 px-3.5 py-2 text-[12.5px] font-semibold text-white hover:bg-amber-500 disabled:opacity-50"
          >
            {busy === "restore" ? <Loader2 className="h-4 w-4 animate-spin" /> : <RotateCcw className="h-4 w-4" />}
            Restore my account
          </button>
        </div>
      )}

      <div className="mt-4 grid gap-3 md:grid-cols-2">
        <Card icon={Download} title="Export" body="Download everything stored in your bank as one JSON file, in the exact format Import accepts. Includes pending assumptions and their threads.">
          <button
            disabled={busy !== null}
            onClick={() => void exportBank()}
            className="inline-flex items-center gap-1.5 rounded-lg border border-gray-300 px-3.5 py-2 text-[12.5px] font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50"
          >
            {busy === "export" ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
            Download my data
          </button>
        </Card>

        <Card icon={Upload} title="Import" body="Restore a previous export into this bank. Memories already present are kept and skipped; nothing is overwritten.">
          <input
            ref={fileRef}
            type="file"
            accept="application/json,.json"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void importBank(f);
              e.target.value = "";
            }}
          />
          <button
            disabled={busy !== null || deleting}
            onClick={() => fileRef.current?.click()}
            className="inline-flex items-center gap-1.5 rounded-lg border border-gray-300 px-3.5 py-2 text-[12.5px] font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50"
          >
            {busy === "import" ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
            Choose an export file
          </button>
        </Card>

        <Card danger icon={Trash2} title="Erase all memories" body="Delete every memory in this bank right now. Your account, plan, API key and connections stay. Export first if you might want them back.">
          {confirm === "erase" ? null : (
            <button
              disabled={busy !== null || deleting}
              onClick={() => { setConfirm("erase"); setWord(""); setPassword(""); }}
              className="rounded-lg border border-red-300 px-3.5 py-2 text-[12.5px] font-medium text-red-700 hover:bg-red-50 disabled:opacity-50"
            >
              Erase all memories
            </button>
          )}
        </Card>

        <Card danger icon={UserX} title="Delete my account" body="Cancels your subscription, revokes your API key and connections, and erases everything after seven days. You can restore the account during those seven days.">
          {confirm === "delete" ? null : (
            <button
              disabled={busy !== null || deleting}
              onClick={() => { setConfirm("delete"); setWord(""); setPassword(""); }}
              className="rounded-lg bg-red-600 px-3.5 py-2 text-[12.5px] font-semibold text-white hover:bg-red-500 disabled:opacity-50"
            >
              Delete my account
            </button>
          )}
        </Card>
      </div>

      {confirm && (
        <div className="mt-4 rounded-lg border border-red-300 bg-red-50 p-4">
          <p className="text-[13px] font-semibold text-red-800">
            {confirm === "erase" ? "Erase every memory in this bank?" : "Delete this account?"}
          </p>
          <p className="mt-1 text-[12.5px] text-red-700">
            {confirm === "erase"
              ? "This cannot be undone. Type ERASE to confirm."
              : "Your subscription is cancelled now and everything is erased in seven days unless you restore. Type DELETE to confirm."}
            {authStatus === "signedIn" && " You'll verify your sign-in first."}
          </p>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <input
              value={word}
              onChange={(e) => setWord(e.target.value)}
              placeholder={expectedWord}
              className="w-36 rounded-lg border border-gray-300 px-3 py-2 text-[12.5px] outline-none focus:border-red-400"
            />
            {needsPassword && (
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Your password"
                className="w-40 rounded-lg border border-gray-300 px-3 py-2 text-[12.5px] outline-none focus:border-red-400"
              />
            )}
            <button
              disabled={busy !== null || !confirmReady}
              onClick={() => void (confirm === "erase" ? erase() : del())}
              className="inline-flex items-center gap-1.5 rounded-lg bg-red-600 px-3.5 py-2 text-[12.5px] font-semibold text-white hover:bg-red-500 disabled:opacity-50"
            >
              {busy === confirm ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
              {confirm === "erase" ? "Yes, erase everything" : "Yes, delete my account"}
            </button>
            <button
              disabled={busy !== null}
              onClick={() => { setConfirm(null); setWord(""); setPassword(""); }}
              className="rounded-lg border border-gray-300 px-3.5 py-2 text-[12.5px] font-medium text-gray-600 hover:bg-gray-50"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {(note || error) && (
        <p className={`mt-4 rounded-lg px-3 py-2 text-[12.5px] ${error ? "bg-red-50 text-red-600" : "bg-emerald-50 text-emerald-700"}`}>
          {error ?? note}
        </p>
      )}
    </section>
  );
}
