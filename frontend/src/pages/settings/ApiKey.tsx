// Settings › API key (Refine PR 9, 5.8 item 6). The key is shown once at
// creation and stored as a hash, so nothing but its existence can be
// shown (no prefix, created date or last-used tool is stored). Regenerate
// is a TypedConfirm that names which kinds of tools break: key-based
// ones (from the connect registry, narrowed to the tools the user said
// they use at signup when known) lose memory until the new key is
// pasted; signed-in (OAuth) tools keep working.
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Check, Copy, KeyRound } from "lucide-react";
import { api, errorMessage } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { CONNECT_TOOLS } from "@/lib/connect-tools";
import { useSelectedCustomer } from "@/lib/selected-customer";
import { Modal, TypedConfirm, useToast } from "@/components/ui";
import { BTN_PRIMARY, BTN_SECONDARY, Section } from "./shared";

const GENERIC_KEY_TOOLS = ["claude_code", "cursor", "vscode_copilot", "chatgpt"];

// Which tools break and which keep working, in the user's own terms.
export function regenerateImpact(saidTools: string[]): { breaks: string[]; keeps: string[] } {
  const known = CONNECT_TOOLS.filter((t) => saidTools.includes(t.id));
  const pool = known.length > 0 ? known : CONNECT_TOOLS.filter((t) => GENERIC_KEY_TOOLS.includes(t.id) || t.auth === "oauth");
  const breaks = pool.filter((t) => t.auth === "key").map((t) => t.label);
  const keeps = pool.filter((t) => t.auth === "oauth").map((t) => t.label);
  return { breaks, keeps };
}

function list(names: string[]): string {
  if (names.length <= 1) return names.join("");
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

export function ApiKey() {
  const { selectedCustomerId } = useSelectedCustomer();
  const { status: authStatus, reauthProvider, reauthenticate, me } = useAuth();
  const toast = useToast();
  const needsPassword = authStatus === "signedIn" && reauthProvider() === "password";
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | undefined>();
  const [fresh, setFresh] = useState<string | null>(null);
  const [prefix, setPrefix] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const status = useQuery({ queryKey: ["onboarding-status"], queryFn: () => api.onboardingStatus(), enabled: me?.kind === "user" });
  const impact = regenerateImpact(status.data?.ai_tools ?? []);

  const regenerate = async (password?: string) => {
    setBusy(true);
    setError(undefined);
    try {
      if (authStatus === "signedIn") await reauthenticate(needsPassword ? password : undefined);
      const out = await api.rotateApiKey(selectedCustomerId!);
      setFresh(out.api_key);
      setConfirming(false);
      toast("New key created.");
    } catch (e) {
      setError(errorMessage(e, "Couldn't create a new key."));
    } finally {
      setBusy(false);
    }
  };
  const copy = async () => {
    if (!fresh) return;
    try { await navigator.clipboard.writeText(fresh); setCopied(true); setTimeout(() => setCopied(false), 1600); } catch { /* selectable */ }
  };
  const saved = () => {
    if (fresh) setPrefix(fresh.slice(0, 12) + "…");
    setFresh(null);
  };

  return (
    <div className="space-y-4">
      <Section title="Your Crystal API key" help="Authenticates key-based tools (Claude Code, Cursor, VS Code, ChatGPT and others) and the API. The console never needs it.">
        <div className="flex flex-wrap items-center gap-3 rounded-lg border border-line bg-inset px-3 py-2.5 text-[13px]">
          <KeyRound aria-hidden className="h-4 w-4 text-fg-3" />
          {prefix ? (
            <span className="text-fg">New key created just now. Only the prefix is kept here: <code className="font-mono">{prefix}</code></span>
          ) : (
            <span className="text-fg-2">Shown once when the workspace was created and stored as a hash, so it cannot be shown again.</span>
          )}
        </div>
        {!confirming && (
          <button type="button" onClick={() => setConfirming(true)} className={BTN_SECONDARY + " mt-3"}>Regenerate key</button>
        )}
        {confirming && (
          <TypedConfirm word="REGENERATE" confirmLabel="Regenerate key" requirePassword={needsPassword} busy={busy} error={error}
            onConfirm={(pw) => void regenerate(pw)} onCancel={() => { setConfirming(false); setError(undefined); }}
            message={<>
              The current key stops working right away.
              {impact.breaks.length > 0 && <> {list(impact.breaks)} will lose memory until you paste the new key.</>}
              {impact.keeps.length > 0 && <> {list(impact.keeps)} {impact.keeps.length === 1 ? "keeps" : "keep"} working (signed in, no key).</>}
            </>} />
        )}
      </Section>

      <Modal open={!!fresh} onClose={saved} title="Your new key"
        footer={<button type="button" onClick={saved} className={BTN_PRIMARY}>I saved it</button>}>
        <p className="text-[13px] text-fg-2">Copy it now. It is shown only this once.</p>
        <div className="mt-3 flex items-center gap-2 rounded-lg border border-line bg-inset px-3 py-2.5">
          <code className="min-w-0 flex-1 break-all font-mono text-[12.5px] text-fg">{fresh}</code>
          <button type="button" onClick={() => void copy()} aria-label="Copy key" className="shrink-0 rounded-md border border-line p-1.5 text-fg-2 hover:bg-surface hover:text-fg">
            {copied ? <Check aria-hidden className="h-4 w-4 text-ok" /> : <Copy aria-hidden className="h-4 w-4" />}
          </button>
        </div>
        <p className="mt-3 text-[12.5px] text-fg-3">Paste it into every key-based tool. Signed-in tools need nothing.</p>
      </Modal>
    </div>
  );
}
