import { Fragment, useEffect, useRef, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  ArrowUp, Check, ChevronRight, Copy, Download, FileText, Gem, Key, Loader2,
  RotateCcw, Settings2, ThumbsDown, ThumbsUp, X, Zap,
} from "lucide-react";
import { api, ApiError } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";
import { useAuth } from "@/lib/auth";
import { CrystalButton } from "@/components/ui";
import type { AgentRunResponse, AgentToolCall, DocumentArtifact } from "@/lib/types";
import { cn, fmtNum } from "@/lib/utils";

interface PlaygroundTurn {
  id: string; user: string; assistant: string | null; result: AgentRunResponse | null;
  error: string | null; sentAt: number; feedback?: "up" | "down"; learnResult?: { crystals_written: number };
}

const SUGGESTIONS = [
  "What do you know about this project?",
  "Summarize the most recent document I uploaded",
  "Where is the sparse key format defined?",
];

// ── Markdown-lite ──────────────────────────────────────────────────
// Deliberately small: fenced code blocks (with copy), inline `code`,
// and **bold**. Everything else renders as the model wrote it. No
// parser dependency, no HTML injection — plain React nodes only.

function CodeBlock({ lang, code }: { lang: string; code: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try { await navigator.clipboard.writeText(code); setCopied(true); setTimeout(() => setCopied(false), 1600); } catch {}
  };
  return (
    <div className="chat-code group/code relative my-2 overflow-hidden rounded-lg">
      <div className="flex items-center justify-between border-b border-gray-200 px-3 py-1.5">
        <span className="text-[10px] font-medium uppercase tracking-wider text-gray-400">{lang || "code"}</span>
        <button onClick={copy} className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium text-gray-400 transition-colors hover:bg-gray-100 hover:text-gray-700">
          {copied ? <><Check className="h-3 w-3 text-emerald-400" /> Copied</> : <><Copy className="h-3 w-3" /> Copy</>}
        </button>
      </div>
      <pre className="overflow-x-auto px-3 py-2.5 text-gray-700">{code}</pre>
    </div>
  );
}

function InlineMd({ text }: { text: string }) {
  // `code` first, then **bold** inside the plain spans.
  const parts = text.split(/(`[^`\n]+`)/g);
  return (
    <>
      {parts.map((part, i) => {
        if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
          return <code key={i} className="chat-inline-code">{part.slice(1, -1)}</code>;
        }
        const boldParts = part.split(/(\*\*[^*]+\*\*)/g);
        return (
          <Fragment key={i}>
            {boldParts.map((b, j) =>
              b.startsWith("**") && b.endsWith("**") && b.length > 4
                ? <strong key={j} className="font-semibold text-gray-900">{b.slice(2, -2)}</strong>
                : <Fragment key={j}>{b}</Fragment>
            )}
          </Fragment>
        );
      })}
    </>
  );
}

function Markdown({ text }: { text: string }) {
  const segments = text.split(/```(\w*)\n?([\s\S]*?)```/g);
  // split with two capture groups yields [text, lang, code, text, ...]
  const nodes: React.ReactNode[] = [];
  for (let i = 0; i < segments.length; i += 3) {
    const plain = segments[i];
    if (plain) nodes.push(<span key={`t${i}`} className="whitespace-pre-wrap"><InlineMd text={plain} /></span>);
    if (i + 2 < segments.length) {
      nodes.push(<CodeBlock key={`c${i}`} lang={segments[i + 1]} code={segments[i + 2].replace(/\n$/, "")} />);
    }
  }
  return <>{nodes}</>;
}

// ── Small pieces ───────────────────────────────────────────────────

function TypingDots() {
  return (
    <span className="inline-flex items-center gap-1 px-0.5">
      {[0, 1, 2].map((d) => (
        <span key={d} className="h-1.5 w-1.5 rounded-full bg-brand-400 animate-dot-bounce" style={{ animationDelay: `${d * 0.15}s` }} />
      ))}
    </span>
  );
}

function CrystalAvatar() {
  return (
    <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-brand-50 ring-1 ring-brand-200/60">
      <Gem className="h-3.5 w-3.5 text-brand-400" />
    </div>
  );
}

// Humanize CRYS's tool names for the activity trace. Unknown tools fall back
// to the raw name (forward-compatible with P4d generation tools).
const TOOL_LABELS: Record<string, string> = {
  content_search: "Searched documents",
  knowledge_search: "Searched knowledge",
  navigation_search: "Scanned keys",
  depth_search: "Deep analysis",
  memory_recall: "Recalled memory",
  memory_store: "Stored memory",
  web_search: "Web search",
  cognition_run: "Ran cognition",
  create_document: "Wrote document",
};

function toolLabel(name: string): string {
  return TOOL_LABELS[name] ?? name;
}

function toolInputSummary(input: Record<string, unknown>): string {
  if (typeof input?.query === "string") return input.query;
  for (const v of Object.values(input ?? {})) {
    if (typeof v === "string" && v) return v;
  }
  return Object.keys(input ?? {}).join(", ");
}

// P4d — narrow a CRYS tool output to a document artifact (the shape
// agent/tools/artifacts.py::create_document returns). Anything else returns
// null and the tool falls back to the generic trace rendering.
function asDocument(output: unknown): DocumentArtifact | null {
  if (
    output != null &&
    typeof output === "object" &&
    (output as { type?: unknown }).type === "document" &&
    typeof (output as { content?: unknown }).content === "string" &&
    typeof (output as { filename?: unknown }).filename === "string"
  ) {
    return output as DocumentArtifact;
  }
  return null;
}

function fmtBytes(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return "0 B";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

// One CRYS tool call: the tool_use request (name + input summary) with an
// expandable input/output detail. P4d adds media-specific result cards
// (image / file / video) by switching on tool name / output shape here.
function ToolCard({ call }: { call: AgentToolCall }) {
  const [open, setOpen] = useState(false);
  const doc = asDocument(call.output);
  // For a generated document the filename is a cleaner one-liner than the
  // raw input (whose first string field is the full document body).
  const summary = doc ? doc.filename : toolInputSummary(call.input);
  const output =
    typeof call.output === "string" ? call.output : JSON.stringify(call.output, null, 2);
  return (
    <div className="overflow-hidden rounded-md border border-gray-100 bg-white/70">
      <button onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-1.5 px-2.5 py-1.5 text-left text-[11px] transition-colors hover:bg-gray-50">
        <ChevronRight className={cn("h-3 w-3 shrink-0 text-gray-300 transition-transform", open && "rotate-90")} />
        <span className="shrink-0 font-medium text-gray-600">{toolLabel(call.tool_name)}</span>
        {summary && <span className="truncate text-gray-400">{summary}</span>}
        {call.is_error && <span className="ml-auto shrink-0 font-medium text-red-500">error</span>}
      </button>
      {open && (
        <div className="space-y-1.5 border-t border-gray-100 px-2.5 py-2">
          <div>
            <div className="mb-0.5 text-[9px] font-semibold uppercase tracking-wider text-gray-300">input</div>
            <pre className="max-h-40 overflow-auto rounded bg-gray-50 px-2 py-1 text-[10px] leading-relaxed text-gray-600">{JSON.stringify(call.input, null, 2)}</pre>
          </div>
          <div>
            <div className="mb-0.5 text-[9px] font-semibold uppercase tracking-wider text-gray-300">output</div>
            {doc ? (
              <div className="rounded bg-gray-50 px-2 py-1 text-[10px] leading-relaxed text-gray-500">
                produced <span className="font-medium text-gray-700">{doc.filename}</span> ({fmtBytes(doc.bytes)}) — shown above
              </div>
            ) : (
              <pre className="max-h-48 overflow-auto rounded bg-gray-50 px-2 py-1 text-[10px] leading-relaxed text-gray-600">{output}</pre>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// The run footer doubles as the entry point to CRYS's tool-by-tool trace:
// the summary line stays quiet by default; clicking expands each tool call
// (the tool_use request + its result).
function RunFooter({ result }: { result: AgentRunResponse }) {
  const [open, setOpen] = useState(false);
  const calls = result.tool_calls ?? [];
  const toolCount = calls.length;
  return (
    <div className="mt-2">
      <button onClick={() => toolCount > 0 && setOpen((o) => !o)} disabled={toolCount === 0}
        className={cn(
          "flex w-full flex-wrap items-center gap-x-3 gap-y-1.5 rounded-lg border border-gray-100 bg-gray-50/60 px-3 py-1.5 text-[11px] text-gray-400 transition-colors",
          toolCount > 0 ? "hover:bg-gray-100/60" : "cursor-default"
        )}>
        <span className="inline-flex items-center gap-1.5 font-medium uppercase tracking-wider text-gray-400">
          <Gem className="h-3 w-3 text-brand-400" /> CRYS
        </span>
        {toolCount > 0 && (
          <span className="inline-flex items-center gap-1">
            <ChevronRight className={cn("h-3 w-3 transition-transform", open && "rotate-90")} />
            {toolCount} tool{toolCount === 1 ? "" : "s"}
          </span>
        )}
        <span>{result.iterations} iter</span>
        <span>{fmtNum(result.prompt_tokens)} → {fmtNum(result.completion_tokens)} tok</span>
      </button>
      {open && toolCount > 0 && (
        <div className="mt-1.5 space-y-1">
          {calls.map((c, i) => <ToolCard key={c.tool_use_id || `${i}`} call={c} />)}
        </div>
      )}
    </div>
  );
}

// A generated document, surfaced as a download + preview card so the
// artifact is a first-class result rather than JSON buried in the trace.
// md previews through the same Markdown-lite renderer as chat; txt/html
// preview as monospace source.
function DocumentCard({ doc }: { doc: DocumentArtifact }) {
  const [preview, setPreview] = useState(false);
  const download = () => {
    try {
      const blob = new Blob([doc.content], { type: doc.mime || "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = doc.filename || "document";
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch {}
  };
  return (
    <div className="overflow-hidden rounded-xl border border-brand-200/50 bg-brand-50/30">
      <div className="flex items-center gap-2.5 px-3 py-2.5">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-white ring-1 ring-brand-200/60">
          <FileText className="h-4 w-4 text-brand-500" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-medium text-gray-800">{doc.title || doc.filename}</div>
          <div className="truncate text-[11px] text-gray-400">
            {doc.filename} · {doc.format.toUpperCase()} · {fmtBytes(doc.bytes)}
          </div>
        </div>
        <button onClick={() => setPreview((p) => !p)}
          className="shrink-0 rounded-lg px-2 py-1 text-[11px] font-medium text-gray-500 transition-colors hover:bg-white hover:text-gray-800">
          {preview ? "Hide" : "Preview"}
        </button>
        <button onClick={download}
          className="inline-flex shrink-0 items-center gap-1.5 rounded-lg bg-brand-600 px-2.5 py-1.5 text-[11px] font-medium text-zinc-50 shadow-glow transition-colors hover:bg-brand-500">
          <Download className="h-3.5 w-3.5" /> Download
        </button>
      </div>
      {preview && (
        <div className="border-t border-brand-200/40 bg-white px-3 py-2.5 text-sm leading-relaxed text-gray-700">
          {doc.format === "md" ? (
            <Markdown text={doc.content} />
          ) : (
            <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words text-[12px] text-gray-600">{doc.content}</pre>
          )}
        </div>
      )}
    </div>
  );
}

// Scan a CRYS run for document artifacts and render them as cards between
// the answer bubble and the tool trace, so a produced file is visible
// without expanding the trace.
function Artifacts({ result }: { result: AgentRunResponse }) {
  const docs = (result.tool_calls ?? [])
    .map((c) => asDocument(c.output))
    .filter((d): d is DocumentArtifact => d !== null);
  if (docs.length === 0) return null;
  return (
    <div className="mt-2 space-y-2">
      {docs.map((d, i) => <DocumentCard key={`${d.filename}-${i}`} doc={d} />)}
    </div>
  );
}

// ── Page ───────────────────────────────────────────────────────────

export function ChatPlayground() {
  const { selectedCustomerId } = useSelectedCustomer();
  const [input, setInput] = useState("");
  const [turns, setTurns] = useState<PlaygroundTurn[]>([]);
  const [showSettings, setShowSettings] = useState(false);
  const [upstreamKey, setUpstreamKey] = useState("");
  const [keyStatus, setKeyStatus] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Tenant-safe customer resolution (Accounts Phase C fix, 2026-07-06):
  // a pinned tenant cannot list customers (the guard 401s the cross-
  // tenant list — correctly), so they fetch their OWN record instead.
  // Admins and self-host keep the list path.
  const { status: authStatus, me } = useAuth();
  const isTenant = authStatus === "signedIn" && me?.role === "owner";
  const customers = useQuery({
    queryKey: ["customers"],
    queryFn: api.listCustomers,
    enabled: !!selectedCustomerId && !isTenant,
  });
  const ownCustomer = useQuery({
    queryKey: ["own-customer", selectedCustomerId],
    queryFn: () => api.getCustomer(selectedCustomerId!),
    enabled: !!selectedCustomerId && isTenant,
  });
  const customer = isTenant
    ? ownCustomer.data
    : customers.data?.items.find((c) => c.id === selectedCustomerId);

  const sendMutation = useMutation({
    mutationFn: async (text: string): Promise<AgentRunResponse> => {
      if (!customer || !selectedCustomerId) throw new Error("Not ready");
      // Full conversation history each call — CRYS is stateless (P0.17),
      // so prior turns are replayed as plain user/assistant text (the
      // agent re-derives its own tool context within the run).
      const messages: { role: "user" | "assistant"; content: string }[] = [];
      for (const t of turns) {
        messages.push({ role: "user" as const, content: t.user });
        if (t.assistant) messages.push({ role: "assistant" as const, content: t.assistant });
      }
      messages.push({ role: "user" as const, content: text });
      return api.adminAgent(selectedCustomerId, { messages });
    },
  });

  useEffect(() => { if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight; }, [turns]);

  const autoGrow = () => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  };

  const send = async (preset?: string) => {
    const text = (preset ?? input).trim();
    if (!text || !customer || sendMutation.isPending) return;
    const turnId = crypto.randomUUID();
    const sentAt = Date.now();
    setTurns((p) => [...p, { id: turnId, user: text, assistant: null, result: null, error: null, sentAt }]);
    setInput("");
    if (inputRef.current) inputRef.current.style.height = "auto";
    try {
      const res = await sendMutation.mutateAsync(text);
      setTurns((p) => p.map((t) => (t.id === turnId ? { ...t, assistant: res.final_text || "(empty)", result: res } : t)));
    } catch (e) {
      setTurns((p) => p.map((t) => (t.id === turnId ? { ...t, error: e instanceof ApiError ? `${e.status}: ${JSON.stringify(e.body)}` : String(e) } : t)));
    }
  };

  const feedback = async (turn: PlaygroundTurn, signal: "up" | "down", comment?: string) => {
    if (!selectedCustomerId) return;
    try {
      const r = await api.adminLearn(selectedCustomerId, { prompt: turn.user, response: turn.assistant || "", outcome: signal === "up" ? "pass" : "fail", signal: signal === "down" ? (comment || "Incorrect") : undefined });
      setTurns((p) => p.map((t) => (t.id === turn.id ? { ...t, feedback: signal, learnResult: r } : t)));
    } catch {}
  };

  if (!selectedCustomerId) {
    return (
      <div className="flex h-full items-center justify-center px-8">
        <div className="text-center">
          <p className="text-sm font-semibold text-gray-900">No customer selected</p>
          <p className="mt-1 text-sm text-gray-500">Pick a customer in the sidebar to start chatting.</p>
        </div>
      </div>
    );
  }

  const empty = turns.length === 0;

  return (
    <div className="flex h-screen flex-col">
      {/* ── Header bar ── */}
      <div className="glass z-10 border-b border-gray-200">
        <div className="mx-auto flex h-14 max-w-3xl items-center justify-between px-4">
          <div className="flex items-center gap-2.5">
            <h1 className="text-[15px] font-semibold text-gray-900">Chat</h1>
            {customer && (
              <span className="rounded-md border border-gray-200 bg-gray-50 px-2 py-0.5 font-mono text-[11px] text-gray-500">
                CRYS
              </span>
            )}
          </div>
          <div className="flex items-center gap-1">
            {turns.length > 0 && (
              <button onClick={() => setTurns([])} title="New conversation"
                className="inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium text-gray-500 transition-colors hover:bg-gray-100 hover:text-gray-900">
                <RotateCcw className="h-3.5 w-3.5" /> New chat
              </button>
            )}
            <button onClick={() => setShowSettings((v) => !v)} title="Connection settings"
              className={cn("rounded-lg p-2 transition-colors", showSettings ? "bg-brand-50 text-brand-400" : "text-gray-500 hover:bg-gray-100 hover:text-gray-900")}>
              <Settings2 className="h-4 w-4" />
            </button>
          </div>
        </div>
        {showSettings && (
          <div className="border-t border-gray-100">
            <div className="mx-auto flex max-w-3xl items-center gap-2 px-4 py-2.5">
              <Key className="h-4 w-4 shrink-0 text-gray-400" />
              <input type="password" value={upstreamKey} onChange={(e) => setUpstreamKey(e.target.value)}
                placeholder="Upstream API key (sk-ant-...)"
                className="flex-1 bg-transparent font-mono text-xs text-gray-700 placeholder:text-gray-400 focus:outline-none" />
              <CrystalButton size="sm" variant={keyStatus === "Saved" ? "ghost" : "primary"} disabled={!upstreamKey.trim()}
                onClick={async () => {
                  if (!upstreamKey.trim() || !selectedCustomerId) return;
                  setKeyStatus("…"); await api.updateUpstreamKey(selectedCustomerId, upstreamKey.trim());
                  setKeyStatus("Saved"); setTimeout(() => setKeyStatus(null), 2000);
                }}>{keyStatus === "Saved" ? <><Check className="h-3 w-3" /> Saved</> : keyStatus || "Set key"}</CrystalButton>
              <button onClick={() => setShowSettings(false)} className="rounded p-1 text-gray-400 hover:text-gray-700"><X className="h-3.5 w-3.5" /></button>
            </div>
          </div>
        )}
      </div>

      {/* ── Messages ── */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl px-4 pb-6 pt-6">
          {empty ? (
            <div className="flex min-h-[55vh] flex-col items-center justify-center text-center">
              <div className="mb-5 flex h-14 w-14 items-center justify-center rounded-2xl bg-brand-50 shadow-glow ring-1 ring-brand-200/50">
                <Gem className="h-6 w-6 text-brand-400" />
              </div>
              <h2 className="prism-text text-2xl font-semibold tracking-tight">What do you want to know?</h2>
              <p className="mt-2 max-w-md text-sm leading-relaxed text-gray-500">
                CRYS answers using this customer's knowledge bank — its reasoning and tool calls appear under each reply.
              </p>
              <div className="mt-7 flex flex-wrap justify-center gap-2">
                {SUGGESTIONS.map((s) => (
                  <button key={s} onClick={() => send(s)}
                    className="rounded-full border border-gray-200 bg-white px-3.5 py-1.5 text-xs text-gray-600 transition-all hover:border-brand-300/40 hover:bg-brand-50 hover:text-brand-700">
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {turns.map((t) => (
                <div key={t.id} className="animate-fade-up space-y-3">
                  {/* User */}
                  <div className="flex justify-end">
                    <div className="max-w-[80%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-user-bubble px-4 py-2.5 text-sm leading-relaxed text-zinc-50 shadow-card">
                      {t.user}
                    </div>
                  </div>
                  {/* Assistant */}
                  <div className="flex items-start gap-3">
                    <CrystalAvatar />
                    <div className="min-w-0 max-w-[85%] flex-1">
                      {t.assistant !== null ? (
                        <div className="rounded-2xl rounded-tl-md border border-gray-100 bg-white px-4 py-3 text-sm leading-relaxed text-gray-700 shadow-card">
                          <Markdown text={t.assistant} />
                        </div>
                      ) : t.error ? (
                        <div className="rounded-2xl rounded-tl-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">{t.error}</div>
                      ) : (
                        <div className="inline-flex items-center rounded-2xl rounded-tl-md border border-gray-100 bg-white px-4 py-3.5 shadow-card">
                          <TypingDots />
                        </div>
                      )}

                      {t.result && <Artifacts result={t.result} />}

                      {t.result && <RunFooter result={t.result} />}

                      {t.assistant && !t.feedback && (
                        <div className="mt-1.5 flex gap-0.5">
                          <button onClick={() => feedback(t, "up")} title="Good answer — cache it"
                            className="rounded-md p-1.5 text-gray-300 transition-colors hover:bg-emerald-50 hover:text-emerald-500">
                            <ThumbsUp className="h-3.5 w-3.5" />
                          </button>
                          <button onClick={() => { const c = window.prompt("What was wrong?"); feedback(t, "down", c || undefined); }} title="Wrong — teach it"
                            className="rounded-md p-1.5 text-gray-300 transition-colors hover:bg-red-50 hover:text-red-500">
                            <ThumbsDown className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      )}
                      {t.feedback === "up" && (
                        <span className="mt-1.5 inline-flex items-center gap-1 text-[11px] font-medium text-emerald-500"><Zap className="h-3 w-3" /> Cached</span>
                      )}
                      {t.feedback === "down" && t.learnResult && (
                        <span className="mt-1.5 inline-flex items-center gap-1 text-[11px] font-medium text-amber-500"><Gem className="h-3 w-3" /> Learned — +{t.learnResult.crystals_written} crystals</span>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* ── Composer ── */}
      <div className="glass border-t border-gray-200">
        <div className="mx-auto max-w-3xl px-4 py-4">
          <div className="flex items-end gap-2 rounded-2xl border border-gray-200 bg-white px-3 py-2 shadow-card transition-colors focus-within:border-brand-500/60 focus-within:shadow-facet">
            <textarea ref={inputRef} value={input} rows={1}
              onChange={(e) => { setInput(e.target.value); autoGrow(); }}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
              placeholder="Ask anything about this bank…"
              className="max-h-[200px] flex-1 resize-none bg-transparent px-1.5 py-1.5 text-sm leading-relaxed text-gray-800 placeholder:text-gray-400 focus:outline-none"
              disabled={sendMutation.isPending || !customer} />
            <button onClick={() => send()} disabled={sendMutation.isPending || !input.trim() || !customer}
              className="mb-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-brand-600 text-zinc-50 shadow-glow transition-all hover:bg-brand-500 disabled:opacity-30 disabled:shadow-none">
              {sendMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <ArrowUp className="h-4 w-4" />}
            </button>
          </div>
          <p className="mt-2 text-center text-[11px] text-gray-400">
            <kbd className="rounded border border-gray-200 bg-gray-50 px-1 py-0.5 font-mono text-[10px]">Enter</kbd> to send ·{" "}
            <kbd className="rounded border border-gray-200 bg-gray-50 px-1 py-0.5 font-mono text-[10px]">Shift+Enter</kbd> for a new line
          </p>
        </div>
      </div>
    </div>
  );
}
