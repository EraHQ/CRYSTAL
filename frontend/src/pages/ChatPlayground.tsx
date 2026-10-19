// Chat — Console Refine PR 8 (docs/CONSOLE_UX_AUDIT.md 5.2).
//
// Ask what your connected tools would know; the answer comes only from
// this workspace's memory. Each answer says which memories answered:
// the recall and search steps in the trace are numbered, the numbers
// appear under the answer, and the expanded trace shows each step in
// plain words with its query and the memory it found (no JSON). The
// provider key field moved to Settings; feedback is Helpful / Not right
// with an inline correction; a failed turn shows an ErrorBanner with
// Retry. The streaming ticker and the stateless full-history replay
// (P0.17) are unchanged.
import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ArrowUp, Check, ChevronRight, Copy, Download, FileText, Gem, Loader2, RotateCcw, X, History as HistoryIcon } from "lucide-react";
import { Link } from "react-router-dom";
import { api, errorMessage } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";
import { useAuth } from "@/lib/auth";
import { Drawer, ErrorBanner, PageHeader, Skeleton, useToast } from "@/components/ui";
import type { AgentRunResponse, AgentToolCall, DocumentArtifact } from "@/lib/types";
import { cn } from "@/lib/utils";
import { modelLabel } from "@/pages/settings/ModelInference";

interface TurnActivity { id: string; tool: string; status: "running" | "ok" | "error"; ms?: number }

interface PlaygroundTurn {
  id: string; user: string; assistant: string | null; result: AgentRunResponse | null;
  error: string | null; sentAt: number; doneAt?: number; feedback?: "up" | "down"; learnResult?: { crystals_written: number };
  activity?: TurnActivity[]; phase?: string | null; draft?: string;
}

const STATIC_SUGGESTIONS = [
  "What did I decide most recently?",
  "Summarize the most recent document I uploaded",
  "What questions could memory not answer lately?",
];

// ── Markdown-lite ──────────────────────────────────────────────────
// Fenced code blocks (with copy), inline `code`, **bold**. No parser
// dependency, no HTML injection; plain React nodes only.

function CodeBlock({ lang, code }: { lang: string; code: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try { await navigator.clipboard.writeText(code); setCopied(true); setTimeout(() => setCopied(false), 1600); } catch { /* clipboard unavailable */ }
  };
  return (
    <div className="chat-code my-2 overflow-hidden rounded-lg border border-line">
      <div className="flex items-center justify-between border-b border-line px-3 py-1.5">
        <span className="text-[11px] font-medium uppercase tracking-wider text-fg-3">{lang || "code"}</span>
        <button type="button" onClick={copy} className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-medium text-fg-3 hover:bg-inset hover:text-fg">
          {copied ? <><Check aria-hidden className="h-3 w-3 text-ok" /> Copied</> : <><Copy aria-hidden className="h-3 w-3" /> Copy</>}
        </button>
      </div>
      <pre className="overflow-x-auto px-3 py-2.5 text-[12.5px] text-fg-2">{code}</pre>
    </div>
  );
}

function InlineMd({ text }: { text: string }) {
  const parts = text.split(/(`[^`\n]+`)/g);
  return (
    <>
      {parts.map((part, i) => {
        if (part.startsWith("`") && part.endsWith("`") && part.length > 2) return <code key={i} className="chat-inline-code">{part.slice(1, -1)}</code>;
        const boldParts = part.split(/(\*\*[^*]+\*\*)/g);
        return (
          <Fragment key={i}>
            {boldParts.map((b, j) => b.startsWith("**") && b.endsWith("**") && b.length > 4
              ? <strong key={j} className="font-semibold text-fg">{b.slice(2, -2)}</strong>
              : <Fragment key={j}>{b}</Fragment>)}
          </Fragment>
        );
      })}
    </>
  );
}

export function Markdown({ text }: { text: string }) {
  const segments = text.split(/```(\w*)\n?([\s\S]*?)```/g);
  const nodes: React.ReactNode[] = [];
  for (let i = 0; i < segments.length; i += 3) {
    const plain = segments[i];
    if (plain) nodes.push(<span key={`t${i}`} className="whitespace-pre-wrap"><InlineMd text={plain} /></span>);
    if (i + 2 < segments.length) nodes.push(<CodeBlock key={`c${i}`} lang={segments[i + 1]} code={segments[i + 2].replace(/\n$/, "")} />);
  }
  return <>{nodes}</>;
}

function TypingDots() {
  return (
    <span className="inline-flex items-center gap-1 px-0.5" aria-label="Working">
      {[0, 1, 2].map((d) => <span key={d} className="h-1.5 w-1.5 rounded-full bg-accent animate-dot-bounce" style={{ animationDelay: `${d * 0.15}s` }} />)}
    </span>
  );
}

function CrystalAvatar() {
  return (
    <div aria-hidden className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-accent-soft ring-1 ring-accent/30">
      <Gem className="h-3.5 w-3.5 text-accent" />
    </div>
  );
}

// ── The trace: what each step did, in plain words ─────────────────

// Tool names (agent/tools/*) in plain words; the raw name rides in `title`.
const TOOL_LABELS: Record<string, string> = {
  content_search: "Searched documents", knowledge_search: "Recalled memory", navigation_search: "Scanned memory keys",
  depth_search: "Read deeply", key_scan: "Scanned memory keys", crystal_recall: "Recalled memory", memory_recall: "Recalled memory",
  mem0_recall: "Recalled this conversation", crystal_write: "Saved a memory", memory_store: "Saved a memory", mem0_write: "Saved this turn",
  web_search: "Searched the web", web_fetch: "Read a web page", cognition_run: "Ran research", create_document: "Wrote a document",
};
const RECALL_TOOLS = new Set(["content_search", "knowledge_search", "navigation_search", "depth_search", "key_scan", "crystal_recall", "memory_recall", "mem0_recall"]);
export function toolLabel(name: string): string { return TOOL_LABELS[name] ?? name.replace(/_/g, " "); }

export interface TraceMemory { text: string; source: string | null; crystalId: string | null }
export interface TraceStep {
  n: number | null; call: AgentToolCall; label: string; query: string;
  memories: TraceMemory[]; factCount: number; documents: number; note: string | null;
}

function stringField(o: unknown, k: string): string | null {
  const v = o && typeof o === "object" ? (o as Record<string, unknown>)[k] : undefined;
  return typeof v === "string" && v.trim() ? v : null;
}
function listField(o: unknown, k: string): unknown[] {
  const v = o && typeof o === "object" ? (o as Record<string, unknown>)[k] : undefined;
  return Array.isArray(v) ? v : [];
}

export function buildTrace(result: AgentRunResponse): { steps: TraceStep[]; memories: number; documents: number } {
  let n = 0, memories = 0, documents = 0;
  const steps: TraceStep[] = (result.tool_calls ?? []).map((call) => {
    const isRecall = RECALL_TOOLS.has(call.tool_name);
    const out = typeof call.output === "string" ? (() => { try { return JSON.parse(call.output) as unknown; } catch { return call.output; } })() : call.output;
    const query = stringField(call.input, "query") ?? stringField(call.input, "key_prefix") ?? stringField(call.input, "key") ?? "";
    const found: TraceMemory[] = [];
    let factCount = 0, docs = 0;
    if (isRecall) {
      const crystals = listField(out, "crystals");
      for (const c of crystals) {
        for (const f of listField(c, "facts")) {
          found.push({ text: stringField(f, "claim_text") ?? "", source: stringField(f, "prompt_text"), crystalId: stringField(c, "crystal_id") });
        }
      }
      const findings = listField(out, "findings");
      for (const f of findings) {
        found.push({ text: stringField(f, "content_preview") ?? "", source: stringField(f, "key"), crystalId: stringField(f, "crystal_id") });
      }
      const ids = listField(out, "matched_fact_ids");
      const crystalIds = listField(out, "matched_crystal_ids").filter((x): x is string => typeof x === "string");
      if (found.length === 0) {
        const injection = stringField(out, "injection_text");
        if (injection) found.push({ text: injection, source: null, crystalId: crystalIds[0] ?? null });
      }
      factCount = Math.max(ids.length, found.length);
      if (call.tool_name === "content_search") { docs = factCount; documents += docs; } else { memories += factCount; }
    }
    const note = call.is_error ? "This step failed." : isRecall && factCount === 0 ? "Nothing matched." : stringField(out, "tier_note");
    return { n: isRecall ? ++n : null, call, label: toolLabel(call.tool_name), query, memories: found.filter((m) => m.text), factCount, documents: docs, note };
  });
  return { steps, memories, documents };
}

function TraceStepRow({ step, highlighted }: { step: TraceStep; highlighted: boolean }) {
  return (
    <li id={`trace-${step.call.tool_use_id}`} className={cn("rounded-lg border px-3 py-2 text-[12.5px]", highlighted ? "border-accent bg-accent-soft/40" : "border-line bg-surface")}>
      <div className="flex flex-wrap items-center gap-2">
        {step.n != null && <span className="inline-flex h-5 min-w-5 items-center justify-center rounded-full bg-accent px-1.5 text-[11px] font-semibold text-on-accent">{step.n}</span>}
        <span className="font-medium text-fg" title={step.call.tool_name}>{step.label}</span>
        {step.query && <span className="min-w-0 flex-1 truncate text-fg-3">“{step.query}”</span>}
        {step.call.is_error && <span className="text-bad">failed</span>}
      </div>
      {step.memories.length > 0 && (
        <ul className="mt-1.5 space-y-1">
          {step.memories.slice(0, 6).map((m, i) => (
            <li key={i} className="rounded-md bg-inset px-2.5 py-1.5">
              <p className="whitespace-pre-wrap break-words text-fg">{m.text}</p>
              {(m.source || m.crystalId) && (
                <p className="mt-0.5 text-[11.5px] text-fg-3">
                  {m.source}{m.source && m.crystalId ? " · " : ""}
                  {m.crystalId && <Link to={`/bank/${m.crystalId}`} className="text-accent hover:underline">open crystal</Link>}
                </p>
              )}
            </li>
          ))}
          {step.memories.length > 6 && <li className="text-[11.5px] text-fg-3">{step.memories.length - 6} more</li>}
        </ul>
      )}
      {step.note && <p className="mt-1 text-[11.5px] text-fg-3">{step.note}</p>}
    </li>
  );
}

function Trace({ result, seconds, openStep, onToggle }: { result: AgentRunResponse; seconds: number | null; openStep: string | null; onToggle: (open: boolean) => void }) {
  const [open, setOpen] = useState(false);
  const { steps, memories, documents } = useMemo(() => buildTrace(result), [result]);
  useEffect(() => { if (openStep) setOpen(true); }, [openStep]);
  const parts = [
    `Looked up ${memories} memor${memories === 1 ? "y" : "ies"}`,
    documents ? `${documents} document passage${documents === 1 ? "" : "s"}` : null,
    steps.some((s) => s.call.tool_name === "create_document") ? "wrote a document" : null,
  ].filter(Boolean);
  const line = parts.length > 1 ? `${parts[0]} and ${parts.slice(1).join(", ")}` : parts[0];
  return (
    <div className="mt-2">
      <button type="button" onClick={() => { setOpen((o) => !o); onToggle(!open); }} aria-expanded={open}
        className="flex w-full items-center gap-2 rounded-lg border border-line bg-inset px-3 py-1.5 text-left text-[12px] text-fg-2 hover:bg-inset/70">
        <ChevronRight aria-hidden className={cn("h-3 w-3 transition-transform", open && "rotate-90")} />
        <span>{line}{seconds != null ? ` · ${seconds.toFixed(1)}s` : ""}{modelLabel(result.model) ? ` · ${modelLabel(result.model)}` : ""}</span>
      </button>
      {open && (
        <ol aria-label="Trace" className="mt-1.5 space-y-1.5">
          {steps.length === 0 && <li className="text-[12px] text-fg-3">No lookups: the answer came from the conversation alone.</li>}
          {steps.map((s) => <TraceStepRow key={s.call.tool_use_id} step={s} highlighted={openStep === s.call.tool_use_id} />)}
        </ol>
      )}
    </div>
  );
}

// ── Document artifacts ────────────────────────────────────────────

function asDocument(output: unknown): DocumentArtifact | null {
  if (output != null && typeof output === "object" && (output as { type?: unknown }).type === "document"
    && typeof (output as { content?: unknown }).content === "string" && typeof (output as { filename?: unknown }).filename === "string") {
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

function DocumentCard({ doc }: { doc: DocumentArtifact }) {
  const [preview, setPreview] = useState(false);
  const download = () => {
    try {
      const blob = new Blob([doc.content], { type: doc.mime || "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = doc.filename || "document";
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch { /* download unavailable */ }
  };
  return (
    <div className="overflow-hidden rounded-xl border border-accent/30 bg-accent-soft/40">
      <div className="flex items-center gap-2.5 px-3 py-2.5">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-surface ring-1 ring-accent/30"><FileText aria-hidden className="h-4 w-4 text-accent" /></div>
        <div className="min-w-0 flex-1">
          <div className="truncate text-[13.5px] font-medium text-fg">{doc.title || doc.filename}</div>
          <div className="truncate text-[11.5px] text-fg-3">{doc.filename} · {doc.format.toUpperCase()} · {fmtBytes(doc.bytes)}</div>
        </div>
        <button type="button" onClick={() => setPreview((p) => !p)} className="shrink-0 rounded-md px-2 py-1 text-[12px] font-medium text-fg-2 hover:bg-surface hover:text-fg">{preview ? "Hide" : "Preview"}</button>
        <button type="button" onClick={download} className="inline-flex shrink-0 items-center gap-1.5 rounded-md bg-accent px-2.5 py-1.5 text-[12px] font-medium text-on-accent hover:bg-accent-hover">
          <Download aria-hidden className="h-3.5 w-3.5" /> Download
        </button>
      </div>
      {preview && (
        <div className="border-t border-accent/20 bg-surface px-3 py-2.5 text-[13.5px] leading-relaxed text-fg-2">
          {doc.format === "md" ? <Markdown text={doc.content} /> : <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words text-[12px] text-fg-2">{doc.content}</pre>}
        </div>
      )}
    </div>
  );
}

function Artifacts({ result }: { result: AgentRunResponse }) {
  const docs = (result.tool_calls ?? []).map((c) => asDocument(c.output)).filter((d): d is DocumentArtifact => d !== null);
  if (docs.length === 0) return null;
  return <div className="mt-2 space-y-2">{docs.map((d, i) => <DocumentCard key={`${d.filename}-${i}`} doc={d} />)}</div>;
}

// ── Page ───────────────────────────────────────────────────────────

const newSequenceId = () => (crypto.randomUUID ? crypto.randomUUID() : `seq_${Date.now()}`).slice(0, 64);

export function ChatPlayground() {
  const { selectedCustomerId } = useSelectedCustomer();
  const toast = useToast();
  const [input, setInput] = useState("");
  const [turns, setTurns] = useState<PlaygroundTurn[]>([]);
  const [downFor, setDownFor] = useState<string | null>(null);
  const [downText, setDownText] = useState("");
  const [sequenceId, setSequenceId] = useState<string>(newSequenceId);
  const [showHistory, setShowHistory] = useState(false);
  const [loadingSession, setLoadingSession] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [openStep, setOpenStep] = useState<{ turn: string; step: string } | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // A pinned tenant cannot list customers; they fetch their own record.
  const { status: authStatus, me } = useAuth();
  const isTenant = authStatus === "signedIn" && me?.role === "owner";
  const customers = useQuery({ queryKey: ["customers"], queryFn: api.listCustomers, enabled: !!selectedCustomerId && !isTenant });
  const ownCustomer = useQuery({ queryKey: ["own-customer", selectedCustomerId], queryFn: () => api.getCustomer(selectedCustomerId!), enabled: !!selectedCustomerId && isTenant });
  const customer = isTenant ? ownCustomer.data : customers.data?.items.find((c) => c.id === selectedCustomerId);

  // 5.2 item 5: three suggestions from the bank's top domains, static fallback.
  const topics = useQuery({ queryKey: ["chat-topics", selectedCustomerId], queryFn: () => api.listCrystals(selectedCustomerId!, { limit: 100 }), enabled: !!selectedCustomerId, staleTime: 5 * 60_000 });
  const suggestions = useMemo(() => {
    const counts = new Map<string, number>();
    for (const c of topics.data?.items ?? []) {
      const first = (c.headline_key ?? "").split("|")[0]?.trim();
      if (first && first !== "General" && first !== "Reflections") counts.set(first, (counts.get(first) ?? 0) + 1);
    }
    const top = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 3).map(([d]) => `What do you know about ${d}?`);
    return ["What do you remember about me?", ...(top.length ? top : STATIC_SUGGESTIONS)];
  }, [topics.data]);

  const history = useQuery({ queryKey: ["chat-sessions", selectedCustomerId], queryFn: () => api.listChatSessions(selectedCustomerId!), enabled: !!selectedCustomerId && showHistory });
  const loadSession = async (sid: string) => {
    if (!selectedCustomerId) return;
    setLoadingSession(sid);
    try {
      const data = await api.getChatSession(selectedCustomerId, sid);
      // Loading the transcript IS the resume: the full history is sent each turn.
      setTurns(data.turns.map((t: any, i: number) => ({
        id: `hist_${i}`, user: t.user, assistant: t.assistant,
        result: t.tool_calls?.length ? ({ tool_calls: t.tool_calls, final_text: t.assistant } as AgentRunResponse) : null,
        error: null, sentAt: t.at ? new Date(t.at).getTime() : Date.now(),
      })) as PlaygroundTurn[]);
      setSequenceId(sid);
      setShowHistory(false);
    } finally {
      setLoadingSession(null);
    }
  };

  const applyStreamEvent = (turnId: string, event: string, payload: any) => {
    setTurns((p) => p.map((t) => {
      if (t.id !== turnId) return t;
      if (event === "iteration_started") return { ...t, phase: "thinking…" };
      if (event === "text_delta") return { ...t, phase: null, draft: (t.draft || "") + (payload.text || "") };
      if (event === "tool_calls") {
        const added: TurnActivity[] = (payload.calls || []).map((c: any) => ({ id: c.tool_use_id || `${c.name}_${Date.now()}`, tool: c.name || "?", status: "running" as const }));
        return { ...t, phase: null, activity: [...(t.activity || []), ...added] };
      }
      if (event === "tool_result") {
        return { ...t, activity: (t.activity || []).map((a) => a.id === payload.tool_use_id ? { ...a, status: payload.is_error ? ("error" as const) : ("ok" as const), ms: payload.duration_ms } : a) };
      }
      if (event === "notice") {
        const k = payload.kind;
        const phase = k === "compacted" ? "tidying the conversation…" : k === "deadline" ? "out of time, composing…" : k === "tool_call_truncated" ? "retrying a cut-off step…"
          : k === "max_iterations" ? "wrapping up…" : k === "model_error" ? "model hiccup, recovering…" : null;
        return phase ? { ...t, phase } : t;
      }
      return t;
    }));
  };

  useEffect(() => { if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight; }, [turns]);

  const autoGrow = () => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  };

  const send = async (preset?: string, replaceTurnId?: string) => {
    const text = (preset ?? input).trim();
    if (!text || !customer || !selectedCustomerId || pending) return;
    const turnId = crypto.randomUUID();
    const sentAt = Date.now();
    const prior = replaceTurnId ? turns.filter((t) => t.id !== replaceTurnId) : turns;
    setTurns([...prior, { id: turnId, user: text, assistant: null, result: null, error: null, sentAt, activity: [], phase: "thinking…" }]);
    if (!preset) setInput("");
    if (inputRef.current) inputRef.current.style.height = "auto";
    // Full conversation history each call (P0.17); built once so the
    // stream and its fallback replay the identical history.
    const messages: { role: "user" | "assistant"; content: string }[] = [];
    for (const t of prior) {
      messages.push({ role: "user" as const, content: t.user });
      if (t.assistant) messages.push({ role: "assistant" as const, content: t.assistant });
    }
    messages.push({ role: "user" as const, content: text });
    setPending(true);
    const settle = (res: AgentRunResponse) =>
      setTurns((p) => p.map((t) => (t.id === turnId ? { ...t, assistant: res.final_text || "(empty)", result: res, phase: null, doneAt: Date.now() } : t)));
    try {
      settle(await api.adminAgentStream(selectedCustomerId, { messages, metadata: { sequence_id: sequenceId } }, (event, payload) => applyStreamEvent(turnId, event, payload)));
    } catch {
      try {
        settle(await api.adminAgent(selectedCustomerId, { messages, metadata: { sequence_id: sequenceId } }));
      } catch (e) {
        setTurns((p) => p.map((t) => (t.id === turnId ? { ...t, error: errorMessage(e, "The request failed."), phase: null } : t)));
      }
    } finally {
      setPending(false);
    }
  };

  const feedback = async (turn: PlaygroundTurn, signal: "up" | "down", comment?: string) => {
    if (!selectedCustomerId) return;
    try {
      const r = await api.adminLearn(selectedCustomerId, { prompt: turn.user, response: turn.assistant || "", outcome: signal === "up" ? "pass" : "fail", signal: signal === "down" ? (comment || "Incorrect") : undefined });
      setTurns((p) => p.map((t) => (t.id === turn.id ? { ...t, feedback: signal, learnResult: r } : t)));
      toast(signal === "up" ? "Saved. Similar questions reuse this answer." : "Correction saved to memory.");
    } catch (e) {
      toast(errorMessage(e, "Couldn't save that."));
    }
  };

  if (!selectedCustomerId) {
    return (
      <div className="flex h-full items-center justify-center px-8">
        <div className="text-center">
          <p className="text-[14px] font-semibold text-fg">No customer selected</p>
          <p className="mt-1 text-[13px] text-fg-2">Pick a customer above to start chatting.</p>
        </div>
      </div>
    );
  }

  const empty = turns.length === 0;
  const headerBtn = "inline-flex items-center gap-1.5 rounded-md border border-line px-2.5 py-1.5 text-[12.5px] font-medium text-fg-2 hover:bg-inset hover:text-fg";

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-line bg-surface">
        <div className="mx-auto max-w-3xl px-4 pt-4">
          <PageHeader title="Chat" meta="Ask what your connected tools would know. Answers come only from this workspace's memory." className="mb-3"
            actions={
              <>
                <button type="button" onClick={() => setShowHistory(true)} className={headerBtn}><HistoryIcon aria-hidden className="h-3.5 w-3.5" /> History</button>
                {turns.length > 0 && (
                  <button type="button" onClick={() => { setTurns([]); setSequenceId(newSequenceId()); }} className={headerBtn}><RotateCcw aria-hidden className="h-3.5 w-3.5" /> New chat</button>
                )}
              </>
            } />
        </div>
      </div>

      {showHistory && (
        <Drawer open onClose={() => setShowHistory(false)} title="Past conversations">
          {history.isLoading && <Skeleton rows={4} />}
          {history.isError && <ErrorBanner title="Couldn't load past conversations" error={history.error} onRetry={() => history.refetch()} />}
          {!history.isLoading && !history.data?.sessions?.length && <p className="text-[13px] text-fg-3">No past conversations yet.</p>}
          <ul className="divide-y divide-line">
            {(history.data?.sessions ?? []).map((s: any) => (
              <li key={s.sequence_id}>
                <button type="button" disabled={loadingSession !== null} onClick={() => void loadSession(s.sequence_id)}
                  className="flex w-full items-center justify-between gap-3 py-2 text-left hover:bg-inset disabled:opacity-50">
                  <span className="min-w-0 flex-1 truncate text-[13px] text-fg">{loadingSession === s.sequence_id ? "Loading…" : (s.title || "(untitled)")}</span>
                  <span className="shrink-0 text-[12px] text-fg-3">{s.turn_count} turn{s.turn_count === 1 ? "" : "s"}</span>
                </button>
              </li>
            ))}
          </ul>
        </Drawer>
      )}

      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl px-4 pb-6 pt-6">
          {empty ? (
            <div className="flex min-h-[50vh] flex-col items-center justify-center text-center">
              <div className="mb-5 flex h-14 w-14 items-center justify-center rounded-2xl bg-accent-soft ring-1 ring-accent/30"><Gem aria-hidden className="h-6 w-6 text-accent" /></div>
              <h2 className="text-[22px] font-[650] tracking-tight text-fg">What do you want to check?</h2>
              <p className="mt-2 max-w-md text-[13.5px] leading-relaxed text-fg-2">Ask the way you would in Claude, Cursor or ChatGPT. You will see which memories answered.</p>
              <div className="mt-6 flex flex-wrap justify-center gap-2">
                {suggestions.map((s) => (
                  <button key={s} type="button" onClick={() => void send(s)} disabled={!customer}
                    className="rounded-full border border-line bg-surface px-3.5 py-1.5 text-[12.5px] text-fg-2 hover:border-accent/40 hover:bg-accent-soft hover:text-accent disabled:opacity-50">
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {turns.map((t) => {
                const steps = t.result ? buildTrace(t.result).steps.filter((s) => s.n != null) : [];
                return (
                  <div key={t.id} className="animate-fade-up space-y-3">
                    <div className="flex justify-end">
                      <div className="max-w-[80%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-accent px-4 py-2.5 text-[13.5px] leading-relaxed text-on-accent">{t.user}</div>
                    </div>
                    <div className="flex items-start gap-3">
                      <CrystalAvatar />
                      <div className="min-w-0 max-w-[85%] flex-1">
                        {t.assistant !== null ? (
                          <div className="rounded-2xl rounded-tl-md border border-line bg-surface px-4 py-3 text-[13.5px] leading-relaxed text-fg">
                            <Markdown text={t.assistant} />
                            {steps.length > 0 && (
                              <div className="mt-2 flex flex-wrap items-center gap-1 border-t border-line pt-2 text-[11.5px] text-fg-3">
                                <span>From memory</span>
                                {steps.map((s) => (
                                  <button key={s.call.tool_use_id} type="button" onClick={() => setOpenStep({ turn: t.id, step: s.call.tool_use_id })}
                                    title={`${s.label}${s.query ? `: ${s.query}` : ""}`}
                                    className="inline-flex h-5 min-w-5 items-center justify-center rounded-full bg-accent-soft px-1.5 text-[11px] font-semibold text-accent hover:bg-accent hover:text-on-accent">
                                    {s.n}
                                  </button>
                                ))}
                              </div>
                            )}
                          </div>
                        ) : t.error ? (
                          <ErrorBanner title="That didn't go through" message={t.error} onRetry={() => void send(t.user, t.id)} />
                        ) : (
                          <div className="rounded-2xl rounded-tl-md border border-line bg-surface px-4 py-3">
                            {(t.activity?.length || t.phase || t.draft) ? (
                              <div className="space-y-1.5">
                                {(t.activity || []).map((a) => (
                                  <div key={a.id} className="flex items-center gap-2 text-[12.5px]">
                                    {a.status === "running" ? <Loader2 aria-hidden className="h-3 w-3 animate-spin text-accent" /> : a.status === "error" ? <X aria-hidden className="h-3 w-3 text-bad" /> : <Check aria-hidden className="h-3 w-3 text-ok" />}
                                    <span className="text-fg-2" title={a.tool}>{toolLabel(a.tool)}</span>
                                    {a.ms != null && a.status !== "running" && <span className="text-[11px] text-fg-3">{(a.ms / 1000).toFixed(1)}s</span>}
                                  </div>
                                ))}
                                {t.draft && <div className="whitespace-pre-wrap pt-0.5 text-[13.5px] leading-relaxed text-fg">{t.draft}</div>}
                                {t.phase && <div className="flex items-center gap-2 pt-0.5 text-[12px] text-fg-3"><TypingDots /> <span>{t.phase}</span></div>}
                              </div>
                            ) : <TypingDots />}
                          </div>
                        )}

                        {t.result && <Artifacts result={t.result} />}
                        {t.result && (
                          <Trace result={t.result} seconds={t.doneAt ? (t.doneAt - t.sentAt) / 1000 : null}
                            openStep={openStep?.turn === t.id ? openStep.step : null} onToggle={(o) => { if (!o) setOpenStep(null); }} />
                        )}

                        {t.assistant && !t.feedback && downFor !== t.id && (
                          <div className="mt-1.5 flex gap-2 text-[12px]">
                            <button type="button" onClick={() => void feedback(t, "up")} className="rounded-md px-2 py-1 text-fg-3 hover:bg-ok-soft hover:text-ok">Helpful</button>
                            <button type="button" onClick={() => { setDownFor(t.id); setDownText(""); }} className="rounded-md px-2 py-1 text-fg-3 hover:bg-bad-soft hover:text-bad">Not right</button>
                          </div>
                        )}
                        {t.assistant && !t.feedback && downFor === t.id && (
                          <form className="mt-2 rounded-lg border border-line bg-inset p-2.5" onSubmit={(e) => { e.preventDefault(); const c = downText.trim(); setDownFor(null); void feedback(t, "down", c || undefined); }}>
                            <label htmlFor={`down-${t.id}`} className="block text-[12px] font-medium text-fg-2">What was wrong? Your note is saved to memory as a correction.</label>
                            <textarea id={`down-${t.id}`} value={downText} onChange={(e) => setDownText(e.target.value)} rows={2} maxLength={2000} autoFocus
                              placeholder="What should the answer have been?"
                              className="mt-1 w-full rounded-md border border-line bg-surface px-2.5 py-1.5 text-[13px] text-fg outline-none focus:border-accent" />
                            <div className="mt-2 flex flex-wrap items-center gap-2">
                              <span className="text-[11.5px] text-fg-3">Uses a little AI capacity.</span>
                              <span className="flex-1" />
                              <button type="button" onClick={() => setDownFor(null)} className="rounded-md border border-line px-2.5 py-1 text-[12.5px] font-medium text-fg-2 hover:bg-surface hover:text-fg">Cancel</button>
                              <button type="submit" className="rounded-md bg-accent px-2.5 py-1 text-[12.5px] font-semibold text-on-accent hover:bg-accent-hover">Send correction</button>
                            </div>
                          </form>
                        )}
                        {t.feedback === "up" && <span className="mt-1.5 inline-block text-[11.5px] font-medium text-ok">Saved for similar questions</span>}
                        {t.feedback === "down" && t.learnResult && <span className="mt-1.5 inline-block text-[11.5px] font-medium text-warn">Correction saved{t.learnResult.crystals_written ? ` · ${t.learnResult.crystals_written} new memor${t.learnResult.crystals_written === 1 ? "y" : "ies"}` : ""}</span>}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      <div className="border-t border-line bg-surface">
        <div className="mx-auto max-w-3xl px-4 py-3">
          <div className="flex items-end gap-2 rounded-2xl border border-line bg-inset px-3 py-2 focus-within:border-accent">
            <textarea ref={inputRef} value={input} rows={1} aria-label="Message"
              onChange={(e) => { setInput(e.target.value); autoGrow(); }}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void send(); } }}
              placeholder="Ask about your memory"
              className="max-h-[200px] flex-1 resize-none bg-transparent px-1.5 py-1.5 text-[13.5px] leading-relaxed text-fg placeholder:text-fg-3 focus:outline-none"
              disabled={pending || !customer} />
            <button type="button" onClick={() => void send()} disabled={pending || !input.trim() || !customer} aria-label="Send"
              className="mb-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-accent text-on-accent hover:bg-accent-hover disabled:opacity-30">
              {pending ? <Loader2 aria-hidden className="h-4 w-4 animate-spin" /> : <ArrowUp aria-hidden className="h-4 w-4" />}
            </button>
          </div>
          <p className="mt-1.5 text-center text-[11.5px] text-fg-3">Enter to send · Shift Enter for a new line</p>
        </div>
      </div>
    </div>
  );
}
