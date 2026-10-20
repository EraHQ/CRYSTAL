// Markdown-lite for chat answers, briefs and run write-ups. No parser
// dependency, no HTML injection; plain React nodes only. Handles what
// the models actually emit: fenced code (with copy), inline `code`,
// **bold**, [links](https://…), headings, bullet and numbered lists,
// pipe tables, horizontal rules, paragraphs, and the literal <br> some
// models write for a line break. Anything else shows as typed.
// Lived in ChatPlayground until 2026-10-19 (headings and tables rendered
// raw in chat and in the Scheduled runs results); moved here so every
// surface renders the same way.
import { Fragment, ReactNode, useState } from "react";
import { Check, Copy } from "lucide-react";

export function CodeBlock({ lang, code }: { lang: string; code: string }) {
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

function bold(text: string): ReactNode[] {
  return text.split(/(\*\*[^*\n]+\*\*)/g).map((b, j) => b.startsWith("**") && b.endsWith("**") && b.length > 4
    ? <strong key={j} className="font-semibold text-fg">{b.slice(2, -2)}</strong>
    : <Fragment key={j}>{b}</Fragment>);
}

// [text](https://…) only; anything else stays as typed, so a javascript:
// or data: target can never become a link.
function linksAndBold(text: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g;
  let last = 0;
  let k = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(<Fragment key={k++}>{bold(text.slice(last, m.index))}</Fragment>);
    out.push(<a key={k++} href={m[2]} target="_blank" rel="noreferrer noopener" className="text-accent hover:underline">{m[1]}</a>);
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(<Fragment key={k++}>{bold(text.slice(last))}</Fragment>);
  return out;
}

export function InlineMd({ text }: { text: string }) {
  const parts = text.split(/(`[^`\n]+`)/g);
  return (
    <>
      {parts.map((part, i) => part.startsWith("`") && part.endsWith("`") && part.length > 2
        ? <code key={i} className="chat-inline-code">{part.slice(1, -1)}</code>
        : <Fragment key={i}>{linksAndBold(part)}</Fragment>)}
    </>
  );
}

export type Block =
  | { kind: "h"; level: number; text: string }
  | { kind: "p"; text: string }
  | { kind: "ul"; items: string[] }
  | { kind: "ol"; items: string[]; start: number }
  | { kind: "hr" }
  | { kind: "table"; rows: string[][] };

const TABLE_SEPARATOR = /^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$/;

/** Lines → blocks. A blank line ends a paragraph, list or table. */
export function parseBlocks(text: string): Block[] {
  const lines = text.replace(/<br\s*\/?>/gi, "\n").split("\n");
  const blocks: Block[] = [];
  let para: string[] = [];
  let cur: Block | null = null;
  const flush = () => {
    if (para.length) { blocks.push({ kind: "p", text: para.join("\n") }); para = []; }
    cur = null;
  };
  for (const line of lines) {
    const t = line.trim();
    if (!t) { flush(); continue; }
    let m: RegExpExecArray | null;
    if ((m = /^(#{1,6})\s+(.*)$/.exec(t))) { flush(); blocks.push({ kind: "h", level: m[1].length, text: m[2].replace(/\s+#+$/, "") }); continue; }
    if (/^([-*_])(\s*\1){2,}$/.test(t)) { flush(); blocks.push({ kind: "hr" }); continue; }
    if ((m = /^[-*•]\s+(.*)$/.exec(t))) {
      if (cur?.kind === "ul") cur.items.push(m[1]);
      else { flush(); cur = { kind: "ul", items: [m[1]] }; blocks.push(cur); }
      continue;
    }
    if ((m = /^(\d+)[.)]\s+(.*)$/.exec(t))) {
      if (cur?.kind === "ol") cur.items.push(m[2]);
      else { flush(); cur = { kind: "ol", items: [m[2]], start: Number(m[1]) || 1 }; blocks.push(cur); }
      continue;
    }
    if (t.startsWith("|")) {
      if (TABLE_SEPARATOR.test(t)) continue;
      const cells = t.replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
      if (cur?.kind === "table") cur.rows.push(cells);
      else { flush(); cur = { kind: "table", rows: [cells] }; blocks.push(cur); }
      continue;
    }
    if (cur) cur = null;
    para.push(line);
  }
  flush();
  return blocks;
}

function Blocks({ text }: { text: string }) {
  return (
    <>
      {parseBlocks(text).map((b, i) => {
        switch (b.kind) {
          case "h":
            return <p key={i} role="heading" aria-level={b.level} className={b.level <= 2 ? "mb-1 mt-3 text-[14.5px] font-semibold text-fg first:mt-0" : "mb-0.5 mt-2.5 text-[13.5px] font-semibold text-fg first:mt-0"}><InlineMd text={b.text} /></p>;
          case "ul":
            return <ul key={i} className="my-1.5 list-disc space-y-0.5 pl-5">{b.items.map((it, j) => <li key={j}><InlineMd text={it} /></li>)}</ul>;
          case "ol":
            return <ol key={i} start={b.start} className="my-1.5 list-decimal space-y-0.5 pl-5">{b.items.map((it, j) => <li key={j}><InlineMd text={it} /></li>)}</ol>;
          case "hr":
            return <hr key={i} className="my-3 border-line" />;
          case "table":
            return (
              <div key={i} className="my-2 overflow-x-auto">
                <table className="w-full text-[12.5px]">
                  <thead><tr>{b.rows[0].map((c, j) => <th key={j} className="border-b border-line px-2 py-1 text-left font-semibold text-fg"><InlineMd text={c} /></th>)}</tr></thead>
                  <tbody>{b.rows.slice(1).map((r, j) => <tr key={j}>{r.map((c, k) => <td key={k} className="border-b border-line px-2 py-1 align-top"><InlineMd text={c} /></td>)}</tr>)}</tbody>
                </table>
              </div>
            );
          default:
            return <p key={i} className="my-1.5 whitespace-pre-wrap first:mt-0 last:mb-0"><InlineMd text={b.text} /></p>;
        }
      })}
    </>
  );
}

export function Markdown({ text }: { text: string }) {
  const segments = text.split(/```(\w*)\n?([\s\S]*?)```/g);
  const nodes: ReactNode[] = [];
  for (let i = 0; i < segments.length; i += 3) {
    const plain = segments[i];
    if (plain && plain.trim()) nodes.push(<Blocks key={`t${i}`} text={plain} />);
    if (i + 2 < segments.length) nodes.push(<CodeBlock key={`c${i}`} lang={segments[i + 1]} code={segments[i + 2].replace(/\n$/, "")} />);
  }
  return <>{nodes}</>;
}
