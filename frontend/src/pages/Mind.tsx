// Mind (2026-10-21): the whole memory as one scene. Anthony: "establish
// a structure for the consciousness we have built for the whole system:
// knowledge, assumptions, duplicates, contradictions. Basically if you
// could look at a human brain and see each segment, thought etc."
//
// The scene (components/mind/scene.ts, three.js) draws what one read of
// /mind returns: regions laid out from the facts' own vectors, every
// fact, the sources, bonds, contradictions, duplicates, assumptions,
// gaps, and the runs and recalls of the window as streams from the core.
// This page is the HUD around it: the inspector for whatever is picked
// (every glow, arc and thread is clickable), the legend, and the strip
// that replays the window hour by hour.
import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Orbit } from "lucide-react";
import { api, errorMessage } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";
import { EmptyState, ErrorBanner, PageHeader, Skeleton } from "@/components/ui";
import type { MindScene } from "@/lib/types";
import { cn, fmtRelative } from "@/lib/utils";
import { suggestionLabel } from "@/pages/Conflicts";
import type { MindSceneHandle, MindSelection } from "@/components/mind/scene";

const KIND_LABEL: Record<MindSelection["kind"], string> = {
  overview: "Right now", region: "Region", fact: "Fact", source: "Source", bond: "Bond",
  conflict: "Disagreement", assumption: "Assumption", gap: "Gap", run: "Thought", recall: "Recall",
};

function fmtClock(iso: string | number | null | undefined): string {
  if (iso == null) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
}

/** The name of a picked thing, for the tooltip and the inspector eyebrow. */
export function nameOf(data: MindScene, sel: MindSelection | null): string {
  if (!sel) return "";
  switch (sel.kind) {
    case "overview": return "The whole mind";
    case "region": return data.regions.find((r) => r.id === sel.id)?.name ?? "Region";
    case "fact": return data.facts.find((f) => f.id === sel.id)?.key || "Fact";
    case "source": return data.sources.find((s) => s.id === sel.id)?.label ?? "Source";
    case "bond": {
      const [a, b] = sel.id.split("|");
      const bond = data.bonds.find((x) => x.a === a && x.b === b);
      const name = (r: number | null) => data.regions.find((x) => x.id === r)?.name ?? "a cluster";
      return bond ? `Bond: ${name(bond.region_a)} and ${name(bond.region_b)}` : "Bond";
    }
    case "conflict": {
      const c = data.conflicts.find((x) => x.id === sel.id);
      return c ? `${c.kind === "duplicate" ? "Duplicate" : "Disagreement"}: ${c.subject ?? c.a.claim.slice(0, 40)}` : "Disagreement";
    }
    case "assumption": return "Assumption: " + (data.assumptions.find((x) => x.id === sel.id)?.statement ?? "");
    case "gap": return "Gap: " + (data.gaps.find((x) => x.id === sel.id)?.missing ?? "");
    case "run": return data.runs.find((x) => x.id === sel.id)?.title ?? "Run";
    case "recall": return "Recall: " + (data.recalls.find((x) => x.id === sel.id)?.question ?? "");
  }
}

/** What the memory was doing at an hour of the window, from its events. */
export function doingAt(data: MindScene, hour: number): string {
  const since = Date.parse(data.window.since);
  const at = since + hour * 3_600_000;
  const before = data.events.filter((e) => Date.parse(e.at) <= at).sort((a, b) => Date.parse(b.at) - Date.parse(a.at))[0];
  if (before && at - Date.parse(before.at) < 2 * 3_600_000) return before.label ?? before.type.replace(/_/g, " ");
  const lanes = data.lanes.map((l) => ({ name: l.name, calls: l.hours[Math.min(l.hours.length - 1, Math.max(0, Math.floor(hour)))]?.calls ?? 0 }))
    .filter((l) => l.calls > 0).sort((a, b) => b.calls - a.calls);
  if (lanes.length) return `${lanes[0].name.toLowerCase()} (${lanes[0].calls} model calls this hour)`;
  return "quiet";
}

const Eyebrow = ({ children, tone }: { children: React.ReactNode; tone?: "red" | "cyan" | "vio" | "ok" | "warn" }) => (
  <p className={cn("text-[11px] font-medium uppercase tracking-[0.08em] text-fg-3",
    tone === "red" && "text-bad", tone === "cyan" && "text-info", tone === "vio" && "text-vio", tone === "ok" && "text-ok", tone === "warn" && "text-warn")}>{children}</p>
);
const Rows = ({ items }: { items: Array<{ text: string; note?: string; hot?: boolean }> }) => (
  <div className="flex flex-col gap-1.5 text-[13px]">
    {items.map((it, i) => (
      <div key={i} className={cn("rounded-lg border border-line px-2.5 py-2", it.hot && "border-bad/40 text-bad")}>
        {it.text}{it.note && <span className="text-fg-3"> · {it.note}</span>}
      </div>
    ))}
  </div>
);
const Btn = ({ to, children, primary }: { to: string; children: React.ReactNode; primary?: boolean }) => (
  <Link to={to} className={cn("rounded-md border px-2.5 py-1.5 text-[12.5px] font-medium",
    primary ? "border-accent bg-accent text-on-accent hover:bg-accent-hover" : "border-line text-fg-2 hover:bg-inset hover:text-fg")}>{children}</Link>
);

export function Inspector({ data, sel, onPick }: { data: MindScene; sel: MindSelection; onPick: (sel: MindSelection) => void }) {
  const regionName = (r: number | null | undefined) => data.regions.find((x) => x.id === r)?.name ?? "a region";
  const regionBtn = (r: number | null | undefined) => r == null ? null : (
    <button type="button" onClick={() => onPick({ kind: "region", id: r })} className="rounded-md border border-line px-2.5 py-1.5 text-[12.5px] font-medium text-fg-2 hover:bg-inset hover:text-fg">Fly to {regionName(r)}</button>
  );
  switch (sel.kind) {
    case "overview": {
      const t = data.totals;
      return (
        <>
          <Eyebrow>Right now</Eyebrow>
          <h2 className="text-[16px] font-semibold text-fg">{t.facts} facts in {t.regions} regions</h2>
          <p className="text-[13px] text-fg-2">Last {data.window.hours} hours: {t.runs} runs, {t.recalls} recalls, {data.sources.length} sources read. Bright points are trusted; dim ones arrived recently. Red arcs disagree, violet bridges are assumptions, dark voids are gaps. Click anything.</p>
          <div className="grid grid-cols-2 gap-2">
            {[["Disagreements", t.contradictions, "text-bad"], ["Duplicates", t.duplicates, "text-info"], ["Assumptions waiting", t.assumptions_waiting, "text-vio"], ["Gaps", t.gaps, "text-fg-2"]].map(([label, n, tone]) => (
              <div key={label as string} className="rounded-lg border border-line p-2.5"><div className={cn("font-mono text-[20px]", tone as string)}>{n as number}</div><div className="text-[12px] text-fg-3">{label as string}</div></div>
            ))}
          </div>
        </>
      );
    }
    case "region": {
      const r = data.regions.find((x) => x.id === sel.id); if (!r) return null;
      const facts = data.facts.filter((f) => f.region === r.id).sort((a, b) => b.hits - a.hits || (b.tier === "trusted" ? 1 : 0) - (a.tier === "trusted" ? 1 : 0)).slice(0, 6);
      const sources = data.sources.filter((s) => s.region === r.id).slice(0, 4);
      return (
        <>
          <Eyebrow tone="warn">Region · {r.name}</Eyebrow>
          <h2 className="text-[16px] font-semibold text-fg">{r.facts} facts in {r.clusters} clusters · {r.trusted} trusted</h2>
          {sources.length > 0 && <p className="text-[13px] text-fg-2">Fed by {sources.map((s) => s.label).join(", ")}{data.sources.filter((s) => s.region === r.id).length > 4 ? " and more" : ""}.</p>}
          <Rows items={facts.map((f) => ({ text: f.claim, note: f.tier === "trusted" ? "trusted" : f.tier === "reviewed" ? "reviewed" : "new" }))} />
          <div className="flex flex-wrap gap-2"><Btn to="/" primary>Open in Crystal Bank</Btn></div>
        </>
      );
    }
    case "fact": {
      const f = data.facts.find((x) => x.id === sel.id); if (!f) return null;
      const src = data.sources.find((s) => s.id === f.doc_id);
      return (
        <>
          <Eyebrow>Fact · {regionName(f.region)}</Eyebrow>
          <h2 className="text-[16px] font-semibold text-fg">{f.claim}</h2>
          <Rows items={[{ text: f.key || "(no key)", note: "key" }, { text: f.tier === "trusted" ? "Trusted" : f.tier === "reviewed" ? "Reviewed" : f.tier === "wrong" ? "Marked wrong" : "New, unvetted", note: `${f.hits} recalls` },
            { text: src ? src.label : f.source_kind === "model_reasoning" ? "A conversation" : f.source_kind === "agent_inferred" ? "Inferred by the memory" : "A document", note: f.learned_at ? `learned ${fmtRelative(f.learned_at)}` : undefined }]} />
          <div className="flex flex-wrap gap-2"><Btn to={`/bank/${encodeURIComponent(f.crystal_id)}`} primary>Open its cluster</Btn>{regionBtn(f.region)}</div>
        </>
      );
    }
    case "source": {
      const s = data.sources.find((x) => x.id === sel.id); if (!s) return null;
      return (
        <>
          <Eyebrow>Source · {regionName(s.region)}</Eyebrow>
          <h2 className="text-[16px] font-semibold text-fg">{s.label}</h2>
          <p className="text-[13px] text-fg-2">{s.facts} facts came from it{s.learned_at ? `, read ${fmtRelative(s.learned_at)}` : ""}.</p>
          <div className="flex flex-wrap gap-2"><Btn to="/knowledge" primary>Open in Knowledge</Btn>{regionBtn(s.region)}</div>
        </>
      );
    }
    case "bond": {
      const [a, b] = sel.id.split("|");
      const bond = data.bonds.find((x) => x.a === a && x.b === b); if (!bond) return null;
      return (
        <>
          <Eyebrow>Bond · {regionName(bond.region_a)} and {regionName(bond.region_b)}</Eyebrow>
          <h2 className="text-[16px] font-semibold text-fg">Two clusters that travel together</h2>
          <p className="text-[13px] text-fg-2">{bond.type === "co_cited" ? "Cited together in answers" : bond.type === "co_routed" ? "Routed to together" : bond.type === "chained" ? "Chained by the memory" : bond.type === "gap_subject" ? "Share an open question" : bond.type === "key_adjacent" ? "Neighbors by key" : bond.type === "vector_similar" ? "Neighbors by meaning" : bond.type ?? "Bonded"}, weight {bond.weight.toFixed(2)}. Bonds are what the assumptions scan walks first.</p>
          <div className="flex flex-wrap gap-2"><Btn to={`/bank/${encodeURIComponent(bond.a)}`}>Open one cluster</Btn><Btn to={`/bank/${encodeURIComponent(bond.b)}`}>Open the other</Btn></div>
        </>
      );
    }
    case "conflict": {
      const c = data.conflicts.find((x) => x.id === sel.id); if (!c) return null;
      const dup = c.kind === "duplicate";
      return (
        <>
          <Eyebrow tone={dup ? "cyan" : "red"}>{dup ? "Duplicate" : "Disagreement"}{c.subject ? ` · ${c.subject}` : ""}{c.status !== "open" ? ` · ${c.status}` : ""}</Eyebrow>
          <h2 className="text-[16px] font-semibold text-fg">{dup ? "Two facts that say the same thing" : "Two facts that cannot both be true"}</h2>
          <Rows items={[{ text: c.a.claim, note: "A" }, { text: c.b.claim, note: "B" }]} />
          {c.reasoning && (<><Eyebrow>{dup ? "Why these look like the same fact" : "Why it looks like a contradiction"}</Eyebrow><p className="text-[13px] text-fg-2">{c.reasoning}</p></>)}
          {c.recommendation && <p className="text-[13px] text-fg"><span className="font-medium">Suggested: {suggestionLabel(c.recommendation, dup ? "dedup_scan" : "contradiction_scan")}.</span>{c.recommendation.reason ? <span className="text-fg-2"> {c.recommendation.reason}</span> : null}</p>}
          <div className="flex flex-wrap gap-2"><Btn to="/conflicts" primary>Decide in Conflicts</Btn>{regionBtn(c.a.region)}</div>
        </>
      );
    }
    case "assumption": {
      const a = data.assumptions.find((x) => x.id === sel.id); if (!a) return null;
      return (
        <>
          <Eyebrow tone="vio">Assumption{a.confidence != null ? ` · ${Math.round(a.confidence * 100)}% confident` : ""} · {a.state === "waiting" ? "waiting for you" : a.state}</Eyebrow>
          <h2 className="text-[16px] font-semibold text-fg">{a.statement}</h2>
          {a.rationale && (<><Eyebrow>Why</Eyebrow><p className="text-[13px] text-fg-2">{a.rationale}</p></>)}
          {a.parents.length > 0 && (<><Eyebrow>Built from</Eyebrow><Rows items={a.parents.map((p) => ({ text: p.label, note: regionName(p.region) }))} /></>)}
          <div className="flex flex-wrap gap-2"><Btn to={`/assumptions/${encodeURIComponent(a.id)}`} primary>Review in Assumptions</Btn></div>
        </>
      );
    }
    case "gap": {
      const g = data.gaps.find((x) => x.id === sel.id); if (!g) return null;
      return (
        <>
          <Eyebrow>Gap · {g.priority}{g.region != null ? ` · ${regionName(g.region)}` : ""}</Eyebrow>
          <h2 className="text-[16px] font-semibold text-fg">{g.missing}</h2>
          <p className="text-[13px] text-fg-2">{g.source === "gap_discovery" ? "The memory noticed it does not know this." : g.source === "run_residual" ? "A research run left this unverified." : "Asked and not answered."}{g.disposition === "researchable" ? " Background research can take it." : ""}{g.opened_at ? ` Opened ${fmtRelative(g.opened_at)}.` : ""}</p>
          <div className="flex flex-wrap gap-2"><Btn to="/cognition" primary>Open in Cognition</Btn>{regionBtn(g.region)}</div>
        </>
      );
    }
    case "run": {
      const r = data.runs.find((x) => x.id === sel.id); if (!r) return null;
      return (
        <>
          <Eyebrow tone="ok">Thought · {r.type.replace(/_/g, " ")} · {r.started_at ? fmtClock(r.started_at) : ""}</Eyebrow>
          <h2 className="text-[16px] font-semibold text-fg">{r.title}</h2>
          <p className="text-[13px] text-fg-2">{r.status}{r.outcome ? ` · ${r.outcome.replace(/_/g, " ")}` : ""}{r.finished_at && r.started_at ? ` · ${Math.max(1, Math.round((Date.parse(r.finished_at) - Date.parse(r.started_at)) / 1000))}s` : ""}{r.region != null ? ` · landed in ${regionName(r.region)}` : ""}</p>
          {r.summary && <p className="text-[13px] text-fg-2">{r.summary}</p>}
          <div className="flex flex-wrap gap-2"><Btn to={`/activity/run/${encodeURIComponent(r.id)}`} primary>Open the run</Btn>{regionBtn(r.region)}</div>
        </>
      );
    }
    case "recall": {
      const q = data.recalls.find((x) => x.id === sel.id); if (!q) return null;
      return (
        <>
          <Eyebrow>Recall · {q.client ?? "a client"} · {fmtClock(q.at)}</Eyebrow>
          <h2 className="text-[16px] font-semibold text-fg">{q.question}</h2>
          <p className="text-[13px] text-fg-2">{q.match} match{q.latency_ms != null ? ` in ${q.latency_ms} ms` : ""}{q.model ? ` · ${q.model}` : ""}{q.touched.length ? ` · ${q.touched.length} facts touched` : ""}{q.region != null ? ` in ${regionName(q.region)}` : ""}.</p>
          <div className="flex flex-wrap gap-2"><Btn to="/activity" primary>Open in Activity</Btn>{regionBtn(q.region)}</div>
        </>
      );
    }
  }
}

export function Mind() {
  const { selectedCustomerId } = useSelectedCustomer();
  const stageRef = useRef<HTMLDivElement>(null);
  const labelsRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<MindSceneHandle | null>(null);
  const [sel, setSel] = useState<MindSelection>({ kind: "overview" });
  const [hover, setHover] = useState<{ sel: MindSelection; x: number; y: number } | null>(null);
  const [scrub, setScrub] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const reduced = useMemo(() => typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches, []);

  const q = useQuery({
    queryKey: ["mind", selectedCustomerId],
    queryFn: () => api.getMind(selectedCustomerId!, 24),
    enabled: !!selectedCustomerId,
    staleTime: 60_000,
  });
  const data = q.data;
  const hours = data?.window.hours ?? 24;
  const at = scrub ?? hours;

  // Build the scene once per data set; tear it down with it.
  useEffect(() => {
    if (!data || !stageRef.current || !labelsRef.current) return;
    let handle: MindSceneHandle | null = null;
    let cancelled = false;
    import("@/components/mind/scene").then(({ createMindScene }) => {
      if (cancelled || !stageRef.current || !labelsRef.current) return;
      handle = createMindScene(stageRef.current, data, {
        labelsEl: labelsRef.current,
        reducedMotion: reduced,
        onSelect: (s) => setSel(s),
        onHover: (s, x, y) => setHover(s ? { sel: s, x, y } : null),
      });
      sceneRef.current = handle;
      if (typeof ResizeObserver !== "undefined") {
        const ro = new ResizeObserver(() => handle?.resize());
        ro.observe(stageRef.current);
        const orig = handle.destroy;
        handle.destroy = () => { ro.disconnect(); orig(); };
      }
    });
    return () => { cancelled = true; handle?.destroy(); sceneRef.current = null; };
  }, [data, reduced]);

  useEffect(() => { sceneRef.current?.setScrub(at); }, [at]);
  useEffect(() => {
    if (!playing) { sceneRef.current?.setAutoRotate(false); return; }
    sceneRef.current?.setAutoRotate(true);
    let raf = 0, last = performance.now();
    const tick = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000); last = now;
      setScrub((v) => {
        const next = Math.min(hours, (v ?? 0) + dt * (hours / 30));
        if (next >= hours) setPlaying(false);
        return next;
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, hours]);

  if (!selectedCustomerId) {
    return (<><PageHeader title="Mind" /><EmptyState icon={Orbit} title="No customer selected" help="Pick a customer above to see their memory as one scene." /></>);
  }

  const pick = (s: MindSelection) => { setSel(s); sceneRef.current?.select(s, true); };
  const sinceMs = data ? Date.parse(data.window.since) : 0;
  const clockAt = (h: number) => fmtClock(sinceMs + h * 3_600_000);

  return (
    <div>
      <PageHeader title="Mind" meta="The whole memory as one scene: every fact by meaning, the regions they form, and what the memory itself has noticed, formed, asked and done. Drag to orbit, scroll to zoom, click anything." />
      {q.isError && <ErrorBanner message={errorMessage(q.error, "Couldn't load the mind.")} />}
      {q.isLoading && <Skeleton className="h-[560px] w-full rounded-xl" />}
      {data && data.facts.length === 0 && (
        <EmptyState icon={Orbit} title="Nothing to show yet" help="Store a few facts or ingest a document and the mind takes shape here." />
      )}
      {data && data.facts.length > 0 && (
        <div className="flex flex-wrap gap-4">
          <div className="relative min-w-0 flex-[999_1_560px]">
            <div className="relative h-[min(72vh,760px)] min-h-[440px] w-full overflow-hidden rounded-xl border border-line bg-[#05070c]">
              <div ref={stageRef} className="absolute inset-0" aria-label="The memory as a nebula" />
              <div ref={labelsRef} className="mind-labels pointer-events-none absolute inset-0" />
              {hover && (
                <div className="pointer-events-none absolute z-10 max-w-[260px] rounded-md border border-line bg-surface/90 px-2 py-1 text-[12px] text-fg backdrop-blur" style={{ left: hover.x - (stageRef.current?.getBoundingClientRect().left ?? 0) + 14, top: hover.y - (stageRef.current?.getBoundingClientRect().top ?? 0) + 14 }}>
                  {nameOf(data, hover.sel)}
                </div>
              )}
              <div className="pointer-events-none absolute bottom-3 left-3 hidden flex-col gap-1 rounded-lg border border-line bg-surface/85 px-3 py-2 text-[12px] text-fg-2 backdrop-blur min-[720px]:flex" aria-label="Legend">
                <span><i className="mr-2 inline-block h-2.5 w-2.5 rounded-full" style={{ background: "#fff", boxShadow: "0 0 8px #62cdfc" }} />Trusted fact</span>
                <span><i className="mr-2 inline-block h-2.5 w-2.5 rounded-full opacity-40" style={{ background: "#fff" }} />New, unvetted</span>
                <span><i className="mr-2 inline-block h-2.5 w-2.5 rounded-full border border-dashed border-vio" />Assumption</span>
                <span><i className="mr-2 inline-block h-0.5 w-4 bg-bad align-middle" />Contradiction</span>
                <span><i className="mr-2 inline-block h-0.5 w-4 bg-info align-middle" />Duplicate</span>
                <span><i className="mr-2 inline-block h-2.5 w-2.5 rounded-full border border-dotted border-fg-2" />Gap</span>
                <span><i className="mr-2 inline-block h-2.5 w-2.5 rounded-sm bg-fg-3" />Source</span>
                <span><i className="mr-2 inline-block h-0.5 w-4 bg-ok align-middle" />Thought in flight</span>
              </div>
            </div>

            <div className="mt-3 rounded-xl border border-line bg-surface p-3" aria-label={`Last ${hours} hours`}>
              <div className="mb-2 flex flex-wrap items-center gap-3">
                <span className="text-[11px] font-medium uppercase tracking-[0.08em] text-fg-3">Last {hours} hours</span>
                <button type="button" onClick={() => { if (!playing && at >= hours - 0.01) setScrub(0); setPlaying((p) => !p); }}
                  className="rounded-md border border-line px-2.5 py-1 text-[12.5px] font-medium text-fg-2 hover:bg-inset hover:text-fg">{playing ? "Pause" : "Replay"}</button>
                <span className="font-mono text-[12px] text-fg-2" data-testid="mind-when">{at >= hours - 0.01 ? "Now" : clockAt(at)} · {doingAt(data, Math.min(at, hours - 0.01))}</span>
              </div>
              <label htmlFor="mind-scrub" className="sr-only">Replay the window</label>
              <input id="mind-scrub" type="range" min={0} max={hours} step={0.05} value={at} className="w-full accent-accent"
                onChange={(e) => { setPlaying(false); setScrub(parseFloat(e.target.value)); }} />
              <div className="mt-2 grid grid-cols-[96px_minmax(0,1fr)] items-center gap-x-3 gap-y-1">
                {data.lanes.filter((l) => l.hours.some((h) => h.calls > 0)).map((lane) => {
                  const max = Math.max(1, ...lane.hours.map((h) => h.tokens));
                  return (
                    <div key={lane.name} className="contents">
                      <div className="truncate text-right text-[11.5px] text-fg-3">{lane.name}</div>
                      <div className="flex h-4 items-end gap-px border-b border-line" title={`${lane.hours.reduce((n, h) => n + h.calls, 0)} model calls`}>
                        {lane.hours.map((h, i) => (
                          <span key={i} aria-hidden="true" className={cn("flex-1 rounded-t-[1px]", i / hours <= at / hours ? "bg-accent" : "bg-fg-3/30")} style={{ height: `${Math.max(6, Math.round((h.tokens / max) * 100))}%`, opacity: h.calls ? 0.9 : 0.15 }} />
                        ))}
                      </div>
                    </div>
                  );
                })}
                <div />
                <div className="flex justify-between font-mono text-[10px] text-fg-3"><span>{clockAt(0)}</span><span>{clockAt(hours / 2)}</span><span className="text-accent">now</span></div>
              </div>
            </div>
          </div>

          <aside className="flex min-w-0 flex-[1_1_300px] flex-col gap-2.5 rounded-xl border border-line bg-surface p-4 min-[720px]:max-w-[380px]" aria-label="Inspector" aria-live="polite">
            <Inspector data={data} sel={sel} onPick={pick} />
            {sel.kind !== "overview" && (
              <button type="button" onClick={() => pick({ kind: "overview" })} className="self-start text-[12.5px] text-fg-3 hover:text-fg">Back to the whole mind</button>
            )}
            <p className="text-[12px] text-fg-3">{KIND_LABEL[sel.kind]} · refreshed {fmtRelative(data.generated_at)}</p>
          </aside>
        </div>
      )}
    </div>
  );
}
