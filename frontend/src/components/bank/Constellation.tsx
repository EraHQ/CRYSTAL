// Constellation — the bank as a force graph (ratified 2026-07-15).
// Crystals are nodes (size = fact count, color = quality tier),
// clustered around derived hub nodes (the shelf axis, Obsidian
// tag-style). Real crystal_edges (co-query, weighted) and chains
// draw between crystals — the graph literally densifies as the bank
// gets USED. Click a crystal to open the reader.
import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  forceCenter, forceCollide, forceLink, forceManyBody, forceSimulation,
  type SimulationLinkDatum, type SimulationNodeDatum,
} from "d3-force";
import { authedFetch } from "@/lib/api";

export interface ConstellationCrystal {
  id: string;
  title: string;
  group: string;
  factCount: number;
  tier: string;
}

interface GraphEdge { a: string; b: string; weight: number; type?: string }

interface Node extends SimulationNodeDatum {
  id: string;
  label: string;
  hub: boolean;
  r: number;
  tier?: string;
}
interface Link extends SimulationLinkDatum<Node> {
  weight: number;
  kind: "member" | "edge" | "chain";
}

async function fetchGraph(customerId: string): Promise<{ edges: GraphEdge[]; chains: Array<{ a: string; b: string }> }> {
  const res = await authedFetch(`/admin/api/bank/graph?customer_id=${encodeURIComponent(customerId)}`);
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}

const TIER_FILL: Record<string, string> = {
  verified: "#14b8a6",
  established: "#0ea5e9",
  quarantine: "#f59e0b",
  untiered: "#9ca3af",
};

export function Constellation({ customerId, crystals, onOpen }: {
  customerId: string;
  crystals: ConstellationCrystal[];
  onOpen: (id: string) => void;
}) {
  const graph = useQuery({
    queryKey: ["bank-graph", customerId],
    queryFn: () => fetchGraph(customerId),
  });

  const wrapRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 900, h: 560 });
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() =>
      setSize({ w: el.clientWidth, h: Math.max(480, el.clientHeight) }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const [positioned, setPositioned] = useState<{ nodes: Node[]; links: Link[] }>({ nodes: [], links: [] });
  const [hover, setHover] = useState<string | null>(null);

  const built = useMemo(() => {
    const hubs = [...new Set(crystals.map((c) => c.group))];
    const nodes: Node[] = [
      ...hubs.map((h) => ({ id: `hub:${h}`, label: h, hub: true, r: 22 })),
      ...crystals.map((c) => ({
        id: c.id, label: c.title, hub: false, tier: c.tier || "untiered",
        r: Math.max(6, Math.min(18, 5 + Math.sqrt(c.factCount) * 3)),
      })),
    ];
    const ids = new Set(nodes.map((n) => n.id));
    const links: Link[] = crystals.map((c) => ({
      source: `hub:${c.group}`, target: c.id, weight: 0.4, kind: "member" as const,
    }));
    for (const e of graph.data?.edges ?? []) {
      if (ids.has(e.a) && ids.has(e.b)) links.push({ source: e.a, target: e.b, weight: e.weight ?? 1, kind: "edge" });
    }
    for (const c of graph.data?.chains ?? []) {
      if (ids.has(c.a) && ids.has(c.b)) links.push({ source: c.a, target: c.b, weight: 0.8, kind: "chain" });
    }
    return { nodes, links };
  }, [crystals, graph.data]);

  useEffect(() => {
    const nodes = built.nodes.map((n) => ({ ...n }));
    const links = built.links.map((l) => ({ ...l }));
    const sim = forceSimulation<Node>(nodes)
      .force("link", forceLink<Node, Link>(links).id((d) => d.id)
        .distance((l) => (l.kind === "member" ? 70 : 110))
        .strength((l) => (l.kind === "member" ? 0.5 : Math.min(0.8, 0.2 + l.weight * 0.15))))
      .force("charge", forceManyBody().strength(-140))
      .force("center", forceCenter(size.w / 2, size.h / 2))
      .force("collide", forceCollide<Node>().radius((d) => d.r + 6))
      .stop();
    for (let i = 0; i < 260; i++) sim.tick();
    setPositioned({ nodes, links });
  }, [built, size.w, size.h]);

  const { nodes, links } = positioned;
  const neighbor = useMemo(() => {
    if (!hover) return null;
    const set = new Set<string>([hover]);
    for (const l of links) {
      const s = (l.source as Node).id, t = (l.target as Node).id;
      if (s === hover) set.add(t);
      if (t === hover) set.add(s);
    }
    return set;
  }, [hover, links]);

  return (
    <div ref={wrapRef} className="w-full" style={{ height: "calc(100vh - 200px)", minHeight: 480 }}>
      {graph.isError && (
        <p className="text-xs text-amber-600 mb-2">Couldn't load edges — showing cluster membership only.</p>
      )}
      <svg width={size.w} height={size.h} className="select-none">
        {links.map((l, i) => {
          const s = l.source as Node, t = l.target as Node;
          const dim = neighbor && !(neighbor.has(s.id) && neighbor.has(t.id));
          return (
            <line key={i} x1={s.x} y1={s.y} x2={t.x} y2={t.y}
              stroke={l.kind === "member" ? "#d1d5db" : l.kind === "chain" ? "#a78bfa" : "#818cf8"}
              strokeWidth={l.kind === "member" ? 1 : Math.min(4, 1 + l.weight)}
              strokeDasharray={l.kind === "chain" ? "4 3" : undefined}
              opacity={dim ? 0.08 : l.kind === "member" ? 0.5 : 0.75} />
          );
        })}
        {nodes.map((n) => {
          const dim = neighbor && !neighbor.has(n.id);
          return (
            <g key={n.id} transform={`translate(${n.x},${n.y})`} opacity={dim ? 0.18 : 1}
              onMouseEnter={() => setHover(n.id)} onMouseLeave={() => setHover(null)}
              onClick={() => !n.hub && onOpen(n.id)}
              style={{ cursor: n.hub ? "default" : "pointer" }}>
              <circle r={n.r}
                fill={n.hub ? "#f3f4f6" : TIER_FILL[n.tier ?? "untiered"] ?? TIER_FILL.untiered}
                stroke={n.hub ? "#9ca3af" : "white"} strokeWidth={n.hub ? 1.5 : 1.5} />
              <text y={n.hub ? 4 : n.r + 12} textAnchor="middle"
                className={n.hub ? "fill-gray-600" : "fill-gray-500"}
                style={{ fontSize: n.hub ? 11 : 9.5, fontWeight: n.hub ? 600 : 400, pointerEvents: "none" }}>
                {n.label.length > 26 ? n.label.slice(0, 25) + "…" : n.label}
              </text>
            </g>
          );
        })}
      </svg>
      <p className="text-[10px] text-gray-400 -mt-5 pl-1">
        size = fact count · color = tier · solid indigo = co-queried together (thicker = more often) · dashed violet = chained
      </p>
    </div>
  );
}
