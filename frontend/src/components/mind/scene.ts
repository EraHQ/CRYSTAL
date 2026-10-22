// The Mind scene (2026-10-21): the whole memory as a nebula you orbit.
//
// Imperative three.js, no React inside: the page hands it a container,
// a labels layer and the scene data, and gets back a handle to select,
// scrub the window and tear down. Everything drawn comes from the
// /mind read: regions are nebulae, facts are points (bright when
// trusted), bonds are threads, contradictions red arcs, duplicates cyan
// ties, assumptions dashed violet bridges, gaps dark voids, runs and
// recalls particle streams from the core. Every arc, thread and stream
// is clickable (an invisible tube follows it). The scrub replays the
// window: each thing enters with its own motion at the hour it happened.
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import type { MindScene } from "@/lib/types";

export type MindSelection =
  | { kind: "overview" }
  | { kind: "region"; id: number }
  | { kind: "fact"; id: string }
  | { kind: "source"; id: string }
  | { kind: "bond"; id: string }
  | { kind: "conflict"; id: string }
  | { kind: "assumption"; id: string }
  | { kind: "gap"; id: string }
  | { kind: "run"; id: string }
  | { kind: "recall"; id: string };

export interface MindSceneOptions {
  labelsEl: HTMLElement;
  onSelect: (sel: MindSelection) => void;
  onHover: (sel: MindSelection | null, x: number, y: number) => void;
  reducedMotion?: boolean;
}

export interface MindSceneHandle {
  select: (sel: MindSelection, fly?: boolean) => void;
  setScrub: (hours: number) => void;
  setAutoRotate: (on: boolean) => void;
  resize: () => void;
  destroy: () => void;
}

const COLORS = {
  bg: 0x05070c, white: 0xeef2ff, cyan: 0x62cdfc, amber: 0xf3b64a, red: 0xff6a6a, violet: 0xb793ff, green: 0x63d9a1, grey: 0x7d8aa3,
};
const REGION_PALETTE = [0xf3b64a, 0x62cdfc, 0x63d9a1, 0xb793ff, 0x7d8aa3, 0xff9f6a, 0x7fd6e8, 0xd9b8ff, 0xa8e38f, 0xffc9a3, 0x9fb6ff, 0xe8d37f];
const INTRO = 0.9; // hours of the window an item takes to enter

type V3 = [number, number, number];
type EnterKind = "sprite" | "points" | "fade" | "scale" | "burst" | "draw" | "stream";
type EnterBase = Partial<Record<"w" | "h" | "op" | "size" | "s" | "count", number>> & { stream?: Stream };
interface Timed { obj: THREE.Object3D; t: number; kind: EnterKind; base: EnterBase }
interface Stream { pts: THREE.Points; curve: THREE.Curve<THREE.Vector3>; phase: Float32Array; speed: number; reach: number }
interface Hit { mesh: THREE.Mesh; sel: MindSelection; t: number }
interface LabelDef { el: HTMLButtonElement; pos: THREE.Vector3; sel: MindSelection; t: number; small: boolean }

function vec(p: V3 | null | undefined, fallback?: THREE.Vector3): THREE.Vector3 {
  if (!p) return (fallback ?? new THREE.Vector3()).clone();
  return new THREE.Vector3(p[0], p[1], p[2]);
}
function sameSel(a: MindSelection | null, b: MindSelection | null): boolean {
  if (!a || !b) return a === b;
  return a.kind === b.kind && (a as { id?: unknown }).id === (b as { id?: unknown }).id;
}
function hoursSince(iso: string | null | undefined, since: number): number {
  if (!iso) return 0;
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return 0;
  return Math.max(0, (t - since) / 3_600_000);
}
function seeded(seed: number): () => number {
  let s = seed >>> 0;
  return () => { s += 0x6D2B79F5; let r = Math.imul(s ^ (s >>> 15), 1 | s); r ^= r + Math.imul(r ^ (r >>> 7), 61 | r); return ((r ^ (r >>> 14)) >>> 0) / 4294967296; };
}

function glowTexture(inner: string, outer: string): THREE.Texture {
  const size = 128;
  const cv = document.createElement("canvas"); cv.width = cv.height = size;
  const g = cv.getContext("2d")!;
  const grd = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grd.addColorStop(0, inner); grd.addColorStop(0.35, "rgba(255,255,255,.35)"); grd.addColorStop(1, outer);
  g.fillStyle = grd; g.fillRect(0, 0, size, size);
  const tx = new THREE.CanvasTexture(cv); tx.colorSpace = THREE.SRGBColorSpace; return tx;
}

export function createMindScene(container: HTMLElement, data: MindScene, opts: MindSceneOptions): MindSceneHandle {
  const reduced = !!opts.reducedMotion;
  const since = Date.parse(data.window.since);
  const hours = data.window.hours;
  const rnd = seeded(20261021);
  const gauss = () => (rnd() + rnd() + rnd() - 1.5) * 1.3;

  const width = () => container.clientWidth || 800;
  const height = () => container.clientHeight || 600;

  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(width(), height());
  renderer.setClearColor(COLORS.bg, 1);
  container.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(COLORS.bg, 0.0045);
  const camera = new THREE.PerspectiveCamera(55, width() / height(), 0.1, 2000);
  camera.position.set(0, 42, 128);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.06;
  controls.autoRotate = !reduced; controls.autoRotateSpeed = 0.35;
  // Close enough to pick one fact; far enough to see the whole bank.
  controls.minDistance = 1.5; controls.maxDistance = 420; controls.zoomSpeed = 1.15;
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  // Bloom is a halo, not a floodlight: a dense bank's center stayed
  // legible only once the strength and threshold came down (2026-10-21).
  composer.addPass(new UnrealBloomPass(new THREE.Vector2(width(), height()), 0.75, 0.5, 0.32));
  composer.addPass(new OutputPass());

  const dotTex = glowTexture("rgba(255,255,255,1)", "rgba(255,255,255,0)");
  const softTex = glowTexture("rgba(255,255,255,.55)", "rgba(255,255,255,0)");

  const timed: Timed[] = [];
  const hits: Hit[] = [];
  const pulsing: THREE.Material[] = [];
  const streams: Stream[] = [];
  const labels: LabelDef[] = [];
  const disposables: Array<{ dispose: () => void }> = [renderer, composer as unknown as { dispose: () => void }, dotTex, softTex];
  const enter = (obj: THREE.Object3D, t: number, kind: EnterKind, base: Timed["base"]) => { timed.push({ obj, t, kind, base }); return obj; };
  const track = (o: THREE.Object3D) => { const g = (o as THREE.Mesh).geometry, m = (o as THREE.Mesh).material; if (g) disposables.push(g); if (m) disposables.push(m as THREE.Material); return o; };

  const pickSphere = (pos: THREE.Vector3, radius: number, sel: MindSelection, t: number) => {
    const m = new THREE.Mesh(new THREE.SphereGeometry(radius, 8, 8), new THREE.MeshBasicMaterial({ visible: false }));
    m.position.copy(pos); scene.add(track(m)); hits.push({ mesh: m, sel, t }); return m;
  };
  const curveHit = (curve: THREE.Curve<THREE.Vector3>, radius: number, sel: MindSelection, t: number) => {
    const m = new THREE.Mesh(new THREE.TubeGeometry(curve, 24, radius, 6, false), new THREE.MeshBasicMaterial({ visible: false }));
    scene.add(track(m)); hits.push({ mesh: m, sel, t }); return m;
  };
  const glowDot = (pos: THREE.Vector3, color: number, size: number, opacity = 1) => {
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: dotTex, color, transparent: true, opacity, blending: THREE.AdditiveBlending, depthWrite: false }));
    sp.position.copy(pos); sp.scale.set(size, size, 1); scene.add(track(sp)); return sp;
  };
  const arc = (a: THREE.Vector3, b: THREE.Vector3, color: number, o: { lift?: number; dashed?: boolean; opacity?: number } = {}) => {
    const mid = a.clone().lerp(b, 0.5); const lift = o.lift ?? 10;
    const out = mid.clone().normalize(); if (out.length() < 0.01) out.set(0, 1, 0);
    mid.add(out.multiplyScalar(lift)).add(new THREE.Vector3(0, lift * 0.6, 0));
    const curve = new THREE.QuadraticBezierCurve3(a, mid, b);
    const geo = new THREE.BufferGeometry().setFromPoints(curve.getPoints(48));
    const mat = o.dashed
      ? new THREE.LineDashedMaterial({ color, dashSize: 1.6, gapSize: 1.1, transparent: true, opacity: o.opacity ?? 0.9, blending: THREE.AdditiveBlending, depthWrite: false })
      : new THREE.LineBasicMaterial({ color, transparent: true, opacity: o.opacity ?? 0.95, blending: THREE.AdditiveBlending, depthWrite: false });
    const line = new THREE.Line(geo, mat); if (o.dashed) line.computeLineDistances();
    scene.add(track(line)); return { line, curve, mat, count: 49 };
  };
  const addLabel = (pos: THREE.Vector3, text: string, sel: MindSelection, t: number, small = false) => {
    const el = document.createElement("button"); el.type = "button";
    el.className = "mind-label" + (small ? " mind-label-small" : ""); el.textContent = text;
    el.addEventListener("click", () => { handle.select(sel, true); opts.onSelect(sel); });
    opts.labelsEl.appendChild(el); labels.push({ el, pos: pos.clone(), sel, t, small });
  };

  // ---------- starfield ----------
  {
    const n = 1600, pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { const r = 400 + rnd() * 500, th = rnd() * Math.PI * 2, ph = Math.acos(2 * rnd() - 1); pos.set([r * Math.sin(ph) * Math.cos(th), r * Math.cos(ph), r * Math.sin(ph) * Math.sin(th)], i * 3); }
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    scene.add(track(new THREE.Points(g, new THREE.PointsMaterial({ size: 1.6, color: 0x9fb0d0, map: dotTex, transparent: true, opacity: 0.55, depthWrite: false, blending: THREE.AdditiveBlending }))));
  }

  // ---------- the core ----------
  const core = new THREE.Mesh(new THREE.SphereGeometry(2.2, 32, 32), new THREE.MeshBasicMaterial({ color: 0xdcecff }));
  scene.add(track(core));
  const coreGlow = glowDot(new THREE.Vector3(), COLORS.cyan, 12, 0.55);
  (coreGlow.material as THREE.SpriteMaterial).map = softTex;
  pickSphere(new THREE.Vector3(), 5, { kind: "overview" }, 0);
  addLabel(new THREE.Vector3(0, -7, 0), `${data.totals.facts} facts · ${data.totals.clusters} clusters`, { kind: "overview" }, 0);

  // ---------- regions, facts, sources ----------
  const regionCenter = new Map<number, THREE.Vector3>();
  const regionColor = new Map<number, number>();
  data.regions.forEach((r, i) => { regionCenter.set(r.id, vec(r.center)); regionColor.set(r.id, REGION_PALETTE[i % REGION_PALETTE.length]); });
  const factPos = new Map<string, THREE.Vector3>();
  const byRegion = new Map<number, typeof data.facts>();
  data.facts.forEach((f) => { factPos.set(f.id, vec(f.pos)); const l = byRegion.get(f.region) ?? []; l.push(f); byRegion.set(f.region, l); });

  const pointClouds: Array<{ points: THREE.Points; ids: string[] }> = [];
  data.regions.forEach((r) => {
    const color = regionColor.get(r.id)!;
    const center = regionCenter.get(r.id)!;
    const members = byRegion.get(r.id) ?? [];
    const t0 = members.length ? Math.min(...members.map((f) => hoursSince(f.learned_at, since))) : 0;
    for (let k = 0; k < 3; k++) {
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: softTex, color, transparent: true, opacity: 0.06 + k * 0.02, blending: THREE.AdditiveBlending, depthWrite: false }));
      sp.position.copy(center).add(new THREE.Vector3(gauss() * 4, gauss() * 2, gauss() * 4));
      const s = (12 + Math.min(members.length, 60) * 0.18) - k * 3; sp.scale.set(s, s * 0.8, 1); scene.add(track(sp));
      enter(sp, t0 + k * 0.05, "sprite", { w: s, h: s * 0.8, op: sp.material.opacity });
    }
    // facts as points: tier sets brightness; learned_at sets when they enter
    const base = new THREE.Color(color), white = new THREE.Color(0xffffff);
    const pos: number[] = [], col: number[] = [], ids: string[] = [];
    members.forEach((f) => {
      const p = factPos.get(f.id)!;
      const trusted = f.tier === "trusted", reviewed = f.tier === "reviewed";
      const c = white.clone().lerp(base, trusted ? 0.15 : 0.45);
      const dim = trusted ? 1 : reviewed ? 0.62 : 0.32;
      pos.push(p.x, p.y, p.z); col.push(c.r * dim, c.g * dim, c.b * dim); ids.push(f.id);
    });
    if (ids.length) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
      geo.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
      const pts = new THREE.Points(geo, new THREE.PointsMaterial({ size: 2.4, map: dotTex, vertexColors: true, transparent: true, opacity: 0.9, depthWrite: false, blending: THREE.AdditiveBlending }));
      scene.add(track(pts)); pointClouds.push({ points: pts, ids });
      enter(pts, t0 + 0.15, "points", { size: 2.4, op: 0.9 });
      // threads inside a cluster: each fact to the next of its cluster
      const byCrystal = new Map<string, THREE.Vector3[]>();
      members.forEach((f) => { const l = byCrystal.get(f.crystal_id) ?? []; l.push(factPos.get(f.id)!); byCrystal.set(f.crystal_id, l); });
      const lp: number[] = [];
      byCrystal.forEach((ps) => { for (let i = 0; i + 1 < ps.length; i++) lp.push(ps[i].x, ps[i].y, ps[i].z, ps[i + 1].x, ps[i + 1].y, ps[i + 1].z); });
      if (lp.length) {
        const lg = new THREE.BufferGeometry(); lg.setAttribute("position", new THREE.Float32BufferAttribute(lp, 3));
        const threads = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.16, blending: THREE.AdditiveBlending, depthWrite: false }));
        scene.add(track(threads)); enter(threads, t0 + 0.5, "fade", { op: 0.16 });
      }
    }
    pickSphere(center, 12, { kind: "region", id: r.id }, t0);
    addLabel(center.clone().add(new THREE.Vector3(0, 12, 0)), `${r.name} · ${r.facts}`, { kind: "region", id: r.id }, t0);
  });

  data.sources.forEach((s, i) => {
    const center = regionCenter.get(s.region); if (!center) return;
    const dir = center.clone().normalize(); if (dir.length() < 0.01) dir.set(1, 0, 0);
    const p = center.clone().add(dir.multiplyScalar(13 + (i % 4) * 2.5)).add(new THREE.Vector3(gauss() * 3, gauss() * 3, gauss() * 3));
    const cube = new THREE.Mesh(new THREE.BoxGeometry(1.5, 1.5, 1.5), new THREE.MeshBasicMaterial({ color: 0x8b9ab5 }));
    cube.position.copy(p); cube.rotation.set(rnd(), rnd(), rnd()); scene.add(track(cube));
    const t = hoursSince(s.learned_at, since);
    enter(cube, t, "scale", { s: 1 });
    pickSphere(p, 2.2, { kind: "source", id: s.id }, t);
  });

  // ---------- bonds between clusters ----------
  data.bonds.forEach((b) => {
    const a = vec(b.a_pos), c = vec(b.b_pos);
    const lg = new THREE.BufferGeometry().setFromPoints([a, c]);
    const line = new THREE.Line(lg, new THREE.LineBasicMaterial({ color: 0x9fb0d0, transparent: true, opacity: Math.min(0.35, 0.08 + b.weight * 0.1), blending: THREE.AdditiveBlending, depthWrite: false }));
    scene.add(track(line)); enter(line, 0, "fade", { op: (line.material as THREE.LineBasicMaterial).opacity });
    curveHit(new THREE.LineCurve3(a, c), 1.2, { kind: "bond", id: `${b.a}|${b.b}` }, 0);
  });

  // ---------- the system's own thoughts ----------
  data.conflicts.forEach((c) => {
    const ca = regionCenter.get(c.a.region ?? -1), cb = regionCenter.get(c.b.region ?? -1);
    const a = vec(c.a.pos, ca), b = vec(c.b.pos, cb);
    if (a.distanceTo(b) < 0.5) b.add(new THREE.Vector3(2.5, 0.8, 1));
    const color = c.kind === "contradiction" ? COLORS.red : COLORS.cyan;
    const open = c.status === "open";
    const r = arc(a, b, color, { lift: c.kind === "contradiction" ? 6 : 1.5, opacity: open ? 0.95 : 0.35 });
    if (open && c.kind === "contradiction") pulsing.push(r.mat);
    const t = hoursSince(c.found_at, since);
    const d1 = glowDot(a, color, c.kind === "contradiction" ? 5 : 4.5, open ? 1 : 0.4), d2 = glowDot(b, color, c.kind === "contradiction" ? 5 : 4.5, open ? 1 : 0.4);
    enter(r.line, t, "draw", { count: r.count }); enter(d1, t, "burst", { s: 5 }); enter(d2, t, "burst", { s: 5 });
    curveHit(r.curve, 1.6, { kind: "conflict", id: c.id }, t);
  });

  data.assumptions.forEach((s, i) => {
    const [pa, pb] = s.parents;
    const a = vec(pa?.pos, regionCenter.get(pa?.region ?? -1) ?? new THREE.Vector3(10 + i * 3, 0, 0));
    const b = vec(pb?.pos, regionCenter.get(pb?.region ?? -1) ?? a.clone().add(new THREE.Vector3(8, 4, 8)));
    const dim = s.state === "invalidated" ? 0.35 : 1;
    const l = arc(a, b, COLORS.violet, { lift: 14, dashed: true, opacity: 0.8 * dim });
    const mid = l.curve.getPoint(0.5);
    const ring = new THREE.Mesh(new THREE.TorusGeometry(2.2, 0.25, 8, 40), new THREE.MeshBasicMaterial({ color: COLORS.violet, transparent: true, opacity: dim }));
    ring.position.copy(mid); scene.add(track(ring));
    const g = glowDot(mid, COLORS.violet, 7, 0.7 * dim);
    const t = hoursSince(s.formed_at, since);
    enter(l.line, t, "draw", { count: l.count }); enter(ring, t, "scale", { s: 1 }); enter(g, t, "burst", { s: 7 });
    curveHit(l.curve, 1.8, { kind: "assumption", id: s.id }, t);
    pickSphere(mid, 3.5, { kind: "assumption", id: s.id }, t);
    addLabel(mid, s.statement.length > 48 ? s.statement.slice(0, 47) + "…" : s.statement, { kind: "assumption", id: s.id }, t, true);
  });

  data.gaps.forEach((gp, i) => {
    const center = regionCenter.get(gp.region ?? -1);
    // A void sits at the edge of its region, each on its own bearing;
    // a gap with no region rings the whole bank.
    const p = center
      ? center.clone().add(new THREE.Vector3(Math.cos(i * 2.1) * 16, -6 + (i % 5) * 3, Math.sin(i * 2.1) * 16))
      : new THREE.Vector3(Math.cos(i * 0.9) * 95, -12 + (i % 4) * 6, Math.sin(i * 0.9) * 95);
    const hole = new THREE.Mesh(new THREE.SphereGeometry(3.4, 24, 24), new THREE.MeshBasicMaterial({ color: COLORS.bg, transparent: true, opacity: 0.92 }));
    hole.position.copy(p); scene.add(track(hole));
    const n = 40, pos = new Float32Array(n * 3);
    for (let k = 0; k < n; k++) { const th = (k / n) * Math.PI * 2; pos.set([p.x + Math.cos(th) * 4.2, p.y + Math.sin(th * 2) * 0.6, p.z + Math.sin(th) * 4.2], k * 3); }
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    const rim = new THREE.Points(g, new THREE.PointsMaterial({ size: 1.4, color: 0xc3ccdd, map: dotTex, transparent: true, opacity: 0.9, depthWrite: false, blending: THREE.AdditiveBlending }));
    scene.add(track(rim));
    const t = hoursSince(gp.opened_at, since);
    enter(hole, t, "scale", { s: 1 }); enter(rim, t, "points", { size: 1.4, op: 0.9 });
    pickSphere(p, 5, { kind: "gap", id: gp.id }, t);
    addLabel(p.clone().add(new THREE.Vector3(0, 6, 0)), "Gap · " + (gp.missing.length > 40 ? gp.missing.slice(0, 39) + "…" : gp.missing), { kind: "gap", id: gp.id }, t, true);
  });

  // ---------- thoughts in flight: runs and recalls as streams from the core ----------
  const stream = (target: THREE.Vector3, color: number, t: number, sel: MindSelection, count = 36, opacity = 0.95) => {
    const mid = target.clone().multiplyScalar(0.5).add(new THREE.Vector3(gauss() * 10, 18 + gauss() * 4, gauss() * 10));
    const curve = new THREE.CatmullRomCurve3([new THREE.Vector3(0, 0, 0), mid, target]);
    const pos = new Float32Array(count * 3), phase = new Float32Array(count);
    for (let i = 0; i < count; i++) phase[i] = i / count;
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    const pts = new THREE.Points(g, new THREE.PointsMaterial({ size: 2.6, color, map: dotTex, transparent: true, opacity, depthWrite: false, blending: THREE.AdditiveBlending }));
    const trail = new THREE.Line(new THREE.BufferGeometry().setFromPoints(curve.getPoints(60)), new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.12, blending: THREE.AdditiveBlending, depthWrite: false }));
    scene.add(track(pts)); scene.add(track(trail));
    const s: Stream = { pts, curve, phase, speed: 0.12 + rnd() * 0.05, reach: 1 }; streams.push(s);
    enter(pts, t, "stream", { stream: s }); enter(trail, t, "draw", { count: 61 });
    curveHit(curve, 1.6, sel, t);
    return s;
  };
  data.runs.forEach((r, i) => {
    const target = r.landing ? vec(r.landing) : regionCenter.get(r.region ?? -1)?.clone() ?? new THREE.Vector3(Math.cos(i) * 60, 8, Math.sin(i) * 60);
    const t = hoursSince(r.started_at, since);
    const color = r.status === "failed" ? COLORS.red : r.type === "research" ? COLORS.green : COLORS.cyan;
    stream(target, color, t, { kind: "run", id: r.id });
    if (r.landing) {
      const dot = glowDot(target, color, 7); enter(dot, t + 0.7, "burst", { s: 7 });
      pickSphere(target, 4, { kind: "run", id: r.id }, t);
    }
    addLabel(target.clone().add(new THREE.Vector3(0, 5, 0)), r.title.length > 44 ? r.title.slice(0, 43) + "…" : r.title, { kind: "run", id: r.id }, t, true);
  });
  data.recalls.slice(0, 40).forEach((q, i) => {
    const touched = q.touched.map((id) => factPos.get(id)).filter(Boolean) as THREE.Vector3[];
    const target = touched.length
      ? touched.reduce((acc, p) => acc.add(p), new THREE.Vector3()).multiplyScalar(1 / touched.length)
      : regionCenter.get(q.region ?? -1)?.clone() ?? new THREE.Vector3(Math.cos(i * 1.3) * 55, 4, Math.sin(i * 1.3) * 55);
    stream(target, COLORS.white, hoursSince(q.at, since), { kind: "recall", id: q.id }, 16, 0.6);
  });

  // ---------- labels ----------
  const tmp = new THREE.Vector3();
  let selected: MindSelection | null = null;
  let scrubValue = hours;
  function placeLabels() {
    const w = width(), h = height();
    labels.forEach((d) => {
      if (d.t > scrubValue) { d.el.style.display = "none"; return; }
      tmp.copy(d.pos).project(camera);
      const behind = tmp.z > 1; const x = (tmp.x * 0.5 + 0.5) * w, y = (-tmp.y * 0.5 + 0.5) * h;
      const dist = camera.position.distanceTo(d.pos);
      const vis = !behind && x > -80 && x < w + 80 && y > 8 && y < h - 8 && (!d.small || dist < 110);
      d.el.style.display = vis ? "" : "none";
      if (vis) { d.el.style.left = x + "px"; d.el.style.top = y + "px"; d.el.style.opacity = String(Math.max(0.35, Math.min(1, 160 / dist))); }
      d.el.classList.toggle("on", sameSel(selected, d.sel));
    });
  }

  // ---------- selection and flight ----------
  const fly = { active: false, from: new THREE.Vector3(), to: new THREE.Vector3(), tFrom: new THREE.Vector3(), tTo: new THREE.Vector3(), start: 0 };
  function targetFor(sel: MindSelection): { look: THREE.Vector3; dist: number } {
    if (sel.kind === "overview") return { look: new THREE.Vector3(), dist: 130 };
    if (sel.kind === "region") return { look: regionCenter.get(sel.id) ?? new THREE.Vector3(), dist: 48 };
    if (sel.kind === "fact") return { look: factPos.get(sel.id) ?? new THREE.Vector3(), dist: 9 };
    const h = hits.find((x) => sameSel(x.sel, sel));
    if (!h) return { look: new THREE.Vector3(), dist: 60 };
    if (!h.mesh.geometry.boundingSphere) h.mesh.geometry.computeBoundingSphere();
    return { look: h.mesh.geometry.boundingSphere!.center.clone().add(h.mesh.position), dist: sel.kind === "bond" || sel.kind === "recall" || sel.kind === "run" ? 60 : 34 };
  }
  function select(sel: MindSelection, flyTo = false) {
    selected = sel;
    if (!flyTo) return;
    const { look, dist } = targetFor(sel);
    const dir = camera.position.clone().sub(controls.target).normalize();
    if (dir.length() < 0.01) dir.set(0, 0.4, 1).normalize();
    fly.from.copy(camera.position); fly.to.copy(look).add(dir.multiplyScalar(dist)).add(new THREE.Vector3(0, dist * 0.18, 0));
    fly.tFrom.copy(controls.target); fly.tTo.copy(look); fly.start = performance.now(); fly.active = true;
    controls.autoRotate = false;
  }

  // ---------- picking ----------
  const ray = new THREE.Raycaster(); ray.params.Points = { threshold: 1.1 };
  const mouse = new THREE.Vector2(); let downAt: [number, number] | null = null;
  const visibleHits = () => hits.filter((h) => h.t <= scrubValue).map((h) => h.mesh);
  function pick(e: PointerEvent): MindSelection | null {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
    ray.setFromCamera(mouse, camera);
    // facts first (a point under the cursor beats the region sphere around it)
    for (const pc of pointClouds) {
      if (!pc.points.visible) continue;
      const hit = ray.intersectObject(pc.points, false)[0];
      if (hit && hit.index !== undefined) return { kind: "fact", id: pc.ids[hit.index] };
    }
    const hit = ray.intersectObjects(visibleHits(), false)[0];
    if (!hit) return null;
    return hits.find((h) => h.mesh === hit.object)?.sel ?? null;
  }
  const onDown = (e: PointerEvent) => { downAt = [e.clientX, e.clientY]; };
  const onUp = (e: PointerEvent) => {
    if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 6) return;
    const sel = pick(e) ?? { kind: "overview" as const };
    select(sel, sel.kind !== "overview"); opts.onSelect(sel);
  };
  const onMove = (e: PointerEvent) => {
    if (e.pointerType === "touch") return;
    const sel = pick(e);
    renderer.domElement.style.cursor = sel ? "pointer" : "";
    opts.onHover(sel, e.clientX, e.clientY);
  };
  renderer.domElement.addEventListener("pointerdown", onDown);
  renderer.domElement.addEventListener("pointerup", onUp);
  renderer.domElement.addEventListener("pointermove", onMove);
  const onLeave = () => opts.onHover(null, 0, 0);
  renderer.domElement.addEventListener("pointerleave", onLeave);

  // ---------- the scrub: everything enters at its hour, with motion ----------
  const ease = (p: number) => 1 - Math.pow(1 - p, 3);
  function setScrub(v: number) {
    scrubValue = v;
    timed.forEach((x) => {
      const p = Math.max(0, Math.min(1, (v - x.t) / INTRO));
      x.obj.visible = p > 0;
      if (p <= 0) return;
      const e = ease(p), b = x.base;
      const mat = (x.obj as THREE.Mesh).material as THREE.Material & { opacity: number; size?: number };
      switch (x.kind) {
        case "sprite": x.obj.scale.set((b.w ?? 1) * (0.2 + 0.8 * e), (b.h ?? 1) * (0.2 + 0.8 * e), 1); mat.opacity = (b.op ?? 1) * e; break;
        case "points": (mat as THREE.PointsMaterial).size = (b.size ?? 1) * (0.3 + 0.7 * e); mat.opacity = (b.op ?? 1) * e; break;
        case "fade": mat.opacity = (b.op ?? 1) * e; break;
        case "scale": x.obj.scale.setScalar(Math.max(0.001, (b.s ?? 1) * e)); break;
        case "burst": { const s = (b.s ?? 1) * (1 + 2.2 * (1 - e)); x.obj.scale.set(s, s, 1); mat.opacity = 0.4 + 0.6 * e; break; }
        case "draw": (x.obj as THREE.Line).geometry.setDrawRange(0, Math.max(2, Math.floor((b.count ?? 2) * e))); break;
        case "stream": if (b.stream) b.stream.reach = e; break;
      }
    });
  }
  setScrub(hours);

  // ---------- loop ----------
  const clock = new THREE.Clock();
  let raf = 0; let alive = true;
  function frame() {
    if (!alive) return;
    raf = requestAnimationFrame(frame);
    const dt = Math.min(clock.getDelta(), 0.05), now = performance.now();
    if (!reduced) {
      const s = 1 + Math.sin(now / 900) * 0.06; core.scale.setScalar(s); coreGlow.scale.set(12 * s, 12 * s, 1);
      pulsing.forEach((m, i) => { (m as THREE.Material & { opacity: number }).opacity = 0.55 + 0.45 * (0.5 + 0.5 * Math.sin(now / 420 + i)); });
      streams.forEach((st) => {
        if (!st.pts.visible) return;
        const arr = st.pts.geometry.attributes.position.array as Float32Array;
        for (let i = 0; i < st.phase.length; i++) { st.phase[i] = (st.phase[i] + dt * st.speed) % 1; const p = st.curve.getPoint(Math.min(st.phase[i], st.reach)); arr[i * 3] = p.x; arr[i * 3 + 1] = p.y; arr[i * 3 + 2] = p.z; }
        st.pts.geometry.attributes.position.needsUpdate = true;
      });
    }
    if (fly.active) {
      const k = Math.min(1, (now - fly.start) / 1100), e = 1 - Math.pow(1 - k, 3);
      camera.position.lerpVectors(fly.from, fly.to, e); controls.target.lerpVectors(fly.tFrom, fly.tTo, e);
      if (k >= 1) fly.active = false;
    }
    controls.update();
    scene.traverse((o) => { if ((o as THREE.Mesh).geometry?.type === "TorusGeometry") o.lookAt(camera.position); });
    composer.render();
    placeLabels();
  }
  frame();

  const handle: MindSceneHandle = {
    select,
    setScrub,
    setAutoRotate: (on) => { controls.autoRotate = on && !reduced; },
    resize: () => { camera.aspect = width() / height(); camera.updateProjectionMatrix(); renderer.setSize(width(), height()); composer.setSize(width(), height()); },
    destroy: () => {
      alive = false; cancelAnimationFrame(raf);
      renderer.domElement.removeEventListener("pointerdown", onDown);
      renderer.domElement.removeEventListener("pointerup", onUp);
      renderer.domElement.removeEventListener("pointermove", onMove);
      renderer.domElement.removeEventListener("pointerleave", onLeave);
      controls.dispose();
      labels.forEach((l) => l.el.remove());
      disposables.forEach((d) => { try { d.dispose(); } catch { /* already gone */ } });
      renderer.domElement.remove();
    },
  };
  return handle;
}
