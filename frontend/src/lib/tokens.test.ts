// Console Refine PR 1 (docs/CONSOLE_UX_AUDIT.md Part 2): the semantic
// token layer is complete and the remap traps are gone from components.
//
// The dark remap in tailwind.config.js makes `text-white` render as the
// card surface (#151823) and leaves indigo / sky / teal / slate and every
// *-900 shade unmapped, so those classes paint light-theme colors onto
// the dark console (invisible wordmark, 2:1 tab text, bright chips).
// This pin is the spec's definition-of-done grep, run on every test
// pass, plus a check that every CSS variable a class or style reads is
// defined for both themes.
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const SRC = fileURLToPath(new URL("..", import.meta.url));
const ROOT = join(SRC, "..");

function walk(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p, out);
    else if (/\.(tsx|ts|css)$/.test(name) && !/\.test\.tsx?$/.test(name)) out.push(p);
  }
  return out;
}

const FILES = walk(SRC);
const TSX = FILES.filter((f) => f.endsWith(".tsx"));

// The remap traps, as the spec's grep states them, plus the two the
// console found live: `white/NN` opacity utilities (a dark tint, not a
// light one) and status *-900 text shades (unmapped, unreadable).
const TRAPS: Array<[string, RegExp]> = [
  ["text-white / bg-white", /\btext-white\b|\bbg-white\b/],
  ["unmapped indigo/sky/teal/slate", /\b(indigo|sky|teal|slate)-[0-9]/],
  ["white/NN opacity utility", /\b[a-z]+-white\/[0-9]/],
  ["status *-900 shade", /\b(text|bg|border)-(amber|green|emerald|red|rose|blue|violet|purple|cyan|yellow|orange)-900\b/],
];

describe("semantic tokens (Refine PR 1)", () => {
  it("no component uses a remap trap class", () => {
    const hits: string[] = [];
    for (const f of TSX) {
      const lines = readFileSync(f, "utf8").split("\n");
      lines.forEach((line, i) => {
        for (const [label, re] of TRAPS) {
          if (re.test(line)) hits.push(`${f.slice(SRC.length)}:${i + 1} ${label}: ${line.trim().slice(0, 100)}`);
        }
      });
    }
    expect(hits).toEqual([]);
  });

  it("every CSS variable read anywhere is defined for the dark and the light theme", () => {
    const css = readFileSync(join(SRC, "index.css"), "utf8");
    const dark = css.match(/:root\s*\{([^}]*)\}/)?.[1] ?? "";
    const light = css.match(/:root\[data-theme="light"\]\s*\{([^}]*)\}/)?.[1] ?? "";
    const defined = (block: string) => new Set([...block.matchAll(/--([a-z0-9-]+)\s*:/g)].map((m) => m[1]));
    const darkSet = defined(dark);
    const lightSet = defined(light);
    expect(darkSet.size).toBeGreaterThanOrEqual(23);
    expect([...darkSet].sort()).toEqual([...lightSet].sort());

    const used = new Set<string>();
    const sources = [...FILES, join(ROOT, "tailwind.config.js")];
    for (const f of sources) {
      for (const m of readFileSync(f, "utf8").matchAll(/var\(--([a-z0-9-]+)\)/g)) used.add(m[1]);
    }
    const missing = [...used].filter((v) => !darkSet.has(v)).sort();
    expect(missing).toEqual([]);
    // The spec's token table, by variable name (Part 2.2).
    for (const v of ["bg", "sf", "in", "bd", "bd2", "tx", "tx2", "tx3", "ac", "ac2", "ac-sf", "ac-tx", "on-ac",
      "ok", "ok-sf", "warn", "warn-sf", "bad", "bad-sf", "info", "info-sf", "vio", "vio-sf"]) {
      expect(darkSet.has(v), `--${v} missing`).toBe(true);
    }
  });

  it("the Tailwind names in the spec exist and text-accent is the light accent text", () => {
    const cfg = readFileSync(join(ROOT, "tailwind.config.js"), "utf8");
    for (const name of ["canvas", "surface", "inset", "line", "line-strong", "fg", "fg-2", "fg-3", "accent",
      "accent-hover", "accent-soft", "on-accent", "ok", "ok-soft", "warn", "warn-soft", "bad", "bad-soft",
      "info", "info-soft", "vio", "vio-soft"]) {
      expect(cfg, `color ${name}`).toMatch(new RegExp(`(^|[\\s{,])"?${name}"?\\s*:\\s*"rgb\\(var\\(--`, "m"));
    }
    expect(cfg).toMatch(/textColor:\s*\{[^}]*accent:\s*"rgb\(var\(--ac-tx\)/);
    expect(cfg).toMatch(/pop:\s*"/);
  });
});
